// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Main driver.
// Phase 2: CompileBlock is a proper code emitter. For each pair it emits:
//   - cycle++ and TPC advance inline
//   - ARM64 BL calls to stall helpers with compile-time-resolved uregs/lregs ptrs
//   - Upper instruction via recVU1_UpperTable (direct BL, no runtime table lookup)
//   - Lower instruction via recVU1_LowerTable (same)
//   - Inline branch/ebit countdown
// VF/VI hazard pairs fall back to vu1Exec for correctness.

#include "Common.h"
#include "GS.h"
#include "Gif_Unit.h"
#include "Memory.h"
#include "MTVU.h"
#include "VUmicro.h"
#include "VUops.h"
#include "arm64/AsmHelpers.h"
#include "arm64/iVU1micro_arm64.h"

#include <cfenv>
#include <cstring>

using namespace vixl::aarch64;

// Global instance
recArmVU1 CpuArmVU1;

// VU1 per-cycle interpreter entry point (defined in VU1microInterp.cpp)
extern void vu1Exec(VURegs* VU);

// Flush helpers declared in VU1microInterp.cpp
extern void _vuFlushAll(VURegs* VU);
extern void _vuXGKICKTransfer(s32 cycles, bool flush);

// ============================================================================
//  Rec emitter dispatch tables (defined in iVU1Upper/Lower_arm64.cpp)
// ============================================================================

using VU1RecFn = void (*)();
extern VU1RecFn recVU1_UpperTable[64];
extern VU1RecFn recVU1_LowerTable[128];

// ============================================================================
//  Block cache
// ============================================================================

// VU1_PROGSIZE / VU1_PROGMASK come from VUmicro.h
static constexpr u32 VU1_NUM_SLOTS       = VU1_PROGSIZE / 8; // one slot per instruction-pair
static constexpr u32 VU1_MAX_BLOCK_PAIRS = 256;

// Reserve a slice at the start of the JIT region for the constant pool.
static constexpr u32 POOL_SIZE = 64 * 1024;

struct VU1BlockEntry
{
	u8*  codeEntry; // nullptr = not yet compiled
	u32  numPairs;
};

static VU1BlockEntry s_blocks[VU1_NUM_SLOTS];
static u8* s_code_base  = nullptr;
static u8* s_code_write = nullptr;
static u8* s_code_end   = nullptr;
static ArmConstantPool s_pool;

// ============================================================================
//  Runtime helper functions called from compiled blocks
// ============================================================================

// Check D/T bits at runtime (depends on VU0 FBRST which is a runtime value).
static void vu1CheckDTBits(u32 upper)
{
	if (upper & 0x10000000) // D flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x400)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x200;
			hwIntcIrq(INTC_VU1);
			VU1.ebit = 1;
		}
	}
	if (upper & 0x08000000) // T flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x800)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x400;
			hwIntcIrq(INTC_VU1);
			VU1.ebit = 1;
		}
	}
}

// End-of-microprogram cleanup (called when ebit countdown hits 0).
static void vu1EbitDone(VURegs* VU)
{
	VU->VIBackupCycles = 0;
	_vuFlushAll(VU);
	VU0.VI[REG_VPU_STAT].UL &= ~0x100;
	vif1Regs.stat.VEW = false;

	if (VU1.xgkickenable)
		_vuXGKICKTransfer(0, true);
	if (INSTANT_VU1)
		VU1.xgkicklastcycle = cpuRegs.cycle;
}

// Handle takedelaybranch state when branch countdown fires.
static void vu1HandleDelayBranch(VURegs* VU)
{
	if (VU->takedelaybranch)
	{
		VU->branch          = 1;
		VU->branchpc        = VU->delaybranchpc;
		VU->takedelaybranch = false;
	}
}

// Decrement VIBackupCycles after stalls, matching the interpreter.
// cyclesBefore is VU->cycle - 1 (the cycle count before this pair's cycle++).
// Stalls may advance VU->cycle further; the elapsed count determines how much
// to decrement VIBackupCycles.
static void vu1DecrementVIBackup(VURegs* VU, u64 cyclesBefore)
{
	if (VU->VIBackupCycles > 0)
	{
		u32 elapsed = static_cast<u32>(VU->cycle - cyclesBefore);
		if (elapsed >= VU->VIBackupCycles)
			VU->VIBackupCycles = 0;
		else
			VU->VIBackupCycles -= static_cast<u8>(elapsed);
	}
}

// ============================================================================
//  Block analysis helpers
// ============================================================================

static bool PairHasEbit(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU1.Micro + pc + 4);
	return (upper >> 30) & 1;
}

static bool PairHasBranch(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU1.Micro + pc + 4);
	if ((upper >> 31) & 1)
		return false; // I-bit: lower field is immediate, not an opcode
	const u32 lower = *reinterpret_cast<const u32*>(VU1.Micro + pc);
	_VURegsNum lregs{};
	VU1regs_LOWER_OPCODE[lower >> 25](&lregs);
	return lregs.pipe == VUPIPE_BRANCH;
}

static u32 AnalyzeBlock(u32 startPC)
{
	u32 pairs = 0;
	u32 pc    = startPC;

	while (pairs < VU1_MAX_BLOCK_PAIRS)
	{
		const bool ebit   = PairHasEbit(pc);
		const bool branch = PairHasBranch(pc);

		pairs++;
		pc = (pc + 8) & (VU1_PROGSIZE - 1);

		if (ebit || branch)
		{
			// Include the one delay-slot pair then stop.
			pairs++;
			break;
		}
	}

	return pairs;
}

// ============================================================================
//  Block compilation
// ============================================================================

// Pinned VU1 base register used throughout compiled blocks.
// x23 is callee-saved (AAPCS64) and not clobbered by C function calls.
static const auto VU1_BASE_REG = x23;

static u8* CompileBlock(u32 startPC, u32 numPairs)
{
	// --- Size check ---
	const size_t data_size    = numPairs * 2 * sizeof(_VURegsNum);
	const size_t code_worst   = static_cast<size_t>(numPairs) * 512 + 64;
	const size_t total_needed = data_size + code_worst;

	if (static_cast<size_t>(s_code_end - s_code_write) < total_needed)
	{
		DEV_LOG("VU1 JIT: code buffer full, resetting");
		std::memset(s_blocks, 0, sizeof(s_blocks));
		s_code_write = s_code_base;
		s_pool.Reset();
	}

	// --- Data section: pre-computed uregs/lregs for every pair ---
	// Layout: [uregs[0..N-1]] [lregs[0..N-1]] (before the code, in JIT buffer)
	u8* const data_base = s_code_write;
	_VURegsNum* const uregs_data = reinterpret_cast<_VURegsNum*>(data_base);
	_VURegsNum* const lregs_data = uregs_data + numPairs;

	// Zero the entire data section first — some regs functions don't set all
	// fields (e.g., branch regs don't set 'cycles'). The interpreter does
	// 'lregs.cycles = 0' before calling the regs function; we match that by
	// zeroing the whole array.
	std::memset(data_base, 0, data_size);

	{
		u32 pc = startPC;
		for (u32 i = 0; i < numPairs; i++)
		{
			const u32 upper = *reinterpret_cast<const u32*>(VU1.Micro + pc + 4);
			const u32 lower = *reinterpret_cast<const u32*>(VU1.Micro + pc);

			VU1.code = upper;
			VU1regs_UPPER_OPCODE[upper & 0x3f](&uregs_data[i]);

			if (!((upper >> 31) & 1))
			{
				// Non-I-bit: lower field is an instruction.
				VU1.code = lower;
				VU1regs_LOWER_OPCODE[lower >> 25](&lregs_data[i]);
			}
			// I-bit pairs: lregs_data[i] stays zeroed (no lower instruction).

			pc = (pc + 8) & (VU1_PROGSIZE - 1);
		}
	}

	// Code section starts after data, 4-byte aligned.
	u8* code_start = data_base + data_size;
	code_start = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(code_start) + 3) & ~3ULL);

	armSetAsmPtr(code_start, static_cast<size_t>(s_code_end - code_start), &s_pool);
	u8* const entry = armStartBlock();

	// --- Prologue: save callee-saved regs, pin VU1_BASE_REG = &VU1 ---
	// 32-byte frame: [sp+0..7] = x29/x30, [sp+16..23] = x22, [sp+24..31] = x23
	armAsm->Stp(x29, x30, MemOperand(sp, -32, PreIndex));
	armAsm->Stp(x22, VU1_BASE_REG, MemOperand(sp, 16));
	armAsm->Mov(x29, sp);
	armMoveAddressToReg(VU1_BASE_REG, &VU1);

	// Compile-time constants for field offsets used throughout the loop.
	const int64_t cycle_off    = (int64_t)offsetof(VURegs, cycle);
	const int64_t code_off     = (int64_t)offsetof(VURegs, code);
	const int64_t branch_off   = (int64_t)offsetof(VURegs, branch);
	const int64_t branchpc_off = (int64_t)offsetof(VURegs, branchpc);
	const int64_t ebit_off     = (int64_t)offsetof(VURegs, ebit);
	const int64_t tpc_off      = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_TPC * (int64_t)sizeof(REG_VI));
	const int64_t regi_off     = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_I   * (int64_t)sizeof(REG_VI));
	const int64_t fmacwpos_off = (int64_t)offsetof(VURegs, fmacwritepos);

	// --- Per-pair code emission ---
	u32 pc = startPC;
	for (u32 i = 0; i < numPairs; i++)
	{
		const u32 upper     = *reinterpret_cast<const u32*>(VU1.Micro + pc + 4);
		const u32 lower     = *reinterpret_cast<const u32*>(VU1.Micro + pc);
		const bool ibit     = (upper >> 31) & 1;
		const bool ebit_set = (upper >> 30) & 1;
		const bool dbit_set = (upper >> 28) & 1;
		const bool tbit_set = (upper >> 27) & 1;
		const _VURegsNum& uregs = uregs_data[i];
		const _VURegsNum& lregs = lregs_data[i];

		// Detect VF/VI write-before-read hazard at compile time.
		// If present, fall back to vu1Exec for this pair for correctness.
		// The TPC at this point already equals `pc` (set by the previous pair),
		// so vu1Exec can run directly without adjustment.
		const bool vf_hazard = !ibit && uregs.VFwrite != 0 &&
			(lregs.VFread0 == uregs.VFwrite || lregs.VFread1 == uregs.VFwrite);
		const bool vi_hazard = !ibit &&
			(uregs.VIwrite & (1u << REG_CLIP_FLAG)) &&
			(lregs.VIread  & (1u << REG_CLIP_FLAG));

		if (vf_hazard || vi_hazard)
		{
			// Full interpreter fallback for this pair.
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1Exec));
			pc = (pc + 8) & (VU1_PROGSIZE - 1);
			continue;
		}

		// 1. VU->cycle++
		//    Save cycle-1 (= cycle before this pair) in x22 for VIBackupCycles.
		//    x22 is callee-saved and already saved/restored in our prologue/epilogue.
		armAsm->Ldr(x4, MemOperand(VU1_BASE_REG, cycle_off));
		armAsm->Mov(x22, x4);  // cyclesBefore = old cycle (before ++)
		armAsm->Add(x4, x4, 1);
		armAsm->Str(x4, MemOperand(VU1_BASE_REG, cycle_off));

		// 2. Advance TPC to next pair (compile-time constant).
		const u32 new_tpc = (pc + 8) & VU1_PROGMASK;
		armAsm->Mov(w4, new_tpc);
		armAsm->Str(w4, MemOperand(VU1_BASE_REG, tpc_off));

		// 3. E-bit: set VU->ebit = 2 (bit 30 of upper — compile-time known).
		if (ebit_set)
		{
			armAsm->Mov(w4, 2u);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, ebit_off));
		}

		// 4. D/T bits: depend on VU0 FBRST (runtime). Only emit when actually set.
		if (dbit_set || tbit_set)
		{
			armAsm->Mov(w0, upper);
			armEmitCall(reinterpret_cast<const void*>(vu1CheckDTBits));
		}

		// 5. Test upper stalls. uregs pointer is stable — lives in the JIT buffer
		//    allocated above. No runtime VU1regs table lookup needed.
		armAsm->Mov(x0, VU1_BASE_REG);
		armMoveAddressToReg(x1, &uregs_data[i]);
		armEmitCall(reinterpret_cast<const void*>(_vuTestUpperStalls));

		// 5b. Test lower stalls BEFORE TestPipes (non-I-bit only).
		//     TestLowerStalls may advance VU->cycle (FDIV/EFU/ALU stalls);
		//     TestPipes needs to see the updated cycle to flush FMAC correctly.
		if (!ibit)
		{
			armAsm->Mov(x0, VU1_BASE_REG);
			armMoveAddressToReg(x1, &lregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuTestLowerStalls));
		}

		// 6. Test pipes (always, after lower stalls for non-I-bit).
		armAsm->Mov(x0, VU1_BASE_REG);
		armEmitCall(reinterpret_cast<const void*>(_vuTestPipes));

		// 6b. Decrement VIBackupCycles (needed for correct VI backup reads
		//     in branch instructions). x22 holds cycle value before this pair.
		armAsm->Mov(x0, VU1_BASE_REG);
		armAsm->Mov(x1, x22);
		armEmitCall(reinterpret_cast<const void*>(vu1DecrementVIBackup));

		// 7. Execute upper instruction.
		//    Set VU->code at runtime (interpreter reads it for register fields).
		//    Set VU1.code at compile time so the rec emitter resolves the correct
		//    interpreter function pointer via VU1_UPPER_OPCODE[code & 0x3f].
		armAsm->Mov(w4, upper);
		armAsm->Str(w4, MemOperand(VU1_BASE_REG, code_off));
		VU1.code = upper; // compile-time context for the rec emitter
		recVU1_UpperTable[upper & 0x3f](); // emits BL to specific interpreter fn

		// 8. Lower instruction handling.
		if (ibit)
		{
			// I-bit: lower field is a float immediate — load into VI[REG_I].
			armAsm->Mov(w4, lower);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, regi_off));
		}
		else
		{
			// Execute lower instruction (stalls already tested above).
			armAsm->Mov(w4, lower);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, code_off));
			VU1.code = lower; // compile-time context
			recVU1_LowerTable[lower >> 25](); // emits BL to specific interpreter fn
		}

		// 9. FMAC clear (if either pipe is FMAC).
		if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
		{
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(_vuClearFMAC));
		}

		// 10. Add upper stalls.
		armAsm->Mov(x0, VU1_BASE_REG);
		armMoveAddressToReg(x1, &uregs_data[i]);
		armEmitCall(reinterpret_cast<const void*>(_vuAddUpperStalls));

		// 11. Add lower stalls (skipped when I-bit suppressed lower).
		if (!ibit)
		{
			armAsm->Mov(x0, VU1_BASE_REG);
			armMoveAddressToReg(x1, &lregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuAddLowerStalls));
		}

		// 12. Branch countdown (inline).
		{
			Label skip_branch;
			armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, branch_off));
			armAsm->Cbz(w4, &skip_branch);        // branch == 0: nothing to do
			armAsm->Subs(w4, w4, 1);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, branch_off));
			armAsm->B(&skip_branch, ne);           // still > 0: keep counting
			// branch just reached 0: set TPC = branchpc
			armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, branchpc_off));
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, tpc_off));
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1HandleDelayBranch));
			armAsm->Bind(&skip_branch);
		}

		// 13. Ebit countdown (inline).
		{
			Label skip_ebit;
			armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, ebit_off));
			armAsm->Cbz(w4, &skip_ebit);          // ebit == 0: nothing to do
			armAsm->Subs(w4, w4, 1);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, ebit_off));
			armAsm->B(&skip_ebit, ne);             // still > 0: keep counting
			// ebit just reached 0: end of microprogram
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1EbitDone));
			armAsm->Bind(&skip_ebit);
		}

		// 14. FMAC write-position advance (wraps mod 4).
		if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
		{
			armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, fmacwpos_off));
			armAsm->Add(w4, w4, 1);
			armAsm->And(w4, w4, 3);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, fmacwpos_off));
		}

		pc = (pc + 8) & (VU1_PROGSIZE - 1);
	}

	// --- Epilogue ---
	armAsm->Ldp(x22, VU1_BASE_REG, MemOperand(sp, 16));
	armAsm->Ldp(x29, x30, MemOperand(sp, 32, PostIndex));
	armAsm->Ret();

	u8* end = armEndBlock();
	s_code_write = end;
	return entry;
}

// ============================================================================
//  recArmVU1
// ============================================================================

recArmVU1::recArmVU1()
{
	m_Idx = 1;
	IsInterpreter = false;
}

void recArmVU1::Reserve()
{
	u8* const buf     = SysMemory::GetVU1Rec();
	u8* const buf_end = SysMemory::GetVU1RecEnd();

	s_pool.Init(buf, POOL_SIZE);
	s_code_base  = buf + POOL_SIZE;
	s_code_write = s_code_base;
	s_code_end   = buf_end;

	std::memset(s_blocks, 0, sizeof(s_blocks));
}

void recArmVU1::Shutdown()
{
	s_pool.Destroy();
	s_code_base  = nullptr;
	s_code_write = nullptr;
	s_code_end   = nullptr;
	std::memset(s_blocks, 0, sizeof(s_blocks));
}

void recArmVU1::Reset()
{
	VU1.fmacwritepos = 0;
	VU1.fmacreadpos  = 0;
	VU1.fmaccount    = 0;
	VU1.ialuwritepos = 0;
	VU1.ialureadpos  = 0;
	VU1.ialucount    = 0;

	std::memset(s_blocks, 0, sizeof(s_blocks));
	if (s_code_base)
		s_code_write = s_code_base;
	s_pool.Reset();
}

void recArmVU1::SetStartPC(u32 startPC)
{
	VU1.start_pc = startPC;
}

void recArmVU1::Step()
{
	VU1.VI[REG_TPC].UL &= VU1_PROGMASK;
	vu1Exec(&VU1);
}

void recArmVU1::Execute(u32 cycles)
{
	const FPControlRegisterBackup fpcr_backup(EmuConfig.Cpu.VU1FPCR);

	VU1.VI[REG_TPC].UL <<= 3;
	const u64 startcycles = VU1.cycle;

	while ((VU1.cycle - startcycles) < cycles)
	{
		if (!(VU0.VI[REG_VPU_STAT].UL & 0x100))
		{
			if (VU1.branch == 1)
			{
				VU1.VI[REG_TPC].UL = VU1.branchpc;
				VU1.branch = 0;
			}
			break;
		}

		const u32 pc   = VU1.VI[REG_TPC].UL & (VU1_PROGSIZE - 1);
		const u32 slot = pc / 8;

		VU1BlockEntry& blk = s_blocks[slot];

		if (!blk.codeEntry)
		{
			const u32 numPairs = AnalyzeBlock(pc);
			blk.numPairs  = numPairs;
			blk.codeEntry = CompileBlock(pc, numPairs);
		}

		using BlockFn = void (*)();
		reinterpret_cast<BlockFn>(blk.codeEntry)();
	}

	VU1.VI[REG_TPC].UL >>= 3;
	VU1.nextBlockCycles = (VU1.cycle - cpuRegs.cycle) + 1;
}

void recArmVU1::Clear(u32 addr, u32 size)
{
	const u32 first        = addr / 8;
	const u32 last         = (addr + size + 7) / 8;
	const u32 clamped_last = std::min(last, VU1_NUM_SLOTS);

	if (first >= VU1_NUM_SLOTS)
		return;

	for (u32 i = first; i < clamped_last; i++)
		s_blocks[i].codeEntry = nullptr;
}
