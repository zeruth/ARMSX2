// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Main driver.
// Adapted from the VU1 ARM64 recompiler. VU0 has 4KB micro memory,
// no XGKICK, and different VPU_STAT bits.

#include "Common.h"
#include "GS.h"
#include "Memory.h"
#include "VIF.h"
#include "VUmicro.h"
#include "VUops.h"
#include "arm64/AsmHelpers.h"
#include "arm64/arm64Emitter.h"
#include "arm64/iVU0micro_arm64.h"

#include <cstring>

using namespace vixl::aarch64;

// Global instance
recArmVU0 CpuArmVU0;

// VU0 per-cycle interpreter entry point
extern void vu0Exec(VURegs* VU);

// Flush helpers
extern void _vuFlushAll(VURegs* VU);

// ============================================================================
//  Rec emitter dispatch tables — reuse VU1's tables since the instruction
//  set is identical. The tables emit code that operates through a pinned
//  base register (x23), which we point at &VU0 instead of &VU1.
//  The only caveat: interpreter-stub entries call VU1_UPPER/LOWER_OPCODE.
//  For VU0 we need VU0_UPPER/LOWER_OPCODE. We define VU0-specific tables.
// ============================================================================

using VU0RecFn = void (*)();
extern VU0RecFn recVU0_UpperTable[64];
extern VU0RecFn recVU0_LowerTable[128];

// ============================================================================
//  Block cache
// ============================================================================

static constexpr u32 VU0_NUM_SLOTS       = VU0_PROGSIZE / 8;
static constexpr u32 VU0_MAX_BLOCK_PAIRS = 128;

static constexpr u32 POOL_SIZE = 32 * 1024;

struct VU0BlockEntry
{
	u8*  codeEntry;
	u32  numPairs;
};

static VU0BlockEntry s_blocks[VU0_NUM_SLOTS];
static u8* s_code_base  = nullptr;
static u8* s_code_write = nullptr;
static u8* s_code_end   = nullptr;
static ArmConstantPool s_pool;

// ============================================================================
//  Runtime helpers
// ============================================================================

static void vu0CheckDTBits(u32 upper)
{
	if (upper & 0x10000000) // D flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x4)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x2;
			hwIntcIrq(INTC_VU0);
			VU0.ebit = 1;
		}
	}
	if (upper & 0x08000000) // T flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x8)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x4;
			hwIntcIrq(INTC_VU0);
			VU0.ebit = 1;
		}
	}
}

static void vu0EbitDone(VURegs* VU)
{
	VU->VIBackupCycles = 0;
	_vuFlushAll(VU);
	VU0.VI[REG_VPU_STAT].UL &= ~0x1;
	vif0Regs.stat.VEW = false;
}

static void vu0HandleDelayBranch(VURegs* VU)
{
	if (VU->takedelaybranch)
	{
		VU->branch          = 1;
		VU->branchpc        = VU->delaybranchpc;
		VU->takedelaybranch = false;
	}
}

static void vu0DecrementVIBackup(VURegs* VU, u64 cyclesBefore)
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
//  Block analysis
// ============================================================================

static bool PairHasEbit(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
	return (upper >> 30) & 1;
}

static bool PairHasBranch(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
	if ((upper >> 31) & 1)
		return false;
	const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + pc);
	_VURegsNum lregs{};
	VU0regs_LOWER_OPCODE[lower >> 25](&lregs);
	return lregs.pipe == VUPIPE_BRANCH;
}

static u32 AnalyzeBlock(u32 startPC)
{
	u32 pairs = 0;
	u32 pc    = startPC;

	while (pairs < VU0_MAX_BLOCK_PAIRS)
	{
		const bool ebit   = PairHasEbit(pc);
		const bool branch = PairHasBranch(pc);

		pairs++;
		pc = (pc + 8) & (VU0_PROGSIZE - 1);

		if (ebit || branch)
		{
			pairs++;
			break;
		}
	}

	return pairs;
}

// ============================================================================
//  Block compilation
// ============================================================================

static const auto VU0_BASE_REG = x23;

static u8* CompileBlock(u32 startPC, u32 numPairs)
{
	const size_t data_size    = numPairs * 2 * sizeof(_VURegsNum);
	const size_t code_worst   = static_cast<size_t>(numPairs) * 512 + 64;
	const size_t total_needed = data_size + code_worst;

	if (static_cast<size_t>(s_code_end - s_code_write) < total_needed)
	{
		std::memset(s_blocks, 0, sizeof(s_blocks));
		s_code_write = s_code_base;
		s_pool.Reset();
	}

	u8* const data_base = s_code_write;
	_VURegsNum* const uregs_data = reinterpret_cast<_VURegsNum*>(data_base);
	_VURegsNum* const lregs_data = uregs_data + numPairs;

	std::memset(data_base, 0, data_size);

	{
		u32 pc = startPC;
		for (u32 i = 0; i < numPairs; i++)
		{
			const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
			const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + pc);

			VU0.code = upper;
			VU0regs_UPPER_OPCODE[upper & 0x3f](&uregs_data[i]);

			if (!((upper >> 31) & 1))
			{
				VU0.code = lower;
				VU0regs_LOWER_OPCODE[lower >> 25](&lregs_data[i]);
			}

			pc = (pc + 8) & (VU0_PROGSIZE - 1);
		}
	}

	u8* code_start = data_base + data_size;
	code_start = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(code_start) + 3) & ~3ULL);

	armSetAsmPtr(code_start, static_cast<size_t>(s_code_end - code_start), &s_pool);
	u8* const entry = armStartBlock();

	armAsm->Stp(x29, x30, MemOperand(sp, -32, PreIndex));
	armAsm->Stp(x22, VU0_BASE_REG, MemOperand(sp, 16));
	armAsm->Mov(x29, sp);
	armMoveAddressToReg(VU0_BASE_REG, &VU0);

	const int64_t cycle_off    = (int64_t)offsetof(VURegs, cycle);
	const int64_t code_off     = (int64_t)offsetof(VURegs, code);
	const int64_t branch_off   = (int64_t)offsetof(VURegs, branch);
	const int64_t branchpc_off = (int64_t)offsetof(VURegs, branchpc);
	const int64_t ebit_off     = (int64_t)offsetof(VURegs, ebit);
	const int64_t tpc_off      = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_TPC * (int64_t)sizeof(REG_VI));
	const int64_t regi_off     = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_I   * (int64_t)sizeof(REG_VI));
	const int64_t fmacwpos_off = (int64_t)offsetof(VURegs, fmacwritepos);
	const int64_t flags_off    = (int64_t)offsetof(VURegs, flags);
	const int64_t vpu_stat_off = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_VPU_STAT * (int64_t)sizeof(REG_VI));

	// Epilogue label — jumped to when we need early exit mid-block
	Label early_exit;

	u32 pc = startPC;
	for (u32 i = 0; i < numPairs; i++)
	{
		const u32 upper     = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
		const u32 lower     = *reinterpret_cast<const u32*>(VU0.Micro + pc);
		const bool ibit     = (upper >> 31) & 1;
		const bool ebit_set = (upper >> 30) & 1;
		const bool dbit_set = (upper >> 28) & 1;
		const bool tbit_set = (upper >> 27) & 1;
		const _VURegsNum& uregs = uregs_data[i];
		const _VURegsNum& lregs = lregs_data[i];

		const bool vf_hazard = !ibit && uregs.VFwrite != 0 &&
			(lregs.VFread0 == uregs.VFwrite || lregs.VFread1 == uregs.VFwrite);
		const bool vi_hazard = !ibit &&
			(uregs.VIwrite & (1u << REG_CLIP_FLAG)) &&
			(lregs.VIread  & (1u << REG_CLIP_FLAG));
		const bool mbit_set    = ((upper >> 29) & 1) != 0;
		const bool fmac_pipe   = (uregs.pipe == VUPIPE_FMAC) || (lregs.pipe == VUPIPE_FMAC);
		const bool branch_pipe = !ibit && (lregs.pipe == VUPIPE_BRANCH);

		// Group bisects (see arm64Emitter.h). When the corresponding aspect macro
		// is defined, any pair where the aspect applies falls back to vu0Exec for
		// the entire pair instead of running the per-pair native machinery.
		bool fallback = false;
#ifdef INTERP_VU0_PAIR
		fallback = true;
#endif
		// Hazard fallback is always on: native VF/CLIP_FLAG save/restore (the
		// thing _vu0Exec lines 104-158 / microVU does to make lower see pre-upper
		// values) is not yet implemented in this JIT. INTERP_VU0_HAZARD is kept as
		// a documentation handle but does not toggle behavior today.
		if (vf_hazard || vi_hazard) fallback = true;
#ifdef INTERP_VU0_MBIT
		if (mbit_set) fallback = true;
#endif
#ifdef INTERP_VU0_DTBITS
		if (dbit_set || tbit_set) fallback = true;
#endif
#ifdef INTERP_VU0_EBIT
		if (ebit_set) fallback = true;
#endif
#ifdef INTERP_VU0_BRANCH
		if (branch_pipe) fallback = true;
#endif
#ifdef INTERP_VU0_FMAC
		if (fmac_pipe) fallback = true;
#endif

		if (fallback)
		{
			// Per-pair fallback — call vu0Exec for the whole pair. vu0Exec
			// internally bumps cycle and advances TPC by 8 from its current
			// value, so the JIT must NOT touch cycle/TPC here.
			armAsm->Mov(x0, VU0_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu0Exec));
		}
		else
		{
			// 1. VU->cycle++
			armAsm->Ldr(x4, MemOperand(VU0_BASE_REG, cycle_off));
			armAsm->Mov(x22, x4);
			armAsm->Add(x4, x4, 1);
			armAsm->Str(x4, MemOperand(VU0_BASE_REG, cycle_off));

			// 2. Advance TPC
			const u32 new_tpc = (pc + 8) & VU0_PROGMASK;
			armAsm->Mov(w4, new_tpc);
			armAsm->Str(w4, MemOperand(VU0_BASE_REG, tpc_off));

			// 3. E-bit
			if (ebit_set)
			{
				armAsm->Mov(w4, 2u);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, ebit_off));
			}

			// 3b. M-bit (early-exit signal to EE — VU0 only)
			// Mirrors _vu0Exec at VU0microInterp.cpp:40-44 and microVU at
			// microVU_Compile.inl:892. Without this, the EE never observes
			// VUFLAG_MFLAGSET and waits forever for VU0 M-bit completion.
			if (mbit_set)
			{
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, flags_off));
				armAsm->Orr(w4, w4, VUFLAG_MFLAGSET);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, flags_off));
			}

			// 4. D/T bits
			if (dbit_set || tbit_set)
			{
				armAsm->Mov(w0, upper);
				armEmitCall(reinterpret_cast<const void*>(vu0CheckDTBits));
			}

			// 5. Upper stalls
			armAsm->Mov(x0, VU0_BASE_REG);
			armMoveAddressToReg(x1, &uregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuTestUpperStalls));

			// 5b. Lower stalls
			if (!ibit)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armMoveAddressToReg(x1, &lregs_data[i]);
				armEmitCall(reinterpret_cast<const void*>(_vuTestLowerStalls));
			}

			// 6. Test pipes
			armAsm->Mov(x0, VU0_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(_vuTestPipes));

			// 6b. VIBackupCycles
			armAsm->Mov(x0, VU0_BASE_REG);
			armAsm->Mov(x1, x22);
			armEmitCall(reinterpret_cast<const void*>(vu0DecrementVIBackup));

			// 7. Upper instruction
			armAsm->Mov(w4, upper);
			armAsm->Str(w4, MemOperand(VU0_BASE_REG, code_off));
			VU0.code = upper;
			recVU0_UpperTable[upper & 0x3f]();

			// 8. Lower instruction
			if (ibit)
			{
				armAsm->Mov(w4, lower);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, regi_off));
			}
			else
			{
				armAsm->Mov(w4, lower);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, code_off));
				VU0.code = lower;
				recVU0_LowerTable[lower >> 25]();
			}

			// 9. FMAC clear
			if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(_vuClearFMAC));
			}

			// 10. Upper stalls add
			armAsm->Mov(x0, VU0_BASE_REG);
			armMoveAddressToReg(x1, &uregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuAddUpperStalls));

			// 11. Lower stalls add
			if (!ibit)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armMoveAddressToReg(x1, &lregs_data[i]);
				armEmitCall(reinterpret_cast<const void*>(_vuAddLowerStalls));
			}

			// 12. Branch countdown
			{
				Label skip_branch;
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, branch_off));
				armAsm->Cbz(w4, &skip_branch);
				armAsm->Subs(w4, w4, 1);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, branch_off));
				armAsm->B(&skip_branch, ne);
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, branchpc_off));
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, tpc_off));
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu0HandleDelayBranch));
				armAsm->Bind(&skip_branch);
			}

			// 13. Ebit countdown
			{
				Label skip_ebit;
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, ebit_off));
				armAsm->Cbz(w4, &skip_ebit);
				armAsm->Subs(w4, w4, 1);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, ebit_off));
				armAsm->B(&skip_ebit, ne);
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu0EbitDone));
				armAsm->Bind(&skip_ebit);
			}

			// 14. FMAC write-position advance
			if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
			{
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, fmacwpos_off));
				armAsm->Add(w4, w4, 1);
				armAsm->And(w4, w4, 3);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, fmacwpos_off));
			}
		}

		pc = (pc + 8) & (VU0_PROGSIZE - 1);

		// After each pair (except last): check VPU_STAT and MFLAGSET
		if (i < numPairs - 1)
		{
			armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, vpu_stat_off));
			armAsm->Tbz(w4, 0, &early_exit);
			armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, flags_off));
			armAsm->Tbnz(w4, 1, &early_exit);
		}
	}

	// Epilogue (also used as early-exit target when MFLAGSET or VPU_STAT fires)
	armAsm->Bind(&early_exit);
	armAsm->Ldp(x22, VU0_BASE_REG, MemOperand(sp, 16));
	armAsm->Ldp(x29, x30, MemOperand(sp, 32, PostIndex));
	armAsm->Ret();

	u8* end = armEndBlock();
	s_code_write = end;
	return entry;
}

// ============================================================================
//  recArmVU0
// ============================================================================

recArmVU0::recArmVU0()
{
	m_Idx = 0;
	IsInterpreter = false;
}

void recArmVU0::Reserve()
{
	u8* const buf     = SysMemory::GetVU0Rec();
	u8* const buf_end = SysMemory::GetVU0RecEnd();

	s_pool.Init(buf, POOL_SIZE);
	s_code_base  = buf + POOL_SIZE;
	s_code_write = s_code_base;
	s_code_end   = buf_end;

	std::memset(s_blocks, 0, sizeof(s_blocks));
}

void recArmVU0::Shutdown()
{
	s_pool.Destroy();
	s_code_base  = nullptr;
	s_code_write = nullptr;
	s_code_end   = nullptr;
	std::memset(s_blocks, 0, sizeof(s_blocks));
}

void recArmVU0::Reset()
{
	VU0.fmacwritepos = 0;
	VU0.fmacreadpos  = 0;
	VU0.fmaccount    = 0;
	VU0.ialuwritepos = 0;
	VU0.ialureadpos  = 0;
	VU0.ialucount    = 0;

	std::memset(s_blocks, 0, sizeof(s_blocks));
	if (s_code_base)
		s_code_write = s_code_base;
	s_pool.Reset();
}

void recArmVU0::SetStartPC(u32 startPC)
{
	VU0.start_pc = startPC;
}

void recArmVU0::Step()
{
	VU0.VI[REG_TPC].UL &= VU0_PROGMASK;
	vu0Exec(&VU0);
}

void recArmVU0::Execute(u32 cycles)
{
	const FPControlRegisterBackup fpcr_backup(EmuConfig.Cpu.VU0FPCR);

	VU0.VI[REG_TPC].UL <<= 3;
	VU0.flags &= ~VUFLAG_MFLAGSET;
	const u64 startcycles = VU0.cycle;

	while ((VU0.cycle - startcycles) < cycles)
	{
		if (!(VU0.VI[REG_VPU_STAT].UL & 0x1))
		{
			if (VU0.branch)
			{
				VU0.VI[REG_TPC].UL = VU0.branchpc;
				VU0.branch = 0;
			}
			break;
		}
		if (VU0.flags & VUFLAG_MFLAGSET)
			break;

		const u32 pc   = VU0.VI[REG_TPC].UL & (VU0_PROGSIZE - 1);
		const u32 slot = pc / 8;

		VU0BlockEntry& blk = s_blocks[slot];

		if (!blk.codeEntry)
		{
			const u32 numPairs = AnalyzeBlock(pc);
			blk.numPairs  = numPairs;
			blk.codeEntry = CompileBlock(pc, numPairs);
		}

		using BlockFn = void (*)();
		reinterpret_cast<BlockFn>(blk.codeEntry)();
	}

	VU0.VI[REG_TPC].UL >>= 3;

	if (EmuConfig.Speedhacks.EECycleRate != 0 && (!EmuConfig.Gamefixes.VUSyncHack || EmuConfig.Speedhacks.EECycleRate < 0))
	{
		u64 cycle_change = VU0.cycle - startcycles;
		VU0.cycle -= cycle_change;
		switch (std::min(static_cast<int>(EmuConfig.Speedhacks.EECycleRate), static_cast<int>(cycle_change)))
		{
			case -3: cycle_change = static_cast<u64>(cycle_change * 2.0f);       break;
			case -2: cycle_change = static_cast<u64>(cycle_change * 1.6666667f); break;
			case -1: cycle_change = static_cast<u64>(cycle_change * 1.3333333f); break;
			case  1: cycle_change = static_cast<u64>(cycle_change / 1.3f);       break;
			case  2: cycle_change = static_cast<u64>(cycle_change / 1.8f);       break;
			case  3: cycle_change = static_cast<u64>(cycle_change / 3.0f);       break;
			default: break;
		}
		VU0.cycle += cycle_change;
	}

	VU0.nextBlockCycles = (VU0.cycle - cpuRegs.cycle) + 1;
}

void recArmVU0::Clear(u32 addr, u32 size)
{
	const u32 first        = addr / 8;
	const u32 last         = (addr + size + 7) / 8;
	const u32 clamped_last = std::min(last, VU0_NUM_SLOTS);

	if (first >= VU0_NUM_SLOTS)
		return;

	for (u32 i = first; i < clamped_last; i++)
		s_blocks[i].codeEntry = nullptr;
}
