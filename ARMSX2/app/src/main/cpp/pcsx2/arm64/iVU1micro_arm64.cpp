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

// Flag-deferral state owned by iVU1Upper_arm64.cpp. Set per-pair before
// dispatching the upper emitter — when false, FMAC arithmetic emitters
// skip the BL vu1_fmac_writeback and inline a NEON clamp + store instead.
extern bool g_vu1NeedsFlags;

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

// (vu1DecrementVIBackup removed — now inlined directly into the per-pair
//  loop via emitDecrementVIBackup, see below.)

// ============================================================================
//  Specialized stall helpers — invoked from JIT with compile-time-constant
//  args. These exist so the per-pair codegen does NOT have to dereference
//  a runtime _VURegsNum* and re-do the pipe switch every pair (the way the
//  generic _vuTest*Stalls / _vuAdd*Stalls helpers do).
//
//  Each helper is called from the JIT only when the corresponding compile-
//  time precondition is true (e.g. vu1_TestFMACStallReg is only emitted
//  when uregs.pipe == VUPIPE_FMAC AND uregs.VFread{0,1} != 0).
// ============================================================================

// Mirrors _vuFMACTestStall in VUops.cpp:210, but takes (reg,xyzw) directly
// in argument registers instead of via _VURegsNum*.
static void vu1_TestFMACStallReg(VURegs* VU, u32 reg, u32 xyzw)
{
	u32 i = 0;
	for (int currentpipe = VU->fmacreadpos; i < VU->fmaccount;
	     currentpipe = (currentpipe + 1) & 3, i++)
	{
		if ((VU->cycle - VU->fmac[currentpipe].sCycle) >= VU->fmac[currentpipe].Cycle)
			continue;

		if ((VU->fmac[currentpipe].regupper == reg && (VU->fmac[currentpipe].xyzwupper & xyzw))
			|| (VU->fmac[currentpipe].reglower == reg && (VU->fmac[currentpipe].xyzwlower & xyzw)))
		{
			u64 newCycle = VU->fmac[currentpipe].Cycle + VU->fmac[currentpipe].sCycle;
			if (newCycle > VU->cycle)
				VU->cycle = newCycle;
		}
	}
}

// FDIV pipe wait portion of _vuTestFDIVStalls (the FMAC test is called
// separately by the JIT when needed).
static void vu1_TestFDIVPipeWait(VURegs* VU)
{
	if (VU->fdiv.enable != 0)
	{
		u64 newCycle = VU->fdiv.Cycle + VU->fdiv.sCycle;
		if (newCycle > VU->cycle)
			VU->cycle = newCycle;
	}
}

// EFU pipe wait portion of _vuTestEFUStalls. NOTE: this mutates
// efu.Cycle (decrements by 1) — see the comment in VUops.cpp:269 for why.
static void vu1_TestEFUPipeWait(VURegs* VU)
{
	if (VU->efu.enable == 0)
		return;
	VU->efu.Cycle -= 1;
	u64 newCycle = VU->efu.sCycle + VU->efu.Cycle;
	if (newCycle > VU->cycle)
		VU->cycle = newCycle;
}

// Mirrors _vuTestALUStalls (VUops.cpp:278) — takes the constant VIread mask
// directly. Used for branch instructions (VUPIPE_BRANCH lower).
static void vu1_TestALUStallReg(VURegs* VU, u32 VIread)
{
	u32 i = 0;
	for (int currentpipe = VU->ialureadpos; i < VU->ialucount;
	     currentpipe = (currentpipe + 1) & 3, i++)
	{
		if ((VU->cycle - VU->ialu[currentpipe].sCycle) >= VU->ialu[currentpipe].Cycle)
			continue;

		if (VU->ialu[currentpipe].reg & VIread)
		{
			u64 newCycle = VU->ialu[currentpipe].Cycle + VU->ialu[currentpipe].sCycle;
			if (newCycle > VU->cycle)
				VU->cycle = newCycle;
		}
	}
}

// Combined ClearFMAC + AddUpperStalls + AddLowerStalls into a single helper
// for FMAC pairs. Replaces three BL calls with one. The ABI is:
//   x0 = VU
//   w1 = regUpper  (uregs.VFwrite if upper pipe is FMAC, else 0)
//   w2 = xyzwUpper (uregs.VFwxyzw if upper pipe is FMAC, else 0)
//   w3 = regLower  (lregs.VFwrite if lower pipe is FMAC, else 0)
//   w4 = xyzwLower (lregs.VFwxyzw if lower pipe is FMAC, else 0)
//   w5 = flagregBoth (upper.VIwrite | lower.VIwrite if both contributing,
//                     matching the OR in _vuAddFMACStalls' isUpper=false branch)
static void vu1_FMACAddPair(VURegs* VU, u32 regUpper, u32 xyzwUpper,
                             u32 regLower, u32 xyzwLower, u32 flagregBoth)
{
	const int i = VU->fmacwritepos;
	VU->fmac[i].regupper   = regUpper;
	VU->fmac[i].xyzwupper  = xyzwUpper;
	VU->fmac[i].reglower   = regLower;
	VU->fmac[i].xyzwlower  = xyzwLower;
	VU->fmac[i].flagreg    = static_cast<int>(flagregBoth);
	VU->fmac[i].sCycle     = VU->cycle;
	VU->fmac[i].Cycle      = 4;
	VU->fmac[i].macflag    = VU->macflag;
	VU->fmac[i].statusflag = VU->statusflag;
	VU->fmac[i].clipflag   = VU->clipflag;
	VU->fmaccount++;
}

// FDIV pipe add — only called when the JIT has determined at compile time
// that lregs.pipe == VUPIPE_FDIV AND (lregs.VIwrite & (1 << REG_Q)) != 0.
// cycles is a compile-time constant (lregs.cycles).
static void vu1_FDIVAdd(VURegs* VU, int cycles)
{
	VU->fdiv.enable = 1;
	VU->fdiv.sCycle = VU->cycle;
	VU->fdiv.Cycle  = cycles;
	VU->fdiv.reg.F  = VU->q.F;
	VU->fdiv.statusflag = VU->statusflag;
}

// EFU pipe add — only called when lregs.pipe == VUPIPE_EFU AND VIwrite has
// REG_P bit set. cycles is compile-time constant.
static void vu1_EFUAdd(VURegs* VU, int cycles)
{
	VU->efu.enable = 1;
	VU->efu.sCycle = VU->cycle;
	VU->efu.Cycle  = cycles;
	VU->efu.reg.F  = VU->p.F;
}

// IALU pipe add — only called when lregs.pipe == VUPIPE_IALU AND cycles != 0.
// cycles and VIwrite are compile-time constants.
static void vu1_IALUAdd(VURegs* VU, int cycles, u32 VIwrite)
{
	const int i = VU->ialuwritepos;
	VU->ialu[i].sCycle = VU->cycle;
	VU->ialu[i].Cycle  = cycles;
	VU->ialu[i].reg    = static_cast<int>(VIwrite);
	VU->ialuwritepos = (VU->ialuwritepos + 1) & 3;
	VU->ialucount++;
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

// ============================================================================
//  Inline emit helpers for per-pair housekeeping
//
//  These replace the BL _vuTest*Stalls / BL _vuClearFMAC / BL _vuAdd*Stalls /
//  BL vu1DecrementVIBackup calls with compile-time-specialized inline code.
//  Most pipes (NOP, MOVE, LQ, IADD, FCAND, ...) end up emitting *zero*
//  instructions for stall housekeeping; only FMAC/FDIV/EFU/IALU/BRANCH
//  pipes emit real work.
//
//  All helpers assume:
//    x23 = &VU1 (VU1_BASE_REG, pinned for the entire block)
//    x22 = cyclesBefore (set by step 1 of every pair)
//    x4-x7, x0-x3 are scratch (clobbered freely)
// ============================================================================

// Emit BL vu1_TestFMACStallReg(VU, reg, xyzw) only when reg != 0.
// This is the inner-loop FMAC hazard test. Both VFread0 and VFread1 may
// require a call.
static void emitFMACStallChecks(const _VURegsNum& regs)
{
	if (regs.VFread0 != 0)
	{
		armAsm->Mov(x0, VU1_BASE_REG);
		armAsm->Mov(w1, regs.VFread0);
		armAsm->Mov(w2, regs.VFr0xyzw);
		armEmitCall(reinterpret_cast<const void*>(vu1_TestFMACStallReg));
	}
	if (regs.VFread1 != 0)
	{
		armAsm->Mov(x0, VU1_BASE_REG);
		armAsm->Mov(w1, regs.VFread1);
		armAsm->Mov(w2, regs.VFr1xyzw);
		armEmitCall(reinterpret_cast<const void*>(vu1_TestFMACStallReg));
	}
}

// Inline replacement for BL _vuTestUpperStalls.
// Upper instructions only have an FMAC pipe; everything else is a no-op.
static void emitTestUpperStalls(const _VURegsNum& uregs)
{
	if (uregs.pipe == VUPIPE_FMAC)
		emitFMACStallChecks(uregs);
}

// Inline replacement for BL _vuTestLowerStalls.
// Lower instructions can be FMAC, FDIV, EFU, or BRANCH (ALU). Other pipes
// (IALU, NONE) are no-ops.
static void emitTestLowerStalls(const _VURegsNum& lregs)
{
	switch (lregs.pipe)
	{
		case VUPIPE_FMAC:
			emitFMACStallChecks(lregs);
			break;
		case VUPIPE_FDIV:
			emitFMACStallChecks(lregs);
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1_TestFDIVPipeWait));
			break;
		case VUPIPE_EFU:
			emitFMACStallChecks(lregs);
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1_TestEFUPipeWait));
			break;
		case VUPIPE_BRANCH:
			// Unconditional B/BAL have VIread == 0; the ALU stall test
			// would be a no-op, so skip the BL entirely.
			if (lregs.VIread != 0)
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armAsm->Mov(w1, lregs.VIread);
				armEmitCall(reinterpret_cast<const void*>(vu1_TestALUStallReg));
			}
			break;
		default:
			break;
	}
}

// Inline replacement for BL vu1DecrementVIBackup.
// VIBackupCycles is a u8 field; in the common case it's 0 and we skip the
// whole block via CBZ. Otherwise we compute (VU->cycle - x22) and saturate.
//
// Uses w4, w5, x6 as scratch (all caller-saved).
static void emitDecrementVIBackup(int64_t cycle_off, int64_t vibackup_off)
{
	Label skip;

	// w4 = VIBackupCycles (zero-extended from u8)
	armAsm->Ldrb(w4, MemOperand(VU1_BASE_REG, vibackup_off));
	armAsm->Cbz(w4, &skip);

	// x6 = VU->cycle (u64)
	armAsm->Ldr(x6, MemOperand(VU1_BASE_REG, cycle_off));
	// x6 = elapsed = VU->cycle - cyclesBefore (cyclesBefore is in x22)
	armAsm->Sub(x6, x6, x22);

	// elapsed is at most ~few; w6 is fine. Compare against u8 VIBackupCycles in w4.
	armAsm->Cmp(w6, w4);
	Label do_subtract;
	armAsm->B(&do_subtract, lo); // elapsed < VIBackupCycles
	// elapsed >= VIBackupCycles → store 0
	armAsm->Strb(wzr, MemOperand(VU1_BASE_REG, vibackup_off));
	armAsm->B(&skip);

	armAsm->Bind(&do_subtract);
	armAsm->Sub(w4, w4, w6);
	armAsm->Strb(w4, MemOperand(VU1_BASE_REG, vibackup_off));

	armAsm->Bind(&skip);
}

// Inline replacement for BL _vuClearFMAC + BL _vuAddUpperStalls + BL
// _vuAddLowerStalls. Combined into a single BL when at least one side
// is FMAC. Returns true if anything was emitted (the caller still needs
// to handle non-FMAC lower pipes separately).
static void emitFMACAddPair(const _VURegsNum& uregs, const _VURegsNum& lregs)
{
	const bool upperFMAC = (uregs.pipe == VUPIPE_FMAC);
	const bool lowerFMAC = (lregs.pipe == VUPIPE_FMAC);
	if (!upperFMAC && !lowerFMAC)
		return;

	const u32 regUpper   = upperFMAC ? uregs.VFwrite  : 0u;
	const u32 xyzwUpper  = upperFMAC ? uregs.VFwxyzw  : 0u;
	const u32 regLower   = lowerFMAC ? lregs.VFwrite  : 0u;
	const u32 xyzwLower  = lowerFMAC ? lregs.VFwxyzw  : 0u;
	const u32 flagregBoth = (upperFMAC ? uregs.VIwrite : 0u) |
	                        (lowerFMAC ? lregs.VIwrite : 0u);

	armAsm->Mov(x0, VU1_BASE_REG);
	armAsm->Mov(w1, regUpper);
	armAsm->Mov(w2, xyzwUpper);
	armAsm->Mov(w3, regLower);
	armAsm->Mov(w4, xyzwLower);
	armAsm->Mov(w5, flagregBoth);
	armEmitCall(reinterpret_cast<const void*>(vu1_FMACAddPair));
}

// Inline replacement for BL _vuAddLowerStalls when the lower pipe is
// NOT FMAC (FMAC is handled by emitFMACAddPair above). Handles FDIV/EFU/IALU.
static void emitLowerNonFMACAdd(const _VURegsNum& lregs)
{
	switch (lregs.pipe)
	{
		case VUPIPE_FDIV:
			if (lregs.VIwrite & (1u << REG_Q))
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armAsm->Mov(w1, lregs.cycles);
				armEmitCall(reinterpret_cast<const void*>(vu1_FDIVAdd));
			}
			break;
		case VUPIPE_EFU:
			if (lregs.VIwrite & (1u << REG_P))
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armAsm->Mov(w1, lregs.cycles);
				armEmitCall(reinterpret_cast<const void*>(vu1_EFUAdd));
			}
			break;
		case VUPIPE_IALU:
			if (lregs.cycles != 0)
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armAsm->Mov(w1, lregs.cycles);
				armAsm->Mov(w2, lregs.VIwrite);
				armEmitCall(reinterpret_cast<const void*>(vu1_IALUAdd));
			}
			break;
		default:
			break;
	}
}

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

	// --- Flag-deferral analysis ---
	// For each FMAC pair, determine whether its MAC/STATUS flag updates are
	// observable. Two reasons to keep them:
	//   (a) Some later same-block pair reads MAC/STATUS/CLIP via FMxxx/FSxxx/
	//       FCxxx (detected via lregs.VIread bits).
	//   (b) The pair is one of the LAST 4 FMAC ops in the block — the FMAC
	//       pipe has 4 slots and ~4-cycle latency, so these writes have not
	//       reached VI[FLAG] before the block ends; the next block's
	//       _vuTestPipes will flush them.
	//
	// When neither holds, the FMAC arithmetic emitters skip BL vu1_fmac_writeback
	// entirely and emit a NEON clamp + store instead — typically 5-7 instructions
	// instead of a function call doing per-lane flag math.
	bool pair_needs_flags[VU1_MAX_BLOCK_PAIRS];
	{
		constexpr u32 FLAG_READ_MASK = (1u << REG_MAC_FLAG)
		                              | (1u << REG_STATUS_FLAG)
		                              | (1u << REG_CLIP_FLAG);
		bool sawFlagReader = false; // any pair > current reads flags
		int  fmacFromEnd   = 0;     // count of FMAC pairs at indices > current
		for (int i = static_cast<int>(numPairs) - 1; i >= 0; i--)
		{
			const _VURegsNum& uregs = uregs_data[i];
			const _VURegsNum& lregs = lregs_data[i];
			const bool isFmacPair = (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC);

			bool needsFlags = false;
			if (isFmacPair)
			{
				if (fmacFromEnd < 4 || sawFlagReader)
					needsFlags = true;
				fmacFromEnd++;
			}
			pair_needs_flags[i] = needsFlags;

			// Update sawFlagReader for the NEXT (earlier) iteration. The
			// current pair's own flag read does not pull its own flag write
			// — pipe latency means a same-pair FMxxx reads VI[FLAG] from
			// 4+ cycles ago, not the upper FMAC's just-now-written value.
			if ((uregs.VIread | lregs.VIread) & FLAG_READ_MASK)
				sawFlagReader = true;
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
	const int64_t vibackup_off = (int64_t)offsetof(VURegs, VIBackupCycles);

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

		// Detect every VF/VI hazard that _vu1Exec (VU1microInterp.cpp:108-163)
		// resolves via save/restore or discard. The native machinery does
		// neither, so all four cases must fall back to vu1Exec:
		//
		//   VF: upper writes vfX, lower also writes vfX        -> discard lower
		//   VF: upper writes vfX, lower reads  vfX             -> save/restore VF
		//   CLIP: upper writes CLIP, lower writes CLIP         -> discard lower
		//   CLIP: upper writes CLIP, lower reads  CLIP         -> save/restore CLIP
		//
		// The TPC at this point already equals `pc` (set by the previous pair),
		// so vu1Exec can run directly without adjustment.
		//
		// Without the discard cases, the JIT runs upper then lower
		// sequentially and lower's write silently clobbers upper's FMAC
		// result whenever both target the same VF.
		const bool vf_hazard = !ibit && uregs.VFwrite != 0 &&
			(lregs.VFwrite == uregs.VFwrite ||
			 lregs.VFread0 == uregs.VFwrite ||
			 lregs.VFread1 == uregs.VFwrite);
		const bool vi_hazard = !ibit &&
			(uregs.VIwrite & (1u << REG_CLIP_FLAG)) &&
			((lregs.VIwrite & (1u << REG_CLIP_FLAG)) ||
			 (lregs.VIread  & (1u << REG_CLIP_FLAG)));

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

		// 5. Test upper stalls — compile-time-specialized inline. Most upper
		//    instructions are non-FMAC and emit zero work here.
		emitTestUpperStalls(uregs);

		// 5b. Test lower stalls BEFORE TestPipes (non-I-bit only).
		//     TestLowerStalls may advance VU->cycle (FDIV/EFU/ALU stalls);
		//     TestPipes needs to see the updated cycle to flush FMAC correctly.
		if (!ibit)
			emitTestLowerStalls(lregs);

		// 6. Test pipes (always, after lower stalls for non-I-bit).
		armAsm->Mov(x0, VU1_BASE_REG);
		armEmitCall(reinterpret_cast<const void*>(_vuTestPipes));

		// 6b. Decrement VIBackupCycles (needed for correct VI backup reads
		//     in branch instructions). x22 holds cycle value before this pair.
		//     Inlined: common case is VIBackupCycles==0 → CBZ skips entire block.
		emitDecrementVIBackup(cycle_off, vibackup_off);

		// 7. Execute upper instruction.
		//    Set VU->code at runtime (interpreter reads it for register fields).
		//    Set VU1.code at compile time so the rec emitter resolves the correct
		//    interpreter function pointer via VU1_UPPER_OPCODE[code & 0x3f].
		armAsm->Mov(w4, upper);
		armAsm->Str(w4, MemOperand(VU1_BASE_REG, code_off));
		VU1.code = upper; // compile-time context for the rec emitter
		g_vu1NeedsFlags = pair_needs_flags[i]; // flag-deferral hint for FMAC emitters
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

		// 9-11. FMAC clear + AddUpperStalls + AddLowerStalls fused.
		//       emitFMACAddPair handles ClearFMAC + the FMAC sides of
		//       AddUpper/AddLowerStalls in a single BL (skipped entirely
		//       when neither side is FMAC). emitLowerNonFMACAdd handles
		//       FDIV/EFU/IALU adds for non-FMAC lower pipes.
		//       For I-bit pairs lregs is all-zero (pipe == VUPIPE_NONE),
		//       so passing it directly is safe — both helpers no-op on it.
		emitFMACAddPair(uregs, lregs);
		if (!ibit)
			emitLowerNonFMACAdd(lregs);

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
