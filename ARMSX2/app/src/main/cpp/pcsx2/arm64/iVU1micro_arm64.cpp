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

// Deferred XGKICK fire helper — defined in iVU1Lower_arm64.cpp.
// Called one pair after an XGKICK to match microVU's 1-pair delay semantics.
extern void vu1_XGKICK_fire_deferred(VURegs* VU);

// Recognize XGKICK by raw lower opcode word. Dispatch path is
//   recVU1_LowerTable[0x40] -> recVU1_LowerOP_Table[0x3C] (T3_00)
//     -> recVU1_LowerOP_T3_00_Table[0x1B] = recVU1_XGKICK
// so the unique bit pattern is:
//   (lower >> 25) == 0x40, (lower & 0x3f) == 0x3C, ((lower >> 6) & 0x1f) == 0x1B
static inline bool isXgkickOp(u32 lower)
{
	return ((lower >> 25) == 0x40u) &&
	       ((lower & 0x3fu) == 0x3Cu) &&
	       (((lower >> 6) & 0x1fu) == 0x1Bu);
}

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
extern u32 g_vu1CurrentPC;

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

// Set by vu1EbitDone when the ebit countdown reaches 0 (microprogram finished).
// Under non-MTVU this is redundant with the VPU_STAT 0x100 clear, but under
// THREAD_VU1 we can't touch VPU_STAT from the VU thread (cross-thread race
// on EE state), so the dispatch loop uses this flag to break instead.
// Reset at the top of recArmVU1::Execute. Single-writer, single-reader on
// the same thread (VU thread under MTVU, EE thread otherwise).
static bool s_vu1_program_ended = false;

// ============================================================================
//  Runtime helper functions called from compiled blocks
// ============================================================================

// Check D/T bits at runtime (depends on FBRST which is a runtime value).
//
// Under THREAD_VU1 this runs on the MTVU thread, so:
//   1. Read vu1Thread.vuFBRST (EE-thread snapshot sent via ExecuteVU) rather
//      than VU0.VI[REG_FBRST].UL (live EE-thread state — cross-thread race).
//   2. Do NOT write VU0.VI[REG_VPU_STAT] or call hwIntcIrq() from the VU
//      thread. Instead, atomically OR a flag into vu1Thread.mtvuInterrupts;
//      Get_MTVUChanges() (MTVU.cpp:351) processes it on the EE thread after
//      the MTVU execute completes, doing the VPU_STAT update and the IRQ
//      raise there.
// Mirrors x86 microVU's mVUTBit / mVUEBit + mVUDTendProgram path
// (microVU_Misc.inl:272-282, microVU_Branch.inl:335-375).
// D-bit under MTVU ends the program via InterruptFlagVUEBit (no IRQ) — same
// as x86 microVU's D-bit path, which calls mVUDTendProgram → mVUEBit.
static void vu1CheckDTBits(u32 upper)
{
	const u32 fbrst = THREAD_VU1 ? vu1Thread.vuFBRST : VU0.VI[REG_FBRST].UL;

	if (upper & 0x10000000) // D flag
	{
		if (fbrst & 0x400)
		{
			if (THREAD_VU1)
			{
				vu1Thread.mtvuInterrupts.fetch_or(VU_Thread::InterruptFlagVUEBit, std::memory_order_release);
			}
			else
			{
				VU0.VI[REG_VPU_STAT].UL |= 0x200;
				hwIntcIrq(INTC_VU1);
			}
			VU1.ebit = 1;
		}
	}
	if (upper & 0x08000000) // T flag
	{
		if (fbrst & 0x800)
		{
			if (THREAD_VU1)
			{
				vu1Thread.mtvuInterrupts.fetch_or(VU_Thread::InterruptFlagVUTBit, std::memory_order_release);
			}
			else
			{
				VU0.VI[REG_VPU_STAT].UL |= 0x400;
				hwIntcIrq(INTC_VU1);
			}
			VU1.ebit = 1;
		}
	}
}

// End-of-microprogram cleanup (called when ebit countdown hits 0).
//
// No XGKICK drain here: step 13 (this call) runs before step 14/15 within
// the same pair, so a pair-local pending kick is always drained by step 15
// or by the block-end drain in CompileBlock. And since vu1_XGKICK no longer
// touches VU1.xgkickenable, there's no legacy interpreter-style pending
// state to clean up on our behalf.
static void vu1EbitDone(VURegs* VU)
{
	VU->VIBackupCycles = 0;
	_vuFlushAll(VU);
	// VPU_STAT running bit + VEW: under THREAD_VU1, vu1ExecMicro on the EE
	// thread already cleared VPU_STAT (VU1micro.cpp:52) before queuing the
	// MTVU execute, and VEW is owned by the VIF DMA path (Vif1_Dma.cpp:258).
	// Writing them from here under MTVU is a cross-thread race on EE state.
	// x86 microVU doesn't touch either at end-of-microprogram; we match that.
	if (!THREAD_VU1)
	{
		VU0.VI[REG_VPU_STAT].UL &= ~0x100;
		vif1Regs.stat.VEW = false;
	}
	// Signal the dispatch loop that the microprogram is finished. Under
	// non-MTVU the VPU_STAT clear above also breaks the loop, but that
	// gate is unreliable under THREAD_VU1 (VPU_STAT 0x100 is cleared on
	// the EE side before queueing and, with INSTANT_VU1, is never re-set),
	// so the loop uses this flag as its termination signal.
	s_vu1_program_ended = true;
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

// Stage C1 (2026-04-11): vu1_FMACAddPair / vu1_FDIVAdd / vu1_EFUAdd /
// vu1_IALUAdd were removed. The corresponding per-pair pipeline adds are
// now emitted as inline store sequences by emitFMACAddPair and
// emitLowerNonFMACAdd below, eliminating four BL round-trips per pair.

// VU1-specialized _vuTestPipes. Mirrors the body of _vuTestPipes / _vuFMACflush /
// _vuFDIVflush / _vuEFUflush / _vuIALUflush from VUops.cpp, with two deletions:
//
//  1. No XGKICK transfer block. The arm64 rec bypasses VU1.xgkickenable via
//     the vu1_XGKICK capture hack (see project_rec_vu1_xgkick_hack) — kicks
//     are fired one pair later by vu1_XGKICK_fire_deferred. `_vuTestPipes`'s
//     `if (VU1.xgkickenable) _vuXGKICKTransfer(...)` would never trigger for
//     us anyway, so eliding it avoids a dead load+branch every pair.
//
//  2. No `do { } while (flushed)` retry loop. None of the four flush functions
//     enqueue anything, so a single pass is always equivalent to the fixpoint.
//
// Called once per pair (step 6 of CompileBlock), so keeping it tight matters.
// Kept in sync with the originals — when any of the VUops.cpp flush bodies
// change, this must be updated too.
static void vu1_TestPipes_VU1(VURegs* VU)
{
	// --- FMAC flush ---
	for (int i = VU->fmacreadpos; VU->fmaccount > 0; i = (i + 1) & 3)
	{
		if ((VU->cycle - VU->fmac[i].sCycle) < VU->fmac[i].Cycle)
			break;

		if (VU->fmac[i].flagreg & (1 << REG_CLIP_FLAG))
			VU->VI[REG_CLIP_FLAG].UL = VU->fmac[i].clipflag;

		if (VU->fmac[i].flagreg & (1 << REG_STATUS_FLAG))
			VU->VI[REG_STATUS_FLAG].UL = (VU->VI[REG_STATUS_FLAG].UL & 0x30)
				| (VU->fmac[i].statusflag & 0xFC0)
				| (VU->fmac[i].statusflag & 0xF);
		else
			VU->VI[REG_STATUS_FLAG].UL = (VU->VI[REG_STATUS_FLAG].UL & 0xFF0)
				| (VU->fmac[i].statusflag & 0xF)
				| ((VU->fmac[i].statusflag & 0xF) << 6);
		VU->VI[REG_MAC_FLAG].UL = VU->fmac[i].macflag;

		VU->fmacreadpos = (VU->fmacreadpos + 1) & 3;
		VU->fmaccount--;
	}

	// --- FDIV flush ---
	if (VU->fdiv.enable != 0
		&& (VU->cycle - VU->fdiv.sCycle) >= VU->fdiv.Cycle)
	{
		VU->fdiv.enable = 0;
		VU->VI[REG_Q].UL = VU->fdiv.reg.UL;
		VU->VI[REG_STATUS_FLAG].UL = (VU->VI[REG_STATUS_FLAG].UL & 0xFCF)
			| (VU->fdiv.statusflag & 0xC30);
	}

	// --- EFU flush ---
	if (VU->efu.enable != 0
		&& (VU->cycle - VU->efu.sCycle) >= VU->efu.Cycle)
	{
		VU->efu.enable = 0;
		VU->VI[REG_P].UL = VU->efu.reg.UL;
	}

	// --- IALU flush (pop only, no flag writes) ---
	for (int i = VU->ialureadpos; VU->ialucount > 0; i = (i + 1) & 3)
	{
		if ((VU->cycle - VU->ialu[i].sCycle) < VU->ialu[i].Cycle)
			break;
		VU->ialureadpos = (VU->ialureadpos + 1) & 3;
		VU->ialucount--;
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

// Stage C2 (2026-04-11): pinned VU->cycle register for the duration of a
// compiled block. Loaded once at block entry, used directly by step 1
// (cycle++), step 6b (VIBackup decrement), and every inline pipeline add
// (FMAC/FDIV/EFU/IALU sCycle store). Flushed back to memory before any BL
// that reads or writes `VU->cycle`, and reloaded afterwards when the BL
// may have mutated it. Block-end flushes back to memory before restoring
// the caller's x21. Callee-saved — BLs won't clobber it.
static const auto VU1_CYCLE_REG = x21;

// Emit `Str x21, [VU1_BASE, cycle_off]`. Call immediately before a BL that
// reads `VU->cycle`.
static void emitFlushCycleReg(int64_t cycle_off)
{
	armAsm->Str(VU1_CYCLE_REG, MemOperand(VU1_BASE_REG, cycle_off));
}

// Emit `Ldr x21, [VU1_BASE, cycle_off]`. Call immediately after a BL that
// may have mutated `VU->cycle`.
static void emitReloadCycleReg(int64_t cycle_off)
{
	armAsm->Ldr(VU1_CYCLE_REG, MemOperand(VU1_BASE_REG, cycle_off));
}

// Stage C3 (2026-04-11): pinned VU->fmacwritepos / VU->ialuwritepos registers
// for the duration of a compiled block. Loaded once at block entry, used
// directly by emitFMACAddPair (slot address math) / emitLowerNonFMACAdd (IALU
// slot address + wpos advance) / step 14 (FMAC wpos advance). Flushed before
// and reloaded after vu1Exec (the only BL we emit that reads/writes wpos —
// every other BL touches only fmacreadpos / fmaccount / ialureadpos /
// ialucount, which stay memory-resident). Block-end flushes back before the
// epilogue restores the caller's x24/x25.
//
// These are 32-bit-wide (u32) fields; we use w24/w25 for all arithmetic and
// let the implicit zero-extend-on-32-bit-write rule keep x24/x25 valid as
// the 64-bit form for the slot-address math in emitFMACAddPair /
// emitLowerNonFMACAdd.
static const auto VU1_FMAC_WPOS_REG = w24;
static const auto VU1_IALU_WPOS_REG = w25;

static void emitFlushWposRegs(int64_t fmacwpos_off, int64_t ialuwpos_off)
{
	armAsm->Str(VU1_FMAC_WPOS_REG, MemOperand(VU1_BASE_REG, fmacwpos_off));
	armAsm->Str(VU1_IALU_WPOS_REG, MemOperand(VU1_BASE_REG, ialuwpos_off));
}

static void emitReloadWposRegs(int64_t fmacwpos_off, int64_t ialuwpos_off)
{
	armAsm->Ldr(VU1_FMAC_WPOS_REG, MemOperand(VU1_BASE_REG, fmacwpos_off));
	armAsm->Ldr(VU1_IALU_WPOS_REG, MemOperand(VU1_BASE_REG, ialuwpos_off));
}

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
//    x23 = &VU1       (VU1_BASE_REG, pinned for the entire block)
//    x22 = cyclesBefore (set by step 1 of every pair; Mov'd from x21)
//    x21 = VU->cycle   (VU1_CYCLE_REG, Stage C2 hoisted cycle counter)
//    x4-x7, x0-x3 are scratch (clobbered freely)
//
//  Any helper that emits a BL to a function which reads or writes
//  `VU->cycle` must flush x21 to memory first (emitFlushCycleReg) and, if
//  the BL may have mutated cycle, reload afterwards (emitReloadCycleReg).
// ============================================================================

// Emit BL vu1_TestFMACStallReg(VU, reg, xyzw) only when reg != 0 AND the
// compile-time pipeline tracker has not already proven no FMAC slot aliases
// (skip0/skip1 flags come from Stage A of the mVUregs port — see the
// "Compile-time pipeline state tracking" pre-walk in CompileBlock).
//
// vu1_TestFMACStallReg reads `VU->cycle` and conditionally writes it when
// a stall adjustment is needed, so the Stage C2 cached cycle register
// (x21) must be flushed/reloaded around each BL.
static void emitFMACStallChecks(const _VURegsNum& regs, bool skip0, bool skip1)
{
	const int64_t cycle_off = (int64_t)offsetof(VURegs, cycle);

	if (!skip0 && regs.VFread0 != 0)
	{
		emitFlushCycleReg(cycle_off);
		armAsm->Mov(x0, VU1_BASE_REG);
		armAsm->Mov(w1, regs.VFread0);
		armAsm->Mov(w2, regs.VFr0xyzw);
		armEmitCall(reinterpret_cast<const void*>(vu1_TestFMACStallReg));
		emitReloadCycleReg(cycle_off);
	}
	if (!skip1 && regs.VFread1 != 0)
	{
		emitFlushCycleReg(cycle_off);
		armAsm->Mov(x0, VU1_BASE_REG);
		armAsm->Mov(w1, regs.VFread1);
		armAsm->Mov(w2, regs.VFr1xyzw);
		armEmitCall(reinterpret_cast<const void*>(vu1_TestFMACStallReg));
		emitReloadCycleReg(cycle_off);
	}
}

// Inline replacement for BL _vuTestUpperStalls.
// Upper instructions only have an FMAC pipe; everything else is a no-op.
static void emitTestUpperStalls(const _VURegsNum& uregs, bool skipFMACStall0, bool skipFMACStall1)
{
	if (uregs.pipe == VUPIPE_FMAC)
		emitFMACStallChecks(uregs, skipFMACStall0, skipFMACStall1);
}

// Inline replacement for BL _vuTestLowerStalls.
// Lower instructions can be FMAC, FDIV, EFU, or BRANCH (ALU). Other pipes
// (IALU, NONE) are no-ops. Stage B threads through FDIV/EFU/ALU wait skip
// flags in addition to Stage A's FMAC stall skips.
//
// EFU wait note: vu1_TestEFUPipeWait has a mandatory `efu.Cycle -= 1` side
// effect when enable!=0, so skipping is ONLY sound when the pre-walk proved
// the EFU pipe is entirely empty at this pair (no in-block add AND carry-in
// worst-case retired, gate = 54 cycles). Same reasoning applies to FDIV wait
// (gate 12) and ALU stall check (gate 3).
static void emitTestLowerStalls(const _VURegsNum& lregs,
	bool skipFMACStall0, bool skipFMACStall1,
	bool skipFDIVWait, bool skipEFUWait, bool skipALUStall)
{
	const int64_t cycle_off = (int64_t)offsetof(VURegs, cycle);

	switch (lregs.pipe)
	{
		case VUPIPE_FMAC:
			emitFMACStallChecks(lregs, skipFMACStall0, skipFMACStall1);
			break;
		case VUPIPE_FDIV:
			emitFMACStallChecks(lregs, skipFMACStall0, skipFMACStall1);
			if (!skipFDIVWait)
			{
				emitFlushCycleReg(cycle_off);
				armAsm->Mov(x0, VU1_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu1_TestFDIVPipeWait));
				emitReloadCycleReg(cycle_off);
			}
			break;
		case VUPIPE_EFU:
			emitFMACStallChecks(lregs, skipFMACStall0, skipFMACStall1);
			if (!skipEFUWait)
			{
				emitFlushCycleReg(cycle_off);
				armAsm->Mov(x0, VU1_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu1_TestEFUPipeWait));
				emitReloadCycleReg(cycle_off);
			}
			break;
		case VUPIPE_BRANCH:
			// Unconditional B/BAL have VIread == 0; the ALU stall test
			// would be a no-op, so skip the BL entirely.
			if (!skipALUStall && lregs.VIread != 0)
			{
				emitFlushCycleReg(cycle_off);
				armAsm->Mov(x0, VU1_BASE_REG);
				armAsm->Mov(w1, lregs.VIread);
				armEmitCall(reinterpret_cast<const void*>(vu1_TestALUStallReg));
				emitReloadCycleReg(cycle_off);
			}
			break;
		default:
			break;
	}
}

// Inline replacement for BL vu1DecrementVIBackup.
// VIBackupCycles is a u8 field; in the common case it's 0 and we skip the
// whole block via CBZ. Otherwise we compute (VU->cycle - x22) and saturate.
// Stage C2: the VU->cycle load is skipped — we use the cached VU1_CYCLE_REG
// (x21) directly, which is always up-to-date at this point in the pair
// (step 1 bumped it, the lower-stall BLs have been flushed/reloaded, and
// step 6's TestPipes BL does not write cycle).
//
// Uses w4, w5, x6 as scratch (all caller-saved).
static void emitDecrementVIBackup(int64_t /*cycle_off*/, int64_t vibackup_off)
{
	Label skip;

	// w4 = VIBackupCycles (zero-extended from u8)
	armAsm->Ldrb(w4, MemOperand(VU1_BASE_REG, vibackup_off));
	armAsm->Cbz(w4, &skip);

	// x6 = elapsed = VU->cycle - cyclesBefore
	//       cycle is in VU1_CYCLE_REG (x21), cyclesBefore is in x22.
	armAsm->Sub(x6, VU1_CYCLE_REG, x22);

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

// Stage C1 inline FMAC pipeline add. Writes directly into
// &VU->fmac[fmacwritepos] and bumps fmaccount, matching the body that used
// to live in vu1_FMACAddPair. fmacwritepos itself is advanced in step 14
// (unchanged) — this helper uses the pre-advance value, same as before.
// Stage C2: sCycle is stored directly from the pinned VU1_CYCLE_REG (x21),
// eliminating the `Ldr x6, [VU1_BASE, cycle_off]` that C1 emitted.
// Stage C3: fmacwritepos is read directly from the pinned VU1_FMAC_WPOS_REG
// (x24/w24), eliminating the `Ldr w4, [VU1_BASE, fmacwpos_off]` that C1/C2
// emitted. x24's upper 32 bits are guaranteed zero by the zero-extend-on-
// 32-bit-write rule — every write to w24 in this file is a 32-bit op.
//
// Scratch: w5/x5, x6, x7. x7 ends up holding &VU->fmac[wpos]; the per-field
// offsets (offsetof(fmacPipe, ...)) bake into each Str's MemOperand immediate.
static void emitFMACAddPair(const _VURegsNum& uregs, const _VURegsNum& lregs)
{
	const bool upperFMAC = (uregs.pipe == VUPIPE_FMAC);
	const bool lowerFMAC = (lregs.pipe == VUPIPE_FMAC);
	if (!upperFMAC && !lowerFMAC)
		return;

	const u32 regUpper    = upperFMAC ? uregs.VFwrite  : 0u;
	const u32 xyzwUpper   = upperFMAC ? uregs.VFwxyzw  : 0u;
	const u32 regLower    = lowerFMAC ? lregs.VFwrite  : 0u;
	const u32 xyzwLower   = lowerFMAC ? lregs.VFwxyzw  : 0u;
	const u32 flagregBoth = (upperFMAC ? uregs.VIwrite : 0u) |
	                        (lowerFMAC ? lregs.VIwrite : 0u);

	const int64_t fmac_off       = (int64_t)offsetof(VURegs, fmac);
	const int64_t fmaccount_off  = (int64_t)offsetof(VURegs, fmaccount);
	const int64_t macflag_off    = (int64_t)offsetof(VURegs, macflag);
	const int64_t statusflag_off = (int64_t)offsetof(VURegs, statusflag);
	const int64_t clipflag_off   = (int64_t)offsetof(VURegs, clipflag);

	const int64_t f_regupper   = (int64_t)offsetof(fmacPipe, regupper);
	const int64_t f_reglower   = (int64_t)offsetof(fmacPipe, reglower);
	const int64_t f_flagreg    = (int64_t)offsetof(fmacPipe, flagreg);
	const int64_t f_xyzwupper  = (int64_t)offsetof(fmacPipe, xyzwupper);
	const int64_t f_xyzwlower  = (int64_t)offsetof(fmacPipe, xyzwlower);
	const int64_t f_sCycle     = (int64_t)offsetof(fmacPipe, sCycle);
	const int64_t f_Cycle      = (int64_t)offsetof(fmacPipe, Cycle);
	const int64_t f_macflag    = (int64_t)offsetof(fmacPipe, macflag);
	const int64_t f_statusflag = (int64_t)offsetof(fmacPipe, statusflag);
	const int64_t f_clipflag   = (int64_t)offsetof(fmacPipe, clipflag);

	// x5 = wpos * 48 = (wpos * 3) << 4 — x24 is the zero-extended 64-bit
	// view of w24 (VU1_FMAC_WPOS_REG), safe because every write to w24 is
	// 32-bit and therefore zeroes the top half.
	armAsm->Add(x5, x24, Operand(x24, LSL, 1));
	armAsm->Lsl(x5, x5, 4);
	// x7 = VU1_BASE + wpos*48 (fmac_off baked into each Str's imm).
	armAsm->Add(x7, VU1_BASE_REG, x5);

	auto storeImm32 = [&](u32 value, int64_t field_off) {
		if (value == 0)
		{
			armAsm->Str(wzr, MemOperand(x7, fmac_off + field_off));
		}
		else
		{
			armAsm->Mov(w6, value);
			armAsm->Str(w6, MemOperand(x7, fmac_off + field_off));
		}
	};

	storeImm32(regUpper,    f_regupper);
	storeImm32(regLower,    f_reglower);
	storeImm32(flagregBoth, f_flagreg);
	storeImm32(xyzwUpper,   f_xyzwupper);
	storeImm32(xyzwLower,   f_xyzwlower);

	// sCycle (u64) = VU->cycle — use the cached VU1_CYCLE_REG directly.
	armAsm->Str(VU1_CYCLE_REG, MemOperand(x7, fmac_off + f_sCycle));

	// Cycle = 4 (compile-time constant — FMAC latency is always 4).
	armAsm->Mov(w6, 4);
	armAsm->Str(w6, MemOperand(x7, fmac_off + f_Cycle));

	// macflag / statusflag / clipflag snapshotted from VU->*.
	armAsm->Ldr(w6, MemOperand(VU1_BASE_REG, macflag_off));
	armAsm->Str(w6, MemOperand(x7, fmac_off + f_macflag));
	armAsm->Ldr(w6, MemOperand(VU1_BASE_REG, statusflag_off));
	armAsm->Str(w6, MemOperand(x7, fmac_off + f_statusflag));
	armAsm->Ldr(w6, MemOperand(VU1_BASE_REG, clipflag_off));
	armAsm->Str(w6, MemOperand(x7, fmac_off + f_clipflag));

	// fmaccount++
	armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, fmaccount_off));
	armAsm->Add(w4, w4, 1);
	armAsm->Str(w4, MemOperand(VU1_BASE_REG, fmaccount_off));
}

// Stage C1 inline pipeline add for non-FMAC lower pipes (FDIV/EFU/IALU).
// FMAC lowers are handled by emitFMACAddPair above. All field stores are
// emitted directly into VU->fdiv / VU->efu / VU->ialu[ialuwritepos], so
// there is no BL into a C helper.
//
// Stage C2: all sCycle stores write VU1_CYCLE_REG (x21) directly, skipping
// the `Ldr x4, [VU1_BASE, cycle_off]` that C1 emitted.
//
// Scratch: w4/x4, w5/x5, x6, x7. Matches emitFMACAddPair's scratch usage.
static void emitLowerNonFMACAdd(const _VURegsNum& lregs)
{
	switch (lregs.pipe)
	{
		case VUPIPE_FDIV:
			if (lregs.VIwrite & (1u << REG_Q))
			{
				const int64_t statusflag_off = (int64_t)offsetof(VURegs, statusflag);
				const int64_t q_off          = (int64_t)offsetof(VURegs, q);
				const int64_t fdiv_off       = (int64_t)offsetof(VURegs, fdiv);
				const int64_t d_enable       = (int64_t)offsetof(fdivPipe, enable);
				const int64_t d_reg          = (int64_t)offsetof(fdivPipe, reg);
				const int64_t d_sCycle       = (int64_t)offsetof(fdivPipe, sCycle);
				const int64_t d_Cycle        = (int64_t)offsetof(fdivPipe, Cycle);
				const int64_t d_statusflag   = (int64_t)offsetof(fdivPipe, statusflag);

				// enable = 1
				armAsm->Mov(w4, 1);
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, fdiv_off + d_enable));
				// sCycle (u64) = VU->cycle (cached in x21)
				armAsm->Str(VU1_CYCLE_REG, MemOperand(VU1_BASE_REG, fdiv_off + d_sCycle));
				// Cycle = lregs.cycles (compile-time)
				armAsm->Mov(w4, static_cast<u32>(lregs.cycles));
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, fdiv_off + d_Cycle));
				// reg.F = VU->q.F (first 4 bytes of the REG_VI union)
				armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, q_off));
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, fdiv_off + d_reg));
				// statusflag = VU->statusflag
				armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, statusflag_off));
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, fdiv_off + d_statusflag));
			}
			break;

		case VUPIPE_EFU:
			if (lregs.VIwrite & (1u << REG_P))
			{
				const int64_t p_off    = (int64_t)offsetof(VURegs, p);
				const int64_t efu_off  = (int64_t)offsetof(VURegs, efu);
				const int64_t e_enable = (int64_t)offsetof(efuPipe, enable);
				const int64_t e_reg    = (int64_t)offsetof(efuPipe, reg);
				const int64_t e_sCycle = (int64_t)offsetof(efuPipe, sCycle);
				const int64_t e_Cycle  = (int64_t)offsetof(efuPipe, Cycle);

				armAsm->Mov(w4, 1);
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, efu_off + e_enable));
				armAsm->Str(VU1_CYCLE_REG, MemOperand(VU1_BASE_REG, efu_off + e_sCycle));
				armAsm->Mov(w4, static_cast<u32>(lregs.cycles));
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, efu_off + e_Cycle));
				armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, p_off));
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, efu_off + e_reg));
			}
			break;

		case VUPIPE_IALU:
			if (lregs.cycles != 0)
			{
				const int64_t ialu_off      = (int64_t)offsetof(VURegs, ialu);
				const int64_t ialucount_off = (int64_t)offsetof(VURegs, ialucount);
				const int64_t i_reg         = (int64_t)offsetof(ialuPipe, reg);
				const int64_t i_sCycle      = (int64_t)offsetof(ialuPipe, sCycle);
				const int64_t i_Cycle       = (int64_t)offsetof(ialuPipe, Cycle);

				// Stage C3: ialuwritepos is held live in x25/w25 —
				// x25 is the zero-extended 64-bit view (every write to
				// w25 in this file is 32-bit, which zeros the top half).
				// x5 = wpos * 24 = (wpos * 3) << 3
				armAsm->Add(x5, x25, Operand(x25, LSL, 1));
				armAsm->Lsl(x5, x5, 3);
				// x7 = VU1_BASE + wpos*24
				armAsm->Add(x7, VU1_BASE_REG, x5);

				// sCycle (u64) = VU->cycle (cached in x21)
				armAsm->Str(VU1_CYCLE_REG, MemOperand(x7, ialu_off + i_sCycle));
				// Cycle = lregs.cycles (compile-time)
				armAsm->Mov(w6, static_cast<u32>(lregs.cycles));
				armAsm->Str(w6, MemOperand(x7, ialu_off + i_Cycle));
				// reg = lregs.VIwrite (compile-time)
				armAsm->Mov(w6, lregs.VIwrite);
				armAsm->Str(w6, MemOperand(x7, ialu_off + i_reg));

				// Stage C3: ialuwritepos = (wpos + 1) & 3 — in-register,
				// no memory store. Block-end epilogue flushes x25 back.
				armAsm->Add(VU1_IALU_WPOS_REG, VU1_IALU_WPOS_REG, 1);
				armAsm->And(VU1_IALU_WPOS_REG, VU1_IALU_WPOS_REG, 3);

				// ialucount++ (stays memory-resident — bumped by us,
				// decremented by vu1_TestPipes_VU1 / vu1EbitDone).
				armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, ialucount_off));
				armAsm->Add(w4, w4, 1);
				armAsm->Str(w4, MemOperand(VU1_BASE_REG, ialucount_off));
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

	// --- Compile-time pipeline state tracking (Stages A+B) ---
	// Pre-walk the block to decide which stall-check / TestPipes BLs can be
	// proven unnecessary at compile time. Tracks the four VU pipes:
	//   FMAC (4-slot ring, Cycle=4 fixed)
	//   IALU (4-slot ring, per-slot Cycle)
	//   FDIV (single slot, per-slot Cycle, max 13)
	//   EFU  (single slot, per-slot Cycle, max 54)
	//
	// Soundness: ct_cycle advances by exactly 1 per pair (no stall-induced
	// bumps). Runtime cycle can only LEAD ours (stalls advance runtime but
	// not our model), so for any slot, (runtime_cycle - runtime_sCycle) >=
	// (ct_cycle - ct_sCycle). "Slot absent in our model" implies "slot
	// absent at runtime" — elision is one-way safe.
	//
	// Carry-in: runtime's rings may hold entries at block entry that our
	// model can't see. Each pipe has a "carry-in gate" — ct_cycle threshold
	// past which all possible carry-in is guaranteed retired:
	//   FMAC: > 3  (max Cycle=4, latest delta-sCycle=-1 matures at ct_cycle=3)
	//   IALU: > 3  (max Cycle=4)
	//   FDIV: > 12 (max Cycle=13)
	//   EFU : > 54 (max Cycle=54 for EATAN family)
	// Elision only fires once the relevant gate is cleared.
	struct CTFmacSlot
	{
		u8 regupper, xyzwupper;
		u8 reglower, xyzwlower;
		int sCycle;
		bool valid;
	};
	CTFmacSlot ct_fmac[4] = {};
	int ct_fmac_wpos = 0, ct_fmac_rpos = 0, ct_fmac_count = 0;

	struct CTIaluSlot
	{
		u32 reg;    // VIwrite bits
		int sCycle;
		int cycles;
		bool valid;
	};
	CTIaluSlot ct_ialu[4] = {};
	int ct_ialu_wpos = 0, ct_ialu_rpos = 0, ct_ialu_count = 0;

	bool ct_fdiv_pending = false;
	int  ct_fdiv_sCycle  = 0;
	int  ct_fdiv_cycles  = 0;

	bool ct_efu_pending = false;
	int  ct_efu_sCycle  = 0;
	int  ct_efu_cycles  = 0;

	constexpr int CARRY_IN_GATE_FMAC = 3;
	constexpr int CARRY_IN_GATE_IALU = 3;
	constexpr int CARRY_IN_GATE_FDIV = 12;
	constexpr int CARRY_IN_GATE_EFU  = 54;

	struct PerPairSkip
	{
		bool skipUpperFMACStall0;
		bool skipUpperFMACStall1;
		bool skipLowerFMACStall0;
		bool skipLowerFMACStall1;
		bool skipLowerFDIVWait;
		bool skipLowerEFUWait;
		bool skipLowerALUStall;
		bool skipTestPipes;
	};
	PerPairSkip skip_info[VU1_MAX_BLOCK_PAIRS] = {};

	{
		int ct_cycle = 0;
		u32 pc_walk = startPC;
		for (u32 i = 0; i < numPairs; i++)
		{
			const u32 upper = *reinterpret_cast<const u32*>(VU1.Micro + pc_walk + 4);
			const bool ibit = (upper >> 31) & 1;
			const _VURegsNum& uregs = uregs_data[i];
			const _VURegsNum& lregs = lregs_data[i];

			// step 1: cycle++
			ct_cycle++;

			const bool fmac_carry_safe = (ct_cycle > CARRY_IN_GATE_FMAC);
			const bool ialu_carry_safe = (ct_cycle > CARRY_IN_GATE_IALU);
			const bool fdiv_carry_safe = (ct_cycle > CARRY_IN_GATE_FDIV);
			const bool efu_carry_safe  = (ct_cycle > CARRY_IN_GATE_EFU);

			auto aliasFmac = [&](u8 reg, u8 xyzw) -> bool {
				if (reg == 0)
					return false;
				int idx = ct_fmac_rpos;
				for (int n = 0; n < ct_fmac_count; n++)
				{
					const CTFmacSlot& slot = ct_fmac[idx];
					if (slot.valid)
					{
						if (slot.regupper == reg && (slot.xyzwupper & xyzw))
							return true;
						if (slot.reglower == reg && (slot.xyzwlower & xyzw))
							return true;
					}
					idx = (idx + 1) & 3;
				}
				return false;
			};

			auto aliasIalu = [&](u32 VIread) -> bool {
				if (VIread == 0)
					return false;
				int idx = ct_ialu_rpos;
				for (int n = 0; n < ct_ialu_count; n++)
				{
					const CTIaluSlot& slot = ct_ialu[idx];
					if (slot.valid && (slot.reg & VIread))
						return true;
					idx = (idx + 1) & 3;
				}
				return false;
			};

			// step 5 upper stalls — FMAC only
			if (fmac_carry_safe && uregs.pipe == VUPIPE_FMAC)
			{
				skip_info[i].skipUpperFMACStall0 = !aliasFmac(uregs.VFread0, uregs.VFr0xyzw);
				skip_info[i].skipUpperFMACStall1 = !aliasFmac(uregs.VFread1, uregs.VFr1xyzw);
			}

			// step 5b lower stalls — FMAC/FDIV/EFU do FMAC checks first, plus
			// their respective wait helpers. BRANCH does an ALU stall check.
			if (!ibit)
			{
				const bool lowerDoesFMACCheck =
					(lregs.pipe == VUPIPE_FMAC) ||
					(lregs.pipe == VUPIPE_FDIV) ||
					(lregs.pipe == VUPIPE_EFU);
				if (lowerDoesFMACCheck && fmac_carry_safe)
				{
					skip_info[i].skipLowerFMACStall0 = !aliasFmac(lregs.VFread0, lregs.VFr0xyzw);
					skip_info[i].skipLowerFMACStall1 = !aliasFmac(lregs.VFread1, lregs.VFr1xyzw);
				}

				switch (lregs.pipe)
				{
					case VUPIPE_FDIV:
						// Elide wait BL when FDIV definitely not pending:
						// no in-block add AND carry-in gate cleared.
						skip_info[i].skipLowerFDIVWait = !ct_fdiv_pending && fdiv_carry_safe;
						break;
					case VUPIPE_EFU:
						// vu1_TestEFUPipeWait has a mandatory `efu.Cycle -= 1`
						// side effect when enable!=0, so elision is only safe
						// when we know enable=0 for certain (no in-block add
						// AND carry-in retired — gate 54).
						skip_info[i].skipLowerEFUWait = !ct_efu_pending && efu_carry_safe;
						break;
					case VUPIPE_BRANCH:
						if (lregs.VIread != 0 && ialu_carry_safe)
							skip_info[i].skipLowerALUStall = !aliasIalu(lregs.VIread);
						break;
					default:
						break;
				}
			}

			// step 6 TestPipes: decide elision BEFORE retiring, then retire.
			// The BL is a no-op when nothing matures at this pair AND all
			// pipes' carry-in gates have cleared.
			{
				const bool fmacMature = (ct_fmac_count > 0)
					&& ct_fmac[ct_fmac_rpos].valid
					&& (ct_cycle - ct_fmac[ct_fmac_rpos].sCycle) >= 4;
				const bool fdivMature = ct_fdiv_pending
					&& (ct_cycle - ct_fdiv_sCycle) >= ct_fdiv_cycles;
				const bool efuMature  = ct_efu_pending
					&& (ct_cycle - ct_efu_sCycle) >= ct_efu_cycles;
				const bool ialuMature = (ct_ialu_count > 0)
					&& ct_ialu[ct_ialu_rpos].valid
					&& (ct_cycle - ct_ialu[ct_ialu_rpos].sCycle) >= ct_ialu[ct_ialu_rpos].cycles;

				skip_info[i].skipTestPipes =
					!fmacMature && !fdivMature && !efuMature && !ialuMature
					&& fmac_carry_safe && fdiv_carry_safe
					&& efu_carry_safe  && ialu_carry_safe;
			}

			// Retire mature slots in the CT model.
			while (ct_fmac_count > 0)
			{
				CTFmacSlot& head = ct_fmac[ct_fmac_rpos];
				if (!head.valid || (ct_cycle - head.sCycle) < 4)
					break;
				head.valid = false;
				ct_fmac_rpos = (ct_fmac_rpos + 1) & 3;
				ct_fmac_count--;
			}
			while (ct_ialu_count > 0)
			{
				CTIaluSlot& head = ct_ialu[ct_ialu_rpos];
				if (!head.valid || (ct_cycle - head.sCycle) < head.cycles)
					break;
				head.valid = false;
				ct_ialu_rpos = (ct_ialu_rpos + 1) & 3;
				ct_ialu_count--;
			}
			if (ct_fdiv_pending && (ct_cycle - ct_fdiv_sCycle) >= ct_fdiv_cycles)
				ct_fdiv_pending = false;
			if (ct_efu_pending && (ct_cycle - ct_efu_sCycle) >= ct_efu_cycles)
				ct_efu_pending = false;

			// step 11 adds
			{
				const bool uFMAC = (uregs.pipe == VUPIPE_FMAC);
				const bool lFMAC = !ibit && (lregs.pipe == VUPIPE_FMAC);
				if (uFMAC || lFMAC)
				{
					CTFmacSlot& slot = ct_fmac[ct_fmac_wpos];
					slot.regupper  = uFMAC ? uregs.VFwrite : 0;
					slot.xyzwupper = uFMAC ? uregs.VFwxyzw : 0;
					slot.reglower  = lFMAC ? lregs.VFwrite : 0;
					slot.xyzwlower = lFMAC ? lregs.VFwxyzw : 0;
					slot.sCycle    = ct_cycle;
					slot.valid     = true;
					ct_fmac_wpos = (ct_fmac_wpos + 1) & 3;
					if (ct_fmac_count < 4)
						ct_fmac_count++;
				}
			}

			if (!ibit)
			{
				switch (lregs.pipe)
				{
					case VUPIPE_FDIV:
						if (lregs.VIwrite & (1u << REG_Q))
						{
							ct_fdiv_pending = true;
							ct_fdiv_sCycle  = ct_cycle;
							ct_fdiv_cycles  = lregs.cycles;
						}
						break;
					case VUPIPE_EFU:
						if (lregs.VIwrite & (1u << REG_P))
						{
							ct_efu_pending = true;
							ct_efu_sCycle  = ct_cycle;
							ct_efu_cycles  = lregs.cycles;
						}
						break;
					case VUPIPE_IALU:
						if (lregs.cycles != 0)
						{
							CTIaluSlot& slot = ct_ialu[ct_ialu_wpos];
							slot.reg    = lregs.VIwrite;
							slot.sCycle = ct_cycle;
							slot.cycles = lregs.cycles;
							slot.valid  = true;
							ct_ialu_wpos = (ct_ialu_wpos + 1) & 3;
							if (ct_ialu_count < 4)
								ct_ialu_count++;
						}
						break;
					default:
						break;
				}
			}

			pc_walk = (pc_walk + 8) & (VU1_PROGSIZE - 1);
		}
	}

	// Code section starts after data, 4-byte aligned.
	u8* code_start = data_base + data_size;
	code_start = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(code_start) + 3) & ~3ULL);

	armSetAsmPtr(code_start, static_cast<size_t>(s_code_end - code_start), &s_pool);
	u8* const entry = armStartBlock();

	// --- Prologue: save callee-saved regs, pin VU1_BASE_REG = &VU1 ---
	// 64-byte frame (Stage C3 expanded from 48):
	//   [sp+0..7]   = x29 (fp)
	//   [sp+8..15]  = x30 (lr)
	//   [sp+16..23] = x21 (VU1_CYCLE_REG — Stage C2 cached VU->cycle)
	//   [sp+24..31] = x22 (cyclesBefore scratch)
	//   [sp+32..39] = x23 (VU1_BASE_REG)
	//   [sp+40..47] = x24 (VU1_FMAC_WPOS_REG — Stage C3 cached fmacwritepos)
	//   [sp+48..55] = x25 (VU1_IALU_WPOS_REG — Stage C3 cached ialuwritepos)
	//   [sp+56..63] = unused pad for 16-byte alignment
	armAsm->Stp(x29, x30, MemOperand(sp, -64, PreIndex));
	armAsm->Stp(VU1_CYCLE_REG, x22, MemOperand(sp, 16));
	armAsm->Stp(VU1_BASE_REG, x24, MemOperand(sp, 32));
	armAsm->Str(x25, MemOperand(sp, 48));
	armAsm->Mov(x29, sp);
	armMoveAddressToReg(VU1_BASE_REG, &VU1);

	// Compile-time constants for field offsets used throughout the loop.
	const int64_t cycle_off     = (int64_t)offsetof(VURegs, cycle);
	const int64_t code_off      = (int64_t)offsetof(VURegs, code);
	const int64_t branch_off    = (int64_t)offsetof(VURegs, branch);
	const int64_t branchpc_off  = (int64_t)offsetof(VURegs, branchpc);
	const int64_t ebit_off      = (int64_t)offsetof(VURegs, ebit);
	const int64_t tpc_off       = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_TPC * (int64_t)sizeof(REG_VI));
	const int64_t regi_off      = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_I   * (int64_t)sizeof(REG_VI));
	const int64_t fmacwpos_off  = (int64_t)offsetof(VURegs, fmacwritepos);
	const int64_t ialuwpos_off  = (int64_t)offsetof(VURegs, ialuwritepos);
	const int64_t vibackup_off  = (int64_t)offsetof(VURegs, VIBackupCycles);

	// Stage C2: prime the pinned cycle register from memory. Every subsequent
	// step 1 in the per-pair loop bumps x21 in place and does NOT store back
	// to memory; the block-end flush (pre-epilogue) writes it out once.
	armAsm->Ldr(VU1_CYCLE_REG, MemOperand(VU1_BASE_REG, cycle_off));

	// Stage C3: prime the pinned FMAC/IALU write-position registers from
	// memory. Every FMAC-pipe pair reads x24 directly for the slot address
	// math (no per-pair memory load) and step 14 bumps w24 in place without
	// touching memory. Same for w25 / IALU. The block-end flush (pre-
	// epilogue) writes both back in a single Str pair.
	emitReloadWposRegs(fmacwpos_off, ialuwpos_off);

	// --- Per-pair code emission ---
	// XGKICK cycle-delay tracking (mirrors microVU mVUinfo.doXGKICK).
	// When a pair captures an XGKICK (vu1_XGKICK stashes the addr in
	// s_vu1_pending_xgkick_addr), the *next* pair fires the deferred
	// transfer AFTER its own opcodes so any store on that pair has
	// committed before GIF walks VU1.Mem. If the next pair is itself an
	// XGKICK, the prior kick is fired *before* that pair's lower emit
	// (see step 8a), so pair k's kick always reaches GIF before pair k+1
	// overwrites the scratch with its own captured addr.
	bool pending_xgkick_fire = false;
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
			// Full interpreter fallback for this pair. vu1Exec runs a complete
			// interpreter pair, including the _vuTest*/_vuAdd* pipeline helpers
			// which read AND write VU->cycle — flush x21 first, reload after.
			// Stage C3: vu1Exec's inner driver loop (_vu1Exec in
			// VU1microInterp.cpp) also advances fmacwritepos AND _vuAddIALUStalls
			// advances ialuwritepos, so x24/x25 must be flushed+reloaded
			// across this BL too.
			emitFlushCycleReg(cycle_off);
			emitFlushWposRegs(fmacwpos_off, ialuwpos_off);
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1Exec));
			emitReloadCycleReg(cycle_off);
			emitReloadWposRegs(fmacwpos_off, ialuwpos_off);
			// Honor pending XGKICK fire from prior pair. Hazard fallbacks
			// never carry an XGKICK themselves (XGKICK has no VF write and
			// doesn't touch CLIP), so fire unconditionally when pending.
			if (pending_xgkick_fire)
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu1_XGKICK_fire_deferred));
				pending_xgkick_fire = false;
			}
			pc = (pc + 8) & (VU1_PROGSIZE - 1);
			continue;
		}

		// 1. VU->cycle++ — Stage C2 uses the cached VU1_CYCLE_REG (x21).
		//    x22 latches "cycle before this pair" for the VIBackupCycles
		//    decrement at step 6b. Both x21 and x22 are callee-saved and
		//    already saved/restored in our prologue/epilogue. No memory
		//    store here; the block-end flush writes x21 to VU->cycle once.
		armAsm->Mov(x22, VU1_CYCLE_REG);
		armAsm->Add(VU1_CYCLE_REG, VU1_CYCLE_REG, 1);

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
		//    instructions are non-FMAC and emit zero work here. Stage A uses
		//    skip_info[i] to elide FMAC stall-check BLs when the compile-time
		//    ring buffer proves no alias exists.
		emitTestUpperStalls(uregs,
			skip_info[i].skipUpperFMACStall0,
			skip_info[i].skipUpperFMACStall1);

		// 5b. Test lower stalls BEFORE TestPipes (non-I-bit only).
		//     TestLowerStalls may advance VU->cycle (FDIV/EFU/ALU stalls);
		//     TestPipes needs to see the updated cycle to flush FMAC correctly.
		//     Stage B adds FDIV/EFU/ALU wait skip flags.
		if (!ibit)
			emitTestLowerStalls(lregs,
				skip_info[i].skipLowerFMACStall0,
				skip_info[i].skipLowerFMACStall1,
				skip_info[i].skipLowerFDIVWait,
				skip_info[i].skipLowerEFUWait,
				skip_info[i].skipLowerALUStall);

		// 6. Test pipes (after lower stalls for non-I-bit). Uses the VU1-
		//    specialized helper that skips the XGKICK block and the do-while
		//    retry loop — see vu1_TestPipes_VU1 definition above. Stage B
		//    elides the BL entirely when the pre-walk proved nothing matures
		//    at this pair AND all pipes' carry-in gates have cleared. Stage
		//    C2 flushes the cached cycle register before the BL so the
		//    helper's flush checks read the up-to-date value; it does not
		//    write cycle so no reload is needed.
		if (!skip_info[i].skipTestPipes)
		{
			emitFlushCycleReg(cycle_off);
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1_TestPipes_VU1));
		}

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
			// 8a. Back-to-back XGKICK sequencing. If the prior pair captured
			//     an XGKICK and this pair's lower is also an XGKICK, fire the
			//     prior kick NOW — before vu1_XGKICK clobbers the scratch
			//     with the new addr. Pair k+1's upper has already emitted
			//     above (step 7) and upper ops don't write VU1.Mem, so firing
			//     here doesn't race with any pending store.
			if (pending_xgkick_fire && isXgkickOp(lower))
			{
				armAsm->Mov(x0, VU1_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu1_XGKICK_fire_deferred));
				pending_xgkick_fire = false;
			}
			// Execute lower instruction (stalls already tested above).
			armAsm->Mov(w4, lower);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, code_off));
			VU1.code = lower; // compile-time context
			g_vu1CurrentPC = pc; // compile-time PC for native branches
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

		// 13. Ebit countdown (inline). vu1EbitDone calls _vuFlushAll which
		//     writes VU->cycle (pipeline drain can advance the cycle to
		//     retire still-pending slots), so flush/reload the cached
		//     cycle register around the BL.
		{
			Label skip_ebit;
			armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, ebit_off));
			armAsm->Cbz(w4, &skip_ebit);          // ebit == 0: nothing to do
			armAsm->Subs(w4, w4, 1);
			armAsm->Str(w4, MemOperand(VU1_BASE_REG, ebit_off));
			armAsm->B(&skip_ebit, ne);             // still > 0: keep counting
			// ebit just reached 0: end of microprogram
			emitFlushCycleReg(cycle_off);
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1EbitDone));
			emitReloadCycleReg(cycle_off);
			armAsm->Bind(&skip_ebit);
		}

		// 14. FMAC write-position advance (wraps mod 4). Stage C3: hoisted
		// into w24 — no memory load/store here; the block-end flush writes
		// the final value back in one store.
		if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
		{
			armAsm->Add(VU1_FMAC_WPOS_REG, VU1_FMAC_WPOS_REG, 1);
			armAsm->And(VU1_FMAC_WPOS_REG, VU1_FMAC_WPOS_REG, 3);
		}

		// 15. XGKICK deferred fire. A pending kick from the prior pair is
		//     emitted here — AFTER this pair's opcodes so any store has
		//     committed before GIF walks VU1.Mem. Back-to-back XGKICK was
		//     already handled at step 8a, so if we reach here with pending
		//     set, this pair's lower is guaranteed to be non-XGKICK.
		if (pending_xgkick_fire)
		{
			armAsm->Mov(x0, VU1_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu1_XGKICK_fire_deferred));
			pending_xgkick_fire = false;
		}
		// Re-arm for the next pair if this one captured an XGKICK.
		if (!ibit && isXgkickOp(lower))
			pending_xgkick_fire = true;

		pc = (pc + 8) & (VU1_PROGSIZE - 1);
	}

	// Block-end XGKICK drain. If the last pair was XGKICK we never got a
	// chance to emit the deferred fire inside the loop — drain here so the
	// scratch (s_vu1_pending_xgkick_addr) never carries state into the next
	// compiled block. The file-local static assumption in iVU1Lower_arm64.cpp
	// depends on this drain firing on every exit path.
	if (pending_xgkick_fire)
	{
		armAsm->Mov(x0, VU1_BASE_REG);
		armEmitCall(reinterpret_cast<const void*>(vu1_XGKICK_fire_deferred));
	}

	// Stage C2: flush the cached cycle register to memory before restoring
	// the caller's x21. From here on VU->cycle is authoritative again.
	emitFlushCycleReg(cycle_off);
	// Stage C3: flush the cached FMAC/IALU write-position registers to
	// memory before restoring the caller's x24/x25.
	emitFlushWposRegs(fmacwpos_off, ialuwpos_off);

	// --- Epilogue (64-byte frame; mirrors the prologue layout above) ---
	armAsm->Ldr(x25, MemOperand(sp, 48));
	armAsm->Ldp(VU1_BASE_REG, x24, MemOperand(sp, 32));
	armAsm->Ldp(VU1_CYCLE_REG, x22, MemOperand(sp, 16));
	armAsm->Ldp(x29, x30, MemOperand(sp, 64, PostIndex));
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
	s_vu1_program_ended = false;

	while ((VU1.cycle - startcycles) < cycles)
	{
		// Termination gate.
		//   Non-MTVU: read VPU_STAT 0x100 live — external clears (FBRST reset
		//     on the EE thread) stop us mid-execute, and vu1EbitDone's own
		//     clear ends the program normally.
		//   MTVU: VPU_STAT 0x100 is cleared by vu1ExecMicro before the queue
		//     and never re-set under INSTANT_VU1, so we can't use it. Break
		//     on s_vu1_program_ended (set by vu1EbitDone on the VU thread).
		//     Matches x86 microVU.cpp:381-385 which skips the VPU_STAT gate
		//     entirely under THREAD_VU1.
		const bool stopped = THREAD_VU1
			? s_vu1_program_ended
			: !(VU0.VI[REG_VPU_STAT].UL & 0x100);
		if (stopped)
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
