// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Main file
// Provides block compilation, dispatchers, and execution loop.

#include "Common.h"
#include "CDVD/CDVD.h"
#include "Elfheader.h"
#include "GS.h"
#include "Host.h"
#include "Memory.h"
#include "Patch.h"
#include "R3000A.h"
#include "R5900OpcodeTables.h"
#include "VMManager.h"
#include "vtlb.h"

#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"
#include "x86/BaseblockEx.h"

// EEINST: per-instruction info used by the recompiler for register liveness analysis.
// This mirrors the definition in x86/iCore.h but avoids dragging in x86 register allocator deps.
struct EEINST
{
	u16 info;
	u8 regs[34];
	u8 fpuregs[33];
	u8 vfregs[34];
	u8 viregs[16];
	u8 writeType[3], writeReg[3];
	u8 readType[4], readReg[4];
};
extern EEINST* g_pCurInstInfo;

#include "common/AlignedMalloc.h"
#include "common/Console.h"
#include "common/FastJmp.h"
#include "common/HeapArray.h"
#include "common/Perf.h"

using namespace R5900;

// ============================================================================
//  State — mirrors the x86 recompiler's state
// ============================================================================

static bool eeRecNeedsReset = false;
static bool eeCpuExecuting = false;
static bool eeRecExitRequested = false;
static bool g_resetEeScalingStats = false;

#define PC_GETBLOCK(x) PC_GETBLOCK_(x, recLUT)

u32 maxrecmem = 0;
alignas(16) static uptr recLUT[_64kb];
alignas(16) static u32 hwLUT[_64kb];

static __fi u32 HWADDR(u32 mem) { return hwLUT[mem >> 16] + mem; }

u32 s_nBlockCycles = 0;
bool s_nBlockInterlocked = false;
u32 pc;
int g_branch;

alignas(16) GPR_reg64 g_cpuConstRegs[32] = {};
u32 g_cpuHasConstReg = 0, g_cpuFlushedConstReg = 0;
bool g_cpuFlushedPC, g_cpuFlushedCode, g_recompilingDelaySlot, g_maySignalException;

eeProfiler EE::Profiler;

// Defined in x86/iCore.cpp on x86 — we need our own on ARM64
EEINST* g_pCurInstInfo = nullptr;

// ============================================================================
//  Code buffer and block management
// ============================================================================

static DynamicHeapArray<u8, 4096> recRAMCopy;
static DynamicHeapArray<BASEBLOCK, 4096> recLutReserve_RAM;
static DynamicHeapArray<BASEBLOCK, 4096> recLutUnmapped;
static size_t recLutEntries;
static bool extraRam;

static BASEBLOCK* recRAM = nullptr;
static BASEBLOCK* recROM = nullptr;
static BASEBLOCK* recROM1 = nullptr;
static BASEBLOCK* recROM2 = nullptr;

static BaseBlocks recBlocks;
static u8* recPtr = nullptr;
static u8* recPtrEnd = nullptr;
EEINST* s_pInstCache = nullptr;
static u32 s_nInstCacheSize = 0;

static BASEBLOCK* s_pCurBlock = nullptr;
static BASEBLOCKEX* s_pCurBlockEx = nullptr;
u32 s_nEndBlock = 0;
u32 s_branchTo;
static bool s_branchIsUnconditional; // true for J/JAL, false for BEQ/BNE etc.
static bool s_nBlockFF;

static int* s_pCode = nullptr;

GPR_reg64 s_saveConstRegs[32];
static u32 s_saveHasConstReg = 0, s_saveFlushedConstReg = 0;
static EEINST* s_psaveInstInfo = nullptr;

static u32 s_savenBlockCycles = 0;

// Constant pool for the recompiler code buffer
static ArmConstantPool s_recConstPool;

// ============================================================================
//  Dispatcher pointers (filled by _DynGen_Dispatchers)
// ============================================================================

static const void* DispatcherEvent = nullptr;
static const void* DispatcherReg = nullptr;
static const void* JITCompile = nullptr;
static const void* EnterRecompiledCode = nullptr;
static const void* DispatchBlockDiscard = nullptr;
static const void* DispatchPageReset = nullptr;
static const void* UnmappedRecLUTPage = nullptr;

// ============================================================================
//  Forward declarations
// ============================================================================

static void recRecompile(const u32 startpc);
static void recResetRaw();
static void recError(u32 error);
static void iBranchTest(u32 newpc = 0xffffffff);
static void ClearRecLUT(BASEBLOCK* base, int count);
static u32 scaleblockcycles();
static void recExitExecution();
static void dyna_block_discard(u32 start, u32 sz);
static void dyna_page_reset(u32 start, u32 sz);
static void recClear(u32 addr, u32 size);
static void memory_protect_recompiled_code(u32 startpc, u32 size);

// ============================================================================
//  ARM64 codegen helpers
// ============================================================================

void armFlushConstRegs()
{
	for (int i = 1; i < 32; i++)
	{
		if ((g_cpuHasConstReg & (1u << i)) && !(g_cpuFlushedConstReg & (1u << i)))
		{
			// Write constant value to cpuRegs.GPR[i].SD[0] (lower 64 bits only).
			// Upper 64 bits (UD[1]) are left untouched — matches interpreter behavior.
			s64 val = g_cpuConstRegs[i].SD[0];
			if (val == 0)
			{
				armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(i)));
			}
			else
			{
				armAsm->Mov(RSCRATCHGPR, static_cast<u64>(val));
				armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(i)));
			}
			g_cpuFlushedConstReg |= (1u << i);
		}
	}
}

// Flush a single constant register to memory if it has a const value that
// hasn't been written back yet. No-op otherwise. Used by call sites that
// bypass armLoadGPR* and read GPR memory directly (128-bit MMI/QMTC2 loads,
// PLZCW upper-half access) so the const value is coherent before the read.
void armFlushConstReg(int reg)
{
	if (reg <= 0 || reg >= 32)
		return;
	const u32 bit = 1u << reg;
	if (!(g_cpuHasConstReg & bit) || (g_cpuFlushedConstReg & bit))
		return;

	const s64 val = g_cpuConstRegs[reg].SD[0];
	if (val == 0)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(reg)));
	}
	else
	{
		armAsm->Mov(RSCRATCHGPR, static_cast<u64>(val));
		armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(reg)));
	}
	g_cpuFlushedConstReg |= bit;
}

// Drop a register's const tracking AFTER first committing its value to memory.
// The early `GPR_DEL_CONST(_Rd_)` pattern at the top of an op is unsafe when
// _Rd_ aliases _Rs_/_Rt_: clearing the const flag causes the subsequent
// armLoadGPR* on the operand to fall through to an LDR that reads stale
// memory (the const value never reached cpuRegs.GPR). Flushing first ensures
// memory is coherent regardless of whether _Rd_ aliases an operand. The
// follow-up armStoreGPR* still overwrites _Rd_; the wasted store is
// harmless. Use this anywhere you would have written GPR_DEL_CONST(_Rd_)
// before loading source operands.
void armDelConstReg(int reg)
{
	armFlushConstReg(reg);
	GPR_DEL_CONST(reg);
}

// ============================================================================
//  Tier 1 GPR cache — Phase C plumbing (NOT wired to ops yet)
// ============================================================================
// See arm64Emitter.h for the API contract and rationale. This is pure
// infrastructure: no instruction emitter currently calls armGprAlloc/Flush/
// Invalidate. Block entry (recRecompile main loop) calls armGprCacheReset
// so the slot table starts in a defined state once Phase D begins wiring
// individual ops to the cache.

ArmGprCacheSlot g_armGprCache[32];
u16 g_armGprCachePoolUsed;

// Pool of caller-saved host regs available for cache slots. Order is the
// allocation order (front-first). Indices into this array are what get
// stored in g_armGprCachePoolUsed.
//   x4..x6  : RSCRATCHGPR{,2,3}            — used by per-instruction codegen
//   x9..x10 : armEmit{Set,Flush,Reload}Cycle scratch
//   x16..x17: vixl IP0/IP1
//   x19..x28: pinned state regs (RCPUSTATE, RCYCLE, RMEMBASE, RRECLUT,
//             RFASTMEMBASE, RDELAYSLOTGPR + reserved Phase D-H slots)
// Everything else in 0..30 is either an arg (x0..x3), the FP/LR (x29/x30),
// or already excluded above. That leaves x7, x8, x11..x15.
static constexpr u8 kArmGprCachePool[] = {7, 8, 11, 12, 13, 14, 15};
static constexpr int kArmGprCachePoolSize = static_cast<int>(sizeof(kArmGprCachePool) / sizeof(kArmGprCachePool[0]));
static_assert(kArmGprCachePoolSize <= 16, "g_armGprCachePoolUsed bitmask is u16");

// Linear scan: pool host_code → index in kArmGprCachePool. -1 if not in pool.
static int armGprPoolIndex(u8 host_code)
{
	for (int i = 0; i < kArmGprCachePoolSize; i++)
	{
		if (kArmGprCachePool[i] == host_code)
			return i;
	}
	return -1;
}

// Acquire a free pool slot (or evict the lowest-indexed cached MIPS GPR).
// Returns the host register code. Marks the index used in g_armGprCachePoolUsed.
static u8 armGprAcquirePoolSlot()
{
	for (int i = 0; i < kArmGprCachePoolSize; i++)
	{
		const u16 bit = static_cast<u16>(1u << i);
		if (!(g_armGprCachePoolUsed & bit))
		{
			g_armGprCachePoolUsed |= bit;
			return kArmGprCachePool[i];
		}
	}

	// Pool full — evict the lowest-indexed cached MIPS GPR. Phase C uses
	// the simplest possible policy; Phase D will refine (LRU / use-count).
	for (int g = 1; g < 32; g++)
	{
		ArmGprCacheSlot& slot = g_armGprCache[g];
		if (slot.host_code == 0xff)
			continue;
		const int slot_idx = armGprPoolIndex(slot.host_code);
		if (slot_idx < 0)
			continue;

		if (slot.dirty)
		{
			armAsm->Str(a64::XRegister(slot.host_code),
				a64::MemOperand(RCPUSTATE, GPR_OFFSET(g)));
		}
		const u8 host = slot.host_code;
		slot.host_code = 0xff;
		slot.dirty = false;
		slot.sxw = false;
		// The pool index bit stays set — we hand it straight to the new owner.
		return host;
	}

	pxFailRel("armGprAcquirePoolSlot: no slots free and nothing to evict");
	return 0xff;
}

void armGprCacheReset()
{
	for (int i = 0; i < 32; i++)
	{
		g_armGprCache[i].host_code = 0xff;
		g_armGprCache[i].dirty = false;
		g_armGprCache[i].sxw = false;
	}
	g_armGprCachePoolUsed = 0;
}

bool armGprIsCached(int gpr)
{
	return gpr >= 0 && gpr < 32 && g_armGprCache[gpr].host_code != 0xff;
}

a64::XRegister armGprAlloc(int gpr, bool for_write)
{
	pxAssertMsg(gpr > 0 && gpr < 32, "armGprAlloc: GPR0 has no cache slot (use xzr)");

	ArmGprCacheSlot& slot = g_armGprCache[gpr];

	if (slot.host_code != 0xff)
	{
		// Already cached — just hand back the existing host reg.
		if (for_write)
		{
			slot.dirty = true;
			slot.sxw = false;
		}
		return a64::XRegister(slot.host_code);
	}

	// Not cached — pick a pool slot and (for reads) populate it.
	const u8 host = armGprAcquirePoolSlot();
	a64::XRegister reg(host);

	if (!for_write)
	{
		if (GPR_IS_CONST1(gpr))
		{
			// Const-tracked: emit MOV-imm. The const tracker is left alone;
			// other read sites in this block can still see the const.
			armAsm->Mov(reg, static_cast<u64>(g_cpuConstRegs[gpr].SD[0]));
		}
		else
		{
			armAsm->Ldr(reg, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
		}
	}
	// for_write: caller will store next; no load needed.

	slot.host_code = host;
	slot.dirty = for_write;
	slot.sxw = false;
	return reg;
}

a64::XRegister armGprAllocTmp()
{
	const u8 host = armGprAcquirePoolSlot();
	return a64::XRegister(host);
}

void armGprReleaseTmp(const a64::Register& reg)
{
	const int idx = armGprPoolIndex(static_cast<u8>(reg.GetCode()));
	pxAssertMsg(idx >= 0, "armGprReleaseTmp: register is not in the cache pool");
	g_armGprCachePoolUsed &= ~static_cast<u16>(1u << idx);
}

void armGprFlush(int gpr)
{
	if (gpr <= 0 || gpr >= 32)
		return;
	ArmGprCacheSlot& slot = g_armGprCache[gpr];
	if (slot.host_code == 0xff || !slot.dirty)
		return;
	armAsm->Str(a64::XRegister(slot.host_code),
		a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	slot.dirty = false;
}

void armGprInvalidate(int gpr)
{
	if (gpr <= 0 || gpr >= 32)
		return;
	ArmGprCacheSlot& slot = g_armGprCache[gpr];
	if (slot.host_code == 0xff)
		return;
	if (slot.dirty)
	{
		armAsm->Str(a64::XRegister(slot.host_code),
			a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	}
	const int idx = armGprPoolIndex(slot.host_code);
	if (idx >= 0)
		g_armGprCachePoolUsed &= ~static_cast<u16>(1u << idx);
	slot.host_code = 0xff;
	slot.dirty = false;
	slot.sxw = false;
}

void armGprFlushAll()
{
	for (int g = 1; g < 32; g++)
	{
		ArmGprCacheSlot& slot = g_armGprCache[g];
		if (slot.host_code != 0xff && slot.dirty)
			armGprFlush(g);
	}
}

void armGprInvalidateAll()
{
	for (int g = 1; g < 32; g++)
	{
		if (g_armGprCache[g].host_code != 0xff)
			armGprInvalidate(g);
	}
	// Pool should be empty now; reset explicitly in case any tmps leaked.
	// (Phase C: tmp leaks are a bug — Phase D ops MUST release every tmp.)
	g_armGprCachePoolUsed = 0;
}

// ============================================================================
//  Cycle delta helpers (Phase B)
// ============================================================================
// RCYCLE (x20) holds (s64)(cpuRegs.cycle - cpuRegs.nextEventCycle).
// Negative = budget remaining; >= 0 = event due.
// Per-block accounting becomes a single ADDS + B.MI continue, replacing the
// 7-instruction LDR/ADD/STR/LDR/CMP/B.LO/B sequence used previously.
//
// The in-memory cpuRegs.cycle is stale during JIT execution (the in-flight
// delta lives in x20). It is reconciled at every "exit to C++" point:
//   - DispatcherEvent (before recEventTest)
//   - armBranchCallInterpreter (before the BL into the interp branch helper)
// armCallInterpreter does NOT writeback — per-instruction interp helpers do
// not read cpuRegs.cycle, matching the prior s_nBlockCycles model.

// Emit: cpuRegs.cycle = cpuRegs.nextEventCycle + RCYCLE
//       (i.e. flush the in-flight delta back to memory)
static void armWritebackCycle()
{
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x0, a64::x1, RCYCLE);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
}

// Emit: RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
//       (i.e. reload the delta from memory after C++ may have changed either)
static void armReloadCycle()
{
	armAsm->Ldr(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Sub(RCYCLE, a64::x0, a64::x1);
}

// Emit: dst = cpuRegs.nextEventCycle + RCYCLE
// JIT-current cycle for ops that need to read "cycle now" (e.g. COP0 Count).
// Reading CYCLE_OFFSET directly would be stale by potentially many blocks
// since RCYCLE only writes back at iBranchTest / DispatcherEvent.
void armEmitLoadCurrentCycle(const a64::Register& dst)
{
	armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(dst, dst, RCYCLE);
}

// Inline cpuSetNextEventDelta(delta) — see header for math derivation.
//   if (RCYCLE + delta) < 0:
//     nextEventCycle += (RCYCLE + delta)
//     RCYCLE = -delta
// Clobbers x9, x10.
void armEmitSetNextEventDelta(s32 delta)
{
	a64::Label skip;
	// x9 = RCYCLE + delta, sets flags. ADDS supports a 12-bit unsigned imm
	// (optionally LSL #12); the macro assembler spills to a temp if needed.
	armAsm->Adds(a64::x9, RCYCLE, delta);
	// If result >= 0 (signed), the existing schedule is already at or before
	// (cycle + delta), so leave it.
	armAsm->B(&skip, a64::ge);
	// nextEventCycle += x9 (pulls the event forward by the missing budget)
	armAsm->Ldr(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x10, a64::x10, a64::x9);
	armAsm->Str(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	// RCYCLE = -delta
	armAsm->Mov(RCYCLE, -static_cast<s64>(delta));
	armAsm->Bind(&skip);
}

// Flush RCYCLE → cpuRegs.cycle BEFORE a C++ call that reads cycle or
// reschedules events. Required because Phase B keeps the JIT-current cycle
// only in RCYCLE; cpuRegs.cycle in memory is stale by the block's accumulated
// cycles. Without this flush, callees like CPU_INT compute new nec from a
// stale "now" and schedule events too late.
//   cpuRegs.cycle = cpuRegs.nextEventCycle + RCYCLE
// Uses x9 as scratch (caller-saved, free pre-call).
void armEmitFlushCycleBeforeCall()
{
	armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x9, a64::x9, RCYCLE);
	armAsm->Str(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
}

// Reload RCYCLE from memory AFTER a C++ call that may have mutated nec
// (and that was preceded by armEmitFlushCycleBeforeCall, so cpuRegs.cycle
// is current). Reconstructs the Phase B invariant:
//   RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
// Uses x9/x10 as scratch — does NOT clobber x0/w0 (vtlb_memRead's return).
void armEmitReloadCycleAfterCall()
{
	armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	armAsm->Ldr(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Sub(RCYCLE, a64::x9, a64::x10);
}

void armLoadGPR64(const a64::Register& dst, int gpr)
{
	if (gpr == 0)
		armAsm->Mov(dst, a64::xzr);
	else if (GPR_IS_CONST1(gpr))
		armAsm->Mov(dst, static_cast<u64>(g_cpuConstRegs[gpr].SD[0]));
	else
		armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

void armLoadGPR32(const a64::Register& dst, int gpr)
{
	if (gpr == 0)
		armAsm->Mov(dst.IsX() ? dst : a64::Register(dst.GetCode(), 64), a64::xzr);
	else if (GPR_IS_CONST1(gpr))
		armAsm->Mov(dst, g_cpuConstRegs[gpr].UL[0]);
	else
		armAsm->Ldr(a64::WRegister(dst.GetCode()), a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

void armStoreGPR64SignExt32(const a64::Register& src_w, int gpr)
{
	if (gpr == 0)
		return;

	// Sign-extend 32-bit value to 64-bit, store to SD[0] only.
	// Upper 64 bits (UD[1]) left untouched — matches interpreter behavior.
	a64::XRegister src_x(src_w.GetCode());
	armAsm->Sxtw(src_x, a64::WRegister(src_w.GetCode()));
	armAsm->Str(src_x, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	GPR_DEL_CONST(gpr);
}

void armStoreGPR64(const a64::Register& src_x, int gpr)
{
	if (gpr == 0)
		return;
	// Store to SD[0] only. Upper 64 bits (UD[1]) left untouched.
	armAsm->Str(src_x, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	GPR_DEL_CONST(gpr);
}

void armFlushPC()
{
	if (!g_cpuFlushedPC)
	{
		armAsm->Mov(RWSCRATCH, pc);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		g_cpuFlushedPC = true;
	}
}

void armFlushCode()
{
	if (!g_cpuFlushedCode)
	{
		armAsm->Mov(RWSCRATCH, cpuRegs.code);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CODE_OFFSET));
		g_cpuFlushedCode = true;
	}
}

void armCallInterpreter(void (*func)())
{
    armFlushPC();
    armFlushCode();
    armFlushConstRegs();

    // Phase B: the JIT-current cycle lives in RCYCLE (x20); cpuRegs.cycle in
    // memory is stale. Many interpreter stubs (COP2/VU0 CFC2/CTC2/QMFC2, MMI,
    // etc.) transitively call CPU_INT / cpuSetNextEventDelta, which read
    // cpuRegs.cycle to schedule events. Without flushing first, events are
    // scheduled relative to an old "now" and nextEventCycle goes wrong;
    // without reloading after, the RCYCLE invariant breaks and the next
    // iBranchTest miscomputes budget — either way, a hang.
    armEmitFlushCycleBeforeCall();
    armEmitCall((const void*)func);
    armEmitReloadCycleAfterCall();

    // RMEMBASE (x21) is callee-saved — no reload needed.
    g_cpuHasConstReg = 1;
    g_cpuFlushedConstReg = 1;
}

// Branch-call variant: used for interpreter branch/syscall/trap stubs.
// Matches the upstream x86 recBranchCall:
//   1) Writes the in-flight cycle delta back and forces an event check
//      (cpuRegs.cycle = nextEventCycle + RCYCLE; nextEventCycle = cycle)
//   2) Flushes state and calls the standard interpreter function
//      (which calls doBranch → intUpdateCPUCycles → intEventTest,
//       properly counting delay slot cycles and processing events)
//   3) Reloads RCYCLE from memory (interp may have advanced cycle and/or
//      scheduled a new nextEventCycle)
//   4) Sets g_branch = 2
void armBranchCallInterpreter(void (*func)())
{
	// cpuRegs.cycle = nextEventCycle + RCYCLE  (writeback the in-flight delta)
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x0, a64::x1, RCYCLE);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	// nextEventCycle = cycle  (force the post-call iBranchTest into DispatcherEvent)
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));

	// Flush PC, code, and const regs — but NOT cycle (already flushed above).
	// We cannot delegate to armCallInterpreter here because it calls
	// armEmitFlushCycleBeforeCall, which would re-read the nec we just set
	// to cycle and add RCYCLE again, double-counting the accumulated delta.
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)func);
	armEmitReloadCycleAfterCall();

	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
}

// ============================================================================
//  Cycle scaling (same algorithm as x86)
// ============================================================================

#define DEFAULT_SCALED_BLOCKS() (s_nBlockCycles >> 3)

static u32 scaleblockcycles_calculation()
{
	const bool lowcycles = (s_nBlockCycles <= 40);
	const s8 cyclerate = EmuConfig.Speedhacks.EECycleRate;
	u32 scale_cycles = 0;

	if (cyclerate == 0 || lowcycles || cyclerate < -99 || cyclerate > 3)
		scale_cycles = DEFAULT_SCALED_BLOCKS();
	else if (cyclerate > 1)
		scale_cycles = s_nBlockCycles >> (2 + cyclerate);
	else if (cyclerate == 1)
		scale_cycles = DEFAULT_SCALED_BLOCKS() / 1.3f;
	else if (cyclerate == -1)
		scale_cycles = (s_nBlockCycles <= 80 || s_nBlockCycles > 168 ? 5 : 7) * s_nBlockCycles / 32;
	else
		scale_cycles = ((5 + (-2 * (cyclerate + 1))) * s_nBlockCycles) >> 5;

	return (scale_cycles < 1) ? 1 : scale_cycles;
}

static u32 scaleblockcycles()
{
	return scaleblockcycles_calculation();
}

u32 scaleblockcycles_clear()
{
	u32 scaled = scaleblockcycles_calculation();
	const s8 cyclerate = EmuConfig.Speedhacks.EECycleRate;
	const bool lowcycles = (s_nBlockCycles <= 40);

	if (!lowcycles && cyclerate > 1)
		s_nBlockCycles &= (0x1 << (cyclerate + 2)) - 1;
	else
		s_nBlockCycles &= 0x7;

	return scaled;
}

// ============================================================================
//  ARM64 Dispatchers
// ============================================================================

static void recEventTest()
{
	_cpuEventTest_Shared();

	if (eeRecExitRequested)
	{
		eeRecExitRequested = false;
		recExitExecution();
	}
}

// Dispatcher: jump to block at cpuRegs.pc
static const void* _DynGen_DispatcherReg()
{
	u8* retval = armStartBlock();

	// w0 = cpuRegs.pc
	armAsm->Ldr(a64::w0, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	// x1 = recLUT[pc >> 16] (has negative offset baked in by recLUT_SetPage)
	armAsm->Lsr(a64::w1, a64::w0, 16);
	armAsm->Ldr(a64::x1, a64::MemOperand(RRECLUT, a64::x1, a64::LSL, 3));
	// PC_GETBLOCK_ = recLUT[pc>>16] + pc * (sizeof(BASEBLOCK) / 4)
	// sizeof(BASEBLOCK) = 8, so scale = 2. Must use full pc, not masked.
	armAsm->Lsl(a64::x2, a64::x0, 1); // x2 = (u64)pc * 2 (w0 was zero-extended by ldr)
	armAsm->Ldr(a64::x3, a64::MemOperand(a64::x1, a64::x2));
	armAsm->Br(a64::x3);

	armEndBlock();
	return retval;
}

// Event dispatcher: writeback the cycle delta, call recEventTest, reload, then
// jump to DispatcherReg.
// On x86, DispatcherEvent falls through into DispatcherReg (contiguous code).
// On ARM64 each armStartBlock/armEndBlock pair introduces alignment padding,
// so we must use an explicit jump instead of relying on fallthrough.
static const void* _DynGen_DispatcherEvent()
{
	pxAssert(DispatcherReg); // must be generated first
	u8* retval = armStartBlock();

	// recEventTest reads cpuRegs.cycle and may schedule a new nextEventCycle.
	// Flush the in-flight delta before the call, reload after.
	armWritebackCycle();
	armEmitCall((const void*)recEventTest);
	armReloadCycle();
	armEmitJmp(DispatcherReg);

	armEndBlock();
	return retval;
}

// JIT compile: called when we hit an uncompiled block
static const void* _DynGen_JITCompile()
{
	u8* retval = armStartBlock();

	// arg1 = cpuRegs.pc
	armAsm->Ldr(RWARG1, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armEmitCall((const void*)recRecompile);

	// Now dispatch to the newly compiled block
	armAsm->Ldr(a64::w0, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armAsm->Lsr(a64::w1, a64::w0, 16);
	armAsm->Ldr(a64::x1, a64::MemOperand(RRECLUT, a64::x1, a64::LSL, 3));
	armAsm->Lsl(a64::x2, a64::x0, 1); // x2 = (u64)pc * 2
	armAsm->Ldr(a64::x3, a64::MemOperand(a64::x1, a64::x2));
	armAsm->Br(a64::x3);

	armEndBlock();
	return retval;
}

// Enter recompiled code: called from C++ to start execution
static const void* _DynGen_EnterRecompiledCode()
{
	u8* retval = armStartBlock();

	// Save callee-saved registers and set up stack frame
	armBeginStackFrame(false);

	// Load pinned state registers
	// (fpuRegs is reached via RCPUSTATE + FPUREGS_BASE — no dedicated reg.)
	armMoveAddressToReg(RCPUSTATE, &cpuRegs);
	armMoveAddressToReg(RRECLUT, recLUT);

	// Load RAM base if available
	if (eeMem)
		armMoveAddressToReg(RMEMBASE, eeMem->Main);

	// Load fastmem base for VTLB direct access
	if (vtlb_private::vtlbdata.fastmem_base)
		armMoveAddressToReg(RFASTMEMBASE, (void*)vtlb_private::vtlbdata.fastmem_base);

	// Initialize RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
	armReloadCycle();

	// Jump to the dispatcher
	armEmitJmp(DispatcherReg);

	armEndBlock();
	return retval;
}

static const void* _DynGen_DispatchBlockDiscard()
{
	u8* retval = armStartBlock();
	armEmitCall((const void*)dyna_block_discard);
	armEmitJmp(DispatcherReg);
	armEndBlock();
	return retval;
}

static const void* _DynGen_DispatchPageReset()
{
	u8* retval = armStartBlock();
	armEmitCall((const void*)dyna_page_reset);
	armEmitJmp(DispatcherReg);
	armEndBlock();
	return retval;
}

static const void* _DynGen_UnmappedRecLUTPage()
{
	u8* retval = armStartBlock();
	// Pass the actual cpuRegs.pc so recError can report it
	armAsm->Ldr(RWARG1, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armEmitCall((const void*)recError);
	armEndBlock();
	return retval;
}

static void _DynGen_Dispatchers()
{
	// DispatcherReg must be generated before DispatcherEvent (Event jumps to Reg).
	// On x86 these are contiguous (fallthrough), but on ARM64 each block has
	// alignment padding, so DispatcherEvent uses an explicit jump instead.
	DispatcherReg = _DynGen_DispatcherReg();
	DispatcherEvent = _DynGen_DispatcherEvent();
	JITCompile = _DynGen_JITCompile();
	EnterRecompiledCode = _DynGen_EnterRecompiledCode();
	DispatchBlockDiscard = _DynGen_DispatchBlockDiscard();
	DispatchPageReset = _DynGen_DispatchPageReset();
	UnmappedRecLUTPage = _DynGen_UnmappedRecLUTPage();

	recBlocks.SetJITCompile(JITCompile);
}

// ============================================================================
//  Block end — cycle counting and dispatch
// ============================================================================

static void iBranchTest(u32 newpc)
{
	// RCYCLE += scaleblockcycles();
	// if (RCYCLE < 0)  goto DispatcherReg (or linked block);  // still have budget
	// else             goto DispatcherEvent;                  // event due
	//
	// RCYCLE = (cycle - nextEventCycle), so result-negative (N=1) means
	// cycle is still behind nextEventCycle, i.e. budget remaining. B.MI catches
	// that. The fall-through path covers both result==0 and result>0.

	u32 cycles = scaleblockcycles();

	if (EmuConfig.Speedhacks.WaitLoop && s_nBlockFF && newpc == s_branchTo)
	{
		// Wait loop optimization: cycle = max(cycle + n, nextEventCycle)
		// In delta form: RCYCLE = max(RCYCLE + n, 0), then jump to DispatcherEvent.
		armAsm->Adds(RCYCLE, RCYCLE, cycles);
		armAsm->Csel(RCYCLE, a64::xzr, RCYCLE, a64::mi);
		armEmitJmp(DispatcherEvent);
	}
	else
	{
		// Normal path: bump the delta, branch on sign.
		armAsm->Adds(RCYCLE, RCYCLE, cycles);
		armEmitCondBranch(a64::mi, DispatcherReg); // N=1: still have budget
		armEmitJmp(DispatcherEvent);               // fall through: event due
	}
}

// ============================================================================
//  Instruction recompilation
// ============================================================================

void recompileNextInstruction(bool delayslot, bool swapped_delay_slot)
{
	if (EmuConfig.EnablePatches)
		Patch::ApplyDynamicPatches(pc);

	s_pCode = (int*)PSM(pc);
	pxAssert(s_pCode);

	const int old_code = cpuRegs.code;
	EEINST* old_inst_info = g_pCurInstInfo;

	cpuRegs.code = *(int*)s_pCode;

	if (!delayslot)
	{
		pc += 4;
		g_cpuFlushedPC = false;
		g_cpuFlushedCode = false;
	}
	else
	{
		g_recompilingDelaySlot = true;
	}

	g_pCurInstInfo++;

	const OPCODE& opcode = GetCurrentInstruction();

	// NOP check
	if (cpuRegs.code == 0x00000000)
	{
		s_nBlockCycles += 9 * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));
	}
	else
	{
		s_nBlockCycles += opcode.cycles * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));

		if (opcode.recompile)
			opcode.recompile();
		else
		{
			// No recompiler implementation — fall back to interpreter
			armCallInterpreter(opcode.interpret);
		}

		// Phase D: op-local GPR cache discipline.
		// Migrated ops (ALU only as of Phase D) use armGprAlloc to bind MIPS
		// GPRs to host pool regs within their codegen. Commit any dirty slots
		// to memory and drop the cache before moving on to the next opcode —
		// non-migrated ops still read/write memory directly via armLoadGPR /
		// armStoreGPR, so the cache must not survive across the boundary.
		// No-op when nothing was cached (the common case).
		armGprInvalidateAll();
	}

	if (delayslot)
	{
		pc += 4;
		g_cpuFlushedPC = false;
		g_cpuFlushedCode = false;
		g_recompilingDelaySlot = false;
	}

	cpuRegs.code = old_code;
	g_pCurInstInfo = old_inst_info;
}

// ============================================================================
//  Block compilation
// ============================================================================

static void recError(u32 error)
{
	static u32 lastError = ~0u;
	static int errorCount = 0;

	if (error == lastError)
	{
		if (++errorCount > 5)
		{
			Console.Error("EE ARM64 recError: pc=%08X repeated %d times, halting", error, errorCount);
			VMManager::SetPaused(true);
			cpuRegs.branch = 0;
			recExitExecution();
			return;
		}
	}
	else
	{
		lastError = error;
		errorCount = 1;
	}

	Console.Error("EE ARM64 Recompiler Error: pc=%08X (recLUT page %04X unmapped)", error, error >> 16);
	cpuRegs.branch = 0;
	recExitExecution();
}

u8* recBeginThunk()
{
	if (recPtr >= recPtrEnd)
		eeRecNeedsReset = true;

	armSetAsmPtr(recPtr, recPtrEnd - recPtr, nullptr);
	recPtr = armStartBlock();
	return recPtr;
}

u8* recEndThunk()
{
	u8* block_end = armEndBlock();
	pxAssert(block_end < SysMemory::GetEERecEnd());
	recPtr = block_end;
	return block_end;
}

static void recRecompile(const u32 startpc)
{
	pxAssert(startpc);

	// Check if we need to reset the code buffer
	if (recPtr >= recPtrEnd)
		eeRecNeedsReset = true;

	if (HWADDR(startpc) == VMManager::Internal::GetCurrentELFEntryPoint())
		VMManager::Internal::EntryPointCompilingOnCPUThread();

	if (eeRecNeedsReset)
	{
		eeRecNeedsReset = false;
		recResetRaw();
	}

	// Set up the assembler to write to the code buffer
	armSetAsmPtr(recPtr, recPtrEnd - recPtr, &s_recConstPool);
	u8* blockStart = armStartBlock();

	s_pCurBlock = PC_GETBLOCK(startpc);
	pxAssert(s_pCurBlock->GetFnptr() == (uptr)JITCompile);

	s_pCurBlockEx = recBlocks.Get(HWADDR(startpc));
	pxAssert(!s_pCurBlockEx || s_pCurBlockEx->startpc != HWADDR(startpc));
	s_pCurBlockEx = recBlocks.New(HWADDR(startpc), (uptr)blockStart);
	pxAssert(s_pCurBlockEx);

	// EELOAD hooks for fast boot
	if (HWADDR(startpc) == EELOAD_START)
	{
		const u32 mainjump = memRead32(EELOAD_START + 0x9c);
		if (mainjump >> 26 == 3) // JAL
			g_eeloadMain = ((EELOAD_START + 0xa0) & 0xf0000000U) | (mainjump << 2 & 0x0fffffffU);
	}

	if (g_eeloadMain && HWADDR(startpc) == HWADDR(g_eeloadMain))
	{
		armEmitCall((void*)eeloadHook);
		if (VMManager::Internal::IsFastBootInProgress())
		{
			const u32 typeAexecjump = memRead32(EELOAD_START + 0x470);
			const u32 typeBexecjump = memRead32(EELOAD_START + 0x5B0);
			const u32 typeCexecjump = memRead32(EELOAD_START + 0x618);
			const u32 typeDexecjump = memRead32(EELOAD_START + 0x600);
			if ((typeBexecjump >> 26 == 3) || (typeCexecjump >> 26 == 3) || (typeDexecjump >> 26 == 3))
				g_eeloadExec = EELOAD_START + 0x2B8;
			else if (typeAexecjump >> 26 == 3)
				g_eeloadExec = EELOAD_START + 0x170;
			else
				Console.WriteLn("recRecompile: Could not enable launch arguments for fast boot mode.");
		}
	}

	if (g_eeloadExec && HWADDR(startpc) == HWADDR(g_eeloadExec))
		armEmitCall((void*)eeloadHook2);

	g_branch = 0;

	// Reset recompiler state
	s_nBlockCycles = 0;
	s_nBlockInterlocked = false;
	pc = startpc;
	g_cpuHasConstReg = g_cpuFlushedConstReg = 1; // r0 is always const 0
	pxAssert(g_cpuConstRegs[0].UD[0] == 0);

	// Determine block boundaries
	u32 i = startpc;
	s_nEndBlock = 0xffffffff;
	s_branchTo = -1;
	s_branchIsUnconditional = false;

	while (1)
	{
		BASEBLOCK* pblock = PC_GETBLOCK(i);

		if (i != startpc)
		{
			if ((i & 0xffc) == 0x0) // page boundary
			{
				s_nEndBlock = i;
				break;
			}

			if (pblock->GetFnptr() != (uptr)JITCompile)
			{
				s_nEndBlock = i;
				break;
			}
		}

		cpuRegs.code = *(int*)PSM(i);

		switch (cpuRegs.code >> 26)
		{
			case 0: // special
				if (_Funct_ == 8 || _Funct_ == 9) // JR, JALR
				{
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				else if (_Funct_ == 12 || _Funct_ == 13) // SYSCALL, BREAK
				{
					s_nEndBlock = i + 4;
					goto StartRecomp;
				}
				break;
			case 1: // regimm
				if (_Rt_ < 4 || (_Rt_ >= 16 && _Rt_ < 20))
				{
					s_branchTo = _Imm_ * 4 + i + 4;
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;
			case 2: // J
			case 3: // JAL
				s_branchTo = (_Target_ << 2) | ((i + 4) & 0xf0000000);
				s_branchIsUnconditional = true;
				s_nEndBlock = i + 8;
				goto StartRecomp;
			case 4: case 5: case 6: case 7: // BEQ, BNE, BLEZ, BGTZ
			case 20: case 21: case 22: case 23: // BEQL, BNEL, BLEZL, BGTZL
				s_branchTo = _Imm_ * 4 + i + 4;
				s_nEndBlock = i + 8;
				goto StartRecomp;
			case 16: // COP0 — ERET
				if ((cpuRegs.code & 0x3F) == 24)
				{
					s_nEndBlock = i + 4;
					goto StartRecomp;
				}
				break;
		}

		i += 4;
	}

StartRecomp:

	// Detect wait/spin loops: if the block branches back to its own start
	// and only does loads, immediate arithmetic, nops, or mfc/cfc (i.e. it
	// polls a hardware register), we can fast-forward the cycle counter to
	// the next event instead of spinning.
	s_nBlockFF = false;
	if (s_branchTo == startpc)
	{
		s_nBlockFF = true;

		u32 reads = 0, loads = 1;

		for (i = startpc; i < s_nEndBlock; i += 4)
		{
			if (i == s_nEndBlock - 8)
				continue;
			cpuRegs.code = *(u32*)PSM(i);
			// nop
			if (cpuRegs.code == 0)
				continue;
			// cache, sync
			else if (_Opcode_ == 057 || (_Opcode_ == 0 && _Funct_ == 017))
				continue;
			// imm arithmetic
			else if ((_Opcode_ & 070) == 010 || (_Opcode_ & 076) == 030)
			{
				if (loads & 1 << _Rs_)
				{
					loads |= 1 << _Rt_;
					continue;
				}
				else
					reads |= 1 << _Rs_;
				if (reads & 1 << _Rt_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// common register arithmetic instructions
			else if (_Opcode_ == 0 && (_Funct_ & 060) == 040 && (_Funct_ & 076) != 050)
			{
				if (loads & 1 << _Rs_ && loads & 1 << _Rt_)
				{
					loads |= 1 << _Rd_;
					continue;
				}
				else
					reads |= 1 << _Rs_ | 1 << _Rt_;
				if (reads & 1 << _Rd_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// loads
			else if ((_Opcode_ & 070) == 040 || (_Opcode_ & 076) == 032 || _Opcode_ == 067)
			{
				if (loads & 1 << _Rs_)
				{
					loads |= 1 << _Rt_;
					continue;
				}
				else
					reads |= 1 << _Rs_;
				if (reads & 1 << _Rt_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// mfc*, cfc*
			else if ((_Opcode_ & 074) == 020 && _Rs_ < 4)
			{
				loads |= 1 << _Rt_;
			}
			else
			{
				s_nBlockFF = false;
				break;
			}
		}
	}

	// Build instruction info cache
	{
		u32 numinsts = (s_nEndBlock - startpc) / 4;
		if (numinsts + 1 > s_nInstCacheSize)
		{
			free(s_pInstCache);
			s_nInstCacheSize = numinsts + 1;
			s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
		}
		memset(s_pInstCache, 0, sizeof(EEINST) * (numinsts + 1));

		// Instruction analysis (size will be set after compilation, matching x86)
		g_pCurInstInfo = s_pInstCache;
		for (u32 j = startpc; j < s_nEndBlock; j += 4)
		{
			g_pCurInstInfo++;
			cpuRegs.code = *(u32*)PSM(j);
			// Could add instruction analysis here for register liveness, etc.
		}
	}

	// Emit SMC protection (integrity checks / write-protect) at the top of the block
	memory_protect_recompiled_code(startpc, (s_nEndBlock - startpc) >> 2);

	// Now emit code for each instruction
	g_pCurInstInfo = s_pInstCache;
	g_cpuFlushedPC = false;
	g_cpuFlushedCode = false;
	armGprCacheReset();

	while (!g_branch && pc < s_nEndBlock)
	{
		recompileNextInstruction(false, false);
	}

	pxAssert((pc - startpc) >> 2 <= 0xffff);
	s_pCurBlockEx->size = (pc - startpc) >> 2;

	// Handle block ending
	if (g_branch == 2) // syscall/break — event check, indirect dispatch
	{
		armFlushConstRegs();
		iBranchTest();
	}
	else
	{
		// Branch or fall-through
		if (!g_cpuFlushedPC)
		{
			armAsm->Mov(RWSCRATCH, pc);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		}
		armFlushConstRegs();
		iBranchTest(s_branchTo);
	}

	armEndBlock();

	// armEndBlock() advances armAsmPtr and sets armAsm=nullptr,
	// so use armAsmPtr directly instead of armGetCurrentCodePointer().
	recPtr = armAsmPtr;
	pxAssert((g_cpuHasConstReg & g_cpuFlushedConstReg) == g_cpuHasConstReg);

	// Point the BASEBLOCK at the compiled code so the dispatcher jumps directly
	// to it. Without this, the block stays pointed at JITCompile and gets
	// recompiled on every dispatch — burning through the code buffer.
	s_pCurBlock->SetFnptr((uptr)blockStart);

	if (!(pc & 0x10000000))
		maxrecmem = std::max((pc & ~0xa0000000), maxrecmem);

	s_pCurBlock = nullptr;
	s_pCurBlockEx = nullptr;
}

// ============================================================================
//  Memory management and lifecycle
// ============================================================================

static void recReserveRAM()
{
	recLutEntries = (Ps2MemSize::ExposedRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2) / 4;

	if (recRAMCopy.size() != Ps2MemSize::ExposedRam)
		recRAMCopy.resize(Ps2MemSize::ExposedRam);

	if (recLutReserve_RAM.size() != recLutEntries)
		recLutReserve_RAM.resize(recLutEntries);

	recLutUnmapped.resize(_64kb / 4);

	BASEBLOCK* basepos = recLutReserve_RAM.data();
	recRAM = basepos; basepos += (Ps2MemSize::ExposedRam / 4);
	recROM = basepos; basepos += (Ps2MemSize::Rom / 4);
	recROM1 = basepos; basepos += (Ps2MemSize::Rom1 / 4);
	recROM2 = basepos; basepos += (Ps2MemSize::Rom2 / 4);

	BASEBLOCK* unmapped = recLutUnmapped.data();
	for (int j = 0; j < 0x10000; j++)
		recLUT_SetPage(recLUT, hwLUT, unmapped, j, 0, 0);

	for (int j = 0x0000; j < (int)(Ps2MemSize::ExposedRam / 0x10000); j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x0000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x2000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x3000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x8000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xa000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xb000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xc000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xd000, j, j);
	}

	for (int j = 0x1fc0; j < 0x2000; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM, 0x0000, j, j - 0x1fc0);
		recLUT_SetPage(recLUT, hwLUT, recROM, 0x8000, j, j - 0x1fc0);
		recLUT_SetPage(recLUT, hwLUT, recROM, 0xa000, j, j - 0x1fc0);
	}

	for (int j = 0x1e00; j < 0x1e40; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0x0000, j, j - 0x1e00);
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0x8000, j, j - 0x1e00);
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0xa000, j, j - 0x1e00);
	}

	for (int j = 0x1e40; j < 0x1e80; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0x0000, j, j - 0x1e40);
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0x8000, j, j - 0x1e40);
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0xa000, j, j - 0x1e40);
	}
}

static void recReserve()
{
	Console.WriteLn("ARM64 recReserve: GetEERec=%p GetEERecEnd=%p", SysMemory::GetEERec(), SysMemory::GetEERecEnd());
	recPtr = SysMemory::GetEERec();
	recPtrEnd = SysMemory::GetEERecEnd() - _64kb;
	Console.WriteLn("ARM64 recReserve: recPtr=%p recPtrEnd=%p (capacity=%zu)", recPtr, recPtrEnd, (size_t)(recPtrEnd - recPtr));

	// Initialize constant pool at the end of the code buffer
	s_recConstPool.Init(recPtrEnd, _64kb);
	Console.WriteLn("ARM64 recReserve: constant pool initialized");

	recReserveRAM();
	Console.WriteLn("ARM64 recReserve: RAM reserved");

	pxAssertRel(!s_pInstCache, "InstCache not allocated");
	s_nInstCacheSize = 128;
	s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
	if (!s_pInstCache)
		pxFailRel("Failed to allocate R5900 InstCache array");
	Console.WriteLn("ARM64 recReserve: done");
}

alignas(16) static u16 manual_page[Ps2MemSize::TotalRam >> 12];
alignas(16) static u8 manual_counter[Ps2MemSize::TotalRam >> 12];

// Emit SMC protection at the top of a compiled block.
//
// ProtMode_None / ProtMode_Write:
//   Install OS write-protection on the page at compile time.  No code emitted.
//   The vtlb fault handler clears stale blocks when the game writes to the page.
//
// ProtMode_Manual:
//   Emit per-DWORD integrity checks inline.  If any word differs from the
//   compile-time snapshot, the block was stale and we jump to DispatchBlockDiscard.
//   Counted blocks also accumulate a weighted value into manual_page[]; when the
//   16-bit accumulator overflows the page switches to OS write-protection via
//   DispatchPageReset.
static void memory_protect_recompiled_code(u32 startpc, u32 size)
{
	const u32 inpage_ptr = HWADDR(startpc);
	const u32 inpage_sz  = size * 4; // bytes

	// Kernel thread-stack pages must always use manual protection
	const bool contains_thread_stack =
		((startpc >> 12) == 0x81) || ((startpc >> 12) == 0x80001);
	const vtlb_ProtectionMode PageType =
		contains_thread_stack ? ProtMode_Manual : mmap_GetRamPageInfo(inpage_ptr);

	switch (PageType)
	{
		case ProtMode_NotRequired:
			break;

		case ProtMode_None:
		case ProtMode_Write:
			// Switch to OS write-protection now; the vtlb fault handler will
			// call recClear when the game writes to this page.  No emitted code.
			mmap_MarkCountedRamPage(inpage_ptr);
			manual_page[inpage_ptr >> 12] = 0;
			break;

		case ProtMode_Manual:
		{
			// Pre-load dyna_block_discard args before the integrity loop so
			// they are live at DispatchBlockDiscard regardless of which word
			// fails the check.
			armAsm->Mov(RWARG1, inpage_ptr);
			armAsm->Mov(RWARG2, inpage_sz / 4);

			// Emit one compare per DWORD in the block.
			u32 lpc = inpage_ptr;
			u32 stg = inpage_sz;
			while (stg > 0)
			{
				const u32 snapshot = *(const u32*)PSM(lpc);

				// Load current value from PS2 RAM (RMEMBASE = eeMem->Main)
				armAsm->Mov(RSCRATCHGPR, (uint64_t)lpc);
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RMEMBASE, RSCRATCHGPR));

				// Compare with compile-time snapshot
				armAsm->Mov(RSCRATCHGPR2, (uint64_t)(u32)snapshot);
				armAsm->Cmp(RWSCRATCH, RWSCRATCH2);
				armEmitCondBranch(a64::ne, DispatchBlockDiscard);

				stg -= 4;
				lpc += 4;
			}

			// Counted blocks: accumulate block size into manual_page[page].
			// When the 16-bit accumulator overflows the page switches to OS
			// write-protection (cheaper than continued inline checks).
			if (!contains_thread_stack && manual_counter[inpage_ptr >> 12] <= 3)
			{
				armMoveAddressToReg(RSCRATCHGPR, &manual_page[inpage_ptr >> 12]);
				armAsm->Ldrh(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR));
				armAsm->Add(RWSCRATCH2, RWSCRATCH2, size);
				armAsm->Strh(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR));
				armAsm->Cmp(RWSCRATCH2, 0x10000);
				armEmitCondBranch(a64::ge, DispatchPageReset);
			}
			break;
		}
	}
}

static void ClearRecLUT(BASEBLOCK* base, int count)
{
	for (int i = 0; i < count / 4; i++)
		base[i].SetFnptr((uptr)JITCompile);
}

static void recResetRaw()
{
	Console.WriteLn(Color_StrongBlack, "EE/ARM64 Recompiler Reset");

	if (CHECK_EXTRAMEM != extraRam)
	{
		Console.WriteLn("ARM64 recReset: extra RAM changed, re-reserving");
		recReserveRAM();
		extraRam = !extraRam;
	}

	EE::Profiler.Reset();
	Console.WriteLn("ARM64 recReset: profiler reset, setting up asm ptr=%p capacity=%zu", SysMemory::GetEERec(), (size_t)(recPtrEnd - SysMemory::GetEERec()));

	// Set up assembler at the beginning of the code buffer
	armSetAsmPtr(SysMemory::GetEERec(), recPtrEnd - SysMemory::GetEERec(), &s_recConstPool);
	s_recConstPool.Reset();
	Console.WriteLn("ARM64 recReset: asm ptr set, generating dispatchers");

	_DynGen_Dispatchers();
	// armEndBlock() already advanced armAsmPtr past emitted code and set armAsm=nullptr,
	// so we can't use armGetCurrentCodePointer() (which dereferences armAsm).
	recPtr = armAsmPtr;

	ClearRecLUT(recLutReserve_RAM.data(),
		Ps2MemSize::ExposedRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2);

	for (int j = 0; j < _64kb / 4; j++)
		recLutUnmapped.data()[j].SetFnptr((uptr)UnmappedRecLUTPage);

	recRAMCopy.fill(0);
	maxrecmem = 0;

	if (s_pInstCache)
		memset(s_pInstCache, 0, sizeof(EEINST) * s_nInstCacheSize);

	recBlocks.Reset();
	vtlb_ClearLoadStoreInfo();

	g_branch = 0;
	g_resetEeScalingStats = true;

	memset(manual_page, 0, sizeof(manual_page));
	memset(manual_counter, 0, sizeof(manual_counter));
}

static void recShutdown()
{
	recRAMCopy.deallocate();
	recLutReserve_RAM.deallocate();

	recBlocks.Reset();
	recRAM = recROM = recROM1 = recROM2 = nullptr;

	safe_free(s_pInstCache);
	s_nInstCacheSize = 0;

	recPtr = nullptr;
	recPtrEnd = nullptr;
}

static void recStep()
{
}

static fastjmp_buf m_SetJmp_StateCheck;

static void recExitExecution()
{
	fastjmp_jmp(&m_SetJmp_StateCheck, 1);
}

static void recSafeExitExecution()
{
	eeRecExitRequested = true;

	if (!eeEventTestIsActive)
	{
		cpuRegs.nextEventCycle = 0;
	}
	else
	{
		if (psxRegs.iopCycleEE > 0)
		{
			psxRegs.iopBreak += psxRegs.iopCycleEE;
			psxRegs.iopCycleEE = 0;
		}
	}
}

static void recResetEE()
{
	Console.WriteLn("ARM64 recResetEE: eeCpuExecuting=%d", (int)eeCpuExecuting);
	if (eeCpuExecuting)
	{
		eeRecNeedsReset = true;
		recSafeExitExecution();
		return;
	}

	recResetRaw();
}

static fastjmp_buf m_SetJmp_CancelInstruction;

static void recCancelInstruction()
{
	// An interpreter stub (called via armCallInterpreter) raised an error
	// (Address Error, TLB miss, etc.). The interpreter pre-increments
	// cpuRegs.pc before executing, so armFlushPC already wrote pc+4.
	// Don't exit recExecute — just longjmp back to the dispatch loop so
	// cycles keep accumulating and events fire. This matches the
	// interpreter's intCancelInstruction which longjmps back into its
	// for(;;) loop without exiting intExecute.
	//
	// The x86 JIT marks this as "should never happen" because it compiles
	// loads/stores natively. We hit it because we use interpreter stubs.
	fastjmp_jmp(&m_SetJmp_CancelInstruction, 1);
}

static void recExecute()
{
	// Console.WriteLn("ARM64 recExecute: enter, eeRecNeedsReset=%d EnterRecompiledCode=%p", (int)eeRecNeedsReset, EnterRecompiledCode);
	if (eeRecNeedsReset)
	{
		eeRecNeedsReset = false;
		recResetRaw();
	}

	if (!fastjmp_set(&m_SetJmp_StateCheck))
	{
		eeCpuExecuting = true;

		// Set up the cancel-instruction landing pad. When an interpreter stub
		// calls Cpu->CancelInstruction(), we longjmp here instead of fully
		// exiting recExecute. This keeps us inside the execution loop so we
		// can bump cycles, check events, and re-dispatch — matching the
		// interpreter's behavior where intCancelInstruction re-enters the
		// for(;;) loop.
		if (fastjmp_set(&m_SetJmp_CancelInstruction))
		{
			// Landed here from recCancelInstruction. Add a small cycle bump
			// so the system makes progress, then check events and re-dispatch.
			cpuRegs.cycle += 8;
			if (cpuRegs.cycle >= cpuRegs.nextEventCycle)
				_cpuEventTest_Shared();

			if (eeRecExitRequested)
			{
				eeRecExitRequested = false;
				goto exit;
			}
		}

		// Console.WriteLn("ARM64 recExecute: jumping to EnterRecompiledCode");
		((void (*)())EnterRecompiledCode)();
	}

exit:
	eeCpuExecuting = false;
	EE::Profiler.Print();
}

static void dyna_block_discard(u32 start, u32 sz)
{
	DevCon.WriteLn(Color_StrongGray, "Clearing Manual Block @ 0x%08X  [size=%d]", start, sz * 4);
	recClear(start, sz);
}

static void dyna_page_reset(u32 start, u32 sz)
{
	recClear(start & ~0xfffUL, 0x400);
	manual_counter[start >> 12]++;
	mmap_MarkCountedRamPage(start);
}

static void recClear(u32 addr, u32 size)
{
	if ((addr) >= maxrecmem || !(recLUT[(addr) >> 16] + (addr & ~0xFFFFUL)))
		return;
	addr = HWADDR(addr);

	int blockidx = recBlocks.LastIndex(addr + size * 4 - 4);

	if (blockidx == -1)
		return;

	u32 lowerextent = static_cast<u32>(-1), upperextent = 0, ceiling = static_cast<u32>(-1);

	BASEBLOCKEX* pexblock = recBlocks[blockidx + 1];
	if (pexblock)
		ceiling = pexblock->startpc;

	int toRemoveLast = blockidx;

	while ((pexblock = recBlocks[blockidx]))
	{
		u32 blockstart = pexblock->startpc;
		u32 blockend = pexblock->startpc + pexblock->size * 4;
		BASEBLOCK* pblock = PC_GETBLOCK(blockstart);

		if (pblock == s_pCurBlock)
		{
			if (toRemoveLast != blockidx)
				recBlocks.Remove((blockidx + 1), toRemoveLast);
			toRemoveLast = --blockidx;
			continue;
		}

		if (blockend <= addr)
		{
			lowerextent = std::max(lowerextent, blockend);
			break;
		}

		lowerextent = std::min(lowerextent, blockstart);
		upperextent = std::max(upperextent, blockend);

		pblock->SetFnptr((uptr)JITCompile);

		blockidx--;
	}

	if (toRemoveLast != blockidx)
		recBlocks.Remove((blockidx + 1), toRemoveLast);

	upperextent = std::min(upperextent, ceiling);

	for (u32 cleared = lowerextent; cleared < upperextent; cleared += 4)
	{
		BASEBLOCK* pblock = PC_GETBLOCK(cleared);
		pblock->SetFnptr((uptr)JITCompile);
	}
}

R5900cpu recCpu = {
	recReserve,
	recShutdown,
	recResetEE,
	recStep,
	recExecute,
	recSafeExitExecution,
	recCancelInstruction,
	recClear,
};

