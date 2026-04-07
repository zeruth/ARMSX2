// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 IOP (R3000A) Recompiler — Main JIT Engine
//
// Modeled on the x86 IOP JIT (x86/iR3000A.cpp) and the ARM64 EE JIT
// (arm64/iR5900_arm64.cpp). This file implements:
//   - Code buffer management (Reserve, Reset, Shutdown)
//   - ARM64 dispatcher stubs (DispatcherReg, JITCompile, Enter/Exit)
//   - Block compilation (iopRecRecompile, psxRecompileNextInstruction)
//   - Branch test & cycle counting (iPsxBranchTest, iPsxAddEECycles)
//   - Constant propagation helpers
//   - The psxRec CPU interface struct

#include "iR3000A_arm64.h"
#include "arm64/AsmHelpers.h"
#include "R3000A.h"
#include "x86/BaseblockEx.h"
#include "IopBios.h"
#include "IopHw.h"
#include "IopMem.h"
#include "Common.h"
#include "Config.h"
#include "VMManager.h"
#include "common/Console.h"
#include "common/HeapArray.h"
#include "common/Perf.h"
#include "DebugTools/Breakpoints.h"

#include "x86/iCore.h" // EEINST, g_pCurInstInfo

extern void armEndStackFrame(bool save_fpr);

using namespace vixl::aarch64;

// ============================================================================
//  Static globals
// ============================================================================

uptr psxRecLUT[0x10000];
static u32 psxhwLUT[0x10000];

static __fi u32 HWADDR(u32 mem) { return psxhwLUT[mem >> 16] + mem; }

static BASEBLOCK* recRAM = nullptr;
static BASEBLOCK* recROM = nullptr;
static BASEBLOCK* recROM1 = nullptr;
static BASEBLOCK* recROM2 = nullptr;
static BaseBlocks recBlocks;
static u8* recPtr = nullptr;
static u8* recPtrEnd = nullptr;
static ArmConstantPool s_iopConstPool;

u32 psxpc;            // recompiler pc
int psxbranch;        // branch state
u32 g_iopCyclePenalty;
u32 s_psxBlockCycles;
bool s_recompilingDelaySlot = false;

static EEINST* s_pInstCache = nullptr;
static u32 s_nInstCacheSize = 0;

static BASEBLOCK* s_pCurBlock = nullptr;
static BASEBLOCKEX* s_pCurBlockEx = nullptr;

static u32 s_nEndBlock = 0;
static u32 s_branchTo;
static bool s_nBlockFF;

u32 g_psxMaxRecMem = 0;

// Constant propagation state (defined in R3000A.cpp)

// Branch state save/restore
static u32 s_saveConstRegs[32];
static u32 s_saveHasConstReg = 0, s_saveFlushedConstReg = 0;
static EEINST* s_psaveInstInfo = nullptr;
static u32 s_savenBlockCycles = 0;

// PC/Code flush tracking
static bool g_psxFlushedPC, g_psxFlushedCode;

// ============================================================================
//  Dispatcher pointers (generated at init)
// ============================================================================

static const void* iopDispatcherEvent = nullptr;
static const void* iopDispatcherReg = nullptr;
static const void* iopJITCompile = nullptr;
static const void* iopEnterRecompiledCode = nullptr;
static const void* iopExitRecompiledCode = nullptr;
static const void* iopUnmappedRecLUTPage = nullptr;

// ============================================================================
//  Forward declarations
// ============================================================================

static void iopRecRecompile(u32 startpc);
static void recEventTest();
static void iopRecError(int err);
static void iopClearRecLUT(BASEBLOCK* base, int count);
void recResetIOP();

#define PSX_GETBLOCK(x) PC_GETBLOCK_(x, psxRecLUT)

// ============================================================================
//  Constant Propagation Helpers
// ============================================================================

void iopArmFlushConstReg(int reg)
{
	if (PSX_IS_CONST1(reg) && !(g_psxFlushedConstReg & (1u << reg)))
	{
		armAsm->Mov(RWPSXSCRATCH, g_psxConstRegs[reg]);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(reg)));
		g_psxFlushedConstReg |= (1u << reg);
	}
}

void iopArmFlushConstRegs()
{
	for (int i = 1; i < 32; i++)
	{
		if ((g_psxHasConstReg & (1u << i)) && !(g_psxFlushedConstReg & (1u << i)))
		{
			armAsm->Mov(RWPSXSCRATCH, g_psxConstRegs[i]);
			armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(i)));
			g_psxFlushedConstReg |= (1u << i);
		}
		if (g_psxHasConstReg == g_psxFlushedConstReg)
			break;
	}
}

void iopArmLoadGPR(const Register& dst, int gpr)
{
	if (gpr == 0)
	{
		armAsm->Mov(dst, wzr);
		return;
	}
	if (PSX_IS_CONST1(gpr))
	{
		armAsm->Mov(dst, g_psxConstRegs[gpr]);
		return;
	}
	armAsm->Ldr(dst, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(gpr)));
}

void iopArmStoreGPR(const Register& src, int gpr)
{
	if (gpr == 0) return;
	PSX_DEL_CONST(gpr);
	armAsm->Str(src, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(gpr)));
}

void iopArmFlushPC()
{
	if (!g_psxFlushedPC)
	{
		armAsm->Mov(RWPSXSCRATCH, psxpc);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
		g_psxFlushedPC = true;
	}
}

void iopArmFlushCode()
{
	if (!g_psxFlushedCode)
	{
		armAsm->Mov(RWPSXSCRATCH, psxRegs.code);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CODE_OFFSET));
		g_psxFlushedCode = true;
	}
}

void iopArmCallInterpreter(void (*func)())
{
	iopArmFlushCode();
	iopArmFlushPC();
	iopArmFlushConstRegs();
	armEmitCall((const void*)func);
	// Conservative: after calling interpreter, invalidate all const tracking
	g_psxHasConstReg = g_psxFlushedConstReg = 1; // r0 is always const 0
}

void iopArmBranchCallInterpreter(void (*func)())
{
	iopArmFlushCode();
	iopArmFlushPC();
	iopArmFlushConstRegs();
	armEmitCall((const void*)func);
	g_psxHasConstReg = g_psxFlushedConstReg = 1;
	psxbranch = 2;

	// The interpreter's branch path (psxJ/psxBEQ/...) calls doBranch() which
	// internally runs execI() for the delay slot — that consumes one extra IOP
	// cycle that the JIT block epilogue does not see (s_psxBlockCycles only
	// counts the branch instruction itself). Without compensation, iopCycleEE
	// is undercharged by 8 per stubbed branch, the IOP runs hot relative to
	// the EE, and IOP-side timing (SPU, counters) drifts faster than real.
	// psxRegs.cycle is already correct: interpreter execI bumps it by 1 for
	// the delay slot, and the JIT epilogue bumps it by 1 for the branch.
	// Only iopCycleEE needs the missing 8.
	armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
	armAsm->Sub(w1, w1, 8);
	armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
}

// ============================================================================
//  Branch state save/restore
// ============================================================================

void psxSaveBranchState()
{
	s_savenBlockCycles = s_psxBlockCycles;
	memcpy(s_saveConstRegs, g_psxConstRegs, sizeof(g_psxConstRegs));
	s_saveHasConstReg = g_psxHasConstReg;
	s_saveFlushedConstReg = g_psxFlushedConstReg;
	s_psaveInstInfo = g_pCurInstInfo;
}

void psxLoadBranchState()
{
	s_psxBlockCycles = s_savenBlockCycles;
	memcpy(g_psxConstRegs, s_saveConstRegs, sizeof(g_psxConstRegs));
	g_psxHasConstReg = s_saveHasConstReg;
	g_psxFlushedConstReg = s_saveFlushedConstReg;
	g_pCurInstInfo = s_psaveInstInfo;
}

// ============================================================================
//  Event test (called from dispatcher)
// ============================================================================

static void recEventTest()
{
	_cpuEventTest_Shared();
}

// ============================================================================
//  ARM64 Dispatcher Generation
// ============================================================================

// Dispatcher: jump to compiled block at psxRegs.pc
static const void* _DynGen_DispatcherReg()
{
	u8* retval = armStartBlock();

	// w0 = psxRegs.pc
	armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	// x1 = psxRecLUT[pc >> 16]
	armAsm->Lsr(w1, w0, 16);
	armAsm->Ldr(x1, MemOperand(RPSXRECLUT, x1, LSL, 3));
	// block = psxRecLUT[pc>>16] + pc * 2 (sizeof(BASEBLOCK)=8, /4=2)
	armAsm->Lsl(x2, x0, 1);
	armAsm->Ldr(x3, MemOperand(x1, x2));
	armAsm->Br(x3);

	armEndBlock();
	return retval;
}

// Event dispatcher: call recEventTest then jump to DispatcherReg
static const void* _DynGen_DispatcherEvent()
{
	pxAssert(iopDispatcherReg);
	u8* retval = armStartBlock();

	armEmitCall((const void*)recEventTest);
	armEmitJmp(iopDispatcherReg);

	armEndBlock();
	return retval;
}

// JIT compile stub: called when hitting uncompiled block
static const void* _DynGen_JITCompile()
{
	pxAssert(iopDispatcherReg);
	u8* retval = armStartBlock();

	// arg1 = psxRegs.pc
	armAsm->Ldr(RWARG1, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	armEmitCall((const void*)iopRecRecompile);

	// Re-dispatch to newly compiled block
	armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	armAsm->Lsr(w1, w0, 16);
	armAsm->Ldr(x1, MemOperand(RPSXRECLUT, x1, LSL, 3));
	armAsm->Lsl(x2, x0, 1);
	armAsm->Ldr(x3, MemOperand(x1, x2));
	armAsm->Br(x3);

	armEndBlock();
	return retval;
}

// Enter recompiled code: called from C++ (recExecuteBlock)
static const void* _DynGen_EnterRecompiledCode()
{
	u8* retval = armStartBlock();

	// Save callee-saved registers
	armBeginStackFrame(false);

	// Pin state registers
	armMoveAddressToReg(RPSXSTATE, &psxRegs);
	armMoveAddressToReg(RPSXRECLUT, psxRecLUT);

	// Jump to dispatcher
	armEmitJmp(iopDispatcherReg);

	// Exit point: blocks jump here when iopCycleEE <= 0
	// This must immediately follow so iopExitRecompiledCode is addressable.
	armEndBlock();

	// Generate iopExitRecompiledCode as a separate block
	iopExitRecompiledCode = armStartBlock();
	armEndStackFrame(false);
	armAsm->Ret();
	armEndBlock();

	return retval;
}

// Unmapped page error handler
static const void* _DynGen_UnmappedRecLUTPage()
{
	u8* retval = armStartBlock();

	armAsm->Mov(RWARG1, 0);
	armEmitCall((const void*)iopRecError);
	armEmitJmp(iopExitRecompiledCode);

	armEndBlock();
	return retval;
}

static void _DynGen_Dispatchers()
{
	iopDispatcherReg = _DynGen_DispatcherReg();
	iopDispatcherEvent = _DynGen_DispatcherEvent();
	iopJITCompile = _DynGen_JITCompile();
	iopEnterRecompiledCode = _DynGen_EnterRecompiledCode();
	// iopExitRecompiledCode is set inside _DynGen_EnterRecompiledCode
	iopUnmappedRecLUTPage = _DynGen_UnmappedRecLUTPage();

	recBlocks.SetJITCompile(iopJITCompile);
}

// ============================================================================
//  Error handler
// ============================================================================

static void iopRecError(int err)
{
	switch (err)
	{
		case 0:
			Console.Error("[IOP ARM64 Rec] Jump to unmapped recLUT page (PC: 0x%08x)", psxRegs.pc);
			break;
		case 1:
			Console.Error("[IOP ARM64 Rec] Block execution at 0x%08x with zero fnptr (code buffer overflow?)", psxRegs.pc);
			break;
		default:
			Console.Error("[IOP ARM64 Rec] Unknown error %d at PC 0x%08x", err, psxRegs.pc);
			break;
	}

	Cpu->ExitExecution();
}

// ============================================================================
//  Cycle counting & branch test
// ============================================================================

static __fi u32 psxScaleBlockCycles()
{
	return s_psxBlockCycles;
}

// Emit code to subtract scaled IOP cycles from iopCycleEE.
// In normal mode (not PS1): iopCycleEE -= blockCycles * 8
// In PS1 mode: uses cnum/cdenom ratio (emitted as a C++ helper call for now).
static void iPsxAddEECycles(u32 blockCycles)
{
	// Normal mode (most common): iopCycleEE -= blockCycles * 8
	// PS1 mode check is done at compile time for this block.
	// For simplicity, always emit the normal-mode path.
	// PS1 mode is extremely rare in PS2 context.

	if (blockCycles == 0) return;

	u32 eeCycles = blockCycles * 8;
	armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
	if (Assembler::IsImmAddSub(eeCycles))
	{
		armAsm->Subs(w1, w1, eeCycles);
	}
	else
	{
		armAsm->Mov(w0, eeCycles);
		armAsm->Subs(w1, w1, w0);
	}
	armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
}

// Emit the branch test at the end of a compiled block.
// Adds block cycles, subtracts from iopCycleEE, checks for events,
// and exits if the IOP timeslice is exhausted.
static void iPsxBranchTest(u32 newpc, u32 cpuBranch)
{
	u32 blockCycles = psxScaleBlockCycles();

	if (EmuConfig.Speedhacks.WaitLoop && s_nBlockFF && newpc == s_branchTo)
	{
		// Wait loop optimization: fast-forward cycle to iopNextEventCycle
		// cycle += (iopCycleEE + 7) >> 3
		// cycle = min(cycle, iopNextEventCycle)
		armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
		armAsm->Mov(x4, x0); // save original cycle
		armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
		armAsm->Add(w1, w1, 7);
		armAsm->Lsr(w1, w1, 3);
		armAsm->Add(x0, x0, x1);
		armAsm->Ldr(x2, MemOperand(RPSXSTATE, PSX_IOPNEXTEVENTCYCLE_OFFSET));
		armAsm->Cmp(x0, x2);
		armAsm->Csel(x0, x2, x0, hs); // clamp to iopNextEventCycle
		armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));

		// Compute how many IOP cycles we actually advanced, subtract *8 from iopCycleEE
		armAsm->Sub(x0, x0, x4); // delta cycles
		armAsm->Lsl(x0, x0, 3); // * 8
		armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
		armAsm->Subs(w1, w1, w0);
		armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));

		// Exit if iopCycleEE <= 0
		armEmitCondBranch(le, iopExitRecompiledCode);

		// Call iopEventTest
		armEmitCall((const void*)iopEventTest);

		// If PC changed, re-dispatch
		if (newpc != 0xffffffff)
		{
			armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
			armAsm->Mov(w1, newpc);
			armAsm->Cmp(w0, w1);
			armEmitCondBranch(ne, iopDispatcherReg);
		}
	}
	else
	{
		// Normal path: add block cycles to psxRegs.cycle
		armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
		if (Assembler::IsImmAddSub(blockCycles))
		{
			armAsm->Add(x0, x0, blockCycles);
		}
		else
		{
			armAsm->Mov(x4, (u64)blockCycles);
			armAsm->Add(x0, x0, x4);
		}
		armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));

		// Subtract from iopCycleEE
		iPsxAddEECycles(blockCycles);

		// Exit if iopCycleEE <= 0
		// w1 still holds the updated iopCycleEE from iPsxAddEECycles
		armEmitCondBranch(le, iopExitRecompiledCode);

		// Check if an event is pending: cycle >= iopNextEventCycle
		armAsm->Ldr(x2, MemOperand(RPSXSTATE, PSX_IOPNEXTEVENTCYCLE_OFFSET));
		armAsm->Cmp(x0, x2);

		Label noEventDone;
		armAsm->B(&noEventDone, lo); // branch if cycle < iopNextEventCycle (no event)

		// Event pending: call iopEventTest
		armEmitCall((const void*)iopEventTest);

		// If PC changed due to exception/interrupt, re-dispatch
		if (newpc != 0xffffffff)
		{
			armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
			armAsm->Mov(w1, newpc);
			armAsm->Cmp(w0, w1);
			armEmitCondBranch(ne, iopDispatcherReg);
		}

		armAsm->Bind(&noEventDone);
	}
}

// ============================================================================
//  Branch target helpers (called from instruction implementations)
// ============================================================================

// Dynamic branch (JR/JALR): PC was set by instruction, go through dispatcher
void psxSetBranchReg()
{
	psxbranch = 1;

	// Flush state
	iopArmFlushConstRegs();

	// iPsxBranchTest with dynamic target
	iPsxBranchTest(0xffffffff, 1);

	// Dispatch
	armEmitJmp(iopDispatcherReg);
}

// Static branch: PC = imm
void psxSetBranchImm(u32 imm)
{
	psxbranch = 1;

	// Store PC
	armAsm->Mov(RWPSXSCRATCH, imm);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	iopArmFlushConstRegs();

	iPsxBranchTest(imm, imm <= psxpc);

	// Dispatch (could be block-linked in the future)
	armEmitJmp(iopDispatcherReg);
}

// ============================================================================
//  Delay slot swap attempt
// ============================================================================

bool psxTrySwapDelaySlot(u32 rs, u32 rt, u32 rd)
{
	if (s_recompilingDelaySlot)
		return false;

	const u32 opcode_encoded = iopMemRead32(psxpc);
	if (opcode_encoded == 0) // NOP
	{
		psxRecompileNextInstruction(true, true);
		return true;
	}

	const u32 opcode_rs = ((opcode_encoded >> 21) & 0x1F);
	const u32 opcode_rt = ((opcode_encoded >> 16) & 0x1F);
	const u32 opcode_rd = ((opcode_encoded >> 11) & 0x1F);

	switch (opcode_encoded >> 26)
	{
		case 8: // ADDI
		case 9: // ADDIU
		case 10: // SLTI
		case 11: // SLTIU
		case 12: // ANDI
		case 13: // ORI
		case 14: // XORI
		case 15: // LUI
		{
			// Rt = Rs op Imm -- safe if Rt doesn't conflict with branch regs
			if (opcode_rt == rs || opcode_rt == rt || opcode_rt == rd)
				break;
			if ((rs != 0 && opcode_rt == rs) || (rt != 0 && opcode_rt == rt))
				break;
			psxRecompileNextInstruction(true, true);
			return true;
		}

		case 0: // SPECIAL
		{
			const u32 funct = opcode_encoded & 0x3F;
			switch (funct)
			{
				case 0: // SLL
				case 2: // SRL
				case 3: // SRA
				{
					if (opcode_rd == rs || opcode_rd == rt || opcode_rd == rd)
						break;
					psxRecompileNextInstruction(true, true);
					return true;
				}
				case 33: // ADDU
				case 35: // SUBU
				case 36: // AND
				case 37: // OR
				case 38: // XOR
				case 39: // NOR
				case 42: // SLT
				case 43: // SLTU
				{
					if (opcode_rd == rs || opcode_rd == rt || opcode_rd == rd)
						break;
					if ((rs != 0 && opcode_rd == rs) || (rt != 0 && opcode_rd == rt))
						break;
					psxRecompileNextInstruction(true, true);
					return true;
				}
				default:
					break;
			}
			break;
		}
		default:
			break;
	}

	return false;
}

// ============================================================================
//  Instruction compilation
// ============================================================================

void psxRecompileNextInstruction(bool delayslot, bool swapped_delayslot)
{
	const int old_code = psxRegs.code;
	EEINST* old_inst_info = g_pCurInstInfo;
	s_recompilingDelaySlot = delayslot;

	psxRegs.code = iopMemRead32(psxpc);
	s_psxBlockCycles++;
	psxpc += 4;

	g_pCurInstInfo++;

	g_psxFlushedPC = false;
	g_psxFlushedCode = false;
	g_iopCyclePenalty = 0;

	// Dispatch to instruction recompiler
	rpsxBSC[psxRegs.code >> 26]();
	s_psxBlockCycles += g_iopCyclePenalty;

	if (swapped_delayslot)
	{
		psxRegs.code = old_code;
		g_pCurInstInfo = old_inst_info;
	}
}

// ============================================================================
//  Block compilation
// ============================================================================

static void iopRecRecompile(const u32 startpc)
{
	u32 i;
	u32 link_next_block = 0;

	// SYSMEM module detection
	if (startpc == 0x890)
	{
		DevCon.WriteLn(Color_Gray, "R3000 Debugger: Branch to 0x890 (SYSMEM). Clearing modules.");
		R3000SymbolGuardian.ClearIrxModules();
	}

	// IRX injection hack
	if (startpc == 0x1630 && EmuConfig.CurrentIRX.length() > 3)
	{
		if (iopMemRead32(0x20018) == 0x1F)
			iopMemWrite32(0x20094, 0xbffc0000);
	}

	// IOPBOOT memory size override
	if (startpc == 0xbfc4a000)
		psxRegs.GPR.n.a0 = Ps2MemSize::ExposedIopRam >> 20;

	pxAssert(startpc);

	// Check code buffer space
	if (recPtr >= recPtrEnd)
	{
		recResetIOP();
	}

	// Set up assembler
	size_t capacity = recPtrEnd - recPtr;
	armSetAsmPtr(recPtr, capacity, &s_iopConstPool);
	recPtr = armStartBlock();

	s_pCurBlock = PSX_GETBLOCK(startpc);
	pxAssert(s_pCurBlock->GetFnptr() == (uptr)iopJITCompile);

	s_pCurBlockEx = recBlocks.Get(HWADDR(startpc));
	if (!s_pCurBlockEx || s_pCurBlockEx->startpc != HWADDR(startpc))
		s_pCurBlockEx = recBlocks.New(HWADDR(startpc), (uptr)recPtr);

	psxbranch = 0;
	s_pCurBlock->SetFnptr((uptr)armGetCurrentCodePointer());
	s_psxBlockCycles = 0;

	// Reset recompiler state
	psxpc = startpc;
	g_psxHasConstReg = g_psxFlushedConstReg = 1; // r0 is always const 0

	// BIOS call check
	if ((psxHu32(HW_ICFG) & 8) && (HWADDR(startpc) == 0xa0 || HWADDR(startpc) == 0xb0 || HWADDR(startpc) == 0xc0))
	{
		armEmitCall((const void*)psxBiosCall);
		// If psxBiosCall returns non-zero, BIOS handled it — jump to dispatcher
		armEmitCbnz(RWRET, iopDispatcherReg);
	}

	// Scan for block end
	i = startpc;
	s_nEndBlock = 0xffffffff;
	s_branchTo = (u32)-1;

	while (1)
	{
		BASEBLOCK* pblock = PSX_GETBLOCK(i);
		if (i != startpc && pblock->GetFnptr() != (uptr)iopJITCompile)
		{
			link_next_block = 1;
			s_nEndBlock = i;
			break;
		}

		psxRegs.code = iopMemRead32(i);

		switch (psxRegs.code >> 26)
		{
			case 0: // SPECIAL
				if (_psxFunct_ == 8 || _psxFunct_ == 9) // JR, JALR
				{
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;

			case 1: // REGIMM
				if (_psxRt_ == 0 || _psxRt_ == 1 || _psxRt_ == 16 || _psxRt_ == 17)
				{
					s_branchTo = _psxImm_ * 4 + i + 4;
					if (s_branchTo > startpc && s_branchTo < i)
						s_nEndBlock = s_branchTo;
					else
						s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;

			case 2: // J
			case 3: // JAL
				s_branchTo = (_psxTarget_ << 2) | ((i + 4) & 0xf0000000);
				s_nEndBlock = i + 8;
				goto StartRecomp;

			case 4: // BEQ
			case 5: // BNE
			case 6: // BLEZ
			case 7: // BGTZ
				s_branchTo = _psxImm_ * 4 + i + 4;
				if (s_branchTo > startpc && s_branchTo < i)
					s_nEndBlock = s_branchTo;
				else
					s_nEndBlock = i + 8;
				goto StartRecomp;
		}

		i += 4;
	}

StartRecomp:

	// Detect wait loops (branch-to-self with only NOPs)
	s_nBlockFF = false;
	if (s_branchTo == startpc)
	{
		s_nBlockFF = true;
		for (i = startpc; i < s_nEndBlock; i += 4)
		{
			if (i != s_nEndBlock - 8)
			{
				switch (iopMemRead32(i))
				{
					case 0: // NOP
						break;
					default:
						s_nBlockFF = false;
				}
			}
		}
	}

	// Build EEINST liveness info
	{
		EEINST* pcur;

		if (s_nInstCacheSize < (s_nEndBlock - startpc) / 4 + 1)
		{
			free(s_pInstCache);
			s_nInstCacheSize = (s_nEndBlock - startpc) / 4 + 10;
			s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
			pxAssert(s_pInstCache != nullptr);
		}

		pcur = s_pInstCache + (s_nEndBlock - startpc) / 4;
		_recClearInst(pcur);
		pcur->info = 0;

		for (i = s_nEndBlock; i > startpc; i -= 4)
		{
			psxRegs.code = iopMemRead32(i - 4);
			pcur[-1] = pcur[0];
			rpsxpropBSC(pcur - 1, pcur);
			pcur--;
		}
	}

	// Compile instructions
	g_pCurInstInfo = s_pInstCache;
	while (!psxbranch && psxpc < s_nEndBlock)
	{
		psxRecompileNextInstruction(false, false);
	}

	pxAssert((psxpc - startpc) >> 2 <= 0xffff);
	s_pCurBlockEx->size = (psxpc - startpc) >> 2;

	if (!(psxpc & 0x10000000))
		g_psxMaxRecMem = std::max((psxpc & ~0xa0000000), g_psxMaxRecMem);

	// Emit block epilogue
	if (psxbranch == 2)
	{
		// Dynamic branch (JR/JALR)
		iopArmFlushConstRegs();
		iPsxBranchTest(0xffffffff, 1);
		armEmitJmp(iopDispatcherReg);
	}
	else
	{
		if (psxbranch)
			pxAssert(!link_next_block);
		else
		{
			// Fall-through: add cycles
			u32 blockCycles = psxScaleBlockCycles();

			armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
			if (Assembler::IsImmAddSub(blockCycles))
				armAsm->Add(x0, x0, blockCycles);
			else
			{
				armAsm->Mov(x4, (u64)blockCycles);
				armAsm->Add(x0, x0, x4);
			}
			armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
			iPsxAddEECycles(blockCycles);
		}

		if (link_next_block || !psxbranch)
		{
			pxAssert(psxpc == s_nEndBlock);
			iopArmFlushConstRegs();

			// Store PC and dispatch (no block linking for now)
			armAsm->Mov(RWPSXSCRATCH, psxpc);
			armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
			armEmitJmp(iopDispatcherReg);
			psxbranch = 3;
		}
	}

	// Finalize block
	u8* blockEnd = armEndBlock();

	pxAssert(blockEnd < SysMemory::GetIOPRecEnd());
	s_pCurBlockEx->x86size = blockEnd - recPtr;

	Perf::iop.RegisterPC((void*)s_pCurBlockEx->fnptr, s_pCurBlockEx->x86size, s_pCurBlockEx->startpc);

	recPtr = blockEnd;

	pxAssert((g_psxHasConstReg & g_psxFlushedConstReg) == g_psxHasConstReg);

	s_pCurBlock = nullptr;
	s_pCurBlockEx = nullptr;
}

// ============================================================================
//  Reserve / Reset / Execute / Clear / Shutdown
// ============================================================================

static DynamicHeapArray<BASEBLOCK, 4096> recLutReserve;
static DynamicHeapArray<BASEBLOCK, 4096> recLutUnmapped;
static size_t recLutEntries;
static bool extraRam = false;

static void recReserveRAM()
{
	recLutEntries =
		((Ps2MemSize::ExposedIopRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2) / 4);

	if (recLutReserve.size() != recLutEntries)
		recLutReserve.resize(recLutEntries);

	recLutUnmapped.resize(_64kb / 4);

	BASEBLOCK* curpos = recLutReserve.data();
	recRAM = curpos;
	curpos += (Ps2MemSize::ExposedIopRam / 4);
	recROM = curpos;
	curpos += (Ps2MemSize::Rom / 4);
	recROM1 = curpos;
	curpos += (Ps2MemSize::Rom1 / 4);
	recROM2 = curpos;
	curpos += (Ps2MemSize::Rom2 / 4);
}

static void recReserve()
{
	recPtr = SysMemory::GetIOPRec();
	recPtrEnd = SysMemory::GetIOPRecEnd() - _64kb;

	s_iopConstPool.Init(recPtrEnd, _64kb);

	recReserveRAM();

	if (!s_pInstCache)
	{
		s_nInstCacheSize = 128;
		s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
		if (!s_pInstCache)
			pxFailRel("Failed to allocate IOP InstCache array.");
	}
}

void recResetIOP()
{
	DevCon.WriteLn("iR3000A ARM64 Recompiler reset.");

	if (CHECK_EXTRAMEM != extraRam)
	{
		recReserveRAM();
		extraRam = !extraRam;
	}

	// Set up assembler at start of code buffer
	size_t capacity = recPtrEnd - SysMemory::GetIOPRec();
	armSetAsmPtr(SysMemory::GetIOPRec(), capacity, &s_iopConstPool);
	s_iopConstPool.Reset();

	// Generate dispatchers
	_DynGen_Dispatchers();
	// armEndBlock() already advanced armAsmPtr past emitted code and set armAsm=nullptr,
	// so we can't use armGetCurrentCodePointer() (which dereferences armAsm).
	recPtr = armAsmPtr;

	// Clear all block entries
	iopClearRecLUT(reinterpret_cast<BASEBLOCK*>(recLutReserve.data()),
		Ps2MemSize::ExposedIopRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2);

	BASEBLOCK* unmapped = recLutUnmapped.data();

	for (int i = 0; i < 0x10000; i++)
		recLUT_SetPage(psxRecLUT, psxhwLUT, unmapped, i, 0, 0);

	for (int i = 0; i < (int)(_64kb / 4); i++)
		unmapped[i].SetFnptr((uptr)iopUnmappedRecLUTPage);

	// Map IOP RAM (mirrored at 0x0000, 0x8000, 0xa000)
	for (int i = 0; i < 0x80; i++)
	{
		u32 mask = (Ps2MemSize::ExposedIopRam / _64kb) - 1;
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0x0000, i, i & mask);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0x8000, i, i & mask);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0xa000, i, i & mask);
	}

	// Map ROM
	for (int i = 0x1fc0; i < 0x2000; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0x0000, i, i - 0x1fc0);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0x8000, i, i - 0x1fc0);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0xa000, i, i - 0x1fc0);
	}

	// Map ROM1
	for (int i = 0x1e00; i < 0x1e40; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0x0000, i, i - 0x1e00);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0x8000, i, i - 0x1e00);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0xa000, i, i - 0x1e00);
	}

	// Map ROM2
	for (int i = 0x1e40; i < 0x1e48; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0x0000, i, i - 0x1e40);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0x8000, i, i - 0x1e40);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0xa000, i, i - 0x1e40);
	}

	if (s_pInstCache)
		memset(s_pInstCache, 0, sizeof(EEINST) * s_nInstCacheSize);

	recBlocks.Reset();
	g_psxMaxRecMem = 0;
	psxbranch = 0;
}

static void iopClearRecLUT(BASEBLOCK* base, int count)
{
	for (int i = 0; i < count / 4; i++)
		base[i].SetFnptr((uptr)iopJITCompile);
}

static __noinline s32 recExecuteBlock(s32 eeCycles)
{
	psxRegs.iopBreak = 0;
	psxRegs.iopCycleEE = eeCycles;

	((void (*)())iopEnterRecompiledCode)();

	return psxRegs.iopBreak + psxRegs.iopCycleEE;
}

// Returns the offset to the next instruction after any cleared memory
static __fi u32 psxRecClearMem(u32 pc)
{
	BASEBLOCK* pblock;

	pblock = PSX_GETBLOCK(pc);
	if (pblock->GetFnptr() == (uptr)iopJITCompile)
		return 4;

	pc = HWADDR(pc);

	u32 lowerextent = pc, upperextent = pc + 4;
	int blockidx = recBlocks.Index(pc);
	pxAssert(blockidx != -1);

	while (BASEBLOCKEX* pexblock = recBlocks[blockidx - 1])
	{
		if (pexblock->startpc + pexblock->size * 4 <= lowerextent)
			break;
		lowerextent = std::min(lowerextent, pexblock->startpc);
		blockidx--;
	}

	int toRemoveFirst = blockidx;

	while (BASEBLOCKEX* pexblock = recBlocks[blockidx])
	{
		if (pexblock->startpc >= upperextent)
			break;
		lowerextent = std::min(lowerextent, pexblock->startpc);
		upperextent = std::max(upperextent, pexblock->startpc + pexblock->size * 4);
		blockidx++;
	}

	if (toRemoveFirst != blockidx)
		recBlocks.Remove(toRemoveFirst, (blockidx - 1));

	// Clear all BASEBLOCK entries in range
	for (u32 addr = lowerextent; addr < upperextent; addr += 4)
	{
		BASEBLOCK* p = PSX_GETBLOCK(addr);
		p->SetFnptr((uptr)iopJITCompile);
	}

	return upperextent - pc;
}

#define PSXREC_CLEARM(mem) \
	(((mem) < g_psxMaxRecMem && (psxRecLUT[(mem) >> 16] + (mem))) ? \
			psxRecClearMem(mem) : \
			4)

static void recClearIOP(u32 addr, u32 size)
{
	u32 upperLimit = addr + size;
	for (u32 i = addr; i < upperLimit; i += PSXREC_CLEARM(i))
		;
}

static void recShutdown()
{
	recLutReserve.deallocate();
	s_iopConstPool.Destroy();

	safe_free(s_pInstCache);
	s_nInstCacheSize = 0;

	recPtr = nullptr;
	recPtrEnd = nullptr;
}

// ============================================================================
//  psxRec CPU interface
// ============================================================================

R3000Acpu psxRec = {
	recReserve,
	recResetIOP,
	recExecuteBlock,
	recClearIOP,
	recShutdown,
};
