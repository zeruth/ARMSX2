// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP0 Instructions
// MFC0, MTC0, BC0x, TLB*, ERET, EI, DI

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "COP0.h"
#include "Dmac.h"

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

using namespace R5900;

// CP0 register offsets from RCPUSTATE (x19 = &cpuRegs)
static constexpr s64 CP0_OFFSET(int reg) { return offsetof(cpuRegisters, CP0) + reg * sizeof(u32); }
static constexpr s64 PERF_OFFSET = offsetof(cpuRegisters, PERF);
static constexpr s64 LAST_COP0_CYCLE_OFFSET = offsetof(cpuRegisters, lastCOP0Cycle);
static constexpr s64 LAST_PERF_CYCLE_OFFSET(int n) { return offsetof(cpuRegisters, lastPERFCycle) + n * sizeof(u64); }

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#ifdef INTERP_COP0
#define ISTUB_MFC0     1
#define ISTUB_MTC0     1
#define ISTUB_BC0F     1
#define ISTUB_BC0T     1
#define ISTUB_BC0FL    1
#define ISTUB_BC0TL    1
#define ISTUB_TLBR     1
#define ISTUB_TLBWI    1
#define ISTUB_TLBWR    1
#define ISTUB_TLBP     1
#define ISTUB_ERET     1
#define ISTUB_EI       1
#define ISTUB_DI       1
#else
#define ISTUB_MFC0     0
#define ISTUB_MTC0     0
#define ISTUB_BC0F     0
#define ISTUB_BC0T     0
#define ISTUB_BC0FL    0
#define ISTUB_BC0TL    0
#define ISTUB_TLBR     1   // TLB ops are complex — keep as interp
#define ISTUB_TLBWI    1
#define ISTUB_TLBWR    1
#define ISTUB_TLBP     1
#define ISTUB_ERET     1   // ERET modifies PC/status — keep as interp
#define ISTUB_EI       1   // EI/DI have side effects — keep as interp
#define ISTUB_DI       1
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP0 {

// ============================================================================
//  MFC0 — Move from COP0 register
//  Reads CP0 register _Rd_ into GPR[_Rt_] (sign-extended to 64 bits)
//  Special cases: Count (reg 9), Status (reg 12), PERF (reg 25)
// ============================================================================

#if ISTUB_MFC0
void recMFC0() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::MFC0); }
#else
void recMFC0()
{
	if (_Rd_ == 9)
	{
		// Count register: must update even if _Rt_ == 0
		// Count += cpuRegs.cycle - cpuRegs.lastCOP0Cycle (min 1)
		armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
		armAsm->Ldr(RSCRATCHGPR2, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));
		armAsm->Sub(RSCRATCHGPR3, RSCRATCHGPR, RSCRATCHGPR2);
		// Ensure increment is at least 1
		armAsm->Cmp(RSCRATCHGPR3, 0);
		armAsm->Csinc(RSCRATCHGPR3, RSCRATCHGPR3, a64::xzr, a64::ne);
		// Update Count (32-bit add, using low 32 bits of increment)
		armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
		armAsm->Add(RWSCRATCH2, RWSCRATCH2, RWSCRATCH3);
		armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
		// lastCOP0Cycle = cycle
		armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));

		if (!_Rt_)
			return;

		GPR_DEL_CONST(_Rt_);
		armStoreGPR64SignExt32(RWSCRATCH2, _Rt_);
		return;
	}

	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);

	switch (_Rd_)
	{
		case 12: // Status — mask off reserved bits
		{
			armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(12)));
			armAsm->Mov(RWSCRATCH2, 0xf0c79c1f);
			armAsm->And(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
			armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			break;
		}

		case 25: // PERF registers
		{
			if (0 == (_Imm_ & 1)) // MFPS — read PCCR
			{
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pccr)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			else if (0 == (_Imm_ & 2)) // MFPC 0 — read PCR0
			{
				// Call COP0_UpdatePCCR to get up-to-date counter value
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)COP0_UpdatePCCR);
				armAsm->Blr(RSCRATCHGPR);
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr0)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			else // MFPC 1 — read PCR1
			{
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)COP0_UpdatePCCR);
				armAsm->Blr(RSCRATCHGPR);
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr1)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			break;
		}

		case 24: // Debug breakpoint registers — no-op
			break;

		default:
			armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(_Rd_)));
			armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			break;
	}
}
#endif

// ============================================================================
//  MTC0 — Move to COP0 register
//  Writes GPR[_Rt_] into CP0 register _Rd_
//  Special cases: Count (9), Status (12), Config (16), PERF (25)
// ============================================================================

#if ISTUB_MTC0
void recMTC0() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::MTC0); }
#else
void recMTC0()
{
	switch (_Rd_)
	{
		case 9: // Count
		{
			// lastCOP0Cycle = cycle
			armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
			armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));
			// CP0.r[9] = GPR[rt].UL[0]
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
			break;
		}

		case 12: // Status — call WriteCP0Status
		{
			armLoadGPR32(a64::w0, _Rt_);
			armFlushConstRegs();
			armFlushPC();
			armFlushCode();
			armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)WriteCP0Status);
			armAsm->Blr(RSCRATCHGPR);
			break;
		}

		case 16: // Config — call WriteCP0Config
		{
			armLoadGPR32(a64::w0, _Rt_);
			armFlushConstRegs();
			armFlushPC();
			armFlushCode();
			armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)WriteCP0Config);
			armAsm->Blr(RSCRATCHGPR);
			break;
		}

		case 25: // PERF registers
		{
			if (0 == (_Imm_ & 1)) // MTPS
			{
				if (0 != (_Imm_ & 0x3E)) // only effective when register field is 0
					break;
				// COP0_UpdatePCCR(); pccr = GPR[rt].UL[0]; COP0_DiagnosticPCCR();
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)COP0_UpdatePCCR);
				armAsm->Blr(RSCRATCHGPR);
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pccr)));
				armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)COP0_DiagnosticPCCR);
				armAsm->Blr(RSCRATCHGPR);
			}
			else if (0 == (_Imm_ & 2)) // MTPC 0
			{
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr0)));
				armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
				armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_PERF_CYCLE_OFFSET(0)));
			}
			else // MTPC 1
			{
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr1)));
				armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
				armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_PERF_CYCLE_OFFSET(1)));
			}
			break;
		}

		case 24: // Debug breakpoint registers — no-op
			break;

		default:
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(_Rd_)));
			break;
	}
}
#endif

// ============================================================================
//  BC0F / BC0T / BC0FL / BC0TL — COP0 branch instructions
//  Condition: (((dmacRegs.stat.CIS | ~dmacRegs.pcr.CPC) & 0x3FF) == 0x3FF)
//  BC0T branches when true, BC0F when false.
// ============================================================================

// Helper: evaluate CPCOND0 and set condition result in RDELAYSLOTGPR.
// branchIfTrue: BC0T/BC0TL branch when condition is true.
static void recBC0_setup(bool branchIfTrue)
{
	// CPCOND0 = ((CIS | ~CPC) & 0x3FF) == 0x3FF
	//         ≡ (CPC & ~CIS & 0x3FF) == 0       (De Morgan)
	// Load stat first — the pcr load into w4 clobbers the base in x4.
	armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)&dmacRegs);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR, offsetof(DMACregisters, stat)));
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR, offsetof(DMACregisters, pcr)));
	// CPC & ~CIS (bit-clear: clears bits in pcr that are set in stat)
	armAsm->Bic(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	// Test lower 10 bits — sets Z if all cleared (CPCOND0 true)
	armAsm->Tst(RWSCRATCH, 0x3FF);
	// Cset: 1 = taken.
	// BC0T: taken when eq (condition true). BC0F: taken when ne (condition false).
	armAsm->Cset(RDELAYSLOTGPR, branchIfTrue ? a64::eq : a64::ne);
}

// Non-likely BC0 branch
static void recBC0_helper(bool branchIfTrue)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	recBC0_setup(branchIfTrue);

	armFlushConstRegs();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Mov(RWSCRATCH2, fallthrough);
	armAsm->Cmp(RDELAYSLOTGPR, 0);
	armAsm->Csel(RWSCRATCH, RWSCRATCH, RWSCRATCH2, a64::ne);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}

// Likely BC0 branch (skip delay slot if not taken)
static void recBC0_Likely_helper(bool branchIfTrue)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	recBC0_setup(branchIfTrue);

	a64::Label skipDelaySlot, done;
	// If NOT taken, skip delay slot
	armAsm->Cbz(RDELAYSLOTGPR, &skipDelaySlot);

	// Taken: execute delay slot, branch to target
	armFlushConstRegs();
	recompileNextInstruction(true, false);
	armFlushConstRegs();
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armAsm->B(&done);

	// Not taken: skip delay slot, PC = fallthrough
	armAsm->Bind(&skipDelaySlot);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	armAsm->Bind(&done);
	g_branch = 1;
	g_cpuFlushedPC = true;
}

#if ISTUB_BC0F
void recBC0F()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::BC0F);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC0F()  { recBC0_helper(false); }
#endif

#if ISTUB_BC0T
void recBC0T()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::BC0T);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC0T()  { recBC0_helper(true); }
#endif

#if ISTUB_BC0FL
void recBC0FL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::BC0FL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC0FL() { recBC0_Likely_helper(false); }
#endif

#if ISTUB_BC0TL
void recBC0TL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::BC0TL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC0TL() { recBC0_Likely_helper(true); }
#endif

// ============================================================================
//  TLB instructions — complex, always use interpreter
// ============================================================================

#if ISTUB_TLBR
void recTLBR()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBR); }
#else
void recTLBR()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBR); }
#endif

#if ISTUB_TLBWI
void recTLBWI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWI); }
#else
void recTLBWI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWI); }
#endif

#if ISTUB_TLBWR
void recTLBWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWR); }
#else
void recTLBWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWR); }
#endif

#if ISTUB_TLBP
void recTLBP()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBP); }
#else
void recTLBP()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBP); }
#endif

// ============================================================================
//  ERET — Exception Return
//  Modifies PC and Status, must end the block.
// ============================================================================

#if ISTUB_ERET
void recERET() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::ERET); }
#else
void recERET() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::ERET); }
#endif

// ============================================================================
//  EI / DI — Enable / Disable Interrupts
//  EI enables interrupts and must force an event check (upstream uses recBranchCall).
// ============================================================================

#if ISTUB_EI
void recEI() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::EI); }
#else
void recEI() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::EI); }
#endif

#if ISTUB_DI
void recDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::DI); }
#else
void recDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::DI); }
#endif

} // namespace COP0
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
