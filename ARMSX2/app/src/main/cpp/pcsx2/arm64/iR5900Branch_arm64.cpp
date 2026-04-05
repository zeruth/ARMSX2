// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Branch & Jump Instructions
// BEQ, BNE, BGEZ, BLTZ, BLEZ, BGTZ, J, JAL, JR, JALR + Likely variants

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#ifdef INTERP_BRANCH
#define ISTUB_BEQ      1
#define ISTUB_BNE      1
#define ISTUB_BEQL     1
#define ISTUB_BNEL     1
#define ISTUB_BGEZ     1
#define ISTUB_BGTZ     1
#define ISTUB_BLEZ     1
#define ISTUB_BLTZ     1
#define ISTUB_BGEZL    1
#define ISTUB_BGTZL    1
#define ISTUB_BLEZL    1
#define ISTUB_BLTZL    1
#define ISTUB_BGEZAL   1
#define ISTUB_BLTZAL   1
#define ISTUB_BGEZALL  1
#define ISTUB_BLTZALL  1
#define ISTUB_J        1
#define ISTUB_JAL      1
#define ISTUB_JR       1
#define ISTUB_JALR     1
#define ISTUB_SYSCALL  1
#define ISTUB_BREAK    1
#else
#define ISTUB_BEQ      0
#define ISTUB_BNE      0
#define ISTUB_BEQL     0
#define ISTUB_BNEL     0
#define ISTUB_BGEZ     0
#define ISTUB_BGTZ     0
#define ISTUB_BLEZ     0
#define ISTUB_BLTZ     0
#define ISTUB_BGEZL    0
#define ISTUB_BGTZL    0
#define ISTUB_BLEZL    0
#define ISTUB_BLTZL    0
#define ISTUB_BGEZAL   0
#define ISTUB_BLTZAL   0
#define ISTUB_BGEZALL  0
#define ISTUB_BLTZALL  0
#define ISTUB_J        0
#define ISTUB_JAL      0
#define ISTUB_JR       0
#define ISTUB_JALR     0
#define ISTUB_SYSCALL  0
#define ISTUB_BREAK    0
#endif

// ============================================================================
//  Native codegen helpers (only compiled when at least one native branch exists)
// ============================================================================

#if !ISTUB_BEQ || !ISTUB_BNE || !ISTUB_BEQL || !ISTUB_BNEL || \
    !ISTUB_BGEZ || !ISTUB_BGTZ || !ISTUB_BLEZ || !ISTUB_BLTZ || \
    !ISTUB_BGEZL || !ISTUB_BGTZL || !ISTUB_BLEZL || !ISTUB_BLTZL || \
    !ISTUB_BGEZAL || !ISTUB_BLTZAL || !ISTUB_BGEZALL || !ISTUB_BLTZALL || \
    !ISTUB_J || !ISTUB_JAL || !ISTUB_JR || !ISTUB_JALR || \
    !ISTUB_SYSCALL || !ISTUB_BREAK

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

// Helper: set PC to a known immediate and mark branch done
static void SetBranchImm(u32 imm)
{
	armAsm->Mov(RWSCRATCH, imm);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	g_branch = 1;
	g_cpuFlushedPC = true;
}

// Helper: for conditional branches with two GPR operands
static void recBranch_GPR64(a64::Condition cond, int rs, int rt, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST2(rs, rt))
	{
		bool taken = false;
		switch (cond)
		{
			case a64::eq: taken = (g_cpuConstRegs[rs].SD[0] == g_cpuConstRegs[rt].SD[0]); break;
			case a64::ne: taken = (g_cpuConstRegs[rs].SD[0] != g_cpuConstRegs[rt].SD[0]); break;
			default: break;
		}
		recompileNextInstruction(true, false);
		SetBranchImm(taken ? branchTarget : fallthrough);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);
	if (rt == 0)
		armAsm->Cmp(RSCRATCHGPR, 0);
	else
	{
		armLoadGPR64(RSCRATCHGPR2, rt);
		armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	}

	armAsm->Cset(RDELAYSLOTGPR, cond);

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

// Helper: for conditional branches comparing one GPR against zero
static void recBranch_GPR64_vs_Zero(a64::Condition cond, int rs, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST1(rs))
	{
		bool taken = false;
		s64 val = g_cpuConstRegs[rs].SD[0];
		switch (cond)
		{
			case a64::ge: taken = (val >= 0); break;
			case a64::lt: taken = (val < 0); break;
			case a64::le: taken = (val <= 0); break;
			case a64::gt: taken = (val > 0); break;
			default: break;
		}
		recompileNextInstruction(true, false);
		SetBranchImm(taken ? branchTarget : fallthrough);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);
	armAsm->Cmp(RSCRATCHGPR, 0);
	armAsm->Cset(RDELAYSLOTGPR, cond);

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

// Helper: likely branch — if NOT taken, skip the delay slot entirely
static void recBranch_GPR64_Likely(a64::Condition cond, int rs, int rt, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST2(rs, rt))
	{
		bool taken = false;
		switch (cond)
		{
			case a64::eq: taken = (g_cpuConstRegs[rs].SD[0] == g_cpuConstRegs[rt].SD[0]); break;
			case a64::ne: taken = (g_cpuConstRegs[rs].SD[0] != g_cpuConstRegs[rt].SD[0]); break;
			default: break;
		}
		if (taken)
		{
			recompileNextInstruction(true, false);
			SetBranchImm(branchTarget);
		}
		else
		{
			SetBranchImm(fallthrough);
		}
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);
	if (rt == 0)
		armAsm->Cmp(RSCRATCHGPR, 0);
	else
	{
		armLoadGPR64(RSCRATCHGPR2, rt);
		armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	}

	// If condition is NOT met, skip the delay slot and go to fallthrough
	a64::Label skipDelaySlot, done;
	armAsm->B(&skipDelaySlot, a64::InvertCondition(cond));

	// Condition met: execute delay slot, then branch to target
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

static void recBranch_GPR64_vs_Zero_Likely(a64::Condition cond, int rs, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST1(rs))
	{
		bool taken = false;
		s64 val = g_cpuConstRegs[rs].SD[0];
		switch (cond)
		{
			case a64::ge: taken = (val >= 0); break;
			case a64::lt: taken = (val < 0); break;
			case a64::le: taken = (val <= 0); break;
			case a64::gt: taken = (val > 0); break;
			default: break;
		}
		if (taken)
		{
			recompileNextInstruction(true, false);
			SetBranchImm(branchTarget);
		}
		else
		{
			SetBranchImm(fallthrough);
		}
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);

	// For lt/ge, use Tbz/Tbnz to test sign bit directly (1 instruction vs 2)
	a64::Label skipDelaySlot, done;
	if (cond == a64::lt)
		armAsm->Tbz(RSCRATCHGPR, 63, &skipDelaySlot);
	else if (cond == a64::ge)
		armAsm->Tbnz(RSCRATCHGPR, 63, &skipDelaySlot);
	else
	{
		armAsm->Cmp(RSCRATCHGPR, 0);
		armAsm->B(&skipDelaySlot, a64::InvertCondition(cond));
	}

	// Condition met: execute delay slot, then branch to target
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

#endif // at least one native branch

// ============================================================================
//  Instruction implementations
// ============================================================================

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ---- BEQ ----
#if ISTUB_BEQ
void recBEQ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BEQ); }
#else
void recBEQ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64(a64::eq, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BNE ----
#if ISTUB_BNE
void recBNE() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BNE); }
#else
void recBNE()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		// rs != rs is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64(a64::ne, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BEQL ----
#if ISTUB_BEQL
void recBEQL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BEQL); }
#else
void recBEQL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_Likely(a64::eq, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BNEL ----
#if ISTUB_BNEL
void recBNEL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BNEL); }
#else
void recBNEL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		// Likely: not taken → skip delay slot
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_Likely(a64::ne, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BGEZ ----
#if ISTUB_BGEZ
void recBGEZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZ); }
#else
void recBGEZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 >= 0 is always true
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGTZ ----
#if ISTUB_BGTZ
void recBGTZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGTZ); }
#else
void recBGTZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 > 0 is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::gt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLEZ ----
#if ISTUB_BLEZ
void recBLEZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLEZ); }
#else
void recBLEZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 <= 0 is always true
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::le, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZ ----
#if ISTUB_BLTZ
void recBLTZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZ); }
#else
void recBLTZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 < 0 is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZL ----
#if ISTUB_BGEZL
void recBGEZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZL); }
#else
void recBGEZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGTZL ----
#if ISTUB_BGTZL
void recBGTZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGTZL); }
#else
void recBGTZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::gt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLEZL ----
#if ISTUB_BLEZL
void recBLEZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLEZL); }
#else
void recBLEZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::le, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZL ----
#if ISTUB_BLTZL
void recBLTZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZL); }
#else
void recBLTZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZAL ----
#if ISTUB_BGEZAL
void recBGEZAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZAL); }
#else
void recBGEZAL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZAL ----
#if ISTUB_BLTZAL
void recBLTZAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZAL); }
#else
void recBLTZAL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZALL ----
#if ISTUB_BGEZALL
void recBGEZALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZALL); }
#else
void recBGEZALL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZALL ----
#if ISTUB_BLTZALL
void recBLTZALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZALL); }
#else
void recBLTZALL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- J ----
#if ISTUB_J
void recJ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::J); }
#else
void recJ()
{
	u32 target = (_Target_ << 2) | ((pc + 4) & 0xf0000000);
	recompileNextInstruction(true, false);
	SetBranchImm(target);
}
#endif

// ---- JAL ----
#if ISTUB_JAL
void recJAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JAL); }
#else
void recJAL()
{
	u32 target = (_Target_ << 2) | ((pc + 4) & 0xf0000000);

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	recompileNextInstruction(true, false);
	SetBranchImm(target);
}
#endif

// ---- JR ----
#if ISTUB_JR
void recJR() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JR); }
#else
void recJR()
{
	armLoadGPR64(RDELAYSLOTGPR, _Rs_);

	armFlushConstRegs();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Str(RWDELAYSLOT, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}
#endif

// ---- JALR ----
#if ISTUB_JALR
void recJALR() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JALR); }
#else
void recJALR()
{
	int rd = _Rd_ ? _Rd_ : 31;

	armLoadGPR64(RDELAYSLOTGPR, _Rs_);

	g_cpuConstRegs[rd].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(rd);

	armFlushConstRegs();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Str(RWDELAYSLOT, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}
#endif

// ---- SYSCALL ----
#if ISTUB_SYSCALL
void recSYSCALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::SYSCALL); }
#else
void recSYSCALL()
{
	// Interpreter SYSCALL does cpuRegs.pc -= 4 internally, so flush pc as-is
	// (pc = SYSCALL_addr + 4, interpreter subtracts to get SYSCALL_addr)
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)R5900::Interpreter::OpcodeImpl::SYSCALL);

	// Interpreter may modify any GPR/PC — clear all const tracking
	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
	g_cpuFlushedPC = true;
}
#endif

// ---- BREAK ----
#if ISTUB_BREAK
void recBREAK() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BREAK); }
#else
void recBREAK()
{
	// Interpreter BREAK does cpuRegs.pc -= 4 internally, so flush pc as-is
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)R5900::Interpreter::OpcodeImpl::BREAK);

	// Interpreter may modify any GPR/PC — clear all const tracking
	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
	g_cpuFlushedPC = true;
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
