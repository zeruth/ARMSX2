// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — ALU Instructions
// ADD/ADDU/SUB/SUBU, ADDI/ADDIU, DADD/DADDU/DSUB/DSUBU, DADDI/DADDIU,
// AND/OR/XOR/NOR, ANDI/ORI/XORI, SLT/SLTU, SLTI/SLTIU

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_ALU) || defined(INTERP_EE)
#define ISTUB_ADD      1
#define ISTUB_ADDU     1
#define ISTUB_SUB      1
#define ISTUB_SUBU     1
#define ISTUB_ADDI     1
#define ISTUB_ADDIU    1
#define ISTUB_DADD     1
#define ISTUB_DADDU    1
#define ISTUB_DSUB     1
#define ISTUB_DSUBU    1
#define ISTUB_DADDI    1
#define ISTUB_DADDIU   1
#define ISTUB_AND      1
#define ISTUB_OR       1
#define ISTUB_XOR      1
#define ISTUB_NOR      1
#define ISTUB_ANDI     1
#define ISTUB_ORI      1
#define ISTUB_XORI     1
#define ISTUB_SLT      1
#define ISTUB_SLTU     1
#define ISTUB_SLTI     1
#define ISTUB_SLTIU    1
#else
#define ISTUB_ADD      1  // overflow trap — keep interp
#define ISTUB_ADDU     0
#define ISTUB_SUB      1  // overflow trap — keep interp
#define ISTUB_SUBU     0
#define ISTUB_ADDI     1  // overflow trap — keep interp
#define ISTUB_ADDIU    0
#define ISTUB_DADD     1  // overflow trap — keep interp
#define ISTUB_DADDU    0
#define ISTUB_DSUB     1  // overflow trap — keep interp
#define ISTUB_DSUBU    0
#define ISTUB_DADDI    1  // overflow trap — keep interp
#define ISTUB_DADDIU   0
#define ISTUB_AND      0
#define ISTUB_OR       0
#define ISTUB_XOR      0
#define ISTUB_NOR      0
#define ISTUB_ANDI     0
#define ISTUB_ORI      0
#define ISTUB_XORI     0
#define ISTUB_SLT      0
#define ISTUB_SLTU     0
#define ISTUB_SLTI     0
#define ISTUB_SLTIU    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  ADDU — rd = sign_extend_32(rs + rt)  [no overflow trap]
//  Interpreter: GPR[rd].UD[0] = u64(s64(s32(GPR[rs].UL[0] + GPR[rt].UL[0])))
// ============================================================================

#if ISTUB_ADDU
void recADDU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADDU); }
#else
void recADDU()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR32(RWSCRATCH, _Rs_);
		armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
		return;
	}
	if (!_Rs_)
	{
		armLoadGPR32(RWSCRATCH, _Rt_);
		armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
		return;
	}
	if (_Rs_ == _Rt_)
	{
		armLoadGPR32(RWSCRATCH, _Rs_);
		armAsm->Add(RWSCRATCH, RWSCRATCH, RWSCRATCH);
		armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
		return;
	}

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Add(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
}
#endif

// ============================================================================
//  SUBU — rd = sign_extend_32(rs - rt)  [no overflow trap]
//  Interpreter: GPR[rd].UD[0] = u64(s64(s32(GPR[rs].UL[0] - GPR[rt].UL[0])))
// ============================================================================

#if ISTUB_SUBU
void recSUBU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SUBU); }
#else
void recSUBU()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR32(RWSCRATCH, _Rs_);
		armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
		return;
	}
	if (!_Rs_)
	{
		armLoadGPR32(RWSCRATCH, _Rt_);
		armAsm->Neg(RWSCRATCH, RWSCRATCH);
		armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
		return;
	}

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Sub(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	armStoreGPR64SignExt32(RWSCRATCH, _Rd_);
}
#endif

// ============================================================================
//  ADDIU — rt = sign_extend_32(rs + sign_extend(imm16))  [no overflow trap]
//  Interpreter: GPR[rt].UD[0] = u64(s64(s32(GPR[rs].UL[0] + u32(s32(_Imm_)))))
// ============================================================================

#if ISTUB_ADDIU
void recADDIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADDIU); }
#else
void recADDIU()
{
	if (!_Rt_)
		return;

	// Const propagation disabled — causes downstream corruption (see SLL comment)
	GPR_DEL_CONST(_Rt_);

	const s32 imm = _Imm_;
	if (imm == 0)
	{
		armLoadGPR32(RWSCRATCH, _Rs_);
		armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
		return;
	}
	if (!_Rs_)
	{
		armAsm->Mov(RSCRATCHGPR, static_cast<u64>(static_cast<s64>(imm)));
		armStoreGPR64(RSCRATCHGPR, _Rt_);
		return;
	}

	armLoadGPR32(RWSCRATCH, _Rs_);
	if (imm > 0)
		armAsm->Add(RWSCRATCH, RWSCRATCH, imm);
	else
		armAsm->Sub(RWSCRATCH, RWSCRATCH, -imm);
	armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
}
#endif

// ============================================================================
//  DADDU — rd = rs + rt  (64-bit, no overflow trap)
//  Interpreter: GPR[rd].UD[0] = GPR[rs].UD[0] + GPR[rt].UD[0]
// ============================================================================

#if ISTUB_DADDU
void recDADDU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADDU); }
#else
void recDADDU()
{
	if (!_Rd_)
		return;

	// Const propagation disabled — causes downstream corruption (see SLL comment)
	GPR_DEL_CONST(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (!_Rs_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rt_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (_Rs_ == _Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armAsm->Add(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Add(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  DSUBU — rd = rs - rt  (64-bit, no overflow trap)
//  Interpreter: GPR[rd].UD[0] = GPR[rs].UD[0] - GPR[rt].UD[0]
// ============================================================================

#if ISTUB_DSUBU
void recDSUBU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSUBU); }
#else
void recDSUBU()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (!_Rs_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rt_);
		armAsm->Neg(RSCRATCHGPR, RSCRATCHGPR);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Sub(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  DADDIU — rt = rs + sign_extend(imm16)  (64-bit, no overflow trap)
//  Interpreter: GPR[rt].UD[0] = GPR[rs].UD[0] + u64(s64(_Imm_))
// ============================================================================

#if ISTUB_DADDIU
void recDADDIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADDIU); }
#else
void recDADDIU()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);

	const s32 imm = _Imm_;
	if (imm == 0)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
		return;
	}
	if (!_Rs_)
	{
		armAsm->Mov(RSCRATCHGPR, static_cast<u64>(static_cast<s64>(imm)));
		armStoreGPR64(RSCRATCHGPR, _Rt_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	if (imm > 0)
		armAsm->Add(RSCRATCHGPR, RSCRATCHGPR, imm);
	else
		armAsm->Sub(RSCRATCHGPR, RSCRATCHGPR, -imm);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

// ============================================================================
//  AND / OR / XOR / NOR — 64-bit logical, rd = rs OP rt
// ============================================================================

#if ISTUB_AND
void recAND() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::AND); }
#else
void recAND()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (!_Rs_ || !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (_Rs_ == _Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->And(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_OR
void recOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::OR); }
#else
void recOR()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_ || _Rs_ == _Rt_)
	{
		// 0 | rt = rt, or rs | rs = rs (MIPS "move" idiom)
		armLoadGPR64(RSCRATCHGPR, _Rt_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Orr(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_XOR
void recXOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::XOR); }
#else
void recXOR()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rt_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (!_Rt_)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Eor(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_NOR
void recNOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::NOR); }
#else
void recNOR()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		// ~(0 | 0) = all ones
		armAsm->Mov(RSCRATCHGPR, ~static_cast<u64>(0));
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (_Rs_ == _Rt_ || !_Rt_)
	{
		// ~(rs | rs) = ~rs, or ~(rs | 0) = ~rs
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armAsm->Mvn(RSCRATCHGPR, RSCRATCHGPR);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}
	if (!_Rs_)
	{
		// ~(0 | rt) = ~rt
		armLoadGPR64(RSCRATCHGPR, _Rt_);
		armAsm->Mvn(RSCRATCHGPR, RSCRATCHGPR);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Orr(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armAsm->Mvn(RSCRATCHGPR, RSCRATCHGPR);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  ANDI / ORI / XORI — 64-bit logical with zero-extended immediate
//  Interpreter: GPR[rt].UD[0] = GPR[rs].UD[0] OP (u64)_ImmU_
// ============================================================================

#if ISTUB_ANDI
void recANDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ANDI); }
#else
void recANDI()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);

	if (_ImmU_ == 0)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->And(RSCRATCHGPR, RSCRATCHGPR, (u64)_ImmU_);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

#if ISTUB_ORI
void recORI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ORI); }
#else
void recORI()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	if (_ImmU_ == 0)
	{
		// ORI with 0 is a move
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Orr(RSCRATCHGPR, RSCRATCHGPR, (u64)_ImmU_);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

#if ISTUB_XORI
void recXORI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::XORI); }
#else
void recXORI()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	if (_ImmU_ == 0)
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Eor(RSCRATCHGPR, RSCRATCHGPR, (u64)_ImmU_);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

// ============================================================================
//  SLT / SLTU — Set on Less Than (register)
//  SLT:  GPR[rd] = (GPR[rs].SD[0] < GPR[rt].SD[0]) ? 1 : 0   (signed)
//  SLTU: GPR[rd] = (GPR[rs].UD[0] < GPR[rt].UD[0]) ? 1 : 0   (unsigned)
// ============================================================================

#if ISTUB_SLT
void recSLT() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLT); }
#else
void recSLT()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (_Rs_ == _Rt_)
	{
		// x < x is always false
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	armAsm->Cset(RSCRATCHGPR, a64::lt);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_SLTU
void recSLTU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTU); }
#else
void recSLTU()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armLoadGPR64(RSCRATCHGPR2, _Rt_);
	armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	armAsm->Cset(RSCRATCHGPR, a64::lo);
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  SLTI / SLTIU — Set on Less Than Immediate
//  SLTI:  GPR[rt] = (GPR[rs].SD[0] < sign_extend(imm16)) ? 1 : 0  (signed)
//  SLTIU: GPR[rt] = (GPR[rs].UD[0] < sign_extend(imm16)) ? 1 : 0  (unsigned)
// ============================================================================

#if ISTUB_SLTI
void recSLTI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTI); }
#else
void recSLTI()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	// VIXL MacroAssembler handles Cmp with immediate optimally:
	// small positive → cmp, small negative → cmn, else → mov tmp + cmp
	armAsm->Cmp(RSCRATCHGPR, static_cast<s64>(_Imm_));
	armAsm->Cset(RSCRATCHGPR, a64::lt);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

#if ISTUB_SLTIU
void recSLTIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTIU); }
#else
void recSLTIU()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Cmp(RSCRATCHGPR, static_cast<s64>(_Imm_));
	armAsm->Cset(RSCRATCHGPR, a64::lo);
	armStoreGPR64(RSCRATCHGPR, _Rt_);
}
#endif

// ============================================================================
//  Overflow-trapping ops — kept as interp stubs
//  ADD, ADDI, DADD, DADDI, SUB, DSUB can raise cpuException on overflow
// ============================================================================

#if ISTUB_ADD
void recADD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADD); }
#else
void recADD() {}
#endif

#if ISTUB_ADDI
void recADDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADDI); }
#else
void recADDI() {}
#endif

#if ISTUB_SUB
void recSUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SUB); }
#else
void recSUB() {}
#endif

#if ISTUB_DADD
void recDADD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADD); }
#else
void recDADD() {}
#endif

#if ISTUB_DADDI
void recDADDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADDI); }
#else
void recDADDI() {}
#endif

#if ISTUB_DSUB
void recDSUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSUB); }
#else
void recDSUB() {}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
