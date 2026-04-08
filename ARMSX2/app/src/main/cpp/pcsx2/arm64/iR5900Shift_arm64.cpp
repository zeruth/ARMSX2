// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Shift Instructions
// SLL/SRL/SRA, SLLV/SRLV/SRAV, DSLL/DSRL/DSRA, DSLL32/DSRL32/DSRA32,
// DSLLV/DSRLV/DSRAV

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_SHIFT) || defined(INTERP_EE)
#define ISTUB_SLL      1
#define ISTUB_SRL      1
#define ISTUB_SRA      1
#define ISTUB_SLLV     1
#define ISTUB_SRLV     1
#define ISTUB_SRAV     1
#define ISTUB_DSLL     1
#define ISTUB_DSRL     1
#define ISTUB_DSRA     1
#define ISTUB_DSLL32   1
#define ISTUB_DSRL32   1
#define ISTUB_DSRA32   1
#define ISTUB_DSLLV    1
#define ISTUB_DSRLV    1
#define ISTUB_DSRAV    1
#else
#define ISTUB_SLL      0
#define ISTUB_SRL      0
#define ISTUB_SRA      0
#define ISTUB_SLLV     0
#define ISTUB_SRLV     0
#define ISTUB_SRAV     0
#define ISTUB_DSLL     0
#define ISTUB_DSRL     0
#define ISTUB_DSRA     0
#define ISTUB_DSLL32   0
#define ISTUB_DSRL32   0
#define ISTUB_DSRA32   0
#define ISTUB_DSLLV    0
#define ISTUB_DSRLV    0
#define ISTUB_DSRAV    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  SLL — Shift Left Logical (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] << sa)
// ============================================================================

#if ISTUB_SLL
void recSLL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLL); }
#else
void recSLL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] << _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsl(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SRL — Shift Right Logical (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] >> sa)
// ============================================================================

#if ISTUB_SRL
void recSRL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRL); }
#else
void recSRL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] >> _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsr(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SRA — Shift Right Arithmetic (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].SL[0] >> sa)
// ============================================================================

#if ISTUB_SRA
void recSRA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRA); }
#else
void recSRA()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)((s32)g_cpuConstRegs[_Rt_].UL[0] >> _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Asr(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SLLV — Shift Left Logical Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] << (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SLLV
void recSLLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLLV); }
#else
void recSLLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] << (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  SRLV — Shift Right Logical Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] >> (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SRLV
void recSRLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRLV); }
#else
void recSRLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  SRAV — Shift Right Arithmetic Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].SL[0] >> (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SRAV
void recSRAV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRAV); }
#else
void recSRAV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)((s32)g_cpuConstRegs[_Rt_].UL[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  DSLL — Doubleword Shift Left Logical
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << sa
// ============================================================================

#if ISTUB_DSLL
void recDSLL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLL); }
#else
void recDSLL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] << _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsl(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSRL — Doubleword Shift Right Logical
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> sa
// ============================================================================

#if ISTUB_DSRL
void recDSRL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRL); }
#else
void recDSRL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] >> _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsr(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSRA — Doubleword Shift Right Arithmetic
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> sa
// ============================================================================

#if ISTUB_DSRA
void recDSRA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRA); }
#else
void recDSRA()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = g_cpuConstRegs[_Rt_].SD[0] >> _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Asr(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSLL32 — Doubleword Shift Left Logical +32
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << (sa + 32)
// ============================================================================

#if ISTUB_DSLL32
void recDSLL32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLL32); }
#else
void recDSLL32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] << (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSRL32 — Doubleword Shift Right Logical +32
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> (sa + 32)
// ============================================================================

#if ISTUB_DSRL32
void recDSRL32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRL32); }
#else
void recDSRL32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] >> (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSRA32 — Doubleword Shift Right Arithmetic +32
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> (sa + 32)
// ============================================================================

#if ISTUB_DSRA32
void recDSRA32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRA32); }
#else
void recDSRA32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = g_cpuConstRegs[_Rt_].SD[0] >> (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSLLV — Doubleword Shift Left Logical Variable
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSLLV
void recDSLLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLLV); }
#else
void recDSLLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rt_].UD[0] << (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd, rt, rs);
}
#endif

// ============================================================================
//  DSRLV — Doubleword Shift Right Logical Variable
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSRLV
void recDSRLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRLV); }
#else
void recDSRLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rt_].UD[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd, rt, rs);
}
#endif

// ============================================================================
//  DSRAV — Doubleword Shift Right Arithmetic Variable
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSRAV
void recDSRAV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRAV); }
#else
void recDSRAV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			g_cpuConstRegs[_Rt_].SD[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd, rt, rs);
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
