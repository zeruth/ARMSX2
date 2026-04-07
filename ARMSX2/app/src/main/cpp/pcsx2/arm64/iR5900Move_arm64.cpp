// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Move Instructions
// LUI, MFHI/MFLO, MTHI/MTLO, MFHI1/MFLO1, MTHI1/MTLO1,
// MOVZ, MOVN, MFSA, MTSA, MTSAB, MTSAH

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_MOVE) || defined(INTERP_EE)
#define ISTUB_LUI      1
#define ISTUB_MFHI     1
#define ISTUB_MFLO     1
#define ISTUB_MTHI     1
#define ISTUB_MTLO     1
#define ISTUB_MFHI1    1
#define ISTUB_MFLO1    1
#define ISTUB_MTHI1    1
#define ISTUB_MTLO1    1
#define ISTUB_MOVZ     1
#define ISTUB_MOVN     1
#define ISTUB_MFSA     1
#define ISTUB_MTSA     1
#define ISTUB_MTSAB    1
#define ISTUB_MTSAH    1
#else
#define ISTUB_LUI      0
#define ISTUB_MFHI     0
#define ISTUB_MFLO     0
#define ISTUB_MTHI     0
#define ISTUB_MTLO     0
#define ISTUB_MFHI1    0
#define ISTUB_MFLO1    0
#define ISTUB_MTHI1    0
#define ISTUB_MTLO1    0
#define ISTUB_MOVZ     0
#define ISTUB_MOVN     0
#define ISTUB_MFSA     0
#define ISTUB_MTSA     0
#define ISTUB_MTSAB    0
#define ISTUB_MTSAH    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  LUI — Load Upper Immediate
//  Interpreter: GPR[rt].UD[0] = (s32)(code << 16)
// ============================================================================

#if ISTUB_LUI
void recLUI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LUI); }
#else
void recLUI()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	s64 val = (s64)(s32)(cpuRegs.code << 16);
	armAsm->Mov(RSCRATCHGPR, static_cast<u64>(val));
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
}
#endif

// ============================================================================
//  MFHI / MFLO — Move from HI/LO register
//  Interpreter: GPR[rd].UD[0] = HI.UD[0]  /  GPR[rd].UD[0] = LO.UD[0]
// ============================================================================

#if ISTUB_MFHI
void recMFHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFHI); }
#else
void recMFHI()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_MFLO
void recMFLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFLO); }
#else
void recMFLO()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  MTHI / MTLO — Move to HI/LO register
//  Interpreter: HI.UD[0] = GPR[rs].UD[0]  /  LO.UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTHI
void recMTHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTHI); }
#else
void recMTHI()
{
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

#if ISTUB_MTLO
void recMTLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTLO); }
#else
void recMTLO()
{
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LO_OFFSET));
}
#endif

// ============================================================================
//  MFHI1 / MFLO1 — Move from HI/LO pipeline 1 (upper 64 bits)
//  Interpreter: GPR[rd].UD[0] = HI.UD[1]  /  GPR[rd].UD[0] = LO.UD[1]
// ============================================================================

#if ISTUB_MFHI1
void recMFHI1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFHI1); }
#else
void recMFHI1()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + 8));
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

#if ISTUB_MFLO1
void recMFLO1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFLO1); }
#else
void recMFLO1()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LO_OFFSET + 8));
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  MTHI1 / MTLO1 — Move to HI/LO pipeline 1 (upper 64 bits)
//  Interpreter: HI.UD[1] = GPR[rs].UD[0]  /  LO.UD[1] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTHI1
void recMTHI1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTHI1); }
#else
void recMTHI1()
{
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + 8));
}
#endif

#if ISTUB_MTLO1
void recMTLO1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTLO1); }
#else
void recMTLO1()
{
	armLoadGPR64(RSCRATCHGPR, _Rs_);
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LO_OFFSET + 8));
}
#endif

// ============================================================================
//  MOVZ — Conditional Move if Zero
//  Interpreter: if (!_Rd_) return; if (GPR[rt].UD[0] == 0) GPR[rd].UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MOVZ
void recMOVZ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MOVZ); }
#else
void recMOVZ()
{
	if (!_Rd_)
		return;

	// Conditional move: if rt==0, rd=rs. Must flush rd to memory since
	// the move might not happen at runtime.
	GPR_DEL_CONST(_Rd_);

	// rt const nonzero: move never happens
	if (GPR_IS_CONST1(_Rt_) && g_cpuConstRegs[_Rt_].UD[0] != 0)
		return;

	// rt const zero (or both const): unconditional move
	if (GPR_IS_CONST1(_Rt_))
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	// General case: conditional store
	armLoadGPR64(RSCRATCHGPR, _Rt_);
	a64::Label skip;
	armAsm->Cbnz(RSCRATCHGPR, &skip);
	armLoadGPR64(RSCRATCHGPR2, _Rs_);
	armAsm->Str(RSCRATCHGPR2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
	armAsm->Bind(&skip);
}
#endif

// ============================================================================
//  MOVN — Conditional Move if Not Zero
//  Interpreter: if (!_Rd_) return; if (GPR[rt].UD[0] != 0) GPR[rd].UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MOVN
void recMOVN() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MOVN); }
#else
void recMOVN()
{
	if (!_Rd_)
		return;

	// Conditional move: if rt!=0, rd=rs. Must flush rd to memory since
	// the move might not happen at runtime.
	GPR_DEL_CONST(_Rd_);

	// rt const zero: move never happens
	if (GPR_IS_CONST1(_Rt_) && g_cpuConstRegs[_Rt_].UD[0] == 0)
		return;

	// rt const nonzero (or both const): unconditional move
	if (GPR_IS_CONST1(_Rt_))
	{
		armLoadGPR64(RSCRATCHGPR, _Rs_);
		armStoreGPR64(RSCRATCHGPR, _Rd_);
		return;
	}

	// General case: conditional store
	armLoadGPR64(RSCRATCHGPR, _Rt_);
	a64::Label skip;
	armAsm->Cbz(RSCRATCHGPR, &skip);
	armLoadGPR64(RSCRATCHGPR2, _Rs_);
	armAsm->Str(RSCRATCHGPR2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
	armAsm->Bind(&skip);
}
#endif

// ============================================================================
//  MFSA — Move from Shift Amount register
//  Interpreter: if (!_Rd_) return; GPR[rd].UD[0] = (u64)cpuRegs.sa
// ============================================================================

#if ISTUB_MFSA
void recMFSA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFSA); }
#else
void recMFSA()
{
	if (!_Rd_)
		return;

	GPR_DEL_CONST(_Rd_);
	// sa is u32, zero-extend to 64 via ldr w-reg (w4 load zero-extends into x4)
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
	armStoreGPR64(RSCRATCHGPR, _Rd_);
}
#endif

// ============================================================================
//  MTSA — Move to Shift Amount register
//  Interpreter: cpuRegs.sa = (u32)GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTSA
void recMTSA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSA); }
#else
void recMTSA()
{
	armLoadGPR32(RWSCRATCH, _Rs_);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

// ============================================================================
//  MTSAB — Move to Shift Amount Byte
//  Interpreter: cpuRegs.sa = ((GPR[rs].UL[0] & 0xF) ^ (_Imm_ & 0xF))
// ============================================================================

#if ISTUB_MTSAB
void recMTSAB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSAB); }
#else
void recMTSAB()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 result = (g_cpuConstRegs[_Rs_].UL[0] & 0xF) ^ (_ImmU_ & 0xF);
		armAsm->Mov(RWSCRATCH, result);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		return;
	}

	// w4 = GPR[rs].UL[0]
	armLoadGPR32(RWSCRATCH, _Rs_);
	// w4 = w4 & 0xF
	armAsm->And(RWSCRATCH, RWSCRATCH, 0xF);
	// w4 = w4 ^ (imm & 0xF)
	u32 immMask = _ImmU_ & 0xF;
	if (immMask)
		armAsm->Eor(RWSCRATCH, RWSCRATCH, immMask);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

// ============================================================================
//  MTSAH — Move to Shift Amount Halfword
//  Interpreter: cpuRegs.sa = ((GPR[rs].UL[0] & 0x7) ^ (_Imm_ & 0x7)) << 1
// ============================================================================

#if ISTUB_MTSAH
void recMTSAH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSAH); }
#else
void recMTSAH()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 result = ((g_cpuConstRegs[_Rs_].UL[0] & 0x7) ^ (_ImmU_ & 0x7)) << 1;
		armAsm->Mov(RWSCRATCH, result);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		return;
	}

	// w4 = GPR[rs].UL[0]
	armLoadGPR32(RWSCRATCH, _Rs_);
	// w4 = w4 & 0x7
	armAsm->And(RWSCRATCH, RWSCRATCH, 0x7);
	// w4 = w4 ^ (imm & 0x7)
	u32 immMask = _ImmU_ & 0x7;
	if (immMask)
		armAsm->Eor(RWSCRATCH, RWSCRATCH, immMask);
	// w4 = w4 << 1
	armAsm->Lsl(RWSCRATCH, RWSCRATCH, 1);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
