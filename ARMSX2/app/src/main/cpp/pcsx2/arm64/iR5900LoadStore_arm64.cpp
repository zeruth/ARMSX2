// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Load / Store Instructions
// LB, LBU, LH, LHU, LW, LWU, LD, LQ, LWL, LWR, LDL, LDR, LWC1, LQC2
// SB, SH, SW, SD, SQ, SWL, SWR, SDL, SDR, SWC1, SQC2
//
// Aligned 8/16/32/64-bit loads and stores use VTLB fastmem when available:
// a single LDR/STR through RFASTMEMBASE (x23 = vtlbdata.fastmem_base).
// On fault the signal handler backpatches to a thunk that calls the C
// vtlb_memRead/Write functions (see recVTLB_arm64.cpp).

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"
#include "vtlb.h"

using namespace R5900;

// FPR offset from RCPUSTATE (x19) — for LWC1/SWC1. fpuRegs lives at FPUREGS_BASE
// inside cpuRegistersPack, so we address FPU state without a dedicated base reg.
static constexpr s64 FPR_OFFSET(int reg) { return FPUREGS_BASE + offsetof(fpuRegisters, fpr) + reg * sizeof(FPRreg); }

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_LOAD) || defined(INTERP_EE)
#define ISTUB_LB       1
#define ISTUB_LBU      1
#define ISTUB_LH       1
#define ISTUB_LHU      1
#define ISTUB_LW       1
#define ISTUB_LWU      1
#define ISTUB_LD       1
#define ISTUB_LWC1     1
#define ISTUB_LWL      1
#define ISTUB_LWR      1
#define ISTUB_LDL      1
#define ISTUB_LDR      1
#define ISTUB_LQ       1
#define ISTUB_LQC2     1
#else
#define ISTUB_LB       0
#define ISTUB_LBU      0
#define ISTUB_LH       0
#define ISTUB_LHU      0
#define ISTUB_LW       0
#define ISTUB_LWU      0
#define ISTUB_LD       0
#define ISTUB_LWC1     0
#define ISTUB_LWL      1   // unaligned — keep as interp
#define ISTUB_LWR      1
#define ISTUB_LDL      1
#define ISTUB_LDR      1
#define ISTUB_LQ       1   // 128-bit — keep as interp
#define ISTUB_LQC2     1   // 128-bit VU — keep as interp
#endif

#if defined(INTERP_STORE) || defined(INTERP_EE)
#define ISTUB_SB       1
#define ISTUB_SH       1
#define ISTUB_SW       1
#define ISTUB_SD       1
#define ISTUB_SWC1     1
#define ISTUB_SWL      1
#define ISTUB_SWR      1
#define ISTUB_SDL      1
#define ISTUB_SDR      1
#define ISTUB_SQ       1
#define ISTUB_SQC2     1
#else
#define ISTUB_SB       0
#define ISTUB_SH       0
#define ISTUB_SW       0
#define ISTUB_SD       0
#define ISTUB_SWC1     0
#define ISTUB_SWL      1   // unaligned — keep as interp
#define ISTUB_SWR      1
#define ISTUB_SDL      1
#define ISTUB_SDR      1
#define ISTUB_SQ       1   // 128-bit — keep as interp
#define ISTUB_SQC2     1   // 128-bit VU — keep as interp
#endif

// ============================================================================
//  Helpers: address computation and vtlb call wrappers
// ============================================================================

// Emit code to compute effective address (Rs + sign-extended Imm) into w0.
// Uses constant propagation when Rs is known at compile time.
static void armComputeAddress()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 addr = g_cpuConstRegs[_Rs_].UL[0] + _Imm_;
		armAsm->Mov(a64::w0, addr);
	}
	else
	{
		armLoadGPR32(a64::w0, _Rs_);
		if (_Imm_ != 0)
			armAsm->Add(a64::w0, a64::w0, _Imm_);
	}
}

// Flush state before a slow-path vtlb call.
static void armPreVtlbCall()
{
	armFlushConstRegs();
	armFlushPC();
}

// Record a fastmem load/store for backpatching.
// addr_reg and data_reg are ARM64 register codes.
static void armRecordFastmem(const u8* code_start, u8 addr_reg, u8 data_reg,
	u8 bits, bool is_signed, bool is_load, bool is_fpr)
{
	u32 code_size = static_cast<u32>(armGetCurrentCodePointer() - code_start);
	vtlb_AddLoadStoreInfo(reinterpret_cast<uptr>(code_start), code_size, pc,
		0 /*gpr_bitmask*/, 0 /*fpr_bitmask*/,
		addr_reg, data_reg, bits, is_signed, is_load, is_fpr);
}

// Returns true if we should use the fastmem path for the current instruction.
static bool armUseFastmem()
{
	return CHECK_FASTMEM && !vtlb_IsFaultingPC(pc);
}

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//                              LOADS
// ============================================================================

// ---- LB — Load Byte (sign-extended to 64 bits) ----

#if ISTUB_LB
void recLB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LB); }
#else
void recLB()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsb(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 8, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem8_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Sxtb(a64::x0, a64::w0);
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LBU — Load Byte Unsigned ----

#if ISTUB_LBU
void recLBU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LBU); }
#else
void recLBU()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrb(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 8, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem8_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LH — Load Halfword (sign-extended) ----

#if ISTUB_LH
void recLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LH); }
#else
void recLH()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsh(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 16, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem16_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Sxth(a64::x0, a64::w0);
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LHU — Load Halfword Unsigned ----

#if ISTUB_LHU
void recLHU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LHU); }
#else
void recLHU()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrh(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 16, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem16_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LW — Load Word (sign-extended) ----

#if ISTUB_LW
void recLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LW); }
#else
void recLW()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsw(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64SignExt32(a64::w0, _Rt_);
	}
}
#endif

// ---- LWU — Load Word Unsigned ----

#if ISTUB_LWU
void recLWU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWU); }
#else
void recLWU()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LD — Load Doubleword ----

#if ISTUB_LD
void recLD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LD); }
#else
void recLD()
{
	if (!_Rt_) return;
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 64, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem64_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LWC1 — Load Word to COP1 (FPU register) ----

#if ISTUB_LWC1
void recLWC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWC1); }
#else
void recLWC1()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, false, true, false);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Str(a64::w0, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));
	}
}
#endif

// ---- Unaligned / 128-bit loads — interpreter stubs ----

#if ISTUB_LWL
void recLWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWL); }
#else
void recLWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWL); }
#endif

#if ISTUB_LWR
void recLWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWR); }
#else
void recLWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWR); }
#endif

#if ISTUB_LDL
void recLDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDL); }
#else
void recLDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDL); }
#endif

#if ISTUB_LDR
void recLDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDR); }
#else
void recLDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDR); }
#endif

#if ISTUB_LQ
void recLQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQ); }
#else
void recLQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQ); }
#endif

#if ISTUB_LQC2
void recLQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQC2); }
#else
void recLQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQC2); }
#endif

// ============================================================================
//                              STORES
// ============================================================================

// ---- SB — Store Byte ----

#if ISTUB_SB
void recSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SB); }
#else
void recSB()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Strb(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 8, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem8_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SH — Store Halfword ----

#if ISTUB_SH
void recSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SH); }
#else
void recSH()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Strh(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 16, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem16_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SW — Store Word ----

#if ISTUB_SW
void recSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SW); }
#else
void recSW()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 32, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SD — Store Doubleword ----

#if ISTUB_SD
void recSD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SD); }
#else
void recSD()
{
	armComputeAddress();
	armLoadGPR64(a64::x1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::x1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 64, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem64_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SWC1 — Store Word from COP1 (FPU register) ----

#if ISTUB_SWC1
void recSWC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWC1); }
#else
void recSWC1()
{
	armComputeAddress();
	armAsm->Ldr(a64::w1, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 32, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- Unaligned / 128-bit stores — interpreter stubs ----

#if ISTUB_SWL
void recSWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWL); }
#else
void recSWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWL); }
#endif

#if ISTUB_SWR
void recSWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWR); }
#else
void recSWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWR); }
#endif

#if ISTUB_SDL
void recSDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDL); }
#else
void recSDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDL); }
#endif

#if ISTUB_SDR
void recSDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDR); }
#else
void recSDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDR); }
#endif

#if ISTUB_SQ
void recSQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQ); }
#else
void recSQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQ); }
#endif

#if ISTUB_SQC2
void recSQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQC2); }
#else
void recSQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQC2); }
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
