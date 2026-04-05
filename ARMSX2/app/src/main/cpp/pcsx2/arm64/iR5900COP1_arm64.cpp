// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP1 (FPU) Instructions
// MFC1, MTC1, CFC1, CTC1, BC1x, arithmetic, compare, convert

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// fpuRegisters field offsets from RFPUSTATE (x20 = &fpuRegs)
static constexpr s64 FPR_OFFSET(int reg) { return offsetof(fpuRegisters, fpr) + reg * sizeof(FPRreg); }
static constexpr s64 FPRC_OFFSET(int reg) { return offsetof(fpuRegisters, fprc) + reg * sizeof(u32); }

// COP1 field aliases: _Fs_=bits 15:11, _Ft_=bits 20:16, _Fd_=bits 10:6
#define _Fs_cop1_ _Rd_
#define _Ft_cop1_ _Rt_
#define _Fd_cop1_ _Sa_

// FPU accumulator offset from RFPUSTATE
static constexpr s64 ACC_OFFSET = offsetof(fpuRegisters, ACC);

// PS2 FPU constants
static constexpr u32 PS2_POS_FMAX  = 0x7F7FFFFF;
static constexpr u32 PS2_FPU_FLAG_C  = 0x00800000;
static constexpr u32 PS2_FPU_FLAG_O  = 0x00008000;
static constexpr u32 PS2_FPU_FLAG_U  = 0x00004000;
static constexpr u32 PS2_FPU_FLAG_SO = 0x00000010;
static constexpr u32 PS2_FPU_FLAG_SU = 0x00000008;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#ifdef INTERP_COP1
#define ISTUB_MFC1     1
#define ISTUB_MTC1     1
#define ISTUB_CFC1     1
#define ISTUB_CTC1     1
#define ISTUB_BC1F     1
#define ISTUB_BC1T     1
#define ISTUB_BC1FL    1
#define ISTUB_BC1TL    1
#define ISTUB_ADD_S    1
#define ISTUB_ADDA_S   1
#define ISTUB_SUB_S    1
#define ISTUB_SUBA_S   1
#define ISTUB_ABS_S    1
#define ISTUB_MOV_S    1
#define ISTUB_NEG_S    1
#define ISTUB_MAX_S    1
#define ISTUB_MIN_S    1
#define ISTUB_MUL_S    1
#define ISTUB_DIV_S    1
#define ISTUB_SQRT_S   1
#define ISTUB_RSQRT_S  1
#define ISTUB_MULA_S   1
#define ISTUB_MADD_S   1
#define ISTUB_MSUB_S   1
#define ISTUB_MADDA_S  1
#define ISTUB_MSUBA_S  1
#define ISTUB_C_F      1
#define ISTUB_C_EQ     1
#define ISTUB_C_LT     1
#define ISTUB_C_LE     1
#define ISTUB_CVT_S    1
#define ISTUB_CVT_W    1
#else
#define ISTUB_MFC1     0
#define ISTUB_MTC1     0
#define ISTUB_CFC1     0
#define ISTUB_CTC1     0
#define ISTUB_BC1F     0
#define ISTUB_BC1T     0
#define ISTUB_BC1FL    0
#define ISTUB_BC1TL    0
#define ISTUB_ADD_S    0
#define ISTUB_ADDA_S   0
#define ISTUB_SUB_S    0
#define ISTUB_SUBA_S   0
#define ISTUB_ABS_S    0
#define ISTUB_MOV_S    0
#define ISTUB_NEG_S    0
#define ISTUB_MAX_S    0
#define ISTUB_MIN_S    0
#define ISTUB_MUL_S    0
#define ISTUB_DIV_S    0
#define ISTUB_SQRT_S   0
#define ISTUB_RSQRT_S  0
#define ISTUB_MULA_S   0
#define ISTUB_MADD_S   0
#define ISTUB_MSUB_S   0
#define ISTUB_MADDA_S  0
#define ISTUB_MSUBA_S  0
#define ISTUB_C_F      0
#define ISTUB_C_EQ     0
#define ISTUB_C_LT     0
#define ISTUB_C_LE     0
#define ISTUB_CVT_S    0
#define ISTUB_CVT_W    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP1 {

// ============================================================================
//  MFC1 — Move from FPU register
//  Interpreter: if (!_Rt_) return; GPR[rt].SD[0] = fpuRegs.fpr[fs].SL
//  (sign-extend 32-bit FPR to 64-bit GPR)
// ============================================================================

#if ISTUB_MFC1
void recMFC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MFC1); }
#else
void recMFC1()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
}
#endif

// ============================================================================
//  MTC1 — Move to FPU register
//  Interpreter: fpuRegs.fpr[fs].UL = GPR[rt].UL[0]
//  (32-bit copy from GPR to FPR)
// ============================================================================

#if ISTUB_MTC1
void recMTC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MTC1); }
#else
void recMTC1()
{
	// Load 32-bit from GPR[rt], store to fpr[fs]
	armLoadGPR32(RWSCRATCH, _Rt_);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
}
#endif

// ============================================================================
//  CFC1 — Move from FPU control register
//  Interpreter:
//    if (!_Rt_) return;
//    if (fs == 31) GPR[rt].SD[0] = (s32)fprc[31]
//    else if (fs == 0) GPR[rt].SD[0] = 0x2E00
//    else GPR[rt].SD[0] = 0
// ============================================================================

#if ISTUB_CFC1
void recCFC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CFC1); }
#else
void recCFC1()
{
	if (!_Rt_)
		return;

	GPR_DEL_CONST(_Rt_);
	if (_Fs_cop1_ == 31)
	{
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
		armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
	}
	else if (_Fs_cop1_ == 0)
	{
		// FCR0 = revision register, always 0x2E00
		armAsm->Mov(RSCRATCHGPR, (u64)0x2E00);
		armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	}
	else
	{
		// All other control registers read as 0
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	}
}
#endif

// ============================================================================
//  CTC1 — Move to FPU control register
//  Interpreter: if (fs != 31) return; fprc[fs] = GPR[rt].UL[0]
// ============================================================================

#if ISTUB_CTC1
void recCTC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CTC1); }
#else
void recCTC1()
{
	if (_Fs_cop1_ != 31)
		return;

	armLoadGPR32(RWSCRATCH, _Rt_);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
}
#endif

// ============================================================================
//  BC1F / BC1T / BC1FL / BC1TL — COP1 branches
//  Test FPU condition flag (bit 23 of fprc[31])
// ============================================================================

} // namespace COP1
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP1 {

// Helper: non-likely BC1 branch.
// branchIfSet=false → BC1F (branch when flag clear),
// branchIfSet=true  → BC1T (branch when flag set).
static void recBC1_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Test FPU condition flag
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
	armAsm->Tst(RWSCRATCH, PS2_FPU_FLAG_C);
	// Cset: 1 = taken. BC1F taken when flag clear (eq), BC1T when set (ne).
	armAsm->Cset(RDELAYSLOTGPR, branchIfSet ? a64::ne : a64::eq);

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

// Helper: likely BC1 branch (skip delay slot if not taken).
static void recBC1_Likely_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Test FPU condition flag (bit 23)
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));

	// If NOT taken, skip delay slot.
	// BC1FL not-taken when C SET → Tbnz; BC1TL not-taken when C CLEAR → Tbz.
	// Tbz/Tbnz on bit 23 replaces Tst + B(cond) — saves 1 instruction.
	a64::Label skipDelaySlot, done;
	if (branchIfSet)
		armAsm->Tbz(RWSCRATCH, 23, &skipDelaySlot);  // BC1TL: skip when C clear
	else
		armAsm->Tbnz(RWSCRATCH, 23, &skipDelaySlot); // BC1FL: skip when C set

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

#if ISTUB_BC1F
void recBC1F()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::BC1F);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC1F()  { recBC1_helper(false); }
#endif

#if ISTUB_BC1T
void recBC1T()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::BC1T);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC1T()  { recBC1_helper(true); }
#endif

#if ISTUB_BC1FL
void recBC1FL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::BC1FL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC1FL() { recBC1_Likely_helper(false); }
#endif

#if ISTUB_BC1TL
void recBC1TL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::BC1TL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
#else
void recBC1TL() { recBC1_Likely_helper(true); }
#endif

// ============================================================================
//  FPU codegen helpers
// ============================================================================

// NEON/FP scratch registers (caller-saved)
#define RFSCRATCH0 a64::s0
#define RFSCRATCH1 a64::s1
#define RFSCRATCH2 a64::s2

// Emit PS2 FPU input clamping (fpuDouble equivalent):
//   denormal (exp==0) → ±0, inf/NaN (exp==0xFF) → ±Fmax
// Input: wSrc = raw FPR bits. Output: sDst = clamped float.
// Clobbers: wSrc, RWSCRATCH3
static void armFpuClampInput(const a64::Register& wSrc, const a64::VRegister& sDst)
{
	// Extract exponent field (bits 30:23)
	armAsm->Ubfx(RWSCRATCH3, wSrc, 23, 8);
	// If exponent == 0: flush to ±0 (keep sign, zero rest)
	a64::Label notDenorm, done;
	armAsm->Cbnz(RWSCRATCH3, &notDenorm);
	armAsm->And(wSrc, wSrc, 0x80000000);
	armAsm->B(&done);
	armAsm->Bind(&notDenorm);
	// If exponent == 0xFF: clamp to ±Fmax
	armAsm->Cmp(RWSCRATCH3, 0xFF);
	a64::Label notInf;
	armAsm->B(&notInf, a64::ne);
	armAsm->And(wSrc, wSrc, 0x80000000);
	armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
	armAsm->Orr(wSrc, wSrc, RWSCRATCH3);
	armAsm->Bind(&notInf);
	armAsm->Bind(&done);
	armAsm->Fmov(sDst, wSrc);
}

// Emit PS2 FPU output clamping (checkOverflow + checkUnderflow):
//   infinity → ±Fmax, denormal → ±0
// Input: sDst has the result. Output: wDst = clamped result bits.
// Clobbers: RWSCRATCH3
static void armFpuClampOutput(const a64::VRegister& sSrc, const a64::Register& wDst)
{
	armAsm->Fmov(wDst, sSrc);
	armAsm->Ubfx(RWSCRATCH3, wDst, 23, 8);
	// Overflow: exp == 0xFF → ±Fmax
	armAsm->Cmp(RWSCRATCH3, 0xFF);
	a64::Label notOvf, checkUdf, storeDone;
	armAsm->B(&notOvf, a64::ne);
	armAsm->And(wDst, wDst, 0x80000000);
	armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
	armAsm->Orr(wDst, wDst, RWSCRATCH3);
	armAsm->B(&storeDone);
	armAsm->Bind(&notOvf);
	// Underflow: exp == 0 && mantissa != 0 → ±0
	armAsm->Cbnz(RWSCRATCH3, &storeDone);
	armAsm->Tst(wDst, 0x007FFFFF);
	armAsm->B(&storeDone, a64::eq);
	armAsm->And(wDst, wDst, 0x80000000);
	armAsm->Bind(&storeDone);
}

// Emit a two-operand FPU arithmetic op: fd = clamp(op(clamp(fs), clamp(ft)))
// opFunc emits the actual instruction given (sDst, sSrc0, sSrc1)
template<typename OpFunc>
static void armFpuBinOp(int fd, int fs, int ft, OpFunc opFunc)
{
	// Load and clamp fs
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(fs)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	// Load and clamp ft
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(ft)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	// Operation
	opFunc(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	// Clamp output and store
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(fd)));
}

// Same but stores to ACC instead of fd
template<typename OpFunc>
static void armFpuBinOpAcc(int fs, int ft, OpFunc opFunc)
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(fs)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(ft)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	opFunc(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
}

// ============================================================================
//  FPU arithmetic / compare / convert
// ============================================================================

// ---- ABS_S ----
#if ISTUB_ABS_S
void recABS_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ABS_S); }
#else
void recABS_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->And(RWSCRATCH, RWSCRATCH, 0x7FFFFFFF);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- NEG_S ----
#if ISTUB_NEG_S
void recNEG_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::NEG_S); }
#else
void recNEG_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Eor(RWSCRATCH, RWSCRATCH, 0x80000000);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MOV_S ----
#if ISTUB_MOV_S
void recMOV_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MOV_S); }
#else
void recMOV_S()
{
	if (_Fs_cop1_ == _Fd_cop1_)
		return;
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MAX_S / MIN_S ----
#if ISTUB_MAX_S
void recMAX_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MAX_S); }
#else
void recMAX_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MAX_S); }
#endif

#if ISTUB_MIN_S
void recMIN_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MIN_S); }
#else
void recMIN_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MIN_S); }
#endif

// ---- ADD_S ----
#if ISTUB_ADD_S
void recADD_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ADD_S); }
#else
void recADD_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fadd(d, a, b); });
}
#endif

// ---- SUB_S ----
#if ISTUB_SUB_S
void recSUB_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SUB_S); }
#else
void recSUB_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fsub(d, a, b); });
}
#endif

// ---- MUL_S ----
#if ISTUB_MUL_S
void recMUL_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MUL_S); }
#else
void recMUL_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmul(d, a, b); });
}
#endif

// ---- ADDA_S ----
#if ISTUB_ADDA_S
void recADDA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ADDA_S); }
#else
void recADDA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fadd(d, a, b); });
}
#endif

// ---- SUBA_S ----
#if ISTUB_SUBA_S
void recSUBA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SUBA_S); }
#else
void recSUBA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fsub(d, a, b); });
}
#endif

// ---- MULA_S ----
#if ISTUB_MULA_S
void recMULA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MULA_S); }
#else
void recMULA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmul(d, a, b); });
}
#endif

// ---- MADD_S ----
#if ISTUB_MADD_S
void recMADD_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MADD_S); }
#else
void recMADD_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fadd(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MSUB_S ----
#if ISTUB_MSUB_S
void recMSUB_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MSUB_S); }
#else
void recMSUB_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fsub(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MADDA_S ----
#if ISTUB_MADDA_S
void recMADDA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MADDA_S); }
#else
void recMADDA_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
	armAsm->Fmov(RFSCRATCH1, RWSCRATCH);
	armAsm->Fadd(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
}
#endif

// ---- MSUBA_S ----
#if ISTUB_MSUBA_S
void recMSUBA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MSUBA_S); }
#else
void recMSUBA_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
	armAsm->Fmov(RFSCRATCH1, RWSCRATCH);
	armAsm->Fsub(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, ACC_OFFSET));
}
#endif

// ---- DIV_S / SQRT_S / RSQRT_S ----
#if ISTUB_DIV_S
void recDIV_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::DIV_S); }
#else
void recDIV_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::DIV_S); }
#endif

#if ISTUB_SQRT_S
void recSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SQRT_S); }
#else
void recSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SQRT_S); }
#endif

#if ISTUB_RSQRT_S
void recRSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::RSQRT_S); }
#else
void recRSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::RSQRT_S); }
#endif

// ---- C_F ----
#if ISTUB_C_F
void recC_F() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_F); }
#else
void recC_F()
{
	// Bic clears specific bits: 0x00800000 is a valid ARM64 logical immediate,
	// ~0x00800000 is not — Bic saves the scratch+And pair.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH, RWSCRATCH, PS2_FPU_FLAG_C);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
}
#endif

// ---- C_EQ / C_LT / C_LE ----
static void armFpuCompare(a64::Condition cond)
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fcmp(RFSCRATCH0, RFSCRATCH1);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
	// Bic clears PS2_FPU_FLAG_C (0x00800000 is valid logical immediate; ~C is not).
	armAsm->Bic(RWSCRATCH, RWSCRATCH, PS2_FPU_FLAG_C);
	// Cset → 0 or 1, then Lsl(23) → 0 or PS2_FPU_FLAG_C.
	armAsm->Cset(RWSCRATCH2, cond);
	armAsm->Lsl(RWSCRATCH2, RWSCRATCH2, 23);
	armAsm->Orr(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPRC_OFFSET(31)));
}

#if ISTUB_C_EQ
void recC_EQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_EQ); }
#else
void recC_EQ() { armFpuCompare(a64::eq); }
#endif

#if ISTUB_C_LT
void recC_LT() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_LT); }
#else
void recC_LT() { armFpuCompare(a64::mi); }
#endif

#if ISTUB_C_LE
void recC_LE() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_LE); }
#else
void recC_LE() { armFpuCompare(a64::ls); }
#endif

// ---- CVT_S ----
#if ISTUB_CVT_S
void recCVT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CVT_S); }
#else
void recCVT_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Scvtf(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RFSCRATCH0, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- CVT_W ----
#if ISTUB_CVT_W
void recCVT_W() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CVT_W); }
#else
void recCVT_W()
{
	// Clamp input (denormal → ±0, inf/NaN → ±Fmax), then convert to s32.
	// ARM64 FCVTZS saturates: >MAX→0x7FFFFFFF, <MIN→0x80000000
	// which matches PS2 CVT_W overflow behavior for clamped inputs.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Fcvtzs(RWSCRATCH, RFSCRATCH0);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RFPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

} // namespace COP1
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
