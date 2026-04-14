// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 Native Interpreter Ops — dispatch and implementations.
// Plain C++ functions that directly manipulate cpuRegs, called from the
// interpreter's execI() loop. The interpreter harness handles everything
// else (cycles, events, branches, PC advancement).

#include "Common.h"
#include "R5900.h"
#include "R5900OpcodeTables.h"
#include "VU.h"
#include "COP0.h"
#include "Dmac.h"
#include "arm64/intNativeOps.h"

#include "common/BitUtils.h"
#include <cmath>

#if defined(__aarch64__) || defined(_M_ARM64)

// Forward declaration — defined in VU0.cpp, no header exposes it
extern void vu0Sync();

// Field extraction comes from R5900.h via Common.h:
//   _Rs_, _Rt_, _Rd_, _Funct_, _Imm_, _ImmU_,
//   _JumpTarget_, _BranchTarget_, _SetLink()

// ============================================================================
//  ALU native implementations — no-overflow register ops
// ============================================================================

#if NOP_ADDU
static void nativeADDU()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = u64(s64(s32(cpuRegs.GPR.r[_Rs_].UL[0] + cpuRegs.GPR.r[_Rt_].UL[0])));
}
#endif

#if NOP_SUBU
static void nativeSUBU()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = u64(s64(s32(cpuRegs.GPR.r[_Rs_].UL[0] - cpuRegs.GPR.r[_Rt_].UL[0])));
}
#endif

#if NOP_DADDU
static void nativeDADDU()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] + cpuRegs.GPR.r[_Rt_].UD[0];
}
#endif

#if NOP_DSUBU
static void nativeDSUBU()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] - cpuRegs.GPR.r[_Rt_].UD[0];
}
#endif

#if NOP_AND
static void nativeAND()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] & cpuRegs.GPR.r[_Rt_].UD[0];
}
#endif

#if NOP_OR
static void nativeOR()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] | cpuRegs.GPR.r[_Rt_].UD[0];
}
#endif

#if NOP_XOR
static void nativeXOR()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] ^ cpuRegs.GPR.r[_Rt_].UD[0];
}
#endif

#if NOP_NOR
static void nativeNOR()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = ~(cpuRegs.GPR.r[_Rs_].UD[0] | cpuRegs.GPR.r[_Rt_].UD[0]);
}
#endif

#if NOP_SLT
static void nativeSLT()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (cpuRegs.GPR.r[_Rs_].SD[0] < cpuRegs.GPR.r[_Rt_].SD[0]) ? 1 : 0;
}
#endif

#if NOP_SLTU
static void nativeSLTU()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (cpuRegs.GPR.r[_Rs_].UD[0] < cpuRegs.GPR.r[_Rt_].UD[0]) ? 1 : 0;
}
#endif

// ---- Overflow-trapping register ops ----

#if NOP_ADD
static void nativeADD()
{
	GPR_reg64 result; result.SD[0] = (s64)cpuRegs.GPR.r[_Rs_].SL[0] + cpuRegs.GPR.r[_Rt_].SL[0];
	if ((result.UL[0] >> 31) != (result.UL[1] & 1)) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = result.SD[0];
}
#endif

#if NOP_SUB
static void nativeSUB()
{
	GPR_reg64 result; result.SD[0] = (s64)cpuRegs.GPR.r[_Rs_].SL[0] - cpuRegs.GPR.r[_Rt_].SL[0];
	if ((result.UL[0] >> 31) != (result.UL[1] & 1)) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = result.SD[0];
}
#endif

#if NOP_DADD
static void nativeDADD()
{
	s64 x = cpuRegs.GPR.r[_Rs_].SD[0], y = cpuRegs.GPR.r[_Rt_].SD[0];
	s64 result = x + y;
	if (((~(x ^ y)) & (x ^ result)) < 0) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = result;
}
#endif

#if NOP_DSUB
static void nativeDSUB()
{
	s64 x = cpuRegs.GPR.r[_Rs_].SD[0], y = -cpuRegs.GPR.r[_Rt_].SD[0];
	s64 result = x + y;
	if (((~(x ^ y)) & (x ^ result)) < 0) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = result;
}
#endif

// ---- Immediate ALU ops ----

#if NOP_ADDIU
static void nativeADDIU()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = u64(s64(s32(cpuRegs.GPR.r[_Rs_].UL[0] + u32(s32(_Imm_)))));
}
#endif

#if NOP_DADDIU
static void nativeDADDIU()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] + u64(s64(_Imm_));
}
#endif

#if NOP_ANDI
static void nativeANDI()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] & (u64)_ImmU_;
}
#endif

#if NOP_ORI
static void nativeORI()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] | (u64)_ImmU_;
}
#endif

#if NOP_XORI
static void nativeXORI()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0] ^ (u64)_ImmU_;
}
#endif

#if NOP_SLTI
static void nativeSLTI()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = (cpuRegs.GPR.r[_Rs_].SD[0] < (s64)(_Imm_)) ? 1 : 0;
}
#endif

#if NOP_SLTIU
static void nativeSLTIU()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = (cpuRegs.GPR.r[_Rs_].UD[0] < (u64)(_Imm_)) ? 1 : 0;
}
#endif

// ---- Overflow-trapping immediate ops ----

#if NOP_ADDI
static void nativeADDI()
{
	GPR_reg64 result; result.SD[0] = (s64)cpuRegs.GPR.r[_Rs_].SL[0] + _Imm_;
	if ((result.UL[0] >> 31) != (result.UL[1] & 1)) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].SD[0] = result.SD[0];
}
#endif

#if NOP_DADDI
static void nativeDADDI()
{
	s64 x = cpuRegs.GPR.r[_Rs_].SD[0], y = (s64)_Imm_;
	s64 result = x + y;
	if (((~(x ^ y)) & (x ^ result)) < 0) { cpuException(0x30, cpuRegs.branch); return; }
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].SD[0] = result;
}
#endif

// ============================================================================
//  Load native implementations
// ============================================================================

#if NOP_LB
static void nativeLB()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	s8 val = memRead8(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].SD[0] = val;
}
#endif

#if NOP_LBU
static void nativeLBU()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u8 val = memRead8(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].UD[0] = val;
}
#endif

#if NOP_LH
static void nativeLH()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	s16 val = memRead16(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].SD[0] = val;
}
#endif

#if NOP_LHU
static void nativeLHU()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u16 val = memRead16(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].UD[0] = val;
}
#endif

#if NOP_LW
static void nativeLW()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 val = memRead32(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].SD[0] = (s32)val;
}
#endif

#if NOP_LWU
static void nativeLWU()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 val = memRead32(addr);
	if (_Rt_) cpuRegs.GPR.r[_Rt_].UD[0] = val;
}
#endif

#if NOP_LD
static void nativeLD()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	cpuRegs.GPR.r[_Rt_].UD[0] = memRead64(addr);
}
#endif

#if NOP_LQ
static void nativeLQ()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	if (_Rt_)
		memRead128(addr & ~0xf, (u128*)&cpuRegs.GPR.r[_Rt_]);
	else
	{
		u128 dummy;
		memRead128(addr & ~0xf, dummy);
	}
}
#endif

// Unaligned loads — complex bit manipulation, keep inline
static const u32 LWL_MASK[4] = { 0xffffff, 0x0000ffff, 0x000000ff, 0x00000000 };
static const u32 LWR_MASK[4] = { 0x000000, 0xff000000, 0xffff0000, 0xffffff00 };
static const u8 LWL_SHIFT[4] = { 24, 16, 8, 0 };
static const u8 LWR_SHIFT[4] = { 0, 8, 16, 24 };

#if NOP_LWL
static void nativeLWL()
{
	s32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 3;
	u32 mem = memRead32(addr & ~3);
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].SD[0] = (s32)((cpuRegs.GPR.r[_Rt_].UL[0] & LWL_MASK[shift]) | (mem << LWL_SHIFT[shift]));
}
#endif

#if NOP_LWR
static void nativeLWR()
{
	s32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 3;
	u32 mem = memRead32(addr & ~3);
	if (!_Rt_) return;
	mem = (cpuRegs.GPR.r[_Rt_].UL[0] & LWR_MASK[shift]) | (mem >> LWR_SHIFT[shift]);
	if (shift == 0)
		cpuRegs.GPR.r[_Rt_].SD[0] = (s32)mem;
	else
		cpuRegs.GPR.r[_Rt_].UL[0] = mem;
}
#endif

static const u64 LDL_MASK[8] = {
	0x00ffffffffffffffULL, 0x0000ffffffffffffULL, 0x000000ffffffffffULL, 0x00000000ffffffffULL,
	0x0000000000ffffffULL, 0x000000000000ffffULL, 0x00000000000000ffULL, 0x0000000000000000ULL
};
static const u64 LDR_MASK[8] = {
	0x0000000000000000ULL, 0xff00000000000000ULL, 0xffff000000000000ULL, 0xffffff0000000000ULL,
	0xffffffff00000000ULL, 0xffffffffff000000ULL, 0xffffffffffff0000ULL, 0xffffffffffffff00ULL
};
static const u8 LDL_SHIFT[8] = { 56, 48, 40, 32, 24, 16, 8, 0 };
static const u8 LDR_SHIFT[8] = { 0, 8, 16, 24, 32, 40, 48, 56 };

#if NOP_LDL
static void nativeLDL()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 7;
	u64 mem = memRead64(addr & ~7);
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = (cpuRegs.GPR.r[_Rt_].UD[0] & LDL_MASK[shift]) | (mem << LDL_SHIFT[shift]);
}
#endif

#if NOP_LDR
static void nativeLDR()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 7;
	u64 mem = memRead64(addr & ~7);
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = (cpuRegs.GPR.r[_Rt_].UD[0] & LDR_MASK[shift]) | (mem >> LDR_SHIFT[shift]);
}
#endif

#if NOP_LWC1
static void nativeLWC1()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	fpuRegs.fpr[_Rt_].UL = memRead32(addr);
}
#endif

#if NOP_LQC2
static void nativeLQC2()
{
	vu0Sync();
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + (s16)cpuRegs.code;
	if (_Rt_)
		memRead128(addr, VU0.VF[_Rt_].UQ);
	else
	{
		u128 dummy;
		memRead128(addr, dummy);
	}
}
#endif

// ============================================================================
//  Store native implementations
// ============================================================================

#if NOP_SB
static void nativeSB()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite8(addr, cpuRegs.GPR.r[_Rt_].UC[0]);
}
#endif

#if NOP_SH
static void nativeSH()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite16(addr, cpuRegs.GPR.r[_Rt_].US[0]);
}
#endif

#if NOP_SW
static void nativeSW()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite32(addr, cpuRegs.GPR.r[_Rt_].UL[0]);
}
#endif

#if NOP_SD
static void nativeSD()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite64(addr, cpuRegs.GPR.r[_Rt_].UD[0]);
}
#endif

#if NOP_SQ
static void nativeSQ()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite128(addr & ~0xf, cpuRegs.GPR.r[_Rt_].UQ);
}
#endif

static const u32 SWL_MASK[4] = { 0xffffff00, 0xffff0000, 0xff000000, 0x00000000 };
static const u32 SWR_MASK[4] = { 0x00000000, 0x000000ff, 0x0000ffff, 0x00ffffff };
static const u8 SWL_SHIFT[4] = { 24, 16, 8, 0 };
static const u8 SWR_SHIFT[4] = { 0, 8, 16, 24 };

#if NOP_SWL
static void nativeSWL()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 3;
	u32 mem = memRead32(addr & ~3);
	memWrite32(addr & ~3, (cpuRegs.GPR.r[_Rt_].UL[0] >> SWL_SHIFT[shift]) | (mem & SWL_MASK[shift]));
}
#endif

#if NOP_SWR
static void nativeSWR()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 3;
	u32 mem = memRead32(addr & ~3);
	memWrite32(addr & ~3, (cpuRegs.GPR.r[_Rt_].UL[0] << SWR_SHIFT[shift]) | (mem & SWR_MASK[shift]));
}
#endif

static const u64 SDL_MASK[8] = {
	0xffffffffffffff00ULL, 0xffffffffffff0000ULL, 0xffffffffff000000ULL, 0xffffffff00000000ULL,
	0xffffff0000000000ULL, 0xffff000000000000ULL, 0xff00000000000000ULL, 0x0000000000000000ULL
};
static const u64 SDR_MASK[8] = {
	0x0000000000000000ULL, 0x00000000000000ffULL, 0x000000000000ffffULL, 0x0000000000ffffffULL,
	0x00000000ffffffffULL, 0x000000ffffffffffULL, 0x0000ffffffffffffULL, 0x00ffffffffffffffULL
};
static const u8 SDL_SHIFT[8] = { 56, 48, 40, 32, 24, 16, 8, 0 };
static const u8 SDR_SHIFT[8] = { 0, 8, 16, 24, 32, 40, 48, 56 };

#if NOP_SDL
static void nativeSDL()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 7;
	u64 mem = memRead64(addr & ~7);
	memWrite64(addr & ~7, (cpuRegs.GPR.r[_Rt_].UD[0] >> SDL_SHIFT[shift]) | (mem & SDL_MASK[shift]));
}
#endif

#if NOP_SDR
static void nativeSDR()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	u32 shift = addr & 7;
	u64 mem = memRead64(addr & ~7);
	memWrite64(addr & ~7, (cpuRegs.GPR.r[_Rt_].UD[0] << SDR_SHIFT[shift]) | (mem & SDR_MASK[shift]));
}
#endif

#if NOP_SWC1
static void nativeSWC1()
{
	u32 addr = cpuRegs.GPR.r[_Rs_].UL[0] + _Imm_;
	memWrite32(addr, fpuRegs.fpr[_Rt_].UL);
}
#endif

#if NOP_SQC2
static void nativeSQC2()
{
	vu0Sync();
	u32 addr = _Imm_ + cpuRegs.GPR.r[_Rs_].UL[0];
	memWrite128(addr, VU0.VF[_Rt_].UQ);
}
#endif

// ============================================================================
//  Shift native implementations
// ============================================================================

#if NOP_SLL
static void nativeSLL()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].UL[0] << _Sa_);
}
#endif

#if NOP_SRL
static void nativeSRL()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].UL[0] >> _Sa_);
}
#endif

#if NOP_SRA
static void nativeSRA()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].SL[0] >> _Sa_);
}
#endif

#if NOP_SLLV
static void nativeSLLV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].UL[0] << (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1f));
}
#endif

#if NOP_SRLV
static void nativeSRLV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].UL[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1f));
}
#endif

#if NOP_SRAV
static void nativeSRAV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s32)(cpuRegs.GPR.r[_Rt_].SL[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1f));
}
#endif

#if NOP_DSLL
static void nativeDSLL()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (u64)(cpuRegs.GPR.r[_Rt_].UD[0] << _Sa_);
}
#endif

#if NOP_DSRL
static void nativeDSRL()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rt_].UD[0] >> _Sa_;
}
#endif

#if NOP_DSRA
static void nativeDSRA()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.GPR.r[_Rt_].SD[0] >> _Sa_;
}
#endif

#if NOP_DSLL32
static void nativeDSLL32()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (u64)(cpuRegs.GPR.r[_Rt_].UD[0] << (_Sa_ + 32));
}
#endif

#if NOP_DSRL32
static void nativeDSRL32()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rt_].UD[0] >> (_Sa_ + 32);
}
#endif

#if NOP_DSRA32
static void nativeDSRA32()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.GPR.r[_Rt_].SD[0] >> (_Sa_ + 32);
}
#endif

#if NOP_DSLLV
static void nativeDSLLV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (u64)(cpuRegs.GPR.r[_Rt_].UD[0] << (cpuRegs.GPR.r[_Rs_].UL[0] & 0x3f));
}
#endif

#if NOP_DSRLV
static void nativeDSRLV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (u64)(cpuRegs.GPR.r[_Rt_].UD[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x3f));
}
#endif

#if NOP_DSRAV
static void nativeDSRAV()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s64)(cpuRegs.GPR.r[_Rt_].SD[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x3f));
}
#endif

// ============================================================================
//  Branch native implementations
//  These call intDoBranch() for taken branches and intEventTest() for
//  not-taken cases, matching the interpreter's doBranch() behavior.
// ============================================================================

#if NOP_J
static void nativeJ()
{
	intDoBranch(_JumpTarget_);
}
#endif

#if NOP_JAL
static void nativeJAL()
{
	_SetLink(31);
	intDoBranch(_JumpTarget_);
}
#endif

#if NOP_BEQ
static void nativeBEQ()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] == cpuRegs.GPR.r[_Rt_].SD[0])
		intDoBranch(_BranchTarget_);
	else
		intEventTest();
}
#endif

#if NOP_BNE
static void nativeBNE()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] != cpuRegs.GPR.r[_Rt_].SD[0])
		intDoBranch(_BranchTarget_);
	else
		intEventTest();
}
#endif

#if NOP_BLEZ
static void nativeBLEZ()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] <= 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BGTZ
static void nativeBGTZ()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] > 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BEQL
static void nativeBEQL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] == cpuRegs.GPR.r[_Rt_].SD[0])
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BNEL
static void nativeBNEL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] != cpuRegs.GPR.r[_Rt_].SD[0])
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BLEZL
static void nativeBLEZL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] <= 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BGTZL
static void nativeBGTZL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] > 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_JR
static void nativeJR()
{
	intDoBranch(cpuRegs.GPR.r[_Rs_].UL[0]);
}
#endif

#if NOP_JALR
static void nativeJALR()
{
	const u32 temp = cpuRegs.GPR.r[_Rs_].UL[0];
	if (_Rd_) _SetLink(_Rd_);
	intDoBranch(temp);
}
#endif

#if NOP_SYSCALL
static void nativeSYSCALL()
{
	// SYSCALL/BREAK use the interpreter's own implementation directly,
	// which handles cpuException. We just call through opcode.interpret().
	R5900::Interpreter::OpcodeImpl::SYSCALL();
}
#endif

#if NOP_BREAK
static void nativeBREAK()
{
	R5900::Interpreter::OpcodeImpl::BREAK();
}
#endif

// ---- REGIMM branches ----

#if NOP_BLTZ
static void nativeBLTZ()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BGEZ
static void nativeBGEZ()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BLTZL
static void nativeBLTZL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BGEZL
static void nativeBGEZL()
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BLTZAL
static void nativeBLTZAL()
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BGEZAL
static void nativeBGEZAL()
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
		intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BLTZALL
static void nativeBLTZALL()
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

#if NOP_BGEZALL
static void nativeBGEZALL()
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
		intDoBranch(_BranchTarget_);
	else
	{
		cpuRegs.pc += 4;
		intEventTest();
	}
}
#endif

// ============================================================================
//  Move native implementations
// ============================================================================

#if NOP_LUI
static void nativeLUI()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].UD[0] = (s32)(cpuRegs.code << 16);
}
#endif

#if NOP_MFHI
static void nativeMFHI()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.HI.UD[0];
}
#endif

#if NOP_MFLO
static void nativeMFLO()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.LO.UD[0];
}
#endif

#if NOP_MTHI
static void nativeMTHI()
{
	cpuRegs.HI.UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MTLO
static void nativeMTLO()
{
	cpuRegs.LO.UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MFHI1
static void nativeMFHI1()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.HI.UD[1];
}
#endif

#if NOP_MFLO1
static void nativeMFLO1()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.LO.UD[1];
}
#endif

#if NOP_MTHI1
static void nativeMTHI1()
{
	cpuRegs.HI.UD[1] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MTLO1
static void nativeMTLO1()
{
	cpuRegs.LO.UD[1] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MOVZ
static void nativeMOVZ()
{
	if (!_Rd_) return;
	if (cpuRegs.GPR.r[_Rt_].UD[0] == 0)
		cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MOVN
static void nativeMOVN()
{
	if (!_Rd_) return;
	if (cpuRegs.GPR.r[_Rt_].UD[0] != 0)
		cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MFSA
static void nativeMFSA()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = (u64)cpuRegs.sa;
}
#endif

#if NOP_MTSA
static void nativeMTSA()
{
	cpuRegs.sa = (u32)cpuRegs.GPR.r[_Rs_].UD[0];
}
#endif

#if NOP_MTSAB
static void nativeMTSAB()
{
	cpuRegs.sa = ((cpuRegs.GPR.r[_Rs_].UL[0] & 0xF) ^ (_Imm_ & 0xF));
}
#endif

#if NOP_MTSAH
static void nativeMTSAH()
{
	cpuRegs.sa = ((cpuRegs.GPR.r[_Rs_].UL[0] & 0x7) ^ (_Imm_ & 0x7)) << 1;
}
#endif

// ============================================================================
//  COP0 native implementations
// ============================================================================

#if NOP_MFC0
static void nativeMFC0()
{
	// CP0.Count (rd=9) must be updated even if rt=0
	if ((_Rd_ != 9) && !_Rt_) return;

	switch (_Rd_)
	{
		case 12:
			cpuRegs.GPR.r[_Rt_].SD[0] = (s32)(cpuRegs.CP0.r[_Rd_] & 0xf0c79c1f);
			break;
		case 25:
			if (0 == (_Imm_ & 1))
			{
				cpuRegs.GPR.r[_Rt_].SD[0] = (s32)cpuRegs.PERF.n.pccr.val;
			}
			else if (0 == (_Imm_ & 2))
			{
				COP0_UpdatePCCR();
				cpuRegs.GPR.r[_Rt_].SD[0] = (s32)cpuRegs.PERF.n.pcr0;
			}
			else
			{
				COP0_UpdatePCCR();
				cpuRegs.GPR.r[_Rt_].SD[0] = (s32)cpuRegs.PERF.n.pcr1;
			}
			break;
		case 24:
			break; // debug breakpoint regs — no-op
		case 9:
		{
			s64 incr = cpuRegs.cycle - cpuRegs.lastCOP0Cycle;
			if (incr == 0) incr++;
			cpuRegs.CP0.n.Count += incr;
			cpuRegs.lastCOP0Cycle = cpuRegs.cycle;
			if (!_Rt_) break;
		}
			[[fallthrough]];
		default:
			cpuRegs.GPR.r[_Rt_].SD[0] = (s32)cpuRegs.CP0.r[_Rd_];
	}
}
#endif

#if NOP_MTC0
static void nativeMTC0()
{
	switch (_Rd_)
	{
		case 9:
			cpuRegs.lastCOP0Cycle = cpuRegs.cycle;
			cpuRegs.CP0.r[9] = cpuRegs.GPR.r[_Rt_].UL[0];
			break;
		case 12:
			WriteCP0Status(cpuRegs.GPR.r[_Rt_].UL[0]);
			break;
		case 16:
			WriteCP0Config(cpuRegs.GPR.r[_Rt_].UL[0]);
			break;
		case 24:
			break; // debug breakpoint regs — no-op
		case 25:
			if (0 == (_Imm_ & 1))
			{
				if (0 != (_Imm_ & 0x3E)) break;
				COP0_UpdatePCCR();
				// Only bits 1-9, 11-19, and 31 are writable
				cpuRegs.PERF.n.pccr.val = cpuRegs.GPR.r[_Rt_].UL[0] & 0x800FFBFE;
				COP0_DiagnosticPCCR();
			}
			else if (0 == (_Imm_ & 2))
			{
				cpuRegs.PERF.n.pcr0 = cpuRegs.GPR.r[_Rt_].UL[0];
				cpuRegs.lastPERFCycle[0] = cpuRegs.cycle;
			}
			else
			{
				cpuRegs.PERF.n.pcr1 = cpuRegs.GPR.r[_Rt_].UL[0];
				cpuRegs.lastPERFCycle[1] = cpuRegs.cycle;
			}
			break;
		default:
			cpuRegs.CP0.r[_Rd_] = cpuRegs.GPR.r[_Rt_].UL[0];
			break;
	}
}
#endif

// BC0 branches — condition is DMA CPC
static inline int CPCOND0()
{
	return (((dmacRegs.stat.CIS | ~dmacRegs.pcr.CPC) & 0x3FF) == 0x3ff);
}

#if NOP_BC0F
static void nativeBC0F()
{
	if (CPCOND0() == 0) intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BC0T
static void nativeBC0T()
{
	if (CPCOND0() == 1) intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BC0FL
static void nativeBC0FL()
{
	if (CPCOND0() == 0) intDoBranch(_BranchTarget_);
	else cpuRegs.pc += 4;
}
#endif

#if NOP_BC0TL
static void nativeBC0TL()
{
	if (CPCOND0() == 1) intDoBranch(_BranchTarget_);
	else cpuRegs.pc += 4;
}
#endif

// TLB ops — complex, delegate to interpreter implementations
#if NOP_TLBR
static void nativeTLBR()  { R5900::Interpreter::OpcodeImpl::COP0::TLBR(); }
#endif
#if NOP_TLBWI
static void nativeTLBWI() { R5900::Interpreter::OpcodeImpl::COP0::TLBWI(); }
#endif
#if NOP_TLBWR
static void nativeTLBWR() { R5900::Interpreter::OpcodeImpl::COP0::TLBWR(); }
#endif
#if NOP_TLBP
static void nativeTLBP()  { R5900::Interpreter::OpcodeImpl::COP0::TLBP(); }
#endif

#if NOP_ERET
static void nativeERET()
{
	if (cpuRegs.CP0.n.Status.b.ERL)
	{
		cpuRegs.pc = cpuRegs.CP0.n.ErrorEPC;
		cpuRegs.CP0.n.Status.b.ERL = 0;
	}
	else
	{
		cpuRegs.pc = cpuRegs.CP0.n.EPC;
		cpuRegs.CP0.n.Status.b.EXL = 0;
	}
	cpuUpdateOperationMode();
	cpuSetNextEventDelta(4);
	intSetBranch();
}
#endif

#if NOP_EI
static void nativeEI()
{
	if (cpuRegs.CP0.n.Status.b._EDI || cpuRegs.CP0.n.Status.b.EXL ||
		cpuRegs.CP0.n.Status.b.ERL || (cpuRegs.CP0.n.Status.b.KSU == 0))
	{
		cpuRegs.CP0.n.Status.b.EIE = 1;
		cpuSetNextEventDelta(4);
	}
}
#endif

#if NOP_DI
static void nativeDI()
{
	if (cpuRegs.CP0.n.Status.b._EDI || cpuRegs.CP0.n.Status.b.EXL ||
		cpuRegs.CP0.n.Status.b.ERL || (cpuRegs.CP0.n.Status.b.KSU == 0))
	{
		cpuRegs.CP0.n.Status.b.EIE = 0;
	}
}
#endif

// ============================================================================
//  COP1 (FPU) native implementations
// ============================================================================

// FPU field extraction — local to this file (same as FPU.cpp)
#define _Ft_     ((cpuRegs.code >> 16) & 0x1F)
#define _Fs_     ((cpuRegs.code >> 11) & 0x1F)
#define _Fd_     ((cpuRegs.code >>  6) & 0x1F)

// IEEE 754 constants for PS2 FPU clamping
#define PosInfinity 0x7f800000
#define NegInfinity 0xff800000
#define posFmax     0x7F7FFFFF
#define negFmax     0xFF7FFFFF

// FCR31 flag bits
#define FPUflagC    0X00800000
#define FPUflagI    0X00020000
#define FPUflagD    0X00010000
#define FPUflagO    0X00008000
#define FPUflagU    0X00004000
#define FPUflagSI   0X00000040
#define FPUflagSD   0X00000020
#define FPUflagSO   0X00000010
#define FPUflagSU   0X00000008

// PS2 FPU float conversion: denormals → ±0, infinities → ±Fmax
static inline float fpuDouble(u32 f)
{
	switch (f & 0x7f800000)
	{
		case 0x0:
			f &= 0x80000000;
			return *(float*)&f;
		case 0x7f800000:
			f = (f & 0x80000000) | 0x7f7fffff;
			return *(float*)&f;
		default:
			return *(float*)&f;
	}
}

static inline bool checkOverflow(u32& xReg, u32 cFlagsToSet)
{
	if ((xReg & ~0x80000000) == PosInfinity)
	{
		xReg = (xReg & 0x80000000) | posFmax;
		fpuRegs.fprc[31] |= cFlagsToSet;
		return true;
	}
	else if (cFlagsToSet & FPUflagO)
		fpuRegs.fprc[31] &= ~FPUflagO;
	return false;
}

static inline bool checkUnderflow(u32& xReg, u32 cFlagsToSet)
{
	if (((xReg & 0x7F800000) == 0) && ((xReg & 0x007FFFFF) != 0))
	{
		xReg &= 0x80000000;
		fpuRegs.fprc[31] |= cFlagsToSet;
		return true;
	}
	else if (cFlagsToSet & FPUflagU)
		fpuRegs.fprc[31] &= ~FPUflagU;
	return false;
}

static inline bool checkDivideByZero(u32& xReg, u32 yDivisorReg, u32 zDividendReg, u32 cFlagsToSet1, u32 cFlagsToSet2)
{
	if ((yDivisorReg & 0x7F800000) == 0)
	{
		fpuRegs.fprc[31] |= ((zDividendReg & 0x7F800000) == 0) ? cFlagsToSet2 : cFlagsToSet1;
		xReg = ((yDivisorReg ^ zDividendReg) & 0x80000000) | posFmax;
		return true;
	}
	return false;
}

static inline u32 fp_max(u32 a, u32 b)
{
	return ((s32)a < 0 && (s32)b < 0) ? std::min<s32>(a, b) : std::max<s32>(a, b);
}

static inline u32 fp_min(u32 a, u32 b)
{
	return ((s32)a < 0 && (s32)b < 0) ? std::max<s32>(a, b) : std::min<s32>(a, b);
}

// --- COP1 move/control ops ---

#if NOP_MFC1
static void nativeMFC1()
{
	if (!_Rt_) return;
	cpuRegs.GPR.r[_Rt_].SD[0] = fpuRegs.fpr[_Fs_].SL;
}
#endif

#if NOP_CFC1
static void nativeCFC1()
{
	if (!_Rt_) return;
	if (_Fs_ == 31)
		cpuRegs.GPR.r[_Rt_].SD[0] = (s32)fpuRegs.fprc[31];
	else if (_Fs_ == 0)
		cpuRegs.GPR.r[_Rt_].SD[0] = 0x2E00;
	else
		cpuRegs.GPR.r[_Rt_].SD[0] = 0;
}
#endif

#if NOP_MTC1
static void nativeMTC1()
{
	fpuRegs.fpr[_Fs_].UL = cpuRegs.GPR.r[_Rt_].UL[0];
}
#endif

#if NOP_CTC1
static void nativeCTC1()
{
	if (_Fs_ != 31) return;
	fpuRegs.fprc[_Fs_] = cpuRegs.GPR.r[_Rt_].UL[0];
}
#endif

// --- BC1 branches ---

#if NOP_BC1F
static void nativeBC1F()
{
	if ((fpuRegs.fprc[31] & FPUflagC) == 0) intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BC1T
static void nativeBC1T()
{
	if ((fpuRegs.fprc[31] & FPUflagC) != 0) intDoBranch(_BranchTarget_);
}
#endif

#if NOP_BC1FL
static void nativeBC1FL()
{
	if ((fpuRegs.fprc[31] & FPUflagC) == 0) intDoBranch(_BranchTarget_);
	else cpuRegs.pc += 4;
}
#endif

#if NOP_BC1TL
static void nativeBC1TL()
{
	if ((fpuRegs.fprc[31] & FPUflagC) != 0) intDoBranch(_BranchTarget_);
	else cpuRegs.pc += 4;
}
#endif

// --- S sub-table: single-precision FPU ops ---

#if NOP_ADD_S
static void nativeADD_S()
{
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) + fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_SUB_S
static void nativeSUB_S()
{
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) - fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MUL_S
static void nativeMUL_S()
{
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_DIV_S
static void nativeDIV_S()
{
	if (checkDivideByZero(fpuRegs.fpr[_Fd_].UL, fpuRegs.fpr[_Ft_].UL, fpuRegs.fpr[_Fs_].UL, FPUflagD | FPUflagSD, FPUflagI | FPUflagSI)) return;
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) / fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, 0)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, 0);
}
#endif

#if NOP_SQRT_S
static void nativeSQRT_S()
{
	fpuRegs.fprc[31] &= ~(FPUflagI | FPUflagD);
	if ((fpuRegs.fpr[_Ft_].UL & 0x7F800000) == 0)
		fpuRegs.fpr[_Fd_].UL = fpuRegs.fpr[_Ft_].UL & 0x80000000;
	else if (fpuRegs.fpr[_Ft_].UL & 0x80000000)
	{
		fpuRegs.fprc[31] |= FPUflagI | FPUflagSI;
		fpuRegs.fpr[_Fd_].f = sqrt(fabs(fpuDouble(fpuRegs.fpr[_Ft_].UL)));
	}
	else
		fpuRegs.fpr[_Fd_].f = sqrt(fpuDouble(fpuRegs.fpr[_Ft_].UL));
}
#endif

#if NOP_ABS_S
static void nativeABS_S()
{
	fpuRegs.fpr[_Fd_].UL = fpuRegs.fpr[_Fs_].UL & 0x7fffffff;
	fpuRegs.fprc[31] &= ~(FPUflagO | FPUflagU);
}
#endif

#if NOP_MOV_S
static void nativeMOV_S()
{
	fpuRegs.fpr[_Fd_].UL = fpuRegs.fpr[_Fs_].UL;
}
#endif

#if NOP_NEG_S
static void nativeNEG_S()
{
	fpuRegs.fpr[_Fd_].UL = fpuRegs.fpr[_Fs_].UL ^ 0x80000000;
	fpuRegs.fprc[31] &= ~(FPUflagO | FPUflagU);
}
#endif

#if NOP_RSQRT_S
static void nativeRSQRT_S()
{
	fpuRegs.fprc[31] &= ~(FPUflagD | FPUflagI);
	if ((fpuRegs.fpr[_Ft_].UL & 0x7F800000) == 0)
	{
		fpuRegs.fprc[31] |= FPUflagD | FPUflagSD;
		fpuRegs.fpr[_Fd_].UL = (fpuRegs.fpr[_Ft_].UL & 0x80000000) | posFmax;
		return;
	}
	else if (fpuRegs.fpr[_Ft_].UL & 0x80000000)
	{
		fpuRegs.fprc[31] |= FPUflagI | FPUflagSI;
		FPRreg temp;
		temp.f = sqrt(fabs(fpuDouble(fpuRegs.fpr[_Ft_].UL)));
		fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) / fpuDouble(temp.UL);
	}
	else
	{
		fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.fpr[_Fs_].UL) / sqrt(fpuDouble(fpuRegs.fpr[_Ft_].UL));
	}
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, 0)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, 0);
}
#endif

#if NOP_ADDA_S
static void nativeADDA_S()
{
	fpuRegs.ACC.f = fpuDouble(fpuRegs.fpr[_Fs_].UL) + fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.ACC.UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.ACC.UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_SUBA_S
static void nativeSUBA_S()
{
	fpuRegs.ACC.f = fpuDouble(fpuRegs.fpr[_Fs_].UL) - fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.ACC.UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.ACC.UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MULA_S
static void nativeMULA_S()
{
	fpuRegs.ACC.f = fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.ACC.UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.ACC.UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MADD_S
static void nativeMADD_S()
{
	FPRreg temp;
	temp.f = fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.ACC.UL) + fpuDouble(temp.UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MSUB_S
static void nativeMSUB_S()
{
	FPRreg temp;
	temp.f = fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	fpuRegs.fpr[_Fd_].f = fpuDouble(fpuRegs.ACC.UL) - fpuDouble(temp.UL);
	if (checkOverflow(fpuRegs.fpr[_Fd_].UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.fpr[_Fd_].UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MADDA_S
static void nativeMADDA_S()
{
	fpuRegs.ACC.f += fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.ACC.UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.ACC.UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_MSUBA_S
static void nativeMSUBA_S()
{
	fpuRegs.ACC.f -= fpuDouble(fpuRegs.fpr[_Fs_].UL) * fpuDouble(fpuRegs.fpr[_Ft_].UL);
	if (checkOverflow(fpuRegs.ACC.UL, FPUflagO | FPUflagSO)) return;
	checkUnderflow(fpuRegs.ACC.UL, FPUflagU | FPUflagSU);
}
#endif

#if NOP_CVT_W
static void nativeCVT_W()
{
	if ((fpuRegs.fpr[_Fs_].UL & 0x7F800000) <= 0x4E800000)
		fpuRegs.fpr[_Fd_].SL = (s32)fpuRegs.fpr[_Fs_].f;
	else if ((fpuRegs.fpr[_Fs_].UL & 0x80000000) == 0)
		fpuRegs.fpr[_Fd_].UL = 0x7fffffff;
	else
		fpuRegs.fpr[_Fd_].UL = 0x80000000;
}
#endif

#if NOP_MAX_S
static void nativeMAX_S()
{
	fpuRegs.fpr[_Fd_].UL = fp_max(fpuRegs.fpr[_Fs_].UL, fpuRegs.fpr[_Ft_].UL);
	fpuRegs.fprc[31] &= ~(FPUflagO | FPUflagU);
}
#endif

#if NOP_MIN_S
static void nativeMIN_S()
{
	fpuRegs.fpr[_Fd_].UL = fp_min(fpuRegs.fpr[_Fs_].UL, fpuRegs.fpr[_Ft_].UL);
	fpuRegs.fprc[31] &= ~(FPUflagO | FPUflagU);
}
#endif

// Compare ops — use fpuDouble for PS2-accurate comparison
#if NOP_C_F
static void nativeC_F()
{
	fpuRegs.fprc[31] &= ~FPUflagC;
}
#endif

#if NOP_C_EQ
static void nativeC_EQ()
{
	fpuRegs.fprc[31] = (fpuDouble(fpuRegs.fpr[_Fs_].UL) == fpuDouble(fpuRegs.fpr[_Ft_].UL))
		? (fpuRegs.fprc[31] | FPUflagC)
		: (fpuRegs.fprc[31] & ~FPUflagC);
}
#endif

#if NOP_C_LT
static void nativeC_LT()
{
	fpuRegs.fprc[31] = (fpuDouble(fpuRegs.fpr[_Fs_].UL) < fpuDouble(fpuRegs.fpr[_Ft_].UL))
		? (fpuRegs.fprc[31] | FPUflagC)
		: (fpuRegs.fprc[31] & ~FPUflagC);
}
#endif

#if NOP_C_LE
static void nativeC_LE()
{
	fpuRegs.fprc[31] = (fpuDouble(fpuRegs.fpr[_Fs_].UL) <= fpuDouble(fpuRegs.fpr[_Ft_].UL))
		? (fpuRegs.fprc[31] | FPUflagC)
		: (fpuRegs.fprc[31] & ~FPUflagC);
}
#endif

// W sub-table
#if NOP_CVT_S
static void nativeCVT_S()
{
	fpuRegs.fpr[_Fd_].f = (float)fpuRegs.fpr[_Fs_].SL;
}
#endif

// ============================================================================
//  SPECIAL table dispatch (opcode == 0, keyed on funct field bits [5:0])
// ============================================================================

static bool trySpecial()
{
	switch (_Funct_)
	{
#if NOP_SLL
		case 0x00: nativeSLL(); return true;   // SLL
#endif
#if NOP_SRL
		case 0x02: nativeSRL(); return true;   // SRL
#endif
#if NOP_SRA
		case 0x03: nativeSRA(); return true;   // SRA
#endif
#if NOP_SLLV
		case 0x04: nativeSLLV(); return true;  // SLLV
#endif
#if NOP_SRLV
		case 0x06: nativeSRLV(); return true;  // SRLV
#endif
#if NOP_SRAV
		case 0x07: nativeSRAV(); return true;  // SRAV
#endif
#if NOP_JR
		case 0x08: nativeJR(); return true;    // JR
#endif
#if NOP_JALR
		case 0x09: nativeJALR(); return true;  // JALR
#endif
#if NOP_MOVZ
		case 0x0A: nativeMOVZ(); return true;  // MOVZ
#endif
#if NOP_MOVN
		case 0x0B: nativeMOVN(); return true;  // MOVN
#endif
#if NOP_SYSCALL
		case 0x0C: nativeSYSCALL(); return true;  // SYSCALL
#endif
#if NOP_BREAK
		case 0x0D: nativeBREAK(); return true;    // BREAK
#endif
#if NOP_MFHI
		case 0x10: nativeMFHI(); return true;  // MFHI
#endif
#if NOP_MTHI
		case 0x11: nativeMTHI(); return true;  // MTHI
#endif
#if NOP_MFLO
		case 0x12: nativeMFLO(); return true;  // MFLO
#endif
#if NOP_MTLO
		case 0x13: nativeMTLO(); return true;  // MTLO
#endif
#if NOP_DSLLV
		case 0x14: nativeDSLLV(); return true; // DSLLV
#endif
#if NOP_DSRLV
		case 0x16: nativeDSRLV(); return true; // DSRLV
#endif
#if NOP_DSRAV
		case 0x17: nativeDSRAV(); return true; // DSRAV
#endif
#if NOP_ADD
		case 0x20: nativeADD(); return true;   // ADD
#endif
#if NOP_ADDU
		case 0x21: nativeADDU(); return true;  // ADDU
#endif
#if NOP_SUB
		case 0x22: nativeSUB(); return true;   // SUB
#endif
#if NOP_SUBU
		case 0x23: nativeSUBU(); return true;  // SUBU
#endif
#if NOP_AND
		case 0x24: nativeAND(); return true;   // AND
#endif
#if NOP_OR
		case 0x25: nativeOR(); return true;    // OR
#endif
#if NOP_XOR
		case 0x26: nativeXOR(); return true;   // XOR
#endif
#if NOP_NOR
		case 0x27: nativeNOR(); return true;   // NOR
#endif
#if NOP_MFSA
		case 0x28: nativeMFSA(); return true;  // MFSA
#endif
#if NOP_MTSA
		case 0x29: nativeMTSA(); return true;  // MTSA
#endif
#if NOP_SLT
		case 0x2A: nativeSLT(); return true;   // SLT
#endif
#if NOP_SLTU
		case 0x2B: nativeSLTU(); return true;  // SLTU
#endif
#if NOP_DADD
		case 0x2C: nativeDADD(); return true;  // DADD
#endif
#if NOP_DADDU
		case 0x2D: nativeDADDU(); return true; // DADDU
#endif
#if NOP_DSUB
		case 0x2E: nativeDSUB(); return true;  // DSUB
#endif
#if NOP_DSUBU
		case 0x2F: nativeDSUBU(); return true; // DSUBU
#endif
#if NOP_DSLL
		case 0x38: nativeDSLL(); return true;  // DSLL
#endif
#if NOP_DSRL
		case 0x3A: nativeDSRL(); return true;  // DSRL
#endif
#if NOP_DSRA
		case 0x3B: nativeDSRA(); return true;  // DSRA
#endif
#if NOP_DSLL32
		case 0x3C: nativeDSLL32(); return true; // DSLL32
#endif
#if NOP_DSRL32
		case 0x3E: nativeDSRL32(); return true; // DSRL32
#endif
#if NOP_DSRA32
		case 0x3F: nativeDSRA32(); return true; // DSRA32
#endif
		default: return false;
	}
}

// ============================================================================
//  REGIMM table dispatch (opcode == 1, keyed on rt field bits [20:16])
// ============================================================================

static bool tryRegImm()
{
	switch (_Rt_)
	{
#if NOP_BLTZ
		case 0x00: nativeBLTZ(); return true;     // BLTZ
#endif
#if NOP_BGEZ
		case 0x01: nativeBGEZ(); return true;     // BGEZ
#endif
#if NOP_BLTZL
		case 0x02: nativeBLTZL(); return true;    // BLTZL
#endif
#if NOP_BGEZL
		case 0x03: nativeBGEZL(); return true;    // BGEZL
#endif
#if NOP_BLTZAL
		case 0x10: nativeBLTZAL(); return true;   // BLTZAL
#endif
#if NOP_BGEZAL
		case 0x11: nativeBGEZAL(); return true;   // BGEZAL
#endif
#if NOP_BLTZALL
		case 0x12: nativeBLTZALL(); return true;  // BLTZALL
#endif
#if NOP_BGEZALL
		case 0x13: nativeBGEZALL(); return true;  // BGEZALL
#endif
#if NOP_MTSAB
		case 0x18: nativeMTSAB(); return true;    // MTSAB
#endif
#if NOP_MTSAH
		case 0x19: nativeMTSAH(); return true;    // MTSAH
#endif
		default: return false;
	}
}

// ============================================================================
//  MMI native implementations (opcode 0x1C)
// ============================================================================

// ---- tbl_MMI direct ops (keyed on funct field) ----

#if NOP_MADD
static void nativeMADD()
{
	s64 temp = (s64)((u64)cpuRegs.LO.UL[0] | ((u64)cpuRegs.HI.UL[0] << 32)) +
			  ((s64)cpuRegs.GPR.r[_Rs_].SL[0] * (s64)cpuRegs.GPR.r[_Rt_].SL[0]);
	cpuRegs.LO.SD[0] = (s32)(temp & 0xffffffff);
	cpuRegs.HI.SD[0] = (s32)(temp >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.LO.SD[0];
}
#endif

#if NOP_MADDU
static void nativeMADDU()
{
	u64 tempu = (u64)((u64)cpuRegs.LO.UL[0] | ((u64)cpuRegs.HI.UL[0] << 32)) +
			   ((u64)cpuRegs.GPR.r[_Rs_].UL[0] * (u64)cpuRegs.GPR.r[_Rt_].UL[0]);
	cpuRegs.LO.SD[0] = (s32)(tempu & 0xffffffff);
	cpuRegs.HI.SD[0] = (s32)(tempu >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.LO.SD[0];
}
#endif

#if NOP_PLZCW
static void nativePLZCW()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UL[0] = Common::CountLeadingSignBits(cpuRegs.GPR.r[_Rs_].SL[0]) - 1;
	cpuRegs.GPR.r[_Rd_].UL[1] = Common::CountLeadingSignBits(cpuRegs.GPR.r[_Rs_].SL[1]) - 1;
}
#endif

#if NOP_MULT1
static void nativeMULT1()
{
	s64 temp = (s64)cpuRegs.GPR.r[_Rs_].SL[0] * cpuRegs.GPR.r[_Rt_].SL[0];
	cpuRegs.LO.SD[1] = (s32)(temp & 0xffffffff);
	cpuRegs.HI.SD[1] = (s32)(temp >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.LO.UD[1];
}
#endif

#if NOP_MULTU1
static void nativeMULTU1()
{
	u64 tempu = (u64)cpuRegs.GPR.r[_Rs_].UL[0] * cpuRegs.GPR.r[_Rt_].UL[0];
	cpuRegs.LO.SD[1] = (s32)(tempu & 0xffffffff);
	cpuRegs.HI.SD[1] = (s32)(tempu >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.LO.UD[1];
}
#endif

#if NOP_DIV1
static void nativeDIV1()
{
	if (cpuRegs.GPR.r[_Rs_].UL[0] == 0x80000000 && cpuRegs.GPR.r[_Rt_].UL[0] == 0xffffffff)
	{
		cpuRegs.LO.SD[1] = (s32)0x80000000;
		cpuRegs.HI.SD[1] = (s32)0x0;
	}
	else if (cpuRegs.GPR.r[_Rt_].SL[0] != 0)
	{
		cpuRegs.LO.SD[1] = cpuRegs.GPR.r[_Rs_].SL[0] / cpuRegs.GPR.r[_Rt_].SL[0];
		cpuRegs.HI.SD[1] = cpuRegs.GPR.r[_Rs_].SL[0] % cpuRegs.GPR.r[_Rt_].SL[0];
	}
	else
	{
		cpuRegs.LO.SD[1] = (cpuRegs.GPR.r[_Rs_].SL[0] < 0) ? 1 : -1;
		cpuRegs.HI.SD[1] = cpuRegs.GPR.r[_Rs_].SL[0];
	}
}
#endif

#if NOP_DIVU1
static void nativeDIVU1()
{
	if (cpuRegs.GPR.r[_Rt_].UL[0] != 0)
	{
		cpuRegs.LO.SD[1] = (s32)(cpuRegs.GPR.r[_Rs_].UL[0] / cpuRegs.GPR.r[_Rt_].UL[0]);
		cpuRegs.HI.SD[1] = (s32)(cpuRegs.GPR.r[_Rs_].UL[0] % cpuRegs.GPR.r[_Rt_].UL[0]);
	}
	else
	{
		cpuRegs.LO.SD[1] = -1;
		cpuRegs.HI.SD[1] = cpuRegs.GPR.r[_Rs_].SL[0];
	}
}
#endif

#if NOP_MADD1
static void nativeMADD1()
{
	s64 temp = (s64)((u64)cpuRegs.LO.UL[2] | ((u64)cpuRegs.HI.UL[2] << 32)) +
			  ((s64)cpuRegs.GPR.r[_Rs_].SL[0] * (s64)cpuRegs.GPR.r[_Rt_].SL[0]);
	cpuRegs.LO.SD[1] = (s32)(temp & 0xffffffff);
	cpuRegs.HI.SD[1] = (s32)(temp >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.LO.SD[1];
}
#endif

#if NOP_MADDU1
static void nativeMADDU1()
{
	u64 tempu = (u64)((u64)cpuRegs.LO.UL[2] | ((u64)cpuRegs.HI.UL[2] << 32)) +
			   ((u64)cpuRegs.GPR.r[_Rs_].UL[0] * (u64)cpuRegs.GPR.r[_Rt_].UL[0]);
	cpuRegs.LO.SD[1] = (s32)(tempu & 0xffffffff);
	cpuRegs.HI.SD[1] = (s32)(tempu >> 32);
	if (_Rd_) cpuRegs.GPR.r[_Rd_].SD[0] = cpuRegs.LO.SD[1];
}
#endif

#if NOP_PMFHL
static inline void nativePMFHL_CLAMP(u16& dst, s32 src)
{
	if      (src >  0x7fff) dst = 0x7fff;
	else if (src < -0x8000) dst = 0x8000;
	else                    dst = (u16)src;
}

static void nativePMFHL()
{
	if (!_Rd_) return;
	switch (_Sa_)
	{
		case 0x00: // LW — [LO[0], HI[0], LO[2], HI[2]]
		{
			uint32x4_t lo = vld1q_u32(cpuRegs.LO.UL);
			uint32x4_t hi = vld1q_u32(cpuRegs.HI.UL);
			vst1q_u32(cpuRegs.GPR.r[_Rd_].UL, vtrn1q_u32(lo, hi));
			break;
		}
		case 0x01: // UW — [LO[1], HI[1], LO[3], HI[3]]
		{
			uint32x4_t lo = vld1q_u32(cpuRegs.LO.UL);
			uint32x4_t hi = vld1q_u32(cpuRegs.HI.UL);
			vst1q_u32(cpuRegs.GPR.r[_Rd_].UL, vtrn2q_u32(lo, hi));
			break;
		}
		case 0x02: // SLW — saturate 64-bit pairs to s32, sign-extend to s64
		{
			uint32x4_t lo = vld1q_u32(cpuRegs.LO.UL);
			uint32x4_t hi = vld1q_u32(cpuRegs.HI.UL);
			// LE: int64[n] = HI[n*2]<<32 | LO[n*2]
			int64x2_t vals = vreinterpretq_s64_u32(vtrn1q_u32(lo, hi));
			vst1q_s64((int64_t*)cpuRegs.GPR.r[_Rd_].UD, vmovl_s32(vqmovn_s64(vals)));
			break;
		}
		case 0x03: // LH — even halfwords from LO and HI, interleaved in pairs
		{
			uint16x8_t lo16 = vld1q_u16(cpuRegs.LO.US);
			uint16x8_t hi16 = vld1q_u16(cpuRegs.HI.US);
			// [LO[0],LO[2],LO[4],LO[6]] and [HI[0],HI[2],HI[4],HI[6]]
			uint16x4_t lo_even = vuzp1_u16(vget_low_u16(lo16), vget_high_u16(lo16));
			uint16x4_t hi_even = vuzp1_u16(vget_low_u16(hi16), vget_high_u16(hi16));
			// Interleave in 32-bit pairs: [lo_pair0, hi_pair0, lo_pair1, hi_pair1]
			uint32x2_t lo_u32 = vreinterpret_u32_u16(lo_even);
			uint32x2_t hi_u32 = vreinterpret_u32_u16(hi_even);
			vst1q_u16(cpuRegs.GPR.r[_Rd_].US, vreinterpretq_u16_u32(
					vzip1q_u32(vcombine_u32(lo_u32, lo_u32),
							   vcombine_u32(hi_u32, hi_u32))));
			break;
		}
		case 0x04: // SH — saturate 8 words to s16, interleaved in pairs
		{
			int32x4_t lo = vld1q_s32((const int32_t*)cpuRegs.LO.UL);
			int32x4_t hi = vld1q_s32((const int32_t*)cpuRegs.HI.UL);
			int16x4_t lo_sat = vqmovn_s32(lo);
			int16x4_t hi_sat = vqmovn_s32(hi);
			uint32x2_t lo_u32 = vreinterpret_u32_s16(lo_sat);
			uint32x2_t hi_u32 = vreinterpret_u32_s16(hi_sat);
			vst1q_u16(cpuRegs.GPR.r[_Rd_].US, vreinterpretq_u16_u32(
					vzip1q_u32(vcombine_u32(lo_u32, lo_u32),
							   vcombine_u32(hi_u32, hi_u32))));
			break;
		}
	}
}

#endif

#if NOP_PMTHL
static void nativePMTHL()
{
	if (_Sa_ != 0) return;
	cpuRegs.LO.UL[0] = cpuRegs.GPR.r[_Rs_].UL[0];
	cpuRegs.HI.UL[0] = cpuRegs.GPR.r[_Rs_].UL[1];
	cpuRegs.LO.UL[2] = cpuRegs.GPR.r[_Rs_].UL[2];
	cpuRegs.HI.UL[2] = cpuRegs.GPR.r[_Rs_].UL[3];
}
#endif

#if NOP_PSLLH
static void nativePSLLH()
{
	if (!_Rd_) return;
	const int sa = _Sa_ & 0xf;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rt_].US[i] << sa;
}
#endif

#if NOP_PSRLH
static void nativePSRLH()
{
	if (!_Rd_) return;
	const int sa = _Sa_ & 0xf;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rt_].US[i] >> sa;
}
#endif

#if NOP_PSRAH
static void nativePSRAH()
{
	if (!_Rd_) return;
	const int sa = _Sa_ & 0xf;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rt_].SS[i] >> sa;
}
#endif

#if NOP_PSLLW
static void nativePSLLW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = cpuRegs.GPR.r[_Rt_].UL[i] << _Sa_;
}
#endif

#if NOP_PSRLW
static void nativePSRLW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = cpuRegs.GPR.r[_Rt_].UL[i] >> _Sa_;
}
#endif

#if NOP_PSRAW
static void nativePSRAW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = cpuRegs.GPR.r[_Rt_].SL[i] >> _Sa_;
}
#endif

// ---- tbl_MMI0 ops (funct==0x08, keyed on sa field) ----

#if NOP_PADDW
static void nativePADDW()
{
	if (!_Rd_) return;
	vst1q_u32(cpuRegs.GPR.r[_Rd_].UL,
			  vaddq_u32(vld1q_u32(cpuRegs.GPR.r[_Rs_].UL),
						vld1q_u32(cpuRegs.GPR.r[_Rt_].UL)));
}
#endif

#if NOP_PSUBW
static void nativePSUBW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = cpuRegs.GPR.r[_Rs_].UL[i] - cpuRegs.GPR.r[_Rt_].UL[i];
}
#endif

#if NOP_PCGTW
static void nativePCGTW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = (cpuRegs.GPR.r[_Rs_].SL[i] > cpuRegs.GPR.r[_Rt_].SL[i]) ? 0xFFFFFFFF : 0;
}
#endif

#if NOP_PMAXW
static void nativePMAXW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = (cpuRegs.GPR.r[_Rs_].SL[i] > cpuRegs.GPR.r[_Rt_].SL[i]) ?
			cpuRegs.GPR.r[_Rs_].UL[i] : cpuRegs.GPR.r[_Rt_].UL[i];
}
#endif

#if NOP_PADDH
static void nativePADDH()
{
	if (!_Rd_) return;
	vst1q_u16(cpuRegs.GPR.r[_Rd_].US,
			  vaddq_u16(vld1q_u16(cpuRegs.GPR.r[_Rs_].US),
						vld1q_u16(cpuRegs.GPR.r[_Rt_].US)));
}
#endif

#if NOP_PSUBH
static void nativePSUBH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rs_].US[i] - cpuRegs.GPR.r[_Rt_].US[i];
}
#endif

#if NOP_PCGTH
static void nativePCGTH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = (cpuRegs.GPR.r[_Rs_].SS[i] > cpuRegs.GPR.r[_Rt_].SS[i]) ? 0xFFFF : 0;
}
#endif

#if NOP_PMAXH
static void nativePMAXH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = (cpuRegs.GPR.r[_Rs_].SS[i] > cpuRegs.GPR.r[_Rt_].SS[i]) ?
			cpuRegs.GPR.r[_Rs_].US[i] : cpuRegs.GPR.r[_Rt_].US[i];
}
#endif

#if NOP_PADDB
static void nativePADDB()
{
	if (!_Rd_) return;
	vst1q_s8(cpuRegs.GPR.r[_Rd_].SC,
			 vaddq_s8(vld1q_s8(cpuRegs.GPR.r[_Rs_].SC),
					  vld1q_s8(cpuRegs.GPR.r[_Rt_].SC)));
}
#endif

#if NOP_PSUBB
static void nativePSUBB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
		cpuRegs.GPR.r[_Rd_].SC[i] = cpuRegs.GPR.r[_Rs_].SC[i] - cpuRegs.GPR.r[_Rt_].SC[i];
}
#endif

#if NOP_PCGTB
static void nativePCGTB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
		cpuRegs.GPR.r[_Rd_].UC[i] = (cpuRegs.GPR.r[_Rs_].SC[i] > cpuRegs.GPR.r[_Rt_].SC[i]) ? 0xFF : 0x00;
}
#endif

#if NOP_PADDSW
static void nativePADDSW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
	{
		s64 tmp = (s64)cpuRegs.GPR.r[_Rs_].SL[i] + (s64)cpuRegs.GPR.r[_Rt_].SL[i];
		if (tmp > 0x7FFFFFFF) cpuRegs.GPR.r[_Rd_].UL[i] = 0x7FFFFFFF;
		else if (tmp < (s32)0x80000000) cpuRegs.GPR.r[_Rd_].UL[i] = 0x80000000;
		else cpuRegs.GPR.r[_Rd_].UL[i] = (s32)tmp;
	}
}
#endif

#if NOP_PSUBSW
static void nativePSUBSW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
	{
		s64 tmp = (s64)cpuRegs.GPR.r[_Rs_].SL[i] - (s64)cpuRegs.GPR.r[_Rt_].SL[i];
		if (tmp >= 0x7FFFFFFF) cpuRegs.GPR.r[_Rd_].UL[i] = 0x7FFFFFFF;
		else if (tmp < (s32)0x80000000) cpuRegs.GPR.r[_Rd_].UL[i] = 0x80000000;
		else cpuRegs.GPR.r[_Rd_].UL[i] = (s32)tmp;
	}
}
#endif

#if NOP_PEXTLW
static void nativePEXTLW()
{
	if (!_Rd_) return;
	uint32x4_t rs = vld1q_u32(cpuRegs.GPR.r[_Rs_].UL);
	uint32x4_t rt = vld1q_u32(cpuRegs.GPR.r[_Rt_].UL);
	// vzip1q_u32(rt, rs) = [rt[0], rs[0], rt[1], rs[1]]
	vst1q_u32(cpuRegs.GPR.r[_Rd_].UL, vzip1q_u32(rt, rs));
}
#endif

#if NOP_PPACW
static void nativePPACW()
{
	if (!_Rd_) return;
	uint32x4_t rs = vld1q_u32(cpuRegs.GPR.r[_Rs_].UL);
	uint32x4_t rt = vld1q_u32(cpuRegs.GPR.r[_Rt_].UL);
	// vuzp1q_u32(rt, rs) = [rt[0], rt[2], rs[0], rs[2]]
	vst1q_u32(cpuRegs.GPR.r[_Rd_].UL, vuzp1q_u32(rt, rs));
}
#endif

#if NOP_PADDSH
static void nativePADDSH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
	{
		s32 tmp = (s32)cpuRegs.GPR.r[_Rs_].SS[i] + (s32)cpuRegs.GPR.r[_Rt_].SS[i];
		if (tmp > 0x7FFF) cpuRegs.GPR.r[_Rd_].US[i] = 0x7FFF;
		else if (tmp < (s32)0xffff8000) cpuRegs.GPR.r[_Rd_].US[i] = 0x8000;
		else cpuRegs.GPR.r[_Rd_].US[i] = (s16)tmp;
	}
}
#endif

#if NOP_PSUBSH
static void nativePSUBSH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
	{
		s32 tmp = (s32)cpuRegs.GPR.r[_Rs_].SS[i] - (s32)cpuRegs.GPR.r[_Rt_].SS[i];
		if (tmp >= 0x7FFF) cpuRegs.GPR.r[_Rd_].US[i] = 0x7FFF;
		else if (tmp < (s32)0xffff8000) cpuRegs.GPR.r[_Rd_].US[i] = 0x8000;
		else cpuRegs.GPR.r[_Rd_].US[i] = (s16)tmp;
	}
}
#endif

#if NOP_PEXTLH
static void nativePEXTLH()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[1] = Rs.US[0];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[1]; cpuRegs.GPR.r[_Rd_].US[3] = Rs.US[1];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[2]; cpuRegs.GPR.r[_Rd_].US[5] = Rs.US[2];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[3]; cpuRegs.GPR.r[_Rd_].US[7] = Rs.US[3];
}
#endif

#if NOP_PPACH
static void nativePPACH()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[1] = Rt.US[2];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[4]; cpuRegs.GPR.r[_Rd_].US[3] = Rt.US[6];
	cpuRegs.GPR.r[_Rd_].US[4] = Rs.US[0]; cpuRegs.GPR.r[_Rd_].US[5] = Rs.US[2];
	cpuRegs.GPR.r[_Rd_].US[6] = Rs.US[4]; cpuRegs.GPR.r[_Rd_].US[7] = Rs.US[6];
}
#endif

#if NOP_PADDSB
static void nativePADDSB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
	{
		s16 tmp = (s16)cpuRegs.GPR.r[_Rs_].SC[i] + (s16)cpuRegs.GPR.r[_Rt_].SC[i];
		if (tmp > 0x7F) cpuRegs.GPR.r[_Rd_].UC[i] = 0x7F;
		else if (tmp < (s16)-128) cpuRegs.GPR.r[_Rd_].UC[i] = 0x80;
		else cpuRegs.GPR.r[_Rd_].UC[i] = (s8)tmp;
	}
}
#endif

#if NOP_PSUBSB
static void nativePSUBSB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
	{
		s16 tmp = (s16)cpuRegs.GPR.r[_Rs_].SC[i] - (s16)cpuRegs.GPR.r[_Rt_].SC[i];
		if (tmp >= 0x7F) cpuRegs.GPR.r[_Rd_].UC[i] = 0x7F;
		else if (tmp < (s16)-128) cpuRegs.GPR.r[_Rd_].UC[i] = 0x80;
		else cpuRegs.GPR.r[_Rd_].UC[i] = (s8)tmp;
	}
}
#endif

#if NOP_PEXTLB
static void nativePEXTLB()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UC[0]  = Rt.UC[0]; cpuRegs.GPR.r[_Rd_].UC[1]  = Rs.UC[0];
	cpuRegs.GPR.r[_Rd_].UC[2]  = Rt.UC[1]; cpuRegs.GPR.r[_Rd_].UC[3]  = Rs.UC[1];
	cpuRegs.GPR.r[_Rd_].UC[4]  = Rt.UC[2]; cpuRegs.GPR.r[_Rd_].UC[5]  = Rs.UC[2];
	cpuRegs.GPR.r[_Rd_].UC[6]  = Rt.UC[3]; cpuRegs.GPR.r[_Rd_].UC[7]  = Rs.UC[3];
	cpuRegs.GPR.r[_Rd_].UC[8]  = Rt.UC[4]; cpuRegs.GPR.r[_Rd_].UC[9]  = Rs.UC[4];
	cpuRegs.GPR.r[_Rd_].UC[10] = Rt.UC[5]; cpuRegs.GPR.r[_Rd_].UC[11] = Rs.UC[5];
	cpuRegs.GPR.r[_Rd_].UC[12] = Rt.UC[6]; cpuRegs.GPR.r[_Rd_].UC[13] = Rs.UC[6];
	cpuRegs.GPR.r[_Rd_].UC[14] = Rt.UC[7]; cpuRegs.GPR.r[_Rd_].UC[15] = Rs.UC[7];
}
#endif

#if NOP_PPACB
static void nativePPACB()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UC[0]  = Rt.UC[0];  cpuRegs.GPR.r[_Rd_].UC[1]  = Rt.UC[2];
	cpuRegs.GPR.r[_Rd_].UC[2]  = Rt.UC[4];  cpuRegs.GPR.r[_Rd_].UC[3]  = Rt.UC[6];
	cpuRegs.GPR.r[_Rd_].UC[4]  = Rt.UC[8];  cpuRegs.GPR.r[_Rd_].UC[5]  = Rt.UC[10];
	cpuRegs.GPR.r[_Rd_].UC[6]  = Rt.UC[12]; cpuRegs.GPR.r[_Rd_].UC[7]  = Rt.UC[14];
	cpuRegs.GPR.r[_Rd_].UC[8]  = Rs.UC[0];  cpuRegs.GPR.r[_Rd_].UC[9]  = Rs.UC[2];
	cpuRegs.GPR.r[_Rd_].UC[10] = Rs.UC[4];  cpuRegs.GPR.r[_Rd_].UC[11] = Rs.UC[6];
	cpuRegs.GPR.r[_Rd_].UC[12] = Rs.UC[8];  cpuRegs.GPR.r[_Rd_].UC[13] = Rs.UC[10];
	cpuRegs.GPR.r[_Rd_].UC[14] = Rs.UC[12]; cpuRegs.GPR.r[_Rd_].UC[15] = Rs.UC[14];
}
#endif

#if NOP_PEXT5
static void nativePEXT5()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] =
			((cpuRegs.GPR.r[_Rt_].UL[i] & 0x0000001F) <<  3) |
			((cpuRegs.GPR.r[_Rt_].UL[i] & 0x000003E0) <<  6) |
			((cpuRegs.GPR.r[_Rt_].UL[i] & 0x00007C00) <<  9) |
			((cpuRegs.GPR.r[_Rt_].UL[i] & 0x00008000) << 16);
}
#endif

#if NOP_PPAC5
static void nativePPAC5()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] =
			((cpuRegs.GPR.r[_Rt_].UL[i] >>  3) & 0x0000001F) |
			((cpuRegs.GPR.r[_Rt_].UL[i] >>  6) & 0x000003E0) |
			((cpuRegs.GPR.r[_Rt_].UL[i] >>  9) & 0x00007C00) |
			((cpuRegs.GPR.r[_Rt_].UL[i] >> 16) & 0x00008000);
}
#endif

// ---- tbl_MMI1 ops (funct==0x28, keyed on sa field) ----

#if NOP_PABSW
static void nativePABSW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
	{
		if (cpuRegs.GPR.r[_Rt_].UL[i] == 0x80000000)
			cpuRegs.GPR.r[_Rd_].UL[i] = 0x7fffffff;
		else if (cpuRegs.GPR.r[_Rt_].SL[i] < 0)
			cpuRegs.GPR.r[_Rd_].UL[i] = -cpuRegs.GPR.r[_Rt_].SL[i];
		else
			cpuRegs.GPR.r[_Rd_].UL[i] = cpuRegs.GPR.r[_Rt_].SL[i];
	}
}
#endif

#if NOP_PCEQW
static void nativePCEQW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].UL[i] = (cpuRegs.GPR.r[_Rs_].UL[i] == cpuRegs.GPR.r[_Rt_].UL[i]) ? 0xFFFFFFFF : 0;
}
#endif

#if NOP_PMINW
static void nativePMINW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].SL[i] = (cpuRegs.GPR.r[_Rs_].SL[i] < cpuRegs.GPR.r[_Rt_].SL[i]) ?
			cpuRegs.GPR.r[_Rs_].SL[i] : cpuRegs.GPR.r[_Rt_].SL[i];
}
#endif

#if NOP_PADSBH
static void nativePADSBH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rs_].US[i] - cpuRegs.GPR.r[_Rt_].US[i];
	for (int i = 4; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rs_].US[i] + cpuRegs.GPR.r[_Rt_].US[i];
}
#endif

#if NOP_PABSH
static void nativePABSH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
	{
		if (cpuRegs.GPR.r[_Rt_].US[i] == 0x8000)
			cpuRegs.GPR.r[_Rd_].US[i] = 0x7fff;
		else if (cpuRegs.GPR.r[_Rt_].SS[i] < 0)
			cpuRegs.GPR.r[_Rd_].US[i] = -cpuRegs.GPR.r[_Rt_].SS[i];
		else
			cpuRegs.GPR.r[_Rd_].US[i] = cpuRegs.GPR.r[_Rt_].SS[i];
	}
}
#endif

#if NOP_PCEQH
static void nativePCEQH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = (cpuRegs.GPR.r[_Rs_].US[i] == cpuRegs.GPR.r[_Rt_].US[i]) ? 0xFFFF : 0;
}
#endif

#if NOP_PMINH
static void nativePMINH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
		cpuRegs.GPR.r[_Rd_].US[i] = (cpuRegs.GPR.r[_Rs_].SS[i] < cpuRegs.GPR.r[_Rt_].SS[i]) ?
			cpuRegs.GPR.r[_Rs_].US[i] : cpuRegs.GPR.r[_Rt_].US[i];
}
#endif

#if NOP_PCEQB
static void nativePCEQB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
		cpuRegs.GPR.r[_Rd_].UC[i] = (cpuRegs.GPR.r[_Rs_].UC[i] == cpuRegs.GPR.r[_Rt_].UC[i]) ? 0xFF : 0x00;
}
#endif

#if NOP_PADDUW
static void nativePADDUW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
	{
		s64 tmp = (s64)cpuRegs.GPR.r[_Rs_].UL[i] + (s64)cpuRegs.GPR.r[_Rt_].UL[i];
		cpuRegs.GPR.r[_Rd_].UL[i] = (tmp > 0xffffffff) ? 0xffffffff : (u32)tmp;
	}
}
#endif

#if NOP_PSUBUW
static void nativePSUBUW()
{
	if (!_Rd_) return;
	for (int i = 0; i < 4; i++)
	{
		s64 tmp = (s64)cpuRegs.GPR.r[_Rs_].UL[i] - (s64)cpuRegs.GPR.r[_Rt_].UL[i];
		cpuRegs.GPR.r[_Rd_].UL[i] = (tmp <= 0) ? 0 : (u32)tmp;
	}
}
#endif

#if NOP_PEXTUW
static void nativePEXTUW()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UL[0] = Rt.UL[2]; cpuRegs.GPR.r[_Rd_].UL[1] = Rs.UL[2];
	cpuRegs.GPR.r[_Rd_].UL[2] = Rt.UL[3]; cpuRegs.GPR.r[_Rd_].UL[3] = Rs.UL[3];
}
#endif

#if NOP_PADDUH
static void nativePADDUH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
	{
		s32 tmp = (s32)cpuRegs.GPR.r[_Rs_].US[i] + (s32)cpuRegs.GPR.r[_Rt_].US[i];
		cpuRegs.GPR.r[_Rd_].US[i] = (tmp > 0xFFFF) ? 0xFFFF : (u16)tmp;
	}
}
#endif

#if NOP_PSUBUH
static void nativePSUBUH()
{
	if (!_Rd_) return;
	for (int i = 0; i < 8; i++)
	{
		s32 tmp = (s32)cpuRegs.GPR.r[_Rs_].US[i] - (s32)cpuRegs.GPR.r[_Rt_].US[i];
		cpuRegs.GPR.r[_Rd_].US[i] = (tmp <= 0) ? 0 : (u16)tmp;
	}
}
#endif

#if NOP_PEXTUH
static void nativePEXTUH()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[4]; cpuRegs.GPR.r[_Rd_].US[1] = Rs.US[4];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[5]; cpuRegs.GPR.r[_Rd_].US[3] = Rs.US[5];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[6]; cpuRegs.GPR.r[_Rd_].US[5] = Rs.US[6];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[7]; cpuRegs.GPR.r[_Rd_].US[7] = Rs.US[7];
}
#endif

#if NOP_PADDUB
static void nativePADDUB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
	{
		u16 tmp = (u16)cpuRegs.GPR.r[_Rs_].UC[i] + (u16)cpuRegs.GPR.r[_Rt_].UC[i];
		cpuRegs.GPR.r[_Rd_].UC[i] = (tmp > 0xFF) ? 0xFF : (u8)tmp;
	}
}
#endif

#if NOP_PSUBUB
static void nativePSUBUB()
{
	if (!_Rd_) return;
	for (int i = 0; i < 16; i++)
	{
		s16 tmp = (s16)cpuRegs.GPR.r[_Rs_].UC[i] - (s16)cpuRegs.GPR.r[_Rt_].UC[i];
		cpuRegs.GPR.r[_Rd_].UC[i] = (tmp <= 0) ? 0 : (u8)tmp;
	}
}
#endif

#if NOP_PEXTUB
static void nativePEXTUB()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UC[0]  = Rt.UC[8];  cpuRegs.GPR.r[_Rd_].UC[1]  = Rs.UC[8];
	cpuRegs.GPR.r[_Rd_].UC[2]  = Rt.UC[9];  cpuRegs.GPR.r[_Rd_].UC[3]  = Rs.UC[9];
	cpuRegs.GPR.r[_Rd_].UC[4]  = Rt.UC[10]; cpuRegs.GPR.r[_Rd_].UC[5]  = Rs.UC[10];
	cpuRegs.GPR.r[_Rd_].UC[6]  = Rt.UC[11]; cpuRegs.GPR.r[_Rd_].UC[7]  = Rs.UC[11];
	cpuRegs.GPR.r[_Rd_].UC[8]  = Rt.UC[12]; cpuRegs.GPR.r[_Rd_].UC[9]  = Rs.UC[12];
	cpuRegs.GPR.r[_Rd_].UC[10] = Rt.UC[13]; cpuRegs.GPR.r[_Rd_].UC[11] = Rs.UC[13];
	cpuRegs.GPR.r[_Rd_].UC[12] = Rt.UC[14]; cpuRegs.GPR.r[_Rd_].UC[13] = Rs.UC[14];
	cpuRegs.GPR.r[_Rd_].UC[14] = Rt.UC[15]; cpuRegs.GPR.r[_Rd_].UC[15] = Rs.UC[15];
}
#endif

#if NOP_QFSRV
static void nativeQFSRV()
{
	if (!_Rd_) return;
	u32 sa_amt = cpuRegs.sa << 3;
	if (sa_amt == 0) {
		cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.GPR.r[_Rt_].UD[0];
		cpuRegs.GPR.r[_Rd_].UD[1] = cpuRegs.GPR.r[_Rt_].UD[1];
	} else if (sa_amt < 64) {
		GPR_reg Rd;
		Rd.UD[0] = cpuRegs.GPR.r[_Rt_].UD[0] >> sa_amt;
		Rd.UD[1] = cpuRegs.GPR.r[_Rt_].UD[1] >> sa_amt;
		Rd.UD[0]|= cpuRegs.GPR.r[_Rt_].UD[1] << (64 - sa_amt);
		Rd.UD[1]|= cpuRegs.GPR.r[_Rs_].UD[0] << (64 - sa_amt);
		cpuRegs.GPR.r[_Rd_] = Rd;
	} else {
		GPR_reg Rd;
		Rd.UD[0] = cpuRegs.GPR.r[_Rt_].UD[1] >> (sa_amt - 64);
		Rd.UD[1] = cpuRegs.GPR.r[_Rs_].UD[0] >> (sa_amt - 64);
		if (sa_amt != 64) {
			Rd.UD[0]|= cpuRegs.GPR.r[_Rs_].UD[0] << (128u - sa_amt);
			Rd.UD[1]|= cpuRegs.GPR.r[_Rs_].UD[1] << (128u - sa_amt);
		}
		cpuRegs.GPR.r[_Rd_] = Rd;
	}
}
#endif

// ---- tbl_MMI2 ops (funct==0x09, keyed on sa field) ----

#if NOP_PMADDW
static void nativePMADDW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		s64 temp = ((s64)cpuRegs.GPR.r[_Rs_].SL[ss] * (s64)cpuRegs.GPR.r[_Rt_].SL[ss]);
		s64 temp2 = temp + ((s64)cpuRegs.HI.SL[ss] << 32);
		if (ss == 0)
		{
			if (((cpuRegs.GPR.r[_Rt_].SL[ss] & 0x7FFFFFFF) == 0 || (cpuRegs.GPR.r[_Rt_].SL[ss] & 0x7FFFFFFF) == 0x7FFFFFFF) &&
				cpuRegs.GPR.r[_Rs_].SL[ss] != cpuRegs.GPR.r[_Rt_].SL[ss])
				temp2 += 0x70000000;
		}
		temp2 = (s32)(temp2 / 4294967295);
		cpuRegs.LO.SD[dd] = (s32)(temp & 0xffffffff) + cpuRegs.LO.SL[ss];
		cpuRegs.HI.SD[dd] = (s32)temp2;
		if (_Rd_)
		{
			cpuRegs.GPR.r[_Rd_].UL[dd * 2] = cpuRegs.LO.UL[dd * 2];
			cpuRegs.GPR.r[_Rd_].UL[(dd * 2) + 1] = cpuRegs.HI.UL[dd * 2];
		}
	}
}
#endif

#if NOP_PSLLVW
static void nativePSLLVW()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s64)(s32)(cpuRegs.GPR.r[_Rt_].UL[0] << (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1F));
	cpuRegs.GPR.r[_Rd_].SD[1] = (s64)(s32)(cpuRegs.GPR.r[_Rt_].UL[2] << (cpuRegs.GPR.r[_Rs_].UL[2] & 0x1F));
}
#endif

#if NOP_PSRLVW
static void nativePSRLVW()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s64)(s32)(cpuRegs.GPR.r[_Rt_].UL[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1F));
	cpuRegs.GPR.r[_Rd_].SD[1] = (s64)(s32)(cpuRegs.GPR.r[_Rt_].UL[2] >> (cpuRegs.GPR.r[_Rs_].UL[2] & 0x1F));
}
#endif

#if NOP_PMSUBW
static void nativePMSUBW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		s64 temp = ((s64)cpuRegs.GPR.r[_Rs_].SL[ss] * (s64)cpuRegs.GPR.r[_Rt_].SL[ss]);
		s64 temp2 = ((s64)cpuRegs.HI.SL[ss] << 32) - temp;
		temp2 = (s32)(temp2 / 4294967295);
		cpuRegs.LO.SD[dd] = cpuRegs.LO.SL[ss] - (s32)(temp & 0xffffffff);
		cpuRegs.HI.SD[dd] = (s32)temp2;
		if (_Rd_)
		{
			cpuRegs.GPR.r[_Rd_].UL[dd * 2] = cpuRegs.LO.UL[dd * 2];
			cpuRegs.GPR.r[_Rd_].UL[(dd * 2) + 1] = cpuRegs.HI.UL[dd * 2];
		}
	}
}
#endif

#if NOP_PMFHI
static void nativePMFHI()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.HI.UD[0];
	cpuRegs.GPR.r[_Rd_].UD[1] = cpuRegs.HI.UD[1];
}
#endif

#if NOP_PMFLO
static void nativePMFLO()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = cpuRegs.LO.UD[0];
	cpuRegs.GPR.r[_Rd_].UD[1] = cpuRegs.LO.UD[1];
}
#endif

#if NOP_PINTH
static void nativePINTH()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[1] = Rs.US[4];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[1]; cpuRegs.GPR.r[_Rd_].US[3] = Rs.US[5];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[2]; cpuRegs.GPR.r[_Rd_].US[5] = Rs.US[6];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[3]; cpuRegs.GPR.r[_Rd_].US[7] = Rs.US[7];
}
#endif

#if NOP_PMULTW
static void nativePMULTW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		s64 temp = (s64)cpuRegs.GPR.r[_Rs_].SL[ss] * (s64)cpuRegs.GPR.r[_Rt_].SL[ss];
		cpuRegs.LO.UD[dd] = (s32)(temp & 0xffffffff);
		cpuRegs.HI.UD[dd] = (s32)(temp >> 32);
		if (_Rd_) cpuRegs.GPR.r[_Rd_].SD[dd] = temp;
	}
}
#endif

#if NOP_PDIVW
static void nativePDIVW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		if (cpuRegs.GPR.r[_Rs_].UL[ss] == 0x80000000 && cpuRegs.GPR.r[_Rt_].UL[ss] == 0xffffffff)
		{
			cpuRegs.LO.SD[dd] = (s32)0x80000000;
			cpuRegs.HI.SD[dd] = (s32)0;
		}
		else if (cpuRegs.GPR.r[_Rt_].SL[ss] != 0)
		{
			cpuRegs.LO.SD[dd] = cpuRegs.GPR.r[_Rs_].SL[ss] / cpuRegs.GPR.r[_Rt_].SL[ss];
			cpuRegs.HI.SD[dd] = cpuRegs.GPR.r[_Rs_].SL[ss] % cpuRegs.GPR.r[_Rt_].SL[ss];
		}
		else
		{
			cpuRegs.LO.SD[dd] = (cpuRegs.GPR.r[_Rs_].SL[ss] < 0) ? 1 : -1;
			cpuRegs.HI.SD[dd] = cpuRegs.GPR.r[_Rs_].SL[ss];
		}
	}
}
#endif

#if NOP_PCPYLD
static void nativePCPYLD()
{
	if (!_Rd_) return;
	uint64x2_t rs = vld1q_u64(cpuRegs.GPR.r[_Rs_].UD);
	uint64x2_t rt = vld1q_u64(cpuRegs.GPR.r[_Rt_].UD);
	// Rd = [Rt.UD[0], Rs.UD[0]]
	vst1q_u64(cpuRegs.GPR.r[_Rd_].UD,
			  vcombine_u64(vget_low_u64(rt), vget_low_u64(rs)));
}
#endif

#if NOP_PMADDH
static void nativePMADDH()
{
	s32 temp;
	temp = cpuRegs.LO.UL[0] + (s32)cpuRegs.GPR.r[_Rs_].SS[0] * (s32)cpuRegs.GPR.r[_Rt_].SS[0]; cpuRegs.LO.UL[0] = temp;
	temp = cpuRegs.LO.UL[1] + (s32)cpuRegs.GPR.r[_Rs_].SS[1] * (s32)cpuRegs.GPR.r[_Rt_].SS[1]; cpuRegs.LO.UL[1] = temp;
	temp = cpuRegs.HI.UL[0] + (s32)cpuRegs.GPR.r[_Rs_].SS[2] * (s32)cpuRegs.GPR.r[_Rt_].SS[2]; cpuRegs.HI.UL[0] = temp;
	temp = cpuRegs.HI.UL[1] + (s32)cpuRegs.GPR.r[_Rs_].SS[3] * (s32)cpuRegs.GPR.r[_Rt_].SS[3]; cpuRegs.HI.UL[1] = temp;
	temp = cpuRegs.LO.UL[2] + (s32)cpuRegs.GPR.r[_Rs_].SS[4] * (s32)cpuRegs.GPR.r[_Rt_].SS[4]; cpuRegs.LO.UL[2] = temp;
	temp = cpuRegs.LO.UL[3] + (s32)cpuRegs.GPR.r[_Rs_].SS[5] * (s32)cpuRegs.GPR.r[_Rt_].SS[5]; cpuRegs.LO.UL[3] = temp;
	temp = cpuRegs.HI.UL[2] + (s32)cpuRegs.GPR.r[_Rs_].SS[6] * (s32)cpuRegs.GPR.r[_Rt_].SS[6]; cpuRegs.HI.UL[2] = temp;
	temp = cpuRegs.HI.UL[3] + (s32)cpuRegs.GPR.r[_Rs_].SS[7] * (s32)cpuRegs.GPR.r[_Rt_].SS[7]; cpuRegs.HI.UL[3] = temp;
	if (_Rd_) {
		cpuRegs.GPR.r[_Rd_].UL[0] = cpuRegs.LO.UL[0]; cpuRegs.GPR.r[_Rd_].UL[1] = cpuRegs.HI.UL[0];
		cpuRegs.GPR.r[_Rd_].UL[2] = cpuRegs.LO.UL[2]; cpuRegs.GPR.r[_Rd_].UL[3] = cpuRegs.HI.UL[2];
	}
}
#endif

#if NOP_PHMADH
static void nativePHMADH()
{
	s32 ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[1] * (s32)cpuRegs.GPR.r[_Rt_].SS[1];
	cpuRegs.LO.UL[0] = ft + (s32)cpuRegs.GPR.r[_Rs_].SS[0] * (s32)cpuRegs.GPR.r[_Rt_].SS[0];
	cpuRegs.LO.UL[1] = ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[3] * (s32)cpuRegs.GPR.r[_Rt_].SS[3];
	cpuRegs.HI.UL[0] = ft + (s32)cpuRegs.GPR.r[_Rs_].SS[2] * (s32)cpuRegs.GPR.r[_Rt_].SS[2];
	cpuRegs.HI.UL[1] = ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[5] * (s32)cpuRegs.GPR.r[_Rt_].SS[5];
	cpuRegs.LO.UL[2] = ft + (s32)cpuRegs.GPR.r[_Rs_].SS[4] * (s32)cpuRegs.GPR.r[_Rt_].SS[4];
	cpuRegs.LO.UL[3] = ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[7] * (s32)cpuRegs.GPR.r[_Rt_].SS[7];
	cpuRegs.HI.UL[2] = ft + (s32)cpuRegs.GPR.r[_Rs_].SS[6] * (s32)cpuRegs.GPR.r[_Rt_].SS[6];
	cpuRegs.HI.UL[3] = ft;
	if (_Rd_) {
		cpuRegs.GPR.r[_Rd_].UL[0] = cpuRegs.LO.UL[0]; cpuRegs.GPR.r[_Rd_].UL[1] = cpuRegs.HI.UL[0];
		cpuRegs.GPR.r[_Rd_].UL[2] = cpuRegs.LO.UL[2]; cpuRegs.GPR.r[_Rd_].UL[3] = cpuRegs.HI.UL[2];
	}
}
#endif

#if NOP_PAND
static void nativePAND()
{
	if (!_Rd_) return;
	vst1q_u64(cpuRegs.GPR.r[_Rd_].UD,
			  vandq_u64(vld1q_u64(cpuRegs.GPR.r[_Rs_].UD),
						vld1q_u64(cpuRegs.GPR.r[_Rt_].UD)));
}
#endif

#if NOP_PXOR
static void nativePXOR()
{
	if (!_Rd_) return;
	vst1q_u64(cpuRegs.GPR.r[_Rd_].UD,
			  veorq_u64(vld1q_u64(cpuRegs.GPR.r[_Rs_].UD),
						vld1q_u64(cpuRegs.GPR.r[_Rt_].UD)));
}
#endif

#if NOP_PMSUBH
static void nativePMSUBH()
{
	s32 temp;
	temp = cpuRegs.LO.UL[0] - (s32)cpuRegs.GPR.r[_Rs_].SS[0] * (s32)cpuRegs.GPR.r[_Rt_].SS[0]; cpuRegs.LO.UL[0] = temp;
	temp = cpuRegs.LO.UL[1] - (s32)cpuRegs.GPR.r[_Rs_].SS[1] * (s32)cpuRegs.GPR.r[_Rt_].SS[1]; cpuRegs.LO.UL[1] = temp;
	temp = cpuRegs.HI.UL[0] - (s32)cpuRegs.GPR.r[_Rs_].SS[2] * (s32)cpuRegs.GPR.r[_Rt_].SS[2]; cpuRegs.HI.UL[0] = temp;
	temp = cpuRegs.HI.UL[1] - (s32)cpuRegs.GPR.r[_Rs_].SS[3] * (s32)cpuRegs.GPR.r[_Rt_].SS[3]; cpuRegs.HI.UL[1] = temp;
	temp = cpuRegs.LO.UL[2] - (s32)cpuRegs.GPR.r[_Rs_].SS[4] * (s32)cpuRegs.GPR.r[_Rt_].SS[4]; cpuRegs.LO.UL[2] = temp;
	temp = cpuRegs.LO.UL[3] - (s32)cpuRegs.GPR.r[_Rs_].SS[5] * (s32)cpuRegs.GPR.r[_Rt_].SS[5]; cpuRegs.LO.UL[3] = temp;
	temp = cpuRegs.HI.UL[2] - (s32)cpuRegs.GPR.r[_Rs_].SS[6] * (s32)cpuRegs.GPR.r[_Rt_].SS[6]; cpuRegs.HI.UL[2] = temp;
	temp = cpuRegs.HI.UL[3] - (s32)cpuRegs.GPR.r[_Rs_].SS[7] * (s32)cpuRegs.GPR.r[_Rt_].SS[7]; cpuRegs.HI.UL[3] = temp;
	if (_Rd_) {
		cpuRegs.GPR.r[_Rd_].UL[0] = cpuRegs.LO.UL[0]; cpuRegs.GPR.r[_Rd_].UL[1] = cpuRegs.HI.UL[0];
		cpuRegs.GPR.r[_Rd_].UL[2] = cpuRegs.LO.UL[2]; cpuRegs.GPR.r[_Rd_].UL[3] = cpuRegs.HI.UL[2];
	}
}
#endif

#if NOP_PHMSBH
static void nativePHMSBH()
{
	s32 ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[1] * (s32)cpuRegs.GPR.r[_Rt_].SS[1];
	cpuRegs.LO.UL[0] = ft - (s32)cpuRegs.GPR.r[_Rs_].SS[0] * (s32)cpuRegs.GPR.r[_Rt_].SS[0];
	cpuRegs.LO.UL[1] = ~ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[3] * (s32)cpuRegs.GPR.r[_Rt_].SS[3];
	cpuRegs.HI.UL[0] = ft - (s32)cpuRegs.GPR.r[_Rs_].SS[2] * (s32)cpuRegs.GPR.r[_Rt_].SS[2];
	cpuRegs.HI.UL[1] = ~ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[5] * (s32)cpuRegs.GPR.r[_Rt_].SS[5];
	cpuRegs.LO.UL[2] = ft - (s32)cpuRegs.GPR.r[_Rs_].SS[4] * (s32)cpuRegs.GPR.r[_Rt_].SS[4];
	cpuRegs.LO.UL[3] = ~ft;
	ft = (s32)cpuRegs.GPR.r[_Rs_].SS[7] * (s32)cpuRegs.GPR.r[_Rt_].SS[7];
	cpuRegs.HI.UL[2] = ft - (s32)cpuRegs.GPR.r[_Rs_].SS[6] * (s32)cpuRegs.GPR.r[_Rt_].SS[6];
	cpuRegs.HI.UL[3] = ~ft;
	if (_Rd_) {
		cpuRegs.GPR.r[_Rd_].UL[0] = cpuRegs.LO.UL[0]; cpuRegs.GPR.r[_Rd_].UL[1] = cpuRegs.HI.UL[0];
		cpuRegs.GPR.r[_Rd_].UL[2] = cpuRegs.LO.UL[2]; cpuRegs.GPR.r[_Rd_].UL[3] = cpuRegs.HI.UL[2];
	}
}
#endif

#if NOP_PEXEH
static void nativePEXEH()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[2]; cpuRegs.GPR.r[_Rd_].US[1] = Rt.US[1];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[3] = Rt.US[3];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[6]; cpuRegs.GPR.r[_Rd_].US[5] = Rt.US[5];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[4]; cpuRegs.GPR.r[_Rd_].US[7] = Rt.US[7];
}
#endif

#if NOP_PREVH
static void nativePREVH()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[3]; cpuRegs.GPR.r[_Rd_].US[1] = Rt.US[2];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[1]; cpuRegs.GPR.r[_Rd_].US[3] = Rt.US[0];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[7]; cpuRegs.GPR.r[_Rd_].US[5] = Rt.US[6];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[5]; cpuRegs.GPR.r[_Rd_].US[7] = Rt.US[4];
}
#endif

#if NOP_PMULTH
static void nativePMULTH()
{
	int16x8_t rs = vld1q_s16((const int16_t*)cpuRegs.GPR.r[_Rs_].SS);
	int16x8_t rt = vld1q_s16((const int16_t*)cpuRegs.GPR.r[_Rt_].SS);

	// 16x16->32 for each halfword pair, split into low (SS[0..3]) and high (SS[4..7])
	int32x4_t lo_prods = vmull_s16(vget_low_s16(rs),  vget_low_s16(rt));
	int32x4_t hi_prods = vmull_s16(vget_high_s16(rs), vget_high_s16(rt));

	// LO = [prod0, prod1, prod4, prod5]
	int32x4_t LO = vcombine_s32(vget_low_s32(lo_prods),  vget_low_s32(hi_prods));
	// HI = [prod2, prod3, prod6, prod7]
	int32x4_t HI = vcombine_s32(vget_high_s32(lo_prods), vget_high_s32(hi_prods));

	vst1q_u32(cpuRegs.LO.UL, vreinterpretq_u32_s32(LO));
	vst1q_u32(cpuRegs.HI.UL, vreinterpretq_u32_s32(HI));

	if (_Rd_)
	{
		// Rd = [prod0, prod2, prod4, prod6] — even-indexed products
		int32x4_t rd = vuzp1q_s32(lo_prods, hi_prods);
		vst1q_u32(cpuRegs.GPR.r[_Rd_].UL, vreinterpretq_u32_s32(rd));
	}
}
#endif

#if NOP_PDIVBW
static void nativePDIVBW()
{
	for (int n = 0; n < 4; n++)
	{
		if (cpuRegs.GPR.r[_Rs_].UL[n] == 0x80000000 && cpuRegs.GPR.r[_Rt_].US[0] == 0xffff)
		{
			cpuRegs.LO.SL[n] = (s32)0x80000000;
			cpuRegs.HI.SL[n] = (s32)0x0;
		}
		else if (cpuRegs.GPR.r[_Rt_].US[0] != 0)
		{
			cpuRegs.LO.SL[n] = cpuRegs.GPR.r[_Rs_].SL[n] / cpuRegs.GPR.r[_Rt_].SS[0];
			cpuRegs.HI.SL[n] = cpuRegs.GPR.r[_Rs_].SL[n] % cpuRegs.GPR.r[_Rt_].SS[0];
		}
		else
		{
			cpuRegs.LO.SL[n] = (cpuRegs.GPR.r[_Rs_].SL[n] < 0) ? 1 : -1;
			cpuRegs.HI.SL[n] = cpuRegs.GPR.r[_Rs_].SL[n];
		}
	}
}
#endif

#if NOP_PEXEW
static void nativePEXEW()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UL[0] = Rt.UL[2]; cpuRegs.GPR.r[_Rd_].UL[1] = Rt.UL[1];
	cpuRegs.GPR.r[_Rd_].UL[2] = Rt.UL[0]; cpuRegs.GPR.r[_Rd_].UL[3] = Rt.UL[3];
}
#endif

#if NOP_PROT3W
static void nativePROT3W()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UL[0] = Rt.UL[1]; cpuRegs.GPR.r[_Rd_].UL[1] = Rt.UL[2];
	cpuRegs.GPR.r[_Rd_].UL[2] = Rt.UL[0]; cpuRegs.GPR.r[_Rd_].UL[3] = Rt.UL[3];
}
#endif

// ---- tbl_MMI3 ops (funct==0x29, keyed on sa field) ----

#if NOP_PMADDUW
static void nativePMADDUW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		u64 tempu = (u64)((u64)cpuRegs.LO.UL[ss] | ((u64)cpuRegs.HI.UL[ss] << 32)) +
				   ((u64)cpuRegs.GPR.r[_Rs_].UL[ss] * (u64)cpuRegs.GPR.r[_Rt_].UL[ss]);
		cpuRegs.LO.SD[dd] = (s32)(tempu & 0xffffffff);
		cpuRegs.HI.SD[dd] = (s32)(tempu >> 32);
		if (_Rd_) cpuRegs.GPR.r[_Rd_].UD[dd] = tempu;
	}
}
#endif

#if NOP_PSRAVW
static void nativePSRAVW()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].SD[0] = (s64)(cpuRegs.GPR.r[_Rt_].SL[0] >> (cpuRegs.GPR.r[_Rs_].UL[0] & 0x1F));
	cpuRegs.GPR.r[_Rd_].SD[1] = (s64)(cpuRegs.GPR.r[_Rt_].SL[2] >> (cpuRegs.GPR.r[_Rs_].UL[2] & 0x1F));
}
#endif

#if NOP_PMTHI
static void nativePMTHI()
{
	cpuRegs.HI.UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
	cpuRegs.HI.UD[1] = cpuRegs.GPR.r[_Rs_].UD[1];
}
#endif

#if NOP_PMTLO
static void nativePMTLO()
{
	cpuRegs.LO.UD[0] = cpuRegs.GPR.r[_Rs_].UD[0];
	cpuRegs.LO.UD[1] = cpuRegs.GPR.r[_Rs_].UD[1];
}
#endif

#if NOP_PINTEH
static void nativePINTEH()
{
	if (!_Rd_) return;
	GPR_reg Rs = cpuRegs.GPR.r[_Rs_], Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[1] = Rs.US[0];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[2]; cpuRegs.GPR.r[_Rd_].US[3] = Rs.US[2];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[4]; cpuRegs.GPR.r[_Rd_].US[5] = Rs.US[4];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[6]; cpuRegs.GPR.r[_Rd_].US[7] = Rs.US[6];
}
#endif

#if NOP_PMULTUW
static void nativePMULTUW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		u64 tempu = (u64)cpuRegs.GPR.r[_Rs_].UL[ss] * (u64)cpuRegs.GPR.r[_Rt_].UL[ss];
		cpuRegs.LO.UD[dd] = (s32)(tempu & 0xffffffff);
		cpuRegs.HI.UD[dd] = (s32)(tempu >> 32);
		if (_Rd_) cpuRegs.GPR.r[_Rd_].UD[dd] = tempu;
	}
}
#endif

#if NOP_PDIVUW
static void nativePDIVUW()
{
	for (int pass = 0; pass < 2; pass++)
	{
		const int dd = pass, ss = pass * 2;
		if (cpuRegs.GPR.r[_Rt_].UL[ss] != 0) {
			cpuRegs.LO.SD[dd] = (s32)(cpuRegs.GPR.r[_Rs_].UL[ss] / cpuRegs.GPR.r[_Rt_].UL[ss]);
			cpuRegs.HI.SD[dd] = (s32)(cpuRegs.GPR.r[_Rs_].UL[ss] % cpuRegs.GPR.r[_Rt_].UL[ss]);
		} else {
			cpuRegs.LO.SD[dd] = -1;
			cpuRegs.HI.SD[dd] = cpuRegs.GPR.r[_Rs_].SL[ss];
		}
	}
}
#endif

#if NOP_PCPYUD
static void nativePCPYUD()
{
	if (!_Rd_) return;
	uint64x2_t rs = vld1q_u64(cpuRegs.GPR.r[_Rs_].UD);
	uint64x2_t rt = vld1q_u64(cpuRegs.GPR.r[_Rt_].UD);
	// Rd = [Rs.UD[1], Rt.UD[1]]
	vst1q_u64(cpuRegs.GPR.r[_Rd_].UD,
			  vcombine_u64(vget_high_u64(rs), vget_high_u64(rt)));
}
#endif

#if NOP_POR
static void nativePOR()
{
	if (!_Rd_) return;
	vst1q_u64(cpuRegs.GPR.r[_Rd_].UD,
			  vorrq_u64(vld1q_u64(cpuRegs.GPR.r[_Rs_].UD),
						vld1q_u64(cpuRegs.GPR.r[_Rt_].UD)));
}
#endif

#if NOP_PNOR
static void nativePNOR()
{
	if (!_Rd_) return;
	cpuRegs.GPR.r[_Rd_].UD[0] = ~(cpuRegs.GPR.r[_Rs_].UD[0] | cpuRegs.GPR.r[_Rt_].UD[0]);
	cpuRegs.GPR.r[_Rd_].UD[1] = ~(cpuRegs.GPR.r[_Rs_].UD[1] | cpuRegs.GPR.r[_Rt_].UD[1]);
}
#endif

#if NOP_PEXCH
static void nativePEXCH()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].US[0] = Rt.US[0]; cpuRegs.GPR.r[_Rd_].US[1] = Rt.US[2];
	cpuRegs.GPR.r[_Rd_].US[2] = Rt.US[1]; cpuRegs.GPR.r[_Rd_].US[3] = Rt.US[3];
	cpuRegs.GPR.r[_Rd_].US[4] = Rt.US[4]; cpuRegs.GPR.r[_Rd_].US[5] = Rt.US[6];
	cpuRegs.GPR.r[_Rd_].US[6] = Rt.US[5]; cpuRegs.GPR.r[_Rd_].US[7] = Rt.US[7];
}
#endif

#if NOP_PCPYH
static void nativePCPYH()
{
	if (!_Rd_) return;
	uint16x8_t rt = vld1q_u16(cpuRegs.GPR.r[_Rt_].US);
	// broadcast US[0] across low 4 lanes, US[4] across high 4 lanes
	uint16x4_t lo = vdup_lane_u16(vget_low_u16(rt),  0);
	uint16x4_t hi = vdup_lane_u16(vget_high_u16(rt), 0);
	vst1q_u16(cpuRegs.GPR.r[_Rd_].US, vcombine_u16(lo, hi));
}
#endif

#if NOP_PEXCW
static void nativePEXCW()
{
	if (!_Rd_) return;
	GPR_reg Rt = cpuRegs.GPR.r[_Rt_];
	cpuRegs.GPR.r[_Rd_].UL[0] = Rt.UL[0]; cpuRegs.GPR.r[_Rd_].UL[1] = Rt.UL[2];
	cpuRegs.GPR.r[_Rd_].UL[2] = Rt.UL[1]; cpuRegs.GPR.r[_Rd_].UL[3] = Rt.UL[3];
}
#endif

// ============================================================================
//  MMI table dispatch (opcode == 0x1C, keyed on funct field bits [5:0])
// ============================================================================

// MMI0 sub-dispatch (funct==0x08, keyed on sa field bits [10:6])
#ifdef NATIVE_MMI
static bool tryMMI0()
{
	switch (_Sa_)
	{
#if NOP_PADDW
		case 0x00: nativePADDW(); return true;
#endif
#if NOP_PSUBW
		case 0x01: nativePSUBW(); return true;
#endif
#if NOP_PCGTW
		case 0x02: nativePCGTW(); return true;
#endif
#if NOP_PMAXW
		case 0x03: nativePMAXW(); return true;
#endif
#if NOP_PADDH
		case 0x04: nativePADDH(); return true;
#endif
#if NOP_PSUBH
		case 0x05: nativePSUBH(); return true;
#endif
#if NOP_PCGTH
		case 0x06: nativePCGTH(); return true;
#endif
#if NOP_PMAXH
		case 0x07: nativePMAXH(); return true;
#endif
#if NOP_PADDB
		case 0x08: nativePADDB(); return true;
#endif
#if NOP_PSUBB
		case 0x09: nativePSUBB(); return true;
#endif
#if NOP_PCGTB
		case 0x0A: nativePCGTB(); return true;
#endif
#if NOP_PADDSW
		case 0x10: nativePADDSW(); return true;
#endif
#if NOP_PSUBSW
		case 0x11: nativePSUBSW(); return true;
#endif
#if NOP_PEXTLW
		case 0x12: nativePEXTLW(); return true;
#endif
#if NOP_PPACW
		case 0x13: nativePPACW(); return true;
#endif
#if NOP_PADDSH
		case 0x14: nativePADDSH(); return true;
#endif
#if NOP_PSUBSH
		case 0x15: nativePSUBSH(); return true;
#endif
#if NOP_PEXTLH
		case 0x16: nativePEXTLH(); return true;
#endif
#if NOP_PPACH
		case 0x17: nativePPACH(); return true;
#endif
#if NOP_PADDSB
		case 0x18: nativePADDSB(); return true;
#endif
#if NOP_PSUBSB
		case 0x19: nativePSUBSB(); return true;
#endif
#if NOP_PEXTLB
		case 0x1A: nativePEXTLB(); return true;
#endif
#if NOP_PPACB
		case 0x1B: nativePPACB(); return true;
#endif
#if NOP_PEXT5
		case 0x1E: nativePEXT5(); return true;
#endif
#if NOP_PPAC5
		case 0x1F: nativePPAC5(); return true;
#endif
		default: return false;
	}
}

// MMI1 sub-dispatch (funct==0x28, keyed on sa field bits [10:6])
static bool tryMMI1()
{
	switch (_Sa_)
	{
#if NOP_PABSW
		case 0x01: nativePABSW(); return true;
#endif
#if NOP_PCEQW
		case 0x02: nativePCEQW(); return true;
#endif
#if NOP_PMINW
		case 0x03: nativePMINW(); return true;
#endif
#if NOP_PADSBH
		case 0x04: nativePADSBH(); return true;
#endif
#if NOP_PABSH
		case 0x05: nativePABSH(); return true;
#endif
#if NOP_PCEQH
		case 0x06: nativePCEQH(); return true;
#endif
#if NOP_PMINH
		case 0x07: nativePMINH(); return true;
#endif
#if NOP_PCEQB
		case 0x0A: nativePCEQB(); return true;
#endif
#if NOP_PADDUW
		case 0x10: nativePADDUW(); return true;
#endif
#if NOP_PSUBUW
		case 0x11: nativePSUBUW(); return true;
#endif
#if NOP_PEXTUW
		case 0x12: nativePEXTUW(); return true;
#endif
#if NOP_PADDUH
		case 0x14: nativePADDUH(); return true;
#endif
#if NOP_PSUBUH
		case 0x15: nativePSUBUH(); return true;
#endif
#if NOP_PEXTUH
		case 0x16: nativePEXTUH(); return true;
#endif
#if NOP_PADDUB
		case 0x18: nativePADDUB(); return true;
#endif
#if NOP_PSUBUB
		case 0x19: nativePSUBUB(); return true;
#endif
#if NOP_PEXTUB
		case 0x1A: nativePEXTUB(); return true;
#endif
#if NOP_QFSRV
		case 0x1B: nativeQFSRV(); return true;
#endif
		default: return false;
	}
}

// MMI2 sub-dispatch (funct==0x09, keyed on sa field bits [10:6])
static bool tryMMI2()
{
	switch (_Sa_)
	{
#if NOP_PMADDW
		case 0x00: nativePMADDW(); return true;
#endif
#if NOP_PSLLVW
		case 0x02: nativePSLLVW(); return true;
#endif
#if NOP_PSRLVW
		case 0x03: nativePSRLVW(); return true;
#endif
#if NOP_PMSUBW
		case 0x04: nativePMSUBW(); return true;
#endif
#if NOP_PMFHI
		case 0x08: nativePMFHI(); return true;
#endif
#if NOP_PMFLO
		case 0x09: nativePMFLO(); return true;
#endif
#if NOP_PINTH
		case 0x0A: nativePINTH(); return true;
#endif
#if NOP_PMULTW
		case 0x0C: nativePMULTW(); return true;
#endif
#if NOP_PDIVW
		case 0x0D: nativePDIVW(); return true;
#endif
#if NOP_PCPYLD
		case 0x0E: nativePCPYLD(); return true;
#endif
#if NOP_PMADDH
		case 0x10: nativePMADDH(); return true;
#endif
#if NOP_PHMADH
		case 0x11: nativePHMADH(); return true;
#endif
#if NOP_PAND
		case 0x12: nativePAND(); return true;
#endif
#if NOP_PXOR
		case 0x13: nativePXOR(); return true;
#endif
#if NOP_PMSUBH
		case 0x14: nativePMSUBH(); return true;
#endif
#if NOP_PHMSBH
		case 0x15: nativePHMSBH(); return true;
#endif
#if NOP_PEXEH
		case 0x1A: nativePEXEH(); return true;
#endif
#if NOP_PREVH
		case 0x1B: nativePREVH(); return true;
#endif
#if NOP_PMULTH
		case 0x1C: nativePMULTH(); return true;
#endif
#if NOP_PDIVBW
		case 0x1D: nativePDIVBW(); return true;
#endif
#if NOP_PEXEW
		case 0x1E: nativePEXEW(); return true;
#endif
#if NOP_PROT3W
		case 0x1F: nativePROT3W(); return true;
#endif
		default: return false;
	}
}

// MMI3 sub-dispatch (funct==0x29, keyed on sa field bits [10:6])
static bool tryMMI3()
{
	switch (_Sa_)
	{
#if NOP_PMADDUW
		case 0x00: nativePMADDUW(); return true;
#endif
#if NOP_PSRAVW
		case 0x03: nativePSRAVW(); return true;
#endif
#if NOP_PMTHI
		case 0x08: nativePMTHI(); return true;
#endif
#if NOP_PMTLO
		case 0x09: nativePMTLO(); return true;
#endif
#if NOP_PINTEH
		case 0x0A: nativePINTEH(); return true;
#endif
#if NOP_PMULTUW
		case 0x0C: nativePMULTUW(); return true;
#endif
#if NOP_PDIVUW
		case 0x0D: nativePDIVUW(); return true;
#endif
#if NOP_PCPYUD
		case 0x0E: nativePCPYUD(); return true;
#endif
#if NOP_POR
		case 0x12: nativePOR(); return true;
#endif
#if NOP_PNOR
		case 0x13: nativePNOR(); return true;
#endif
#if NOP_PEXCH
		case 0x1A: nativePEXCH(); return true;
#endif
#if NOP_PCPYH
		case 0x1B: nativePCPYH(); return true;
#endif
#if NOP_PEXCW
		case 0x1E: nativePEXCW(); return true;
#endif
		default: return false;
	}
}
#endif // NATIVE_MMI

// MMI top-level dispatch (funct field)
#if defined(NATIVE_MMI) || NOP_MFHI1 || NOP_MFLO1 || NOP_MTHI1 || NOP_MTLO1
static bool tryMMI()
{
	switch (_Funct_)
	{
#if NOP_MADD
		case 0x00: nativeMADD(); return true;
#endif
#if NOP_MADDU
		case 0x01: nativeMADDU(); return true;
#endif
#if NOP_PLZCW
		case 0x04: nativePLZCW(); return true;
#endif
#ifdef NATIVE_MMI
		case 0x08: return tryMMI0();
		case 0x09: return tryMMI2();
#endif
#if NOP_MFHI1
		case 0x10: nativeMFHI1(); return true;
#endif
#if NOP_MTHI1
		case 0x11: nativeMTHI1(); return true;
#endif
#if NOP_MFLO1
		case 0x12: nativeMFLO1(); return true;
#endif
#if NOP_MTLO1
		case 0x13: nativeMTLO1(); return true;
#endif
#if NOP_MULT1
		case 0x18: nativeMULT1(); return true;
#endif
#if NOP_MULTU1
		case 0x19: nativeMULTU1(); return true;
#endif
#if NOP_DIV1
		case 0x1A: nativeDIV1(); return true;
#endif
#if NOP_DIVU1
		case 0x1B: nativeDIVU1(); return true;
#endif
#if NOP_MADD1
		case 0x20: nativeMADD1(); return true;
#endif
#if NOP_MADDU1
		case 0x21: nativeMADDU1(); return true;
#endif
#ifdef NATIVE_MMI
		case 0x28: return tryMMI1();
		case 0x29: return tryMMI3();
#endif
#if NOP_PMFHL
		case 0x30: nativePMFHL(); return true;
#endif
#if NOP_PMTHL
		case 0x31: nativePMTHL(); return true;
#endif
#if NOP_PSLLH
		case 0x34: nativePSLLH(); return true;
#endif
#if NOP_PSRLH
		case 0x36: nativePSRLH(); return true;
#endif
#if NOP_PSRAH
		case 0x37: nativePSRAH(); return true;
#endif
#if NOP_PSLLW
		case 0x3C: nativePSLLW(); return true;
#endif
#if NOP_PSRLW
		case 0x3E: nativePSRLW(); return true;
#endif
#if NOP_PSRAW
		case 0x3F: nativePSRAW(); return true;
#endif
		default: return false;
	}
}
#endif

// ============================================================================
//  COP0 dispatch (opcode == 0x10, keyed on rs field bits [25:21])
// ============================================================================

// COP0 BC0 sub-dispatch (rs == 0x08, keyed on rt field bits [20:16])
#if NOP_BC0F || NOP_BC0T || NOP_BC0FL || NOP_BC0TL
static bool tryCOP0_BC0()
{
	switch (_Rt_)
	{
#if NOP_BC0F
		case 0x00: nativeBC0F(); return true;
#endif
#if NOP_BC0T
		case 0x01: nativeBC0T(); return true;
#endif
#if NOP_BC0FL
		case 0x02: nativeBC0FL(); return true;
#endif
#if NOP_BC0TL
		case 0x03: nativeBC0TL(); return true;
#endif
		default: return false;
	}
}
#endif

// COP0 C0 sub-dispatch (rs == 0x10, keyed on funct field bits [5:0])
#if NOP_TLBR || NOP_TLBWI || NOP_TLBWR || NOP_TLBP || NOP_ERET || NOP_EI || NOP_DI
static bool tryCOP0_C0()
{
	switch (cpuRegs.code & 0x3F)
	{
#if NOP_TLBR
		case 0x01: nativeTLBR(); return true;
#endif
#if NOP_TLBWI
		case 0x02: nativeTLBWI(); return true;
#endif
#if NOP_TLBWR
		case 0x06: nativeTLBWR(); return true;
#endif
#if NOP_TLBP
		case 0x08: nativeTLBP(); return true;
#endif
#if NOP_ERET
		case 0x18: nativeERET(); return true;
#endif
#if NOP_EI
		case 0x38: nativeEI(); return true;
#endif
#if NOP_DI
		case 0x39: nativeDI(); return true;
#endif
		default: return false;
	}
}
#endif

// COP0 top-level dispatch (rs field)
#if NOP_MFC0 || NOP_MTC0 || NOP_BC0F || NOP_BC0T || NOP_BC0FL || NOP_BC0TL || NOP_TLBR || NOP_TLBWI || NOP_TLBWR || NOP_TLBP || NOP_ERET || NOP_EI || NOP_DI
static bool tryCOP0()
{
	const u32 rs = (cpuRegs.code >> 21) & 0x1F;
	switch (rs)
	{
#if NOP_MFC0
		case 0x00: nativeMFC0(); return true;
#endif
#if NOP_MTC0
		case 0x04: nativeMTC0(); return true;
#endif
#if NOP_BC0F || NOP_BC0T || NOP_BC0FL || NOP_BC0TL
		case 0x08: return tryCOP0_BC0();
#endif
#if NOP_TLBR || NOP_TLBWI || NOP_TLBWR || NOP_TLBP || NOP_ERET || NOP_EI || NOP_DI
		case 0x10: return tryCOP0_C0();
#endif
		default: return false;
	}
}
#endif

// ============================================================================
//  COP1 dispatch (opcode == 0x11, keyed on rs field bits [25:21])
// ============================================================================

// COP1 BC1 sub-dispatch (rs == 0x08, keyed on rt field bits [20:16])
#if NOP_BC1F || NOP_BC1T || NOP_BC1FL || NOP_BC1TL
static bool tryCOP1_BC1()
{
	switch (_Rt_)
	{
#if NOP_BC1F
		case 0x00: nativeBC1F(); return true;
#endif
#if NOP_BC1T
		case 0x01: nativeBC1T(); return true;
#endif
#if NOP_BC1FL
		case 0x02: nativeBC1FL(); return true;
#endif
#if NOP_BC1TL
		case 0x03: nativeBC1TL(); return true;
#endif
		default: return false;
	}
}
#endif

// COP1 S sub-dispatch (rs == 0x10, single-precision, keyed on funct bits [5:0])
#if NOP_ADD_S || NOP_SUB_S || NOP_MUL_S || NOP_DIV_S || NOP_SQRT_S || NOP_ABS_S || NOP_MOV_S || NOP_NEG_S || NOP_RSQRT_S || NOP_ADDA_S || NOP_SUBA_S || NOP_MULA_S || NOP_MADD_S || NOP_MSUB_S || NOP_MADDA_S || NOP_MSUBA_S || NOP_CVT_W || NOP_MAX_S || NOP_MIN_S || NOP_C_F || NOP_C_EQ || NOP_C_LT || NOP_C_LE
static bool tryCOP1_S()
{
	switch (cpuRegs.code & 0x3F)
	{
#if NOP_ADD_S
		case 0x00: nativeADD_S(); return true;
#endif
#if NOP_SUB_S
		case 0x01: nativeSUB_S(); return true;
#endif
#if NOP_MUL_S
		case 0x02: nativeMUL_S(); return true;
#endif
#if NOP_DIV_S
		case 0x03: nativeDIV_S(); return true;
#endif
#if NOP_SQRT_S
		case 0x04: nativeSQRT_S(); return true;
#endif
#if NOP_ABS_S
		case 0x05: nativeABS_S(); return true;
#endif
#if NOP_MOV_S
		case 0x06: nativeMOV_S(); return true;
#endif
#if NOP_NEG_S
		case 0x07: nativeNEG_S(); return true;
#endif
#if NOP_RSQRT_S
		case 0x16: nativeRSQRT_S(); return true;
#endif
#if NOP_ADDA_S
		case 0x18: nativeADDA_S(); return true;
#endif
#if NOP_SUBA_S
		case 0x19: nativeSUBA_S(); return true;
#endif
#if NOP_MULA_S
		case 0x1A: nativeMULA_S(); return true;
#endif
#if NOP_MADD_S
		case 0x1C: nativeMADD_S(); return true;
#endif
#if NOP_MSUB_S
		case 0x1D: nativeMSUB_S(); return true;
#endif
#if NOP_MADDA_S
		case 0x1E: nativeMADDA_S(); return true;
#endif
#if NOP_MSUBA_S
		case 0x1F: nativeMSUBA_S(); return true;
#endif
#if NOP_CVT_W
		case 0x24: nativeCVT_W(); return true;
#endif
#if NOP_MAX_S
		case 0x28: nativeMAX_S(); return true;
#endif
#if NOP_MIN_S
		case 0x29: nativeMIN_S(); return true;
#endif
#if NOP_C_F
		case 0x30: nativeC_F(); return true;
#endif
#if NOP_C_EQ
		case 0x32: nativeC_EQ(); return true;
#endif
#if NOP_C_LT
		case 0x34: nativeC_LT(); return true;
#endif
#if NOP_C_LE
		case 0x36: nativeC_LE(); return true;
#endif
		default: return false;
	}
}
#endif

// COP1 W sub-dispatch (rs == 0x14, word format, keyed on funct bits [5:0])
#if NOP_CVT_S
static bool tryCOP1_W()
{
	switch (cpuRegs.code & 0x3F)
	{
#if NOP_CVT_S
		case 0x20: nativeCVT_S(); return true;
#endif
		default: return false;
	}
}
#endif

// COP1 top-level dispatch (rs field)
#if NOP_MFC1 || NOP_CFC1 || NOP_MTC1 || NOP_CTC1 || NOP_BC1F || NOP_BC1T || NOP_BC1FL || NOP_BC1TL || NOP_ADD_S || NOP_CVT_S
static bool tryCOP1()
{
	const u32 rs = (cpuRegs.code >> 21) & 0x1F;
	switch (rs)
	{
#if NOP_MFC1
		case 0x00: nativeMFC1(); return true;
#endif
#if NOP_CFC1
		case 0x02: nativeCFC1(); return true;
#endif
#if NOP_MTC1
		case 0x04: nativeMTC1(); return true;
#endif
#if NOP_CTC1
		case 0x06: nativeCTC1(); return true;
#endif
#if NOP_BC1F || NOP_BC1T || NOP_BC1FL || NOP_BC1TL
		case 0x08: return tryCOP1_BC1();
#endif
#if NOP_ADD_S || NOP_SUB_S || NOP_MUL_S || NOP_DIV_S || NOP_SQRT_S || NOP_ABS_S || NOP_MOV_S || NOP_NEG_S || NOP_RSQRT_S || NOP_ADDA_S || NOP_SUBA_S || NOP_MULA_S || NOP_MADD_S || NOP_MSUB_S || NOP_MADDA_S || NOP_MSUBA_S || NOP_CVT_W || NOP_MAX_S || NOP_MIN_S || NOP_C_F || NOP_C_EQ || NOP_C_LT || NOP_C_LE
		case 0x10: return tryCOP1_S();
#endif
#if NOP_CVT_S
		case 0x14: return tryCOP1_W();
#endif
		default: return false;
	}
}
#endif

// ============================================================================
//  Top-level dispatch — called from execI() before opcode.interpret()
// ============================================================================

bool arm64TryNativeExec()
{
	const u32 op = cpuRegs.code >> 26;

	switch (op)
	{
		case 0x00: return trySpecial();  // SPECIAL
		case 0x01: return tryRegImm();   // REGIMM
#if NOP_J
		case 0x02: nativeJ(); return true;       // J
#endif
#if NOP_JAL
		case 0x03: nativeJAL(); return true;     // JAL
#endif
#if NOP_BEQ
		case 0x04: nativeBEQ(); return true;     // BEQ
#endif
#if NOP_BNE
		case 0x05: nativeBNE(); return true;     // BNE
#endif
#if NOP_BLEZ
		case 0x06: nativeBLEZ(); return true;    // BLEZ
#endif
#if NOP_BGTZ
		case 0x07: nativeBGTZ(); return true;    // BGTZ
#endif
#if NOP_ADDI
		case 0x08: nativeADDI(); return true;    // ADDI
#endif
#if NOP_ADDIU
		case 0x09: nativeADDIU(); return true;   // ADDIU
#endif
#if NOP_SLTI
		case 0x0A: nativeSLTI(); return true;    // SLTI
#endif
#if NOP_SLTIU
		case 0x0B: nativeSLTIU(); return true;   // SLTIU
#endif
#if NOP_ANDI
		case 0x0C: nativeANDI(); return true;    // ANDI
#endif
#if NOP_ORI
		case 0x0D: nativeORI(); return true;     // ORI
#endif
#if NOP_XORI
		case 0x0E: nativeXORI(); return true;    // XORI
#endif
#if NOP_LUI
		case 0x0F: nativeLUI(); return true;     // LUI
#endif
#if NOP_MFC0 || NOP_MTC0 || NOP_BC0F || NOP_BC0T || NOP_BC0FL || NOP_BC0TL || NOP_TLBR || NOP_TLBWI || NOP_TLBWR || NOP_TLBP || NOP_ERET || NOP_EI || NOP_DI
		case 0x10: return tryCOP0();             // COP0
#endif
#if NOP_MFC1 || NOP_CFC1 || NOP_MTC1 || NOP_CTC1 || NOP_BC1F || NOP_BC1T || NOP_BC1FL || NOP_BC1TL || NOP_ADD_S || NOP_CVT_S
		case 0x11: return tryCOP1();             // COP1
#endif
#if NOP_BEQL
		case 0x14: nativeBEQL(); return true;    // BEQL
#endif
#if NOP_BNEL
		case 0x15: nativeBNEL(); return true;    // BNEL
#endif
#if NOP_BLEZL
		case 0x16: nativeBLEZL(); return true;   // BLEZL
#endif
#if NOP_BGTZL
		case 0x17: nativeBGTZL(); return true;   // BGTZL
#endif
#if NOP_DADDI
		case 0x18: nativeDADDI(); return true;   // DADDI
#endif
#if NOP_DADDIU
		case 0x19: nativeDADDIU(); return true;  // DADDIU
#endif
#if NOP_LDL
		case 0x1A: nativeLDL(); return true;     // LDL
#endif
#if NOP_LDR
		case 0x1B: nativeLDR(); return true;     // LDR
#endif
#if defined(NATIVE_MMI) || NOP_MFHI1 || NOP_MFLO1 || NOP_MTHI1 || NOP_MTLO1
		case 0x1C: return tryMMI();              // MMI
#endif
#if NOP_LQ
		case 0x1E: nativeLQ(); return true;      // LQ
#endif
#if NOP_SQ
		case 0x1F: nativeSQ(); return true;      // SQ
#endif
#if NOP_LB
		case 0x20: nativeLB(); return true;      // LB
#endif
#if NOP_LH
		case 0x21: nativeLH(); return true;      // LH
#endif
#if NOP_LWL
		case 0x22: nativeLWL(); return true;     // LWL
#endif
#if NOP_LW
		case 0x23: nativeLW(); return true;      // LW
#endif
#if NOP_LBU
		case 0x24: nativeLBU(); return true;     // LBU
#endif
#if NOP_LHU
		case 0x25: nativeLHU(); return true;     // LHU
#endif
#if NOP_LWR
		case 0x26: nativeLWR(); return true;     // LWR
#endif
#if NOP_LWU
		case 0x27: nativeLWU(); return true;     // LWU
#endif
#if NOP_SB
		case 0x28: nativeSB(); return true;      // SB
#endif
#if NOP_SH
		case 0x29: nativeSH(); return true;      // SH
#endif
#if NOP_SWL
		case 0x2A: nativeSWL(); return true;     // SWL
#endif
#if NOP_SW
		case 0x2B: nativeSW(); return true;      // SW
#endif
#if NOP_SDL
		case 0x2C: nativeSDL(); return true;     // SDL
#endif
#if NOP_SDR
		case 0x2D: nativeSDR(); return true;     // SDR
#endif
#if NOP_SWR
		case 0x2E: nativeSWR(); return true;     // SWR
#endif
#if NOP_LWC1
		case 0x31: nativeLWC1(); return true;    // LWC1
#endif
#if NOP_LQC2
		case 0x36: nativeLQC2(); return true;    // LQC2
#endif
#if NOP_LD
		case 0x37: nativeLD(); return true;      // LD
#endif
#if NOP_SWC1
		case 0x39: nativeSWC1(); return true;    // SWC1
#endif
#if NOP_SQC2
		case 0x3E: nativeSQC2(); return true;    // SQC2
#endif
#if NOP_SD
		case 0x3F: nativeSD(); return true;      // SD
#endif
		default: return false;
	}
}

#endif // __aarch64__ || _M_ARM64
