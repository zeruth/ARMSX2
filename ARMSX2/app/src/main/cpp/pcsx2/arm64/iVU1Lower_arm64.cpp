// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Lower Instruction Stubs
// Integer ALU, load/store, branches, FDIV, flag ops,
// move/transfer, random, EFU, special (XITOP/XTOP/XGKICK)

#include "Common.h"
#include "VUops.h"
#include "VU.h"
#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"
#include "MTVU.h"
#include <cmath>

using namespace vixl::aarch64;

extern void _vuXGKICKTransfer(s32 cycles, bool flush);

// ============================================================================
//  Native codegen helpers
// ============================================================================

// VU1_BASE_REG — mirrors the definition in iVU1micro_arm64.cpp.
// At runtime, x23 always holds &VU1 throughout a compiled block.
static const auto VU1_BASE_REG = x23;

// Compute byte offset of VI[reg] within VURegs.
static constexpr int64_t viOff(u32 reg)
{
	return static_cast<int64_t>(offsetof(VURegs, VI)) + reg * static_cast<int64_t>(sizeof(REG_VI));
}

// Compute byte offset of VF[reg] within VURegs.
static constexpr int64_t vfOff(u32 reg)
{
	return static_cast<int64_t>(offsetof(VURegs, VF)) + reg * static_cast<int64_t>(sizeof(VECTOR));
}

// Non-inline VI backup wrapper — callable from JIT-emitted code.
// Mirrors _vuBackupVI() in VUops.cpp (which is __fi and can't be linked).
static void vu1BackupVI(VURegs* VU, u32 reg)
{
	if (VU->VIBackupCycles && reg == VU->VIRegNumber)
	{
		VU->VIBackupCycles = 2;
		return;
	}
	VU->VIBackupCycles = 2;
	VU->VIRegNumber = reg;
	VU->VIOldValue = VU->VI[reg].US[0];
}

// Emit a BL to vu1BackupVI(VU, reg).
// Must be emitted before any VI register write.
static void emitBackupVI(u32 reg)
{
	armAsm->Mov(x0, VU1_BASE_REG);
	armAsm->Mov(w1, reg);
	armEmitCall(reinterpret_cast<const void*>(vu1BackupVI));
}

// ============================================================================
//  C wrapper helpers — replicate interpreter logic for JIT-called functions.
//  These use the same field-extraction macros as VUops.cpp, but take VURegs*.
// ============================================================================

// Field extraction (mirrors VUops.cpp macros, but parameter-based)
#define W_Ft(VU) (((VU)->code >> 16) & 0x1F)
#define W_Fs(VU) (((VU)->code >> 11) & 0x1F)
#define W_Fd(VU) (((VU)->code >>  6) & 0x1F)
#define W_It(VU) (W_Ft(VU) & 0xF)
#define W_Is(VU) (W_Fs(VU) & 0xF)
#define W_Id(VU) (W_Fd(VU) & 0xF)
#define W_X(VU)  (((VU)->code >> 24) & 0x1)
#define W_Y(VU)  (((VU)->code >> 23) & 0x1)
#define W_Z(VU)  (((VU)->code >> 22) & 0x1)
#define W_W(VU)  (((VU)->code >> 21) & 0x1)
#define W_XYZW(VU) (((VU)->code >> 21) & 0xF)
#define W_Fsf(VU)   (((VU)->code >> 21) & 0x03)
#define W_Ftf(VU)   (((VU)->code >> 23) & 0x03)
#define W_Imm11(VU) ((s32)((VU)->code & 0x400 ? 0xfffffc00 | ((VU)->code & 0x3ff) : (VU)->code & 0x3ff))

// Type aliases to avoid vixl namespace clashes (vixl defines s16/u16 as registers)
using vu_s16 = int16_t;
using vu_u16 = uint16_t;
using vu_s32 = int32_t;
using vu_u32 = uint32_t;

// Emit inline ARM64 for vuDouble clamping on a W register.
// Flushes denormals to ±0, clamps inf/NaN to ±max if vu1SignOverflow is set.
// wreg: u32 float bits (modified in place)
// wtmp: scratch register (clobbered)
static void emitVuDouble(const Register& wreg, const Register& wtmp)
{
	a64::Label done;
	armAsm->Ubfx(wtmp, wreg, 23, 8); // extract 8-bit exponent

	if (CHECK_VU_SIGN_OVERFLOW(1))
	{
		a64::Label denormal;
		armAsm->Cbz(wtmp, &denormal);           // exp==0 -> denormal
		armAsm->Cmp(wtmp, 0xFF);
		armAsm->B(&done, a64::ne);               // normal -> done
		// Infinity/NaN: clamp to ±max
		armAsm->And(wtmp, wreg, 0x80000000u);
		armAsm->Mov(wreg, 0x7f7fffff);
		armAsm->Orr(wreg, wreg, wtmp);
		armAsm->B(&done);
		armAsm->Bind(&denormal);
		armAsm->And(wreg, wreg, 0x80000000u);    // flush to ±0
	}
	else
	{
		armAsm->Cbnz(wtmp, &done);               // exp!=0 -> done
		armAsm->And(wreg, wreg, 0x80000000u);    // flush to ±0
	}
	armAsm->Bind(&done);
}

// Float denormal/overflow clamping — mirrors vuDouble() in VUops.cpp
static float vu1Double(vu_u32 f)
{
	switch (f & 0x7f800000)
	{
		case 0x0:
			f &= 0x80000000;
			return *(float*)&f;
		case 0x7f800000:
			if (CHECK_VU_SIGN_OVERFLOW(1))
			{
				u32 d = (f & 0x80000000) | 0x7f7fffff;
				return *(float*)&d;
			}
			break;
	}
	return *(float*)&f;
}

// Branch address calculation — mirrors _branchAddr() in VUops.cpp
static s32 vu1BranchAddr(VURegs* VU)
{
	s32 bpc = VU->VI[REG_TPC].SL + (W_Imm11(VU) * 8);
	bpc &= 0x3fff;
	return bpc;
}

// Set branch state — mirrors _setBranch() in VUops.cpp
static void vu1SetBranch(VURegs* VU, u32 bpc)
{
	if (VU->branch == 1)
	{
		VU->delaybranchpc = bpc;
		VU->takedelaybranch = true;
	}
	else
	{
		VU->branch = 2;
		VU->branchpc = bpc;
	}
}

// LFSR advance — mirrors AdvanceLFSR() in VUops.cpp
static void vu1AdvanceLFSR(VURegs* VU)
{
	int x = (VU->VI[REG_R].UL >> 4) & 1;
	int y = (VU->VI[REG_R].UL >> 22) & 1;
	VU->VI[REG_R].UL <<= 1;
	VU->VI[REG_R].UL ^= x ^ y;
	VU->VI[REG_R].UL = (VU->VI[REG_R].UL & 0x7fffff) | 0x3f800000;
}

// EATAN polynomial — mirrors _vuCalculateEATAN() in VUops.cpp
static float vu1CalculateEATAN(float inputvalue)
{
	float eatanconst[9] = { 0.999999344348907f, -0.333298563957214f, 0.199465364217758f, -0.13085337519646f,
							0.096420042216778f, -0.055909886956215f, 0.021861229091883f, -0.004054057877511f,
							0.785398185253143f };
	float result = (eatanconst[0] * inputvalue) + (eatanconst[1] * pow(inputvalue, 3)) + (eatanconst[2] * pow(inputvalue, 5))
					+ (eatanconst[3] * pow(inputvalue, 7)) + (eatanconst[4] * pow(inputvalue, 9)) + (eatanconst[5] * pow(inputvalue, 11))
					+ (eatanconst[6] * pow(inputvalue, 13)) + (eatanconst[7] * pow(inputvalue, 15));
	result += eatanconst[8];
	result = vu1Double(*(u32*)&result);
	return result;
}

// ============================================================================
//  C wrapper functions — called from JIT via BL.
//  Each takes VURegs* VU (passed in x0 = VU1_BASE_REG).
// ============================================================================

// Macro to generate rec function that calls a C wrapper
#define REC_VU1_LOWER_CALL(name) \
	void recVU1_##name() { \
		armAsm->Mov(x0, VU1_BASE_REG); \
		armEmitCall(reinterpret_cast<const void*>(vu1_##name)); \
	}

// --- FDIV wrappers ---
static void vu1_DIV(VURegs* VU)
{
	float ft = vu1Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	float fs = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft == 0.0)
	{
		if (fs == 0.0)
			VU->statusflag |= 0x10;
		else
			VU->statusflag |= 0x20;
		if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
			(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
			VU->q.UL = 0xFF7FFFFF;
		else
			VU->q.UL = 0x7F7FFFFF;
	}
	else
	{
		VU->q.F = fs / ft;
		VU->q.F = vu1Double(VU->q.UL);
	}
}

static void vu1_SQRT(VURegs* VU)
{
	float ft = vu1Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft < 0.0)
		VU->statusflag |= 0x10;
	VU->q.F = sqrt(fabs(ft));
	VU->q.F = vu1Double(VU->q.UL);
}

static void vu1_RSQRT(VURegs* VU)
{
	float ft = vu1Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	float fs = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft == 0.0)
	{
		VU->statusflag |= 0x20;
		if (fs != 0)
		{
			if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
				(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
				VU->q.UL = 0xFF7FFFFF;
			else
				VU->q.UL = 0x7F7FFFFF;
		}
		else
		{
			if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
				(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
				VU->q.UL = 0x80000000;
			else
				VU->q.UL = 0;
			VU->statusflag |= 0x10;
		}
	}
	else
	{
		if (ft < 0.0)
			VU->statusflag |= 0x10;
		float temp = sqrt(fabs(ft));
		VU->q.F = fs / temp;
		VU->q.F = vu1Double(VU->q.UL);
	}
}

// WAITQ/WAITP are NOPs in the interpreter
static void vu1_WAITQ(VURegs* VU) { (void)VU; }
static void vu1_WAITP(VURegs* VU) { (void)VU; }

// --- Load/Store wrappers ---
static void vu1_LQ(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
}

static void vu1_LQD(VURegs* VU)
{
	vu1BackupVI(VU, W_Is(VU));
	if (W_Is(VU) != 0) VU->VI[W_Is(VU)].US[0]--;
	if (W_Ft(VU) == 0) return;
	u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
}

static void vu1_LQI(VURegs* VU)
{
	vu1BackupVI(VU, W_Is(VU));
	if (W_Ft(VU))
	{
		u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
		u32* ptr = (u32*)GET_VU_MEM(VU, addr);
		if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
		if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
		if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
		if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
	}
	if (W_Fs(VU) != 0) VU->VI[W_Is(VU)].US[0]++;
}

static void vu1_SQ(VURegs* VU)
{
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_It(VU)].SS[0]) * 16);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
}

static void vu1_SQD(VURegs* VU)
{
	vu1BackupVI(VU, W_It(VU));
	if (W_Ft(VU) != 0) VU->VI[W_It(VU)].US[0]--;
	u32 addr = (VU->VI[W_It(VU)].US[0] * 16);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
}

static void vu1_SQI(VURegs* VU)
{
	vu1BackupVI(VU, W_It(VU));
	u32 addr = (VU->VI[W_It(VU)].US[0] * 16);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
	if (W_Ft(VU) != 0) VU->VI[W_It(VU)].US[0]++;
}

static void vu1_ILW(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VI[W_It(VU)].US[0] = ptr[0];
	if (W_Y(VU)) VU->VI[W_It(VU)].US[0] = ptr[2];
	if (W_Z(VU)) VU->VI[W_It(VU)].US[0] = ptr[4];
	if (W_W(VU)) VU->VI[W_It(VU)].US[0] = ptr[6];
}

static void vu1_ISW(VURegs* VU)
{
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) { ptr[0] = VU->VI[W_It(VU)].US[0]; ptr[1] = 0; }
	if (W_Y(VU)) { ptr[2] = VU->VI[W_It(VU)].US[0]; ptr[3] = 0; }
	if (W_Z(VU)) { ptr[4] = VU->VI[W_It(VU)].US[0]; ptr[5] = 0; }
	if (W_W(VU)) { ptr[6] = VU->VI[W_It(VU)].US[0]; ptr[7] = 0; }
}

static void vu1_ILWR(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VI[W_It(VU)].US[0] = ptr[0];
	if (W_Y(VU)) VU->VI[W_It(VU)].US[0] = ptr[2];
	if (W_Z(VU)) VU->VI[W_It(VU)].US[0] = ptr[4];
	if (W_W(VU)) VU->VI[W_It(VU)].US[0] = ptr[6];
}

static void vu1_ISWR(VURegs* VU)
{
	u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) { ptr[0] = VU->VI[W_It(VU)].US[0]; ptr[1] = 0; }
	if (W_Y(VU)) { ptr[2] = VU->VI[W_It(VU)].US[0]; ptr[3] = 0; }
	if (W_Z(VU)) { ptr[4] = VU->VI[W_It(VU)].US[0]; ptr[5] = 0; }
	if (W_W(VU)) { ptr[6] = VU->VI[W_It(VU)].US[0]; ptr[7] = 0; }
}

// --- Branch wrappers ---
static void vu1_B(VURegs* VU)
{
	s32 bpc = vu1BranchAddr(VU);
	vu1SetBranch(VU, bpc);
}

static void vu1_BAL(VURegs* VU)
{
	s32 bpc = vu1BranchAddr(VU);
	if (W_It(VU))
	{
		if (VU->branch == 1)
			VU->VI[W_It(VU)].US[0] = (VU->branchpc + 8) / 8;
		else
			VU->VI[W_It(VU)].US[0] = (VU->VI[REG_TPC].UL + 8) / 8;
	}
	vu1SetBranch(VU, bpc);
}

static void vu1_JR(VURegs* VU)
{
	u32 bpc = VU->VI[W_Is(VU)].US[0] * 8;
	vu1SetBranch(VU, bpc);
}

static void vu1_JALR(VURegs* VU)
{
	u32 bpc = VU->VI[W_Is(VU)].US[0] * 8;
	if (W_It(VU))
	{
		if (VU->branch == 1)
			VU->VI[W_It(VU)].US[0] = (VU->branchpc + 8) / 8;
		else
			VU->VI[W_It(VU)].US[0] = (VU->VI[REG_TPC].UL + 8) / 8;
	}
	vu1SetBranch(VU, bpc);
}

static void vu1_IBEQ(VURegs* VU)
{
	vu_s16 dest = VU->VI[W_It(VU)].US[0];
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_It(VU)) dest = VU->VIOldValue;
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (dest == src)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

static void vu1_IBNE(VURegs* VU)
{
	vu_s16 dest = VU->VI[W_It(VU)].US[0];
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_It(VU)) dest = VU->VIOldValue;
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (dest != src)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

static void vu1_IBLTZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src < 0)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

static void vu1_IBGTZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src > 0)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

static void vu1_IBLEZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src <= 0)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

static void vu1_IBGEZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src >= 0)
	{
		s32 bpc = vu1BranchAddr(VU);
		vu1SetBranch(VU, bpc);
	}
}

// --- Flag operation wrappers ---
static void vu1_FSAND(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) & imm;
}

static void vu1_FSEQ(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	if ((VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) == imm)
		VU->VI[W_It(VU)].US[0] = 1;
	else
		VU->VI[W_It(VU)].US[0] = 0;
}

static void vu1_FSOR(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) | imm;
}

static void vu1_FSSET(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7FF);
	VU->statusflag = (imm & 0xFC0) | (VU->statusflag & 0x3F);
}

static void vu1_FMAND(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = VU->VI[W_Is(VU)].US[0] & (VU->VI[REG_MAC_FLAG].UL & 0xFFFF);
}

static void vu1_FMEQ(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	if ((VU->VI[REG_MAC_FLAG].UL & 0xFFFF) == VU->VI[W_Is(VU)].US[0])
		VU->VI[W_It(VU)].US[0] = 1;
	else
		VU->VI[W_It(VU)].US[0] = 0;
}

static void vu1_FMOR(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_MAC_FLAG].UL & 0xFFFF) | VU->VI[W_Is(VU)].US[0];
}

static void vu1_FCAND(VURegs* VU)
{
	if ((VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) & (VU->code & 0xFFFFFF))
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu1_FCEQ(VURegs* VU)
{
	if ((VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) == (VU->code & 0xFFFFFF))
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu1_FCOR(VURegs* VU)
{
	u32 hold = (VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) | (VU->code & 0xFFFFFF);
	if (hold == 0xFFFFFF)
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu1_FCSET(VURegs* VU)
{
	VU->clipflag = (u32)(VU->code & 0xFFFFFF);
}

static void vu1_FCGET(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = VU->VI[REG_CLIP_FLAG].UL & 0x0FFF;
}

// --- Random number generator wrappers ---
static void vu1_RINIT(VURegs* VU)
{
	VU->VI[REG_R].UL = 0x3F800000 | (VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x007FFFFF);
}

static void vu1_RGET(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = VU->VI[REG_R].UL;
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = VU->VI[REG_R].UL;
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = VU->VI[REG_R].UL;
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = VU->VI[REG_R].UL;
}

static void vu1_RNEXT(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	vu1AdvanceLFSR(VU);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = VU->VI[REG_R].UL;
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = VU->VI[REG_R].UL;
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = VU->VI[REG_R].UL;
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = VU->VI[REG_R].UL;
}

static void vu1_RXOR(VURegs* VU)
{
	VU->VI[REG_R].UL = 0x3F800000 | ((VU->VI[REG_R].UL ^ VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]) & 0x007FFFFF);
}

// --- EFU wrappers ---
static void vu1_ESADD(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].i.x) * vu1Double(VU->VF[W_Fs(VU)].i.x)
			+ vu1Double(VU->VF[W_Fs(VU)].i.y) * vu1Double(VU->VF[W_Fs(VU)].i.y)
			+ vu1Double(VU->VF[W_Fs(VU)].i.z) * vu1Double(VU->VF[W_Fs(VU)].i.z);
	VU->p.F = p;
}

static void vu1_ERSADD(VURegs* VU)
{
	float p = (vu1Double(VU->VF[W_Fs(VU)].i.x) * vu1Double(VU->VF[W_Fs(VU)].i.x))
			+ (vu1Double(VU->VF[W_Fs(VU)].i.y) * vu1Double(VU->VF[W_Fs(VU)].i.y))
			+ (vu1Double(VU->VF[W_Fs(VU)].i.z) * vu1Double(VU->VF[W_Fs(VU)].i.z));
	if (p != 0.0) p = 1.0f / p;
	VU->p.F = p;
}

static void vu1_ELENG(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].i.x) * vu1Double(VU->VF[W_Fs(VU)].i.x)
			+ vu1Double(VU->VF[W_Fs(VU)].i.y) * vu1Double(VU->VF[W_Fs(VU)].i.y)
			+ vu1Double(VU->VF[W_Fs(VU)].i.z) * vu1Double(VU->VF[W_Fs(VU)].i.z);
	if (p >= 0) p = sqrt(p);
	VU->p.F = p;
}

static void vu1_ERLENG(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].i.x) * vu1Double(VU->VF[W_Fs(VU)].i.x)
			+ vu1Double(VU->VF[W_Fs(VU)].i.y) * vu1Double(VU->VF[W_Fs(VU)].i.y)
			+ vu1Double(VU->VF[W_Fs(VU)].i.z) * vu1Double(VU->VF[W_Fs(VU)].i.z);
	if (p >= 0)
	{
		p = sqrt(p);
		if (p != 0) p = 1.0f / p;
	}
	VU->p.F = p;
}

static void vu1_EATANxy(VURegs* VU)
{
	float p = 0;
	if (vu1Double(VU->VF[W_Fs(VU)].i.x) != 0)
		p = vu1CalculateEATAN(vu1Double(VU->VF[W_Fs(VU)].i.y) / vu1Double(VU->VF[W_Fs(VU)].i.x));
	VU->p.F = p;
}

static void vu1_EATANxz(VURegs* VU)
{
	float p = 0;
	if (vu1Double(VU->VF[W_Fs(VU)].i.x) != 0)
		p = vu1CalculateEATAN(vu1Double(VU->VF[W_Fs(VU)].i.z) / vu1Double(VU->VF[W_Fs(VU)].i.x));
	VU->p.F = p;
}

static void vu1_ESUM(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].i.x) + vu1Double(VU->VF[W_Fs(VU)].i.y)
			+ vu1Double(VU->VF[W_Fs(VU)].i.z) + vu1Double(VU->VF[W_Fs(VU)].i.w);
	VU->p.F = p;
}

static void vu1_ERCPR(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	if (p != 0) p = 1.0f / p;
	VU->p.F = p;
}

static void vu1_ESQRT(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	if (p >= 0) p = sqrt(p);
	VU->p.F = p;
}

static void vu1_ERSQRT(VURegs* VU)
{
	float p = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	if (p >= 0)
	{
		p = sqrt(p);
		if (p) p = 1.0f / p;
	}
	VU->p.F = p;
}

static void vu1_ESIN(VURegs* VU)
{
	float sinconsts[5] = {1.0f, -0.166666567325592f, 0.008333025500178f, -0.000198074136279f, 0.000002601886990f};
	float p = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	p = (sinconsts[0] * p) + (sinconsts[1] * pow(p, 3)) + (sinconsts[2] * pow(p, 5))
		+ (sinconsts[3] * pow(p, 7)) + (sinconsts[4] * pow(p, 9));
	VU->p.F = vu1Double(*(u32*)&p);
}

static void vu1_EATAN(VURegs* VU)
{
	float p = vu1CalculateEATAN(vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]));
	VU->p.F = p;
}

static void vu1_EEXP(VURegs* VU)
{
	float consts[6] = {0.249998688697815f, 0.031257584691048f, 0.002591371303424f,
						0.000171562001924f, 0.000005430199963f, 0.000000690600018f};
	float p = vu1Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	p = 1.0f + (consts[0] * p) + (consts[1] * pow(p, 2)) + (consts[2] * pow(p, 3))
		+ (consts[3] * pow(p, 4)) + (consts[4] * pow(p, 5)) + (consts[5] * pow(p, 6));
	p = pow(p, 4);
	p = vu1Double(*(u32*)&p);
	p = 1 / p;
	VU->p.F = p;
}

// --- Special wrappers ---
static void vu1_XITOP(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	if (THREAD_VU1)
		VU->VI[W_It(VU)].US[0] = vu1Thread.vifRegs.itop & 0x3FF;
	else
		VU->VI[W_It(VU)].US[0] = VU->GetVifRegs().itop & 0x3FF;
}

static void vu1_XTOP(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	if (THREAD_VU1)
		VU->VI[W_It(VU)].US[0] = (vu_u16)vu1Thread.vifRegs.top;
	else
		VU->VI[W_It(VU)].US[0] = (vu_u16)VU->GetVifRegs().top;
}

static void vu1_XGKICK(VURegs* VU)
{
	if (VU->xgkickenable)
		_vuXGKICKTransfer(0, true);
	u32 addr = (VU->VI[W_Is(VU)].US[0] & 0x3ff) * 16;
	u32 diff = 0x4000 - addr;
	VU->xgkickenable = true;
	VU->xgkickaddr = addr;
	VU->xgkickdiff = diff;
	VU->xgkicksizeremaining = 0;
	VU->xgkickendpacket = false;
	VU->xgkicklastcycle = VU->cycle;
	VU->xgkickcyclecount = 1;
	VU0.VI[REG_VPU_STAT].UL |= (1 << 12);
}

// ============================================================================
//  Per-instruction interp stub toggles (1 = interp, 0 = native)
// ============================================================================

// ---- FDIV group ----
#ifdef INTERP_VU_FDIV
#define ISTUB_VU_DIV     1
#define ISTUB_VU_SQRT    1
#define ISTUB_VU_RSQRT   1
#define ISTUB_VU_WAITQ   1
#define ISTUB_VU_WAITP   1
#else
#define ISTUB_VU_DIV     0
#define ISTUB_VU_SQRT    0
#define ISTUB_VU_RSQRT   0
#define ISTUB_VU_WAITQ   0
#define ISTUB_VU_WAITP   0
#endif

// ---- IALU group ----
#ifdef INTERP_VU_IALU
#define ISTUB_VU_IADD    1
#define ISTUB_VU_ISUB    1
#define ISTUB_VU_IADDI   1
#define ISTUB_VU_IADDIU  1
#define ISTUB_VU_ISUBIU  1
#define ISTUB_VU_IAND    1
#define ISTUB_VU_IOR     1
#else
#define ISTUB_VU_IADD    0
#define ISTUB_VU_ISUB    0
#define ISTUB_VU_IADDI   0
#define ISTUB_VU_IADDIU  0
#define ISTUB_VU_ISUBIU  0
#define ISTUB_VU_IAND    0
#define ISTUB_VU_IOR     0
#endif

// ---- LoadStore group ----
#ifdef INTERP_VU_LOADSTORE
#define ISTUB_VU_LQ      1
#define ISTUB_VU_LQD     1
#define ISTUB_VU_LQI     1
#define ISTUB_VU_SQ      1
#define ISTUB_VU_SQD     1
#define ISTUB_VU_SQI     1
#define ISTUB_VU_ILW     1
#define ISTUB_VU_ISW     1
#define ISTUB_VU_ILWR    1
#define ISTUB_VU_ISWR    1
#else
#define ISTUB_VU_LQ      0
#define ISTUB_VU_LQD     0
#define ISTUB_VU_LQI     0
#define ISTUB_VU_SQ      0
#define ISTUB_VU_SQD     0
#define ISTUB_VU_SQI     0
#define ISTUB_VU_ILW     0
#define ISTUB_VU_ISW     0
#define ISTUB_VU_ILWR    0
#define ISTUB_VU_ISWR    0
#endif

// ---- Branch group ----
#ifdef INTERP_VU_BRANCH
#define ISTUB_VU_B       1
#define ISTUB_VU_BAL     1
#define ISTUB_VU_JR      1
#define ISTUB_VU_JALR    1
#define ISTUB_VU_IBEQ    1
#define ISTUB_VU_IBNE    1
#define ISTUB_VU_IBLTZ   1
#define ISTUB_VU_IBGTZ   1
#define ISTUB_VU_IBLEZ   1
#define ISTUB_VU_IBGEZ   1
#else
#define ISTUB_VU_B       0
#define ISTUB_VU_BAL     0
#define ISTUB_VU_JR      0
#define ISTUB_VU_JALR    0
#define ISTUB_VU_IBEQ    0
#define ISTUB_VU_IBNE    0
#define ISTUB_VU_IBLTZ   0
#define ISTUB_VU_IBGTZ   0
#define ISTUB_VU_IBLEZ   0
#define ISTUB_VU_IBGEZ   0
#endif

// ---- Misc group (move, flag, random, EFU, special) ----
#ifdef INTERP_VU_MISC
#define ISTUB_VU_MOVE    1
#define ISTUB_VU_MR32    1
#define ISTUB_VU_MFIR    1
#define ISTUB_VU_MTIR    1
#define ISTUB_VU_MFP     1
#define ISTUB_VU_FSAND   1
#define ISTUB_VU_FSEQ    1
#define ISTUB_VU_FSOR    1
#define ISTUB_VU_FSSET   1
#define ISTUB_VU_FMAND   1
#define ISTUB_VU_FMEQ    1
#define ISTUB_VU_FMOR    1
#define ISTUB_VU_FCAND   1
#define ISTUB_VU_FCEQ    1
#define ISTUB_VU_FCOR    1
#define ISTUB_VU_FCSET   1
#define ISTUB_VU_FCGET   1
#define ISTUB_VU_RINIT   1
#define ISTUB_VU_RGET    1
#define ISTUB_VU_RNEXT   1
#define ISTUB_VU_RXOR    1
#define ISTUB_VU_ESADD   1
#define ISTUB_VU_ERSADD  1
#define ISTUB_VU_ELENG   1
#define ISTUB_VU_ERLENG  1
#define ISTUB_VU_EATANxy 1
#define ISTUB_VU_EATANxz 1
#define ISTUB_VU_ESUM    1
#define ISTUB_VU_ERCPR   1
#define ISTUB_VU_ESQRT_EFU 1
#define ISTUB_VU_ERSQRT  1
#define ISTUB_VU_ESIN    1
#define ISTUB_VU_EATAN   1
#define ISTUB_VU_EEXP    1
#define ISTUB_VU_XITOP   1
#define ISTUB_VU_XTOP    1
#define ISTUB_VU_XGKICK  1
#else
#define ISTUB_VU_MOVE    0
#define ISTUB_VU_MR32    0
#define ISTUB_VU_MFIR    0
#define ISTUB_VU_MTIR    0
#define ISTUB_VU_MFP     0
#define ISTUB_VU_FSAND   0
#define ISTUB_VU_FSEQ    0
#define ISTUB_VU_FSOR    0
#define ISTUB_VU_FSSET   0
#define ISTUB_VU_FMAND   0
#define ISTUB_VU_FMEQ    0
#define ISTUB_VU_FMOR    0
#define ISTUB_VU_FCAND   0
#define ISTUB_VU_FCEQ    0
#define ISTUB_VU_FCOR    0
#define ISTUB_VU_FCSET   0
#define ISTUB_VU_FCGET   0
#define ISTUB_VU_RINIT   0
#define ISTUB_VU_RGET    0
#define ISTUB_VU_RNEXT   0
#define ISTUB_VU_RXOR    0
#define ISTUB_VU_ESADD   0
#define ISTUB_VU_ERSADD  0
#define ISTUB_VU_ELENG   0
#define ISTUB_VU_ERLENG  0
#define ISTUB_VU_EATANxy 0
#define ISTUB_VU_EATANxz 0
#define ISTUB_VU_ESUM    0
#define ISTUB_VU_ERCPR   0
#define ISTUB_VU_ESQRT_EFU 0
#define ISTUB_VU_ERSQRT  0
#define ISTUB_VU_ESIN    0
#define ISTUB_VU_EATAN   0
#define ISTUB_VU_EEXP    0
#define ISTUB_VU_XITOP   0
#define ISTUB_VU_XTOP    0
#define ISTUB_VU_XGKICK  0
#endif

// ============================================================================
//  Interpreter stub: dispatch through VU1 lower opcode table.
//  VU1.code must be set before the rec function is called.
//  The table handles sub-table dispatch (LowerOP, T3_xx) automatically.
// ============================================================================

// Code-emitter macro: called at block-compile time to emit an ARM64 BL to
// the specific interpreter function for this lower opcode.
// VU1.code is set by CompileBlock (both as compile-time variable and as a
// runtime store) before calling this function.
#define REC_VU1_LOWER_INTERP(name) \
	void recVU1_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU1_LOWER_OPCODE[VU1.code >> 25])); \
	}

// Non-INTERP path (ISTUB=0, no native codegen yet): same emitter as ISTUB=1.
#define REC_VU1_LOWER_EMIT(name) \
	void recVU1_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU1_LOWER_OPCODE[VU1.code >> 25])); \
	}

// ============================================================================
//  FDIV — Division pipeline
// ============================================================================

#if ISTUB_VU_DIV
REC_VU1_LOWER_INTERP(DIV)
#else
void recVU1_DIV() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 ftf = (VU1.code >> 23) & 0x3;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 fsf = (VU1.code >> 21) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load raw u32 float bits
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + ftf * 4));
	armAsm->Ldr(w1, MemOperand(VU1_BASE_REG, vfOff(fs) + fsf * 4));
	// Save raw copies for sign check
	armAsm->Mov(w4, w0);
	armAsm->Mov(w5, w1);
	// vuDouble clamp both operands
	emitVuDouble(w0, w2);
	emitVuDouble(w1, w2);
	// Clear statusflag bits [5:4]
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	// Convert to float
	armAsm->Fmov(s0, w0); // ft
	armAsm->Fmov(s1, w1); // fs

	a64::Label ftNotZero, done;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftNotZero, a64::ne);

	// --- ft == 0 path ---
	a64::Label fsNotZero, signCheck;
	armAsm->Fcmp(s1, 0.0);
	armAsm->B(&fsNotZero, a64::ne);
	// fs==0: statusflag |= 0x10
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->B(&signCheck);
	armAsm->Bind(&fsNotZero);
	// fs!=0: statusflag |= 0x20
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x20);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bind(&signCheck);
	// Sign check: (raw_ft ^ raw_fs) bit 31
	armAsm->Eor(w3, w4, w5);
	a64::Label diffSign;
	armAsm->Tbnz(w3, 31, &diffSign);
	// Same sign: q = 0x7F7FFFFF
	armAsm->Mov(w0, 0x7F7FFFFFu);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));
	armAsm->B(&done);
	armAsm->Bind(&diffSign);
	// Different sign: q = 0xFF7FFFFF
	armAsm->Mov(w0, 0xFF7FFFFFu);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));
	armAsm->B(&done);

	// --- ft != 0 path ---
	armAsm->Bind(&ftNotZero);
	armAsm->Fdiv(s2, s1, s0); // fs / ft
	armAsm->Fmov(w0, s2);
	emitVuDouble(w0, w2);      // clamp result
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));

	armAsm->Bind(&done);
}
#endif

#if ISTUB_VU_SQRT
REC_VU1_LOWER_INTERP(SQRT)
#else
void recVU1_SQRT() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 ftf = (VU1.code >> 23) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load and clamp ft
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + ftf * 4));
	emitVuDouble(w0, w2);
	// Clear statusflag bits [5:4]
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	// Convert to float
	armAsm->Fmov(s0, w0);
	// If negative: statusflag |= 0x10, use fabs
	a64::Label notNeg, store;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&notNeg, a64::ge);
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bind(&notNeg);
	// sqrt(fabs(ft))
	armAsm->Fabs(s0, s0);
	armAsm->Fsqrt(s0, s0);
	// Clamp result
	armAsm->Fmov(w0, s0);
	emitVuDouble(w0, w2);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));
}
#endif

#if ISTUB_VU_RSQRT
REC_VU1_LOWER_INTERP(RSQRT)
#else
void recVU1_RSQRT() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 ftf = (VU1.code >> 23) & 0x3;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 fsf = (VU1.code >> 21) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load raw bits
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + ftf * 4));
	armAsm->Ldr(w1, MemOperand(VU1_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Mov(w4, w0); // save raw ft
	armAsm->Mov(w5, w1); // save raw fs
	emitVuDouble(w0, w2);
	emitVuDouble(w1, w2);
	// Clear statusflag [5:4]
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Fmov(s0, w0); // ft
	armAsm->Fmov(s1, w1); // fs

	a64::Label ftNotZero, done;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftNotZero, a64::ne);

	// --- ft == 0 ---
	// statusflag |= 0x20
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x20);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));

	a64::Label fsIsZero;
	armAsm->Fcmp(s1, 0.0);
	armAsm->B(&fsIsZero, a64::eq);

	// fs != 0, ft == 0: q = ±max based on sign XOR
	a64::Label diffSign1;
	armAsm->Eor(w3, w4, w5);
	armAsm->Tbnz(w3, 31, &diffSign1);
	armAsm->Mov(w0, 0x7F7FFFFFu);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));
	armAsm->B(&done);
	armAsm->Bind(&diffSign1);
	armAsm->Mov(w0, 0xFF7FFFFFu);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));
	armAsm->B(&done);

	armAsm->Bind(&fsIsZero);
	// fs == 0, ft == 0: q = ±0, statusflag |= 0x10
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	a64::Label diffSign2;
	armAsm->Eor(w3, w4, w5);
	armAsm->Tbnz(w3, 31, &diffSign2);
	armAsm->Str(wzr, MemOperand(VU1_BASE_REG, q_off)); // +0
	armAsm->B(&done);
	armAsm->Bind(&diffSign2);
	armAsm->Mov(w0, 0x80000000u);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off)); // -0
	armAsm->B(&done);

	// --- ft != 0 ---
	armAsm->Bind(&ftNotZero);
	// If ft < 0: statusflag |= 0x10
	a64::Label ftPos;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftPos, a64::ge);
	armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->Bind(&ftPos);
	// temp = sqrt(fabs(ft)); q = fs / temp
	armAsm->Fabs(s0, s0);
	armAsm->Fsqrt(s0, s0);
	armAsm->Fdiv(s2, s1, s0); // fs / sqrt(|ft|)
	armAsm->Fmov(w0, s2);
	emitVuDouble(w0, w2);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, q_off));

	armAsm->Bind(&done);
}
#endif

#if ISTUB_VU_WAITQ
REC_VU1_LOWER_INTERP(WAITQ)
#else
void recVU1_WAITQ() { /* NOP — no instructions to emit */ }
#endif

#if ISTUB_VU_WAITP
REC_VU1_LOWER_INTERP(WAITP)
#else
void recVU1_WAITP() { /* NOP — no instructions to emit */ }
#endif

// ============================================================================
//  Integer ALU
// ============================================================================

#if ISTUB_VU_IADD
REC_VU1_LOWER_INTERP(IADD)
#else
void recVU1_IADD() {
	const u32 id = (VU1.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 it = (VU1.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(it)));
	armAsm->Add(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU_ISUB
REC_VU1_LOWER_INTERP(ISUB)
#else
void recVU1_ISUB() {
	const u32 id = (VU1.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 it = (VU1.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(it)));
	armAsm->Sub(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU_IADDI
REC_VU1_LOWER_INTERP(IADDI)
#else
void recVU1_IADDI() {
	const u32 it = (VU1.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	// 5-bit signed immediate at bits [10:6], sign-extended.
	s32 imm = (VU1.code >> 6) & 0x1f;
	imm = ((imm & 0x10) ? static_cast<s32>(0xfffffff0) : 0) | (imm & 0xf);
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	if (imm > 0)
		armAsm->Add(w0, w0, imm);
	else if (imm < 0)
		armAsm->Sub(w0, w0, static_cast<u32>(-imm));
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_IADDIU
REC_VU1_LOWER_INTERP(IADDIU)
#else
void recVU1_IADDIU() {
	const u32 it = (VU1.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	// 15-bit unsigned immediate: bits [24:21] → [14:11], bits [10:0] → [10:0].
	const u32 imm = ((VU1.code >> 10) & 0x7800) | (VU1.code & 0x7ff);
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	if (imm != 0)
	{
		armAsm->Mov(w1, imm);
		armAsm->Add(w0, w0, w1);
	}
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_ISUBIU
REC_VU1_LOWER_INTERP(ISUBIU)
#else
void recVU1_ISUBIU() {
	const u32 it = (VU1.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 imm = ((VU1.code >> 10) & 0x7800) | (VU1.code & 0x7ff);
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	if (imm != 0)
	{
		armAsm->Mov(w1, imm);
		armAsm->Sub(w0, w0, w1);
	}
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_IAND
REC_VU1_LOWER_INTERP(IAND)
#else
void recVU1_IAND() {
	const u32 id = (VU1.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 it = (VU1.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(it)));
	armAsm->And(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU_IOR
REC_VU1_LOWER_INTERP(IOR)
#else
void recVU1_IOR() {
	const u32 id = (VU1.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 it = (VU1.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(it)));
	armAsm->Orr(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(id)));
}
#endif

// ============================================================================
//  Load / Store (VU data memory)
// ============================================================================

#if ISTUB_VU_LQ
REC_VU1_LOWER_INTERP(LQ)
#else
REC_VU1_LOWER_CALL(LQ)
#endif

#if ISTUB_VU_LQD
REC_VU1_LOWER_INTERP(LQD)
#else
REC_VU1_LOWER_CALL(LQD)
#endif

#if ISTUB_VU_LQI
REC_VU1_LOWER_INTERP(LQI)
#else
REC_VU1_LOWER_CALL(LQI)
#endif

#if ISTUB_VU_SQ
REC_VU1_LOWER_INTERP(SQ)
#else
REC_VU1_LOWER_CALL(SQ)
#endif

#if ISTUB_VU_SQD
REC_VU1_LOWER_INTERP(SQD)
#else
REC_VU1_LOWER_CALL(SQD)
#endif

#if ISTUB_VU_SQI
REC_VU1_LOWER_INTERP(SQI)
#else
REC_VU1_LOWER_CALL(SQI)
#endif

#if ISTUB_VU_ILW
REC_VU1_LOWER_INTERP(ILW)
#else
REC_VU1_LOWER_CALL(ILW)
#endif

#if ISTUB_VU_ISW
REC_VU1_LOWER_INTERP(ISW)
#else
REC_VU1_LOWER_CALL(ISW)
#endif

#if ISTUB_VU_ILWR
REC_VU1_LOWER_INTERP(ILWR)
#else
REC_VU1_LOWER_CALL(ILWR)
#endif

#if ISTUB_VU_ISWR
REC_VU1_LOWER_INTERP(ISWR)
#else
REC_VU1_LOWER_CALL(ISWR)
#endif

// ============================================================================
//  Branches
// ============================================================================

#if ISTUB_VU_B
REC_VU1_LOWER_INTERP(B)
#else
REC_VU1_LOWER_CALL(B)
#endif

#if ISTUB_VU_BAL
REC_VU1_LOWER_INTERP(BAL)
#else
REC_VU1_LOWER_CALL(BAL)
#endif

#if ISTUB_VU_JR
REC_VU1_LOWER_INTERP(JR)
#else
REC_VU1_LOWER_CALL(JR)
#endif

#if ISTUB_VU_JALR
REC_VU1_LOWER_INTERP(JALR)
#else
REC_VU1_LOWER_CALL(JALR)
#endif

#if ISTUB_VU_IBEQ
REC_VU1_LOWER_INTERP(IBEQ)
#else
REC_VU1_LOWER_CALL(IBEQ)
#endif

#if ISTUB_VU_IBNE
REC_VU1_LOWER_INTERP(IBNE)
#else
REC_VU1_LOWER_CALL(IBNE)
#endif

#if ISTUB_VU_IBLTZ
REC_VU1_LOWER_INTERP(IBLTZ)
#else
REC_VU1_LOWER_CALL(IBLTZ)
#endif

#if ISTUB_VU_IBGTZ
REC_VU1_LOWER_INTERP(IBGTZ)
#else
REC_VU1_LOWER_CALL(IBGTZ)
#endif

#if ISTUB_VU_IBLEZ
REC_VU1_LOWER_INTERP(IBLEZ)
#else
REC_VU1_LOWER_CALL(IBLEZ)
#endif

#if ISTUB_VU_IBGEZ
REC_VU1_LOWER_INTERP(IBGEZ)
#else
REC_VU1_LOWER_CALL(IBGEZ)
#endif

// ============================================================================
//  Move / Transfer
// ============================================================================

#if ISTUB_VU_MOVE
REC_VU1_LOWER_INTERP(MOVE)
#else
void recVU1_MOVE() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	if (xyzw & 8) { armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 0));  armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));  }
	if (xyzw & 4) { armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 4));  armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));  }
	if (xyzw & 2) { armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 8));  armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));  }
	if (xyzw & 1) { armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 12)); armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 12)); }
}
#endif

#if ISTUB_VU_MR32
REC_VU1_LOWER_INTERP(MR32)
#else
void recVU1_MR32() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	// Save VF[fs].x — needed for W component (W receives the original X).
	armAsm->Ldr(w1, MemOperand(VU1_BASE_REG, vfOff(fs) + 0));
	if (xyzw & 8) { // X = fs.y
		armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 4));
		armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));
	}
	if (xyzw & 4) { // Y = fs.z
		armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 8));
		armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));
	}
	if (xyzw & 2) { // Z = fs.w
		armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + 12));
		armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));
	}
	if (xyzw & 1) { // W = saved fs.x
		armAsm->Str(w1, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
	}
}
#endif

#if ISTUB_VU_MFIR
REC_VU1_LOWER_INTERP(MFIR)
#else
void recVU1_MFIR() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 is = (VU1.code >> 11) & 0xF;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	// Sign-extend VI[is].SS[0] to 32 bits, broadcast to selected VF components.
	armAsm->Ldrsh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	if (xyzw & 8) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));
	if (xyzw & 4) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));
	if (xyzw & 2) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));
	if (xyzw & 1) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
}
#endif

#if ISTUB_VU_MTIR
REC_VU1_LOWER_INTERP(MTIR)
#else
void recVU1_MTIR() {
	const u32 it = (VU1.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 fsf = (VU1.code >> 21) & 0x3; // 0=x, 1=y, 2=z, 3=w
	emitBackupVI(it);
	// Load lower 16 bits of VF[fs].F[fsf] into VI[it].
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_MFP
REC_VU1_LOWER_INTERP(MFP)
#else
void recVU1_MFP() {
	const u32 ft = (VU1.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	// Load P register (32-bit, stored as VI[REG_P].UL).
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_P)));
	if (xyzw & 8) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));
	if (xyzw & 4) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));
	if (xyzw & 2) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));
	if (xyzw & 1) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
}
#endif

// ============================================================================
//  Flag read/write
// ============================================================================

#if ISTUB_VU_FSAND
REC_VU1_LOWER_INTERP(FSAND)
#else
REC_VU1_LOWER_CALL(FSAND)
#endif

#if ISTUB_VU_FSEQ
REC_VU1_LOWER_INTERP(FSEQ)
#else
REC_VU1_LOWER_CALL(FSEQ)
#endif

#if ISTUB_VU_FSOR
REC_VU1_LOWER_INTERP(FSOR)
#else
REC_VU1_LOWER_CALL(FSOR)
#endif

#if ISTUB_VU_FSSET
REC_VU1_LOWER_INTERP(FSSET)
#else
REC_VU1_LOWER_CALL(FSSET)
#endif

#if ISTUB_VU_FMAND
REC_VU1_LOWER_INTERP(FMAND)
#else
REC_VU1_LOWER_CALL(FMAND)
#endif

#if ISTUB_VU_FMEQ
REC_VU1_LOWER_INTERP(FMEQ)
#else
REC_VU1_LOWER_CALL(FMEQ)
#endif

#if ISTUB_VU_FMOR
REC_VU1_LOWER_INTERP(FMOR)
#else
REC_VU1_LOWER_CALL(FMOR)
#endif

#if ISTUB_VU_FCAND
REC_VU1_LOWER_INTERP(FCAND)
#else
REC_VU1_LOWER_CALL(FCAND)
#endif

#if ISTUB_VU_FCEQ
REC_VU1_LOWER_INTERP(FCEQ)
#else
REC_VU1_LOWER_CALL(FCEQ)
#endif

#if ISTUB_VU_FCOR
REC_VU1_LOWER_INTERP(FCOR)
#else
REC_VU1_LOWER_CALL(FCOR)
#endif

#if ISTUB_VU_FCSET
REC_VU1_LOWER_INTERP(FCSET)
#else
REC_VU1_LOWER_CALL(FCSET)
#endif

#if ISTUB_VU_FCGET
REC_VU1_LOWER_INTERP(FCGET)
#else
REC_VU1_LOWER_CALL(FCGET)
#endif

// ============================================================================
//  Random number generator
// ============================================================================

#if ISTUB_VU_RINIT
REC_VU1_LOWER_INTERP(RINIT)
#else
REC_VU1_LOWER_CALL(RINIT)
#endif

#if ISTUB_VU_RGET
REC_VU1_LOWER_INTERP(RGET)
#else
REC_VU1_LOWER_CALL(RGET)
#endif

#if ISTUB_VU_RNEXT
REC_VU1_LOWER_INTERP(RNEXT)
#else
REC_VU1_LOWER_CALL(RNEXT)
#endif

#if ISTUB_VU_RXOR
REC_VU1_LOWER_INTERP(RXOR)
#else
REC_VU1_LOWER_CALL(RXOR)
#endif

// ============================================================================
//  EFU — Elementary Function Unit (VU1 only)
// ============================================================================

#if ISTUB_VU_ESADD
REC_VU1_LOWER_INTERP(ESADD)
#else
REC_VU1_LOWER_CALL(ESADD)
#endif

#if ISTUB_VU_ERSADD
REC_VU1_LOWER_INTERP(ERSADD)
#else
REC_VU1_LOWER_CALL(ERSADD)
#endif

#if ISTUB_VU_ELENG
REC_VU1_LOWER_INTERP(ELENG)
#else
REC_VU1_LOWER_CALL(ELENG)
#endif

#if ISTUB_VU_ERLENG
REC_VU1_LOWER_INTERP(ERLENG)
#else
REC_VU1_LOWER_CALL(ERLENG)
#endif

#if ISTUB_VU_EATANxy
REC_VU1_LOWER_INTERP(EATANxy)
#else
REC_VU1_LOWER_CALL(EATANxy)
#endif

#if ISTUB_VU_EATANxz
REC_VU1_LOWER_INTERP(EATANxz)
#else
REC_VU1_LOWER_CALL(EATANxz)
#endif

#if ISTUB_VU_ESUM
REC_VU1_LOWER_INTERP(ESUM)
#else
REC_VU1_LOWER_CALL(ESUM)
#endif

#if ISTUB_VU_ERCPR
REC_VU1_LOWER_INTERP(ERCPR)
#else
REC_VU1_LOWER_CALL(ERCPR)
#endif

#if ISTUB_VU_ESQRT_EFU
REC_VU1_LOWER_INTERP(ESQRT)
#else
REC_VU1_LOWER_CALL(ESQRT)
#endif

#if ISTUB_VU_ERSQRT
REC_VU1_LOWER_INTERP(ERSQRT)
#else
REC_VU1_LOWER_CALL(ERSQRT)
#endif

#if ISTUB_VU_ESIN
REC_VU1_LOWER_INTERP(ESIN)
#else
REC_VU1_LOWER_CALL(ESIN)
#endif

#if ISTUB_VU_EATAN
REC_VU1_LOWER_INTERP(EATAN)
#else
REC_VU1_LOWER_CALL(EATAN)
#endif

#if ISTUB_VU_EEXP
REC_VU1_LOWER_INTERP(EEXP)
#else
REC_VU1_LOWER_CALL(EEXP)
#endif

// ============================================================================
//  Special — VU/GIF interface
// ============================================================================

#if ISTUB_VU_XITOP
REC_VU1_LOWER_INTERP(XITOP)
#else
REC_VU1_LOWER_CALL(XITOP)
#endif

#if ISTUB_VU_XTOP
REC_VU1_LOWER_INTERP(XTOP)
#else
REC_VU1_LOWER_CALL(XTOP)
#endif

#if ISTUB_VU_XGKICK
REC_VU1_LOWER_INTERP(XGKICK)
#else
REC_VU1_LOWER_CALL(XGKICK)
#endif

// ============================================================================
//  Generic fallback emitter for unknown / reserved lower opcode slots.
//  VU1.code is already set; the interpreter will handle the unknown case
//  (logging / NOP).
// ============================================================================
static void recVU1_Lower_Unknown()
{
	armEmitCall(reinterpret_cast<const void*>(VU1_LOWER_OPCODE[VU1.code >> 25]));
}

// ============================================================================
//  LowerOP sub-table dispatch (compile-time, mirrors interpreter tables)
//
//  When recVU1_LowerTable[0x40] is called, VU1.code is set to the lower
//  instruction word.  The sub-table dispatch reads bits from VU1.code at
//  compile time to route to the correct rec function.  Non-native entries
//  fall back to recVU1_Lower_Unknown (interpreter BL chain).
// ============================================================================
using VU1RecFn = void (*)();

// T3_00 sub-table: indexed by (VU1.code >> 6) & 0x1f
static VU1RecFn recVU1_LowerOP_T3_00_Table[32] = {
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x00
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x04
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x08
	recVU1_MOVE,          recVU1_LQI,           recVU1_DIV,           recVU1_MTIR,           // 0x0C
	recVU1_RNEXT,         recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x10
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x14
	recVU1_Lower_Unknown, recVU1_MFP,           recVU1_XTOP,          recVU1_XGKICK,         // 0x18
	recVU1_ESADD,         recVU1_EATANxy,       recVU1_ESQRT,         recVU1_ESIN,           // 0x1C
};

// T3_01 sub-table: indexed by (VU1.code >> 6) & 0x1f
static VU1RecFn recVU1_LowerOP_T3_01_Table[32] = {
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x00
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x04
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x08
	recVU1_MR32,          recVU1_SQI,           recVU1_SQRT,          recVU1_MFIR,           // 0x0C
	recVU1_RGET,          recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x10
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x14
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_XITOP,         recVU1_Lower_Unknown, // 0x18
	recVU1_ERSADD,        recVU1_EATANxz,       recVU1_ERSQRT,        recVU1_EATAN,          // 0x1C
};

// T3_10 sub-table: indexed by (VU1.code >> 6) & 0x1f
static VU1RecFn recVU1_LowerOP_T3_10_Table[32] = {
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x00
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x04
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x08
	recVU1_Lower_Unknown, recVU1_LQD,           recVU1_RSQRT,         recVU1_ILWR,           // 0x0C
	recVU1_RINIT,         recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x10
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x14
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x18
	recVU1_ELENG,         recVU1_ESUM,          recVU1_ERCPR,         recVU1_EEXP,           // 0x1C
};

// T3_11 sub-table: indexed by (VU1.code >> 6) & 0x1f
static VU1RecFn recVU1_LowerOP_T3_11_Table[32] = {
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x00
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x04
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x08
	recVU1_Lower_Unknown, recVU1_SQD,           recVU1_WAITQ,         recVU1_ISWR,           // 0x0C
	recVU1_RXOR,          recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x10
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x14
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x18
	recVU1_ERLENG,        recVU1_Lower_Unknown, recVU1_WAITP,         recVU1_Lower_Unknown, // 0x1C
};

// T3 dispatch functions (compile-time)
static void recVU1_LowerOP_T3_00() { recVU1_LowerOP_T3_00_Table[(VU1.code >> 6) & 0x1f](); }
static void recVU1_LowerOP_T3_01() { recVU1_LowerOP_T3_01_Table[(VU1.code >> 6) & 0x1f](); }
static void recVU1_LowerOP_T3_10() { recVU1_LowerOP_T3_10_Table[(VU1.code >> 6) & 0x1f](); }
static void recVU1_LowerOP_T3_11() { recVU1_LowerOP_T3_11_Table[(VU1.code >> 6) & 0x1f](); }

// LowerOP main sub-table: indexed by VU1.code & 0x3f
static VU1RecFn recVU1_LowerOP_Table[64] = {
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x00
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x04
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x08
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x0C
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x10
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x14
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x18
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x1C
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x20
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x24
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x28
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x2C
	recVU1_IADD,          recVU1_ISUB,          recVU1_IADDI,         recVU1_Lower_Unknown, // 0x30
	recVU1_IAND,          recVU1_IOR,           recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x34
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, // 0x38
	recVU1_LowerOP_T3_00, recVU1_LowerOP_T3_01, recVU1_LowerOP_T3_10, recVU1_LowerOP_T3_11, // 0x3C
};

// LowerOP dispatch entry point (compile-time)
static void recVU1_LowerOP() { recVU1_LowerOP_Table[VU1.code & 0x3f](); }

// ============================================================================
//  recVU1_LowerTable[128]
//
//  Maps lower opcode index (lower_word >> 25) to a code-emitter function.
//  Layout mirrors VU1_LOWER_OPCODE in VUops.cpp.
//  Index 0x40 dispatches through recVU1_LowerOP sub-table chain.
// ============================================================================

// clang-format off
// 128 entries, 4 per line = 32 lines
VU1RecFn recVU1_LowerTable[128] = {
	// 0x00-0x03: LQ, SQ, unk, unk
	recVU1_LQ,            recVU1_SQ,            recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x04-0x07: ILW, ISW, unk, unk
	recVU1_ILW,           recVU1_ISW,           recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x08-0x0B: IADDIU, ISUBIU, unk, unk
	recVU1_IADDIU,        recVU1_ISUBIU,        recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x0C-0x0F: unk x4
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x10-0x13: FCEQ, FCSET, FCAND, FCOR
	recVU1_FCEQ,          recVU1_FCSET,         recVU1_FCAND,         recVU1_FCOR,
	// 0x14-0x17: FSEQ, FSSET, FSAND, FSOR
	recVU1_FSEQ,          recVU1_FSSET,         recVU1_FSAND,         recVU1_FSOR,
	// 0x18-0x1B: FMEQ, unk, FMAND, FMOR
	recVU1_FMEQ,          recVU1_Lower_Unknown, recVU1_FMAND,         recVU1_FMOR,
	// 0x1C-0x1F: FCGET, unk, unk, unk
	recVU1_FCGET,         recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x20-0x23: B, BAL, unk, unk
	recVU1_B,             recVU1_BAL,           recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x24-0x27: JR, JALR, unk, unk
	recVU1_JR,            recVU1_JALR,          recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x28-0x2B: IBEQ, IBNE, unk, unk
	recVU1_IBEQ,          recVU1_IBNE,          recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x2C-0x2F: IBLTZ, IBGTZ, IBLEZ, IBGEZ
	recVU1_IBLTZ,         recVU1_IBGTZ,         recVU1_IBLEZ,         recVU1_IBGEZ,
	// 0x30-0x3F: unknown x16
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x40: LowerOP (sub-table dispatch via recVU1_LowerOP)
	// 0x41-0x43: unknown
	recVU1_LowerOP,       recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x44-0x47: unknown x4
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x48-0x4B: unknown x4
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x4C-0x4F: unknown x4
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x50-0x5F: unknown x16
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x60-0x6F: unknown x16
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	// 0x70-0x7F: unknown x16
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
	recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown, recVU1_Lower_Unknown,
};
// clang-format on
