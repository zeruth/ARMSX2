// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Lower Instruction Stubs
// Integer ALU, load/store, branches, FDIV, flag ops,
// move/transfer, random, EFU, special (XITOP/XTOP/XGKICK)

#include "Common.h"
#include "VUops.h"
#include "VU.h"
#include "Vif.h"
#include "Vif_Dma.h"
#include "Gif_Unit.h"
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

// Compile-time current pair PC. Set by the per-pair dispatch loop in
// iVU1micro_arm64.cpp before each lower-op emit. Used by native branch
// emitters to resolve PC-relative targets at compile time (since step 2
// of the dispatch loop has already stored (pair_pc+8) & VU1_PROGMASK into
// VI[REG_TPC], runtime TPC is compile-time predictable here).
u32 g_vu1CurrentPC = 0;

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

// XGKICK capture scratch — arm64 analogue of microVU's mVU.VIxgkick. Holds
// the pending XGKICK's VU1.Mem offset between the capture at pair k and the
// deferred fire at (end of) pair k+1. Critically, this is NOT VU1.xgkickenable
// / VU1.xgkickaddr — touching VU1.xgkickenable would trip _vuTestPipes
// (VUops.cpp:203), which calls _vuXGKICKTransfer whenever xgkickenable is set
// and xgkickcyclecount >= 2. Under REC_VU1=true that loop walks
// GetGSPacketSize without the Gif_Unit.h:606 early-exit branch setting the
// EOP top bit, so xgkickendpacket stays 0 and the loop iterates past the
// packet, trips the 0x4000 guard, and cancels valid kicks. By stashing the
// addr in a private static and firing via the direct gifUnit path, the arm64
// rec avoids _vuTestPipes's xgkick branch entirely (mirroring microVU's
// non-hack path which only touches mVU.VIxgkick, never VU1.xgkickenable).
//
// Safe as a file-local static: every compiled VU1 block drains any pending
// kick before returning (pair-loop step 15 or the CompileBlock block-end
// drain), so this never carries state across block boundaries.
static u32 s_vu1_pending_xgkick_addr;

// Forward decl — defined just below. Fires a pending XGKICK via the direct
// gifUnit path. Only called when the compile-time pending_xgkick_fire tracker
// in CompileBlock says a kick is pending, so no runtime pending flag needed.
void vu1_XGKICK_fire_deferred(VURegs* VU);

static void vu1_XGKICK(VURegs* VU)
{
	// Capture only. No VU1.xgkickenable / VU1.xgkickaddr / VPU_STAT writes —
	// keeping VU1 state untouched is what prevents _vuTestPipes (called at
	// step 6 of every following pair in CompileBlock) from observing "kick
	// pending" and walking into the broken _vuXGKICKTransfer loop. Back-to-
	// back XGKICKs are sequenced at compile time in CompileBlock, so no
	// flush-prior path here.
	s_vu1_pending_xgkick_addr = (VU->VI[W_Is(VU)].US[0] & 0x3ff) * 16;
}

// Deferred XGKICK transfer. Emitted by CompileBlock one pair after an XGKICK
// (or immediately before a back-to-back XGKICK capture) so any VU store on
// the intervening pair has committed before the GIF walks VU1.Mem. Arm64
// analogue of microVU's mVU_XGKICK_ / mVU_XGKICK_DELAY path.
//
// IMPORTANT: we deliberately bypass _vuXGKICKTransfer here. That loop is
// written for the interpreter/REC_VU1=false configuration and relies on
// GetGSPacketSize returning with the EOP top-bit set (Gif_Unit.h:606 early-
// exit branch) to know when to stop. With REC_VU1=true the early-exit is
// disabled and line 612 returns without the top bit, so _vuXGKICKTransfer
// never sees xgkickendpacket, re-iterates past the packet, walks stale
// memory, hits the 0x4000 guard and cancels the kick. microVU sidesteps this
// by calling gifUnit.GetGSPacketSize + TransferGSPacketData directly in
// mVU_XGKICK_ (microVU_Lower.inl:1698). This function is the arm64 analogue.
//
// VU arg is unused (kept for ABI compat with armEmitCall, which loads x0 =
// VU1_BASE_REG at every call site).
void vu1_XGKICK_fire_deferred(VURegs* VU)
{
	(void)VU;

	const u32 addr = s_vu1_pending_xgkick_addr & 0x3FF0u;
	const u32 diff = 0x4000u - addr;
	u32 size = gifUnit.GetGSPacketSize(GIF_PATH_1, VU1.Mem, addr, ~0u, true);
	size &= 0xFFFFu; // strip the EOP top bit the early-exit path sets when REC_VU1=false

	if (size == 0)
	{
		// 0x4000 guard tripped — silently drop the kick, matching microVU's
		// mVU_XGKICK_ which would TransferGSPacketData(..., 0, ...). BIOS
		// does this once at boot (see Gif_Unit.h:602 comment).
		return;
	}

	if (size > diff)
	{
		// Wrap: tail of VU memory + wraparound prefix. CopyGSPacketData for
		// the tail so PATH3 arbitration stays sane (mVU_XGKICK_ does the same).
		gifUnit.gifPath[GIF_PATH_1].CopyGSPacketData(&VU1.Mem[addr], diff, true);
		gifUnit.TransferGSPacketData(GIF_TRANS_XGKICK, &VU1.Mem[0], size - diff, true);
	}
	else
	{
		gifUnit.TransferGSPacketData(GIF_TRANS_XGKICK, &VU1.Mem[addr], size, true);
	}

	// VGW release is handled by gifUnit.Execute on the EE thread (Gif_Unit.h:825).
	// We intentionally do NOT touch vif1Regs.stat.VGW or CPU_INT here — under
	// THREAD_VU1 this runs on the MTVU thread and any write to cpuRegs /
	// vif1Regs from here is a cross-thread race. microVU's mVU_XGKICK_
	// (microVU_Lower.inl:1698) takes the same approach: it never touches VGW,
	// leaving that to the GIF arbitration loop on the EE thread.
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

// Decode the 10-bit signed immediate used by LQ/SQ/ILW/ISW.
static inline s32 decodeVuImm10(u32 code)
{
	return (code & 0x400) ? static_cast<s32>((code & 0x3ff) | 0xfffffc00u)
	                      : static_cast<s32>(code & 0x3ff);
}

// Compute the effective VU1.Mem byte offset for (VI[is_reg] + imm) * 16,
// masked to 14 bits with 16-byte alignment (matches u16 cast + & 0x3FFF).
// Result: w0 = 14-bit masked offset (and x0 holds the zero-extended version).
// Clobbers w0.
static void emitComputeVuMemOffset(u32 is_reg, s32 imm)
{
	armAsm->Ldrsh(w0, MemOperand(VU1_BASE_REG, viOff(is_reg)));
	if (imm > 0)
		armAsm->Add(w0, w0, imm);
	else if (imm < 0)
		armAsm->Sub(w0, w0, static_cast<u32>(-imm));
	armAsm->Lsl(w0, w0, 4);
	armAsm->And(w0, w0, 0x3FF0);
}

#if ISTUB_VU_LQ
REC_VU1_LOWER_INTERP(LQ)
#else
void recVU1_LQ() {
	const u32 ft = W_Ft(&VU1);
	if (ft == 0) return;
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const s32 imm = decodeVuImm10(VU1.code);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitComputeVuMemOffset(is, imm);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0); // x0 is zero-extended from And

	if (xyzw == 0xF)
	{
		armAsm->Ldr(q0, MemOperand(x1));
		armAsm->Str(q0, MemOperand(VU1_BASE_REG, vfOff(ft)));
	}
	else
	{
		if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(x1,  0)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  0)); }
		if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(x1,  4)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  4)); }
		if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(x1,  8)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  8)); }
		if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(x1, 12)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 12)); }
	}
}
#endif

#if ISTUB_VU_LQD
REC_VU1_LOWER_INTERP(LQD)
#else
void recVU1_LQD() {
	const u32 ft = W_Ft(&VU1);
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	// Interpreter unconditionally backs up VI[is] before the decrement.
	emitBackupVI(is);

	// Pre-decrement VI[is] — gated on is != 0 so VI[0] (hardwired 0) is left alone.
	if (is != 0)
	{
		armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(is)));
		armAsm->Sub(w2, w2, 1);
		armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(is)));
	}

	if (ft == 0) return;

	// Address = (VI[is] & 0x3FF) * 16 — interpreter uses US[0] (unsigned), but
	// the final 0x3FF0 mask makes sign of the 16->32 extension irrelevant.
	emitComputeVuMemOffset(is, 0);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	if (xyzw == 0xF)
	{
		armAsm->Ldr(q0, MemOperand(x1));
		armAsm->Str(q0, MemOperand(VU1_BASE_REG, vfOff(ft)));
	}
	else
	{
		if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(x1,  0)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  0)); }
		if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(x1,  4)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  4)); }
		if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(x1,  8)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  8)); }
		if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(x1, 12)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 12)); }
	}
}
#endif

#if ISTUB_VU_LQI
REC_VU1_LOWER_INTERP(LQI)
#else
void recVU1_LQI() {
	const u32 ft = W_Ft(&VU1);
	const u32 is = W_Is(&VU1);
	const u32 fs = W_Fs(&VU1); // interpreter gates the increment on W_Fs (bit 15 quirk)
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitBackupVI(is);

	// Load uses VI[is] pre-increment.
	if (ft != 0)
	{
		emitComputeVuMemOffset(is, 0);
		armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
		armAsm->Add(x1, x1, x0);

		if (xyzw == 0xF)
		{
			armAsm->Ldr(q0, MemOperand(x1));
			armAsm->Str(q0, MemOperand(VU1_BASE_REG, vfOff(ft)));
		}
		else
		{
			if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(x1,  0)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  0)); }
			if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(x1,  4)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  4)); }
			if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(x1,  8)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) +  8)); }
			if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(x1, 12)); armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 12)); }
		}
	}

	// Post-increment — interpreter gates on fs != 0, not is != 0.
	if (fs != 0)
	{
		armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(is)));
		armAsm->Add(w2, w2, 1);
		armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(is)));
	}
}
#endif

#if ISTUB_VU_SQ
REC_VU1_LOWER_INTERP(SQ)
#else
void recVU1_SQ() {
	const u32 fs = W_Fs(&VU1);
	const u32 it = W_It(&VU1); // SQ uses It as the base register
	const u32 xyzw = W_XYZW(&VU1);
	const s32 imm = decodeVuImm10(VU1.code);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitComputeVuMemOffset(it, imm);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	if (xyzw == 0xF)
	{
		armAsm->Ldr(q0, MemOperand(VU1_BASE_REG, vfOff(fs)));
		armAsm->Str(q0, MemOperand(x1));
	}
	else
	{
		if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  0)); armAsm->Str(w2, MemOperand(x1,  0)); }
		if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  4)); armAsm->Str(w2, MemOperand(x1,  4)); }
		if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  8)); armAsm->Str(w2, MemOperand(x1,  8)); }
		if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) + 12)); armAsm->Str(w2, MemOperand(x1, 12)); }
	}
}
#endif

#if ISTUB_VU_SQD
REC_VU1_LOWER_INTERP(SQD)
#else
void recVU1_SQD() {
	const u32 fs = W_Fs(&VU1);
	const u32 it = W_It(&VU1);
	const u32 ft = W_Ft(&VU1); // interpreter gates the decrement on W_Ft, not W_It
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitBackupVI(it);

	// Pre-decrement — interpreter quirk: gated on ft != 0, not it != 0.
	if (ft != 0)
	{
		armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
		armAsm->Sub(w2, w2, 1);
		armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
	}

	emitComputeVuMemOffset(it, 0);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	if (xyzw == 0xF)
	{
		armAsm->Ldr(q0, MemOperand(VU1_BASE_REG, vfOff(fs)));
		armAsm->Str(q0, MemOperand(x1));
	}
	else
	{
		if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  0)); armAsm->Str(w2, MemOperand(x1,  0)); }
		if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  4)); armAsm->Str(w2, MemOperand(x1,  4)); }
		if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  8)); armAsm->Str(w2, MemOperand(x1,  8)); }
		if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) + 12)); armAsm->Str(w2, MemOperand(x1, 12)); }
	}
}
#endif

#if ISTUB_VU_SQI
REC_VU1_LOWER_INTERP(SQI)
#else
void recVU1_SQI() {
	const u32 fs = W_Fs(&VU1);
	const u32 it = W_It(&VU1);
	const u32 ft = W_Ft(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitBackupVI(it);

	emitComputeVuMemOffset(it, 0);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	if (xyzw == 0xF)
	{
		armAsm->Ldr(q0, MemOperand(VU1_BASE_REG, vfOff(fs)));
		armAsm->Str(q0, MemOperand(x1));
	}
	else
	{
		if (xyzw & 8) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  0)); armAsm->Str(w2, MemOperand(x1,  0)); }
		if (xyzw & 4) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  4)); armAsm->Str(w2, MemOperand(x1,  4)); }
		if (xyzw & 2) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) +  8)); armAsm->Str(w2, MemOperand(x1,  8)); }
		if (xyzw & 1) { armAsm->Ldr(w2, MemOperand(VU1_BASE_REG, vfOff(fs) + 12)); armAsm->Str(w2, MemOperand(x1, 12)); }
	}

	// Post-increment — same W_Ft gate as SQD's decrement.
	if (ft != 0)
	{
		armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
		armAsm->Add(w2, w2, 1);
		armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
	}
}
#endif

#if ISTUB_VU_ILW
REC_VU1_LOWER_INTERP(ILW)
#else
void recVU1_ILW() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const s32 imm = decodeVuImm10(VU1.code);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitComputeVuMemOffset(is, imm);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	// ptr is u16*: ptr[0]/ptr[2]/ptr[4]/ptr[6] = byte offsets 0/4/8/12.
	// Only the final set bit's value survives — match by emitting in order.
	if (xyzw & 8) { armAsm->Ldrh(w2, MemOperand(x1,  0)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 4) { armAsm->Ldrh(w2, MemOperand(x1,  4)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 2) { armAsm->Ldrh(w2, MemOperand(x1,  8)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 1) { armAsm->Ldrh(w2, MemOperand(x1, 12)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
}
#endif

#if ISTUB_VU_ISW
REC_VU1_LOWER_INTERP(ISW)
#else
void recVU1_ISW() {
	const u32 it = W_It(&VU1);
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const s32 imm = decodeVuImm10(VU1.code);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitComputeVuMemOffset(is, imm);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	// Source is VI[it].US[0] zero-extended; wrapper stores the u16 then zeros
	// the next u16, which equals a 32-bit store of the zero-extended value.
	armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
	if (xyzw & 8) armAsm->Str(w2, MemOperand(x1,  0));
	if (xyzw & 4) armAsm->Str(w2, MemOperand(x1,  4));
	if (xyzw & 2) armAsm->Str(w2, MemOperand(x1,  8));
	if (xyzw & 1) armAsm->Str(w2, MemOperand(x1, 12));
}
#endif

#if ISTUB_VU_ILWR
REC_VU1_LOWER_INTERP(ILWR)
#else
void recVU1_ILWR() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitBackupVI(it);

	emitComputeVuMemOffset(is, 0);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	// ptr is u16*: ptr[0]/ptr[2]/ptr[4]/ptr[6] = byte offsets 0/4/8/12.
	// Last-set xyzw bit wins — emit in order to match interpreter.
	if (xyzw & 8) { armAsm->Ldrh(w2, MemOperand(x1,  0)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 4) { armAsm->Ldrh(w2, MemOperand(x1,  4)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 2) { armAsm->Ldrh(w2, MemOperand(x1,  8)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
	if (xyzw & 1) { armAsm->Ldrh(w2, MemOperand(x1, 12)); armAsm->Strh(w2, MemOperand(VU1_BASE_REG, viOff(it))); }
}
#endif

#if ISTUB_VU_ISWR
REC_VU1_LOWER_INTERP(ISWR)
#else
void recVU1_ISWR() {
	const u32 it = W_It(&VU1);
	const u32 is = W_Is(&VU1);
	const u32 xyzw = W_XYZW(&VU1);
	const int64_t mem_off = static_cast<int64_t>(offsetof(VURegs, Mem));

	emitComputeVuMemOffset(is, 0);
	armAsm->Ldr(x1, MemOperand(VU1_BASE_REG, mem_off));
	armAsm->Add(x1, x1, x0);

	// Source is VI[it].US[0] zero-extended to 32 bits (interpreter stores the
	// u16 then zeros the adjacent u16 — equivalent to a 32-bit zero-extended store).
	armAsm->Ldrh(w2, MemOperand(VU1_BASE_REG, viOff(it)));
	if (xyzw & 8) armAsm->Str(w2, MemOperand(x1,  0));
	if (xyzw & 4) armAsm->Str(w2, MemOperand(x1,  4));
	if (xyzw & 2) armAsm->Str(w2, MemOperand(x1,  8));
	if (xyzw & 1) armAsm->Str(w2, MemOperand(x1, 12));
}
#endif

// ============================================================================
//  Branches
// ============================================================================

// Emit inline vu1SetBranch(VU, bpc). bpc is in w_bpc. Clobbers w4, w5.
// Mirrors _setBranch() in VUops.cpp: if already in a delay slot (branch==1),
// deferred onto delaybranchpc/takedelaybranch; otherwise sets branch=2 and
// branchpc so the per-pair countdown (step 12) fires it two pairs later.
static void emitInlineSetBranch(const Register& w_bpc)
{
	const int64_t branch_off        = static_cast<int64_t>(offsetof(VURegs, branch));
	const int64_t branchpc_off      = static_cast<int64_t>(offsetof(VURegs, branchpc));
	const int64_t delaybranchpc_off = static_cast<int64_t>(offsetof(VURegs, delaybranchpc));
	const int64_t takedelay_off     = static_cast<int64_t>(offsetof(VURegs, takedelaybranch));

	a64::Label is_delay, done;
	armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, branch_off));
	armAsm->Cmp(w4, 1);
	armAsm->B(&is_delay, a64::eq);

	// Normal path: branch != 1 → set branch=2, branchpc=bpc
	armAsm->Mov(w5, 2);
	armAsm->Str(w5, MemOperand(VU1_BASE_REG, branch_off));
	armAsm->Str(w_bpc, MemOperand(VU1_BASE_REG, branchpc_off));
	armAsm->B(&done);

	// Delay-slot-in-delay-slot path: branch == 1 → queue via delaybranchpc
	armAsm->Bind(&is_delay);
	armAsm->Str(w_bpc, MemOperand(VU1_BASE_REG, delaybranchpc_off));
	armAsm->Mov(w5, 1);
	armAsm->Strb(w5, MemOperand(VU1_BASE_REG, takedelay_off));

	armAsm->Bind(&done);
}

// Emit hazard-corrected VI[reg].US[0] read.
// The 2-cycle integer pipeline hazard: if VIBackupCycles > 0 and VIRegNumber
// matches `reg`, return VIOldValue (which holds the pre-write u16) instead of
// VI[reg].US[0]. reg is a compile-time constant.
//   dest  : destination w-reg (result)
//   reg   : VI register number (0..15)
//   signed_read : emit Ldrsh instead of Ldrh (so IBLTZ/IBGTZ/etc. can compare signed)
//   w_tmp : scratch w-reg, clobbered
static void emitHazardVIRead(const Register& dest, u32 reg, bool signed_read, const Register& w_tmp)
{
	const int64_t vibackup_off = static_cast<int64_t>(offsetof(VURegs, VIBackupCycles));
	const int64_t viregnum_off = static_cast<int64_t>(offsetof(VURegs, VIRegNumber));
	const int64_t violdval_off = static_cast<int64_t>(offsetof(VURegs, VIOldValue));

	if (signed_read)
		armAsm->Ldrsh(dest, MemOperand(VU1_BASE_REG, viOff(reg)));
	else
		armAsm->Ldrh(dest, MemOperand(VU1_BASE_REG, viOff(reg)));

	a64::Label done;
	armAsm->Ldrb(w_tmp, MemOperand(VU1_BASE_REG, vibackup_off));
	armAsm->Cbz(w_tmp, &done);                       // no backup active
	armAsm->Ldr(w_tmp, MemOperand(VU1_BASE_REG, viregnum_off));
	armAsm->Cmp(w_tmp, reg);
	armAsm->B(&done, a64::ne);                        // different reg
	// VIOldValue is u32 but holds the original u16 in its low halfword.
	// Ldrsh/Ldrh from the low halfword matches the interpreter's
	// `src = VU->VIOldValue` (truncation to s16 / u16 on assignment).
	if (signed_read)
		armAsm->Ldrsh(dest, MemOperand(VU1_BASE_REG, violdval_off));
	else
		armAsm->Ldrh(dest, MemOperand(VU1_BASE_REG, violdval_off));
	armAsm->Bind(&done);
}

// Compute compile-time branch target for PC-relative branches (B/BAL/IBxx).
// At emit time, step 2 of the per-pair loop has already stored
// (pair_pc+8) & VU1_PROGMASK into VI[REG_TPC], so runtime TPC is known and
// the full _branchAddr() formula resolves to a constant.
static inline u32 vu1ComputePCRelTarget()
{
	const s32 pair_pc = static_cast<s32>(g_vu1CurrentPC);
	const s32 imm11   = W_Imm11(&VU1);
	const s32 tpc_val = static_cast<s32>((pair_pc + 8) & 0x3fff);
	return static_cast<u32>((tpc_val + imm11 * 8) & 0x3fff);
}

// Emit BAL/JALR link register write. Called before emitInlineSetBranch.
// The link value matches _vuBAL/_vuJALR in VUops.cpp: when already in a
// delay slot (branch==1) we link to branchpc+8 (runtime), otherwise we
// link to TPC+8 (compile-time since step 2 has already written TPC).
// Does not touch branch state. it must be nonzero.
static void emitBranchLinkWrite(u32 it)
{
	const int64_t branch_off   = static_cast<int64_t>(offsetof(VURegs, branch));
	const int64_t branchpc_off = static_cast<int64_t>(offsetof(VURegs, branchpc));
	const u32 runtime_tpc = (g_vu1CurrentPC + 8) & 0x3fff;
	const u32 link_normal = (runtime_tpc + 8) / 8;

	a64::Label is_delay, done;
	armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, branch_off));
	armAsm->Cmp(w4, 1);
	armAsm->B(&is_delay, a64::eq);

	// Normal link: (TPC+8)/8 — compile-time constant
	armAsm->Mov(w5, link_normal);
	armAsm->Strh(w5, MemOperand(VU1_BASE_REG, viOff(it)));
	armAsm->B(&done);

	// In-delay-slot link: (branchpc+8)/8 — runtime
	armAsm->Bind(&is_delay);
	armAsm->Ldr(w4, MemOperand(VU1_BASE_REG, branchpc_off));
	armAsm->Add(w4, w4, 8);
	armAsm->Lsr(w4, w4, 3);
	armAsm->Strh(w4, MemOperand(VU1_BASE_REG, viOff(it)));

	armAsm->Bind(&done);
}

#if ISTUB_VU_B
REC_VU1_LOWER_INTERP(B)
#else
void recVU1_B() {
	const u32 bpc = vu1ComputePCRelTarget();
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
}
#endif

#if ISTUB_VU_BAL
REC_VU1_LOWER_INTERP(BAL)
#else
void recVU1_BAL() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 it  = W_It(&VU1);
	if (it != 0)
		emitBranchLinkWrite(it);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
}
#endif

#if ISTUB_VU_JR
REC_VU1_LOWER_INTERP(JR)
#else
void recVU1_JR() {
	const u32 is = W_Is(&VU1);
	// bpc = VI[is].US[0] * 8  (interpreter does no hazard check here)
	armAsm->Ldrh(w3, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Lsl(w3, w3, 3);
	emitInlineSetBranch(w3);
}
#endif

#if ISTUB_VU_JALR
REC_VU1_LOWER_INTERP(JALR)
#else
void recVU1_JALR() {
	const u32 is = W_Is(&VU1);
	const u32 it = W_It(&VU1);
	// Compute bpc into w3 first (preserved across link write — which only touches w4/w5)
	armAsm->Ldrh(w3, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Lsl(w3, w3, 3);
	if (it != 0)
		emitBranchLinkWrite(it);
	emitInlineSetBranch(w3);
}
#endif

#if ISTUB_VU_IBEQ
REC_VU1_LOWER_INTERP(IBEQ)
#else
void recVU1_IBEQ() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 it  = W_It(&VU1);
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, it, false, w4);
	emitHazardVIRead(w7, is, false, w4);
	armAsm->Cmp(w6, w7);
	armAsm->B(&not_taken, a64::ne);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
#endif

#if ISTUB_VU_IBNE
REC_VU1_LOWER_INTERP(IBNE)
#else
void recVU1_IBNE() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 it  = W_It(&VU1);
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, it, false, w4);
	emitHazardVIRead(w7, is, false, w4);
	armAsm->Cmp(w6, w7);
	armAsm->B(&not_taken, a64::eq);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
#endif

#if ISTUB_VU_IBLTZ
REC_VU1_LOWER_INTERP(IBLTZ)
#else
void recVU1_IBLTZ() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, is, true, w4);
	armAsm->Cmp(w6, 0);
	armAsm->B(&not_taken, a64::ge);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
#endif

#if ISTUB_VU_IBGTZ
REC_VU1_LOWER_INTERP(IBGTZ)
#else
void recVU1_IBGTZ() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, is, true, w4);
	armAsm->Cmp(w6, 0);
	armAsm->B(&not_taken, a64::le);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
#endif

#if ISTUB_VU_IBLEZ
REC_VU1_LOWER_INTERP(IBLEZ)
#else
void recVU1_IBLEZ() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, is, true, w4);
	armAsm->Cmp(w6, 0);
	armAsm->B(&not_taken, a64::gt);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
#endif

#if ISTUB_VU_IBGEZ
REC_VU1_LOWER_INTERP(IBGEZ)
#else
void recVU1_IBGEZ() {
	const u32 bpc = vu1ComputePCRelTarget();
	const u32 is  = W_Is(&VU1);
	a64::Label not_taken;
	emitHazardVIRead(w6, is, true, w4);
	armAsm->Cmp(w6, 0);
	armAsm->B(&not_taken, a64::lt);
	armAsm->Mov(w3, bpc);
	emitInlineSetBranch(w3);
	armAsm->Bind(&not_taken);
}
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

// Status flag imm: bit 21 -> bit 11, bits [10:0] as-is. Max value 0xFFF.
static inline u32 decodeFSImm(u32 code)
{
	return (((code >> 21) & 0x1) << 11) | (code & 0x7ff);
}

#if ISTUB_VU_FSAND
REC_VU1_LOWER_INTERP(FSAND)
#else
void recVU1_FSAND() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 imm = decodeFSImm(VU1.code);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(REG_STATUS_FLAG)));
	if (imm == 0)
		armAsm->Mov(w0, 0);
	else
	{
		armAsm->Mov(w1, imm);
		armAsm->And(w0, w0, w1);
	}
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FSEQ
REC_VU1_LOWER_INTERP(FSEQ)
#else
void recVU1_FSEQ() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 imm = decodeFSImm(VU1.code);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(REG_STATUS_FLAG)));
	armAsm->And(w0, w0, 0xFFF);
	armAsm->Mov(w1, imm);
	armAsm->Cmp(w0, w1);
	armAsm->Cset(w0, a64::eq);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FSOR
REC_VU1_LOWER_INTERP(FSOR)
#else
void recVU1_FSOR() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 imm = decodeFSImm(VU1.code);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(REG_STATUS_FLAG)));
	armAsm->And(w0, w0, 0xFFF);
	if (imm != 0)
	{
		armAsm->Mov(w1, imm);
		armAsm->Orr(w0, w0, w1);
	}
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FSSET
REC_VU1_LOWER_INTERP(FSSET)
#else
void recVU1_FSSET() {
	const u32 imm     = decodeFSImm(VU1.code);
	const u32 top     = imm & 0xFC0;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, sf_off));
	armAsm->And(w0, w0, 0x3F);
	if (top != 0)
	{
		armAsm->Mov(w1, top);
		armAsm->Orr(w0, w0, w1);
	}
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, sf_off));
}
#endif

#if ISTUB_VU_FMAND
REC_VU1_LOWER_INTERP(FMAND)
#else
void recVU1_FMAND() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 is = W_Is(&VU1);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(REG_MAC_FLAG)));
	armAsm->And(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FMEQ
REC_VU1_LOWER_INTERP(FMEQ)
#else
void recVU1_FMEQ() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 is = W_Is(&VU1);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(REG_MAC_FLAG)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Cmp(w0, w1);
	armAsm->Cset(w0, a64::eq);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FMOR
REC_VU1_LOWER_INTERP(FMOR)
#else
void recVU1_FMOR() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	const u32 is = W_Is(&VU1);
	armAsm->Ldrh(w0, MemOperand(VU1_BASE_REG, viOff(REG_MAC_FLAG)));
	armAsm->Ldrh(w1, MemOperand(VU1_BASE_REG, viOff(is)));
	armAsm->Orr(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU_FCAND
REC_VU1_LOWER_INTERP(FCAND)
#else
void recVU1_FCAND() {
	const u32 imm = VU1.code & 0xFFFFFF;
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_CLIP_FLAG)));
	if (imm == 0)
		armAsm->Mov(w0, 0);
	else
	{
		armAsm->Mov(w1, imm);
		armAsm->Tst(w0, w1);
		armAsm->Cset(w0, a64::ne);
	}
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(1)));
}
#endif

#if ISTUB_VU_FCEQ
REC_VU1_LOWER_INTERP(FCEQ)
#else
void recVU1_FCEQ() {
	const u32 imm = VU1.code & 0xFFFFFF;
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_CLIP_FLAG)));
	armAsm->And(w0, w0, 0xFFFFFF);
	armAsm->Mov(w1, imm);
	armAsm->Cmp(w0, w1);
	armAsm->Cset(w0, a64::eq);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(1)));
}
#endif

#if ISTUB_VU_FCOR
REC_VU1_LOWER_INTERP(FCOR)
#else
void recVU1_FCOR() {
	const u32 imm = VU1.code & 0xFFFFFF;
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_CLIP_FLAG)));
	armAsm->And(w0, w0, 0xFFFFFF);
	if (imm != 0)
	{
		armAsm->Mov(w1, imm);
		armAsm->Orr(w0, w0, w1);
	}
	armAsm->Mov(w1, 0xFFFFFFu);
	armAsm->Cmp(w0, w1);
	armAsm->Cset(w0, a64::eq);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(1)));
}
#endif

#if ISTUB_VU_FCSET
REC_VU1_LOWER_INTERP(FCSET)
#else
void recVU1_FCSET() {
	const u32 imm = VU1.code & 0xFFFFFF;
	const int64_t cf_off = static_cast<int64_t>(offsetof(VURegs, clipflag));
	if (imm == 0)
		armAsm->Str(wzr, MemOperand(VU1_BASE_REG, cf_off));
	else
	{
		armAsm->Mov(w0, imm);
		armAsm->Str(w0, MemOperand(VU1_BASE_REG, cf_off));
	}
}
#endif

#if ISTUB_VU_FCGET
REC_VU1_LOWER_INTERP(FCGET)
#else
void recVU1_FCGET() {
	const u32 it = W_It(&VU1);
	if (it == 0) return;
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_CLIP_FLAG)));
	armAsm->And(w0, w0, 0xFFF);
	armAsm->Strh(w0, MemOperand(VU1_BASE_REG, viOff(it)));
}
#endif

// ============================================================================
//  Random number generator
// ============================================================================

#if ISTUB_VU_RINIT
REC_VU1_LOWER_INTERP(RINIT)
#else
void recVU1_RINIT() {
	const u32 fs  = W_Fs(&VU1);
	const u32 fsf = W_Fsf(&VU1);
	// VI[REG_R].UL = 0x3F800000 | (VF[fs].UL[fsf] & 0x007FFFFF)
	armAsm->Mov(w0, 0x3F800000u);
	armAsm->Ldr(w1, MemOperand(VU1_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Bfi(w0, w1, 0, 23);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, viOff(REG_R)));
}
#endif

#if ISTUB_VU_RGET
REC_VU1_LOWER_INTERP(RGET)
#else
void recVU1_RGET() {
	const u32 ft = W_Ft(&VU1);
	if (ft == 0) return;
	const u32 xyzw = W_XYZW(&VU1);
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_R)));
	if (xyzw & 8) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));
	if (xyzw & 4) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));
	if (xyzw & 2) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));
	if (xyzw & 1) armAsm->Str(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
}
#endif

#if ISTUB_VU_RNEXT
REC_VU1_LOWER_INTERP(RNEXT)
#else
void recVU1_RNEXT() {
	const u32 ft = W_Ft(&VU1);
	if (ft == 0) return;
	const u32 xyzw = W_XYZW(&VU1);
	// LFSR advance (mirrors AdvanceLFSR in VUops.cpp):
	//   x = (R >> 4) & 1
	//   y = (R >> 22) & 1
	//   R = ((R << 1) ^ (x ^ y)) with exponent forced to 0x3F800000
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_R)));
	armAsm->Ubfx(w1, w0, 4, 1);            // w1 = x
	armAsm->Ubfx(w2, w0, 22, 1);           // w2 = y
	armAsm->Eor(w1, w1, w2);               // w1 = x ^ y
	armAsm->Lsl(w0, w0, 1);                // R <<= 1 (bit0 becomes 0)
	armAsm->Orr(w0, w0, w1);               // bit0 = x^y
	armAsm->Mov(w2, 0x3F800000u);
	armAsm->Bfi(w2, w0, 0, 23);            // w2 = 0x3F800000 | (R & 0x7FFFFF)
	armAsm->Str(w2, MemOperand(VU1_BASE_REG, viOff(REG_R)));
	// Broadcast to selected VF[ft] components.
	if (xyzw & 8) armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 0));
	if (xyzw & 4) armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 4));
	if (xyzw & 2) armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 8));
	if (xyzw & 1) armAsm->Str(w2, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
}
#endif

#if ISTUB_VU_RXOR
REC_VU1_LOWER_INTERP(RXOR)
#else
void recVU1_RXOR() {
	const u32 fs  = W_Fs(&VU1);
	const u32 fsf = W_Fsf(&VU1);
	// VI[REG_R].UL = 0x3F800000 | ((VI[REG_R].UL ^ VF[fs].UL[fsf]) & 0x007FFFFF)
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, viOff(REG_R)));
	armAsm->Ldr(w1, MemOperand(VU1_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Eor(w1, w0, w1);
	armAsm->Mov(w0, 0x3F800000u);
	armAsm->Bfi(w0, w1, 0, 23);
	armAsm->Str(w0, MemOperand(VU1_BASE_REG, viOff(REG_R)));
}
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
