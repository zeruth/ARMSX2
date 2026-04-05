// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Upper Instruction Stubs
// FMAC arithmetic (ADD/SUB/MUL/MADD/MSUB with xyzwqi broadcasts),
// accumulator variants (ADDA/SUBA/MULA/MADDA/MSUBA),
// MAX, MINI, ABS, CLIP, FTOI/ITOF, OPMULA, OPMSUB, NOP

#include "Common.h"
#include "VUops.h"
#include "VU.h"
#include "VUflags.h"
#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"

using namespace vixl::aarch64;

// ============================================================================
//  Native NEON codegen helpers
// ============================================================================

static const auto VU1_BASE_REG = x23;

static constexpr int64_t vfOff(u32 reg)
{
	return static_cast<int64_t>(offsetof(VURegs, VF)) + reg * static_cast<int64_t>(sizeof(VECTOR));
}

static constexpr int64_t viOff(u32 reg)
{
	return static_cast<int64_t>(offsetof(VURegs, VI)) + reg * static_cast<int64_t>(sizeof(REG_VI));
}

static constexpr int64_t accOff()
{
	return static_cast<int64_t>(offsetof(VURegs, ACC));
}

// C helper: MAC flag update + result clamping + writeback.
// Called from JIT after NEON math. Takes the 4 float results, applies
// VU_MACx_UPDATE (which clamps overflow/underflow and sets per-component flags),
// writes to destination with XYZW mask, and updates statusflag.
static void vu1_fmac_writeback(VURegs* VU, VECTOR* dst, u32 xyzw,
                                float rx, float ry, float rz, float rw)
{
	// VF[0] is hardwired to {0,0,0,1} — discard writes but still update flags
	const bool write = (dst != &VU->VF[0]);
	if (xyzw & 8) { u32 v = VU_MACx_UPDATE(VU, rx); if (write) dst->i.x = v; }
	else VU_MACx_CLEAR(VU);
	if (xyzw & 4) { u32 v = VU_MACy_UPDATE(VU, ry); if (write) dst->i.y = v; }
	else VU_MACy_CLEAR(VU);
	if (xyzw & 2) { u32 v = VU_MACz_UPDATE(VU, rz); if (write) dst->i.z = v; }
	else VU_MACz_CLEAR(VU);
	if (xyzw & 1) { u32 v = VU_MACw_UPDATE(VU, rw); if (write) dst->i.w = v; }
	else VU_MACw_CLEAR(VU);
	VU_STAT_UPDATE(VU);
}

// Emit the call to vu1_fmac_writeback.
// Result must be in v5.4S. dst_off = byte offset of destination from VU1_BASE_REG.
static void emitFmacWriteback(int64_t dst_off, u32 xyzw)
{
	// Extract v5 lanes to s0-s3 (ARM64 ABI: float args in s0-s3)
	armAsm->Mov(v3.V4S(), 0, v5.V4S(), 3); // s3 = W
	armAsm->Mov(v2.V4S(), 0, v5.V4S(), 2); // s2 = Z
	armAsm->Mov(v1.V4S(), 0, v5.V4S(), 1); // s1 = Y
	armAsm->Fmov(s0, s5);                   // s0 = X (lane 0)
	// Integer args: x0=VU, x1=dst, w2=xyzw
	armAsm->Mov(x0, VU1_BASE_REG);
	armAsm->Add(x1, VU1_BASE_REG, dst_off);
	armAsm->Mov(w2, xyzw);
	armEmitCall(reinterpret_cast<const void*>(vu1_fmac_writeback));
}

// ============================================================================
//  NEON FMAC emit patterns
//
//  Binary (ADD/SUB/MUL):  dst = VF[fs] OP src2
//  Ternary (MADD/MSUB):   dst = ACC ± VF[fs] * src2
//
//  src2 is loaded into v1:
//    Broadcast x/y/z/w: ldr s1 + dup v1.4s
//    Q register:        ldr w0 from VI[REG_Q], dup v1.4s, w0
//    I register:        ldr w0 from VI[REG_I], dup v1.4s, w0
//    Full vector:       ldr q1 from VF[ft]
// ============================================================================

// Load second operand broadcast from VF[ft] component into v1.4S
static void emitLoadBroadcast(u32 ft, int comp) // comp: 0=x,1=y,2=z,3=w
{
	armAsm->Ldr(s1, MemOperand(VU1_BASE_REG, vfOff(ft) + comp * 4));
	armAsm->Dup(v1.V4S(), v1.V4S(), 0);
}

// Load Q or I register broadcast into v1.4S
static void emitLoadQI(int64_t vi_off)
{
	armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vi_off));
	armAsm->Dup(v1.V4S(), w0);
}

// --- vuDouble-style input clamping for NEON vectors ---
// The interpreter calls vuDouble() on every FMAC input, which:
//   exp=0 (denormal): flush to ±0 (ARM64 FZ mode handles this)
//   exp=0xFF (inf/NaN): clamp to ±MAX_FLOAT (gated on CHECK_VU_SIGN_OVERFLOW)
// We use FMINNM/FMAXNM to clamp: these treat NaN as "missing" (return the
// non-NaN operand), so both ±INF and NaN are clamped to ±MAX_FLOAT.
// Setup: load MAX_FLOAT into v6, -MAX_FLOAT into v7.

static void emitVuClampSetup()
{
	armAsm->Mov(w0, 0xFFFF);
	armAsm->Movk(w0, 0x7F7F, 16); // w0 = 0x7F7FFFFF = MAX_FLOAT
	armAsm->Dup(v6.V4S(), w0);
	armAsm->Fneg(v7.V4S(), v6.V4S());
}

static void emitVuClampVec(const VRegister& vn)
{
	armAsm->Fminnm(vn, vn, v6.V4S());
	armAsm->Fmaxnm(vn, vn, v7.V4S());
}

// --- Binary FMAC: dst = VF[fs] OP src2 ---
// op: 0=ADD, 1=SUB, 2=MUL
static void emitBinaryFmac(int op, u32 fs, int64_t dst_off, u32 xyzw)
{
	// Load VF[fs] → v0
	armAsm->Ldr(q0, MemOperand(VU1_BASE_REG, vfOff(fs)));
	// v1 must already be loaded with src2

	// vuDouble input clamping (matches interpreter's vuDouble on every operand)
	if (CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1))
	{
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
	}

	// Perform NEON op: v5 = v0 OP v1
	switch (op) {
		case 0: armAsm->Fadd(v5.V4S(), v0.V4S(), v1.V4S()); break;
		case 1: armAsm->Fsub(v5.V4S(), v0.V4S(), v1.V4S()); break;
		case 2: armAsm->Fmul(v5.V4S(), v0.V4S(), v1.V4S()); break;
	}
	emitFmacWriteback(dst_off, xyzw);
}

// --- Ternary FMAC: dst = ACC ± VF[fs] * src2 ---
// subtract: false=MADD (ACC + fs*src2), true=MSUB (ACC - fs*src2)
// NOTE: The PS2 VU does NOT have fused multiply-add. It performs a separate
// multiply then add/sub with intermediate rounding. Using FMLA/FMLS would
// produce results differing by 1+ ULP, causing cascading precision errors
// in matrix transforms. We use separate FMUL + FADD/FSUB to match.
static void emitTernaryFmac(bool subtract, u32 fs, int64_t dst_off, u32 xyzw)
{
	// Load VF[fs] → v0
	armAsm->Ldr(q0, MemOperand(VU1_BASE_REG, vfOff(fs)));
	// v1 must already be loaded with src2
	// Load ACC → v5
	armAsm->Ldr(q5, MemOperand(VU1_BASE_REG, accOff()));

	// vuDouble input clamping (all 3 operands, matching interpreter)
	if (CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1))
	{
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
		emitVuClampVec(v5.V4S());
	}

	// Separate multiply: v4 = VF[fs] * src2 (with rounding)
	armAsm->Fmul(v4.V4S(), v0.V4S(), v1.V4S());
	// Separate add/sub: v5 = ACC ± product (with rounding)
	if (subtract)
		armAsm->Fsub(v5.V4S(), v5.V4S(), v4.V4S());
	else
		armAsm->Fadd(v5.V4S(), v5.V4S(), v4.V4S());
	emitFmacWriteback(dst_off, xyzw);
}

// ============================================================================
//  Macro: generate a binary FMAC rec function (ADD/SUB/MUL variants)
//
//  FMAC_BINARY_BC(name, op, comp)    — broadcast VF[ft].comp
//  FMAC_BINARY_Q(name, op)           — Q register broadcast
//  FMAC_BINARY_I(name, op)           — I register broadcast
//  FMAC_BINARY_FULL(name, op)        — full VF[ft] vector
//
//  op: 0=ADD, 1=SUB, 2=MUL
//  comp: 0=x, 1=y, 2=z, 3=w
//  toACC: false=VF[fd], true=ACC
// ============================================================================

#define FMAC_BINARY_BC(name, op, comp, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

#define FMAC_BINARY_Q(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

#define FMAC_BINARY_I(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

#define FMAC_BINARY_FULL(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		armAsm->Ldr(q1, MemOperand(VU1_BASE_REG, vfOff(ft))); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

// ============================================================================
//  Macro: generate a ternary FMAC rec function (MADD/MSUB variants)
// ============================================================================

#define FMAC_TERNARY_BC(name, isSub, comp, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_Q(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_I(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_FULL(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		armAsm->Ldr(q1, MemOperand(VU1_BASE_REG, vfOff(ft))); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

// ============================================================================
//  Per-instruction interp stub toggles (1 = interp, 0 = native)
// ============================================================================

#ifdef INTERP_VU_UPPER
// Group toggle: force all to interpreter
#define ISTUB_VU_ADDx    1
#define ISTUB_VU_ADDy    1
#define ISTUB_VU_ADDz    1
#define ISTUB_VU_ADDw    1
#define ISTUB_VU_ADDq    1
#define ISTUB_VU_ADDi    1
#define ISTUB_VU_ADD     1
#define ISTUB_VU_ADDAx   1
#define ISTUB_VU_ADDAy   1
#define ISTUB_VU_ADDAz   1
#define ISTUB_VU_ADDAw   1
#define ISTUB_VU_ADDAq   1
#define ISTUB_VU_ADDAi   1
#define ISTUB_VU_ADDA    1
#define ISTUB_VU_SUBx    1
#define ISTUB_VU_SUBy    1
#define ISTUB_VU_SUBz    1
#define ISTUB_VU_SUBw    1
#define ISTUB_VU_SUBq    1
#define ISTUB_VU_SUBi    1
#define ISTUB_VU_SUB     1
#define ISTUB_VU_SUBAx   1
#define ISTUB_VU_SUBAy   1
#define ISTUB_VU_SUBAz   1
#define ISTUB_VU_SUBAw   1
#define ISTUB_VU_SUBAq   1
#define ISTUB_VU_SUBAi   1
#define ISTUB_VU_SUBA    1
#define ISTUB_VU_MULx    1
#define ISTUB_VU_MULy    1
#define ISTUB_VU_MULz    1
#define ISTUB_VU_MULw    1
#define ISTUB_VU_MULq    1
#define ISTUB_VU_MULi    1
#define ISTUB_VU_MUL     1
#define ISTUB_VU_MULAx   1
#define ISTUB_VU_MULAy   1
#define ISTUB_VU_MULAz   1
#define ISTUB_VU_MULAw   1
#define ISTUB_VU_MULAq   1
#define ISTUB_VU_MULAi   1
#define ISTUB_VU_MULA    1
#define ISTUB_VU_MADDx   1
#define ISTUB_VU_MADDy   1
#define ISTUB_VU_MADDz   1
#define ISTUB_VU_MADDw   1
#define ISTUB_VU_MADDq   1
#define ISTUB_VU_MADDi   1
#define ISTUB_VU_MADD    1
#define ISTUB_VU_MADDAx  1
#define ISTUB_VU_MADDAy  1
#define ISTUB_VU_MADDAz  1
#define ISTUB_VU_MADDAw  1
#define ISTUB_VU_MADDAq  1
#define ISTUB_VU_MADDAi  1
#define ISTUB_VU_MADDA   1
#define ISTUB_VU_MSUBx   1
#define ISTUB_VU_MSUBy   1
#define ISTUB_VU_MSUBz   1
#define ISTUB_VU_MSUBw   1
#define ISTUB_VU_MSUBq   1
#define ISTUB_VU_MSUBi   1
#define ISTUB_VU_MSUB    1
#define ISTUB_VU_MSUBAx  1
#define ISTUB_VU_MSUBAy  1
#define ISTUB_VU_MSUBAz  1
#define ISTUB_VU_MSUBAw  1
#define ISTUB_VU_MSUBAq  1
#define ISTUB_VU_MSUBAi  1
#define ISTUB_VU_MSUBA   1
#define ISTUB_VU_MAXx    1
#define ISTUB_VU_MAXy    1
#define ISTUB_VU_MAXz    1
#define ISTUB_VU_MAXw    1
#define ISTUB_VU_MAXi    1
#define ISTUB_VU_MAX     1
#define ISTUB_VU_MINIx   1
#define ISTUB_VU_MINIy   1
#define ISTUB_VU_MINIz   1
#define ISTUB_VU_MINIw   1
#define ISTUB_VU_MINIi   1
#define ISTUB_VU_MINI    1
#define ISTUB_VU_ABS     1
#define ISTUB_VU_CLIP    1
#define ISTUB_VU_OPMULA  1
#define ISTUB_VU_OPMSUB  1
#define ISTUB_VU_NOP     1
#define ISTUB_VU_FTOI0   1
#define ISTUB_VU_FTOI4   1
#define ISTUB_VU_FTOI12  1
#define ISTUB_VU_FTOI15  1
#define ISTUB_VU_ITOF0   1
#define ISTUB_VU_ITOF4   1
#define ISTUB_VU_ITOF12  1
#define ISTUB_VU_ITOF15  1
#else
// Per-instruction control: set to 0 to enable native ARM64 codegen
#define ISTUB_VU_ADDx    0
#define ISTUB_VU_ADDy    0
#define ISTUB_VU_ADDz    0
#define ISTUB_VU_ADDw    0
#define ISTUB_VU_ADDq    0
#define ISTUB_VU_ADDi    0
#define ISTUB_VU_ADD     0
#define ISTUB_VU_ADDAx   0
#define ISTUB_VU_ADDAy   0
#define ISTUB_VU_ADDAz   0
#define ISTUB_VU_ADDAw   0
#define ISTUB_VU_ADDAq   0
#define ISTUB_VU_ADDAi   0
#define ISTUB_VU_ADDA    0
#define ISTUB_VU_SUBx    0
#define ISTUB_VU_SUBy    0
#define ISTUB_VU_SUBz    0
#define ISTUB_VU_SUBw    0
#define ISTUB_VU_SUBq    0
#define ISTUB_VU_SUBi    0
#define ISTUB_VU_SUB     0
#define ISTUB_VU_SUBAx   0
#define ISTUB_VU_SUBAy   0
#define ISTUB_VU_SUBAz   0
#define ISTUB_VU_SUBAw   0
#define ISTUB_VU_SUBAq   0
#define ISTUB_VU_SUBAi   0
#define ISTUB_VU_SUBA    0
#define ISTUB_VU_MULx    0
#define ISTUB_VU_MULy    0
#define ISTUB_VU_MULz    0
#define ISTUB_VU_MULw    0
#define ISTUB_VU_MULq    0
#define ISTUB_VU_MULi    0
#define ISTUB_VU_MUL     0
#define ISTUB_VU_MULAx   0
#define ISTUB_VU_MULAy   0
#define ISTUB_VU_MULAz   0
#define ISTUB_VU_MULAw   0
#define ISTUB_VU_MULAq   0
#define ISTUB_VU_MULAi   0
#define ISTUB_VU_MULA    0
#define ISTUB_VU_MADDx   0
#define ISTUB_VU_MADDy   0
#define ISTUB_VU_MADDz   0
#define ISTUB_VU_MADDw   0
#define ISTUB_VU_MADDq   0
#define ISTUB_VU_MADDi   0
#define ISTUB_VU_MADD    0
#define ISTUB_VU_MADDAx  0
#define ISTUB_VU_MADDAy  0
#define ISTUB_VU_MADDAz  0
#define ISTUB_VU_MADDAw  0
#define ISTUB_VU_MADDAq  0
#define ISTUB_VU_MADDAi  0
#define ISTUB_VU_MADDA   0
#define ISTUB_VU_MSUBx   0
#define ISTUB_VU_MSUBy   0
#define ISTUB_VU_MSUBz   0
#define ISTUB_VU_MSUBw   0
#define ISTUB_VU_MSUBq   0
#define ISTUB_VU_MSUBi   0
#define ISTUB_VU_MSUB    0
#define ISTUB_VU_MSUBAx  0
#define ISTUB_VU_MSUBAy  0
#define ISTUB_VU_MSUBAz  0
#define ISTUB_VU_MSUBAw  0
#define ISTUB_VU_MSUBAq  0
#define ISTUB_VU_MSUBAi  0
#define ISTUB_VU_MSUBA   0
#define ISTUB_VU_MAXx    0
#define ISTUB_VU_MAXy    0
#define ISTUB_VU_MAXz    0
#define ISTUB_VU_MAXw    0
#define ISTUB_VU_MAXi    0
#define ISTUB_VU_MAX     0
#define ISTUB_VU_MINIx   0
#define ISTUB_VU_MINIy   0
#define ISTUB_VU_MINIz   0
#define ISTUB_VU_MINIw   0
#define ISTUB_VU_MINIi   0
#define ISTUB_VU_MINI    0
#define ISTUB_VU_ABS     0
#define ISTUB_VU_CLIP    0
#define ISTUB_VU_OPMULA  0
#define ISTUB_VU_OPMSUB  0
#define ISTUB_VU_NOP     0
#define ISTUB_VU_FTOI0   0
#define ISTUB_VU_FTOI4   0
#define ISTUB_VU_FTOI12  0
#define ISTUB_VU_FTOI15  0
#define ISTUB_VU_ITOF0   0
#define ISTUB_VU_ITOF4   0
#define ISTUB_VU_ITOF12  0
#define ISTUB_VU_ITOF15  0
#endif

// ============================================================================
//  Code-emitter macro: called at block-compile time to emit an ARM64 BL to
//  the specific interpreter function for this upper opcode.
//
//  VU1.code is set by CompileBlock (both as a compile-time variable used here
//  to resolve VU1_UPPER_OPCODE[…], and as a runtime store emitted before
//  calling this function).
// ============================================================================

// INTERP path (ISTUB=1): emit BL to the interpreter function.
#define REC_VU1_UPPER_INTERP(name) \
	void recVU1_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); \
	}

// Non-INTERP fallback for non-FMAC instructions (ISTUB=0, no native codegen yet).
#define REC_VU1_UPPER_EMIT(name) \
	void recVU1_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); \
	}

// ============================================================================
//  ADD family — VF[fd] = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU_ADDx
REC_VU1_UPPER_INTERP(ADDx)
#else
FMAC_BINARY_BC(ADDx, 0, 0, false)
#endif

#if ISTUB_VU_ADDy
REC_VU1_UPPER_INTERP(ADDy)
#else
FMAC_BINARY_BC(ADDy, 0, 1, false)
#endif

#if ISTUB_VU_ADDz
REC_VU1_UPPER_INTERP(ADDz)
#else
FMAC_BINARY_BC(ADDz, 0, 2, false)
#endif

#if ISTUB_VU_ADDw
REC_VU1_UPPER_INTERP(ADDw)
#else
FMAC_BINARY_BC(ADDw, 0, 3, false)
#endif

#if ISTUB_VU_ADDq
REC_VU1_UPPER_INTERP(ADDq)
#else
FMAC_BINARY_Q(ADDq, 0, false)
#endif

#if ISTUB_VU_ADDi
REC_VU1_UPPER_INTERP(ADDi)
#else
FMAC_BINARY_I(ADDi, 0, false)
#endif

#if ISTUB_VU_ADD
REC_VU1_UPPER_INTERP(ADD)
#else
FMAC_BINARY_FULL(ADD, 0, false)
#endif

// ============================================================================
//  ADDA family — ACC = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU_ADDAx
REC_VU1_UPPER_INTERP(ADDAx)
#else
FMAC_BINARY_BC(ADDAx, 0, 0, true)
#endif

#if ISTUB_VU_ADDAy
REC_VU1_UPPER_INTERP(ADDAy)
#else
FMAC_BINARY_BC(ADDAy, 0, 1, true)
#endif

#if ISTUB_VU_ADDAz
REC_VU1_UPPER_INTERP(ADDAz)
#else
FMAC_BINARY_BC(ADDAz, 0, 2, true)
#endif

#if ISTUB_VU_ADDAw
REC_VU1_UPPER_INTERP(ADDAw)
#else
FMAC_BINARY_BC(ADDAw, 0, 3, true)
#endif

#if ISTUB_VU_ADDAq
REC_VU1_UPPER_INTERP(ADDAq)
#else
FMAC_BINARY_Q(ADDAq, 0, true)
#endif

#if ISTUB_VU_ADDAi
REC_VU1_UPPER_INTERP(ADDAi)
#else
FMAC_BINARY_I(ADDAi, 0, true)
#endif

#if ISTUB_VU_ADDA
REC_VU1_UPPER_INTERP(ADDA)
#else
FMAC_BINARY_FULL(ADDA, 0, true)
#endif

// ============================================================================
//  SUB family
// ============================================================================

#if ISTUB_VU_SUBx
REC_VU1_UPPER_INTERP(SUBx)
#else
FMAC_BINARY_BC(SUBx, 1, 0, false)
#endif

#if ISTUB_VU_SUBy
REC_VU1_UPPER_INTERP(SUBy)
#else
FMAC_BINARY_BC(SUBy, 1, 1, false)
#endif

#if ISTUB_VU_SUBz
REC_VU1_UPPER_INTERP(SUBz)
#else
FMAC_BINARY_BC(SUBz, 1, 2, false)
#endif

#if ISTUB_VU_SUBw
REC_VU1_UPPER_INTERP(SUBw)
#else
FMAC_BINARY_BC(SUBw, 1, 3, false)
#endif

#if ISTUB_VU_SUBq
REC_VU1_UPPER_INTERP(SUBq)
#else
FMAC_BINARY_Q(SUBq, 1, false)
#endif

#if ISTUB_VU_SUBi
REC_VU1_UPPER_INTERP(SUBi)
#else
FMAC_BINARY_I(SUBi, 1, false)
#endif

#if ISTUB_VU_SUB
REC_VU1_UPPER_INTERP(SUB)
#else
FMAC_BINARY_FULL(SUB, 1, false)
#endif

// ============================================================================
//  SUBA family
// ============================================================================

#if ISTUB_VU_SUBAx
REC_VU1_UPPER_INTERP(SUBAx)
#else
FMAC_BINARY_BC(SUBAx, 1, 0, true)
#endif

#if ISTUB_VU_SUBAy
REC_VU1_UPPER_INTERP(SUBAy)
#else
FMAC_BINARY_BC(SUBAy, 1, 1, true)
#endif

#if ISTUB_VU_SUBAz
REC_VU1_UPPER_INTERP(SUBAz)
#else
FMAC_BINARY_BC(SUBAz, 1, 2, true)
#endif

#if ISTUB_VU_SUBAw
REC_VU1_UPPER_INTERP(SUBAw)
#else
FMAC_BINARY_BC(SUBAw, 1, 3, true)
#endif

#if ISTUB_VU_SUBAq
REC_VU1_UPPER_INTERP(SUBAq)
#else
FMAC_BINARY_Q(SUBAq, 1, true)
#endif

#if ISTUB_VU_SUBAi
REC_VU1_UPPER_INTERP(SUBAi)
#else
FMAC_BINARY_I(SUBAi, 1, true)
#endif

#if ISTUB_VU_SUBA
REC_VU1_UPPER_INTERP(SUBA)
#else
FMAC_BINARY_FULL(SUBA, 1, true)
#endif

// ============================================================================
//  MUL family
// ============================================================================

#if ISTUB_VU_MULx
REC_VU1_UPPER_INTERP(MULx)
#else
FMAC_BINARY_BC(MULx, 2, 0, false)
#endif

#if ISTUB_VU_MULy
REC_VU1_UPPER_INTERP(MULy)
#else
FMAC_BINARY_BC(MULy, 2, 1, false)
#endif

#if ISTUB_VU_MULz
REC_VU1_UPPER_INTERP(MULz)
#else
FMAC_BINARY_BC(MULz, 2, 2, false)
#endif

#if ISTUB_VU_MULw
REC_VU1_UPPER_INTERP(MULw)
#else
FMAC_BINARY_BC(MULw, 2, 3, false)
#endif

#if ISTUB_VU_MULq
REC_VU1_UPPER_INTERP(MULq)
#else
FMAC_BINARY_Q(MULq, 2, false)
#endif

#if ISTUB_VU_MULi
REC_VU1_UPPER_INTERP(MULi)
#else
FMAC_BINARY_I(MULi, 2, false)
#endif

#if ISTUB_VU_MUL
REC_VU1_UPPER_INTERP(MUL)
#else
FMAC_BINARY_FULL(MUL, 2, false)
#endif

// ============================================================================
//  MULA family
// ============================================================================

#if ISTUB_VU_MULAx
REC_VU1_UPPER_INTERP(MULAx)
#else
FMAC_BINARY_BC(MULAx, 2, 0, true)
#endif

#if ISTUB_VU_MULAy
REC_VU1_UPPER_INTERP(MULAy)
#else
FMAC_BINARY_BC(MULAy, 2, 1, true)
#endif

#if ISTUB_VU_MULAz
REC_VU1_UPPER_INTERP(MULAz)
#else
FMAC_BINARY_BC(MULAz, 2, 2, true)
#endif

#if ISTUB_VU_MULAw
REC_VU1_UPPER_INTERP(MULAw)
#else
FMAC_BINARY_BC(MULAw, 2, 3, true)
#endif

#if ISTUB_VU_MULAq
REC_VU1_UPPER_INTERP(MULAq)
#else
FMAC_BINARY_Q(MULAq, 2, true)
#endif

#if ISTUB_VU_MULAi
REC_VU1_UPPER_INTERP(MULAi)
#else
FMAC_BINARY_I(MULAi, 2, true)
#endif

#if ISTUB_VU_MULA
REC_VU1_UPPER_INTERP(MULA)
#else
FMAC_BINARY_FULL(MULA, 2, true)
#endif

// ============================================================================
//  MADD family
// ============================================================================

#if ISTUB_VU_MADDx
REC_VU1_UPPER_INTERP(MADDx)
#else
FMAC_TERNARY_BC(MADDx, false, 0, false)
#endif

#if ISTUB_VU_MADDy
REC_VU1_UPPER_INTERP(MADDy)
#else
FMAC_TERNARY_BC(MADDy, false, 1, false)
#endif

#if ISTUB_VU_MADDz
REC_VU1_UPPER_INTERP(MADDz)
#else
FMAC_TERNARY_BC(MADDz, false, 2, false)
#endif

#if ISTUB_VU_MADDw
REC_VU1_UPPER_INTERP(MADDw)
#else
FMAC_TERNARY_BC(MADDw, false, 3, false)
#endif

#if ISTUB_VU_MADDq
REC_VU1_UPPER_INTERP(MADDq)
#else
FMAC_TERNARY_Q(MADDq, false, false)
#endif

#if ISTUB_VU_MADDi
REC_VU1_UPPER_INTERP(MADDi)
#else
FMAC_TERNARY_I(MADDi, false, false)
#endif

#if ISTUB_VU_MADD
REC_VU1_UPPER_INTERP(MADD)
#else
FMAC_TERNARY_FULL(MADD, false, false)
#endif

// ============================================================================
//  MADDA family
// ============================================================================

#if ISTUB_VU_MADDAx
REC_VU1_UPPER_INTERP(MADDAx)
#else
FMAC_TERNARY_BC(MADDAx, false, 0, true)
#endif

#if ISTUB_VU_MADDAy
REC_VU1_UPPER_INTERP(MADDAy)
#else
FMAC_TERNARY_BC(MADDAy, false, 1, true)
#endif

#if ISTUB_VU_MADDAz
REC_VU1_UPPER_INTERP(MADDAz)
#else
FMAC_TERNARY_BC(MADDAz, false, 2, true)
#endif

#if ISTUB_VU_MADDAw
REC_VU1_UPPER_INTERP(MADDAw)
#else
FMAC_TERNARY_BC(MADDAw, false, 3, true)
#endif

#if ISTUB_VU_MADDAq
REC_VU1_UPPER_INTERP(MADDAq)
#else
FMAC_TERNARY_Q(MADDAq, false, true)
#endif

#if ISTUB_VU_MADDAi
REC_VU1_UPPER_INTERP(MADDAi)
#else
FMAC_TERNARY_I(MADDAi, false, true)
#endif

#if ISTUB_VU_MADDA
REC_VU1_UPPER_INTERP(MADDA)
#else
FMAC_TERNARY_FULL(MADDA, false, true)
#endif

// ============================================================================
//  MSUB family
// ============================================================================

#if ISTUB_VU_MSUBx
REC_VU1_UPPER_INTERP(MSUBx)
#else
FMAC_TERNARY_BC(MSUBx, true, 0, false)
#endif

#if ISTUB_VU_MSUBy
REC_VU1_UPPER_INTERP(MSUBy)
#else
FMAC_TERNARY_BC(MSUBy, true, 1, false)
#endif

#if ISTUB_VU_MSUBz
REC_VU1_UPPER_INTERP(MSUBz)
#else
FMAC_TERNARY_BC(MSUBz, true, 2, false)
#endif

#if ISTUB_VU_MSUBw
REC_VU1_UPPER_INTERP(MSUBw)
#else
FMAC_TERNARY_BC(MSUBw, true, 3, false)
#endif

#if ISTUB_VU_MSUBq
REC_VU1_UPPER_INTERP(MSUBq)
#else
FMAC_TERNARY_Q(MSUBq, true, false)
#endif

#if ISTUB_VU_MSUBi
REC_VU1_UPPER_INTERP(MSUBi)
#else
FMAC_TERNARY_I(MSUBi, true, false)
#endif

#if ISTUB_VU_MSUB
REC_VU1_UPPER_INTERP(MSUB)
#else
FMAC_TERNARY_FULL(MSUB, true, false)
#endif

// ============================================================================
//  MSUBA family
// ============================================================================

#if ISTUB_VU_MSUBAx
REC_VU1_UPPER_INTERP(MSUBAx)
#else
FMAC_TERNARY_BC(MSUBAx, true, 0, true)
#endif

#if ISTUB_VU_MSUBAy
REC_VU1_UPPER_INTERP(MSUBAy)
#else
FMAC_TERNARY_BC(MSUBAy, true, 1, true)
#endif

#if ISTUB_VU_MSUBAz
REC_VU1_UPPER_INTERP(MSUBAz)
#else
FMAC_TERNARY_BC(MSUBAz, true, 2, true)
#endif

#if ISTUB_VU_MSUBAw
REC_VU1_UPPER_INTERP(MSUBAw)
#else
FMAC_TERNARY_BC(MSUBAw, true, 3, true)
#endif

#if ISTUB_VU_MSUBAq
REC_VU1_UPPER_INTERP(MSUBAq)
#else
FMAC_TERNARY_Q(MSUBAq, true, true)
#endif

#if ISTUB_VU_MSUBAi
REC_VU1_UPPER_INTERP(MSUBAi)
#else
FMAC_TERNARY_I(MSUBAi, true, true)
#endif

#if ISTUB_VU_MSUBA
REC_VU1_UPPER_INTERP(MSUBA)
#else
FMAC_TERNARY_FULL(MSUBA, true, true)
#endif

// ============================================================================
//  MAX / MINI
// ============================================================================

#if ISTUB_VU_MAXx
REC_VU1_UPPER_INTERP(MAXx)
#else
void recVU1_MAXx() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MAXy
REC_VU1_UPPER_INTERP(MAXy)
#else
void recVU1_MAXy() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MAXz
REC_VU1_UPPER_INTERP(MAXz)
#else
void recVU1_MAXz() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MAXw
REC_VU1_UPPER_INTERP(MAXw)
#else
void recVU1_MAXw() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MAXi
REC_VU1_UPPER_INTERP(MAXi)
#else
void recVU1_MAXi() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MAX
REC_VU1_UPPER_INTERP(MAX)
#else
void recVU1_MAX() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINIx
REC_VU1_UPPER_INTERP(MINIx)
#else
void recVU1_MINIx() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINIy
REC_VU1_UPPER_INTERP(MINIy)
#else
void recVU1_MINIy() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINIz
REC_VU1_UPPER_INTERP(MINIz)
#else
void recVU1_MINIz() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINIw
REC_VU1_UPPER_INTERP(MINIw)
#else
void recVU1_MINIw() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINIi
REC_VU1_UPPER_INTERP(MINIi)
#else
void recVU1_MINIi() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_MINI
REC_VU1_UPPER_INTERP(MINI)
#else
void recVU1_MINI() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

// ============================================================================
//  ABS, CLIP, OPMULA, OPMSUB, NOP
// ============================================================================

#if ISTUB_VU_ABS
REC_VU1_UPPER_INTERP(ABS)
#else
void recVU1_ABS() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_CLIP
REC_VU1_UPPER_INTERP(CLIP)
#else
void recVU1_CLIP() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_OPMULA
REC_VU1_UPPER_INTERP(OPMULA)
#else
void recVU1_OPMULA() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_OPMSUB
REC_VU1_UPPER_INTERP(OPMSUB)
#else
void recVU1_OPMSUB() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_NOP
REC_VU1_UPPER_INTERP(NOP)
#else
void recVU1_NOP() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

// ============================================================================
//  FTOI / ITOF — float/int conversion
// ============================================================================

#if ISTUB_VU_FTOI0
REC_VU1_UPPER_INTERP(FTOI0)
#else
void recVU1_FTOI0() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_FTOI4
REC_VU1_UPPER_INTERP(FTOI4)
#else
void recVU1_FTOI4() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_FTOI12
REC_VU1_UPPER_INTERP(FTOI12)
#else
void recVU1_FTOI12() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_FTOI15
REC_VU1_UPPER_INTERP(FTOI15)
#else
void recVU1_FTOI15() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_ITOF0
REC_VU1_UPPER_INTERP(ITOF0)
#else
void recVU1_ITOF0() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_ITOF4
REC_VU1_UPPER_INTERP(ITOF4)
#else
void recVU1_ITOF4() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_ITOF12
REC_VU1_UPPER_INTERP(ITOF12)
#else
void recVU1_ITOF12() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

#if ISTUB_VU_ITOF15
REC_VU1_UPPER_INTERP(ITOF15)
#else
void recVU1_ITOF15() { armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); }
#endif

// ============================================================================
//  Generic emitter for opcode indices that dispatch through the FD sub-table
//  (0x3C-0x3F) or the unknown/reserved slots (0x30-0x3B).
//  VU1.code is already set to the current instruction word, so
//  VU1_UPPER_OPCODE[VU1.code & 0x3f] resolves the correct entry.
// ============================================================================
static void recVU1_Upper_FD()
{
	armEmitCall(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f]));
}

// ============================================================================
//  recVU1_UpperTable[64]
//
//  Maps upper opcode index (upper_word & 0x3f) to a code-emitter function.
//  Layout mirrors VU1_UPPER_OPCODE in VUops.cpp.
//
//  Indices 0x30-0x3B are reserved/unknown in the VU ISA; they are populated
//  with recVU1_Upper_FD (which BLs to VU1_UPPER_OPCODE and lets the
//  interpreter handle it, including unknown-instruction logging).
//  Indices 0x3C-0x3F dispatch through the FD sub-table and likewise use
//  recVU1_Upper_FD so the sub-table dispatch is handled transparently.
// ============================================================================
using VU1RecFn = void (*)();

VU1RecFn recVU1_UpperTable[64] = {
	// 0x00-0x03: ADD broadcast
	recVU1_ADDx, recVU1_ADDy, recVU1_ADDz, recVU1_ADDw,
	// 0x04-0x07: SUB broadcast
	recVU1_SUBx, recVU1_SUBy, recVU1_SUBz, recVU1_SUBw,
	// 0x08-0x0B: MADD broadcast
	recVU1_MADDx, recVU1_MADDy, recVU1_MADDz, recVU1_MADDw,
	// 0x0C-0x0F: MSUB broadcast
	recVU1_MSUBx, recVU1_MSUBy, recVU1_MSUBz, recVU1_MSUBw,
	// 0x10-0x13: MAX broadcast
	recVU1_MAXx, recVU1_MAXy, recVU1_MAXz, recVU1_MAXw,
	// 0x14-0x17: MINI broadcast
	recVU1_MINIx, recVU1_MINIy, recVU1_MINIz, recVU1_MINIw,
	// 0x18-0x1B: MUL broadcast
	recVU1_MULx, recVU1_MULy, recVU1_MULz, recVU1_MULw,
	// 0x1C-0x1F: MULq, MAXi, MULi, MINIi
	recVU1_MULq, recVU1_MAXi, recVU1_MULi, recVU1_MINIi,
	// 0x20-0x23: ADDq, MADDq, ADDi, MADDi
	recVU1_ADDq, recVU1_MADDq, recVU1_ADDi, recVU1_MADDi,
	// 0x24-0x27: SUBq, MSUBq, SUBi, MSUBi
	recVU1_SUBq, recVU1_MSUBq, recVU1_SUBi, recVU1_MSUBi,
	// 0x28-0x2B: ADD, MADD, MUL, MAX
	recVU1_ADD, recVU1_MADD, recVU1_MUL, recVU1_MAX,
	// 0x2C-0x2F: SUB, MSUB, OPMSUB, MINI
	recVU1_SUB, recVU1_MSUB, recVU1_OPMSUB, recVU1_MINI,
	// 0x30-0x3B: reserved/unknown — delegate via interpreter
	recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD,
	recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD,
	recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD,
	// 0x3C-0x3F: FD sub-table dispatch (ADDA/SUBA/MULA/MADDA/MSUBA/ABS/CLIP/NOP/FTOI/ITOF/OPMULA)
	recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD, recVU1_Upper_FD,
};
