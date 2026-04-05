// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — MMI (MultiMedia Instructions)
// Packed SIMD ops: PADD*/PSUB*, PCGT*, PMAX/MIN*, PCEQ*, PABS*,
// PSxx shifts, PEXTL*/PPAC*/PEXTU*, PINTH/PINTEH, PADSBH,
// PAND/POR/PXOR/PNOR, PMFHI/LO, PMTHI/LO, PCPYLD/UD,
// PREVH, PCPYH, PLZCW, PEXEH/PEXEW/PEXCH/PEXCW, PROT3W,
// PSLLVW/PSRLVW/PSRAVW, and more.
//
// Native implementations use ARM64 NEON 128-bit SIMD instructions.
// Complex or rarely-used ops remain as interpreter stubs (ISTUB=1).

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace MMI {

// ============================================================================
//  INTERP_MMI master switch: ALL ops become unconditional interpreter stubs.
//  No templates, no _Rd_ guards, no NEON code — just plain interpreter calls.
// ============================================================================

#ifdef INTERP_MMI

#define REC_MMI_STUB(name) \
	void rec##name() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::name); }

// MMI main
REC_MMI_STUB(PLZCW)
REC_MMI_STUB(PMFHL)
REC_MMI_STUB(PMTHL)
REC_MMI_STUB(PSLLH)
REC_MMI_STUB(PSRLH)
REC_MMI_STUB(PSRAH)
REC_MMI_STUB(PSLLW)
REC_MMI_STUB(PSRLW)
REC_MMI_STUB(PSRAW)

// MMI0
REC_MMI_STUB(PADDW)
REC_MMI_STUB(PSUBW)
REC_MMI_STUB(PCGTW)
REC_MMI_STUB(PMAXW)
REC_MMI_STUB(PADDH)
REC_MMI_STUB(PSUBH)
REC_MMI_STUB(PCGTH)
REC_MMI_STUB(PMAXH)
REC_MMI_STUB(PADDB)
REC_MMI_STUB(PSUBB)
REC_MMI_STUB(PCGTB)
REC_MMI_STUB(PADDSW)
REC_MMI_STUB(PSUBSW)
REC_MMI_STUB(PEXTLW)
REC_MMI_STUB(PPACW)
REC_MMI_STUB(PADDSH)
REC_MMI_STUB(PSUBSH)
REC_MMI_STUB(PEXTLH)
REC_MMI_STUB(PPACH)
REC_MMI_STUB(PADDSB)
REC_MMI_STUB(PSUBSB)
REC_MMI_STUB(PEXTLB)
REC_MMI_STUB(PPACB)
REC_MMI_STUB(PEXT5)
REC_MMI_STUB(PPAC5)

// MMI1
REC_MMI_STUB(PABSW)
REC_MMI_STUB(PCEQW)
REC_MMI_STUB(PMINW)
REC_MMI_STUB(PADSBH)
REC_MMI_STUB(PABSH)
REC_MMI_STUB(PCEQH)
REC_MMI_STUB(PMINH)
REC_MMI_STUB(PCEQB)
REC_MMI_STUB(PADDUW)
REC_MMI_STUB(PSUBUW)
REC_MMI_STUB(PEXTUW)
REC_MMI_STUB(PADDUH)
REC_MMI_STUB(PSUBUH)
REC_MMI_STUB(PEXTUH)
REC_MMI_STUB(PADDUB)
REC_MMI_STUB(PSUBUB)
REC_MMI_STUB(PEXTUB)
REC_MMI_STUB(QFSRV)

// MMI2
REC_MMI_STUB(PMADDW)
REC_MMI_STUB(PSLLVW)
REC_MMI_STUB(PMSUBW)
REC_MMI_STUB(PMFHI)
REC_MMI_STUB(PMFLO)
REC_MMI_STUB(PINTH)
REC_MMI_STUB(PMULTW)
REC_MMI_STUB(PDIVW)
REC_MMI_STUB(PCPYLD)
REC_MMI_STUB(PMADDH)
REC_MMI_STUB(PHMADH)
REC_MMI_STUB(PAND)
REC_MMI_STUB(PXOR)
REC_MMI_STUB(PMSUBH)
REC_MMI_STUB(PHMSBH)
REC_MMI_STUB(PEXEH)
REC_MMI_STUB(PREVH)
REC_MMI_STUB(PMULTH)
REC_MMI_STUB(PDIVBW)
REC_MMI_STUB(PEXEW)
REC_MMI_STUB(PROT3W)

// MMI3
REC_MMI_STUB(PMADDUW)
REC_MMI_STUB(PSRLVW)
REC_MMI_STUB(PSRAVW)
REC_MMI_STUB(PMTHI)
REC_MMI_STUB(PMTLO)
REC_MMI_STUB(PINTEH)
REC_MMI_STUB(PCPYUD)
REC_MMI_STUB(POR)
REC_MMI_STUB(PNOR)
REC_MMI_STUB(PMULTUW)
REC_MMI_STUB(PDIVUW)
REC_MMI_STUB(PEXCH)
REC_MMI_STUB(PCPYH)
REC_MMI_STUB(PEXCW)

#undef REC_MMI_STUB

#else // !INTERP_MMI — native NEON implementations with per-op ISTUB toggles

// ============================================================================
//  NEON scratch register aliases (128-bit Q registers, caller-saved)
// ============================================================================

#define RVMMI0  a64::q0
#define RVMMI1  a64::q1
#define RVMMI2  a64::q2

// ============================================================================
//  Per-instruction interp stub toggles
//  Individual ISTUB_* = 1 for interpreter fallback, 0 for native NEON
// ============================================================================

// Simple packed arithmetic — native NEON (direct equivalents)
#define ISTUB_PADDW    0
#define ISTUB_PADDH    0
#define ISTUB_PADDB    0
#define ISTUB_PADDSW   0
#define ISTUB_PADDSH   0
#define ISTUB_PADDSB   0
#define ISTUB_PADDUW   0
#define ISTUB_PADDUH   0
#define ISTUB_PADDUB   0
#define ISTUB_PSUBW    0
#define ISTUB_PSUBH    0
#define ISTUB_PSUBB    0
#define ISTUB_PSUBSW   0
#define ISTUB_PSUBSH   0
#define ISTUB_PSUBSB   0
#define ISTUB_PSUBUW   0
#define ISTUB_PSUBUH   0
#define ISTUB_PSUBUB   0

// Compare / min / max — native NEON
#define ISTUB_PCGTW    0
#define ISTUB_PCGTH    0
#define ISTUB_PCGTB    0
#define ISTUB_PMAXW    0
#define ISTUB_PMAXH    0
#define ISTUB_PMINW    0
#define ISTUB_PMINH    0
#define ISTUB_PCEQW    0
#define ISTUB_PCEQH    0
#define ISTUB_PCEQB    0
#define ISTUB_PABSW    0
#define ISTUB_PABSH    0

// Logic — native NEON
#define ISTUB_PAND     0
#define ISTUB_POR      0
#define ISTUB_PXOR     0
#define ISTUB_PNOR     0

// HI/LO transfers — native (simple 128b load/store)
#define ISTUB_PMFHI    0
#define ISTUB_PMFLO    0
#define ISTUB_PMTHI    0
#define ISTUB_PMTLO    0

// 128-bit copy — native
#define ISTUB_PCPYLD   0
#define ISTUB_PCPYUD   0

// Packed shifts (immediate _Sa_) — native NEON
#define ISTUB_PSLLH    0
#define ISTUB_PSRLH    0
#define ISTUB_PSRAH    0
#define ISTUB_PSLLW    0
#define ISTUB_PSRLW    0
#define ISTUB_PSRAW    0

// Variable-shift — native NEON (Sshl/Ushl with negation)
#define ISTUB_PSLLVW   0
#define ISTUB_PSRLVW   0
#define ISTUB_PSRAVW   0

// Interleave / pack / extract — native (zip1/zip2/uzp1/ext)
#define ISTUB_PEXTLW   0
#define ISTUB_PEXTLH   0
#define ISTUB_PEXTLB   0
#define ISTUB_PPACW    0
#define ISTUB_PPACH    0
#define ISTUB_PPACB    0
#define ISTUB_PEXTUW   0
#define ISTUB_PEXTUH   0
#define ISTUB_PEXTUB   0
#define ISTUB_PINTH    0
#define ISTUB_PINTEH   0

// Misc packed — native NEON
#define ISTUB_PADSBH   0
#define ISTUB_PLZCW    0
#define ISTUB_PREVH    0
#define ISTUB_PCPYH    0

// Shuffle ops — native (Ins sequences)
#define ISTUB_PEXEH    0
#define ISTUB_PEXEW    0
#define ISTUB_PEXCW    0
#define ISTUB_PEXCH    0
#define ISTUB_PROT3W   0

// Complex / HI-LO-writing ops — keep as interpreter stubs
#define ISTUB_PMFHL    1   // 5 modes via sa field
#define ISTUB_PMTHL    1   // HI/LO write with 4 elements
#define ISTUB_PEXT5    1   // RGB5→RGBA8 bit shuffle
#define ISTUB_PPAC5    1   // RGBA8→RGB5 bit shuffle
#define ISTUB_QFSRV    1   // variable byte-shift (runtime sa)
#define ISTUB_PMADDH   1   // 8×16b MAC to HI/LO
#define ISTUB_PHMADH   1   // horizontal MAC
#define ISTUB_PMSUBH   1   // 8×16b MSUB to HI/LO
#define ISTUB_PHMSBH   1   // horizontal MSUB
#define ISTUB_PMULTH   1   // 8×16b multiply to HI/LO
#define ISTUB_PMADDW   1   // 2×32b MAC to HI/LO
#define ISTUB_PMSUBW   1   // 2×32b MSUB to HI/LO
#define ISTUB_PMULTW   1   // 2×32b multiply to HI/LO
#define ISTUB_PMADDUW  1   // 2×32b unsigned MAC to HI/LO
#define ISTUB_PMULTUW  1   // 2×32b unsigned multiply to HI/LO
#define ISTUB_PDIVW    1   // 2×32b signed divide to HI/LO
#define ISTUB_PDIVUW   1   // 2×32b unsigned divide to HI/LO
#define ISTUB_PDIVBW   1   // 4×32/16b divide to HI/LO

// ============================================================================
//  Codegen helpers
// ============================================================================

// Load PS2 128-bit GPR into a NEON Q register.
static __fi void armLoadGPR128(const a64::VRegister& dst, int gpr)
{
	armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

// Store NEON Q register into PS2 128-bit GPR.
static __fi void armStoreGPR128(int gpr, const a64::VRegister& src)
{
	armAsm->Str(src, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

// Binary op template: loads rs→q0, rt→q1, runs opFunc(), stores q0→rd.
template<typename OpFunc>
static void armMMIBinOp(OpFunc opFunc)
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	opFunc();
	armStoreGPR128(_Rd_, RVMMI0);
}

// Unary op template: loads rt→q0, runs opFunc(), stores q0→rd.
template<typename OpFunc>
static void armMMIUnaryOp(OpFunc opFunc)
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	opFunc();
	armStoreGPR128(_Rd_, RVMMI0);
}

// ============================================================================
//  MMI — PLZCW, PMFHL, PMTHL, PSxx shifts (main MMI sub-table)
// ============================================================================

#if ISTUB_PLZCW
void recPLZCW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PLZCW); }
#else
void recPLZCW()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RWSCRATCH,  a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rs_) + 0));
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rs_) + 4));
	armAsm->Cls(RWSCRATCH,  RWSCRATCH);
	armAsm->Cls(RWSCRATCH2, RWSCRATCH2);
	armAsm->Str(RWSCRATCH,  a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_) + 0));
	armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_) + 4));
}
#endif

#if ISTUB_PMFHL
void recPMFHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHL); }
#else
void recPMFHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHL); }
#endif

#if ISTUB_PMTHL
void recPMTHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTHL); }
#else
void recPMTHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTHL); }
#endif

// ---- Packed shifts (immediate sa) ----

#if ISTUB_PSLLH
void recPSLLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLH); }
#else
void recPSLLH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		armAsm->Shl(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSRLH
void recPSRLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLH); }
#else
void recPSRLH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Ushr(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSRAH
void recPSRAH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAH); }
#else
void recPSRAH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Sshr(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSLLW
void recPSLLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLW); }
#else
void recPSLLW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		armAsm->Shl(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

#if ISTUB_PSRLW
void recPSRLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLW); }
#else
void recPSRLW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Ushr(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

#if ISTUB_PSRAW
void recPSRAW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAW); }
#else
void recPSRAW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Sshr(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

// ============================================================================
//  MMI0 — PADDW/H/B, PSUBW/H/B, saturating, unsigned-sat, compare, pack/interleave
// ============================================================================

#if ISTUB_PADDW
void recPADDW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDW); }
#else
void recPADDW() { armMMIBinOp([]() { armAsm->Add(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBW
void recPSUBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBW); }
#else
void recPSUBW() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PCGTW
void recPCGTW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTW); }
#else
void recPCGTW() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PMAXW
void recPMAXW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMAXW); }
#else
void recPMAXW() { armMMIBinOp([]() { armAsm->Smax(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PADDH
void recPADDH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDH); }
#else
void recPADDH() { armMMIBinOp([]() { armAsm->Add(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBH
void recPSUBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBH); }
#else
void recPSUBH() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PCGTH
void recPCGTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTH); }
#else
void recPCGTH() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PMAXH
void recPMAXH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMAXH); }
#else
void recPMAXH() { armMMIBinOp([]() { armAsm->Smax(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PADDB
void recPADDB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDB); }
#else
void recPADDB() { armMMIBinOp([]() { armAsm->Add(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBB
void recPSUBB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBB); }
#else
void recPSUBB() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PCGTB
void recPCGTB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTB); }
#else
void recPCGTB() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PADDSW
void recPADDSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSW); }
#else
void recPADDSW() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBSW
void recPSUBSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSW); }
#else
void recPSUBSW() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PEXTLW
void recPEXTLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLW); }
#else
void recPEXTLW() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PPACW
void recPPACW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACW); }
#else
void recPPACW() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PADDSH
void recPADDSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSH); }
#else
void recPADDSH() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBSH
void recPSUBSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSH); }
#else
void recPSUBSH() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PEXTLH
void recPEXTLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLH); }
#else
void recPEXTLH() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PPACH
void recPPACH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACH); }
#else
void recPPACH() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PADDSB
void recPADDSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSB); }
#else
void recPADDSB() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBSB
void recPSUBSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSB); }
#else
void recPSUBSB() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PEXTLB
void recPEXTLB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLB); }
#else
void recPEXTLB() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

#if ISTUB_PPACB
void recPPACB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACB); }
#else
void recPPACB() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

#if ISTUB_PEXT5
void recPEXT5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXT5); }
#else
void recPEXT5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXT5); }
#endif

#if ISTUB_PPAC5
void recPPAC5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPAC5); }
#else
void recPPAC5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPAC5); }
#endif

#if ISTUB_PADDUW
void recPADDUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUW); }
#else
void recPADDUW() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBUW
void recPSUBUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUW); }
#else
void recPSUBUW() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PEXTUW
void recPEXTUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUW); }
#else
void recPEXTUW() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PADDUH
void recPADDUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUH); }
#else
void recPADDUH() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBUH
void recPSUBUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUH); }
#else
void recPSUBUH() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PEXTUH
void recPEXTUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUH); }
#else
void recPEXTUH() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PADDUB
void recPADDUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUB); }
#else
void recPADDUB() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBUB
void recPSUBUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUB); }
#else
void recPSUBUB() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PEXTUB
void recPEXTUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUB); }
#else
void recPEXTUB() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

// ============================================================================
//  MMI1 — PABSW/H, PCEQW/H/B, PMIN/MAX, PADSBH, QFSRV
// ============================================================================

#if ISTUB_PABSW
void recPABSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PABSW); }
#else
void recPABSW() { armMMIUnaryOp([]() { armAsm->Sqabs(a64::v0.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PCEQW
void recPCEQW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQW); }
#else
void recPCEQW() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PMINW
void recPMINW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMINW); }
#else
void recPMINW() { armMMIBinOp([]() { armAsm->Smin(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PADSBH
void recPADSBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADSBH); }
#else
void recPADSBH()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	armAsm->Sub(a64::v2.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Add(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Ins(a64::v2.D(), 1, a64::v0.D(), 1);
	armStoreGPR128(_Rd_, RVMMI2);
}
#endif

#if ISTUB_PABSH
void recPABSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PABSH); }
#else
void recPABSH() { armMMIUnaryOp([]() { armAsm->Sqabs(a64::v0.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PCEQH
void recPCEQH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQH); }
#else
void recPCEQH() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PMINH
void recPMINH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMINH); }
#else
void recPMINH() { armMMIBinOp([]() { armAsm->Smin(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PCEQB
void recPCEQB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQB); }
#else
void recPCEQB() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSLLVW
void recPSLLVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLVW); }
#else
void recPSLLVW()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Ushl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PSRLVW
void recPSRLVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLVW); }
#else
void recPSRLVW()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Neg(a64::v1.V4S(), a64::v1.V4S());
	armAsm->Ushl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_QFSRV
void recQFSRV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::QFSRV); }
#else
void recQFSRV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::QFSRV); }
#endif

// ============================================================================
//  MMI2 — PMADDW, PSLLVW, PMFHI/LO, PINTH, PMULTW, PDIVW, PCPYLD, PAND, PXOR,
//          PMADDH, PHMADH, PMSUBH, PHMSBH, PMULTH, PDIVBW, PEXEW, PROT3W
// ============================================================================

#if ISTUB_PMADDW
void recPMADDW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDW); }
#else
void recPMADDW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDW); }
#endif

#if ISTUB_PMSUBW
void recPMSUBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBW); }
#else
void recPMSUBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBW); }
#endif

#if ISTUB_PMFHI
void recPMFHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHI); }
#else
void recPMFHI()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RVMMI0, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMFLO
void recPMFLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFLO); }
#else
void recPMFLO()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armAsm->Ldr(RVMMI0, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PINTH
void recPINTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PINTH); }
#else
void recPINTH()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Ext(a64::v2.V16B(), a64::v1.V16B(), a64::v1.V16B(), 8);
	armAsm->Zip1(a64::v0.V8H(), a64::v0.V8H(), a64::v2.V8H());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMULTW
void recPMULTW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTW); }
#else
void recPMULTW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTW); }
#endif

#if ISTUB_PDIVW
void recPDIVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVW); }
#else
void recPDIVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVW); }
#endif

#if ISTUB_PCPYLD
void recPCPYLD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYLD); }
#else
void recPCPYLD()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Ins(a64::v0.D(), 1, a64::v1.D(), 0);
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMADDH
void recPMADDH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDH); }
#else
void recPMADDH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDH); }
#endif

#if ISTUB_PHMADH
void recPHMADH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMADH); }
#else
void recPHMADH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMADH); }
#endif

#if ISTUB_PAND
void recPAND() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PAND); }
#else
void recPAND() { armMMIBinOp([]() { armAsm->And(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PXOR
void recPXOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PXOR); }
#else
void recPXOR()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	if (_Rs_ == _Rt_)
	{
		armAsm->Stp(a64::xzr, a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	armMMIBinOp([]() { armAsm->Eor(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); });
}
#endif

#if ISTUB_PMSUBH
void recPMSUBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBH); }
#else
void recPMSUBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBH); }
#endif

#if ISTUB_PHMSBH
void recPHMSBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMSBH); }
#else
void recPHMSBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMSBH); }
#endif

#if ISTUB_PEXEH
void recPEXEH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXEH); }
#else
void recPEXEH()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.H(), 0, a64::v0.H(), 2);
		armAsm->Ins(a64::v1.H(), 2, a64::v0.H(), 0);
		armAsm->Ins(a64::v1.H(), 4, a64::v0.H(), 6);
		armAsm->Ins(a64::v1.H(), 6, a64::v0.H(), 4);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PREVH
void recPREVH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PREVH); }
#else
void recPREVH() { armMMIUnaryOp([]() { armAsm->Rev64(a64::v0.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PMULTH
void recPMULTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTH); }
#else
void recPMULTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTH); }
#endif

#if ISTUB_PDIVBW
void recPDIVBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVBW); }
#else
void recPDIVBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVBW); }
#endif

#if ISTUB_PEXEW
void recPEXEW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXEW); }
#else
void recPEXEW()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 0, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PROT3W
void recPROT3W() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PROT3W); }
#else
void recPROT3W()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 0, a64::v0.S(), 1);
		armAsm->Ins(a64::v1.S(), 1, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

// ============================================================================
//  MMI3 — PMADDUW, PSRAVW, PMTHI/LO, PINTEH, PCPYUD, POR, PNOR,
//          PMULTUW, PDIVUW, PEXCH, PCPYH, PEXCW
// ============================================================================

#if ISTUB_PMADDUW
void recPMADDUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDUW); }
#else
void recPMADDUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDUW); }
#endif

#if ISTUB_PSRAVW
void recPSRAVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAVW); }
#else
void recPSRAVW()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Neg(a64::v1.V4S(), a64::v1.V4S());
	armAsm->Sshl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMTHI
void recPMTHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTHI); }
#else
void recPMTHI()
{
	armLoadGPR128(RVMMI0, _Rs_);
	armAsm->Str(RVMMI0, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

#if ISTUB_PMTLO
void recPMTLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTLO); }
#else
void recPMTLO()
{
	armLoadGPR128(RVMMI0, _Rs_);
	armAsm->Str(RVMMI0, a64::MemOperand(RCPUSTATE, LO_OFFSET));
}
#endif

#if ISTUB_PINTEH
void recPINTEH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PINTEH); }
#else
void recPINTEH()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Uzp1(a64::v2.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Ext(a64::v1.V16B(), a64::v2.V16B(), a64::v2.V16B(), 8);
	armAsm->Zip1(a64::v0.V8H(), a64::v2.V8H(), a64::v1.V8H());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PCPYUD
void recPCPYUD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYUD); }
#else
void recPCPYUD()
{
	if (!_Rd_) return;
	GPR_DEL_CONST(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	armAsm->Ins(a64::v0.D(), 0, a64::v0.D(), 1);
	armAsm->Ins(a64::v0.D(), 1, a64::v1.D(), 1);
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_POR
void recPOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::POR); }
#else
void recPOR() { armMMIBinOp([]() { armAsm->Orr(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PNOR
void recPNOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PNOR); }
#else
void recPNOR()
{
	armMMIBinOp([]() {
		armAsm->Orr(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B());
		armAsm->Mvn(a64::v0.V16B(), a64::v0.V16B());
	});
}
#endif

#if ISTUB_PMULTUW
void recPMULTUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTUW); }
#else
void recPMULTUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTUW); }
#endif

#if ISTUB_PDIVUW
void recPDIVUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVUW); }
#else
void recPDIVUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVUW); }
#endif

#if ISTUB_PEXCH
void recPEXCH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXCH); }
#else
void recPEXCH()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.H(), 1, a64::v0.H(), 2);
		armAsm->Ins(a64::v1.H(), 2, a64::v0.H(), 1);
		armAsm->Ins(a64::v1.H(), 5, a64::v0.H(), 6);
		armAsm->Ins(a64::v1.H(), 6, a64::v0.H(), 5);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PCPYH
void recPCPYH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYH); }
#else
void recPCPYH()
{
	armMMIUnaryOp([]() {
		armAsm->Dup(a64::v1.V8H(), a64::v0.H(), 0);
		armAsm->Dup(a64::v2.V8H(), a64::v0.H(), 4);
		armAsm->Ins(a64::v1.D(), 1, a64::v2.D(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PEXCW
void recPEXCW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXCW); }
#else
void recPEXCW()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 1, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 1);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#endif // !INTERP_MMI

} // namespace MMI
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
