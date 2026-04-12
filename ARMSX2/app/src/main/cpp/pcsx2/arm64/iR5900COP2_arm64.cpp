// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP2 (VU0 Macro Mode) Instructions
//
// Full dispatch for all VU0 macro-mode opcodes:
//   QMFC2 / CFC2 / QMTC2 / CTC2, BC2x,
//   SPECIAL1 (VADDx..w, VMADD, VMUL, VMAX, VMINI, VIADD, ..., VCALLMS),
//   SPECIAL2 (VADDAx..w, VITOF, VFTOI, VMULA, VSUBA, VOPMSUB, VMOVE, ..., VRXOR).
//
// QMFC2 and QMTC2 are implemented natively using 128-bit Q-register transfers.
// All VU0 math ops dispatch directly to their per-op interpreter functions,
// eliminating the double-dispatch overhead of the old single REC_INTERP(COP2) stub.
//
// Reference: app/src/main/cpp/pcsx2/x86/microVU_Macro.inl

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "VUmicro.h"
#include "VU.h"
#include "iRecAnalysis.h" // EEINST, EEINST_COP2_SYNC_VU0, EEINST_COP2_FINISH_VU0, g_pCurInstInfo

using namespace R5900;

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  INTERP_COP2 master switch + per-op ISTUB toggles
//
//  Set INTERP_COP2 (or INTERP_EE) in arm64Emitter.h to force all COP2 ops to
//  interpreter stubs for bisection.  Per-op ISTUBs let you toggle individual
//  ops when the master switch is off.
// ============================================================================

#if defined(INTERP_COP2) || defined(INTERP_EE)
#define ISTUB_QMFC2  1
#define ISTUB_CFC2   1
#define ISTUB_QMTC2  1
#define ISTUB_CTC2   1
#define ISTUB_BC2F   1
#define ISTUB_BC2T   1
#define ISTUB_BC2FL  1
#define ISTUB_BC2TL  1
#else
// Native: QMFC2/QMTC2 use 128-bit Q-register LDR/STR.
// CFC2/CTC2 have complex special-case register semantics (REG_R, REG_STATUS_FLAG,
// REG_FBRST, REG_CMSAR1); keep as stubs for now.
// BC2x need full branch-delay-slot machinery; keep as stubs.
#define ISTUB_QMFC2  0
#define ISTUB_CFC2   1
#define ISTUB_QMTC2  0
#define ISTUB_CTC2   1
#define ISTUB_BC2F   1
#define ISTUB_BC2T   1
#define ISTUB_BC2FL  1
#define ISTUB_BC2TL  1
#endif

// ============================================================================
//  Helper macro — emit interpreter call for a global-scope VU0 interpreter
//  function.  The functions QMFC2/VADD/VMUL/etc. live in global scope in
//  R5900OpcodeTables.h, not in a sub-namespace.
// ============================================================================
#define REC_COP2_INTERP(name) \
    void recV##name() { armCallInterpreter(::V##name); }

// ============================================================================
//  QMFC2 — Move 128-bit VF[rd] → GPR[rt]
// ============================================================================
void recQMFC2()
{
#if ISTUB_QMFC2
    armCallInterpreter(::QMFC2);
#else
    if (!_Rt_)
        return;
    // I-bit (bit 0 of instruction) or sync/finish flag → fall back to interpreter
    // so the interlock and VU0 micro-execution are handled correctly.
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::QMFC2);
        return;
    }
    // Native 128-bit copy: VU0.VF[_Rd_] → cpuRegs.GPR[_Rt_]
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rd_]));
    armAsm->Ldr(a64::q0, a64::MemOperand(RSCRATCHGPR));
    armAsm->Str(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    GPR_DEL_CONST(_Rt_);
#endif
}

// ============================================================================
//  QMTC2 — Move 128-bit GPR[rt] → VF[rd]
// ============================================================================
void recQMTC2()
{
#if ISTUB_QMTC2
    armCallInterpreter(::QMTC2);
#else
    if (!_Rd_)
        return;
    // I-bit or sync/finish → interpreter handles interlock.
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::QMTC2);
        return;
    }
    // Native 128-bit copy: cpuRegs.GPR[_Rt_] → VU0.VF[_Rd_]
    // Commit any pending const-prop value before the direct 128-bit LDR.
    armFlushConstReg(_Rt_);
    armAsm->Ldr(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rd_]));
    armAsm->Str(a64::q0, a64::MemOperand(RSCRATCHGPR));
#endif
}

// ============================================================================
//  CFC2 / CTC2 — interpreter stubs (complex special-case register handling)
// ============================================================================
void recCFC2() { armCallInterpreter(::CFC2); }
void recCTC2() { armCallInterpreter(::CTC2); }

// ============================================================================
//  BC2x — branch on COP2 condition
//  Interpreter handles delay slot + branch target.  Mark as branch so the
//  block compiler ends the block here.
// ============================================================================
void recBC2F()  { armCallInterpreter(::BC2F);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
void recBC2T()  { armCallInterpreter(::BC2T);  pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
void recBC2FL() { armCallInterpreter(::BC2FL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }
void recBC2TL() { armCallInterpreter(::BC2TL); pc += 4; g_branch = 1; g_cpuFlushedPC = true; }

// ============================================================================
//  VU0 SPECIAL1 math ops — direct per-op interpreter dispatch
//  (no double-dispatch overhead through the COP2 master interpreter switch)
// ============================================================================

// Broadcast scalar ops (x/y/z/w suffix)
REC_COP2_INTERP(ADDx)  REC_COP2_INTERP(ADDy)  REC_COP2_INTERP(ADDz)  REC_COP2_INTERP(ADDw)
REC_COP2_INTERP(SUBx)  REC_COP2_INTERP(SUBy)  REC_COP2_INTERP(SUBz)  REC_COP2_INTERP(SUBw)
REC_COP2_INTERP(MADDx) REC_COP2_INTERP(MADDy) REC_COP2_INTERP(MADDz) REC_COP2_INTERP(MADDw)
REC_COP2_INTERP(MSUBx) REC_COP2_INTERP(MSUBy) REC_COP2_INTERP(MSUBz) REC_COP2_INTERP(MSUBw)
REC_COP2_INTERP(MAXx)  REC_COP2_INTERP(MAXy)  REC_COP2_INTERP(MAXz)  REC_COP2_INTERP(MAXw)
REC_COP2_INTERP(MINIx) REC_COP2_INTERP(MINIy) REC_COP2_INTERP(MINIz) REC_COP2_INTERP(MINIw)
REC_COP2_INTERP(MULx)  REC_COP2_INTERP(MULy)  REC_COP2_INTERP(MULz)  REC_COP2_INTERP(MULw)

// Q/I register ops
REC_COP2_INTERP(MULq)  REC_COP2_INTERP(MAXi)  REC_COP2_INTERP(MULi)  REC_COP2_INTERP(MINIi)
REC_COP2_INTERP(ADDq)  REC_COP2_INTERP(MADDq) REC_COP2_INTERP(ADDi)  REC_COP2_INTERP(MADDi)
REC_COP2_INTERP(SUBq)  REC_COP2_INTERP(MSUBq) REC_COP2_INTERP(SUBi)  REC_COP2_INTERP(MSUBi)

// Full-vector ops
REC_COP2_INTERP(ADD)   REC_COP2_INTERP(MADD)  REC_COP2_INTERP(MUL)   REC_COP2_INTERP(MAX)
REC_COP2_INTERP(SUB)   REC_COP2_INTERP(MSUB)  REC_COP2_INTERP(OPMSUB)REC_COP2_INTERP(MINI)

// Integer ops
REC_COP2_INTERP(IADD)  REC_COP2_INTERP(ISUB)  REC_COP2_INTERP(IADDI)
REC_COP2_INTERP(IAND)  REC_COP2_INTERP(IOR)

// Micro-subroutine calls
REC_COP2_INTERP(CALLMS)  REC_COP2_INTERP(CALLMSR)

// ============================================================================
//  VU0 SPECIAL2 ops — accumulator, conversion, misc
// ============================================================================

// Accumulator ops
REC_COP2_INTERP(ADDAx)  REC_COP2_INTERP(ADDAy)  REC_COP2_INTERP(ADDAz)  REC_COP2_INTERP(ADDAw)
REC_COP2_INTERP(SUBAx)  REC_COP2_INTERP(SUBAy)  REC_COP2_INTERP(SUBAz)  REC_COP2_INTERP(SUBAw)
REC_COP2_INTERP(MADDAx) REC_COP2_INTERP(MADDAy) REC_COP2_INTERP(MADDAz) REC_COP2_INTERP(MADDAw)
REC_COP2_INTERP(MSUBAx) REC_COP2_INTERP(MSUBAy) REC_COP2_INTERP(MSUBAz) REC_COP2_INTERP(MSUBAw)

// Conversion
REC_COP2_INTERP(ITOF0)  REC_COP2_INTERP(ITOF4)  REC_COP2_INTERP(ITOF12) REC_COP2_INTERP(ITOF15)
REC_COP2_INTERP(FTOI0)  REC_COP2_INTERP(FTOI4)  REC_COP2_INTERP(FTOI12) REC_COP2_INTERP(FTOI15)

// Multiply-accumulate (broadcast)
REC_COP2_INTERP(MULAx)  REC_COP2_INTERP(MULAy)  REC_COP2_INTERP(MULAz)  REC_COP2_INTERP(MULAw)
REC_COP2_INTERP(MULAq)  REC_COP2_INTERP(ABS)    REC_COP2_INTERP(MULAi)  REC_COP2_INTERP(CLIPw)

// Accumulator Q/I ops
REC_COP2_INTERP(ADDAq)  REC_COP2_INTERP(MADDAq) REC_COP2_INTERP(ADDAi)  REC_COP2_INTERP(MADDAi)
REC_COP2_INTERP(SUBAq)  REC_COP2_INTERP(MSUBAq) REC_COP2_INTERP(SUBAi)  REC_COP2_INTERP(MSUBAi)

// Full-vector accumulator
REC_COP2_INTERP(ADDA)   REC_COP2_INTERP(MADDA)  REC_COP2_INTERP(MULA)
REC_COP2_INTERP(SUBA)   REC_COP2_INTERP(MSUBA)  REC_COP2_INTERP(OPMULA)

// NOP — hardware no-op, emit nothing
void recVNOP() {}

// Data movement
REC_COP2_INTERP(MOVE)   REC_COP2_INTERP(MR32)

// Load/store
REC_COP2_INTERP(LQI)    REC_COP2_INTERP(SQI)    REC_COP2_INTERP(LQD)    REC_COP2_INTERP(SQD)

// Division (result → Q register, stalls until done via VWAITQ)
REC_COP2_INTERP(DIV)    REC_COP2_INTERP(SQRT)   REC_COP2_INTERP(RSQRT)

// VWAITQ — stall until FDIV pipeline drains
void recVWAITQ() { armCallInterpreter(::VWAITQ); }

// Integer ↔ VF transfer
REC_COP2_INTERP(MTIR)   REC_COP2_INTERP(MFIR)   REC_COP2_INTERP(ILWR)   REC_COP2_INTERP(ISWR)

// Random number generator
REC_COP2_INTERP(RNEXT)  REC_COP2_INTERP(RGET)   REC_COP2_INTERP(RINIT)  REC_COP2_INTERP(RXOR)

// ============================================================================
//  Unknown/invalid COP2 opcode
// ============================================================================
static void rec_C2UNK()
{
    Console.Error("COP2 unknown opcode: %08X", cpuRegs.code);
}

// ============================================================================
//  Sub-dispatchers (forward-declared because they reference each other via tables)
// ============================================================================
static void recCOP2_BC2();
static void recCOP2_SPEC1();
static void recCOP2_SPEC2();

// ============================================================================
//  Dispatch table: recCOP2t[32]
//  Indexed by _Rs_ (bits 25:21 of the instruction).
//  Mirrors x86 microVU_Macro.inl recCOP2t[].
// ============================================================================
static void (*recCOP2t[32])() = {
    rec_C2UNK,     recQMFC2,      recCFC2,       rec_C2UNK,
    rec_C2UNK,     recQMTC2,      recCTC2,       rec_C2UNK,
    recCOP2_BC2,   rec_C2UNK,     rec_C2UNK,     rec_C2UNK,
    rec_C2UNK,     rec_C2UNK,     rec_C2UNK,     rec_C2UNK,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
};

// ============================================================================
//  Dispatch table: recCOP2_BC2t[32]
//  Indexed by _Rt_ (bits 20:16) when _Rs_ == 8.
// ============================================================================
static void (*recCOP2_BC2t[32])() = {
    recBC2F,   recBC2T,   recBC2FL,  recBC2TL,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
};

// ============================================================================
//  Dispatch table: recCOP2SPECIAL1t[64]
//  Indexed by _Funct_ (bits 5:0) when _Rs_ >= 0x10.
//  Matches x86 recCOP2SPECIAL1t[] exactly.
// ============================================================================
static void (*recCOP2SPECIAL1t[64])() = {
    recVADDx,    recVADDy,    recVADDz,    recVADDw,
    recVSUBx,    recVSUBy,    recVSUBz,    recVSUBw,
    recVMADDx,   recVMADDy,   recVMADDz,   recVMADDw,
    recVMSUBx,   recVMSUBy,   recVMSUBz,   recVMSUBw,
    recVMAXx,    recVMAXy,    recVMAXz,    recVMAXw,
    recVMINIx,   recVMINIy,   recVMINIz,   recVMINIw,
    recVMULx,    recVMULy,    recVMULz,    recVMULw,
    recVMULq,    recVMAXi,    recVMULi,    recVMINIi,
    recVADDq,    recVMADDq,   recVADDi,    recVMADDi,
    recVSUBq,    recVMSUBq,   recVSUBi,    recVMSUBi,
    recVADD,     recVMADD,    recVMUL,     recVMAX,
    recVSUB,     recVMSUB,    recVOPMSUB,  recVMINI,
    recVIADD,    recVISUB,    recVIADDI,   rec_C2UNK,
    recVIAND,    recVIOR,     rec_C2UNK,   rec_C2UNK,
    recVCALLMS,  recVCALLMSR, rec_C2UNK,   rec_C2UNK,
    recCOP2_SPEC2, recCOP2_SPEC2, recCOP2_SPEC2, recCOP2_SPEC2,
};

// ============================================================================
//  Dispatch table: recCOP2SPECIAL2t[128]
//  Indexed by (code & 3) | ((code >> 4) & 0x7C) — 7-bit field from bits[10:6|1:0].
//  Matches x86 recCOP2SPECIAL2t[] exactly.
// ============================================================================
static void (*recCOP2SPECIAL2t[128])() = {
    recVADDAx,   recVADDAy,   recVADDAz,   recVADDAw,
    recVSUBAx,   recVSUBAy,   recVSUBAz,   recVSUBAw,
    recVMADDAx,  recVMADDAy,  recVMADDAz,  recVMADDAw,
    recVMSUBAx,  recVMSUBAy,  recVMSUBAz,  recVMSUBAw,
    recVITOF0,   recVITOF4,   recVITOF12,  recVITOF15,
    recVFTOI0,   recVFTOI4,   recVFTOI12,  recVFTOI15,
    recVMULAx,   recVMULAy,   recVMULAz,   recVMULAw,
    recVMULAq,   recVABS,     recVMULAi,   recVCLIPw,
    recVADDAq,   recVMADDAq,  recVADDAi,   recVMADDAi,
    recVSUBAq,   recVMSUBAq,  recVSUBAi,   recVMSUBAi,
    recVADDA,    recVMADDA,   recVMULA,    rec_C2UNK,
    recVSUBA,    recVMSUBA,   recVOPMULA,  recVNOP,
    recVMOVE,    recVMR32,    rec_C2UNK,   rec_C2UNK,
    recVLQI,     recVSQI,     recVLQD,     recVSQD,
    recVDIV,     recVSQRT,    recVRSQRT,   recVWAITQ,
    recVMTIR,    recVMFIR,    recVILWR,    recVISWR,
    recVRNEXT,   recVRGET,    recVRINIT,   recVRXOR,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
};

// ============================================================================
//  Sub-dispatcher implementations
// ============================================================================

static void recCOP2_BC2()   { recCOP2_BC2t[_Rt_](); }
static void recCOP2_SPEC2() { recCOP2SPECIAL2t[(cpuRegs.code & 3) | ((cpuRegs.code >> 4) & 0x7C)](); }

static void recCOP2_SPEC1()
{
    // If analysis pass flagged that VU0 needs to be synced/finished before this
    // macro-mode instruction, emit a call to _vu0FinishMicro.  This matches the
    // x86 mVUFinishVU0() path in recCOP2_SPEC1().
    if (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0)))
        armCallInterpreter(_vu0FinishMicro);
    recCOP2SPECIAL1t[_Funct_]();
}

// ============================================================================
//  Master COP2 dispatch — called from the EE opcode dispatch table
// ============================================================================
void recCOP2() { recCOP2t[_Rs_](); }

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
