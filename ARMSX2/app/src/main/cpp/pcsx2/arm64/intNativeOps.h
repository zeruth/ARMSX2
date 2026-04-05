// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 Native Interpreter Ops — optimized C++ replacements for interpreter opcodes.
// Uses the interpreter's execution harness (cycles, events, branches) but swaps in
// native implementations for individual ops.
//
// Guard structure:
//   Per-system defines enable/disable groups. Comment out = full interpreter.
//   Per-op defines within each group can be individually toggled (1=native, 0=interp).

#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

// ============================================================================
//  Per-system guards — uncomment a line to enable native ops for that group.
//  All commented out = pure interpreter (baseline, known working).
// ============================================================================

#define NATIVE_ALU       // ADDU, SUBU, ADDIU, DADDU, DSUBU, DADDIU, AND/OR/XOR/NOR, SLT/U, etc.
#define NATIVE_SHIFT     // SLL, SRL, SRA, SLLV, SRLV, SRAV, DSLL/DSRL/DSRA + 32 variants
#define NATIVE_MOVE      // LUI, MFHI/LO, MTHI/LO, MOVZ, MOVN, MFSA, MTSA
#define NATIVE_BRANCH    // BEQ, BNE, J, JAL, JR, JALR, BGEZ, BLTZ, SYSCALL, BREAK, etc.
#define NATIVE_LOAD      // LB, LBU, LH, LHU, LW, LWU, LD, LQ, etc.
#define NATIVE_STORE     // SB, SH, SW, SD, SQ, etc.
#define NATIVE_COP0      // MFC0, MTC0, BC0x, TLB*, ERET, EI, DI
#define NATIVE_COP1      // MFC1, MTC1, CFC1, CTC1, BC1x, FPU arith/cmp/cvt
#define NATIVE_MMI       // Multimedia instructions

// ============================================================================
//  Per-op guards — within each enabled group, toggle individual ops.
//  1 = use native implementation, 0 = fall back to interpreter.
// ============================================================================

#ifdef NATIVE_ALU
// No-overflow register ops (SPECIAL funct)
#define NOP_ADDU   1
#define NOP_SUBU   1
#define NOP_DADDU  1
#define NOP_DSUBU  1
#define NOP_AND    1
#define NOP_OR     1
#define NOP_XOR    1
#define NOP_NOR    1
#define NOP_SLT    1
#define NOP_SLTU   1
// Overflow-trapping register ops (SPECIAL funct)
#define NOP_ADD    1
#define NOP_SUB    1
#define NOP_DADD   1
#define NOP_DSUB   1
// Immediate ops (top-level opcode)
#define NOP_ADDIU  1
#define NOP_DADDIU 1
#define NOP_ANDI   1
#define NOP_ORI    1
#define NOP_XORI   1
#define NOP_SLTI   1
#define NOP_SLTIU  1
// Overflow-trapping immediate ops
#define NOP_ADDI   1
#define NOP_DADDI  1
#else
#define NOP_ADDU   0
#define NOP_SUBU   0
#define NOP_DADDU  0
#define NOP_DSUBU  0
#define NOP_AND    0
#define NOP_OR     0
#define NOP_XOR    0
#define NOP_NOR    0
#define NOP_SLT    0
#define NOP_SLTU   0
#define NOP_ADD    0
#define NOP_SUB    0
#define NOP_DADD   0
#define NOP_DSUB   0
#define NOP_ADDIU  0
#define NOP_DADDIU 0
#define NOP_ANDI   0
#define NOP_ORI    0
#define NOP_XORI   0
#define NOP_SLTI   0
#define NOP_SLTIU  0
#define NOP_ADDI   0
#define NOP_DADDI  0
#endif

#ifdef NATIVE_BRANCH
// Top-level branches (opcode field)
#define NOP_J        1
#define NOP_JAL      1
#define NOP_BEQ      1
#define NOP_BNE      1
#define NOP_BLEZ     1
#define NOP_BGTZ     1
#define NOP_BEQL     1
#define NOP_BNEL     1
#define NOP_BLEZL    1
#define NOP_BGTZL    1
// SPECIAL branches (funct field)
#define NOP_JR       1
#define NOP_JALR     1
#define NOP_SYSCALL  1
#define NOP_BREAK    1
// REGIMM branches (rt field)
#define NOP_BLTZ     1
#define NOP_BGEZ     1
#define NOP_BLTZL    1
#define NOP_BGEZL    1
#define NOP_BLTZAL   1
#define NOP_BGEZAL   1
#define NOP_BLTZALL  1
#define NOP_BGEZALL  1
#else
#define NOP_J        0
#define NOP_JAL      0
#define NOP_BEQ      0
#define NOP_BNE      0
#define NOP_BLEZ     0
#define NOP_BGTZ     0
#define NOP_BEQL     0
#define NOP_BNEL     0
#define NOP_BLEZL    0
#define NOP_BGTZL    0
#define NOP_JR       0
#define NOP_JALR     0
#define NOP_SYSCALL  0
#define NOP_BREAK    0
#define NOP_BLTZ     0
#define NOP_BGEZ     0
#define NOP_BLTZL    0
#define NOP_BGEZL    0
#define NOP_BLTZAL   0
#define NOP_BGEZAL   0
#define NOP_BLTZALL  0
#define NOP_BGEZALL  0
#endif

#ifdef NATIVE_MOVE
#define NOP_LUI      1
#define NOP_MFHI     1
#define NOP_MFLO     1
#define NOP_MTHI     1
#define NOP_MTLO     1
#define NOP_MFHI1    1
#define NOP_MFLO1    1
#define NOP_MTHI1    1
#define NOP_MTLO1    1
#define NOP_MOVZ     1
#define NOP_MOVN     1
#define NOP_MFSA     1
#define NOP_MTSA     1
#define NOP_MTSAB    1
#define NOP_MTSAH    1
#else
#define NOP_LUI      0
#define NOP_MFHI     0
#define NOP_MFLO     0
#define NOP_MTHI     0
#define NOP_MTLO     0
#define NOP_MFHI1    0
#define NOP_MFLO1    0
#define NOP_MTHI1    0
#define NOP_MTLO1    0
#define NOP_MOVZ     0
#define NOP_MOVN     0
#define NOP_MFSA     0
#define NOP_MTSA     0
#define NOP_MTSAB    0
#define NOP_MTSAH    0
#endif

#ifdef NATIVE_SHIFT
#define NOP_SLL      1
#define NOP_SRL      1
#define NOP_SRA      1
#define NOP_SLLV     1
#define NOP_SRLV     1
#define NOP_SRAV     1
#define NOP_DSLL     1
#define NOP_DSRL     1
#define NOP_DSRA     1
#define NOP_DSLL32   1
#define NOP_DSRL32   1
#define NOP_DSRA32   1
#define NOP_DSLLV    1
#define NOP_DSRLV    1
#define NOP_DSRAV    1
#else
#define NOP_SLL      0
#define NOP_SRL      0
#define NOP_SRA      0
#define NOP_SLLV     0
#define NOP_SRLV     0
#define NOP_SRAV     0
#define NOP_DSLL     0
#define NOP_DSRL     0
#define NOP_DSRA     0
#define NOP_DSLL32   0
#define NOP_DSRL32   0
#define NOP_DSRA32   0
#define NOP_DSLLV    0
#define NOP_DSRLV    0
#define NOP_DSRAV    0
#endif

#ifdef NATIVE_LOAD
#define NOP_LB       1
#define NOP_LBU      1
#define NOP_LH       1
#define NOP_LHU      1
#define NOP_LW       1
#define NOP_LWU      1
#define NOP_LD       1
#define NOP_LQ       1
#define NOP_LWL      1
#define NOP_LWR      1
#define NOP_LDL      1
#define NOP_LDR      1
#define NOP_LWC1     1
#define NOP_LQC2     1
#else
#define NOP_LB       0
#define NOP_LBU      0
#define NOP_LH       0
#define NOP_LHU      0
#define NOP_LW       0
#define NOP_LWU      0
#define NOP_LD       0
#define NOP_LQ       0
#define NOP_LWL      0
#define NOP_LWR      0
#define NOP_LDL      0
#define NOP_LDR      0
#define NOP_LWC1     0
#define NOP_LQC2     0
#endif

#ifdef NATIVE_STORE
#define NOP_SB       1
#define NOP_SH       1
#define NOP_SW       1
#define NOP_SD       1
#define NOP_SQ       1
#define NOP_SWL      1
#define NOP_SWR      1
#define NOP_SDL      1
#define NOP_SDR      1
#define NOP_SWC1     1
#define NOP_SQC2     1
#else
#define NOP_SB       0
#define NOP_SH       0
#define NOP_SW       0
#define NOP_SD       0
#define NOP_SQ       0
#define NOP_SWL      0
#define NOP_SWR      0
#define NOP_SDL      0
#define NOP_SDR      0
#define NOP_SWC1     0
#define NOP_SQC2     0
#endif

#ifdef NATIVE_COP0
#define NOP_MFC0     1
#define NOP_MTC0     1
#define NOP_BC0F     1
#define NOP_BC0T     1
#define NOP_BC0FL    1
#define NOP_BC0TL    1
#define NOP_TLBR     1
#define NOP_TLBWI    1
#define NOP_TLBWR    1
#define NOP_TLBP     1
#define NOP_ERET     1
#define NOP_EI       1
#define NOP_DI       1
#else
#define NOP_MFC0     0
#define NOP_MTC0     0
#define NOP_BC0F     0
#define NOP_BC0T     0
#define NOP_BC0FL    0
#define NOP_BC0TL    0
#define NOP_TLBR     0
#define NOP_TLBWI    0
#define NOP_TLBWR    0
#define NOP_TLBP     0
#define NOP_ERET     0
#define NOP_EI       0
#define NOP_DI       0
#endif

#ifdef NATIVE_COP1
// COP1 top-level (rs field)
#define NOP_MFC1     1
#define NOP_CFC1     1
#define NOP_MTC1     1
#define NOP_CTC1     1
// BC1 sub-table (rt field)
#define NOP_BC1F     1
#define NOP_BC1T     1
#define NOP_BC1FL    1
#define NOP_BC1TL    1
// S sub-table (funct field) — single-precision ops
#define NOP_ADD_S    1
#define NOP_SUB_S    1
#define NOP_MUL_S    1
#define NOP_DIV_S    1
#define NOP_SQRT_S   1
#define NOP_ABS_S    1
#define NOP_MOV_S    1
#define NOP_NEG_S    1
#define NOP_RSQRT_S  1
#define NOP_ADDA_S   1
#define NOP_SUBA_S   1
#define NOP_MULA_S   1
#define NOP_MADD_S   1
#define NOP_MSUB_S   1
#define NOP_MADDA_S  1
#define NOP_MSUBA_S  1
#define NOP_CVT_W    1
#define NOP_MAX_S    1
#define NOP_MIN_S    1
#define NOP_C_F      1
#define NOP_C_EQ     1
#define NOP_C_LT     1
#define NOP_C_LE     1
// W sub-table (funct field)
#define NOP_CVT_S    1
#else
#define NOP_MFC1     0
#define NOP_CFC1     0
#define NOP_MTC1     0
#define NOP_CTC1     0
#define NOP_BC1F     0
#define NOP_BC1T     0
#define NOP_BC1FL    0
#define NOP_BC1TL    0
#define NOP_ADD_S    0
#define NOP_SUB_S    0
#define NOP_MUL_S    0
#define NOP_DIV_S    0
#define NOP_SQRT_S   0
#define NOP_ABS_S    0
#define NOP_MOV_S    0
#define NOP_NEG_S    0
#define NOP_RSQRT_S  0
#define NOP_ADDA_S   0
#define NOP_SUBA_S   0
#define NOP_MULA_S   0
#define NOP_MADD_S   0
#define NOP_MSUB_S   0
#define NOP_MADDA_S  0
#define NOP_MSUBA_S  0
#define NOP_CVT_W    0
#define NOP_MAX_S    0
#define NOP_MIN_S    0
#define NOP_C_F      0
#define NOP_C_EQ     0
#define NOP_C_LT     0
#define NOP_C_LE     0
#define NOP_CVT_S    0
#endif

#ifdef NATIVE_MMI
// tbl_MMI direct (funct field)
#define NOP_MADD     1
#define NOP_MADDU    1
#define NOP_PLZCW    1
#define NOP_MULT1    1
#define NOP_MULTU1   1
#define NOP_DIV1     1
#define NOP_DIVU1    1
#define NOP_MADD1    1
#define NOP_MADDU1   1
#define NOP_PMFHL    1
#define NOP_PMTHL    1
#define NOP_PSLLH    1
#define NOP_PSRLH    1
#define NOP_PSRAH    1
#define NOP_PSLLW    1
#define NOP_PSRLW    1
#define NOP_PSRAW    1
// tbl_MMI0 (sa field, funct==0x08)
#define NOP_PADDW    1
#define NOP_PSUBW    1
#define NOP_PCGTW    1
#define NOP_PMAXW    1
#define NOP_PADDH    1
#define NOP_PSUBH    1
#define NOP_PCGTH    1
#define NOP_PMAXH    1
#define NOP_PADDB    1
#define NOP_PSUBB    1
#define NOP_PCGTB    1
#define NOP_PADDSW   1
#define NOP_PSUBSW   1
#define NOP_PEXTLW   1
#define NOP_PPACW    1
#define NOP_PADDSH   1
#define NOP_PSUBSH   1
#define NOP_PEXTLH   1
#define NOP_PPACH    1
#define NOP_PADDSB   1
#define NOP_PSUBSB   1
#define NOP_PEXTLB   1
#define NOP_PPACB    1
#define NOP_PEXT5    1
#define NOP_PPAC5    1
// tbl_MMI1 (sa field, funct==0x28)
#define NOP_PABSW    1
#define NOP_PCEQW    1
#define NOP_PMINW    1
#define NOP_PADSBH   1
#define NOP_PABSH    1
#define NOP_PCEQH    1
#define NOP_PMINH    1
#define NOP_PCEQB    1
#define NOP_PADDUW   1
#define NOP_PSUBUW   1
#define NOP_PEXTUW   1
#define NOP_PADDUH   1
#define NOP_PSUBUH   1
#define NOP_PEXTUH   1
#define NOP_PADDUB   1
#define NOP_PSUBUB   1
#define NOP_PEXTUB   1
#define NOP_QFSRV   1
// tbl_MMI2 (sa field, funct==0x09)
#define NOP_PMADDW   1
#define NOP_PSLLVW   1
#define NOP_PSRLVW   1
#define NOP_PMSUBW   1
#define NOP_PMFHI    1
#define NOP_PMFLO    1
#define NOP_PINTH    1
#define NOP_PMULTW   1
#define NOP_PDIVW    1
#define NOP_PCPYLD   1
#define NOP_PMADDH   1
#define NOP_PHMADH   1
#define NOP_PAND     1
#define NOP_PXOR     1
#define NOP_PMSUBH   1
#define NOP_PHMSBH   1
#define NOP_PEXEH    1
#define NOP_PREVH    1
#define NOP_PMULTH   1
#define NOP_PDIVBW   1
#define NOP_PEXEW    1
#define NOP_PROT3W   1
// tbl_MMI3 (sa field, funct==0x29)
#define NOP_PMADDUW  1
#define NOP_PSRAVW   1
#define NOP_PMTHI    1
#define NOP_PMTLO    1
#define NOP_PINTEH   1
#define NOP_PMULTUW  1
#define NOP_PDIVUW   1
#define NOP_PCPYUD   1
#define NOP_POR      1
#define NOP_PNOR     1
#define NOP_PEXCH    1
#define NOP_PCPYH    1
#define NOP_PEXCW    1
#else
// tbl_MMI direct
#define NOP_MADD     0
#define NOP_MADDU    0
#define NOP_PLZCW    0
#define NOP_MULT1    0
#define NOP_MULTU1   0
#define NOP_DIV1     0
#define NOP_DIVU1    0
#define NOP_MADD1    0
#define NOP_MADDU1   0
#define NOP_PMFHL    0
#define NOP_PMTHL    0
#define NOP_PSLLH    0
#define NOP_PSRLH    0
#define NOP_PSRAH    0
#define NOP_PSLLW    0
#define NOP_PSRLW    0
#define NOP_PSRAW    0
// tbl_MMI0
#define NOP_PADDW    0
#define NOP_PSUBW    0
#define NOP_PCGTW    0
#define NOP_PMAXW    0
#define NOP_PADDH    0
#define NOP_PSUBH    0
#define NOP_PCGTH    0
#define NOP_PMAXH    0
#define NOP_PADDB    0
#define NOP_PSUBB    0
#define NOP_PCGTB    0
#define NOP_PADDSW   0
#define NOP_PSUBSW   0
#define NOP_PEXTLW   0
#define NOP_PPACW    0
#define NOP_PADDSH   0
#define NOP_PSUBSH   0
#define NOP_PEXTLH   0
#define NOP_PPACH    0
#define NOP_PADDSB   0
#define NOP_PSUBSB   0
#define NOP_PEXTLB   0
#define NOP_PPACB    0
#define NOP_PEXT5    0
#define NOP_PPAC5    0
// tbl_MMI1
#define NOP_PABSW    0
#define NOP_PCEQW    0
#define NOP_PMINW    0
#define NOP_PADSBH   0
#define NOP_PABSH    0
#define NOP_PCEQH    0
#define NOP_PMINH    0
#define NOP_PCEQB    0
#define NOP_PADDUW   0
#define NOP_PSUBUW   0
#define NOP_PEXTUW   0
#define NOP_PADDUH   0
#define NOP_PSUBUH   0
#define NOP_PEXTUH   0
#define NOP_PADDUB   0
#define NOP_PSUBUB   0
#define NOP_PEXTUB   0
#define NOP_QFSRV   0
// tbl_MMI2
#define NOP_PMADDW   0
#define NOP_PSLLVW   0
#define NOP_PSRLVW   0
#define NOP_PMSUBW   0
#define NOP_PMFHI    0
#define NOP_PMFLO    0
#define NOP_PINTH    0
#define NOP_PMULTW   0
#define NOP_PDIVW    0
#define NOP_PCPYLD   0
#define NOP_PMADDH   0
#define NOP_PHMADH   0
#define NOP_PAND     0
#define NOP_PXOR     0
#define NOP_PMSUBH   0
#define NOP_PHMSBH   0
#define NOP_PEXEH    0
#define NOP_PREVH    0
#define NOP_PMULTH   0
#define NOP_PDIVBW   0
#define NOP_PEXEW    0
#define NOP_PROT3W   0
// tbl_MMI3
#define NOP_PMADDUW  0
#define NOP_PSRAVW   0
#define NOP_PMTHI    0
#define NOP_PMTLO    0
#define NOP_PINTEH   0
#define NOP_PMULTUW  0
#define NOP_PDIVUW   0
#define NOP_PCPYUD   0
#define NOP_POR      0
#define NOP_PNOR     0
#define NOP_PEXCH    0
#define NOP_PCPYH    0
#define NOP_PEXCW    0
#endif

// Returns true if the current instruction (cpuRegs.code) was executed natively.
// Called from execI() before opcode.interpret(). If true, interpret() is skipped.
bool arm64TryNativeExec();

#endif // __aarch64__ || _M_ARM64
