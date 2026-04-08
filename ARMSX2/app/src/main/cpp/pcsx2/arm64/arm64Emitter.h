// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include "arm64/AsmHelpers.h"
#include "R5900.h"
#include "x86/R5900_Profiler.h"
#include "VU.h"

#include <cstddef>

// ============================================================================
//  ARM64 EE Recompiler — Register Conventions
// ============================================================================
//
// ARM64 has 31 GPRs and 32 NEON 128-bit vector registers. We exploit this by
// pinning frequently-accessed emulator state in callee-saved registers, avoiding
// constant loads from memory.
//
// GPR Pinning (callee-saved x19-x28):
//   x19 = &cpuRegs            (EE CPU state base)
//   x20 = &fpuRegs            (FPU state base)
//   x21 = eeMem->Main         (EE RAM base for fastmem)
//   x22 = recLUT              (block dispatch lookup table)
//
// Scratch registers (caller-saved):
//   x0-x3   = arguments / return value
//   x4-x15  = temporaries within a block
//   x16-x17 = vixl scratch (IP0/IP1), also used by AsmHelpers
//
// NEON registers:
//   v0-v7   = caller-saved temporaries
//   v8-v15  = lower 64b callee-saved (future: pin hot PS2 GPRs)
//   v16-v28 = caller-saved temporaries
//   v29-v31 = scratch (reserved by AsmHelpers)

namespace a64 = vixl::aarch64;

// Pinned state registers
#define RCPUSTATE    a64::x19
#define RFPUSTATE    a64::x20
#define RMEMBASE     a64::x21
#define RRECLUT      a64::x22
#define RFASTMEMBASE a64::x23

// Scratch GPRs for codegen (caller-saved, safe within a single instruction's codegen)
#define RSCRATCHGPR  a64::x4
#define RSCRATCHGPR2 a64::x5
#define RSCRATCHGPR3 a64::x6
#define RWSCRATCH    a64::w4
#define RWSCRATCH2   a64::w5
#define RWSCRATCH3   a64::w6

// Callee-saved scratch: survives across function calls (bl to interpreter helpers).
// Use this when a value must survive a delay slot that may call interpreter functions.
// x25 is callee-saved by AAPCS and saved/restored by armBeginStackFrame.
#define RDELAYSLOTGPR  a64::x25
#define RWDELAYSLOT    a64::w25

// ============================================================================
//  cpuRegs field offsets (computed from struct layout)
// ============================================================================

// GPR[n] is at offset n * 16 from cpuRegs base
static constexpr s64 GPR_OFFSET(int reg) { return offsetof(cpuRegisters, GPR) + reg * 16; }
static constexpr s64 HI_OFFSET = offsetof(cpuRegisters, HI);
static constexpr s64 LO_OFFSET = offsetof(cpuRegisters, LO);
static constexpr s64 SA_OFFSET = offsetof(cpuRegisters, sa);
static constexpr s64 PC_OFFSET = offsetof(cpuRegisters, pc);
static constexpr s64 CODE_OFFSET = offsetof(cpuRegisters, code);
static constexpr s64 CYCLE_OFFSET = offsetof(cpuRegisters, cycle);
static constexpr s64 NEXT_EVENT_CYCLE_OFFSET = offsetof(cpuRegisters, nextEventCycle);

// ============================================================================
//  Instruction field extraction macros (match R5900 conventions)
// ============================================================================

// These reference cpuRegs.code which is set before recompiling each instruction
#define _Opcode_     (cpuRegs.code >> 26)
#define _Rs_         ((cpuRegs.code >> 21) & 0x1F)
#define _Rt_         ((cpuRegs.code >> 16) & 0x1F)
#define _Rd_         ((cpuRegs.code >> 11) & 0x1F)
#define _Sa_         ((cpuRegs.code >>  6) & 0x1F)
#define _Funct_      (cpuRegs.code & 0x3F)
#define _Imm_        ((s16)(cpuRegs.code & 0xFFFF))
#define _ImmU_       ((u16)(cpuRegs.code & 0xFFFF))
#define _Target_     (cpuRegs.code & 0x3FFFFFF)

// ============================================================================
//  Constant propagation helpers (shared with x86 — same globals)
// ============================================================================

extern u32 g_cpuHasConstReg, g_cpuFlushedConstReg;
extern GPR_reg64 g_cpuConstRegs[32];

#define GPR_IS_CONST1(reg) ((g_cpuHasConstReg >> (reg)) & 1)
#define GPR_IS_CONST2(reg1, reg2) \
    (((g_cpuHasConstReg >> (reg1)) & 1) & ((g_cpuHasConstReg >> (reg2)) & 1))
#define GPR_SET_CONST(reg)  \
	do { \
		if ((reg) != 0) { \
			g_cpuHasConstReg |= (1u << (reg)); \
			g_cpuFlushedConstReg &= ~(1u << (reg)); \
		} \
	} while (0)
#define GPR_DEL_CONST(reg) \
	do { \
		if ((reg) != 0) { \
			g_cpuHasConstReg &= ~(1u << (reg)); \
			g_cpuFlushedConstReg &= ~(1u << (reg)); \
		} \
	} while (0)

// ============================================================================
//  Block state (shared with x86 recompiler)
// ============================================================================

extern u32 s_nBlockCycles;
extern bool s_nBlockInterlocked;
extern u32 pc;
extern int g_branch;
extern bool g_cpuFlushedPC, g_cpuFlushedCode, g_recompilingDelaySlot, g_maySignalException;

// ============================================================================
//  ARM64 codegen helpers
// ============================================================================

// Flush all constant registers that haven't been written back to memory yet
void armFlushConstRegs();

// Emit code to load a PS2 GPR (64-bit lower half) into an ARM64 register.
// If the GPR is constant, emits a MOV immediate. Otherwise loads from cpuRegs.
void armLoadGPR64(const a64::Register& dst, int gpr);

// Emit code to load a PS2 GPR (32-bit lower word) into a W register.
void armLoadGPR32(const a64::Register& dst, int gpr);

// Emit code to store an ARM64 register into a PS2 GPR with sign extension to 64 bits.
// Writes both 32-bit halves of the 64-bit lower portion.
void armStoreGPR64SignExt32(const a64::Register& src_w, int gpr);

// Emit code to store a 64-bit value into a PS2 GPR.
void armStoreGPR64(const a64::Register& src_x, int gpr);

// Emit a call to an interpreter function, saving/restoring necessary state.
void armCallInterpreter(void (*func)());

// Branch-call variant for interpreter branch/syscall/trap stubs.
// Sets nextEventCycle = cycle, calls the standard interpreter function
// (which handles delay slot execution, cycle counting, and event testing),
// and sets g_branch = 2.
void armBranchCallInterpreter(void (*func)());

// Emit code to flush PC to cpuRegs.pc if it hasn't been flushed yet.
void armFlushPC();

// Emit code to flush cpuRegs.code if it hasn't been flushed yet.
void armFlushCode();

// Allocate space for a backpatch thunk from the rec code buffer.
u8* recBeginThunk();
u8* recEndThunk();

// Bisect: Commented = native
// Un-comment a group to find a general problem area
// Un-comment indiviudal ops to narrow it down to a specific op

//EE
//#define INTERP_EE        // Master
//#define INTERP_BRANCH    // BEQ, BNE, J, JAL, JR, JALR, SYSCALL, BREAK, etc.
//#define INTERP_MOVE      // LUI, MFHI/LO, MTHI/LO, MOVZ, MOVN, MFSA, MTSA, etc.
//#define INTERP_COP0      // MFC0, MTC0, BC0x, TLB*, ERET, EI, DI
//#define INTERP_COP1      // MFC1, MTC1, CFC1, CTC1, BC1x, FPU arith/cmp/cvt
//#define INTERP_COP2      // QMFC2/QMTC2 (native 128b), CFC2/CTC2, BC2x, all VU0 macro math ops
//#define INTERP_ALU       // ADDU, SUBU, ADDIU, DADDU, DSUBU, DADDIU, AND/OR/XOR/NOR, SLT/U, etc.
//#define INTERP_SHIFT     // SLL, SRL, SRA, SLLV, SRLV, SRAV, DSLL/DSRL/DSRA + 32 variants
//#define INTERP_MMI       // All packed SIMD ops: PADD*/PSUB*, PCGT*, PMAX/MIN*, PCEQ*, PABS*, PSxx shifts, etc.
//#define INTERP_LOAD      // LB, LBU, LH, LHU, LW, LWU, LD, LQ, LWL/R, LDL/R, LWC1, LQC2
//#define INTERP_STORE     // SB, SH, SW, SD, SQ, SWL/R, SDL/R, SWC1, SQC2
//#define INTERP_TRAP      // TGEI, TGEIU, TLTI, TLTIU, TEQI, TNEI, TGE, TGEU, TLT, TLTU, TEQ, TNE

//VU0
// Pair-level bisect: comment out a line to force that class of pairs to fall back to vu0Exec.
//#define INTERP_VU0            // Master
//#define INTERP_VU0_PAIR       // Force every pair to fall back to vu0Exec (kills all per-pair native machinery)
//#define INTERP_VU0_HAZARD     // (Dormant — VF/CLIP hazards always fall back; native save/restore not yet implemented)
//#define INTERP_VU0_MBIT       // Fall back to vu0Exec when M-bit (bit 29) is set on the upper instruction
//#define INTERP_VU0_DTBITS     // Fall back to vu0Exec when D-bit or T-bit is set
//#define INTERP_VU0_EBIT       // Fall back to vu0Exec when E-bit is set
//#define INTERP_VU0_BRANCH     // Fall back to vu0Exec when the pair contains a branch lower op


//#define INTERP_VU0_UPPER     // FMAC arith (ADD/SUB/MUL/MADD/MSUB xyzwqi), accum, MAX/MINI, ABS, CLIP, FTOI/ITOF, NOP
//#define INTERP_VU0_LOWER_FDIV       // DIV, SQRT, RSQRT, WAITQ, WAITP
//#define INTERP_VU0_LOWER_IALU       // IADD, ISUB, IADDI, IADDIU, ISUBIU, IAND, IOR
//#define INTERP_VU0_LOWER_LOADSTORE  // LQ, LQD, LQI, SQ, SQD, SQI, ILW, ISW, ILWR, ISWR
//#define INTERP_VU0_LOWER_BRANCH     // B, BAL, JR, JALR, IBEQ, IBNE, IBLTZ, IBGTZ, IBLEZ, IBGEZ
//#define INTERP_VU0_LOWER_MISC       // MOVE, MR32, MFIR, MTIR, MFP, flag ops, random, EFU, XITOP, XTOP, XGKICK

//VU1
//#define INTERP_VU1            // Master
//#define INTERP_VU_UPPER      // FMAC arith (ADD/SUB/MUL/MADD/MSUB xyzwqi), accum, MAX/MINI, ABS, CLIP, FTOI/ITOF, NOP
//#define INTERP_VU_FDIV       // DIV, SQRT, RSQRT, WAITQ, WAITP
//#define INTERP_VU_IALU       // IADD, ISUB, IADDI, IADDIU, ISUBIU, IAND, IOR
//#define INTERP_VU_LOADSTORE  // LQ, LQD, LQI, SQ, SQD, SQI, ILW, ISW, ILWR, ISWR
//#define INTERP_VU_BRANCH     // B, BAL, JR, JALR, IBEQ, IBNE, IBLTZ, IBGTZ, IBLEZ, IBGEZ
//#define INTERP_VU_MISC       // MOVE, MR32, MFIR, MTIR, MFP, flag ops, random, EFU, XITOP, XTOP, XGKICK

//DMAC
//#define INTERP_DMAC          // VIF0, VIF1, GIF, IPU0/1, SIF0/1/2, SPR0/1 + interrupt handlers

//IOP
//#define INTERP_IOP             // Master
//#define INTERP_IOP_ALU         // BISECT: uncommented → per-instruction ISTUBs in iR3000Atables_arm64.cpp
//#define INTERP_IOP_BRANCH      // BEQ/BNE/BLEZ/BGTZ/BLTZ/BGEZ/BLTZAL/BGEZAL/J/JAL/JR/JALR
//#define INTERP_IOP_SHIFT       // SLL/SRL/SRA/SLLV/SRLV/SRAV
//#define INTERP_IOP_MULTDIV     // MULT/MULTU/DIV/DIVU/MFHI/MTHI/MFLO/MTLO
//#define INTERP_IOP_MOVE        // MFHI/MTHI/MFLO/MTLO
//#define INTERP_IOP_LOADSTORE   // LB/LBU/LH/LHU/LW/LWL/LWR/SB/SH/SW/SWL/SWR
//#define INTERP_IOP_COP0        // MFC0/MTC0/CFC0/CTC0/RFE
#define INTERP_IOP_COP2        // All GTE (keep stubbed)
#define INTERP_IOP_SYSTEM      // SYSCALL/BREAK