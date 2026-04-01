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
#define RCPUSTATE   a64::x19
#define RFPUSTATE   a64::x20
#define RMEMBASE    a64::x21
#define RRECLUT     a64::x22

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
#define GPR_IS_CONST2(reg1, reg2) ((g_cpuHasConstReg >> (reg1)) & (g_cpuHasConstReg >> (reg2)) & 1)
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

// Emit code to flush PC to cpuRegs.pc if it hasn't been flushed yet.
void armFlushPC();

// Emit code to flush cpuRegs.code if it hasn't been flushed yet.
void armFlushCode();

// Bisect: comment out a line to enable native codegen for that file.
// All uncommented = all interp stubs. Comment one at a time to find the bug.
//#define INTERP_BRANCH    // BEQ, BNE, J, JAL, JR, JALR, SYSCALL, BREAK, etc.
