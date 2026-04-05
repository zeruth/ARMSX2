// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Interpreter-fallback stubs for all non-branch opcodes.
// Each rec*() emits a call to the corresponding interpreter function.

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Helper macro: emit interpreter call for a standard opcode
#define REC_INTERP(name) \
	void rec##name() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::name); }

// Helper macro: emit interpreter call for an MMI opcode
#define REC_INTERP_MMI(name) \
	void rec##name() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::name); }

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ---- Misc ----
REC_INTERP(Unknown)
REC_INTERP(MMI_Unknown)
REC_INTERP(COP0_Unknown)
REC_INTERP(COP1_Unknown)
REC_INTERP(COP2)
REC_INTERP(CACHE)
REC_INTERP(PREF)
REC_INTERP(SYNC)

// ---- Traps — see iR5900Trap_arm64.cpp ----

// ---- MultDiv (remaining arithmetic) ----
REC_INTERP(MULT)
REC_INTERP(MULTU)
REC_INTERP(MULT1)
REC_INTERP(MULTU1)
REC_INTERP(MADD)
REC_INTERP(MADDU)
REC_INTERP(MADD1)
REC_INTERP(MADDU1)
REC_INTERP(DIV)
REC_INTERP(DIVU)
REC_INTERP(DIV1)
REC_INTERP(DIVU1)

// ---- ALU — see iR5900ALU_arm64.cpp ----
// ---- Shifts — see iR5900Shift_arm64.cpp ----

// ---- Moves — see iR5900Move_arm64.cpp ----

// ---- Loads/Stores — see iR5900LoadStore_arm64.cpp ----

// ---- MMI — see iR5900MMI_arm64.cpp ----

// ---- COP0 — see iR5900COP0_arm64.cpp ----

// ---- COP1 — see iR5900COP1_arm64.cpp ----

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
