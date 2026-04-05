// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Upper Instruction Table
// All entries emit BL to the VU0 interpreter function.
// Native NEON codegen can replace individual entries later.

#include "Common.h"
#include "VUops.h"
#include "VU.h"
#include "arm64/AsmHelpers.h"

using VU0RecFn = void (*)();

static void recVU0_Upper_Interp()
{
	armEmitCall(reinterpret_cast<const void*>(VU0_UPPER_OPCODE[VU0.code & 0x3f]));
}

VU0RecFn recVU0_UpperTable[64] = {
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x00-0x03
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x04-0x07
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x08-0x0B
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x0C-0x0F
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x10-0x13
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x14-0x17
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x18-0x1B
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x1C-0x1F
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x20-0x23
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x24-0x27
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x28-0x2B
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x2C-0x2F
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x30-0x33
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x34-0x37
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x38-0x3B
	recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, recVU0_Upper_Interp, // 0x3C-0x3F
};
