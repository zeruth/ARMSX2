// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Lower Instruction Table
// All entries emit BL to the VU0 interpreter function.

#include "Common.h"
#include "VUops.h"
#include "VU.h"
#include "arm64/AsmHelpers.h"

using VU0RecFn = void (*)();

static void recVU0_Lower_Interp()
{
	armEmitCall(reinterpret_cast<const void*>(VU0_LOWER_OPCODE[VU0.code >> 25]));
}

VU0RecFn recVU0_LowerTable[128] = {
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x00-0x03
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x04-0x07
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x08-0x0B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x0C-0x0F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x10-0x13
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x14-0x17
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x18-0x1B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x1C-0x1F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x20-0x23
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x24-0x27
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x28-0x2B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x2C-0x2F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x30-0x33
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x34-0x37
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x38-0x3B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x3C-0x3F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x40-0x43
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x44-0x47
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x48-0x4B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x4C-0x4F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x50-0x53
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x54-0x57
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x58-0x5B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x5C-0x5F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x60-0x63
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x64-0x67
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x68-0x6B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x6C-0x6F
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x70-0x73
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x74-0x77
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x78-0x7B
	recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, recVU0_Lower_Interp, // 0x7C-0x7F
};
