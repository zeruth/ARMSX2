// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0

#include "common/Console.h"
#include "common/Pcsx2Defs.h"
#include "MTVU.h"
#include "SaveState.h"
#include "vtlb.h"

#include "common/Assertions.h"

void vtlb_DynBackpatchLoadStore(uptr code_address, u32 code_size, u32 guest_pc, u32 guest_addr, u32 gpr_bitmask, u32 fpr_bitmask, u8 address_register, u8 data_register, u8 size_in_bits, bool is_signed, bool is_load, bool is_fpr)
{
  pxFailRel("Not implemented.");
}

bool SaveStateBase::vuJITFreeze()
{
	if(IsSaving())
		vu1Thread.WaitVU();

	Console.Warning("recompiler state is stubbed in arm64!");

	// HACK!!

	// size of microRegInfo structure
	std::array<u8,96> empty_data{};
	Freeze(empty_data);
	Freeze(empty_data);
	return true;
}

// microVU test harness — uses VU interpreter on ARM64.

#include "VUmicro.h"
#include "VU.h"

void mVU0_TestInit()
{
	CpuIntVU0.Reset();
}

void mVU0_TestShutdown() {}

void mVU0_TestWriteProg(const u32* words, u32 count)
{
	std::memcpy(VU0.Micro, words, count * sizeof(u32));
}

void mVU0_TestExec(u32 startPC, u32 cycles)
{
	// Reset pipeline state so each test starts clean
	VU0.fmacwritepos = 0;
	VU0.fmacreadpos = 0;
	VU0.fmaccount = 0;
	VU0.ialuwritepos = 0;
	VU0.ialureadpos = 0;
	VU0.ialucount = 0;
	VU0.cycle = 0;
	VU0.flags = 0;
	VU0.branch = 0;
	VU0.ebit = 0;
	VU0.clipflag = 0;
	VU0.statusflag = 0;
	VU0.macflag = 0;

	// TPC in instruction-index units (>>3 of byte offset)
	VU0.VI[REG_TPC].UL = startPC >> 3;
	// Mark VU0 as running
	VU0.VI[REG_VPU_STAT].UL |= 0x1;

	CpuIntVU0.Execute(cycles);
}

void mVU1_TestInit()
{
	CpuIntVU1.Reset();
}

void mVU1_TestShutdown() {}

void mVU1_TestWriteProg(const u32* words, u32 count)
{
	std::memcpy(VU1.Micro, words, count * sizeof(u32));
}

void mVU1_TestExec(u32 startPC, u32 cycles)
{
	VU1.fmacwritepos = 0;
	VU1.fmacreadpos = 0;
	VU1.fmaccount = 0;
	VU1.ialuwritepos = 0;
	VU1.ialureadpos = 0;
	VU1.ialucount = 0;
	VU1.cycle = 0;
	VU1.flags = 0;
	VU1.branch = 0;
	VU1.ebit = 0;

	VU1.VI[REG_TPC].UL = startPC >> 3;
	// Mark VU1 as running (bit 8 of VU0's VPU_STAT)
	VU0.VI[REG_VPU_STAT].UL |= 0x100;

	CpuIntVU1.Execute(cycles);
}
