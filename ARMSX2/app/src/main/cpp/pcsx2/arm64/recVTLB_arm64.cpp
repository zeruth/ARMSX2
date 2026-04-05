// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VTLB fastmem backpatch handler.
// When a fastmem LDR/STR faults (unmapped page, MMIO, etc.), the signal
// handler calls vtlb_BackpatchLoadStore → vtlb_DynBackpatchLoadStore.
// We generate a "thunk" that performs the slow vtlb_memRead/Write call
// and jumps back to the instruction after the original fastmem access.
// The faulting instruction is then overwritten with a branch to the thunk.

#include "Common.h"
#include "vtlb.h"
#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"

#include "common/Console.h"

using namespace R5900;

void vtlb_DynBackpatchLoadStore(uptr code_address, u32 code_size, u32 guest_pc,
	u32 guest_addr, u32 gpr_bitmask, u32 fpr_bitmask, u8 address_register,
	u8 data_register, u8 size_in_bits, bool is_signed, bool is_load, bool is_fpr)
{
	u8* thunk = recBeginThunk();

	if (is_load)
	{
		// Ensure address is in w0 (RWARG1) for vtlb_memRead<T>
		if (address_register != RWARG1.GetCode())
			armAsm->Mov(RWARG1, a64::WRegister(address_register));

		switch (size_in_bits)
		{
			case 8:  armEmitCall((const void*)&vtlb_memRead<mem8_t>);  break;
			case 16: armEmitCall((const void*)&vtlb_memRead<mem16_t>); break;
			case 32: armEmitCall((const void*)&vtlb_memRead<mem32_t>); break;
			case 64: armEmitCall((const void*)&vtlb_memRead<mem64_t>); break;
			default: pxFailRel("Unexpected fastmem load size"); break;
		}

		// vtlb_memRead returns unsigned in w0/x0. Sign-extend if needed.
		if (is_signed)
		{
			switch (size_in_bits)
			{
				case 8:  armAsm->Sxtb(RXRET, RWRET); break;
				case 16: armAsm->Sxth(RXRET, RWRET); break;
				case 32: armAsm->Sxtw(RXRET, RWRET); break;
				default: break; // 64-bit: no sign extension needed
			}
		}

		// Move result to the data register if it's not already x0
		if (data_register != RXRET.GetCode())
			armAsm->Mov(a64::XRegister(data_register), RXRET);
	}
	else
	{
		// Store: set up (w0 = address, w1/x1 = value) for vtlb_memWrite<T>
		if (address_register == RXARG2.GetCode() && data_register == RXARG1.GetCode())
		{
			// Registers are swapped — use scratch to resolve
			armAsm->Mov(RXVIXLSCRATCH, a64::XRegister(address_register));
			armAsm->Mov(RXARG2, a64::XRegister(data_register));
			armAsm->Mov(RWARG1, a64::WRegister(RXVIXLSCRATCH.GetCode()));
		}
		else
		{
			if (data_register != RXARG2.GetCode())
				armAsm->Mov(RXARG2, a64::XRegister(data_register));
			if (address_register != RWARG1.GetCode())
				armAsm->Mov(RWARG1, a64::WRegister(address_register));
		}

		switch (size_in_bits)
		{
			case 8:  armEmitCall((const void*)&vtlb_memWrite<mem8_t>);  break;
			case 16: armEmitCall((const void*)&vtlb_memWrite<mem16_t>); break;
			case 32: armEmitCall((const void*)&vtlb_memWrite<mem32_t>); break;
			case 64: armEmitCall((const void*)&vtlb_memWrite<mem64_t>); break;
			default: pxFailRel("Unexpected fastmem store size"); break;
		}
	}

	// Jump back to the instruction after the original fastmem access
	armEmitJmp(reinterpret_cast<const void*>(code_address + code_size));

	recEndThunk();

	// Replace the faulting fastmem instruction with a branch to the thunk
	armEmitJmpPtr(reinterpret_cast<void*>(code_address), thunk, true);
}
