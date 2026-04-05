// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Class declaration.
// Initially wraps the VU1 interpreter; individual instructions are gradually
// replaced with native ARM64 codegen via per-instruction ISTUB toggles.

#pragma once

#include "VUmicro.h"

// ============================================================================
//  recArmVU1 — ARM64 VU1 recompiler
// ============================================================================

class recArmVU1 final : public BaseVUmicroCPU
{
public:
	recArmVU1();
	~recArmVU1() override { Shutdown(); }

	const char* GetShortName() const override { return "armVU1"; }
	const char* GetLongName() const override { return "ARM64 VU1 Recompiler"; }

	void Reserve();
	void Shutdown() override;
	void Reset() override;
	void SetStartPC(u32 startPC) override;
	void Step() override;
	void Execute(u32 cycles) override;
	void Clear(u32 addr, u32 size) override;
	void ResumeXGkick() override {}
};

extern recArmVU1 CpuArmVU1;
