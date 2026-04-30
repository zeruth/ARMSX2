// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Main driver.
// Adapted from the VU1 ARM64 recompiler. VU0 has 4KB micro memory,
// no XGKICK, and different VPU_STAT bits.

#include "Common.h"
#include "GS.h"
#include "Memory.h"
#include "Vif.h"
#include "VUmicro.h"
#include "VUops.h"
#include "arm64/AsmHelpers.h"
#include "arm64/arm64Emitter.h"
#include "arm64/iVU0micro_arm64.h"
#include "common/Perf.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <vector>

using namespace vixl::aarch64;

// Global instance
recArmVU0 CpuArmVU0;

// VU0 per-cycle interpreter entry point
extern void vu0Exec(VURegs* VU);

// Flush helpers
extern void _vuFlushAll(VURegs* VU);

// ============================================================================
//  Rec emitter dispatch tables — reuse VU1's tables since the instruction
//  set is identical. The tables emit code that operates through a pinned
//  base register (x23), which we point at &VU0 instead of &VU1.
//  The only caveat: interpreter-stub entries call VU1_UPPER/LOWER_OPCODE.
//  For VU0 we need VU0_UPPER/LOWER_OPCODE. We define VU0-specific tables.
// ============================================================================

using VU0RecFn = void (*)();
extern VU0RecFn recVU0_UpperTable[64];
extern VU0RecFn recVU0_LowerTable[128];

// The generic "unknown lower opcode" emitter, exposed from iVU0Lower_arm64.cpp.
// Used for pointer-compare bad-op detection at block-analysis time.
extern void recVU0_Lower_Unknown();

// ============================================================================
//  Block cache
// ============================================================================

static constexpr u32 VU0_NUM_SLOTS       = VU0_PROGSIZE / 8;
static constexpr u32 VU0_MAX_BLOCK_PAIRS = 128;

static constexpr u32 POOL_SIZE = 32 * 1024;

// Block-linking patch slot. Mirrors VU1's LinkExit. Each compiled block has
// up to 2 of these (single-exit fall-through / unconditional branch, or
// 2-way conditional with split tails). `patch_site` is the address of an
// unconditional B that gets rewritten between fallthrough (unlinked) and
// successor's linkEntry (linked).
struct LinkExit
{
	u32 target_pc;
	u8* patch_site;
	u8* fallthrough;
	u8* current_target;
};
static constexpr u32 LINK_TARGET_NONE = ~0u;

struct VU0BlockEntry
{
	u8*  codeEntry;   // prologue entry — for first-call dispatch from Execute
	u8*  linkEntry;   // post-prologue entry; predecessors B here directly
	u8*  returnExit;  // post-pair-loop label where the exit-selector lives

	// Static link-exit set:
	//   0 : not linkable (ebit, JR/JALR via runtime dispatch, MFLAGSET-only).
	//   1 : single exit (B/BAL, or max-size fall-through).
	//   2 : conditional IBxx — exits[0]=NOT-TAKEN, exits[1]=TAKEN.
	u32      num_exits;
	LinkExit exits[2];

	u32  numPairs;

	// Content-keyed cache key. Snapshot of the VU0.Micro bytes this variant
	// was compiled against (numPairs * 8 bytes). findVariantVU0 picks the
	// variant whose snapshot matches the live micro at dispatch time —
	// surviving across Clear() so re-uploads of the same program get a hit
	// instead of a fresh compile. Heap-owned; freed in destroyVariantVU0.
	u8*  snapshot;

	// Set by Clear() when this variant's slot is in a cleared range. The
	// next dispatch via findVariantVU0 re-runs tryForwardLinkVU0 +
	// patchWaitingPredecessorsVU0 to re-wire incoming/outgoing edges
	// (Clear already unpatched incoming edges for correctness).
	bool needsRelink;
};

// Per-slot cap on the variant deque. Mirrors VU1's kVariantCapPerSlot. 8
// covers any sane number of distinct programs at one PC; the LRU is
// evicted via destroyVariantVU0 when the cap is hit.
static constexpr u32 kVU0VariantCapPerSlot = 8;

// Content-keyed program cache — one deque of compiled variants per slot.
// Most slots carry 0 or 1 variant in steady state; a slot grows a deque
// only when the EE uploads different bytecode programs to the same PC.
// Front of the deque is the MRU variant — findVariantVU0 bubbles hits forward.
static std::deque<VU0BlockEntry*> s_vu0_variants[VU0_NUM_SLOTS];

// Reverse index: for each VU0 slot S, the list of variants that have at
// least one exit targeting S. patchWaitingPredecessorsVU0 walks this to
// find waiters after a fresh compile; Clear() walks it to find predecessors
// that need their incoming exits unpatched.
//
// A variant is added once per UNIQUE target_pc among its exits[]. Variants
// stay in the index for their lifetime; cleaned up by destroyVariantVU0
// (per-eviction) or deleteAllVariantsVU0 (full reset).
static std::vector<VU0BlockEntry*> s_vu0_waitingForSlot[VU0_NUM_SLOTS];

static u8* s_code_base  = nullptr;
static u8* s_code_write = nullptr;
static u8* s_code_end   = nullptr;
static ArmConstantPool s_pool;

// Cycle limit for the current recArmVU0::Execute call. Set to
// `startcycles + cycles` at the top of Execute; read at every linkEntry by
// the inlined cycle-budget check so linked blocks yield back to the outer
// dispatch loop when the budget is exhausted. Without this, tight VU0
// loops that conditionally-link back to themselves (IBxx loop:) — or any
// chain of blocks with no ebit/MFLAGSET inside — never return to Execute
// and hang the emulator. VU0 is single-thread (always EE), so no atomicity
// is required. Mirrors s_vu1_cycle_limit in iVU1micro_arm64.cpp.
static u64 s_vu0_cycle_limit = 0;

// ============================================================================
//  Branch opcode classifiers (VU instruction encoding identical to VU1).
// ============================================================================

// B / BAL — single compile-time-known target, no runtime condition.
static inline bool isUnconditionalBranchOpVU0(u32 lower)
{
	const u32 top7 = lower >> 25;
	return top7 == 0x20u || top7 == 0x21u;
}

// IBEQ / IBNE / IBLTZ / IBGTZ / IBLEZ / IBGEZ — both targets known,
// runtime condition picks.
static inline bool isConditionalBranchOpVU0(u32 lower)
{
	const u32 top7 = lower >> 25;
	return top7 == 0x28u || top7 == 0x29u
	    || (top7 >= 0x2Cu && top7 <= 0x2Fu);
}

// JR / JALR — runtime VI register holds target.
static inline bool isIndirectBranchOpVU0(u32 lower)
{
	const u32 top7 = lower >> 25;
	return top7 == 0x24u || top7 == 0x25u;
}

// ============================================================================
//  Runtime helpers
// ============================================================================

// D/T bit firing on VU0. Fires the INTC IRQ immediately via hwIntcIrq
// rather than x86 microVU's VUFLAG_INTCINTERRUPT deferred-flag pattern
// (microVU_Compile.inl:570). Safe because hwIntcIrq only mutates state
// (INTC_STAT |= 1<<n, then cpuTestINTCInts which at most schedules an EE
// event via cpuSetNextEventDelta(4)). No preemption, no dispatch. The
// timing divergence vs x86 is sub-block (~1 pair) since ebit=1 terminates
// the block immediately after this via step 13's countdown anyway.
static void vu0CheckDTBits(u32 upper)
{
	if (upper & 0x10000000) // D flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x4)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x2;
			hwIntcIrq(INTC_VU0);
			VU0.ebit = 1;
		}
	}
	if (upper & 0x08000000) // T flag
	{
		if (VU0.VI[REG_FBRST].UL & 0x8)
		{
			VU0.VI[REG_VPU_STAT].UL |= 0x4;
			hwIntcIrq(INTC_VU0);
			VU0.ebit = 1;
		}
	}
}

static void vu0EbitDone(VURegs* VU)
{
	VU->VIBackupCycles = 0;
	_vuFlushAll(VU);
	VU0.VI[REG_VPU_STAT].UL &= ~0x1;
	vif0Regs.stat.VEW = false;
}

static void vu0HandleDelayBranch(VURegs* VU)
{
	if (VU->takedelaybranch)
	{
		VU->branch          = 1;
		VU->branchpc        = VU->delaybranchpc;
		VU->takedelaybranch = false;
	}
}

// Predicate: does this lower instruction word decode to a NOP on VU0?
// Matches x86 microVU's isNOP=true on isVU0 for EFU ops, XGKICK, XTOP, and
// the pipeline-wait ops WAITQ/WAITP. Used to elide the per-pair VU0.code
// store and rec-table dispatch scaffolding for these ops — the rec emitters
// themselves already NOP, so the Mov+Str + table indirection was pure waste.
//
// Lower dispatch flow: lower >> 25 == 0x40 → LowerOP sub-table;
//   (lower & 0x3F) in 0x3C..0x3F → T3_xx sub-table;
//   (lower >> 6) & 0x1F → T3_xx entry.
static bool isVU0LowerNOP(u32 lower)
{
	if ((lower >> 25) != 0x40) return false;
	const u32 lower_op = lower & 0x3f;
	if (lower_op < 0x3C) return false;

	const u32 t3_sub = (lower >> 6) & 0x1f;
	switch (lower_op)
	{
		case 0x3C: // T3_00 — XTOP, XGKICK, ESADD, EATANxy, ESQRT, ESIN (all NOP on VU0)
			return (t3_sub >= 0x1A) && (t3_sub <= 0x1F);
		case 0x3D: // T3_01 — ERSADD, EATANxz, ERSQRT, EATAN (NOP). XITOP (0x1A) does real work.
			return (t3_sub >= 0x1C) && (t3_sub <= 0x1F);
		case 0x3E: // T3_10 — ELENG, ESUM, ERCPR, EEXP (all NOP on VU0)
			return (t3_sub >= 0x1C) && (t3_sub <= 0x1F);
		case 0x3F: // T3_11 — WAITQ(0x0E), ERLENG(0x1C), WAITP(0x1E). 0x1D/0x1F are Unknown (skip).
			return (t3_sub == 0x0E) || (t3_sub == 0x1C) || (t3_sub == 0x1E);
	}
	return false;
}

static void vu0DecrementVIBackup(VURegs* VU, u64 cyclesBefore)
{
	if (VU->VIBackupCycles > 0)
	{
		u32 elapsed = static_cast<u32>(VU->cycle - cyclesBefore);
		if (elapsed >= VU->VIBackupCycles)
			VU->VIBackupCycles = 0;
		else
			VU->VIBackupCycles -= static_cast<u8>(elapsed);
	}
}

// ============================================================================
//  Block analysis
// ============================================================================

static bool PairHasEbit(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
	return (upper >> 30) & 1;
}

static bool PairHasBranch(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
	if ((upper >> 31) & 1)
		return false;
	const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + pc);
	_VURegsNum lregs{};
	// Sub-table dispatchers (e.g. LowerOP) read VU0.code internally, so the
	// global must be primed — otherwise the LowerOP index selects from stale
	// state and can hit the Unknown slot (which pxFail-aborts in debug).
	VU0.code = lower;
	VU0regs_LOWER_OPCODE[lower >> 25](&lregs);
	return lregs.pipe == VUPIPE_BRANCH;
}

// Detect pairs containing an illegal/reserved lower opcode so we can truncate
// the block at them. Mirrors x86 microVU's mVUcheckBadOp (microVU_Compile.inl)
// which sets mVUinfo.isEOB when dispatch routes to Unknown. We do pointer
// comparison against recVU0_Lower_Unknown on the top-level dispatch only —
// sub-table Unknown ops still fall through to the interp at runtime and are
// harmless (just log + return), they just don't get the tighter block bound.
// I-bit pairs are exempted: their lower word is the I-register literal, not
// an opcode.
static bool PairHasBadOp(u32 pc)
{
	const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
	if ((upper >> 31) & 1)
		return false;
	// BIOS writes a reversed-NOP pair (0x8000033c for the upper half); x86
	// microVU explicitly excludes this to avoid console spam. Honor the same
	// allowlist.
	if (upper == 0x8000033c)
		return false;
	const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + pc);
	return recVU0_LowerTable[lower >> 25] == recVU0_Lower_Unknown;
}

static u32 AnalyzeBlock(u32 startPC)
{
	u32 pairs = 0;
	u32 pc    = startPC;

	while (pairs < VU0_MAX_BLOCK_PAIRS)
	{
		const bool ebit   = PairHasEbit(pc);
		const bool branch = PairHasBranch(pc);
		const bool bad_op = PairHasBadOp(pc);

		pairs++;
		pc = (pc + 8) & (VU0_PROGSIZE - 1);

		if (ebit || branch)
		{
			pairs++;
			break;
		}
		// Bad op: include the current pair (still dispatches to interp) but
		// no delay slot, and truncate the block. Matches x86 microVU's
		// mVUinfo.isEOB on bad op. Benefit is earlier block boundary so we
		// re-enter dispatch and don't keep compiling past a definitely-bad
		// opcode.
		if (bad_op)
			break;
	}

	return pairs;
}

// ============================================================================
//  Block linking — patch helpers / link-exit analysis
// ============================================================================

struct BlockLinkExits
{
	u32  num_exits;
	u32  target_pcs[2];
	bool indirect;
};

static BlockLinkExits computeBlockLinkExits(u32 startPC, u32 numPairs)
{
	BlockLinkExits out = {};
	out.target_pcs[0] = LINK_TARGET_NONE;
	out.target_pcs[1] = LINK_TARGET_NONE;

	if (numPairs == 0)
		return out;

	// Max-size block: single fall-through.
	if (numPairs >= VU0_MAX_BLOCK_PAIRS)
	{
		out.num_exits     = 1;
		out.target_pcs[0] = (startPC + numPairs * 8u) & VU0_PROGMASK;
		return out;
	}

	// Branch- or ebit-terminated. Terminator at pair numPairs-2 (pair before
	// the delay slot).
	const u32 term_pc = (startPC + (numPairs - 2u) * 8u) & VU0_PROGMASK;
	const u32 upper   = *reinterpret_cast<const u32*>(VU0.Micro + term_pc + 4);

	if ((upper >> 30) & 1)
		return out; // ebit — no successor

	if ((upper >> 31) & 1)
		return out; // I-bit upper — analyze wouldn't have terminated here on a branch

	const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + term_pc);

	const s32 imm11 = (lower & 0x400u)
		? static_cast<s32>(0xFFFFFC00u | (lower & 0x3FFu))
		: static_cast<s32>(lower & 0x3FFu);
	const u32 tpc_val    = (term_pc + 8u) & VU0_PROGMASK;
	const u32 imm_target = (tpc_val + static_cast<u32>(imm11 * 8)) & VU0_PROGMASK;
	const u32 fallthrough_pc = (startPC + numPairs * 8u) & VU0_PROGMASK;

	if (isUnconditionalBranchOpVU0(lower))
	{
		out.num_exits     = 1;
		out.target_pcs[0] = imm_target;
		return out;
	}

	if (isConditionalBranchOpVU0(lower))
	{
		out.num_exits     = 2;
		out.target_pcs[0] = fallthrough_pc;  // not taken
		out.target_pcs[1] = imm_target;      // taken
		return out;
	}

	if (isIndirectBranchOpVU0(lower))
	{
		out.indirect = true;
		return out;
	}

	return out;
}

// Patch a single LinkExit's B-imm26 to jump to `target`. No-op when the
// site already points there. Single-thread: always called on the EE
// thread (recompile / Clear / dispatch all run there for VU0).
static void patchVU0LinkSite(LinkExit& exit, u8* target)
{
	if (!exit.patch_site)
		return;
	if (exit.current_target == target)
		return;
	armEmitJmpPtr(exit.patch_site, target, true);
	exit.current_target = target;
}

static void unpatchVU0LinkSite(LinkExit& exit)
{
	if (!exit.patch_site)
		return;
	patchVU0LinkSite(exit, exit.fallthrough);
}

// Content-keyed variant lookup. Scans the slot's deque for a variant whose
// snapshot matches the live VU0.Micro bytes at `pc`; MRU-bubbles a hit to
// the front. Miss → nullptr.
//
// Fast-reject on the first 8-byte compare before the full memcmp — most
// mismatches fail there. VU0.Micro is 16-byte aligned and slot PCs are
// 8-byte aligned, so the u64 load is well-defined.
static VU0BlockEntry* findVariantVU0(u32 pc)
{
	const u32 slot = pc / 8;
	if (slot >= VU0_NUM_SLOTS)
		return nullptr;
	auto& deque = s_vu0_variants[slot];
	if (deque.empty())
		return nullptr;

	const u8* live = VU0.Micro + pc;
	const u64 live_head = *reinterpret_cast<const u64*>(live);

	for (auto it = deque.begin(); it != deque.end(); ++it)
	{
		VU0BlockEntry* blk = *it;
		const u64 snap_head = *reinterpret_cast<const u64*>(blk->snapshot);
		if (snap_head != live_head)
			continue;
		const u32 snap_bytes = blk->numPairs * 8;
		if (snap_bytes > 8
			&& std::memcmp(blk->snapshot + 8, live + 8, snap_bytes - 8) != 0)
			continue;

		if (it != deque.begin())
		{
			deque.erase(it);
			deque.push_front(blk);
		}
		return blk;
	}
	return nullptr;
}

// Add this block to the reverse index for each unique exit target. Caller
// has already pushed the entry onto its slot's deque.
static void indexVU0VariantExits(VU0BlockEntry* blk)
{
	for (u32 e = 0; e < blk->num_exits; e++)
	{
		const u32 target_pc = blk->exits[e].target_pc;
		if (target_pc == LINK_TARGET_NONE)
			continue;
		bool dup = false;
		for (u32 j = 0; j < e; j++)
		{
			if (blk->exits[j].target_pc == target_pc)
			{
				dup = true;
				break;
			}
		}
		if (dup)
			continue;
		const u32 target_slot = target_pc / 8;
		if (target_slot < VU0_NUM_SLOTS)
			s_vu0_waitingForSlot[target_slot].push_back(blk);
	}
}

// Right after a block compiles: for each live static exit, look up the
// target slot's variant cache for a snapshot-matching entry; if found and
// it has a linkEntry, wire our exit to it.
static void tryForwardLinkVU0(VU0BlockEntry& block)
{
	for (u32 e = 0; e < block.num_exits; e++)
	{
		LinkExit& exit = block.exits[e];
		if (exit.target_pc == LINK_TARGET_NONE)
			continue;
		VU0BlockEntry* target = findVariantVU0(exit.target_pc);
		if (target && target->linkEntry)
			patchVU0LinkSite(exit, target->linkEntry);
	}
}

// Walk the reverse index for `my_slot` and patch any predecessor exit whose
// target is `my_pc` to jump directly into `my_linkEntry`. Predecessors are
// stored as VU0BlockEntry* (multiple variants can target the same slot,
// and a slot can hold multiple variants too).
static void patchWaitingPredecessorsVU0(u32 my_pc, u8* my_linkEntry)
{
	if (!my_linkEntry)
		return;
	const u32 my_slot = my_pc / 8;
	if (my_slot >= VU0_NUM_SLOTS)
		return;
	for (VU0BlockEntry* pred : s_vu0_waitingForSlot[my_slot])
	{
		for (u32 e = 0; e < pred->num_exits; e++)
		{
			LinkExit& exit = pred->exits[e];
			if (exit.patch_site != nullptr
			    && exit.target_pc == my_pc
			    && exit.current_target != my_linkEntry)
			{
				patchVU0LinkSite(exit, my_linkEntry);
			}
		}
	}
}

// Detach and free a single variant. Caller has already removed it from
// s_vu0_variants[my_slot]; this routine drops it from the reverse index
// and frees the snapshot + entry. Compiled code in the JIT buffer is left
// in place — reclaimed at the next deleteAllVariantsVU0 (buffer-full
// reset / Reset / Shutdown).
//
// Predecessors that had patches pointing at this variant's linkEntry get
// reverted to fall-through so they don't jump into code that may be
// overwritten by a future compile occupying the same buffer space.
static void destroyVariantVU0(VU0BlockEntry* blk, u32 my_slot)
{
	if (blk->linkEntry && my_slot < VU0_NUM_SLOTS)
	{
		for (VU0BlockEntry* pred : s_vu0_waitingForSlot[my_slot])
		{
			if (pred == blk)
				continue;
			for (u32 e = 0; e < pred->num_exits; e++)
			{
				LinkExit& exit = pred->exits[e];
				if (exit.current_target == blk->linkEntry)
					unpatchVU0LinkSite(exit);
			}
		}
	}

	for (u32 e = 0; e < blk->num_exits; e++)
	{
		const u32 target_pc = blk->exits[e].target_pc;
		if (target_pc == LINK_TARGET_NONE)
			continue;
		bool dup = false;
		for (u32 j = 0; j < e; j++)
		{
			if (blk->exits[j].target_pc == target_pc)
			{
				dup = true;
				break;
			}
		}
		if (dup)
			continue;
		const u32 target_slot = target_pc / 8;
		if (target_slot >= VU0_NUM_SLOTS)
			continue;
		auto& vec = s_vu0_waitingForSlot[target_slot];
		vec.erase(std::remove(vec.begin(), vec.end(), blk), vec.end());
	}

	delete[] blk->snapshot;
	delete blk;
}

// Wipe every cached variant in every slot. Called on buffer-full reset,
// Shutdown, and Reset. The compiled code buffer itself is reclaimed by
// the caller — this only frees the per-variant heap storage.
static void deleteAllVariantsVU0()
{
	for (u32 i = 0; i < VU0_NUM_SLOTS; i++)
	{
		for (VU0BlockEntry* blk : s_vu0_variants[i])
		{
			delete[] blk->snapshot;
			delete blk;
		}
		s_vu0_variants[i].clear();
		s_vu0_waitingForSlot[i].clear();
	}
}

// JR/JALR runtime dispatcher. Given the runtime TPC, look up the target
// slot's variant cache (content-match against live micro). On hit, the
// JIT-emitted indirect tail tail-Brs to the returned linkEntry. On miss
// (no compiled variant matches live micro), fall through to flushes+Ret;
// Execute's loop will dispatch + compile, and the next visit hits.
static u8* vu0_indirect_dispatch(u32 tpc)
{
	const u32 pc = tpc & VU0_PROGMASK;
	VU0BlockEntry* blk = findVariantVU0(pc);
	return blk ? blk->linkEntry : nullptr;
}

// ============================================================================
//  Block compilation
// ============================================================================

static const auto VU0_BASE_REG = x23;

// VU0_PROFILE_OPS scaffolding (toggle in arm64/InterpFlags.h). Same shape
// as the VU1 macros — register the emit cursor span around per-pair sections.
#ifdef VU0_PROFILE_OPS
	#define VU0_PERF_BEGIN(varname) const u8* varname = armGetCurrentCodePointer()
	#define VU0_PERF_END(varname, fmt, ...) do { \
		const u8* _vu0_pe_end = armGetCurrentCodePointer(); \
		if (_vu0_pe_end > (varname)) { \
			char _vu0_pe_name[64]; \
			std::snprintf(_vu0_pe_name, sizeof(_vu0_pe_name), fmt, ##__VA_ARGS__); \
			Perf::vu0.Register((varname), \
				static_cast<size_t>(_vu0_pe_end - (varname)), _vu0_pe_name); \
		} \
	} while (0)
#else
	#define VU0_PERF_BEGIN(varname) ((void)0)
	#define VU0_PERF_END(varname, fmt, ...) ((void)0)
#endif

static u8* CompileBlock(u32 startPC, u32 numPairs, VU0BlockEntry* out_block)
{
	const size_t data_size    = numPairs * 2 * sizeof(_VURegsNum);
	const size_t code_worst   = static_cast<size_t>(numPairs) * 512 + 64;
	const size_t total_needed = data_size + code_worst;

	if (static_cast<size_t>(s_code_end - s_code_write) < total_needed)
	{
		// Wipe every cached variant. Any predecessor's patched B would now
		// point at the recycled buffer region; nuking the cache forces a
		// fresh compile (which calls patchWaitingPredecessorsVU0 to re-link).
		// The incoming out_block is fresh (caller hasn't pushed it onto a
		// deque yet), so it survives.
		deleteAllVariantsVU0();
		s_code_write = s_code_base;
		s_pool.Reset();
	}

	u8* const data_base = s_code_write;
	_VURegsNum* const uregs_data = reinterpret_cast<_VURegsNum*>(data_base);
	_VURegsNum* const lregs_data = uregs_data + numPairs;

	std::memset(data_base, 0, data_size);

	{
		u32 pc = startPC;
		for (u32 i = 0; i < numPairs; i++)
		{
			const u32 upper = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
			const u32 lower = *reinterpret_cast<const u32*>(VU0.Micro + pc);

			VU0.code = upper;
			VU0regs_UPPER_OPCODE[upper & 0x3f](&uregs_data[i]);

			if (!((upper >> 31) & 1))
			{
				VU0.code = lower;
				VU0regs_LOWER_OPCODE[lower >> 25](&lregs_data[i]);
			}

			pc = (pc + 8) & (VU0_PROGSIZE - 1);
		}
	}

	u8* code_start = data_base + data_size;
	code_start = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(code_start) + 3) & ~3ULL);

	// Static exit set for this block — drives the exit-selector emit below.
	const BlockLinkExits link_info = computeBlockLinkExits(startPC, numPairs);

	armSetAsmPtr(code_start, static_cast<size_t>(s_code_end - code_start), &s_pool);
	u8* const entry = armStartBlock();

	armAsm->Stp(x29, x30, MemOperand(sp, -32, PreIndex));
	armAsm->Stp(x22, VU0_BASE_REG, MemOperand(sp, 16));
	armAsm->Mov(x29, sp);
	armMoveAddressToReg(VU0_BASE_REG, &VU0);

	const int64_t cycle_off    = (int64_t)offsetof(VURegs, cycle);
	const int64_t code_off     = (int64_t)offsetof(VURegs, code);
	const int64_t branch_off   = (int64_t)offsetof(VURegs, branch);
	const int64_t branchpc_off = (int64_t)offsetof(VURegs, branchpc);
	const int64_t ebit_off     = (int64_t)offsetof(VURegs, ebit);
	const int64_t tpc_off      = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_TPC * (int64_t)sizeof(REG_VI));
	const int64_t regi_off     = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_I   * (int64_t)sizeof(REG_VI));
	const int64_t fmacwpos_off = (int64_t)offsetof(VURegs, fmacwritepos);
	const int64_t flags_off    = (int64_t)offsetof(VURegs, flags);
	const int64_t vpu_stat_off = (int64_t)((int64_t)offsetof(VURegs, VI) + REG_VPU_STAT * (int64_t)sizeof(REG_VI));
	const int64_t micro_off    = (int64_t)offsetof(VURegs, Micro);

	// Epilogue label — jumped to when we need early exit (linkEntry gate
	// or per-pair VPU_STAT/MFLAGSET check). Declared before linkEntry so
	// the gate emitted right after linkEntry can branch to it.
	Label early_exit;

	// Block-linking entry point. Linked predecessors B here directly,
	// skipping the prologue (their callee-saves are already saved). The
	// fall-through dispatch (codeEntry → linkEntry) also lands here.
	out_block->linkEntry = armGetCurrentCodePointer();

	// Entry-gate. Every block entry — fall-through from prologue OR direct
	// `B` from a linked predecessor's returnExit — runs through this gate
	// before the per-pair body. Without it, a tail-chain of VU0 blocks
	// (canonical case: IBxx loop: with no ebit, or any closed-cycle link
	// graph) bypasses Execute's outer while loop entirely and never yields
	// — silent hard hang, no log. The per-pair check between pairs i and
	// i+1 inside a block doesn't help: it doesn't fire after the LAST pair
	// of a block, so a block that drops MFLAGSET / clears VPU_STAT on its
	// last pair (e.g. M-bit fallback to vu0Exec) tail-chains anyway.
	//
	// Three failure modes branch to early_exit (epilogue → Ret to Execute):
	//   1. Cycle budget exhausted (cycle >= s_vu0_cycle_limit).
	//   2. VU0 stopped externally (VPU_STAT bit 0 == 0 — set by FBRST reset
	//      or by _vuFlushAll inside vu0Exec when ebit countdown hits 0).
	//   3. M-bit pending (flags bit VUFLAG_MFLAGSET — set inline by the JIT
	//      M-bit path or by vu0Exec on M-bit fallback).
	//
	// VUSyncHack / FullVU0SyncHack honoring: when either is set, fire the
	// gate if the upcoming block WOULD overshoot the limit (current +
	// numPairs >= limit) instead of only when we already have. Mirrors
	// x86 microVU_Branch.inl:116/243/252 and our VU1 gate (iVU1micro
	// linkEntry block at line ~3735). numPairs <= VU0_MAX_BLOCK_PAIRS is
	// well within Add's 12-bit immediate range.
	{
		const bool tight_sync = EmuConfig.Gamefixes.VUSyncHack
		                     || EmuConfig.Gamefixes.FullVU0SyncHack;

		// 1. Cycle budget.
		armMoveAddressToReg(x5, &s_vu0_cycle_limit);
		armAsm->Ldr(x5, MemOperand(x5));
		armAsm->Ldr(x4, MemOperand(VU0_BASE_REG, cycle_off));
		if (tight_sync)
		{
			armAsm->Add(x4, x4, numPairs);
		}
		armAsm->Cmp(x4, x5);
		armAsm->B(&early_exit, hs);

		// 2. VU0 stopped (VPU_STAT bit 0 == 0).
		armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, vpu_stat_off));
		armAsm->Tbz(w4, 0, &early_exit);

		// 3. M-bit pending (VUFLAG_MFLAGSET = 1 << 1).
		armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, flags_off));
		armAsm->Tbnz(w4, 1, &early_exit);
	}

	// IbitHack forces per-op immediate decode from live micro memory (mirrors
	// x86 microVU's ptr32[&curI] reads). When on, VU->code is loaded from
	// VU->Micro[pc] at runtime instead of the JIT-baked instruction word so
	// subsequent C wrappers (LQ/SQ/ILW/ISW) pick up any post-compile patches.
	// Natively-emitted IADDI/IADDIU/ISUBIU consult EmuConfig directly and emit
	// runtime-decode paths of their own.
	const bool use_ibit_hack = EmuConfig.Gamefixes.IbitHack;

	// Tracks whether the previous pair executed a branch op — feeds the
	// "is this pair a branch delay slot?" predicate for D/T bit suppression.
	// Mirrors mVUinfo.isBdelay in x86 microVU_Compile.inl:901.
	bool prev_was_branch = false;

	// Tracks whether the previous pair had its E-bit set — feeds the
	// "branch in E-bit delay slot?" predicate for branch suppression in
	// the lower emit path. Mirrors x86 microVU_Compile.inl branchWarning
	// which sets mVUlow.isNOP when mVUup.eBit && mVUbranch.
	bool prev_was_ebit = false;

	u32 pc = startPC;
	for (u32 i = 0; i < numPairs; i++)
	{
		const u32 upper     = *reinterpret_cast<const u32*>(VU0.Micro + pc + 4);
		const u32 lower     = *reinterpret_cast<const u32*>(VU0.Micro + pc);
		const bool ibit     = (upper >> 31) & 1;
		const bool ebit_set = (upper >> 30) & 1;
		const bool dbit_set = (upper >> 28) & 1;
		const bool tbit_set = (upper >> 27) & 1;
		const _VURegsNum& uregs = uregs_data[i];
		const _VURegsNum& lregs = lregs_data[i];

		// Hazard detection — match every save/restore/discard case in
		// _vu0Exec (VU0microInterp.cpp:104-158). Anything that needs the
		// "lower sees pre-upper state" or "lower discarded" semantics must
		// fall back to vu0Exec because the native machinery does neither.
		//
		//   VF: upper writes vfX, lower also writes vfX        -> discard lower
		//   VF: upper writes vfX, lower reads  vfX             -> save/restore VF
		//   CLIP: upper writes CLIP, lower writes CLIP         -> discard lower
		//   CLIP: upper writes CLIP, lower reads  CLIP         -> save/restore CLIP
		//
		// Without the discard cases, the JIT runs upper then lower
		// sequentially: when both write the same VF, lower's value wins,
		// silently clobbering the upper FMAC result. This manifests as
		// vertex/shadow corruption in titles like SA where transform
		// pipelines pair `MUL vfX, ...` (upper) with `LQI vfX, ...` (lower).
		const bool vf_hazard = !ibit && uregs.VFwrite != 0 &&
			(lregs.VFwrite == uregs.VFwrite ||
			 lregs.VFread0 == uregs.VFwrite ||
			 lregs.VFread1 == uregs.VFwrite);
		const bool vi_hazard = !ibit &&
			(uregs.VIwrite & (1u << REG_CLIP_FLAG)) &&
			((lregs.VIwrite & (1u << REG_CLIP_FLAG)) ||
			 (lregs.VIread  & (1u << REG_CLIP_FLAG)));
		const bool mbit_set    = ((upper >> 29) & 1) != 0;
		const bool fmac_pipe   = (uregs.pipe == VUPIPE_FMAC) || (lregs.pipe == VUPIPE_FMAC);
		const bool branch_pipe = !ibit && (lregs.pipe == VUPIPE_BRANCH);

		// Group bisects (see arm64Emitter.h). When the corresponding aspect macro
		// is defined, any pair where the aspect applies falls back to vu0Exec for
		// the entire pair instead of running the per-pair native machinery.
		bool fallback = false;
#ifdef INTERP_VU0_PAIR
		fallback = true;
#endif
		// Hazard fallback is always on: native VF/CLIP_FLAG save/restore (the
		// thing _vu0Exec lines 104-158 / microVU does to make lower see pre-upper
		// values) is not yet implemented in this JIT. INTERP_VU0_HAZARD is kept as
		// a documentation handle but does not toggle behavior today.
		if (vf_hazard || vi_hazard) fallback = true;
#ifdef INTERP_VU0_MBIT
		if (mbit_set) fallback = true;
#endif
#ifdef INTERP_VU0_DTBITS
		if (dbit_set || tbit_set) fallback = true;
#endif
#ifdef INTERP_VU0_EBIT
		if (ebit_set) fallback = true;
#endif
#ifdef INTERP_VU0_BRANCH
		if (branch_pipe) fallback = true;
#endif

		if (fallback)
		{
			// Per-pair fallback — call vu0Exec for the whole pair. vu0Exec
			// internally bumps cycle and advances TPC by 8 from its current
			// value, so the JIT must NOT touch cycle/TPC here.
			armAsm->Mov(x0, VU0_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(vu0Exec));
		}
		else
		{
			// 1. VU->cycle++
			armAsm->Ldr(x4, MemOperand(VU0_BASE_REG, cycle_off));
			armAsm->Mov(x22, x4);
			armAsm->Add(x4, x4, 1);
			armAsm->Str(x4, MemOperand(VU0_BASE_REG, cycle_off));

			// 2. Advance TPC
			const u32 new_tpc = (pc + 8) & VU0_PROGMASK;
			armAsm->Mov(w4, new_tpc);
			armAsm->Str(w4, MemOperand(VU0_BASE_REG, tpc_off));

			// 3. E-bit
			if (ebit_set)
			{
				armAsm->Mov(w4, 2u);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, ebit_off));
			}

			// 3b. M-bit (early-exit signal to EE — VU0 only)
			// Mirrors _vu0Exec at VU0microInterp.cpp:40-44 and microVU at
			// microVU_Compile.inl:892. Without this, the EE never observes
			// VUFLAG_MFLAGSET and waits forever for VU0 M-bit completion.
			if (mbit_set)
			{
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, flags_off));
				armAsm->Orr(w4, w4, VUFLAG_MFLAGSET);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, flags_off));
			}

			// 5. Upper stalls
			VU0_PERF_BEGIN(_pp_s5);
			armAsm->Mov(x0, VU0_BASE_REG);
			armMoveAddressToReg(x1, &uregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuTestUpperStalls));
			VU0_PERF_END(_pp_s5, "VU0_TestUpper_0x%04x", pc);

			// 5b. Lower stalls
			VU0_PERF_BEGIN(_pp_s5b);
			if (!ibit)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armMoveAddressToReg(x1, &lregs_data[i]);
				armEmitCall(reinterpret_cast<const void*>(_vuTestLowerStalls));
			}
			VU0_PERF_END(_pp_s5b, "VU0_TestLower_0x%04x", pc);

			// 6. Test pipes
			VU0_PERF_BEGIN(_pp_s6);
			armAsm->Mov(x0, VU0_BASE_REG);
			armEmitCall(reinterpret_cast<const void*>(_vuTestPipes));
			VU0_PERF_END(_pp_s6, "VU0_TestPipes_0x%04x", pc);

			// 6b. VIBackupCycles
			armAsm->Mov(x0, VU0_BASE_REG);
			armAsm->Mov(x1, x22);
			armEmitCall(reinterpret_cast<const void*>(vu0DecrementVIBackup));

			// 7. Upper instruction
			if (use_ibit_hack)
			{
				// Live read from micro memory so post-compile patches are visible.
				armAsm->Ldr(x5, MemOperand(VU0_BASE_REG, micro_off));
				armAsm->Ldr(w4, MemOperand(x5, (pc + 4)));
			}
			else
			{
				armAsm->Mov(w4, upper);
			}
			armAsm->Str(w4, MemOperand(VU0_BASE_REG, code_off));
			VU0.code = upper;
			VU0_PERF_BEGIN(_pp_s7);
			recVU0_UpperTable[upper & 0x3f]();
			VU0_PERF_END(_pp_s7, "VU0_U_%02x_0x%04x", upper & 0x3f, pc);

			// 8. Lower instruction
			// NOP the lower when this pair is a branch AND the previous pair
			// set E-bit — "branch in E-bit delay slot" is ISA-undefined.
			// Matches x86 microVU_Compile.inl branchWarning which flags
			// mVUlow.isNOP when the pair is both in an E-bit delay slot and
			// contains a branch. Upper still executes; we just skip the
			// branch rec emission so VU->branch / branchpc stay untouched.
			//
			// Also elide the entire lower emit scaffold (VU0.code store +
			// rec table dispatch) when the lower is a known-NOP on VU0
			// (EFU/XGKICK/XTOP/WAITP/WAITQ) — matches x86 microVU's
			// mVUlow.isNOP pass1 optimization. Saves ~8 bytes of code per
			// NOP op plus a runtime Mov+Str pair.
			const bool suppress_branch = !ibit && branch_pipe && prev_was_ebit;
			const bool lower_is_nop    = !ibit && isVU0LowerNOP(lower);
			if (ibit)
			{
				armAsm->Mov(w4, lower);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, regi_off));
			}
			else if (!lower_is_nop)
			{
				if (use_ibit_hack)
				{
					armAsm->Ldr(x5, MemOperand(VU0_BASE_REG, micro_off));
					armAsm->Ldr(w4, MemOperand(x5, pc));
				}
				else
				{
					armAsm->Mov(w4, lower);
				}
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, code_off));
				VU0.code = lower;
				VU0_PERF_BEGIN(_pp_s8);
				if (!suppress_branch)
					recVU0_LowerTable[lower >> 25]();
				VU0_PERF_END(_pp_s8, "VU0_L_%02x_0x%04x", lower >> 25, pc);
			}

			// 9. FMAC clear
			if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(_vuClearFMAC));
			}

			// 10. Upper stalls add
			armAsm->Mov(x0, VU0_BASE_REG);
			armMoveAddressToReg(x1, &uregs_data[i]);
			armEmitCall(reinterpret_cast<const void*>(_vuAddUpperStalls));

			// 11. Lower stalls add
			if (!ibit)
			{
				armAsm->Mov(x0, VU0_BASE_REG);
				armMoveAddressToReg(x1, &lregs_data[i]);
				armEmitCall(reinterpret_cast<const void*>(_vuAddLowerStalls));
			}

			// 11b. D/T bits — runs AFTER the op so the pair's side effects
			// on VPU_STAT/VI mem-mapped regs happen before the VPU_STAT bit
			// is set, matching x86 microVU_Compile.inl:900-910 which calls
			// mVUDoDBit/mVUDoTBit after mVUexecuteInstruction. D/T → ebit=1
			// is picked up by step 13's ebit countdown below in the same pair.
			//
			// Suppressed when the current pair is itself a branch or is in
			// a branch delay slot — matches x86's `!mVUinfo.isBdelay && !mVUlow.branch`
			// guard (ISA undefined behavior for D/T in these contexts).
			if ((dbit_set || tbit_set) && !branch_pipe && !prev_was_branch)
			{
				armAsm->Mov(w0, upper);
				armEmitCall(reinterpret_cast<const void*>(vu0CheckDTBits));
			}

			// 12. Branch countdown
			{
				Label skip_branch;
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, branch_off));
				armAsm->Cbz(w4, &skip_branch);
				armAsm->Subs(w4, w4, 1);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, branch_off));
				armAsm->B(&skip_branch, ne);
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, branchpc_off));
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, tpc_off));
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu0HandleDelayBranch));
				armAsm->Bind(&skip_branch);
			}

			// 13. Ebit countdown
			{
				Label skip_ebit;
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, ebit_off));
				armAsm->Cbz(w4, &skip_ebit);
				armAsm->Subs(w4, w4, 1);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, ebit_off));
				armAsm->B(&skip_ebit, ne);
				armAsm->Mov(x0, VU0_BASE_REG);
				armEmitCall(reinterpret_cast<const void*>(vu0EbitDone));
				armAsm->Bind(&skip_ebit);
			}

			// 14. FMAC write-position advance
			if (uregs.pipe == VUPIPE_FMAC || lregs.pipe == VUPIPE_FMAC)
			{
				armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, fmacwpos_off));
				armAsm->Add(w4, w4, 1);
				armAsm->And(w4, w4, 3);
				armAsm->Str(w4, MemOperand(VU0_BASE_REG, fmacwpos_off));
			}
		}

		pc = (pc + 8) & (VU0_PROGSIZE - 1);

		// Track branch for next pair's D/T bit suppression (step 11b).
		// Updated regardless of fallback vs native path because the delay
		// slot's D/T check must be suppressed either way.
		prev_was_branch = branch_pipe;

		// Track E-bit for next pair's branch suppression (step 8).
		// Same reasoning: delay-slot context applies regardless of path.
		prev_was_ebit = ebit_set;

		// After each pair (except last): check VPU_STAT and MFLAGSET
		if (i < numPairs - 1)
		{
			armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, vpu_stat_off));
			armAsm->Tbz(w4, 0, &early_exit);
			armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, flags_off));
			armAsm->Tbnz(w4, 1, &early_exit);
		}
	}

	// ----------------------------------------------------------------
	// Exit selector + linkable patch slots (block linking).
	//
	// Layout when num_exits == 1 (B/BAL or max-size fall-through):
	//   returnExit:
	//     [patch_site] B flushes      (← rewritten to successor's linkEntry
	//                                    by tryForwardLink / patchWaitingPredecessors)
	//   flushes:
	//     fall through to early_exit (Ret)
	//
	// Layout when num_exits == 2 (conditional IBxx — both targets known):
	//   returnExit:
	//     Ldr w4, [VU, tpc_off]
	//     Mov w5, taken_target_pc
	//     Cmp w4, w5
	//     B.ne use_not_taken               (hardcoded — NOT a patch slot)
	//     [patch_taken]    B flushes       (exits[1])
	//   use_not_taken:
	//     [patch_not_taken] B flushes      (exits[0])
	//   flushes:
	//     fall through to early_exit (Ret)
	//
	// Layout when indirect (JR/JALR):
	//   returnExit:
	//     Ldr w0, [VU, tpc_off]
	//     Bl vu0_indirect_dispatch     (returns target linkEntry in x0, or null)
	//     Cbz x0, &flushes
	//     Br  x0                       (tail-jump into successor's linkEntry)
	//   flushes:
	//     fall through to early_exit (Ret)
	//
	// Layout when num_exits == 0 and !indirect (ebit-terminated etc.):
	//   No selector — fall directly through to early_exit.
	// ----------------------------------------------------------------
	out_block->returnExit = armGetCurrentCodePointer();
	out_block->num_exits  = link_info.num_exits;
	for (u32 e = 0; e < 2; e++)
	{
		out_block->exits[e].target_pc      = link_info.target_pcs[e];
		out_block->exits[e].patch_site     = nullptr;
		out_block->exits[e].fallthrough    = nullptr;
		out_block->exits[e].current_target = nullptr;
	}

	if (link_info.num_exits == 1)
	{
		u8* patch = armGetCurrentCodePointer();
		Label flushes;
		armAsm->B(&flushes);
		armAsm->Bind(&flushes);
		out_block->exits[0].patch_site     = patch;
		out_block->exits[0].fallthrough    = patch + 4;
		out_block->exits[0].current_target = patch + 4;
	}
	else if (link_info.num_exits == 2)
	{
		armAsm->Ldr(w4, MemOperand(VU0_BASE_REG, tpc_off));
		armAsm->Mov(w5, link_info.target_pcs[1]);
		armAsm->Cmp(w4, w5);

		Label use_not_taken_path;
		Label flushes;
		armAsm->B(&use_not_taken_path, ne);

		u8* patch_taken = armGetCurrentCodePointer();
		armAsm->B(&flushes);                 // exits[1] (taken)

		armAsm->Bind(&use_not_taken_path);
		u8* patch_not_taken = armGetCurrentCodePointer();
		armAsm->B(&flushes);                 // exits[0] (not-taken)

		armAsm->Bind(&flushes);

		out_block->exits[0].patch_site     = patch_not_taken;
		out_block->exits[0].fallthrough    = patch_not_taken + 4;
		out_block->exits[0].current_target = patch_not_taken + 4;

		out_block->exits[1].patch_site     = patch_taken;
		out_block->exits[1].fallthrough    = patch_taken + 8;
		out_block->exits[1].current_target = patch_taken + 8;
	}
	else if (link_info.indirect)
	{
		// JR/JALR: compute target at runtime, dispatch via lookup. On
		// hit, tail-Br to successor's linkEntry. On miss, fall through
		// to flushes (Execute loop will re-dispatch, compile, and the
		// next visit hits the populated cache).
		Label flushes;
		armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, tpc_off));
		armEmitCall(reinterpret_cast<const void*>(vu0_indirect_dispatch));
		armAsm->Cbz(x0, &flushes);
		armAsm->Br(x0);
		armAsm->Bind(&flushes);
	}
	// num_exits == 0 + !indirect: ebit terminated — fall through to epilogue.

	// Epilogue (also used as early-exit target when MFLAGSET or VPU_STAT fires)
	armAsm->Bind(&early_exit);
	armAsm->Ldp(x22, VU0_BASE_REG, MemOperand(sp, 16));
	armAsm->Ldp(x29, x30, MemOperand(sp, 32, PostIndex));
	armAsm->Ret();

	u8* end = armEndBlock();
	s_code_write = end;

	// Register the compiled VU0 block with simpleperf/perfetto so the JIT'd
	// code shows up as `VU0_<startPC>` in profiler reports instead of
	// "unknown unknown". Cost: one map insert per block compile.
	Perf::vu0.RegisterPC(entry, static_cast<size_t>(end - entry), startPC);

	return entry;
}

// ============================================================================
//  recArmVU0
// ============================================================================

recArmVU0::recArmVU0()
{
	m_Idx = 0;
	IsInterpreter = false;
}

void recArmVU0::Reserve()
{
	u8* const buf     = SysMemory::GetVU0Rec();
	u8* const buf_end = SysMemory::GetVU0RecEnd();

	s_pool.Init(buf, POOL_SIZE);
	s_code_base  = buf + POOL_SIZE;
	s_code_write = s_code_base;
	s_code_end   = buf_end;

	deleteAllVariantsVU0();
}

void recArmVU0::Shutdown()
{
	s_pool.Destroy();
	s_code_base  = nullptr;
	s_code_write = nullptr;
	s_code_end   = nullptr;
	deleteAllVariantsVU0();
}

void recArmVU0::Reset()
{
	VU0.fmacwritepos = 0;
	VU0.fmacreadpos  = 0;
	VU0.fmaccount    = 0;
	VU0.ialuwritepos = 0;
	VU0.ialureadpos  = 0;
	VU0.ialucount    = 0;

	deleteAllVariantsVU0();
	if (s_code_base)
		s_code_write = s_code_base;
	s_pool.Reset();
}

void recArmVU0::SetStartPC(u32 startPC)
{
	VU0.start_pc = startPC;
}

void recArmVU0::Step()
{
	VU0.VI[REG_TPC].UL &= VU0_PROGMASK;
	vu0Exec(&VU0);
}

void recArmVU0::Execute(u32 cycles)
{
	const FPControlRegisterBackup fpcr_backup(EmuConfig.Cpu.VU0FPCR);

	VU0.VI[REG_TPC].UL <<= 3;
	VU0.flags &= ~VUFLAG_MFLAGSET;
	const u64 startcycles = VU0.cycle;
	// Publish the cycle limit for the per-linkEntry budget check. Must be
	// set BEFORE the first block runs on this Execute call — without it,
	// any tail-chain of linked VU0 blocks bypasses this while-loop's cycle
	// check and runs forever. Mirrors recArmVU1::Execute.
	s_vu0_cycle_limit = startcycles + cycles;

	while ((VU0.cycle - startcycles) < cycles)
	{
		if (!(VU0.VI[REG_VPU_STAT].UL & 0x1))
		{
			if (VU0.branch)
			{
				VU0.VI[REG_TPC].UL = VU0.branchpc;
				VU0.branch = 0;
			}
			break;
		}
		if (VU0.flags & VUFLAG_MFLAGSET)
			break;

		const u32 pc   = VU0.VI[REG_TPC].UL & (VU0_PROGSIZE - 1);
		const u32 slot = pc / 8;

		// Content-keyed lookup: scan the slot's deque for a variant whose
		// snapshot matches live VU0.Micro at `pc`. A hit bubbles the variant
		// to the deque front (MRU) so subsequent dispatches find it first.
		VU0BlockEntry* blk = findVariantVU0(pc);

		if (!blk)
		{
			// Miss — compile a new variant. Allocate first so the entry is
			// stable while CompileBlock writes linkEntry/exits/etc. into it.
			const u32 numPairs = AnalyzeBlock(pc);
			blk = new VU0BlockEntry{};
			blk->numPairs = numPairs;

			// Snapshot the bytecode this variant compiles against so future
			// dispatches can content-match against it even after Clear()
			// rewrites live VU0.Micro and (eventually) the EE re-uploads
			// the same program. This is what survives the MGS2 thrash
			// pattern.
			const u32 snap_bytes = numPairs * 8;
			blk->snapshot = new u8[snap_bytes];
			std::memcpy(blk->snapshot, VU0.Micro + pc, snap_bytes);

			blk->codeEntry = CompileBlock(pc, numPairs, blk);

			// Cap the per-slot deque. Evict LRU (back) before pushing the
			// new variant; destroyVariantVU0 unpatches predecessors that
			// were linked to the evicted variant so they fall through to
			// the dispatcher (then patchWaitingPredecessorsVU0 below
			// re-links them to the new MRU variant).
			//
			// Eviction must happen BEFORE indexVU0VariantExits so the
			// destroyVariant walk of s_vu0_waitingForSlot[slot] doesn't
			// see the new variant as a predecessor candidate of itself.
			auto& deque = s_vu0_variants[slot];
			if (deque.size() >= kVU0VariantCapPerSlot)
			{
				VU0BlockEntry* victim = deque.back();
				deque.pop_back();
				destroyVariantVU0(victim, slot);
			}

			deque.push_front(blk);
			indexVU0VariantExits(blk);

			// Block-link wiring for the freshly-compiled variant.
			tryForwardLinkVU0(*blk);
			patchWaitingPredecessorsVU0(pc, blk->linkEntry);
		}
		else if (blk->needsRelink)
		{
			// Variant survived a Clear() that unpatched its incoming exit
			// edges — re-wire the graph lazily on the first dispatch
			// post-Clear so repeated hits pay this cost only once.
			tryForwardLinkVU0(*blk);
			patchWaitingPredecessorsVU0(pc, blk->linkEntry);
			blk->needsRelink = false;
		}

		using BlockFn = void (*)();
		reinterpret_cast<BlockFn>(blk->codeEntry)();
	}

	VU0.VI[REG_TPC].UL >>= 3;

	// Skip cycle scaling when either sync gamefix is active and EECycleRate is
	// positive — both VUSyncHack and FullVU0SyncHack tighten VU0 sync, and
	// scaling cycles down would defeat the sync. Matches x86 microVU which
	// treats the two flags as equivalent across microVU_{Compile,Branch,Macro}.inl.
	const bool tight_sync = EmuConfig.Gamefixes.VUSyncHack || EmuConfig.Gamefixes.FullVU0SyncHack;
	if (EmuConfig.Speedhacks.EECycleRate != 0 && (!tight_sync || EmuConfig.Speedhacks.EECycleRate < 0))
	{
		u64 cycle_change = VU0.cycle - startcycles;
		VU0.cycle -= cycle_change;
		switch (std::min(static_cast<int>(EmuConfig.Speedhacks.EECycleRate), static_cast<int>(cycle_change)))
		{
			case -3: cycle_change = static_cast<u64>(cycle_change * 2.0f);       break;
			case -2: cycle_change = static_cast<u64>(cycle_change * 1.6666667f); break;
			case -1: cycle_change = static_cast<u64>(cycle_change * 1.3333333f); break;
			case  1: cycle_change = static_cast<u64>(cycle_change / 1.3f);       break;
			case  2: cycle_change = static_cast<u64>(cycle_change / 1.8f);       break;
			case  3: cycle_change = static_cast<u64>(cycle_change / 3.0f);       break;
			default: break;
		}
		VU0.cycle += cycle_change;
	}

	VU0.nextBlockCycles = (VU0.cycle - cpuRegs.cycle) + 1;
}

void recArmVU0::Clear(u32 addr, u32 size)
{
	// Slot-keyed invalidation. Clear() is called on very hot paths (VIF MPG
	// uploads, VU0 DMA, SPR DMA, every CPU store to VU memory), so any
	// per-call work shows up in frame budget.
	//
	// History:
	//   - 4af448458 "fix: VU0 clear()" — full-scan span-aware. Caused
	//     graphical dropouts + lag in Twinsanity (512 iterations per call).
	//   - 8c0f6b240 "fix: regressions" — bounded span-aware (~130 iterations
	//     per call). Still cost substantial perf across many games.
	//
	// Both attempts were chasing an audit-speculative stale-block bug —
	// "blocks starting before the patched range whose span reaches into
	// it" — that was NEVER confirmed in any actual game. Real-world VU
	// memory patches are either full-program reloads (caught by slot-keyed
	// invalidation since all block start PCs land in the clear range) or
	// writes to unused data regions (no block cares). The theoretical
	// scenario requires mid-program patches to hot code, which games
	// overwhelmingly avoid.
	//
	// x86 microVU solves the cross-block case via program-level hashing
	// (mVU.prog.isSame / mVUcacheProg), a completely different architecture
	// that we don't mirror. If a real game ever pins down the stale-block
	// scenario, revisit — but don't pay the scan cost speculatively.
	const u32 first        = addr / 8;
	const u32 last         = (addr + size + 7) / 8;
	const u32 clamped_last = std::min(last, VU0_NUM_SLOTS);

	if (first >= VU0_NUM_SLOTS)
		return;

	// Block-linking invalidation: any predecessor that has a B patched to a
	// linkEntry inside the cleared range needs that exit reverted to its
	// fallthrough — otherwise the predecessor's next execution jumps into
	// code that may be overwritten by a fresh compile occupying the same
	// buffer region. Walk the reverse index for each cleared slot.
	for (u32 ts = first; ts < clamped_last; ts++)
	{
		for (VU0BlockEntry* pred : s_vu0_waitingForSlot[ts])
		{
			for (u32 e = 0; e < pred->num_exits; e++)
			{
				LinkExit& exit = pred->exits[e];
				if (!exit.patch_site || exit.target_pc == LINK_TARGET_NONE)
					continue;
				const u32 target_slot = exit.target_pc / 8;
				if (target_slot >= first && target_slot < clamped_last)
					unpatchVU0LinkSite(exit);
			}
		}
	}

	// Mark variants in the cleared range as needing relink on next dispatch.
	// We deliberately do NOT delete them: if the EE re-uploads identical
	// bytes later (the MGS2 thrash pattern), findVariantVU0 will match
	// against the preserved snapshot and reuse the compiled code without
	// re-emitting. On the first post-Clear dispatch, `needsRelink` triggers
	// tryForwardLinkVU0 + patchWaitingPredecessorsVU0 to re-wire exits.
	for (u32 i = first; i < clamped_last; i++)
	{
		for (VU0BlockEntry* blk : s_vu0_variants[i])
			blk->needsRelink = true;
	}
}
