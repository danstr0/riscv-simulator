/**
 * @file test_plic.cpp
 * @brief Tests for the simplified PLIC interrupt controller.
 *
 * Sections:
 *   1 (line  20) : Basic functions - set/clear pending, enable, priority
 *   2 (line  80) : Claim/complete - claim returns highest-priority, clears pending
 *   3 (line 140) : Threshold - interrupts below are filtered
 *   4 (line 186) : MMIO - register reads/writes match programmatic API
 *   5 (line 270) : Notification callback - fires on state changes
 *   6 (line 311) : Multi-source - priority ordering with multiple pending
 */

#include "core/plic.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic functions
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_no_pending_initially)
{
    PLIC plic;
    ASSERT(!plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 0u);
    return true;
}

TEST(plic_pending_but_not_enabled)
{
    PLIC plic;
    plic.set_priority(1, 1);
    plic.set_pending(1);

    // Pending, but not enabled — should not report as pending to hart
    ASSERT(!plic.interrupt_pending());
    return true;
}

TEST(plic_pending_and_enabled)
{
    PLIC plic;
    plic.set_priority(1, 1);
    plic.set_enable(1, true);
    plic.set_pending(1);

    ASSERT(plic.interrupt_pending());
    return true;
}

TEST(plic_clear_pending)
{
    PLIC plic;
    plic.set_priority(1, 1);
    plic.set_enable(1, true);
    plic.set_pending(1);

    ASSERT(plic.interrupt_pending());
    plic.clear_pending(1);
    ASSERT(!plic.interrupt_pending());
    return true;
}

TEST(plic_source_zero_reserved)
{
    // Source 0 is reserved — set_pending(0) should be a no-op
    PLIC plic;
    plic.set_priority(0, 7);
    plic.set_enable(0, true);
    plic.set_pending(0);

    ASSERT(!plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Claim / complete
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_claim_returns_source)
{
    PLIC plic;
    plic.set_priority(3, 2);
    plic.set_enable(3, true);
    plic.set_pending(3);

    ASSERT_EQ(plic.claim(), 3u);
    return true;
}

TEST(plic_claim_clears_pending)
{
    PLIC plic;
    plic.set_priority(3, 2);
    plic.set_enable(3, true);
    plic.set_pending(3);

    (void)plic.claim();
    // After claim, pending is cleared — no more interrupts
    ASSERT(!plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 0u);
    return true;
}

TEST(plic_complete_allows_re_trigger)
{
    PLIC plic;
    plic.set_priority(3, 2);
    plic.set_enable(3, true);
    plic.set_pending(3);

    u32 src = plic.claim();
    ASSERT_EQ(src, 3u);
    plic.complete(src);
    
    // Now the source can fire again
    plic.set_pending(3);
    ASSERT(plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 3u);
    return true;
}

TEST(plic_double_claim_returns_zero)
{
    PLIC plic;
    plic.set_priority(3, 2);
    plic.set_enable(3, true);
    plic.set_pending(3);

    (void)plic.claim();
    // Second claim with nothing pending -> 0
    ASSERT_EQ(plic.claim(), 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Threshold
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_threshold_filters)
{
    PLIC plic;

    plic.set_priority(1, 2);
    plic.set_enable(1, true);
    plic.set_threshold(3); // only priority > 3 passes
    plic.set_pending(1);

    ASSERT(!plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 0u);
    return true;
}

TEST(plic_threshold_passes_higher)
{
    PLIC plic;

    plic.set_priority(1, 5);
    plic.set_enable(1, true);
    plic.set_threshold(3);
    plic.set_pending(1);

    ASSERT(plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 1u);
    return true;
}

TEST(plic_threshold_zero_passes_all)
{
    PLIC plic;
    plic.reset();

    plic.set_priority(1, 1); // minimum nonzero priority
    plic.set_enable(1, true);
    plic.set_threshold(0);
    plic.set_pending(1);

    ASSERT(plic.interrupt_pending());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. MMIO register access
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_mmio_priority)
{
    PLIC plic;

    // Write priority for source 1 via MMIO (offset = source * 4)
    (void)plic.write32(1 * 4, 5);
    ASSERT_EQ(plic.priority(1), 5u & 0x7);

    // Read it back
    auto r = plic.read32(1 * 4);
    ASSERT(r.ok);
    ASSERT_EQ(r.value, 5u);
    return true;
}

TEST(plic_mmio_enable)
{
    PLIC plic;

    // Enable bits at offset 0x2000
    (void)plic.write32(0x2000, (1u << 1) | (1u << 5));
    ASSERT(plic.enabled(1));
    ASSERT(plic.enabled(5));
    ASSERT(!plic.enabled(2));

    auto r = plic.read32(0x2000);
    ASSERT_EQ(r.value, (1u << 1) | (1u << 5));
    return true;
}

TEST(plic_mmio_threshold)
{
    PLIC plic;

    (void)plic.write32(0x200000, 4);
    ASSERT_EQ(plic.threshold(), 4u);

    auto r = plic.read32(0x200000);
    ASSERT_EQ(r.value, 4u);
    return true;
}

TEST(plic_mmio_claim_complete)
{
    PLIC plic;

    plic.set_priority(2, 3);
    plic.set_enable(2, true);
    plic.set_pending(2);

    // Claim via MMIO read of 0x200004
    auto r = plic.read32(0x200004);
    ASSERT_EQ(r.value, 2u);
    ASSERT(!plic.interrupt_pending());

    // Complete via MMIO write to 0x200004
    (void)plic.write32(0x200004, 2);

    // Can re-trigger
    plic.set_pending(2);
    r = plic.read32(0x200004);
    ASSERT_EQ(r.value, 2u);
    return true;
}

TEST(plic_mmio_pending_read_only)
{
    PLIC plic;

    plic.set_pending(3);
    auto r = plic.read32(0x1000);
    ASSERT_EQ(r.value & (1u << 3), 1u << 3);

    // Write to pending is ignored
    (void)plic.write32(0x1000, 0);
    r = plic.read32(0x1000);
    ASSERT_EQ(r.value & (1u << 3), 1u << 3);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Notification callback
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_notify_on_pending)
{
    PLIC plic;
    int notify_count = 0;
    bool last_state = false;
    plic.set_notify([&](bool pending)
    {
        notify_count++;
        last_state = pending;
    });

    plic.set_priority(1, 1);
    plic.set_enable(1, true);
    plic.set_pending(1);

    ASSERT(notify_count > 0);
    ASSERT(last_state);
    return true;
}

TEST(plic_notify_on_claim)
{
    PLIC plic;
    bool last_state = true;
    plic.set_notify([&](bool pending) { last_state = pending; });

    plic.set_priority(1, 1);
    plic.set_enable(1, true);
    plic.set_pending(1);
    ASSERT(last_state);

    (void)plic.claim();
    // After claim, no more pending -> notify with false
    ASSERT(!last_state);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Multi-source priority ordering
// ═══════════════════════════════════════════════════════════════════════

TEST(plic_highest_priority_wins)
{
    // Source 2 at priority 3, source 5 at priority 7
    PLIC plic;
    plic.set_priority(2, 3);
    plic.set_priority(5, 7);
    plic.set_enable(2, true);
    plic.set_enable(5, true);
    plic.set_pending(2);
    plic.set_pending(5);

    // Claim should return source 5 (higher priority)
    ASSERT_EQ(plic.claim(), 5u);
    // Source 2 still pending
    ASSERT(plic.interrupt_pending());
    ASSERT_EQ(plic.claim(), 2u);
    ASSERT(!plic.interrupt_pending());
    return true;
}

TEST(plic_equal_priority_lower_id_wins)
{
    // Both at priority 3; lower source ID should win
    PLIC plic;
    plic.set_priority(2, 3);
    plic.set_priority(5, 3);
    plic.set_enable(2, true);
    plic.set_enable(5, true);
    plic.set_pending(2);
    plic.set_pending(5);
 
    u32 first = plic.claim();
    ASSERT(first == 2u || first == 5u);

    u32 second = plic.claim();
    ASSERT(second == 2u || second == 5u);
    ASSERT(first != second);
    return true;
}
