/**
 * @file test_timer.cpp
 * @brief Tests for the CLINT-style machine timer.
 *
 * Sections:
 *   1 (line  20) : Basic state
 *   2 (line  40) : Compare match — interrupt fires when mtime >= mtimecmp
 *   3 (line  98) : Clear — writing new mtimecmp clears interrupt
 *   4 (line 133) : MMIO — register reads/writes
 *   5 (line 199) : Notification callback
 *   6 (line 236) : Edge cases — overflow, immediate match
 */

#include "core/timer.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic state
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_no_interrupt_initially)
{
    Timer timer;
    ASSERT(!timer.interrupt_pending());
    return true;
}

TEST(timer_default_compare_is_max)
{
    Timer timer;
    // Default mtimecmp = ~0ULL, so even a large cycle count won't trigger
    timer.tick(1'000'000);
    ASSERT(!timer.interrupt_pending());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Compare match
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_fires_at_compare)
{
    Timer timer;
    timer.set_compare(100);
    ASSERT(!timer.interrupt_pending());

    timer.tick(99);
    ASSERT(!timer.interrupt_pending());

    timer.tick(100);
    ASSERT(timer.interrupt_pending());
    return true;
}

TEST(timer_fires_past_compare)
{
    Timer timer;
    timer.set_compare(100);
    timer.tick(200);
    ASSERT(timer.interrupt_pending());
    return true;
}

TEST(timer_tick_returns_true_on_transition)
{
    Timer timer;
    timer.set_compare(50);

    bool fired = timer.tick(49);
    ASSERT(!fired);

    fired = timer.tick(50);
    ASSERT(fired);

    // Already pending — tick doesn't "re-fire"
    fired = timer.tick(51);
    ASSERT(!fired);
    ASSERT(timer.interrupt_pending());
    return true;
}

TEST(timer_stays_pending)
{
    Timer timer;
    timer.set_compare(10);
    timer.tick(10);
    ASSERT(timer.interrupt_pending());

    // Advancing further doesn't clear it
    timer.tick(100);
    ASSERT(timer.interrupt_pending());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Clear by writing new mtimecmp
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_clear_by_new_compare)
{
    Timer timer;
    timer.set_compare(50);
    timer.tick(60);
    ASSERT(timer.interrupt_pending());

    // Write a future compare value -> clears the interrupt
    timer.set_compare(200);
    ASSERT(!timer.interrupt_pending());
    return true;
}

TEST(timer_re_fires_after_clear)
{
    Timer timer;
    timer.set_compare(50);
    timer.tick(60);
    ASSERT(timer.interrupt_pending());

    timer.set_compare(100);
    ASSERT(!timer.interrupt_pending());

    timer.tick(99);
    ASSERT(!timer.interrupt_pending());

    timer.tick(100);
    ASSERT(timer.interrupt_pending());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. MMIO register access
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_mmio_read_mtime)
{
    Timer timer;
    timer.tick(0x12345678);
 
    auto lo = timer.read32(0x00); // MTIME_LO
    ASSERT(lo.ok);
    ASSERT_HEX_EQ(lo.value, 0x12345678u);
 
    auto hi = timer.read32(0x04); // MTIME_HI
    ASSERT(hi.ok);
    ASSERT_EQ(hi.value, 0u); // cycle count fits in 32 bits
    return true;
}

TEST(timer_mmio_write_compare)
{
    Timer timer;

    (void)timer.write32(0x0C, 0);   // MTIMECMP_HI = 0
    (void)timer.write32(0x08, 100); // MTIMECMP_LO = 100
    ASSERT_EQ(timer.compare(), 100u);

    timer.tick(100);
    ASSERT(timer.interrupt_pending());
    return true;
}

TEST(timer_mmio_write_compare_clears)
{
    Timer timer;
    timer.set_compare(50);
    timer.tick(60);
    ASSERT(timer.interrupt_pending());

    // Write new compare via MMIO
    (void)timer.write32(0x08, 200);
    ASSERT(!timer.interrupt_pending());
    ASSERT_EQ(timer.compare(), 200u);
    return true;
}

TEST(timer_mmio_mtime_read_only)
{
    Timer timer;
    timer.tick(42);

    // Writing to mtime should be ignored
    (void)timer.write32(0x00, 999);
    auto r = timer.read32(0x00);
    ASSERT_EQ(r.value, 42u);  // unchanged
    return true;
}

TEST(timer_mmio_invalid_address)
{
    Timer timer;
    auto r = timer.read32(0x20); // out of range
    ASSERT(!r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Notification callback
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_notify_on_fire)
{
    Timer timer;
    int notify_count = 0;
    bool last_state = false;
    timer.set_notify([&](bool pending)
    {
        notify_count++;
        last_state = pending;
    });

    timer.set_compare(50);
    timer.tick(50);
    ASSERT(notify_count > 0);
    ASSERT(last_state);
    return true;
}

TEST(timer_notify_on_clear)
{
    Timer timer;
    bool last_state = true;
    timer.set_notify([&](bool pending) { last_state = pending; });

    timer.set_compare(50);
    timer.tick(60);
    ASSERT(last_state);

    timer.set_compare(200);
    ASSERT(!last_state);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Edge cases
// ═══════════════════════════════════════════════════════════════════════

TEST(timer_immediate_match)
{
    Timer timer;
    timer.tick(100);
    // Set compare to current time -> fires immediately
    timer.set_compare(100);
    ASSERT(timer.interrupt_pending());
    return true;
}

TEST(timer_compare_zero)
{
    Timer timer;
    timer.set_compare(0);
    // mtime starts at 0, 0 >= 0 -> pending
    timer.tick(0);
    ASSERT(timer.interrupt_pending());
    return true;
}
