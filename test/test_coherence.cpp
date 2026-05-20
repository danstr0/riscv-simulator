/**
 * @file test_coherence.cpp
 * @brief Tests for the MESI cache coherence controller.
 *
 * Sections:
 *   1 (line 106) : Read miss — no other copy → Exclusive
 *   2 (line 123) : Read miss — another core has Exclusive → both Shared
 *   3 (line 142) : Read miss — another core has Modified → writeback + both Shared
 *   4 (line 172) : Write miss — no other copy → Modified
 *   5 (line 185) : Write to Exclusive — silent upgrade to Modified
 *   6 (line 204) : Write to Shared — invalidate others → Modified
 *   7 (line 229) : Write miss — another core Modified → writeback + invalidate → Modified
 *   8 (line 254) : Eviction — removes directory entry
 *   9 (line 287) : Statistics — coherence traffic counters
 *  10 (line 316) : Data integrity — writes visible across cores after coherence
 */

#include "core/cache.hpp"
#include "core/coherence.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ── Test fixture ───────────────────────────────────────────────────────

// Creates a 2-core setup with L1 caches backed by a shared memory
struct CoherenceSetup
{
    std::shared_ptr<FlatMemory> main_mem;
    std::shared_ptr<Cache>      l1_0; // Core 0 L1
    std::shared_ptr<Cache>      l1_1; // Core 1 L1
    CoherenceController         ctrl;

    static CoherenceSetup create(u32 num_cores = 2)
    {
        auto mem = std::make_shared<FlatMemory>(0, 0x10000);

        CacheConfig l1_cfg = {
            .size_bytes    = 1024,
            .line_size     = 64,
            .associativity = 4,
            .hit_latency   = 1,
            .miss_penalty  = 10,
            .replacement   = ReplacementPolicy::LRU,
            .write_pol     = WritePolicy::WRITE_BACK,
            .write_alloc   = WriteAllocate::ALLOCATE,
        };

        auto l1_0 = std::make_shared<Cache>(l1_cfg, mem);
        auto l1_1 = std::make_shared<Cache>(l1_cfg, mem);

        std::vector<Cache*> caches = {l1_0.get(), l1_1.get()};

        return CoherenceSetup{
            mem,
            l1_0,
            l1_1,
            CoherenceController(caches, mem, 64),
        };
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  1. Read miss — no other copy → Exclusive
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_read_miss_exclusive) {
    auto s = CoherenceSetup::create();
    s.main_mem->write32(0x100u, 0xDEAD'BEEFu);

    // Core 0 reads — no one else has it
    s.ctrl.handle_read_miss(0, 0x100u);

    // After handling, controller should mark core 0 as Exclusive
    ASSERT_EQ(s.ctrl.line_state(0, 0x100u), MESIState::Exclusive);
    ASSERT_EQ(s.ctrl.line_state(1, 0x100u), MESIState::Invalid);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Read miss — another core has Exclusive → both Shared
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_read_shared_from_exclusive) {
    auto s = CoherenceSetup::create();
    s.main_mem->write32(0x100u, 42);

    // Core 0 reads → Exclusive
    s.ctrl.handle_read_miss(0, 0x100u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x100u), MESIState::Exclusive);

    // Core 1 reads the same line → both become Shared
    s.ctrl.handle_read_miss(1, 0x100u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x100u), MESIState::Shared);
    ASSERT_EQ(s.ctrl.line_state(1, 0x100u), MESIState::Shared);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Read miss — another core has Modified → writeback + Shared
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_read_from_modified) {
    auto s = CoherenceSetup::create();
    s.main_mem->write32(0x100u, 0);

    // Core 0 gets Exclusive, then writes → Modified
    s.ctrl.handle_read_miss(0, 0x100u);
    // Trigger the actual L1 fetch so the line is present
    s.l1_0->read32(0x100u);
    // Write to it (makes it dirty in L1 and Modified in directory)
    s.l1_0->write32(0x100u, 0xCAFE'BABEu);
    s.ctrl.handle_write_miss(0, 0x100u);  // E→M
    ASSERT_EQ(s.ctrl.line_state(0, 0x100u), MESIState::Modified);

    // Core 1 reads → core 0 writes back, both become Shared
    s.ctrl.handle_read_miss(1, 0x100u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x100u), MESIState::Shared);
    ASSERT_EQ(s.ctrl.line_state(1, 0x100u), MESIState::Shared);

    // The writeback should have pushed 0xCAFEBABE to main memory
    ASSERT_EQ(s.main_mem->read32(0x100u).value, 0xCAFE'BABEu);

    ASSERT(s.ctrl.stats().writebacks_forced > 0);
    ASSERT(s.ctrl.stats().shared_transfers > 0);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Write miss — no other copy → Modified
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_write_miss_modified) {
    auto s = CoherenceSetup::create();

    s.ctrl.handle_write_miss(0, 0x200u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x200u), MESIState::Modified);
    ASSERT_EQ(s.ctrl.line_state(1, 0x200u), MESIState::Invalid);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Write to Exclusive — silent upgrade
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_exclusive_to_modified) {
    auto s = CoherenceSetup::create();

    // Core 0 reads → Exclusive
    s.ctrl.handle_read_miss(0, 0x300u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x300u), MESIState::Exclusive);

    // Core 0 writes → Modified (silent, no invalidations needed)
    u32 lat = s.ctrl.handle_write_miss(0, 0x300u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x300u), MESIState::Modified);
    ASSERT_EQ(lat, 0u);  // silent upgrade has zero additional latency
    ASSERT_EQ(s.ctrl.stats().invalidations_sent, 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Write to Shared — invalidate others
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_shared_to_modified_invalidates) {
    auto s = CoherenceSetup::create();

    // Both cores read → Shared
    s.ctrl.handle_read_miss(0, 0x400u);
    s.ctrl.handle_read_miss(1, 0x400u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x400u), MESIState::Shared);
    ASSERT_EQ(s.ctrl.line_state(1, 0x400u), MESIState::Shared);

    // Populate L1s so snoop_invalidate can find the line
    s.l1_0->read32(0x400u);
    s.l1_1->read32(0x400u);

    // Core 0 writes → invalidates core 1, core 0 becomes Modified
    s.ctrl.handle_write_miss(0, 0x400u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x400u), MESIState::Modified);
    ASSERT_EQ(s.ctrl.line_state(1, 0x400u), MESIState::Invalid);
    ASSERT(s.ctrl.stats().invalidations_sent > 0);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Write miss — another core Modified → writeback + invalidate
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_write_steals_modified) {
    auto s = CoherenceSetup::create();
    s.main_mem->write32(0x500u, 100);

    // Core 0 gets line, writes → Modified
    s.ctrl.handle_read_miss(0, 0x500u);
    s.l1_0->read32(0x500u);
    s.l1_0->write32(0x500u, 999);
    s.ctrl.handle_write_miss(0, 0x500u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x500u), MESIState::Modified);

    // Core 1 writes to same line → core 0 writes back and is invalidated
    s.ctrl.handle_write_miss(1, 0x500u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x500u), MESIState::Invalid);
    ASSERT_EQ(s.ctrl.line_state(1, 0x500u), MESIState::Modified);

    // Core 0's dirty data should have been written back to memory
    ASSERT_EQ(s.main_mem->read32(0x500u).value, 999u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  8. Eviction — removes directory entry
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_eviction_cleans_directory) {
    auto s = CoherenceSetup::create();

    s.ctrl.handle_read_miss(0, 0x600u);
    ASSERT_EQ(s.ctrl.num_tracked_lines(), 1u);

    s.ctrl.notify_eviction(0, 0x600u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x600u), MESIState::Invalid);
    // With no valid entries, the directory entry should be cleaned up
    ASSERT_EQ(s.ctrl.num_tracked_lines(), 0u);
    return true;
}

TEST(mesi_eviction_partial) {
    auto s = CoherenceSetup::create();

    // Both cores have the line
    s.ctrl.handle_read_miss(0, 0x700u);
    s.ctrl.handle_read_miss(1, 0x700u);
    ASSERT_EQ(s.ctrl.num_tracked_lines(), 1u);

    // Core 0 evicts — core 1 still has it
    s.ctrl.notify_eviction(0, 0x700u);
    ASSERT_EQ(s.ctrl.line_state(0, 0x700u), MESIState::Invalid);
    ASSERT_EQ(s.ctrl.line_state(1, 0x700u), MESIState::Shared);
    ASSERT_EQ(s.ctrl.num_tracked_lines(), 1u); // entry still exists
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  9. Statistics
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_stats_tracking) {
    auto s = CoherenceSetup::create();

    // Read miss from core 0 → l2_fetch
    s.ctrl.handle_read_miss(0, 0x800u);
    ASSERT_EQ(s.ctrl.stats().read_misses, 1u);
    ASSERT_EQ(s.ctrl.stats().l2_fetches, 1u);

    // Read from core 1 (E→S in core 0) → another l2_fetch
    s.ctrl.handle_read_miss(1, 0x800u);
    ASSERT_EQ(s.ctrl.stats().read_misses, 2u);

    // Write from core 0 (S→M, invalidates core 1)
    s.l1_0->read32(0x800u);
    s.l1_1->read32(0x800u);
    s.ctrl.handle_write_miss(0, 0x800u);
    ASSERT_EQ(s.ctrl.stats().write_misses, 1u);
    ASSERT_EQ(s.ctrl.stats().upgrades, 1u);
    ASSERT(s.ctrl.stats().invalidations_sent >= 1u);

    s.ctrl.reset_stats();
    ASSERT_EQ(s.ctrl.stats().read_misses, 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  10. Data integrity across cores
// ═══════════════════════════════════════════════════════════════════════

TEST(mesi_data_visible_after_coherence) {
    auto s = CoherenceSetup::create();
    s.main_mem->write32(0x900u, 0);

    // Core 0 reads and writes
    s.ctrl.handle_read_miss(0, 0x900u);
    s.l1_0->read32(0x900u);
    s.l1_0->write32(0x900u, 42);
    s.ctrl.handle_write_miss(0, 0x900u);

    // Core 1 wants to read — triggers coherence
    s.ctrl.handle_read_miss(1, 0x900u);

    // The writeback should have pushed 42 to main memory
    // Core 1's L1 will fetch from main memory
    ASSERT_EQ(s.main_mem->read32(0x900u).value, 42u);
    return true;
}
