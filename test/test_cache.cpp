/**
 * @file test_cache.cpp
 * @brief Tests for the parameterised cache and cache hierarchy.
 *
 * Sections:
 *   1 (line 117) : Basic cache operations
 *   2 (line 189) : Set associativity and eviction
 *   3 (line 241) : Replacement policies
 *   4 (line 387) : Write policies
 *   5 (line 446) : Cache control
 *   6 (line 560) : Cache hierarchy
 *   7 (line 769) : Edge cases
 *   8 (line 863) : Statistics
 */

#include "core/cache.hpp"
#include "test_framework.hpp"
#include <cmath>

using namespace riscv;

// ── Helpers ────────────────────────────────────────────────────────────

// 1 KB, 64-byte lines, 2-way, write-back / write-allocate / LRU
static CacheConfig simple_config()
{
    return {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 2,
        .hit_latency   = 1,
        .miss_penalty  = 10,
    };
}

// Same geometry but with a specific replacement policy
static CacheConfig config_with_replacement(ReplacementPolicy rp)
{
    auto c       = simple_config();
    c.replacement = rp;
    return c;
}

// Same geometry but write-through + write-allocate
static CacheConfig write_through_config()
{
    auto c       = simple_config();
    c.write_pol  = WritePolicy::WRITE_THROUGH;
    return c;
}

// Same geometry but write-back + no-allocate on write miss
static CacheConfig no_allocate_config()
{
    auto c          = simple_config();
    c.write_alloc   = WriteAllocate::NO_ALLOCATE;
    return c;
}

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic cache operations - hit, miss, line fill
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_read_miss_then_hit) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0xDEAD'BEEFu);

    Cache cache(simple_config(), mem);

    auto r1 = cache.read32(0x100u);
    ASSERT(r1.ok);
    ASSERT_HEX_EQ(r1.value, 0xDEAD'BEEFu);
    ASSERT_EQ(cache.stats().misses, 1u);
    ASSERT_EQ(cache.stats().hits, 0u);
    ASSERT_EQ(r1.cycles, 1u + 10u);

    auto r2 = cache.read32(0x100u);
    ASSERT_HEX_EQ(r2.value, 0xDEAD'BEEFu);
    ASSERT_EQ(cache.stats().hits, 1u);
    ASSERT_EQ(r2.cycles, 1u);
    return true;
}

TEST(cache_write_allocate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write32(0x200u, 0x1234'5678u);
    ASSERT_EQ(cache.stats().misses, 1u);

    auto r = cache.read32(0x200u);
    ASSERT_HEX_EQ(r.value, 0x1234'5678u);
    ASSERT_EQ(cache.stats().hits, 1u);
    return true;
}

TEST(cache_line_fill) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0x1111'1111u);
    mem->write32(0x104u, 0x2222'2222u);
    mem->write32(0x108u, 0x3333'3333u);

    Cache cache(simple_config(), mem);

    cache.read32(0x100u);
    ASSERT_EQ(cache.stats().misses, 1u);

    // Same line — should all hit
    ASSERT_HEX_EQ(cache.read32(0x104u).value, 0x2222'2222u);
    ASSERT_HEX_EQ(cache.read32(0x108u).value, 0x3333'3333u);
    ASSERT_EQ(cache.stats().hits, 2u);
    return true;
}

TEST(cache_different_lines) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0xAAAA'AAAAu);
    mem->write32(0x200u, 0xBBBB'BBBBu);

    Cache cache(simple_config(), mem);

    cache.read32(0x100u);
    cache.read32(0x200u);
    ASSERT_EQ(cache.stats().misses, 2u);

    cache.read32(0x100u);
    cache.read32(0x200u);
    ASSERT_EQ(cache.stats().hits, 2u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Set associativity and eviction
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_associativity_eviction) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = simple_config();
    Cache cache(cfg, mem);

    // Stride to hit the same set: num_sets * line_size
    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0x000u, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    ASSERT_EQ(cache.stats().misses, 2u);

    // Both should hit
    cache.read32(a);
    cache.read32(b);
    ASSERT_EQ(cache.stats().hits, 2u);

    // c evicts LRU (a)
    cache.read32(c);
    ASSERT_EQ(cache.stats().evictions, 1u);

    cache.read32(b); // hit
    cache.read32(c); // hit
    cache.read32(a); // miss (was evicted)
    ASSERT_EQ(cache.stats().misses, 4u);
    return true;
}

TEST(cache_lru_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = simple_config();
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    cache.read32(a); // touch a again — now b is LRU

    cache.read32(c); // evicts b

    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);                    // hit
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency + cfg.miss_penalty); // miss
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Replacement policies
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_fifo_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = config_with_replacement(ReplacementPolicy::FIFO);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a); // ways: [a(fifo=0), -]
    cache.read32(b); // ways: [a(fifo=0), b(fifo=1)]
    cache.read32(a); // touch a again — FIFO ignores recency

    cache.read32(c); // evicts a

    // a should miss, b should hit
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency + cfg.miss_penalty);
    return true;
}

TEST(cache_random_replacement) {
    // Verify this doesn't crash
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = config_with_replacement(ReplacementPolicy::RANDOM);
    Cache cache(cfg, mem);

    for (u32 i = 0; i < 1000; ++i)
        cache.read32(i * 4);

    ASSERT(cache.stats().misses > 0u);
    ASSERT(cache.stats().hits + cache.stats().misses == 1000u);
    return true;
}

TEST(cache_mru_replacement) {
    // After accessing a, b, then c, the MRU victim is b
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = config_with_replacement(ReplacementPolicy::MRU);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a); // ways: [a, -]
    cache.read32(b); // ways: [a, b]; b is MRU

    // MRU evicts b
    cache.read32(c);

    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency + cfg.miss_penalty);
    return true;
}

TEST(cache_mru_for_scanning) {
    // MRU is advantageous on sequential scans that exceed cache 
    // capacity: it keeps the oldest lines, which are the ones most
    // likely to be revisited in a cyclic scan.
    auto mem_lru = std::make_shared<FlatMemory>(0, 0x100000u);
    auto mem_mru = std::make_shared<FlatMemory>(0, 0x100000u);

    auto cfg_lru = config_with_replacement(ReplacementPolicy::LRU);
    auto cfg_mru = config_with_replacement(ReplacementPolicy::MRU);
    Cache lru(cfg_lru, mem_lru);
    Cache mru(cfg_mru, mem_mru);

    u32 range = cfg_lru.size_bytes * 2;
    for (int pass = 0; pass < 3; ++pass)
    {
        for (u32 addr = 0; addr < range; addr += 64)
	{
            lru.read32(addr);
            mru.read32(addr);
        }
    }

    // MRU should achieve a meaningfully higher hit rate
    ASSERT(mru.stats().hit_rate() > lru.stats().hit_rate());
    return true;
}

TEST(cache_plru_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = config_with_replacement(ReplacementPolicy::PLRU);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    cache.read32(a); // tree should point toward b
    
    cache.read32(c); // evicts b 

    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency + cfg.miss_penalty);
    return true;
}

TEST(cache_plru_4way) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    CacheConfig cfg = {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 4,
        .hit_latency   = 1,
        .miss_penalty  = 10,
        .replacement   = ReplacementPolicy::PLRU,
    };
    Cache cache(cfg, mem);
 
    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride, d = c + stride, e = d + stride;
 
    // Fill all 4 ways
    cache.read32(a);
    cache.read32(b);
    cache.read32(c);
    cache.read32(d);
 
    // Touch a and c to mark them as recently used
    cache.read32(a);
    cache.read32(c);
 
    // e should evict one of {b, d}
    cache.read32(e);
 
    // a and c should still hit
    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(c).cycles, cfg.hit_latency);
 
    // At least one of {b, d} should miss
    auto rb = cache.read32(b);
    auto rd = cache.read32(d);
    bool one_evicted = (rb.cycles > cfg.hit_latency)
	                || (rd.cycles > cfg.hit_latency);
    ASSERT(one_evicted);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Write policies
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_writeback_dirty) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    auto cfg = simple_config(); // write-back by default
    Cache cache(cfg, mem);

    cache.write32(0x100u, 0xCAFE'BABEu);

    // Cache has the value, but backing memory does not
    ASSERT_HEX_EQ(cache.read32(0x100u).value, 0xCAFE'BABEu);
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x0000'0000u);

    // Force eviction
    u32 stride = cfg.num_sets() * cfg.line_size;
    cache.read32(0x100u + stride);
    cache.read32(0x100u + 2 * stride);

    // Now backing memory should have the value
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0xCAFE'BABEu);
    ASSERT_EQ(cache.stats().dirty_evictions, 1u);
    return true;
}

TEST(cache_write_through) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(write_through_config(), mem);

    cache.write32(0x100u, 0xDEAD'C0DEu);

    // Backing memory should have the value immediately
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0xDEAD'C0DEu);

    // The cache should also have it
    ASSERT_HEX_EQ(cache.read32(0x100u).value, 0xDEAD'C0DEu);
    ASSERT_EQ(cache.stats().hits, 1u);
    return true;
}

TEST(cache_write_no_allocate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(no_allocate_config(), mem);

    // Goes directly to backing memory, does NOT fill the cache line
    cache.write32(0x100u, 0xBEEF'BEEFu);
    ASSERT_EQ(cache.stats().misses, 1u);

    // The value should be in backing memory
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0xBEEF'BEEFu);

    // ...but NOT in the cache
    auto r = cache.read32(0x100u);
    ASSERT_EQ(cache.stats().misses, 2u);  // Another miss
    ASSERT_HEX_EQ(r.value, 0xBEEF'BEEFu); // Gets it from memory
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Cache control
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_invalidate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0xAAAA'AAAAu);
    Cache cache(simple_config(), mem);

    cache.read32(0x100u);
    ASSERT(cache.lookup(0x100u));

    cache.invalidate(0x100u);
    ASSERT(!cache.lookup(0x100u));

    cache.read32(0x100u);
    ASSERT_EQ(cache.stats().misses, 2u);
    return true;
}

TEST(cache_invalidate_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    for (addr_t a = 0; a < 1024; a += 64) cache.read32(a);
    u64 misses_before = cache.stats().misses;

    // Verify all hit
    for (addr_t a = 0; a < 1024; a += 64) cache.read32(a);
    ASSERT_EQ(cache.stats().misses, misses_before);

    cache.invalidate_all();

    // All should miss again
    for (addr_t a = 0; a < 1024; a += 64) cache.read32(a);
    ASSERT(cache.stats().misses > misses_before);
    return true;
}

TEST(cache_writeback_keeps_valid) {
    // writeback() should flush dirty data but keep the line valid
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write32(0x100u, 0x1111'1111u);
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x0000'0000u); // not yet in memory

    cache.writeback(0x100u);
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x1111'1111u); // now in memory

    // Line should still be cached (valid) — read should hit
    u64 hits_before = cache.stats().hits;
    cache.read32(0x100u);
    ASSERT_EQ(cache.stats().hits, hits_before + 1);
    return true;
}

TEST(cache_flush_invalidates) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write32(0x100u, 0x2222'2222u);
    cache.flush(0x100u);

    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x2222'2222u); // written back
    ASSERT(!cache.lookup(0x100u));                          // invalidated

    // Read should miss
    ASSERT_EQ(cache.read32(0x100u).cycles, simple_config().hit_latency + simple_config().miss_penalty);
    return true;
}

TEST(cache_writeback_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write32(0x100u, 0x1111'1111u);
    cache.write32(0x200u, 0x2222'2222u);
    cache.write32(0x300u, 0x3333'3333u);

    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x0000'0000u);

    cache.writeback_all();

    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0x1111'1111u);
    ASSERT_HEX_EQ(mem->read32(0x200u).value, 0x2222'2222u);
    ASSERT_HEX_EQ(mem->read32(0x300u).value, 0x3333'3333u);

    // Lines should still be valid (writeback, not flush)
    ASSERT(cache.lookup(0x100u));
    ASSERT(cache.lookup(0x200u));
    return true;
}

TEST(cache_load_bypasses_and_invalidates) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    // Load value into cache
    cache.read32(0x100u);
    ASSERT(cache.lookup(0x100u));

    // Bulk load new data directly to memory — should invalidate cached copy
    u8 data[4] = {0xEFu, 0xBEu, 0xADu, 0xDEu};
    cache.load(0x100u, std::span<const u8>(data, 4));

    ASSERT(!cache.lookup(0x100u));
    ASSERT_HEX_EQ(mem->read32(0x100u).value, 0xDEAD'BEEFu);

    // Reading again should re-fetch from memory
    ASSERT_HEX_EQ(cache.read32(0x100u).value, 0xDEAD'BEEFu);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Cache hierarchy
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_hierarchy_basic) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    mem->write32(0x1000u, 0xDEAD'C0DEu);

    CacheHierarchyConfig config;
    config.levels = {
        {.size_bytes = 1024, .line_size = 64, .associativity = 2,
         .hit_latency = 1, .miss_penalty = 5},
        {.size_bytes = 4096, .line_size = 64, .associativity = 4,
         .hit_latency = 5, .miss_penalty = 50},
    };

    CacheHierarchy hierarchy(config, mem);
 
    auto r1 = hierarchy.read32(0x1000u);
    ASSERT_HEX_EQ(r1.value, 0xDEAD'C0DEu);
    ASSERT_EQ(hierarchy.level(0).stats().misses, 1u);

    auto r2 = hierarchy.read32(0x1000u);
    ASSERT_HEX_EQ(r2.value, 0xDEAD'C0DEu);
    ASSERT_EQ(hierarchy.level(0).stats().hits, 1u);
    return true;
}

TEST(cache_hierarchy_l2_hit) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    mem->write32(0x1000u, 0x1234'5678u);

    CacheHierarchyConfig config;
    config.levels = {
        {.size_bytes = 256, .line_size = 64, .associativity = 2,
         .hit_latency = 1, .miss_penalty = 5},
        {.size_bytes = 4096, .line_size = 64, .associativity = 4,
         .hit_latency = 5, .miss_penalty = 50},
    };

    CacheHierarchy hierarchy(config, mem);

    hierarchy.read32(0x1000u);

    // Thrash L1 (256B = 2 sets × 2 ways × 64B) to evict 0x1000
    u32 l1_stride = 2 * 64;  // num_sets * line_size for L1
    for (int i = 0; i < 4; ++i) {
        hierarchy.read32(0x2000u + static_cast<u32>(i) * l1_stride);
    }

    u64 l1_misses = hierarchy.level(0).stats().misses;
    hierarchy.read32(0x1000u);
    ASSERT(hierarchy.level(0).stats().misses > l1_misses); // L1 miss
    return true;
}

TEST(cache_hierarchy_flush_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    CacheHierarchyConfig config = CacheHierarchyConfig::typical_2level();
    CacheHierarchy hierarchy(config, mem);

    hierarchy.write32(0x1000u, 0xAAAA'AAAAu);
    hierarchy.flush_all();

    ASSERT_HEX_EQ(mem->read32(0x1000u).value, 0xAAAA'AAAAu);
    ASSERT(!hierarchy.level(0).lookup(0x1000u));
    return true;
}

TEST(cache_hierarchy_stats) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    CacheHierarchyConfig config = CacheHierarchyConfig::typical_2level();
    CacheHierarchy hierarchy(config, mem);

    for (int i = 0; i < 100; ++i) hierarchy.read32(0x1000u);

    auto hs = hierarchy.get_stats();
    ASSERT_EQ(hs.level_stats.size(), 2u);
    ASSERT_EQ(hs.total_accesses, 100u);
    ASSERT(hs.level_stats[0].hit_rate() > 0.95);

    hierarchy.reset_stats();
    ASSERT_EQ(hierarchy.level(0).stats().reads, 0u);
    return true;
}

// ── Three-level hierarchy ──────────────────────────────────────────────

static CacheHierarchyConfig three_level_config()
{
    return {
        .levels = {
            CacheConfig{
                .size_bytes = 1024, .line_size = 64, .associativity = 4,
                .hit_latency = 1, .miss_penalty = 5,
            },
            CacheConfig{
                .size_bytes = 4096, .line_size = 64, .associativity = 8,
                .hit_latency = 5, .miss_penalty = 20,
            },
            CacheConfig{
                .size_bytes = 16384, .line_size = 64, .associativity = 16,
                .hit_latency = 20, .miss_penalty = 100,
            },
        }
    };
}

TEST(cache_hierarchy_3level_read_miss_populates_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0xDEAD'BEEFu);
    CacheHierarchy hier(three_level_config(), mem);

    // Cold miss: data travels from memory through L3, L2, into L1
    auto r = hier.read32(0x100u);
    ASSERT(r.ok);
    ASSERT_EQ(r.value, 0xDEAD'BEEFu);

    // All three levels should now hold the line
    ASSERT(hier.level(0).lookup(0x100u));
    ASSERT(hier.level(1).lookup(0x100u));
    ASSERT(hier.level(2).lookup(0x100u));
    return true;
}

TEST(cache_hierarchy_3level_l1_hit) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x200u, 42);
    CacheHierarchy hier(three_level_config(), mem);

    hier.read32(0x200u); // cold miss
    auto r = hier.read32(0x200u); // should hit L1
    ASSERT(r.ok);
    ASSERT_EQ(r.value, 42u);
    ASSERT_EQ(r.cycles, 1u); // L1 hit latency
    return true;
}

TEST(cache_hierarchy_3level_writeback_propagates) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x300u, 0);
    CacheHierarchy hier(three_level_config(), mem);

    hier.read32(0x300u);
    hier.write32(0x300u, 0xCAFE'BABEu);

    // Data is dirty in L1, not yet in main memory
    ASSERT_EQ(mem->read32(0x300u).value, 0u);

    // Flush pushes dirty data through all levels to memory
    hier.flush_all();
    ASSERT_EQ(mem->read32(0x300u).value, 0xCAFE'BABEu);
    return true;
}

TEST(cache_hierarchy_3level_l1_eviction_lands_in_l2) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000u);
    CacheHierarchy hier(three_level_config(), mem);

    auto cfg = three_level_config().levels[0];
    u32 stride = cfg.num_sets() * cfg.line_size;

    // Fill one L1 set completely, then access one more to evict
    for (u32 i = 0; i <= cfg.associativity; ++i)
        hier.read32(i * stride);

    // The evicted line should still be in L2
    ASSERT(hier.level(0).stats().evictions >= 1u);

    // Re-read the first line: should miss L1 but hit L2
    u64 l2_hits_before = hier.level(1).stats().hits;
    hier.read32(0);
    ASSERT(hier.level(1).stats().hits > l2_hits_before);
    return true;
}

TEST(cache_hierarchy_3level_stats_aggregate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    CacheHierarchy hier(three_level_config(), mem);

    for (u32 i = 0; i < 100; ++i)
        hier.read32(i * 4);

    auto hs = hier.get_stats();
    ASSERT_EQ(hs.level_stats.size(), 3u);
    ASSERT_EQ(hs.total_accesses, hs.level_stats[0].reads 
                               + hs.level_stats[0].writes);
    ASSERT(hs.total_latency > 0u);

    hier.reset_stats();
    auto hs2 = hier.get_stats();
    ASSERT_EQ(hs2.level_stats[0].reads, 0u);
    return true;
}

TEST(cache_hierarchy_3level_invalidate_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    CacheHierarchy hier(three_level_config(), mem);

    hier.read32(0x400u);
    ASSERT(hier.level(0).lookup(0x400u));

    hier.invalidate_all();
    ASSERT(!hier.level(0).lookup(0x400u));
    ASSERT(!hier.level(1).lookup(0x400u));
    ASSERT(!hier.level(2).lookup(0x400u));
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Edge cases
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_byte_access) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0x4433'2211u);
    Cache cache(simple_config(), mem);

    ASSERT_HEX_EQ(cache.read8(0x100u).value, 0x11u);
    ASSERT_HEX_EQ(cache.read8(0x101u).value, 0x22u);
    ASSERT_HEX_EQ(cache.read8(0x102u).value, 0x33u);
    ASSERT_HEX_EQ(cache.read8(0x103u).value, 0x44u);

    ASSERT_EQ(cache.stats().misses, 1u);
    ASSERT_EQ(cache.stats().hits, 3u);
    return true;
}

TEST(cache_half_access) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    mem->write32(0x100u, 0xBBBB'AAAAu);
    Cache cache(simple_config(), mem);

    ASSERT_HEX_EQ(cache.read16(0x100u).value, 0xAAAAu);
    ASSERT_HEX_EQ(cache.read16(0x102u).value, 0xBBBBu);
    return true;
}

TEST(cache_cross_line_read) {
    // Place a 4-byte value straddling two cache lines
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    // Line boundary at 0x40 (offset 62–65 straddles lines 0x00 and 0x40)
    mem->write8(0x3Eu, 0x78u);
    mem->write8(0x3Fu, 0x56u);
    mem->write8(0x40u, 0x34u);
    mem->write8(0x41u, 0x12u);

    Cache cache(simple_config(), mem);

    auto r = cache.read32(0x3Eu);
    ASSERT_HEX_EQ(r.value, 0x1234'5678u);
    // Should have caused 2 misses (two different lines)
    ASSERT_EQ(cache.stats().misses, 2u);
    return true;
}

TEST(cache_cross_line_write) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write32(0x3Eu, 0xAABB'CCDDu);

    // Read back the bytes individually to verify correct split
    ASSERT_HEX_EQ(cache.read8(0x3Eu).value, 0xDDu);
    ASSERT_HEX_EQ(cache.read8(0x3Fu).value, 0xCCu);
    ASSERT_HEX_EQ(cache.read8(0x40u).value, 0xBBu);
    ASSERT_HEX_EQ(cache.read8(0x41u).value, 0xAAu);
    return true;
}

TEST(cache_unaligned_in_line) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.write8(0x101u, 0xABu);
    cache.write8(0x102u, 0xCDu);

    ASSERT_HEX_EQ(cache.read8(0x101u).value, 0xABu);
    ASSERT_HEX_EQ(cache.read8(0x102u).value, 0xCDu);
    return true;
}

TEST(cache_config_validation) {
    ASSERT(simple_config().valid());

    // Non-power-of-2 line size
    CacheConfig bad1 = simple_config();
    bad1.line_size = 48;
    ASSERT(!bad1.valid());

    // Size not divisible by (line_size × associativity)
    CacheConfig bad2 = simple_config();
    bad2.size_bytes = 100;
    ASSERT(!bad2.valid());

    // Zero associativity
    CacheConfig bad3 = simple_config();
    bad3.associativity = 0;
    ASSERT(!bad3.valid());

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  8. Statistics
// ═══════════════════════════════════════════════════════════════════════

TEST(cache_hit_rate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    for (int iter = 0; iter < 10; ++iter)
    {
        cache.read32(0x100u);
        cache.read32(0x200u);
        cache.read32(0x300u);
    }

    ASSERT_EQ(cache.stats().misses, 3u);
    ASSERT_EQ(cache.stats().hits, 27u);

    double hr = cache.stats().hit_rate();
    ASSERT(hr == 0.9);
    return true;
}

TEST(cache_latency_tracking) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    CacheConfig cfg = {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 2,
        .hit_latency   = 4,
        .miss_penalty  = 100,
    };
    Cache cache(cfg, mem);

    for (int i = 0; i < 10; ++i) cache.read32(0x100u);

    // 1 miss × (4+100) + 9 hits × 4 = 104 + 36 = 140
    ASSERT_EQ(cache.stats().total_latency, 140u);

    double avg = cache.stats().avg_latency();
    ASSERT(avg == 14.0);
    return true;
}

TEST(cache_stats_reset) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000u);
    Cache cache(simple_config(), mem);

    cache.read32(0x100u);
    ASSERT(cache.stats().reads > 0u);

    cache.stats().reset();

    ASSERT_EQ(cache.stats().reads, 0u);
    ASSERT_EQ(cache.stats().writes, 0u);
    ASSERT_EQ(cache.stats().hits, 0u);
    ASSERT_EQ(cache.stats().misses, 0u);
    ASSERT_EQ(cache.stats().total_latency, 0u);
    return true;
}
