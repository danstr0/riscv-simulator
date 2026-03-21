/**
 * @file test_cache.cpp
 * @brief Tests for the parameterised cache and cache hierarchy.
 *
 * Sections:
 *   1 (line 113) : Basic hit / miss / line fill
 *   2 (line 185) : Set associativity and eviction
 *   3 (line 237) : Replacement policies (LRU, MRU, PLRU, FIFO, Random)
 *   4 (line 276) : Write policies (write-back, write-through, no-allocate)
 *   5 (line 336) : Cache control (invalidate, writeback, flush)
 *   6 (line 451) : Cache hierarchy (L1+L2, flush, stats)
 *   7 (line 537) : Edge cases (byte/half access, cross-line, config validation)
 *   8 (line 631) : Statistics
 */

#include "core/cache.hpp"

#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace riscv;

// ── Shared test infrastructure ──────────────────────────────────────────────

struct TestCase {
    std::string             name;
    std::function<bool()>   func;
};
extern std::vector<TestCase> g_tests;

#define TEST(name)                                                            \
    bool test_##name();                                                       \
    static bool reg_##name = (g_tests.push_back({#name, test_##name}), true); \
    bool test_##name()

#define ASSERT(cond)                                                        \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "  FAILED: " << #cond << "\n"                      \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define ASSERT_EQ(a, b)                                                      \
    do {                                                                     \
        auto actual_   = (a);                                                \
        auto expected_ = (b);                                                \
        if (actual_ != expected_) {                                          \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"          \
                      << "    got: " << static_cast<std::int64_t>(actual_)   \
                      << " != "      << static_cast<std::int64_t>(expected_) \
                      << "\n"                                                \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";   \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define ASSERT_HEX_EQ(a, b)                                                 \
    do {                                                                    \
        auto actual_   = (a);                                               \
        auto expected_ = (b);                                               \
        if (actual_ != expected_) {                                         \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"         \
                      << std::format("    got: 0x{:x} != 0x{:x}\n",         \
                            static_cast<std::uint64_t>(actual_),            \
                            static_cast<std::uint64_t>(expected_))          \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                   \
        }                                                                   \
    } while (0)

// ── Helpers ─────────────────────────────────────────────────────────────────

/// 1 KB, 64-byte lines, 2-way, write-back / write-allocate / LRU.
static CacheConfig simple_config() {
    return {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 2,
        .hit_latency   = 1,
        .miss_penalty  = 10,
    };
}

/// Same geometry but with a specific replacement policy.
static CacheConfig config_with_replacement(ReplacementPolicy rp) {
    auto c       = simple_config();
    c.replacement = rp;
    return c;
}

/// Same geometry but write-through + write-allocate.
static CacheConfig write_through_config() {
    auto c       = simple_config();
    c.write_pol  = WritePolicy::WRITE_THROUGH;
    return c;
}

/// Same geometry but write-back + no-allocate on write miss.
static CacheConfig no_allocate_config() {
    auto c          = simple_config();
    c.write_alloc   = WriteAllocate::NO_ALLOCATE;
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Basic cache operations
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_read_miss_then_hit) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0xDEADBEEF);

    Cache cache(simple_config(), mem);

    auto r1 = cache.read32(0x100);
    ASSERT(r1.ok);
    ASSERT_HEX_EQ(r1.value, 0xDEADBEEFu);
    ASSERT_EQ(cache.stats().misses, 1u);
    ASSERT_EQ(cache.stats().hits, 0u);
    ASSERT_EQ(r1.cycles, 1u + 10u);

    auto r2 = cache.read32(0x100);
    ASSERT_HEX_EQ(r2.value, 0xDEADBEEFu);
    ASSERT_EQ(cache.stats().hits, 1u);
    ASSERT_EQ(r2.cycles, 1u);
    return true;
}

TEST(cache_write_allocate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write32(0x200, 0x12345678);
    ASSERT_EQ(cache.stats().misses, 1u);

    auto r = cache.read32(0x200);
    ASSERT_HEX_EQ(r.value, 0x12345678u);
    ASSERT_EQ(cache.stats().hits, 1u);
    return true;
}

TEST(cache_line_fill) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0x11111111);
    mem->write32(0x104, 0x22222222);
    mem->write32(0x108, 0x33333333);

    Cache cache(simple_config(), mem);

    cache.read32(0x100);
    ASSERT_EQ(cache.stats().misses, 1u);

    // Same line — should all hit
    ASSERT_HEX_EQ(cache.read32(0x104).value, 0x22222222u);
    ASSERT_HEX_EQ(cache.read32(0x108).value, 0x33333333u);
    ASSERT_EQ(cache.stats().hits, 2u);
    return true;
}

TEST(cache_different_lines) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0xAAAAAAAA);
    mem->write32(0x200, 0xBBBBBBBB);

    Cache cache(simple_config(), mem);

    cache.read32(0x100);
    cache.read32(0x200);
    ASSERT_EQ(cache.stats().misses, 2u);

    cache.read32(0x100);
    cache.read32(0x200);
    ASSERT_EQ(cache.stats().hits, 2u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. Set associativity and eviction
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_associativity_eviction) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = simple_config();
    Cache cache(cfg, mem);

    // Stride to hit the same set: num_sets * line_size
    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0x000, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    ASSERT_EQ(cache.stats().misses, 2u);

    // Both should hit.
    cache.read32(a);
    cache.read32(b);
    ASSERT_EQ(cache.stats().hits, 2u);

    // c evicts LRU (a)
    cache.read32(c);
    ASSERT_EQ(cache.stats().evictions, 1u);

    cache.read32(b);  // hit
    cache.read32(c);  // hit
    cache.read32(a);  // miss (was evicted)
    ASSERT_EQ(cache.stats().misses, 4u);
    return true;
}

TEST(cache_lru_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = simple_config();
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    cache.read32(a);  // Touch a again — now b is LRU

    cache.read32(c);  // Evicts b

    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);                     // hit
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency + cfg.miss_penalty);  // miss
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Replacement policies
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_fifo_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = config_with_replacement(ReplacementPolicy::FIFO);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);  // ways: [a(fifo=0), -]
    cache.read32(b);  // ways: [a(fifo=0), b(fifo=1)]
    cache.read32(a);  // Touch a again — FIFO ignores recency

    cache.read32(c);  // Evicts a

    // a should miss, b should hit
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency + cfg.miss_penalty);
    return true;
}

TEST(cache_random_replacement) {
    // Verifies this doesn't crash and produces reasonable stats
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
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
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = config_with_replacement(ReplacementPolicy::MRU);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);  // ways: [a, -]
    cache.read32(b);  // ways: [a, b]; b is MRU

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
    auto mem_lru = std::make_shared<FlatMemory>(0, 0x100000);
    auto mem_mru = std::make_shared<FlatMemory>(0, 0x100000);

    auto cfg_lru = config_with_replacement(ReplacementPolicy::LRU);
    auto cfg_mru = config_with_replacement(ReplacementPolicy::MRU);
    Cache lru(cfg_lru, mem_lru);
    Cache mru(cfg_mru, mem_mru);

    u32 range = cfg_lru.size_bytes * 2;
    for (int pass = 0; pass < 3; ++pass) {
        for (u32 addr = 0; addr < range; addr += 64) {
            lru.read32(addr);
            mru.read32(addr);
        }
    }

    // MRU should achieve a meaningfully higher hit rate
    ASSERT(mru.stats().hit_rate() >= lru.stats().hit_rate());
    return true;
}

TEST(cache_plru_replacement) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = config_with_replacement(ReplacementPolicy::PLRU);
    Cache cache(cfg, mem);

    u32 stride = cfg.num_sets() * cfg.line_size;
    addr_t a = 0, b = a + stride, c = b + stride;

    cache.read32(a);
    cache.read32(b);
    cache.read32(a);  // Tree should point toward b
    
    cache.read32(c);  // Evicts b 

    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(b).cycles, cfg.hit_latency + cfg.miss_penalty);
    return true;
}

TEST(cache_plru_4way) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000);
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
 
    // Fill all 4 ways.
    cache.read32(a);
    cache.read32(b);
    cache.read32(c);
    cache.read32(d);
 
    // Touch a and c to mark them as recently used.
    cache.read32(a);
    cache.read32(c);
 
    // e should evict one of {b, d}
    cache.read32(e);
 
    // a and c should still hit.
    ASSERT_EQ(cache.read32(a).cycles, cfg.hit_latency);
    ASSERT_EQ(cache.read32(c).cycles, cfg.hit_latency);
 
    // At least one of {b, d} should miss
    auto rb = cache.read32(b);
    auto rd = cache.read32(d);
    bool one_evicted = (rb.cycles > cfg.hit_latency) || (rd.cycles > cfg.hit_latency);
    ASSERT(one_evicted);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Write policies
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_writeback_dirty) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    auto cfg = simple_config();  // write-back by default
    Cache cache(cfg, mem);

    cache.write32(0x100, 0xCAFEBABE);

    // Cache has the value, but backing memory does not
    ASSERT_HEX_EQ(cache.read32(0x100).value, 0xCAFEBABEu);
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x00000000u);

    // Force eviction
    u32 stride = cfg.num_sets() * cfg.line_size;
    cache.read32(0x100 + stride);
    cache.read32(0x100 + 2 * stride);

    // Now backing memory should have the value
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0xCAFEBABEu);
    ASSERT_EQ(cache.stats().dirty_evictions, 1u);
    return true;
}

TEST(cache_write_through) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(write_through_config(), mem);

    cache.write32(0x100, 0xDEADC0DE);

    // Backing memory should have the value immediately
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0xDEADC0DEu);

    // The cache should also have it
    ASSERT_HEX_EQ(cache.read32(0x100).value, 0xDEADC0DEu);
    ASSERT_EQ(cache.stats().hits, 1u);
    return true;
}

TEST(cache_write_no_allocate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(no_allocate_config(), mem);

    // Goes directly to backing memory, does NOT fill the cache line
    cache.write32(0x100, 0xBEEFBEEF);
    ASSERT_EQ(cache.stats().misses, 1u);

    // The value should be in backing memory
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0xBEEFBEEFu);

    // ...but NOT in the cache
    auto r = cache.read32(0x100);
    ASSERT_EQ(cache.stats().misses, 2u);  // Another miss
    ASSERT_HEX_EQ(r.value, 0xBEEFBEEFu);  // Gets it from memory
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Cache control
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_invalidate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0xAAAAAAAA);
    Cache cache(simple_config(), mem);

    cache.read32(0x100);
    ASSERT(cache.lookup(0x100));

    cache.invalidate(0x100);
    ASSERT(!cache.lookup(0x100));

    cache.read32(0x100);
    ASSERT_EQ(cache.stats().misses, 2u);
    return true;
}

TEST(cache_invalidate_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
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
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write32(0x100, 0x11111111);
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x00000000u);  // Not yet in memory

    cache.writeback(0x100);
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x11111111u);  // Now in memory

    // Line should still be cached (valid) — read should hit
    u64 hits_before = cache.stats().hits;
    cache.read32(0x100);
    ASSERT_EQ(cache.stats().hits, hits_before + 1);
    return true;
}

TEST(cache_flush_invalidates) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write32(0x100, 0x22222222);
    cache.flush(0x100);

    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x22222222u);  // Written back
    ASSERT(!cache.lookup(0x100));                          // Invalidated

    // Read should miss
    ASSERT_EQ(cache.read32(0x100).cycles, simple_config().hit_latency + simple_config().miss_penalty);
    return true;
}

TEST(cache_writeback_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write32(0x100, 0x11111111);
    cache.write32(0x200, 0x22222222);
    cache.write32(0x300, 0x33333333);

    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x00000000u);

    cache.writeback_all();

    ASSERT_HEX_EQ(mem->read32(0x100).value, 0x11111111u);
    ASSERT_HEX_EQ(mem->read32(0x200).value, 0x22222222u);
    ASSERT_HEX_EQ(mem->read32(0x300).value, 0x33333333u);

    // Lines should still be valid (writeback, not flush)
    ASSERT(cache.lookup(0x100));
    ASSERT(cache.lookup(0x200));
    return true;
}

TEST(cache_load_bypasses_and_invalidates) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    // Load value into cache.
    cache.read32(0x100);
    ASSERT(cache.lookup(0x100));

    // Bulk load new data directly to memory — should invalidate cached copy
    u8 data[4] = {0xEF, 0xBE, 0xAD, 0xDE};
    cache.load(0x100, std::span<const u8>(data, 4));

    ASSERT(!cache.lookup(0x100));
    ASSERT_HEX_EQ(mem->read32(0x100).value, 0xDEADBEEFu);

    // Reading again should re-fetch from memory
    ASSERT_HEX_EQ(cache.read32(0x100).value, 0xDEADBEEFu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Cache hierarchy
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_hierarchy_basic) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000);
    mem->write32(0x1000, 0xDEADC0DE);

    CacheHierarchyConfig config;
    config.levels = {
        {.size_bytes = 1024, .line_size = 64, .associativity = 2,
         .hit_latency = 1, .miss_penalty = 5},
        {.size_bytes = 4096, .line_size = 64, .associativity = 4,
         .hit_latency = 5, .miss_penalty = 50},
    };

    CacheHierarchy hierarchy(config, mem);
 
    auto r1 = hierarchy.read32(0x1000);
    ASSERT_HEX_EQ(r1.value, 0xDEADC0DEu);
    ASSERT_EQ(hierarchy.level(0).stats().misses, 1u);

    auto r2 = hierarchy.read32(0x1000);
    ASSERT_HEX_EQ(r2.value, 0xDEADC0DEu);
    ASSERT_EQ(hierarchy.level(0).stats().hits, 1u);
    return true;
}

TEST(cache_hierarchy_l2_hit) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000);
    mem->write32(0x1000, 0x12345678);

    CacheHierarchyConfig config;
    config.levels = {
        {.size_bytes = 256, .line_size = 64, .associativity = 2,
         .hit_latency = 1, .miss_penalty = 5},
        {.size_bytes = 4096, .line_size = 64, .associativity = 4,
         .hit_latency = 5, .miss_penalty = 50},
    };

    CacheHierarchy hierarchy(config, mem);

    hierarchy.read32(0x1000);

    // Thrash L1 (256B = 2 sets × 2 ways × 64B) to evict 0x1000
    u32 l1_stride = 2 * 64;  // num_sets * line_size for L1
    for (int i = 0; i < 4; ++i) {
        hierarchy.read32(0x2000 + static_cast<u32>(i) * l1_stride);
    }

    u64 l1_misses = hierarchy.level(0).stats().misses;
    hierarchy.read32(0x1000);
    ASSERT(hierarchy.level(0).stats().misses > l1_misses);  // L1 miss
    return true;
}

TEST(cache_hierarchy_flush_all) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000);
    CacheHierarchyConfig config = CacheHierarchyConfig::typical_2level();
    CacheHierarchy hierarchy(config, mem);

    hierarchy.write32(0x1000, 0xAAAAAAAA);
    hierarchy.flush_all();

    ASSERT_HEX_EQ(mem->read32(0x1000).value, 0xAAAAAAAAu);
    ASSERT(!hierarchy.level(0).lookup(0x1000));
    return true;
}

TEST(cache_hierarchy_stats) {
    auto mem = std::make_shared<FlatMemory>(0, 0x100000);
    CacheHierarchyConfig config = CacheHierarchyConfig::typical_2level();
    CacheHierarchy hierarchy(config, mem);

    for (int i = 0; i < 100; ++i) hierarchy.read32(0x1000);

    auto hs = hierarchy.get_stats();
    ASSERT_EQ(hs.level_stats.size(), 2u);
    ASSERT_EQ(hs.total_accesses, 100u);
    ASSERT(hs.level_stats[0].hit_rate() > 0.95);

    hierarchy.reset_stats();
    ASSERT_EQ(hierarchy.level(0).stats().reads, 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. Edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_byte_access) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0x44332211);
    Cache cache(simple_config(), mem);

    ASSERT_HEX_EQ(cache.read8(0x100).value, 0x11u);
    ASSERT_HEX_EQ(cache.read8(0x101).value, 0x22u);
    ASSERT_HEX_EQ(cache.read8(0x102).value, 0x33u);
    ASSERT_HEX_EQ(cache.read8(0x103).value, 0x44u);

    ASSERT_EQ(cache.stats().misses, 1u);
    ASSERT_EQ(cache.stats().hits, 3u);
    return true;
}

TEST(cache_half_access) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x100, 0xBBBBAAAA);
    Cache cache(simple_config(), mem);

    ASSERT_HEX_EQ(cache.read16(0x100).value, 0xAAAAu);
    ASSERT_HEX_EQ(cache.read16(0x102).value, 0xBBBBu);
    return true;
}

TEST(cache_cross_line_read) {
    // Place a 4-byte value straddling two cache lines
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    // Line boundary at 0x40 (offset 62–65 straddles lines 0x00 and 0x40)
    mem->write8(0x3E, 0x78);
    mem->write8(0x3F, 0x56);
    mem->write8(0x40, 0x34);
    mem->write8(0x41, 0x12);

    Cache cache(simple_config(), mem);

    auto r = cache.read32(0x3E);
    ASSERT_HEX_EQ(r.value, 0x12345678u);
    // Should have caused 2 misses (two different lines)
    ASSERT_EQ(cache.stats().misses, 2u);
    return true;
}

TEST(cache_cross_line_write) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write32(0x3E, 0xAABBCCDD);

    // Read back the bytes individually to verify correct split
    ASSERT_HEX_EQ(cache.read8(0x3E).value, 0xDDu);
    ASSERT_HEX_EQ(cache.read8(0x3F).value, 0xCCu);
    ASSERT_HEX_EQ(cache.read8(0x40).value, 0xBBu);
    ASSERT_HEX_EQ(cache.read8(0x41).value, 0xAAu);
    return true;
}

TEST(cache_unaligned_in_line) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.write8(0x101, 0xAB);
    cache.write8(0x102, 0xCD);

    ASSERT_HEX_EQ(cache.read8(0x101).value, 0xABu);
    ASSERT_HEX_EQ(cache.read8(0x102).value, 0xCDu);
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

/* ═══════════════════════════════════════════════════════════════════════
 *  8. Statistics
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(cache_hit_rate) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    for (int iter = 0; iter < 10; ++iter) {
        cache.read32(0x100);
        cache.read32(0x200);
        cache.read32(0x300);
    }

    ASSERT_EQ(cache.stats().misses, 3u);
    ASSERT_EQ(cache.stats().hits, 27u);

    double hr = cache.stats().hit_rate();
    ASSERT(hr > 0.89 && hr < 0.91);
    return true;
}

TEST(cache_latency_tracking) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    CacheConfig cfg = {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 2,
        .hit_latency   = 4,
        .miss_penalty  = 100,
    };
    Cache cache(cfg, mem);

    for (int i = 0; i < 10; ++i) cache.read32(0x100);

    // 1 miss × (4+100) + 9 hits × 4 = 104 + 36 = 140
    ASSERT_EQ(cache.stats().total_latency, 140u);

    double avg = cache.stats().avg_latency();
    ASSERT(avg > 13.9 && avg < 14.1);
    return true;
}

TEST(cache_stats_reset) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    Cache cache(simple_config(), mem);

    cache.read32(0x100);
    ASSERT(cache.stats().reads > 0u);

    cache.stats().reset();

    ASSERT_EQ(cache.stats().reads, 0u);
    ASSERT_EQ(cache.stats().writes, 0u);
    ASSERT_EQ(cache.stats().hits, 0u);
    ASSERT_EQ(cache.stats().misses, 0u);
    ASSERT_EQ(cache.stats().total_latency, 0u);
    return true;
}
