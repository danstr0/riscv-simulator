/**
 * @file cache.hpp
 * @brief Parameterised cache hierarchy for cycle-accurate memory modelling.
 *
 * This provides a flexible, set-associative cache simulation designed for
 * Pareto-front architectural sweeps. It allows for detailed benchmarking
 * across size, associativity, and replacement/write policies.
 *
 * @section arch_sec Architecture
 * - **Cache**: Represents a single level (L1, L2, L3) with flat-vector storage.
 * - **CacheHierarchy**: A chain of Cache levels ending in a backing memory.
 * Implements the @ref Memory interface for drop-in CPU integration.
 *
 * @section address_sec Address Decomposition
 * For an address $A$, the components are derived via:
 * - **Block Offset**: $A \pmod{\text{LineSize}}$
 * - **Set Index**: $\lfloor A / \text{LineSize} \rfloor \pmod{\text{NumSets}}$
 * - **Tag**: $\lfloor A / (\text{LineSize} \times \text{NumSets}) \rfloor$
 *
 * @section policy_sec Parameterization Axes
 * - **Geometry**: Total size, line size, associativity.
 * - **Replacement**: LRU, Pseudo-LRU (Tree-based), MRU, FIFO, Random.
 * - **Write Policy**: Write-Back vs. Write-Through.
 * - **Allocation**: Write-Allocate vs. No-Write-Allocate.
 *
 * @see Patterson, D. A., & Hennessy, J. L. "Computer Organization and Design: The
 * Hardware/Software Interface (RISC-V Edition)", Sections 5.3-5.4.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <bit>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace riscv {

// ── Policy enums ───────────────────────────────────────────────────────

enum class ReplacementPolicy : u8 {
    LRU,     ///< Least recently used.
    MRU,     ///< Most recently used.
    PLRU,    ///< Pseudo-LRU (Tree-based).
    RANDOM,  ///< Uniform random.
    FIFO,    ///< First-in first-out.
};

enum class WritePolicy: u8 {
    WRITE_BACK,     ///< Write to cache only; writeback on eviction.
    WRITE_THROUGH,  ///< Write to cache AND next level on every write.
};

enum class WriteAllocate : u8 {
    ALLOCATE,     ///< On write miss, fetch line into cache then write.
    NO_ALLOCATE,  ///< On write miss, write directly to next level.
};

enum class CacheType : u8 {
    UNIFIED,      ///< Serves both instructions and data.
    INSTRUCTION,  ///< I-cache (reads only in normal operation).
    DATA,         ///< D-cache (reads and writes).
};

// ── Cache configuration ────────────────────────────────────────────────

struct CacheConfig {
    u32 size_bytes;          ///< Total cache size.
    u32 line_size     = 64;  ///< Cache line size in bytes.
    u32 associativity = 8;   ///< N-way set associative.
    u32 hit_latency   = 4;   ///< Cycles for a hit.
    u32 miss_penalty  = 10;  ///< Additional cycles on a miss.

    ReplacementPolicy replacement = ReplacementPolicy::LRU;
    WritePolicy       write_pol   = WritePolicy::WRITE_BACK;
    WriteAllocate     write_alloc = WriteAllocate::ALLOCATE;
    CacheType         type        = CacheType::UNIFIED;

    /** @name Derived geometry */
    /** @{ */
    [[nodiscard]] constexpr u32 num_sets() const noexcept
    {
        return size_bytes / (line_size * associativity);
    }
    
    [[nodiscard]] constexpr u32 num_lines() const noexcept
    {
        return size_bytes / line_size;
    }
    /** @} */

    /**
     * @brief Validate that the configuration is self-consistent.
     *
     * All sizes must be powers of two; total size must be evenly
     * divisible by (line size × associativity).
     */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return line_size > 0
            && associativity > 0
            && size_bytes > 0
            && std::has_single_bit(line_size)
            && std::has_single_bit(num_sets())
            && (size_bytes % (line_size * associativity)) == 0;
    }

    /** @name Static presets for common architectures */
    /** @{ */
    static constexpr CacheConfig L1_typical()
    {
        return {
            .size_bytes    = 32 * 1024, 
            .line_size     = 64,
            .associativity = 8,
            .hit_latency   = 4,
            .miss_penalty  = 10,
        };
    }

    static constexpr CacheConfig L2_typical()
    {
        return {
            .size_bytes    = 256 * 1024,
            .line_size     = 64,
            .associativity = 8,
            .hit_latency   = 12,
            .miss_penalty  = 50,
        };
    }

    static constexpr CacheConfig L3_typical()
    {
        return {
            .size_bytes    = 8 * 1024 * 1024,
            .line_size     = 64,
            .associativity = 16,
            .hit_latency   = 40,
            .miss_penalty  = 100,
        };
    }

    /** @} */
};

// ── Cache line metadata ────────────────────────────────────────────────
// Payload data lives in Cache::storage_, not here.

struct CacheLine {
    bool        valid       = false;
    bool        dirty       = false;
    addr_t      tag         = 0;
    addr_t 	    line_addr   = 0;  ///< Full address of the start of this line.
    mutable u64 last_access = 0;  ///< Timestamp for LRU.
    mutable u32 fifo_order  = 0;  ///< Insertion order for FIFO.
};

// ── Cache statistics ───────────────────────────────────────────────────

struct CacheStats {
    u64 reads           = 0;
    u64 writes          = 0;
    u64 hits            = 0;
    u64 misses          = 0;
    u64 evictions       = 0;
    u64 dirty_evictions = 0;  ///< Evictions that required a writeback.
    u64 total_latency   = 0;  ///< Sum of per-access latencies.

    [[nodiscard]] double hit_rate() const noexcept
    {
        u64 total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / static_cast<double>(total) : 0.0;
    }

    [[nodiscard]] double miss_rate() const noexcept
    {
        return 1.0 - hit_rate();
    }

    [[nodiscard]] double avg_latency() const noexcept
    {
        u64 accesses = reads + writes;
        return accesses > 0 ? static_cast<double>(total_latency) / static_cast<double>(accesses) : 0.0;
    }

    void reset() noexcept { *this = CacheStats{}; }
};

// ── Single cache level ─────────────────────────────────────────────────

class Cache : public Memory {
public:
    Cache(CacheConfig, std::shared_ptr<Memory> next_level);

    /** @name Memory interface */
    /** @{ */
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write32(addr_t addr, u32 value) override;
    
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /** @} */

    /** @name Cache control */
    /** @{ */
    [[nodiscard]] bool lookup(addr_t addr) const;
    void invalidate(addr_t addr);
    void invalidate_all();
    void writeback(addr_t addr);  ///< Flush dirty line WITHOUT invalidating.
    void writeback_all();         ///< Flush all dirty lines WITHOUT invalidating.
    void flush(addr_t addr);      ///< Writeback + invalidate.
    void flush_all();             ///< Writeback + invalidate everything.
    /** @} */

    /** @name Accessors */
    /** @{ */
    [[nodiscard]] const CacheStats&  stats()  const noexcept { return stats_; }
    [[nodiscard]]       CacheStats&  stats()        noexcept { return stats_; }
    [[nodiscard]] const CacheConfig& config() const noexcept { return config_; }
    /** @} */

    /// Debug dump to stdout.
    void dump() const;

private:
    CacheConfig             config_;
    std::shared_ptr<Memory> next_level_;

    /*
     * Flat storage: all line payloads in one contiguous allocation.
     * Line i's data starts at storage_[i * config_.line_size].
     */
    mutable std::vector<u8>                     storage_;
    mutable std::vector<std::vector<CacheLine>> sets_;

    mutable CacheStats stats_;
    mutable u64        access_counter_ = 0;

    /* Per-set FIFO counters (only used when replacement == FIFO). */
    mutable std::vector<u32> fifo_counters_;

    /* Per-set PLRU tree bits (only used with replacement == PLRU) */
    mutable std::vector<std::vector<u8>> plru_bits_;

    /** @name Address decomposition */
    /** @{ */
    [[nodiscard]] u32    offset_of(addr_t addr)    const noexcept;
    [[nodiscard]] u32    index_of(addr_t addr)     const noexcept;
    [[nodiscard]] addr_t tag_of(addr_t addr)       const noexcept;
    [[nodiscard]] addr_t line_addr_of(addr_t addr) const noexcept;
    /** @} */


    /** @name Internal helpers */
    /** @{ */

    /// Get a pointer into storage_ for the given set + way.
    [[nodiscard]] u8*       line_data(u32 set_idx, u32 way) noexcept;
    [[nodiscard]] const u8* line_data(u32 set_idx, u32 way) const noexcept;
    
    struct LineRef {
        CacheLine* meta;
        u8*        data;
    };

    struct ConstLineRef {
        const CacheLine* meta;
        const u8*        data;
    };
    
    [[nodiscard]] std::optional<LineRef>      find_line(addr_t addr);
    [[nodiscard]] std::optional<ConstLineRef> find_line(addr_t addr) const;

    LineRef  allocate_line(addr_t addr);
    void     evict_line(u32 set_idx, u32 way);
    void     writeback_line(u32 set_idx, u32 way);
    void     fetch_line(addr_t line_addr, u32 set_idx, u32 way);
    [[nodiscard]] u32 find_victim(u32 set_idx) const;

    /**
     * @brief Update replacement metadata after an access to the given way.
     * Called on every hit and after every allocation.
     */
    void touch_way(u32 set_idx, u32 way) const;

    [[nodiscard]] MemoryResult read_bytes(addr_t addr, u32 size) const;
    MemoryResult write_bytes(addr_t addr, const u8* bytes, u32 size);

    /// Global line index for flat storage: set_idx * associativity + way.
    [[nodiscard]] u32 line_index(u32 set_idx, u32 way) const noexcept
    {
        return set_idx * config_.associativity + way;
    }

    /** @} */
};

// ── Cache hierarchy ────────────────────────────────────────────────────

struct CacheHierarchyConfig {
    std::vector<CacheConfig> levels;

    static CacheHierarchyConfig typical_2level()
    {
        return {
            .levels = {
                CacheConfig::L1_typical(),
                CacheConfig::L2_typical()
            }
        };
    }

    static CacheHierarchyConfig typical_3level()
    {
        return {
            .levels = {
                CacheConfig::L1_typical(),
                CacheConfig::L2_typical(),
                CacheConfig::L3_typical(),
            }
        };
    }
};

/**
 * @brief Chains multiple @ref Cache levels backed by a main memory.
 *
 * All accesses enter through L1. On an L1 miss, L1's next_level (L2)
 * is consulted, and so on. The hierarchy itself implements the Memory
 * interface so the CPU can use it as a drop-in replacement for
 * FlatMemory or MMIOBus.
 *
 * Latency model: the latency returned by a read/write is the sum of
 * hit_latency at every level that was hit, plus miss_penalty at every
 * level that missed.
 */
class CacheHierarchy : public Memory {
public:
    CacheHierarchy(CacheHierarchyConfig config, std::shared_ptr<Memory> main_memory);

    /** @name Memory interface */
    /** @{ */
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write32(addr_t addr, u32 value) override;

    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /** @} */

    /** @name Cache control */
    /** @{ */
    void flush_all();       ///< Writeback and invalidate all levels.
    void invalidate_all();  ///< Invalidate without writeback.
    /** @} */

    /** @name Introspection */
    /** @{ */
    [[nodiscard]] size_t       num_levels() const noexcept { return caches_.size(); }
    [[nodiscard]]       Cache& level(size_t n)       { return *caches_.at(n); }
    [[nodiscard]] const Cache& level(size_t n) const { return *caches_.at(n); }

    struct HierarchyStats {
        std::vector<CacheStats> level_stats;
        u64 total_accesses = 0;
        u64 total_latency  = 0;

        [[nodiscard]] double avg_latency() const noexcept
        {
            return total_accesses > 0
                ? static_cast<double>(total_latency) / static_cast<double>(total_accesses)
                : 0.0;
        }
    };

    [[nodiscard]] HierarchyStats get_stats() const;
    void reset_stats();
    /** @} */

private:
    std::vector<std::shared_ptr<Cache>> caches_;
    std::shared_ptr<Memory>             main_memory_;
};

} // namespace riscv
