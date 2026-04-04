/**
 * @file coherence.hpp
 * @brief Directory-Based MESI Cache Coherence Controller.
 *
 * This module implements a full-map directory coherence protocol. It serves as
 * the serialization point for all L1 cache misses.
 *
 * @section mesi_protocol The MESI Protocol
 * This implements the standard four-state MESI protocol:
 * - **Modified (M)**: Line is present only in this cache and is dirty.
 * - **Exclusive (E)**: Line is present only in this cache and is clean.
 * - **Shared (S)**: Line may be present in other caches and is clean.
 * - **Invalid (I)**: Line is not present in this cache.
 *
 * @section directory_arch Directory Architecture
 * Unlike bus-based "snoopy" protocols that rely on broadcast, this controller
 * uses a directory (shadow tags) to track the state of every line in every L1.
 * This reduces interconnect traffic by only sending snoop/invalidation messages
 * to cores known to hold a copy of the data.
 *
 * @see "Computer Organization and Design: The Hardware/Software Interface 
 * (RISC-V Edition)" (Hennessy & Patterson), Chapter 5.12.
 */

#pragma once

#include "cache.hpp"
#include "types.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace riscv {

/**
 * @brief Architectural Coherence States.
 */
enum class MESIState : u8 {
    Invalid   = 0,  ///< Line is not cached.
    Shared    = 1,  ///< Line is clean; other cores may also hold it in 'S'.
    Exclusive = 2,  ///< Line is clean; this core is the sole owner.
    Modified  = 3,  ///< Line is dirty; this core is the sole owner.
};

/**
 * @brief Coherence Transaction Statistics.
 */
struct CoherenceStats {
    u64 read_misses        = 0;  ///< Total L1 read misses processed.
    u64 write_misses       = 0;  ///< Total L1 write misses/upgrades processed.
    u64 upgrades           = 0;  ///< S -> M transitions (ownership acquisitions).
    u64 invalidations_sent = 0;  ///< Number of remote invalidation messages.
    u64 writebacks_forced  = 0;  ///< M -> S/I transitions requiring a data writeback.
    u64 shared_transfers   = 0;  ///< Cache-to-cache "intervention" transfers.
    u64 l2_fetches         = 0;  ///< Requests satisfied by the shared L2/RAM.

    void reset() noexcept { *this = CoherenceStats{}; }
};

/**
 * @brief Shadow Tag Directory Entry.
 *
 * Tracks the coherence state for a single cache line across all participating cores.
 */
struct DirectoryEntry {
    /** @brief State per core ID. */
    std::vector<MESIState> core_states;

    explicit DirectoryEntry(u32 num_cores)
        : core_states(num_cores, MESIState::Invalid) {}
};

/**
 * @brief Centralized Coherence Controller (Directory).
 */
class CoherenceController {
public:
    CoherenceController(std::vector<Cache*> l1_caches,
                        std::shared_ptr<Memory> shared_mem,
                        u32 line_size = 64);

    /** @name Core Request Interface */
    /** @{ */

    /**
     * @brief Resolves a Read Miss.
     * * If the line is 'M' in another core, that core is forced to write back
     * and transition to 'S'. Data is then supplied to the requesting core.
     *
     * @return Total transaction latency in cycles.
     */
    u32 handle_read_miss(u32 core_id, addr_t addr);

    /**
     * @brief Resolves a Write Miss or Upgrade.
     *
     * @return Total transaction latency in cycles.
     */
    u32 handle_write_miss(u32 core_id, addr_t addr);

    /** @brief Notifies the controller that a core is evicting a line. */
    void notify_eviction(u32 core_id, addr_t addr);
    /** @} */

    /** @name Statistics */
    /** @{ */
    [[nodiscard]] const CoherenceStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_.reset(); }
    /** @} */

    /** @name State Inspection */
    /** @{ */
    [[nodiscard]] MESIState line_state(u32 core_id, addr_t addr) const;
    [[nodiscard]] u32 num_cores()         const noexcept { return static_cast<u32>(l1_caches_.size()); }
    [[nodiscard]] u32 num_tracked_lines() const noexcept { return static_cast<u32>(directory_.size()); }
    /** @} */

private:
    std::vector<Cache*>     l1_caches_;
    std::shared_ptr<Memory> shared_mem_;
    u32                     line_size_;
    u32                     num_cores_;

    /** @brief The coherence directory. */
    std::unordered_map<addr_t, DirectoryEntry> directory_;

    CoherenceStats stats_;

    /** @brief Gets/creates a directory entry for the line containing @p addr. */
    DirectoryEntry& get_entry(addr_t addr);

    /** @brief Aligns address to line boundary. */
    [[nodiscard]] addr_t line_align(addr_t addr) const noexcept
    {
        return addr & ~static_cast<addr_t>(line_size_ - 1);
    }

    /**
     * @brief Invalidates all copies of a line in cores other than @p requesting_core.
     *
     * @return Additional latency for the snoops.
     */
    u32 invalidate_others(u32 requesting_core, addr_t line_addr, DirectoryEntry& entry);

    /**
     * @brief Finds a core that has the line in Modified state (if any).
     *
     * @return Core ID or -1.
     */
    [[nodiscard]] i32 find_modified_owner(const DirectoryEntry& entry) const;

    /** @brief Finds any core that has the line in Shared or Exclusive state. */
    [[nodiscard]] i32 find_any_sharer(const DirectoryEntry& entry, u32 exclude_core) const;

    /** @name Latency Constants */
    /** @{ */
    static constexpr u32 kSnoopLatency        = 2;  ///< Cycles per snoop message.
    static constexpr u32 kTransferLatency     = 5;  ///< Cycles for cache-to-cache transfer.
    static constexpr u32 kInvalidationLatency = 2;  ///< Cycles per invalidation.
    /** @} */
};

} // namespace riscv
