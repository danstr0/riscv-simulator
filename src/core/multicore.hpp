/**
 * @file multicore.hpp
 * @brief Synchronous Multi-Core SoC Simulation Environment.
 *
 * This module models a cycle-synchronous, shared-memory multiprocessor
 * with directory-based coherence and memory-mapped I/O devices.
 *
 * @section system_topology System Topology
 * The system utilizes a centralized directory-based coherence controller
 * to manage private L1 data caches. All cores share a unified L2/L3 cache
 * hierarchy before reaching the main memory (DRAM) model.
 *
 * @section interrupt_delivery Interrupt Delivery
 * Interrupts are routed via a Platform-Level Interrupt Controller (PLIC).
 * External devices (NIC, Timer) signal the PLIC, which performs priority
 * arbitration before asserting the @c MEIP bit in a specific core's @c mip
 * register.
 *
 * @section memory_map Memory Address Map
 * ┌──────────────┬─────────────┬────────────────────┬────────────┐
 * │ Base Address │ End Address │ Description        │ Attributes │
 * ├──────────────┼─────────────┼────────────────────┼────────────┤
 * │ 0x0000_0000  │ 0x0FFF_FFFF │ Main Memory (DRAM) │ Cached/RW  │
 * │ 0x1000_0000  │ 0x1000_0FFF │ PLIC               │ MMIO/RW    │
 * │ 0x1000_1000  │ 0x1000_100F │ Machine Timer      │ MMIO/RW    │
 * │ 0x1000_2000  │ 0x1000_207F │ NIC                │ MMIO/RW    │
 * └──────────────┴─────────────┴────────────────────┴────────────┘
 */

#pragma once

#include "cache.hpp"
#include "coherence.hpp"
#include "csr.hpp"
#include "memory.hpp"
#include "nic.hpp"
#include "pipeline.hpp"
#include "plic.hpp"
#include "timer.hpp"
#include "types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace riscv {

// ── Multi-core configuration ───────────────────────────────────────────

/**
 * @brief System Configuration.
 *
 * Defines the architectural parameters used to instantiate the SoC.
 */
struct MultiCoreConfig {
    /** @brief Number of symmetric multiprocessing cores. */
    u32 num_cores = 2;

    /** @brief Per-core execution pipeline configuration. */
    PipelineConfig pipeline = {};

    /** @brief Private L1 data cache configuration (per-core). */
    CacheConfig l1d = CacheConfig::L1_typical();

    /** @brief Shared L2 cache. */
    CacheConfig l2 = CacheConfig::L2_typical();

    /** @brief Shared L3 cache. Set size to 0 to bypass. */
    CacheConfig l3 = CacheConfig::L3_typical();

    /** @brief Physical memory footprint. Defaults to 256MB. */
    u32 main_memory_size = 256 * 1024 * 1024;  /* 256 MB */

    /** @brief PLIC interrupt source assignments. */
    u32 nic_plic_source   = 1;
    u32 timer_plic_source = 0;
};

/**
 * @brief System Statistics.
 */
struct MultiCoreStats {
    std::vector<PipelineStats> core_stats;
    CoherenceStats             coherence;
    std::vector<CacheStats>    l1d_stats;
    CacheStats                 l2_stats;
    CacheStats                 l3_stats;
};

/**
 * @brief Unified SoC simulation container and global execution driver.
 * * Calling
 * @c tick() advances the state of all pipelines, caches, and the coherence
 * directory by one global clock cycle.
 */
class MultiCoreCPU {
public:
    explicit MultiCoreCPU(MultiCoreConfig config = {});

    /** @name Core Access */
    /** @{ */
    [[nodiscard]] u32 num_cores() const noexcept { return config_.num_cores; }

    [[nodiscard]]       PipelinedCPU& core(u32 id)       { return *cores_.at(id); }
    [[nodiscard]] const PipelinedCPU& core(u32 id) const { return *cores_.at(id); }
    /** @} */

    /** @name Program Loading */
    /** @{ */
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /** @} */

    /** @name Execution Control */
    /** @{ */

    /**
     * @brief Advances every sub-component (Cores, PLIC, etc.) by 1 cycle.
     *
     * @return True if at least one core is still executing instructions.
     */
    bool tick();

    /** @brief Run the simulation for @p n cycles. */
    cycle_t run_cycles(cycle_t n);

    /** @brief Run the simulation until all cores have halted. */
    cycle_t run_until_all_halted(cycle_t max_cycles = 10'000'000);

    /** @brief Current global cycle count. */
    [[nodiscard]] cycle_t cycles() const noexcept { return cycle_count_; }
    /** @} */

    /** @name Device Access */
    /** @{ */
    [[nodiscard]] PLIC&   plic()        noexcept { return plic_; }
    [[nodiscard]] Timer&  timer()       noexcept { return timer_; }
    [[nodiscard]] Memory& main_memory() noexcept { return *main_mem_; }

    /**
     * @brief Attach a NIC to the system.
     *
     * Maps into the system's address space and wires its line interrupt to the PLIC.
     */
    void attach_nic(std::shared_ptr<NIC> nic, addr_t mmio_base = 0x1000'2000);

    [[nodiscard]] NIC* nic() noexcept { return nic_.get(); }
    /** @} */

    /** @name Cache Hierarchy Access */
    /** @{ */
    [[nodiscard]] Cache& l1d(u32 core_id) { return *l1d_caches_.at(core_id); }
    [[nodiscard]] Cache& l2()             { return *l2_cache_; }
    [[nodiscard]] Cache* l3()             { return l3_cache_.get(); }
    [[nodiscard]] CoherenceController& coherence() { return *coherence_; }
    /** @} */

    /** @name Statistics */
    /** @{ */
    [[nodiscard]] MultiCoreStats get_stats() const;
    void reset_stats();
    /** @} */

    /** @name Configuration */
    /** @{ */
    [[nodiscard]] const MultiCoreConfig& config() const noexcept { return config_; }

    /** @brief Set each core's starting PC. */
    void set_core_pc(u32 core_id, addr_t pc);

    /** @brief Set a per-core register. */
    void set_core_reg(u32 core_id, reg_idx_t r, u32 value);

    void reset();
    /** @} */

private:
    MultiCoreConfig config_;

    /** @name Memory Hierarchy */
    /** @{ */
    std::shared_ptr<FlatMemory> main_mem_;
    std::shared_ptr<Cache>      l3_cache_;  ///< Shared L3 (may be null).
    std::shared_ptr<Cache>      l2_cache_;  ///< Shared L2.
    std::vector<std::shared_ptr<Cache>> l1d_caches_;  ///< Per-core L1 data.

    std::unique_ptr<CoherenceController> coherence_;

    std::vector<std::unique_ptr<PipelinedCPU>> cores_;

    /** @name Devices */
    /** @{ */
    PLIC                 plic_;
    Timer                timer_;
    std::shared_ptr<NIC> nic_;
    /** @} */

    cycle_t cycle_count_ = 0;

    void wire_interrupts();
};

} // namespace riscv
