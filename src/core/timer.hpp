/**
 * @file timer.hpp
 * @brief Core Local Interruptor (CLINT) machine timer.
 *
 * Maintains a 64-bit real-time counter (@c mtime) and a 64-bit compare
 * register (@c mtimecmp). Asserts a level-triggered Machine Timer Interrupt
 * whenever @c mtime ≥ @c mtimecmp.
 *
 *
 * @par MMIO register map
 * @code
 *   ┌────────┬───────────┬───────┬────────┬─────────────────────────────────┐
 *   │ Offset │ Name      │ Width │ Access │ Description                     │
 *   ├────────┼───────────┼───────┼────────┼─────────────────────────────────┤
 *   │ 0x00   │ mtime     │ u32   │ RO     │ Timer Register (low 32 bits)    │
 *   │ 0x04   │ mtimeh    │ u32   │ RO     │ Timer Register (high 32 bits)   │
 *   │ 0x08   │ mtimecmp  │ u32   │ RW     │ Compare Register (low 32 bits)  │
 *   │ 0x0C   │ mtimecmph │ u32   │ RW     │ Compare Register (high 32 bits) │
 *   └────────┴───────────┴───────┴────────┴─────────────────────────────────┘
 * @endcode
 *
 * @note In this implementation, @c mtime is read-only to software. Updates
 * occur via the simulation @c tick() method.
 *
 * @see RISC-V Privileged ISA Specification, Section 3.1.2 (Machine-Level 
 * Memory-Mapped Registers).
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <functional>

namespace riscv {

/**
 * @brief CLINT machine-mode timer.
 *
 * @c mtime is read-only to software; it advances via the simulation @c tick() method.
 * All registers are 32-bit; sub-word accesses are rejected.
 */
class Timer : public Memory {
public:
    Timer() = default;

    static constexpr addr_t REG_SIZE = 0x10;  ///< MMIO region size.

    /// @name Memory-mapped interface
    /// @{
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t)  const override { return {0, 1, false}; }

    /// Writes to @c mtimecmp trigger immediate interrupt re-evaluation.
    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t, u16) override { return {0, 1, false}; }
    MemoryResult write8(addr_t, u8)   override { return {0, 1, false}; }

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /// @}

    /// @name Simulation interface
    /// @{

    /**
     * @brief Advances @c mtime to the given cycle count.
     * @return @c true if the interrupt signal transitioned from deasserted to asserted.
     */
    bool tick(cycle_t cycle);

    /// @return @c true if @c mtime ≥ @c mtimecmp.
    [[nodiscard]] bool interrupt_pending() const { return pending_; }
    /// @}

    /// @name Configuration
    /// @{
    void set_compare(u64 cmp) { mtimecmp_ = cmp; check(); }

    [[nodiscard]] u64     compare()      const { return mtimecmp_; }
    [[nodiscard]] cycle_t current_time() const { return mtime_; }

    /// Callback for interrupt signal transitions (simulates the MTIP wire).
    using NotifyCallback = std::function<void(bool pending)>;
    void set_notify(NotifyCallback cb) { notify_cb_ = std::move(cb); }

    /// Reset @c mtime to 0, @c mtimecmp to @c UINT64_MAX, clear pending.
    void reset();
    /// @}

private:
    cycle_t mtime_    = 0;
    u64     mtimecmp_ = ~u64{0};
    bool    pending_  = false;

    NotifyCallback notify_cb_;

    /// Re-evaluate @c mtime ≥ @c mtimecmp and fire callback on transitions.
    void check();

    /// @name MMIO address offsets
    /// @{
    static constexpr addr_t MTIME_LO    = 0x00;
    static constexpr addr_t MTIME_HI    = 0x04;
    static constexpr addr_t MTIMECMP_LO = 0x08;
    static constexpr addr_t MTIMECMP_HI = 0x0C;
    /// @}
};

} // namespace riscv
