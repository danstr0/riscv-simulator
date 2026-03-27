/**
 * @file timer.h
 * @brief Core Local Interuptor (CLINT) Machine Timer.
 *
 * Implements the RISC-V Machine-mode timer facility. This module maintains
 * a 64-bit real-time counter (@c mtime) and a 64-bit compare register (@c mtimecmp).
 *
 * @section behavior Interrupt Semantics
 * The timer asserts a Machine Timer Interrupt (MTI) whenever the value of
 * @c mtime is greater than or equal to @c mtimecmp. The interrupt remains
 * asserted (level-triggered) until software writes a value to @c mtimecmp
 * that is greater than the current @c mtime.
 *
 * @section mmio Memory-Mapped Register Map
 * ┌────────┬───────────┬───────┬────────┬─────────────────────────────────┐
 * │ Offset │ Name      │ Width │ Access │ Description                     │
 * ├────────┼───────────┼───────┼────────┼─────────────────────────────────┤
 * │ 0x00   │ mtime     │ u32   │ RO     │ Timer Register (low 32 bits)    │
 * │ 0x04   │ mtimeh    │ u32   │ RO     │ Timer Register (high 32 bits)   │
 * │ 0x08   │ mtimecmp  │ u32   │ RW     │ Compare Register (low 32 bits)  │
 * │ 0x0C   │ mtimecmph │ u32   │ RW     │ Compare Register (high 32 bits) │
 * └────────┴───────────┴───────┴────────┴─────────────────────────────────┘
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
 * @brief Machine-mode Timer and Comparator.
 */
class Timer : public Memory {
public:
    Timer() = default;

    /** @name MMIO Interface 
     * 
     * Standard 32-bit accessors for the CLINT register block.
     */
    /** @{ */
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override { return {0, 1, false}; }

    /**
     * @brief Writes to mtimecmp.
     * 
     * @note Updates to either the high or low word trigger an immediate
     * re-evaluation of the interrupt line.
     */
    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override { (void)value; return {0, 1, false}; }
    MemoryResult write8(addr_t addr, u8 value)   override { (void)value; return {0, 1, false}; }

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /** @} */

    /** @name Simulation Interface */
    /** @{ */

    /**
     * @brief Advances the internal @c mtime counter.
     *
     * @param cycle The current absolute simulation cycle count.
     *
     * @return True if the interrupt signal transitioned to pending.
     */
    bool tick(cycle_t cycle);

    /** @brief Returns the current state of the timer interrupt line. */
    [[nodiscard]] bool interrupt_pending() const { return pending_; }
    /** @} */

    /** @name Programmatic Configuration */
    /** @{ */

    /** @brief Direct 64-bit update of the compare register. */
    void set_compare(u64 cmp) { mtimecmp_ = cmp; check(); }

    [[nodiscard]] u64 compare() const { return mtimecmp_; }
    [[nodiscard]] cycle_t current_time() const { return mtime_; }

    /** @brief Registers a callback for interrupt signal transitions. */
    using NotifyCallback = std::function<void(bool pending)>;
    void set_notify(NotifyCallback cb) { notify_cb_ = std::move(cb); }

    /** @brief Resets mtime to 0 and mtimecmp to UINT64_MAX. */
    void reset();

    static constexpr addr_t REG_SIZE = 0x10;

private:
    cycle_t mtime_    = 0;        ///< Internal 64-bit time counter.
    u64     mtimecmp_ = ~u64{0};  ///< 64-bit comparator.
    bool    pending_  = false;    ///< Latch for the interrupt signal state.

    NotifyCallback notify_cb_;

    /** @brief Re-evaluates the condition mtime >= mtimecmp. */
    void check();

    static constexpr addr_t MTIME_LO    = 0x00;
    static constexpr addr_t MTIME_HI    = 0x04;
    static constexpr addr_t MTIMECMP_LO = 0x08;
    static constexpr addr_t MTIMECMP_HI = 0x0C;
};

} // namespace riscv
