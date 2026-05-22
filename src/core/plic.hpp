/**
 * @file plic.hpp
 * @brief Simplified Platform-Level Interrupt Controller (PLIC) for RISC-V.
 *
 * Models a single-hart PLIC conforming to the RISC-V PLIC Specification v1.0.0,
 * supporting up to 31 interrupt sources (source 0 is reserved as "no interrupt").
 *
 * @par Interrupt lifecycle
 * A device asserts an interrupt via @c set_pending(). If the source is enabled
 * and its priority exceeds the threshold, the notify callback fires. The CPU
 * claims the interrupt (returning the highest-priority pending ID and clearing
 * the pending bit), services it, then writes the ID back to complete.
 *
 * @see RISC-V PLIC Specification v1.0.0.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <array>
#include <functional>

namespace riscv {

/// Static PLIC configuration.
struct PLICConfig
{
    /// Maximum interrupt sources. Source 0 is reserved; devices use 1-31.
    static constexpr u32 MAX_SOURCES = 32;
};

/**
 * @brief MMIO-mapped Platform-Level Interrupt Controller.
 *
 * All registers are 32-bit; sub-word accesses are rejected.
 */
class PLIC : public Memory {
public:
    PLIC() = default;

    static constexpr addr_t REG_SIZE = 0x200008;  ///< MMIO region size.

    /// @name Memory-mapped interface
    /// @{
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t)  const override { return {0, 1, false}; }

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t, u16) override { return {0, 1, false}; }
    MemoryResult write8(addr_t, u8)   override { return {0, 1, false}; }
    
    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /// @}

    /// @name Device signal interface
    /// @{

    /// Assert an interrupt from source @p source (1-31).
    void set_pending(u32 source);

    /// Manually clear a pending bit.
    void clear_pending(u32 source);

    /// @return @c true if any enabled source exceeds the priority threshold.
    [[nodiscard]] bool interrupt_pending() const;

    /**
     * @brief Claim the highest-priority pending interrupt for Hart 0.
     * @return Source ID (1-31), or 0 if none pending.
     */
    [[nodiscard]] u32 claim();

    /// Complete a previously claimed interrupt, allowing it to trigger again.
    void complete(u32 source);
    /// @}

    /// @name Register access
    /// @{
    void set_priority(u32 source, u32 priority);
    void set_enable(u32 source, bool enable);
    void set_threshold(u32 threshold) { threshold_ = threshold; }

    [[nodiscard]] u32  priority(u32 source) const;
    [[nodiscard]] bool enabled(u32 source) const;
    [[nodiscard]] u32  threshold() const { return threshold_; }

    /// Callback simulating the physical MEIP wire to the CPU.
    using NotifyCallback = std::function<void(bool pending)>;
    void set_notify(NotifyCallback cb) { notify_cb_ = std::move(cb); }

    /// Reset all state to power-on defaults.
    void reset();
    /// @}

private:
    std::array<u32, PLICConfig::MAX_SOURCES> priorities_{};
    u32         pending_bits_ = 0;  ///< Bitmask of pending interrupts (bit i = source i).
    u32         enable_bits_  = 0;  ///< Bitmask of enabled interrupts for Hart 0.
    u32         threshold_    = 0;  ///< Interrupts with priority ≤ threshold are masked.
    mutable u32 claimed_      = 0;  ///< Currently claimed source (0 = none).

    NotifyCallback notify_cb_;

    /// Evaluate priority tree and invoke notify callback.
    void notify() const;

    /// @name MMIO address map
    /// @{
    static constexpr addr_t PRIORITY_BASE  = 0x000000;
    static constexpr addr_t PENDING_BASE   = 0x001000;
    static constexpr addr_t ENABLE_BASE    = 0x002000;
    static constexpr addr_t THRESHOLD_ADDR = 0x200000;
    static constexpr addr_t CLAIM_ADDR     = 0x200004;
    /// @}
};

} // namespace riscv
