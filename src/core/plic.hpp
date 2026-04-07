/**
 * @file plic.hpp
 * @brief Simplified Platform-Level Interrupt Controller for RISC-V.
 *
 * This class models a simplified RISC-V PLIC conforming to the 1.0.0 specification.
 * It serves as a central hub for multiplexing multiple device interrupts into a
 * single external interrupt signal for a single execution target (Hart 0).
 *
 * @section interrupt_flow Interrupt Lifecycle
 * 1. **Gateway**: A device asserts an interrupt via `set_pending()`.
 * 2. **Notification**: If the source is enabled and its priority exceeds the
 * configured threshold, the `NotifyCallback` is triggered.
 * 3. **Claim**: The CPU reads the `CLAIM_ADDR`. The PLIC returns the ID of the
 * highest-priority pending interrupt and atomically clears the pending bit.
 * 4. **Completion**: After servicing the interrupt, the CPU writes the ID back
 * to `CLAIM_ADDR` (Complete) to allow the source to trigger again.
 *
 * @note This implementation supports up to 31 active sources. Source 0 is
 * hardwired to "no interrupt" per the specification.
 *
 * @see RISC-V PLIC Specification v1.0.0
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <array>
#include <functional>

namespace riscv {

/**
 * @brief Static configuration for the PLIC.
 */
struct PLICConfig {
    /** 
     * @brief Maximum number of supported interrupt sources.
     *
     * Source 0 is reserved by the ISA; actual devices occupy 1 to 31.
     */
    static constexpr u32 MAX_SOURCES = 32;
};

/**
 * @brief Models the Platform-Level Interrupt Controller MMIO device.
 */
class PLIC : public Memory {
public:
    PLIC() = default;

    /** @brief MMIO region size (for MMIOBus mapping). */
    static constexpr addr_t REG_SIZE = 0x200008;

    /** @name Memory-Mapped I/O Interface
     *
     * All PLIC registers are defined as 32-bit words. 8-bit and 16-bit
     * accesses are treated as invalid or ignored to match hardware behavior.
     */
    /** @{ */
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override { return {0, 1, false}; }

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override { (void)value; return {0, 1, false}; }
    MemoryResult write8(addr_t addr, u8 value)   override { (void)value; return {0, 1, false}; }
    
    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /** @} */

    /** @name Hardware Signal Interface (Gateways) 
     *
     * Methods used by peripheral devices to signal state changes.
     */
    /** @{ */

    /**
     * @brief Asserts an interrupt signal from a peripheral.
     *
     * @param source The Interrupt ID (1 to 31).
     */
    void set_pending(u32 source);

    /** @brief Manually clear a pending bit (typically for simulation reset). */
    void clear_pending(u32 source);

    /** @brief Returns true if the PLIC is currently asserting an interrupt to the CPU. */
    [[nodiscard]] bool interrupt_pending() const;

    /**
     * @brief Performs the "Claim" operation for Hart 0.
     * 
     * @return The ID of the highest priority pending interrupt, or 0 if none.
     */
    [[nodiscard]] u32 claim();

    /**
     * @brief Performs the "Completion" operation.
     *
     * @param source The ID previously returned by a claim.
     */
    void complete(u32 source);
    /** @} */

    /**
     * @name Register Accessors
     *
     * Direct programmatic access to the PLIC internal state.
     */
    /** @{ */
    void set_priority(u32 source, u32 priority);
    void set_enable(u32 source, bool enable);
    void set_threshold(u32 threshold) { threshold_ = threshold; }

    [[nodiscard]] u32 priority(u32 source) const;
    [[nodiscard]] bool enabled(u32 source) const;
    [[nodiscard]] u32 threshold() const { return threshold_; }

    /** 
     * @brief Sets the callback for the external interrupt line.
     *
     * This simulates the physical wire connecting the PLIC to the CPU's `meip` bit.
     */
    using NotifyCallback = std::function<void(bool pending)>;
    void set_notify(NotifyCallback cb) { notify_cb_ = std::move(cb); }

    /** @brief Resets all priorities, enables, and pending bits to zero. */
    void reset();
    /** @} */

private:
    std::array<u32, PLICConfig::MAX_SOURCES> priorities_{};
    u32 pending_bits_    = 0;  ///< Bitmask of pending interrupts (Source 1 = Bit 1).
    u32 enable_bits_     = 0;  ///< Bitmask of enabled interrupts for Hart 0.
    u32 threshold_       = 0;  ///< Priority threshold; interrupts <= threshold are masked.
    mutable u32 claimed_ = 0;  ///< Currently active (claimed but not completed) source.

    NotifyCallback notify_cb_;

    /** @brief Evaluates the priority tree and invokes the notify callback if state changed. */
    void notify() const;

    /** @name MMIO Memory Map */
    /** @{ */
    static constexpr addr_t PRIORITY_BASE  = 0x000000; ///< Priority of source i (u32)
    static constexpr addr_t PENDING_BASE   = 0x001000; ///< Pending bit for source i
    static constexpr addr_t ENABLE_BASE    = 0x002000; ///< Enable bit for source i
    static constexpr addr_t THRESHOLD_ADDR = 0x200000; ///< Hart 0 priority threshold
    static constexpr addr_t CLAIM_ADDR     = 0x200004; ///< Hart 0 claim/complete register
    /** @} */
};

} // namespace riscv
