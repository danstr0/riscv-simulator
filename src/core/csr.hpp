/**
 * @file csr.hpp
 * @brief Machine-mode Control and Status Register (CSR) file.
 *
 * Implements a subset of RISC-V M-mode CSRs for trap vectoring,
 * interrupt masking, and exception handling.
 *
 * @par Trap mechanics
 * On trap entry: MIE is saved to MPIE, MIE is cleared, PC is saved
 * to @c mepc, and the cause is written to @c mcause. On @c mret: MPIE
 * is restored to MIE and execution resumes at @c mepc.
 *
 * @see RISC-V Privileged Architecture Specification, Section 2.1 (Control
 * and Status Registers).
 */

#pragma once

#include "types.hpp"

namespace riscv {

/// Standard RISC-V machine-mode CSR addresses.
namespace CSRAddr
{
    constexpr u32 MSTATUS = 0x300;  ///< Machine status register.
    constexpr u32 MIE     = 0x304;  ///< Machine interrupt-enable register.
    constexpr u32 MTVEC   = 0x305;  ///< Machine trap-handler base address.
    constexpr u32 MEPC    = 0x341;  ///< Machine exception program counter.
    constexpr u32 MCAUSE  = 0x342;  ///< Machine trap cause.
    constexpr u32 MIP     = 0x344;  ///< Machine interrupt pending.
}

/// Bitmasks for the mstatus register.
namespace MStatus
{
    constexpr u32 MIE  = 1u << 3;  ///< Machine Interrupt Enable (global).
    constexpr u32 MPIE = 1u << 7;  ///< Machine Previous Interrupt Enable.
}

/// Interrupt-pending bit positions for @c mie and @c mip registers.
namespace MInterrupt
{
    constexpr u32 MSIE = 1u << 3;   ///< Machine software interrupt.
    constexpr u32 MTIE = 1u << 7;   ///< Machine timer interrupt.
    constexpr u32 MEIE = 1u << 11;  ///< Machine external interrupt.
}

/// Exception and interrupt cause codes for @c mcause.
namespace MCause
{
    constexpr u32 INTERRUPT_BIT = 1u << 31;  ///< Bit 31 distinguishes interrupts from exceptions.

    constexpr u32 M_SOFTWARE    = INTERRUPT_BIT | 3;   ///< Machine software interrupt.
    constexpr u32 M_TIMER       = INTERRUPT_BIT | 7;   ///< Machine timer interrupt.
    constexpr u32 M_EXTERNAL    = INTERRUPT_BIT | 11;  ///< Machine external interrupt.

    constexpr u32 ECALL_M       = 11;  ///< Environment call from M-mode.
    constexpr u32 BREAKPOINT    = 3;   ///< @c ebreak instruction.
}

/**
 * @brief Container for machine-mode CSR state.
 *
 * Encapsulates the privileged register file and provides helpers
 * for trap entry/exit transitions.
 */
class CSRFile {
public:
    CSRFile() { reset(); }

    /// Reset all CSRs to architectural power-on defaults.
    void reset() noexcept 
    {
        mstatus_ = 0;
        mie_     = 0;
        mtvec_   = 0;
        mepc_    = 0;
        mcause_  = 0;
        mip_     = 0;
    }

    /// @name Architectural interface - software-driven CSR access
    /// @{

    /// Reads a CSR value. Returns @c 0 for unimplemented addresses.
    [[nodiscard]] u32 read(u32 addr) const noexcept
    {
        switch (addr)
        {
            case CSRAddr::MSTATUS: return mstatus_;
            case CSRAddr::MIE:     return mie_;
            case CSRAddr::MTVEC:   return mtvec_;
            case CSRAddr::MEPC:    return mepc_;
            case CSRAddr::MCAUSE:  return mcause_;
            case CSRAddr::MIP:     return mip_;
            default:               return 0;
        }
    }

    /// Write a value to a CSR. Unimplemented addresses are silently ignored.
    void write(u32 addr, u32 value) noexcept
    {
        switch (addr)
        {
            case CSRAddr::MSTATUS:
                // Only MIE and MPIE are writable
                mstatus_ = value & (MStatus::MIE | MStatus::MPIE);
                break;
            case CSRAddr::MIE:
                mie_ = value & (MInterrupt::MSIE | MInterrupt::MTIE | MInterrupt::MEIE);
                break;
            case CSRAddr::MTVEC:
                mtvec_ = value;
                break;
            case CSRAddr::MEPC:
                mepc_ = value & ~1u;  // Bit 0 always zero (IALIGN=32)
                break;
            case CSRAddr::MCAUSE:
                mcause_ = value;
                break;
            case CSRAddr::MIP:
                // Hardware sets MTIP/MEIP directly; all bits writable here for simplicity
                mip_ = value & (MInterrupt::MSIE | MInterrupt::MTIE | MInterrupt::MEIE);
                break;
            default:
                break;
        }
    }

    /// @return @c true if @p addr is implemented in this CSR file.
    [[nodiscard]] bool valid(u32 addr) const noexcept
    {
        switch (addr)
        {
            case CSRAddr::MSTATUS:
            case CSRAddr::MIE:
            case CSRAddr::MTVEC:
            case CSRAddr::MEPC:
            case CSRAddr::MCAUSE:
            case CSRAddr::MIP:
                return true;
            default:
                return false;
        }
    }
    /// @}

    /// @name Trap management
    /// @{

    /**
     * @brief Check if any enabled interrupt is pending.
     * @return True if mstatus.MIE is set and a bit is set in both @c mip and @c mie.
     */
    [[nodiscard]] bool interrupt_pending() const noexcept
    {
        return (mstatus_ & MStatus::MIE) && (mip_ & mie_);
    }

    /// @return The @c mcause code for the highest-priority pending interrupt, or @c 0 if none.
    [[nodiscard]] u32 pending_cause() const noexcept
    {
        if (!(mstatus_ & MStatus::MIE)) return 0;

        u32 pending = mip_ & mie_;
        if (pending & MInterrupt::MEIE) return MCause::M_EXTERNAL;
        if (pending & MInterrupt::MTIE) return MCause::M_TIMER;
        if (pending & MInterrupt::MSIE) return MCause::M_SOFTWARE;
        return 0;
    }

    /**
     * @brief Enter a trap: save MIE to MPIE, disable interrupts, record PC and cause.
     * @param pc    Address of the faulting or interrupted instruction.
     * @param cause The @c mcause value (see MCause namespace).
     */
    void enter_trap(addr_t pc, u32 cause) noexcept
    {
        if (mstatus_ & MStatus::MIE)
            mstatus_ |= MStatus::MPIE;
        else
            mstatus_ &= ~MStatus::MPIE;

        mstatus_ &= ~MStatus::MIE;
        mepc_   = pc;
        mcause_ = cause;
    }

    /**
     * @brief Execute mret: restore MIE from MPIE, set MPIE, return saved PC.
     * @return The @c mepc address to resume execution at.
     */
    [[nodiscard]] addr_t mret() noexcept
    {
        if (mstatus_ & MStatus::MPIE)
            mstatus_ |= MStatus::MIE;
        else
            mstatus_ &= ~MStatus::MIE;

        mstatus_ |= MStatus::MPIE;
        return mepc_;
    }
    /// @}

    /// @name Hardware signal integration
    /// @{

    /// Assert an interrupt-pending bit in @c mip.
    void set_mip_bit(u32 bit)   noexcept { mip_ |= bit; }
    /// Deassert an interrupt-pending bit in @c mip.
    void clear_mip_bit(u32 bit) noexcept { mip_ &= ~bit; }

    [[nodiscard]] u32 mstatus() const noexcept { return mstatus_; }
    [[nodiscard]] u32 mie()     const noexcept { return mie_; }
    [[nodiscard]] u32 mip()     const noexcept { return mip_; }
    [[nodiscard]] u32 mtvec()   const noexcept { return mtvec_; }
    [[nodiscard]] u32 mepc()    const noexcept { return mepc_; }
    [[nodiscard]] u32 mcause()  const noexcept { return mcause_; }
    /// @}

private:
    u32 mstatus_ = 0;
    u32 mie_     = 0;
    u32 mtvec_   = 0;
    u32 mepc_    = 0;
    u32 mcause_  = 0;
    u32 mip_     = 0;
};

} // namespace riscv
