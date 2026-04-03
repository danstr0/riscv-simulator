/**
 * @file csr.hpp
 * @brief Machine-mode Command and Status Register (CSR) State File.
 *
 * This module implements a subset of the RISC-V Machine-mode CSRs. It provides
 * the necessary state for trap vectoring, interrupt masking, and exception
 * delegation required for system-level benchmarking.
 *
 * @section trap_mechanics Trap Entry and Exit
 * When a trap is taken (via @c enter_trap):
 * 1. The current interrupt enable bit (MIE) is moved to the previous bit (MPIE).
 * 2. Interrupts are globally disabled (MIE = 0).
 * 3. The current Program Counter is saved to @c mepc.
 * 4. The cause of the trap is written to @c mcause.
 *
 * Upon executing @c mret, the MPIE bit is restored to MIE, effectively
 * re-enabling interrupts to their state prior to the trap.
 *
 * @see RISC-V Privileged Architecture Specification, Section 2.1 (Control and 
 * Status Registers)
 */

#pragma once

#include "types.hpp"

#include <unordered_map>

namespace riscv {

/**
 * @brief Standard RISC-V machine-mode CSR addresses.
 */
namespace CSRAddr {
    constexpr u32 MSTATUS = 0x300;  ///< Machine status register.
    constexpr u32 MIE     = 0x304;  ///< Machine interrupt-enable register.
    constexpr u32 MTVEC   = 0x305;  ///< Machine trap-handler base address.
    constexpr u32 MEPC    = 0x341;  ///< Machine exception program counter.
    constexpr u32 MCAUSE  = 0x342;  ///< Machine trap cause.
    constexpr u32 MIP     = 0x344;  ///< Machine interrupt pending.
}

/**
 * @brief Bitmasks for the mstatus register.
 */
namespace MStatus {
    constexpr u32 MIE  = 1u << 3;  ///< Machine Interrupt Enable (Global).
    constexpr u32 MPIE = 1u << 7;  ///< Machine Previous Interrupt Enable.
}

/**
 * @brief Interrupt indices for mie and mip registers.
 */
namespace MInterrupt {
    constexpr u32 MSIE = 1u << 3;   ///< Machine Software Interrupt.
    constexpr u32 MTIE = 1u << 7;   ///< Machine Timer Interrupt.
    constexpr u32 MEIE = 1u << 11;  ///< Machine External Interrupt (via PLIC).
}

/**
 * @brief Exception and Interrupt Cause Codes.
 */
namespace MCause {
    /** @brief Bit 31 of mcause indicates if the trap was an interrupt. */
    constexpr u32 INTERRUPT_BIT = 1u << 31;

    /* Standard interrupt causes */
    constexpr u32 M_SOFTWARE    = INTERRUPT_BIT | 3;
    constexpr u32 M_TIMER       = INTERRUPT_BIT | 7;
    constexpr u32 M_EXTERNAL    = INTERRUPT_BIT | 11;

    /* Standard exception causes */
    constexpr u32 ECALL_M       = 11;  ///< Environment call from M-mode.
    constexpr u32 BREAKPOINT    = 3;   ///< ebreak instruction.
}

/**
 * @brief Container for Machine-mode CSR state.
 *
 * This encapsulates the state of the privileged register file and
 * provides atomic-style updates for trap transitions.
 */
class CSRFile {
public:
    CSRFile() { reset(); }

    /** @brief Resets all CSRs to their architectural power-on state. */
    void reset() noexcept 
    {
        mstatus_ = 0;
        mie_     = 0;
        mtvec_   = 0;
        mepc_    = 0;
        mcause_  = 0;
        mip_     = 0;
    }

    /** @name Architectural Interface 
     *
     * Methods for software-driven CSR access.
     */
    /** @{ */

    /** @brief Reads a CSR value. Returns 0 for unsupported addresses. */
    [[nodiscard]] u32 read(u32 addr) const noexcept
    {
        switch (addr) {
            case CSRAddr::MSTATUS: return mstatus_;
            case CSRAddr::MIE:     return mie_;
            case CSRAddr::MTVEC:   return mtvec_;
            case CSRAddr::MEPC:    return mepc_;
            case CSRAddr::MCAUSE:  return mcause_;
            case CSRAddr::MIP:     return mip_;
            default:               return 0;
        }
    }

    /** @brief Writes a value to a CSR. */
    void write(u32 addr, u32 value) noexcept
    {
        switch (addr) {
            case CSRAddr::MSTATUS:
                /* Only MIE and MPIE are writable */
                mstatus_ = value & (MStatus::MIE | MStatus::MPIE);
                break;
            case CSRAddr::MIE:
                mie_ = value & (MInterrupt::MSIE | MInterrupt::MTIE | MInterrupt::MEIE);
                break;
            case CSRAddr::MTVEC:
                mtvec_ = value;
                break;
            case CSRAddr::MEPC:
                mepc_ = value & ~1u;  /* Low bit must be 0 */
                break;
            case CSRAddr::MCAUSE:
                mcause_ = value;
                break;
            case CSRAddr::MIP:
                /*
                 * Only MSIP is software-writable; MTIP and MEIP are read-only.
                 * For simplicity, writes are allowed to all bits the device
                 * implementations set them directly.
                 */
                mip_ = value & (MInterrupt::MSIE | MInterrupt::MTIE | MInterrupt::MEIE);
                break;
            default:
                break;
        }
    }

    /** @brief Validates if the CSR address is implemented in this model. */
    [[nodiscard]] bool valid(u32 addr) const noexcept
    {
        switch (addr) {
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
    /** @} */

    /** @name Trap Management
     *
     * Helper methods to manage hardware-driven state changes.
     */
    /** @{ */
    
    /**
     * @brief Checks if an interrupt should be taken.
     *
     * @return True if @c mstatus.MIE is set AND a bit is set in both @c mip and @c mie.
     */
    [[nodiscard]] bool interrupt_pending() const noexcept
    {
        return (mstatus_ & MStatus::MIE) && (mip_ & mie_);
    }

    /** @brief Returns the @c mcause code for the highest-priority pending interrupt. */
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
     * @brief Updates CSR state to reflect a hardware trap entry.
     *
     * @param pc The address of the instruction that was interrupted or caused the exception.
     *
     * @param cause The mcause value (see namespace MCause).
     */
    void enter_trap(addr_t pc, u32 cause) noexcept
    {
        if (mstatus_ & MStatus::MIE)
            mstatus_ |= MStatus::MPIE;
        else
            mstatus_ &= ~MStatus::MIE;

        mstatus_ &= ~MStatus::MIE;
        mepc_   = pc;
        mcause_ = cause;
    }

    /**
     * @brief Restores state after an mret instruction.
     *
     * @return The address saved in @c mepc to which the CPU should return.
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

    /** @} */

    /** @name Hardware Signal Integration */
    /** @{ */
    void set_mip_bit(u32 bit)   noexcept { mip_ |= bit; }
    void clear_mip_bit(u32 bit) noexcept { mip_ &= ~bit; }

    [[nodiscard]] u32 mstatus() const noexcept { return mstatus_; }
    [[nodiscard]] u32 mie()     const noexcept { return mie_; }
    [[nodiscard]] u32 mip()     const noexcept { return mip_; }
    [[nodiscard]] u32 mtvec()   const noexcept { return mtvec_; }
    [[nodiscard]] u32 mepc()    const noexcept { return mepc_; }
    [[nodiscard]] u32 mcause()  const noexcept { return mcause_; }
    /** @} */

private:
    u32 mstatus_ = 0;
    u32 mie_     = 0;
    u32 mtvec_   = 0;
    u32 mepc_    = 0;
    u32 mcause_  = 0;
    u32 mip_     = 0;
};

} // namespace riscv
