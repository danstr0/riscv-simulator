/**
 * @file types.hpp
 * @brief Foundational type system for the RISC-V simulator.
 *
 * Fixed-width integer aliases, address/cycle types, exception hierarchy,
 * and bit-manipulation utilities used throughout the simulator. All 
 * bit-manipulation helpers are constexpr and noexcept for use in hot 
 * decode/execute paths.
 *
 * @par Host requirement
 * A little-endian host is required to match the RISC-V base ISA.
 * This is enforced at compile time via static_assert.
 *
 * @see Waterman, A. S. (2016). "Design of the RISC-V Instruction Set Architecture."
 * UC Berkeley Technical Report.
 * @see RISC-V Unprivileged ISA Specification, v20260120, Sections 2.1.3, 2.1.6.
 * @see RISC-V ABIs Specification, Version 1.0, Chapter 1.1.
 */

#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <string>

namespace riscv {

/// @name Fixed-width integer aliases
/// @{

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using addr_t    = u32;  ///< 32-bit address for RV32.
using cycle_t   = u64;  ///< 64-bit cycle counter.
using reg_idx_t = u8;   ///< Register index (0-31).

/// @}

/// Compile-time check that the host is little-endian.
static_assert(std::endian::native == std::endian::little,
              "This implementation requires a little-endian host.");

/// @name ABI register and vector register names
/// @{

/**
 * @brief ABI register name table, indexed by hardware register number.
 * @see RISC-V ABIs Specification, Chapter 1.1 (Integer Register Convention).
 */
inline constexpr std::array<const char*, 32> kRegNames = {
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8",   "s9", "s10","s11","t3", "t4", "t5", "t6"
};

/// Returns the ABI name for a register @p r, or "???" if out of range.
[[nodiscard]] constexpr const char* reg_name(reg_idx_t r) noexcept
{
    return r < 32 ? kRegNames[r] : "???";
}

/// Returns a vector register name (e.g., "v0", "v31"). 
[[nodiscard]] inline std::string vreg_name(reg_idx_t r)
{
    return r < 32 ? std::format("v{}", r) : "???";
}

/// @}

/// @name Exception hierarchy
/// @{

/// Base exception for all CPU-related faults.
class CpuException : public std::runtime_error {
public:
    explicit CpuException(const std::string& msg) : std::runtime_error(msg) {}
};

/// Unrecognized instruction encoding.
class IllegalInstructionException : public CpuException {
public:
    IllegalInstructionException(u32 inst, addr_t pc)
        : CpuException(std::format("Illegal instruction 0x{:08x} at PC 0x{:08x}", inst, pc))
        , instruction_(inst)
        , pc_(pc) {}

    [[nodiscard]] u32    instruction() const noexcept { return instruction_; }
    [[nodiscard]] addr_t pc()          const noexcept { return pc_; }

private:
    u32    instruction_;
    addr_t pc_;
};

/// Memory read/write to an unmapped address.
class MemoryAccessException : public CpuException {
public:
    MemoryAccessException(addr_t addr, bool write)
        : CpuException(std::format("Memory {} fault at 0x{:08x}",
                                   write ? "write" : "read", addr))
        , address_(addr)
        , is_write_(write) {}

    [[nodiscard]] addr_t address()  const noexcept { return address_; }
    [[nodiscard]] bool   is_write() const noexcept { return is_write_; }

private:
    addr_t address_;
    bool   is_write_;
};

/**
 * @brief Memory access violating natural alignment.
 *
 * This implementation traps rather than supports misaligned access 
 * (the spec permits either behavior).
 */
class MisalignedAccessException : public CpuException {
public:
    MisalignedAccessException(addr_t addr, size_t align)
        : CpuException(std::format("Misaligned access at 0x{:08x} (requires {}-byte alignment)",
                                   addr, align))
        , address_(addr)
        , alignment_(align) {}

    [[nodiscard]] addr_t address()   const noexcept { return address_; }
    [[nodiscard]] size_t alignment() const noexcept { return alignment_; }

private:
    addr_t address_;
    size_t alignment_;
};

/// @}

/// @name Bit-manipulation utilities
/// @{

/**
 * @brief Sign-extend a B-bit value to a full i32.
 *
 * Masks to B bits, then replicates the sign bit (bit B-1) into the
 * upper 32-B bits. Used extensively in immediate extraction during
 * instruction decoding.
 *
 * @tparam B  Number of significant bits (1 ≤ B ≤ 32).
 * @param value  Raw bitfield extracted from the instruction word.
 * @see RISC-V Unprivileged ISA Specification, Section 2.1.3.
 */
template<unsigned B>
[[nodiscard]] constexpr i32 sign_extend(u32 value) noexcept
{
    static_assert(B > 0 && B <= 32, "Bit width must be in [1, 32]");

    if constexpr (B == 32)
        return static_cast<i32>(value);
    else 
    {
        constexpr u32 sign_bit = 1u << (B - 1);
        constexpr u32 mask     = (1u << B) - 1;
        value &= mask;
        return static_cast<i32>((value ^ sign_bit) - sign_bit);
    }
}

/// Extract bits [hi:lo] (inclusive) from @p value.
[[nodiscard]] constexpr u32 bits(u32 value, unsigned hi, unsigned lo) noexcept
{
    assert(hi >= lo && "bits(): hi must be >= lo");

    unsigned width = hi - lo + 1;
    u32 mask = (width >= 32) ? ~u32{0} : (1u << width) - 1;
    return (value >> lo) & mask;
}

/// Extract a single bit at position @p pos.
[[nodiscard]] constexpr u32 bit(u32 value, unsigned pos) noexcept
{
    return (value >> pos) & 1;
}

/// @}

} // namespace riscv
