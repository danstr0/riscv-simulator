/**
 * @file types.hpp
 * @brief Foundational type system for the CPU simulator.
 *
 * Provides fixed-width integer aliases, address/cycle types, exception
 * hierarchy, and low-level bit-manipulation utilities used throughout the
 * simulator. All helpers are constexpr and noexcept so they can be
 * evaluated at compile time and used in hot decode/execute paths without
 * overhead.
 *
 * @section ARCHITECTURAL_ASSUMPTIONS
 * This simulator is designed for the RISC-V base ISA, which is strictly
 * little-endian. To ensure performance and simplicity, the memory subsystem
 * utilizes @c std::memcpy for raw byte-to-integer mappings.
 * * ** Constraint ** This implementation requires a little-endian host (e.g., x86-64).
 * Running on a big-endian host will result in incorrect multi-byte integer
 * interpretation.
 *
 * @see Waterman, A. S. (2016). "Design of the RISC-V Instruction Set Architecture."
 * UC Berkeley Technical Report.
 * @see RISC-V Unprivileged ISA Specification, v20260120, Sections 2.1.3, 2.1.6.
 * @see RISC-V ABIs Specification, Version 1.0, Chapter 1.1.
 */

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace riscv {

/* ───────────────────────────────────────────────────────────────────────
 * Fixed-Width Integer Aliases
 * ─────────────────────────────────────────────────────────────────────── */

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

/// 32-bit address for RV32.
using addr_t = u32;

/// Cycle counter - 64-bit to avoid overflow.
using cycle_t = u64;

/// Register index (0-31).
using reg_idx_t = u8;

/* ───────────────────────────────────────────────────────────────────────
 * Endianness Guard
 * ───────────────────────────────────────────────────────────────────────
 * RISC-V base ISA is little-endian. This asserts the host's native endianness
 * matches to ensure that direct memory copies (std::memcpy) are valid.
 */
static_assert(std::endian::native == std::endian::little,
              "This implementation assumes a little-endian host to match the RISC-V base ISA.");

/* ═══════════════════════════════════════════════════════════════════════
 * ABI Register Names
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief RISC-V Application Binary Interface (ABI) register names.
 * 
 * While the hardware refers to registers as x0-x31, the ABI assigns
 * semantic roles (e.g., 'sp' for stack pointer). This table maps
 * hardware indices [0-31] to their standard symbolic names.
 *
 * @see RISC-V ABIs Specification, Chapter 1.1 (Integer Register Convention).
 */
inline constexpr std::array<const char*, 32> kRegNames = {
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8",   "s9", "s10","s11","t3", "t4", "t5", "t6"
};

/**
 * @brief Returns the ABI name for a given register index.
 * @param r The register index (0-31).
 * @return A string view of the ABI name, or "???" if index is out of bounds.
 */
[[nodiscard]] constexpr const char* reg_name(reg_idx_t r) noexcept
{
    return r < 32 ? kRegNames[r] : "???";
}

/* ═══════════════════════════════════════════════════════════════════════
 * Exception Hierarchy
 * ═══════════════════════════════════════════════════════════════════════ */

/// Base exception for all CPU-related faults.
class CpuException : public std::runtime_error {
public:
    explicit CpuException(const std::string& msg) : std::runtime_error(msg) {}
};

/// Raised when the decoder encounters an unrecognized instruction encoding.
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

/// Raised when a memory read/write targets an unmapped address range.
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
 * Raised when a memory access violates natural alignment requirements.
 *
 * This implementation traps on misaligned access (RISC-V spec §2.1.6 permits
 * either trapping or supporting misaligned access).
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

/* ═══════════════════════════════════════════════════════════════════════
 * Bit-Manipulation Utilities
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Sign-extend a B-bit value to a full i32.
 *
 * The value is first masked to B bits, then the sign bit (bit B-1) is
 * replicated into the upper 32-B bits. This is used extensively in
 * immediate extraction during instruction decoding.
 *
 * @tparam B  Number of significant bits (1 <= B <= 32).
 * @param value  The raw bitfield extracted from the instruction.
 * @see RISC-V Spec §2.1.3, Figure 1 for immediate formats.
 */
template<unsigned B>
[[nodiscard]] constexpr i32 sign_extend(u32 value) noexcept
{
    static_assert(B > 0 && B <= 32, "Bit width must be in [1, 32]");

    if constexpr (B == 32) {
        return static_cast<i32>(value);
    } else {
        constexpr u32 sign_bit = 1u << (B - 1);
        constexpr u32 mask     = (1u << B) - 1;
        value &= mask;
        return static_cast<i32>((value ^ sign_bit) - sign_bit);
    }
}

/**
 * @brief Extract bits [hi:lo] (inclusive) from @p value.
 * @param value  The source word.
 * @param hi  The upper bit index.
 * @param lo  The lower bit index.
 * @pre hi > lo
 * @pre hi - lo + 1 <= 32
 */
[[nodiscard]] constexpr u32 bits(u32 value, unsigned hi, unsigned lo) noexcept
{
    unsigned width = hi - lo + 1;
    u32 mask = (width >= 32) ? ~u32{0} : (1u << width) - 1;
    return (value >> lo) & mask;
}

/** @brief Extract a single bit at position @p pos. */
[[nodiscard]] constexpr u32 bit(u32 value, unsigned pos) noexcept
{
    return (value >> pos) & 1;
}

} // namespace riscv
