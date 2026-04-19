/**
 * @file vector_state.hpp
 * @brief Architectural state for a simplified RISC-V Vector (RVV) 1.0 implementation.
 *
 * @par Subset limitations
 * SEW is fixed at 32-bit, LMUL at 1 (no register grouping), and vstart
 * is hardwired to 0. VLEN must be a power of 2, minimum 64 bits.
 *
 * @see RISC-V Unprivileged ISA Specification, Chapter 31.
 */

#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace riscv {

/// Hardware configuration parameters for the vector unit.
struct VectorConfig {
    u32 vlen = 128; ///< Vector register length in bits. Must be a power of 2, minimum 64.

    /// @return Number of bytes per vector register (VLENB).
    [[nodiscard]] constexpr u32 vlenb() const noexcept { return vlen / 8; }

    /// @return Maximum elements at SEW=32 per register.
    [[nodiscard]] constexpr u32 vlmax_sew32() const noexcept { return vlen / 32; }

    /// @return True if VLEN satisfies hardware constraints (power of 2, >= 64).
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return vlen >= 64 && std::has_single_bit(vlen);
    }
};

/**
 * @brief Represents the vtype CSR.
 *
 * Encodes how vector registers are interpreted. Fields map to 
 * bits [0:7] and bit [31] of the vtype CSR.
 */
struct VType {
    u32 sew   = 32;     ///< Selected element width in bits.
    u32 lmul  = 1;      ///< Register grouping multiplier.
    bool vta  = false;  ///< Tail agnostic: tail elements may be overwritten with 1s.
    bool vma  = false;  ///< Mask agnostic: masked-off elements may be overwritten with 1s.
    bool vill = false;  ///< Illegal: any vector instruction using this state traps.

    /**
     * @brief Encodes the struct into the 32-bit vtype CSR format.
     * @return Encoded value (bit 31 = vill, bits [5:3] = vsew).
     */
    [[nodiscard]] constexpr u32 encode() const noexcept
    {
        if (vill) return 1u << 31;

        u32 vsew_field = 0;
        switch(sew)
	{
            case 8:  vsew_field = 0b000; break;
            case 16: vsew_field = 0b001; break;
            case 32: vsew_field = 0b010; break;
            case 64: vsew_field = 0b011; break;
            default: return 1u << 31; // invalid sew → vill
        }
        return (static_cast<u32>(vma) << 7)
             | (static_cast<u32>(vta) << 6)
             | (vsew_field << 3);
    }

    /**
     * @brief Decodes a vsetvli immediate into a VType struct. 
     * @note Sets vill if SEW is not 32-bit.
     */
    static VType decode(u32 zimm)
    {
        VType vt;
        u32 vsew = (zimm >> 3) & 0x7;
        switch (vsew)
	{
            case 0b000: vt.sew = 8;  break;
            case 0b001: vt.sew = 16; break;
            case 0b010: vt.sew = 32; break;
            case 0b011: vt.sew = 64; break;
            default:    vt.vill = true; return vt;
        }
        if (vt.sew != 32) vt.vill = true;
        vt.vta  = (zimm >> 6) & 1;
        vt.vma  = (zimm >> 7) & 1;
        vt.lmul = 1;
        return vt;
    }
};

/**
 * @brief Managed storage for the 32 vector registers (v0-v31).
 *
 * Uses a flat byte buffer. Element offset within the buffer:
 * @code
 *   (reg_index * VLENB) + (elem_index * SEW/8)
 * @endcode
 */
class VectorRegFile {
public:
    explicit VectorRegFile(VectorConfig config = {})
        : config_(config)
        , data_(32 * config.vlenb(), 0)
    {
        assert(config_.valid());
    }

    [[nodiscard]] u32 vlenb() const noexcept { return config_.vlenb(); }
    [[nodiscard]] u32 vlmax() const noexcept { return config_.vlmax_sew32(); }

    /// Read a 32-bit element from register @p vreg at index @p elem.
    [[nodiscard]] u32 get_elem32(u32 vreg, u32 elem) const noexcept
    {
        assert(vreg < 32 && elem < vlmax());
        u32 value;
        std::memcpy(&value, &data_[(vreg * vlenb()) + (elem * 4)], 4);
        return value;
    }

    /// Write a 32-bit element to register @p vreg at index @p elem.
    void set_elem32(u32 vreg, u32 elem, u32 value) noexcept
    {
        assert(vreg < 32 && elem < vlmax());
        std::memcpy(&data_[(vreg * vlenb()) + (elem * 4)], &value, 4);
    }

    /// Read mask bit @p elem from v0 (bit @p elem of register v0).
    [[nodiscard]] bool get_mask_bit(u32 elem) const noexcept
    {
        u32 byte_idx = elem / 8;
        u32 bit_idx  = elem % 8;
        return (data_[byte_idx] >> bit_idx) & 1;
    }

    /// Set mask bit @p elem in v0.
    void set_mask_bit(u32 elem, bool val) noexcept
    {
        u32 byte_idx = elem / 8;
        u32 bit_idx  = elem % 8;
        if (val)
            data_[byte_idx] |=  (1u << bit_idx);
        else
            data_[byte_idx] &= ~(1u << bit_idx);
    }

    /// @return Mutable pointer to the raw storage of register @p vreg.
    [[nodiscard]] u8* reg_data(u32 vreg) noexcept
    {
        return &data_[vreg * vlenb()];
    }

    /// Zero all register storage.
    void reset() noexcept
    {
        std::fill(data_.begin(), data_.end(), u8{0});
    }

    [[nodiscard]] const VectorConfig& config() const noexcept { return config_; }

private:
    VectorConfig    config_;
    std::vector<u8> data_; ///< Flat storage: 32 registers, vlenb bytes each.
};

/// Complete vector extension state: registers, vtype CSR, and vector length.
struct VectorState {
    VectorRegFile regs;
    VType         vtype{};
    u32           vl     = 0;  ///< Current vector length. Elements i < vl are active.
    u32           vstart = 0;  ///< Reserved for future trap-resume support (currently unused).

    explicit VectorState(VectorConfig config = {})
        : regs(config) {}

    /**
     * @brief Computes VLMAX for the current VLEN and vtype.SEW.
     * @return Elements per register, or 0 if vtype is illegal.
     */
    [[nodiscard]] u32 vlmax() const noexcept
    {
        if (vtype.vill || vtype.sew == 0) return 0;
        return regs.config().vlen / vtype.sew;
    }

    /**
     * @brief Implements the vsetvli instruction.
     *
     * Sets vector configuration from @p zimm and calculates vl as
     * min(avl, VLMAX).
     *
     * @param avl  Application vector length requested.
     * @param zimm Immediate encoding for the new vtype.
     * @return The resulting vl.
     */
    u32 vsetvli(u32 avl, u32 zimm)
    {
        vtype = VType::decode(zimm);
        if (vtype.vill)
	{
            vl = 0;
            return 0;
        }
        u32 max = vlmax();
        vl = (avl <= max) ? avl : max;
        return vl;
    }

    /// Reset all vector state to power-on defaults.
    void reset()
    {
        regs.reset();
        vtype = VType{};
        vl = 0;
        vstart = 0;
    }
};

} // namespace riscv
