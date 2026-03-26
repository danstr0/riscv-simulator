/**
 * @file vector_state.hpp
 * @brief Architectural state for a simplified RISC-V Vector (RVV) 1.0 implementation.
 *
 * @section limitations Subset Limitations
 * This implementation targets a restricted subset of the RVV 1.0 specification:
 * - **SEW (Selected Element Width):** Fixed at 32-bit.
 * - **LMUL (Length Multiplier):** Fixed at 1 (no register grouping).
 * - **vstart:** Hardwired to 0 (no support for resumes after traps).
 * - **VLENL:** Must be a power of 2, minimum 64 bits.
 *
 * @see RISC-V Unprivileged ISA Specification, Chapter 30 ("V" Standard Extension for
 * Vector Operations, Version 1.0).
 */

#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace riscv {

/**
 * @brief Hardware configuration parameters for the Vector Unit.
 */
struct VectorConfig {
    /** @brief Vector register length in bits. Must be a power of 2 >= 64. */
    u32 vlen = 128;

    /** @return Number of bytes per vector register (VLENB). */
    [[nodiscard]] constexpr u32 vlenb() const noexcept { return vlen / 8; }

    /** @retur Maximum elements (SEW=32) that can fit in a single register. */
    [[nodiscard]] constexpr u32 vlmax_sew32() const noexcept { return vlen / 32; }

    /** @brief Validates hardware constraints (Power of 2, VLEN >= 64). */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return vlen >= 64 && std::has_single_bit(vlen);
    }
};

/**
 * @brief Represents the `vtype` CSR (Vector Type Register).
 * * Encodes how vector registers are interpreted.
 * 
 * Fields map to bits [0:7] and bit [31] of the vtype CSR.
 */
struct VType {
    u32 sew   = 32;     ///< Selected element width in bits.
    u32 lmul  = 1;      ///< Register grouping multiplier.
    bool vta  = false;  ///< Vector tail agnostic: if true, tail elements can be overwritten with 1s.
    bool vma  = false;  ///< Vector mask agnostic: if true, masked-off elements can be overwritten with 1s.
    bool vill = false;  ///< Illegal: if true, any vector instruction using this state traps.

    /**
     * @brief Encodes the struct into a 32-bit RISC-V CSR format.
     *
     * @return 32-bit encoded value (bit 31 is vill, bits 3-5 are vsew).
     */
    [[nodiscard]] constexpr u32 encode() const noexcept
    {
        if (vill) return 1u << 31;
        u32 vsew_field = 0;
        switch(sew) {
            case 8:  vsew_field = 0b000; break;
            case 16: vsew_field = 0b001; break;
            case 32: vsew_field = 0b010; break;
            case 64: vsew_field = 0b011; break;
            default: break;
        }
        return (static_cast<u32>(vma) << 7)
             | (static_cast<u32>(vta) << 6)
             | (vsew_field << 3);
    }

    /** @brief Decodes a `vsetvli` immediate into a VType struct. */
    static VType decode(u32 zimm) {
        VType vt;
        u32 vsew = (zimm >> 3) & 0x7;
        switch (vsew) {
            case 0b000: vt.sew = 8;  break;
            case 0b001: vt.sew = 16; break;
            case 0b010: vt.sew = 32; break;
            case 0b011: vt.sew = 64; break;
            default:    vt.vill = true; return vt;
        }
        // Only 32-bit supported
        if (vt.sew != 32) vt.vill = true;
        vt.vta = (zimm >> 6) & 1;
        vt.vma = (zimm >> 7) & 1;
        vt.lmul = 1;
        return vt;
    }
};

/**
 * @brief Managed storage for the 32 vector registers (v0-v31).
 *
 * * Uses a flat byte-buffer for performance. Mapping:
 * Offset = (RegisterIndex * VLENB) + (ElementIndex * (SEW/8))
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

    /**
     * @brief Read a 32-bit element.
     *
     * @param vreg Register index (0-31).
     * @param elem Element index (0 to VLMAX-1).
     */
    [[nodiscard]] u32 get_elem32(u32 vreg, u32 elem) const noexcept
    {
        assert(vreg < 32 && elem < vlmax());
        u32 value;
        std::memcpy(&value, &data_[(vreg * vlenb()) + (elem * 4)], 4);
        return value;
    }

    void set_elem32(u32 vreg, u32 elem, u32 value) noexcept
    {
        assert(vreg < 32 && elem < vlmax());
        std::memcpy(&data_[(vreg * vlenb()) + (elem * 4)], &value, 4);
    }

    /**
     * @brief Access mask bits stored in v0.
     *
     * @note In RVV, the mask for element @p i is bit @p i of register v0.
     */
    [[nodiscard]] bool get_mask_bit(u32 elem) const noexcept
    {
        u32 byte_idx = elem / 8;
        u32 bit_idx  = elem % 8;
        return (data_[byte_idx] >> bit_idx) & 1;  /* v0 starts at offset 0 */
    }

    void set_mask_bit(u32 elem, bool val) noexcept
    {
        u32 byte_idx = elem / 8;
        u32 bit_idx  = elem % 8;
        if (val)
            data_[byte_idx] |= (1u << bit_idx);
        else
            data_[byte_idx] &= ~(1u << bit_idx);
    }

    /** @brief Returns a pointer to the start of a specific vector register. */
    [[nodiscard]] u8* reg_data(u32 vreg) noexcept
    {
        return &data_[vreg * vlenb()];
    }

    void reset() noexcept
    {
        std::fill(data_.begin(), data_.end(), u8{0});
    }

    [[nodiscard]] const VectorConfig& config() const noexcept { return config_; }

private:
    VectorConfig    config_;
    std::vector<u8> data_;  ///< Flat storage: 32 registers × vlenb bytes.
};

/**
 * @brief Complete Vector Extension state including CSRs and Registers.
 */
struct VectorState {
    VectorRegFile regs;
    VType         vtype{};
    u32           vl     = 0;  ///< Current vector length. Elements i < vl are "active".
    u32           vstart = 0;  ///< Starting element index for instructions.

    explicit VectorState(VectorConfig config = {})
        : regs(config) {}

    /**
     * @brief Computes VLMAX based on the current hardware VLEN and software vtype.SEW.
     *
     * @return Elements per register, or 0 if vtype is illegal.
     */
    [[nodiscard]] u32 vlmax() const noexcept
    {
        if (vtype.vill || vtype.sew == 0) return 0;
        return regs.config().vlen / vtype.sew;  /* LMUL = 1 assumed */
    }

    /**
     * @brief Implements the `vsetvli` instruction logic.
     *
     * * Sets the vector configuration and calculates the active vector length (vl).
     *
     * @param avl Application vector length.
     * @param zimm The immediate encoding for the new vtype.
     *
     * @return The resulting `vl`.
     */
    u32 vsetvli(u32 avl, u32 zimm)
    {
        vtype = VType::decode(zimm);
        if (vtype.vill) {
            vl = 0;
            return 0;
        }
        u32 max = vlmax();
        vl = (avl <= max) ? avl : max;
        return vl;
    }

    /** @brief Wipes all vector state to power-on defaults. */
    void reset()
    {
        regs.reset();
        vtype = VType{};
        vl = 0;
        vstart = 0;
    }
};

} // namespace riscv
