/**
 * @file decoder.hpp
 * @brief Instruction decoder for the RV32I base integer instruction set.
 *
 * Implements the full decode stage: opcode dispatch, field extraction, and
 * immediate reconstruction for all six instruction formats (R, I, S, B, U, J).
 *
 * @section DECODE_PROCESS Decode Process
 * 1. **Opcode Identification:** The lower 7 bits ([6:0]) determine the @ref Format.
 * 2. **Field Extraction:** Standard fields (rd, rs1, rs2, funct3, funct7) are sliced.
 * 3. **Immediate Reconstruction:** Bit-shuffling is performed according to the
 * instruction format to produce a sign-extended 32-bit immediate.
 *
 * @see RISC-V Unprivileged ISA Specification §2.1, §6.1.1, §12.1, §13.1, §30.1.
 */

#pragma once

#include "types.hpp"

#include <string>

namespace riscv {

/**
 * @brief Primary Opcodes (inst[6:0]).
 *
 * Encodings for the RV32I base ISA, as well as the RV32A extension,
 * are defined here.
 */
enum class Opcode : u8 {
    LOAD      = 0b0000011,  ///< LB, LH, LW, LBU, LHU
    MISC_MEM  = 0b0001111,  ///< FENCE
    OP_IMM    = 0b0010011,  ///< ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
    AUIPC     = 0b0010111,  ///< AUIPC
    STORE     = 0b0100011,  ///< SB, SH, SW
    AMO       = 0b0101111,  ///< LR.W, SC.W, AMO*
    OP        = 0b0110011,  ///< ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
    LUI       = 0b0110111,  ///< LUI
    BRANCH    = 0b1100011,  ///< BEQ, BNE, BLT, BGE, BLTU, BGEU
    JALR      = 0b1100111,  ///< JALR
    JAL       = 0b1101111,  ///< JAL
    SYSTEM    = 0b1110011,  ///< ECALL, EBREAK
    VL        = 0b0000111,  ///< Vector loads (VLE32, etc.)	
    VS        = 0b0100111,  ///< Vector stores (VSE32, etc.)
    OP_V      = 0b1010111,  ///< Vector ALU + VSETVLI
};

/**
 * @brief Decoded Operation Tags.
 * * Represents the specific architectural operation to be performed.
 */
enum class Op : u8 {
    /* Loads (I-type) */
    LB, LH, LW, LBU, LHU,

    /* Stores (S-type) */
    SB, SH, SW,

    /* Branches (B-type) */
    BEQ, BNE, BLT, BGE, BLTU, BGEU,

    /* Jumps */
    JAL,   ///< J-type
    JALR,  ///< I-type

    /* Upper immediate (U-type) */
    LUI, AUIPC,

    /* Register-immediate arithmetic (I-type) */
    ADDI, SLTI, SLTIU, XORI, ORI, ANDI,
    SLLI, SRLI, SRAI,

    /* Register-register arithmetic (R-type) */
    ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,

    /* RV32M integer multiply/divide (R-type, funct7 = 0000001) */
    MUL, MULH, MULHSU, MULHU,
    DIV, DIVU, REM, REMU,

    /* RV32A atomic memory operations (R-type) */
    LR_W,       ///< Load-reserved word
    SC_W,       ///< Store-conditional word
    AMOSWAP_W,  ///< Atomic swap
    AMOADD_W,   ///< Atomic add
    AMOXOR_W,   ///< Atomic XOR
    AMOAND_W,   ///< Atomic AND
    AMOOR_W,    ///< Atomic OR 
    AMOMIN_W,   ///< Atomic signed minimum
    AMOMAX_W,   ///< Atomic signed maximum
    AMOMINU_W,  ///< Atomic unsigned minimum
    AMOMAXU_W,  ///< Atomic unsigned maximum

    /* RVV vector extension subset (opcode = 1010111 for ALU, 0000111/0100111 for load/store) */
    VSETVLI,             ///< Set vector length (immediate vtype)
    VSETIVLI,            ///< Set vector length (immediate AVL + vtype)
    VLE32,               ///< Vector load, 32-bit elements, unit-stride
    VSE32,               ///< Vector store, 32-bit elements, unit-stride
    VADD_VV, VADD_VX,    ///< Vector add
    VSUB_VV, VSUB_VX,    ///< Vector subtract
    VAND_VV, VAND_VX,    ///< Vector AND
    VOR_VV,  VOR_VX,     ///< Vector OR
    VXOR_VV, VXOR_VX,    ///< Vector XOR
    VSLL_VX,             ///< Vector shift left by scalar
    VSRL_VX,             ///< Vector shift right by scalar
    VMSEQ_VV, VMSEQ_VX,  ///< Set mask if equal
    VMSLT_VV,            ///< Set mask if less-than (signed)
    VMSLTU_VV,           ///< Set mask if less-than (unsigned)
    VMAND_MM,            ///< Mask AND
    VMNAND_MM,           ///< Mask NAND
    VMANDN_MM,           ///< Mask AND-NOT
    VMXOR_MM,            ///< Mask XOR
    VMOR_MM,             ///< Mask OR
    VMNOR_MM,            ///< Mask NOR
    VMORN_MM,            ///< Mask OR-NOT
    VMXNOR_MM,           ///< Mask XNOR
    VREDSUM_VS,          ///< Reduction: sum
    VMV_V_X,             ///< Splat scalar to vector
    VMV_X_S,             ///< Extract element 0 to scalar

    /* System / synchronization */
    FENCE,
    ECALL,
    EBREAK,

    /* CSR access */
    CSRRW,   ///< Atomic read/write CSR
    CSRRS,   ///< Atomic read and set bits
    CSRRC,   ///< Atomic read and clear bits
    CSRRWI,  ///< Immediate variant of CSRRW
    CSRRSI,  ///< Immediate variant of CSRRS
    CSRRCI,  ///< Immediate variant of CSRRC

    /* Trap return */
    MRET,    ///< Return from machine-mode trap

    /** Sentinel for unrecognized or malformed instructions. */
    INVALID,
};

/**
 * @brief Instruction Formats as defined by the ISA.
 * * Each format dictates how the 32-bit word is partitioned into registers
 * and immediates.
 */
enum class Format : u8 {
    R,  ///< Reg-Reg:   [funct7][rs2][rs1][funct3][rd][opcode]
    I,  ///< Immediate: [imm11:0][rs1][funct3][rd][opcode]
    S,  ///< Store:     [imm11:5][rs2][rs1][funct3][imm4:0][opcode]
    B,  ///< Branch:    [imm12][imm10:5][rs2][rs1][funct3][imm4:1][imm11][opcode]
    U,  ///< Upper Imm: [imm31:12][rd][opcode]
    J,  ///< Jump:      [imm20][imm10:1][imm11][imm19:12][rd][opcode]
};

/**
 * @brief Fully decoded representation of a RV32I instruction.
 *
 * @note **Invariants maintained by the decoder:**
 * - @c imm is always sign-extended to 32 bits.
 * - For **U-type**, @c imm contains the value with trailing zeros (shifted left 12).
 * - For **Shift-Immediates**, @c imm contains only the 5-bit `shamt` [4:0].
 */
struct DecodedInst {
    Op     op     = Op::INVALID;  ///< The abstract operation.
    Format format = Format::R;    ///< The instruction's layout format.
    
    reg_idx_t rd  = 0;  ///< Destination register [11:7].
    reg_idx_t rs1 = 0;  ///< Source register 1 [19:15].
    reg_idx_t rs2 = 0;  ///< Source register 2 [24:20].
    i32       imm = 0;  ///< Sign-extended immediate value.
    
    u32    raw = 0;     ///< Original 32-bit instruction word for debugging.
    addr_t pc  = 0;     ///< Address of this instruction.
    
    /* ────────── Query Helpers ────────── */

    /** @brief Returns true if this instruction updates a destination register. */
    [[nodiscard]] constexpr bool writes_rd() const noexcept
    {
        switch (op) {
            case Op::SB: case Op::SH: case Op::SW:
            case Op::BEQ: case Op::BNE: case Op::BLT:
            case Op::BGE: case Op::BLTU: case Op::BGEU:
            case Op::FENCE: case Op::ECALL: case Op::EBREAK:
            case Op::INVALID:
                return false;
            default:
                return rd != 0;  ///< x0 is immutable.
        }
    }
    
    [[nodiscard]] constexpr bool reads_rs1() const noexcept
    {
        switch (op) {
            case Op::LUI: case Op::AUIPC: case Op::JAL:
            case Op::FENCE: case Op::ECALL: case Op::EBREAK:
            case Op::INVALID:
                return false;
            default:
                return true;
        }
    }
    
    [[nodiscard]] constexpr bool reads_rs2() const noexcept
    {
        return (format == Format::R || format == Format::S || format == Format::B);
    }
    
    [[nodiscard]] constexpr bool is_load()   const noexcept { return op >= Op::LB && op <= Op::LHU; }
    [[nodiscard]] constexpr bool is_store()  const noexcept { return op >= Op::SB && op <= Op::SW; }
    [[nodiscard]] constexpr bool is_branch() const noexcept { return op >= Op::BEQ && op <= Op::BGEU; }
    [[nodiscard]] constexpr bool is_jump()   const noexcept { return op == Op::JAL || op == Op::JALR; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return op >= Op::LR_W && op <= Op::AMOMAXU_W; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return op >= Op::VSETVLI && op <= Op::VMV_X_S; }

    /** @brief Generates standard RISC-V assembly string (e.g., "addi a0, sp, 16"). */
    [[nodiscard]] std::string disassemble() const;
};

/**
 * @brief Static Instruction Decoder.
 *
 * Transforms raw machine code into executable structures. This class is
 * purely functional and maintains no state.
 */
class Decoder {
public:
    Decoder() = delete;

    /**
     * @brief Decodes a 32-bit RISC-V instruction word.
     *
     * @param instruction  The raw word fetched from memory.
     * @param pc  The program counter where this instruction resides.
     * @return DecodedInst structure. Chec @c .op == Op::INVALID for decode failures.
     */
    [[nodiscard]] static DecodedInst decode(u32 instruction, addr_t pc = 0);
    
private:
    /* Internal dispatchers grouped by major opcode */
    static DecodedInst decode_load(u32 inst, addr_t pc);
    static DecodedInst decode_store(u32 inst, addr_t pc);
    static DecodedInst decode_branch(u32 inst, addr_t pc);
    static DecodedInst decode_op_imm(u32 inst, addr_t pc);
    static DecodedInst decode_op(u32 inst, addr_t pc);
    static DecodedInst decode_amo(u32 inst, addr_t pc);
    static DecodedInst decode_vector(u32 inst, addr_t pc);
    static DecodedInst decode_system(u32 inst, addr_t pc);
    
    /**
     * @name Immediate Extraction
     * Helpers for reconstructing immediates from scrambled bitfields.
     * See Specification §2.1.3, Figure 1.
     * @{
     */
    [[nodiscard]] static i32 extract_i_imm(u32 inst);
    [[nodiscard]] static i32 extract_s_imm(u32 inst);
    [[nodiscard]] static i32 extract_b_imm(u32 inst);
    [[nodiscard]] static i32 extract_u_imm(u32 inst);
    [[nodiscard]] static i32 extract_j_imm(u32 inst);
    /** @} */
};

/** @brief Returns the lowercase mnemonic (e.g., "lui") for an operation. */
[[nodiscard]] const char* op_name(Op op);

} // namespace riscv
