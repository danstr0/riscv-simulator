/**
 * @file decoder.cpp
 * @brief Implementation of the RV32I instruction decoder.
 *
 * The decode pipeline follows a multi-level dispatch:
 * 1. Opcode Extraction: Slices bits [6:0] to determine the base @ref Opcode.
 * 2. Helper Dispatch: Hands off the word to a format-aware static method.
 * 3. Field Mapping: Reconstructs the internal @ref Op tag using funct3/funct7.
 *
 * @note All immediate reconstruction logic follows the mappings defined in
 * RISC-V Unprivileged ISA Spec §2.1.
 */

#include "decoder.hpp"

#include <format>

namespace riscv {

/* ═══════════════════════════════════════════════════════════════════════
 * Mnemonic Lookup
 * ═══════════════════════════════════════════════════════════════════════ */

const char* op_name(Op op) {
    switch (op) {
        case Op::LB:      return "lb";
        case Op::LH:      return "lh";
        case Op::LW:      return "lw";
        case Op::LBU:     return "lbu";
        case Op::LHU:     return "lhu";
        case Op::SB:      return "sb";
        case Op::SH:      return "sh";
        case Op::SW:      return "sw";
        case Op::BEQ:     return "beq";
        case Op::BNE:     return "bne";
        case Op::BLT:     return "blt";
        case Op::BGE:     return "bge";
        case Op::BLTU:    return "bltu";
        case Op::BGEU:    return "bgeu";
        case Op::JAL:     return "jal";
        case Op::JALR:    return "jalr";
        case Op::LUI:     return "lui";
        case Op::AUIPC:   return "auipc";
        case Op::ADDI:    return "addi";
        case Op::SLTI:    return "slti";
        case Op::SLTIU:   return "sltiu";
        case Op::XORI:    return "xori";
        case Op::ORI:     return "ori";
        case Op::ANDI:    return "andi";
        case Op::SLLI:    return "slli";
        case Op::SRLI:    return "srli";
        case Op::SRAI:    return "srai";
        case Op::ADD:     return "add";
        case Op::SUB:     return "sub";
        case Op::SLL:     return "sll";
        case Op::SLT:     return "slt";
        case Op::SLTU:    return "sltu";
        case Op::XOR:     return "xor";
        case Op::SRL:     return "srl";
        case Op::SRA:     return "sra";
        case Op::OR:      return "or";
        case Op::AND:     return "and";
        case Op::FENCE:   return "fence";
        case Op::ECALL:   return "ecall";
        case Op::EBREAK:  return "ebreak";
        case Op::INVALID: return "invalid";
    }
    return "???";
}

/* ═══════════════════════════════════════════════════════════════════════
 * Disassembly
 * ═══════════════════════════════════════════════════════════════════════ */

std::string DecodedInst::disassemble() const
{
    switch(format) {
        case Format::R:
            return std::format("{} {}, {}, {}",
                op_name(op), reg_name(rd), reg_name(rs1), reg_name(rs2));

        case Format::I:
            if (is_load() || op == Op::JALR) {
                return std::format("{} {}, {}({})",
                    op_name(op), reg_name(rd), imm, reg_name(rs1));
            }
            return std::format("{} {}, {}, {}",
                op_name(op), reg_name(rd), reg_name(rs1), imm);

        case Format::S:
            return std::format("{} {}, {}({})",
                op_name(op), reg_name(rs2), imm, reg_name(rs1));

        case Format::B:
            return std::format("{} {}, {}, 0x{:x}",
                op_name(op), reg_name(rs1), reg_name(rs2),
                static_cast<u32>(pc + imm));

        case Format::U:
            /* U-type immediates are displayed as shifted 20-bit hex values. */
            return std::format("{} {}, 0x{:x}",
                op_name(op), reg_name(rd), static_cast<u32>(imm) >> 12);

        case Format::J:
            return std::format("{} {}, 0x{:x}",
                op_name(op), reg_name(rd), static_cast<u32>(pc + imm));
    }
    return "???";
}

/* ═══════════════════════════════════════════════════════════════════════
 * Immediate extraction (Spec §2.1.3)
 * ═══════════════════════════════════════════════════════════════════════ */

i32 Decoder::extract_i_imm(u32 inst)
{
    /* I-type: imm[11:0] = inst[31:20] */
    return sign_extend<12>(inst >> 20);
}

i32 Decoder::extract_s_imm(u32 inst)
{
    /* S-type: imm[11:5]=inst[31:25], imm[4:0]=inst[11:7] */
    u32 imm = (bits(inst, 31, 25) << 5) | bits(inst, 11, 7);
    return sign_extend<12>(imm);
}

i32 Decoder::extract_b_imm(u32 inst)
{
    /* B-type: imm[12]=inst[31], imm[11]=inst[7], imm[10:5]=inst[30:25], imm[4:1]=inst[11:8] */
    /* Bit 0 is implicitly 0. */
    u32 imm = (bit(inst, 31)      << 12)
            | (bit(inst, 7)       << 11)
            | (bits(inst, 30, 25) << 5)
            | (bits(inst, 11, 8)  << 1);
    return sign_extend<13>(imm);
}

i32 Decoder::extract_u_imm(u32 inst)
{
    /* U-type: imm[31:12] = inst[31:12], lower 12 bits zeroed. */
    return static_cast<i32>(inst & 0xFFFFF000u);
}

i32 Decoder::extract_j_imm(u32 inst)
{
    /* J-type: imm[20]=inst[31], imm[19:12]=inst[19:12], imm[11]=inst[20], imm[10:1]=inst[30:21] */
    /* Bit 0 is implicitly 0. */
    u32 imm = (bit(inst, 31)      << 20)
            | (bits(inst, 19, 12) << 12)
            | (bit(inst, 20)      << 11)
            | (bits(inst, 30, 21) << 1);
    return sign_extend<21>(imm);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Per-opcode decode helpers
 * ═══════════════════════════════════════════════════════════════════════ */

DecodedInst Decoder::decode_load(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = bits(inst, 11, 7);
    d.rs1    = bits(inst, 19, 15);
    d.imm    = extract_i_imm(inst);

    switch (bits(inst, 14, 12)) {  /* funct3 */
        case 0b000: d.op = Op::LB;  break;
        case 0b001: d.op = Op::LH;  break;
        case 0b010: d.op = Op::LW;  break;
        case 0b100: d.op = Op::LBU; break;
        case 0b101: d.op = Op::LHU; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_store(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::S;
    d.raw    = inst;
    d.pc     = pc;
    d.rs1    = bits(inst, 19, 15);
    d.rs2    = bits(inst, 24, 20);
    d.imm    = extract_s_imm(inst);

    switch(bits(inst, 14 , 12)) {  /* funct3 */
        case 0b000: d.op = Op::SB; break;
        case 0b001: d.op = Op::SH; break;
        case 0b010: d.op = Op::SW; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_branch(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::B;
    d.raw    = inst;
    d.pc     = pc;
    d.rs1    = bits(inst, 19, 15);
    d.rs2    = bits(inst, 24, 20);
    d.imm    = extract_b_imm(inst);

    switch (bits(inst, 14, 12)) {  /* funct3 */
        case 0b000: d.op = Op::BEQ;  break;
        case 0b001: d.op = Op::BNE;  break;
        case 0b100: d.op = Op::BLT;  break;
        case 0b101: d.op = Op::BGE;  break;
        case 0b110: d.op = Op::BLTU; break;
        case 0b111: d.op = Op::BGEU; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_op_imm(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = bits(inst, 11, 7);
    d.rs1    = bits(inst, 19, 15);
    d.imm    = extract_i_imm(inst);

    u32 funct3 = bits(inst, 14, 12);
    u32 funct7 = bits(inst, 31, 25);

    switch (funct3) {
        case 0b000: d.op = Op::ADDI;  break;
        case 0b010: d.op = Op::SLTI;  break;
        case 0b011: d.op = Op::SLTIU; break;
        case 0b100: d.op = Op::XORI;  break;
        case 0b110: d.op = Op::ORI;   break;
        case 0b111: d.op = Op::ANDI;  break;
        
        case 0b001:  /* SLLI */
            if (funct7 == 0b0000000) {
                d.op  = Op::SLLI;
                d.imm = bits(inst, 24, 20);  // shamt
            } else {
                d.op = Op::INVALID;
            }
            break;

        case 0b101:  /* SRLI / SRAI */
            d.imm = bits(inst, 24, 20);  // shamt
            if (funct7 == 0b0000000)      d.op = Op::SRLI;
            else if (funct7 == 0b0100000) d.op = Op::SRAI;
            else                          d.op = Op::INVALID;
            break;

        default: d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_op(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::R;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = bits(inst, 11, 7);
    d.rs1    = bits(inst, 19, 15);
    d.rs2    = bits(inst, 24, 20);

    u32 funct3 = bits(inst, 14, 12);
    u32 funct7 = bits(inst, 31, 25);

    switch (funct3) {
        case 0b000:
            d.op = (funct7 == 0b0000000) ? Op::ADD
                 : (funct7 == 0b0100000) ? Op::SUB
                 : Op::INVALID;
            break;
        case 0b001: d.op = (funct7 == 0b0000000) ? Op::SLL  : Op::INVALID; break;
        case 0b010: d.op = (funct7 == 0b0000000) ? Op::SLT  : Op::INVALID; break;
        case 0b011: d.op = (funct7 == 0b0000000) ? Op::SLTU : Op::INVALID; break;
        case 0b100: d.op = (funct7 == 0b0000000) ? Op::XOR  : Op::INVALID; break;
        case 0b101:
            d.op = (funct7 == 0b0000000) ? Op::SRL
                 : (funct7 == 0b0100000) ? Op::SRA
                 : Op::INVALID;
            break;
        case 0b110: d.op = (funct7 == 0b0000000) ? Op::OR   : Op::INVALID; break;
        case 0b111: d.op = (funct7 == 0b0000000) ? Op::AND  : Op::INVALID; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_system(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = bits(inst, 11, 7);
    d.rs1    = bits(inst, 19, 15);
    d.imm    = extract_i_imm(inst);

    u32 funct3 = bits(inst, 14, 12);

    if (funct3 == 0b000) {
        if (inst == 0x00000073)      d.op = Op::ECALL;
        else if (inst == 0x00100073) d.op = Op::EBREAK;
        else                         d.op = Op::INVALID;
    } else {
        d.op = Op::INVALID;  // CSR operations not in base RV32I
    }
    return d;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main Decode Entry Point
 * ═══════════════════════════════════════════════════════════════════════ */

DecodedInst Decoder::decode(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.raw = inst;
    d.pc  = pc;

    auto opcode = static_cast<Opcode>(bits(inst, 6, 0));

    switch (opcode) {
        case Opcode::LOAD:   return decode_load(inst, pc);
        case Opcode::STORE:  return decode_store(inst, pc);
        case Opcode::BRANCH: return decode_branch(inst, pc);
        case Opcode::OP_IMM: return decode_op_imm(inst, pc);
        case Opcode::OP:     return decode_op(inst, pc);
        case Opcode::SYSTEM: return decode_system(inst, pc);

        case Opcode::LUI:
        case Opcode::AUIPC:
            d.format = Format::U;
            d.op     = (opcode == Opcode::LUI) ? Op::LUI : Op::AUIPC;
            d.rd     = bits(inst, 11, 7);
            d.imm    = extract_u_imm(inst);
            return d;

        case Opcode::JAL:
            d.format = Format::J;
            d.op     = Op::JAL;
            d.rd     = bits(inst, 11, 7);
            d.imm    = extract_j_imm(inst);
            return d;

        case Opcode::JALR:
            if (bits(inst, 14, 12) != 0b000) { d.op = Op::INVALID; return d; }
            d.format = Format::I;
            d.op     = Op::JALR;
            d.rd     = bits(inst, 11, 7);
            d.rs1    = bits(inst, 19, 15);
            d.imm    = extract_i_imm(inst);
            return d;

        case Opcode::MISC_MEM:
            d.format = Format::I;
            d.op     = Op::FENCE;
            d.rd     = bits(inst, 11, 7);
            d.rs1    = bits(inst, 19, 15);
            d.imm    = extract_i_imm(inst);
            return d;

        default:
            d.op = Op::INVALID;
            return d;
    }
}

} // namespace riscv
