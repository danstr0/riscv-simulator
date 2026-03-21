/**
 * @file executor.cpp
 * @brief Implementation of the RV32I instruction executor.
 * * Key architectural behaviors, such as x0 hard-wiring and
 * JALR address alignment, are enforced here.
 */

#include "executor.hpp"

#include <iostream>
#include <format>

namespace riscv {

Executor::Executor(Memory& memory)
    : memory_(memory)
{
    reset();
}

void Executor::reset()
{
    regs_.fill(0);
    pc_ = 0;
    stats_.reset();
}

void Executor::dump_regs() const
{
    std::cout << std::format("--- Architectural State (PC: 0x{:08x}) ---\n", pc_);
    for (int i = 0; i < 32; i += 4) {
        for (int j = 0; j < 4; ++j) {
            std::cout << std::format("{:>4s}: 0x{:08x}  ",
                                     reg_name(static_cast<reg_idx_t>(i + j)),
                                     regs_[i + j]);
        }
        std::cout << "\n";
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Memory Operations
 * ═══════════════════════════════════════════════════════════════════════ */

ExecuteResult Executor::execute_load(const DecodedInst& inst)
{
    ExecuteResult result;

    /* Address calculation: rs1 + sign-extended immediate. */
    /* Spec §2.1.6: "The effective address is obtained by adding register r1
     * to the sign-extended 12-bit offset." */
    addr_t addr = regs_[inst.rs1] + static_cast<u32>(inst.imm);
    MemoryResult mem;

    switch(inst.op) {
        case Op::LB:
            mem = memory_.read8(addr);
            if (mem.ok) set_reg(inst.rd, static_cast<u32>(sign_extend<8>(mem.value)));
            break;
        case Op::LH:
            mem = memory_.read16(addr);
            if (mem.ok) set_reg(inst.rd, static_cast<u32>(sign_extend<16>(mem.value)));
            break;
        case Op::LW:
            mem = memory_.read32(addr);
            if (mem.ok) set_reg(inst.rd, mem.value);
            break;
        case Op::LBU:
            mem = memory_.read8(addr);
            if (mem.ok) set_reg(inst.rd, mem.value & 0xFFu);
            break;
        case Op::LHU:
            mem = memory_.read16(addr);
            if (mem.ok) set_reg(inst.rd, mem.value & 0xFFFFu);
            break;
        default:  [[unlikely]]
            result.ok = false;
            return result;
    }

    result.ok     = mem.ok;
    result.cycles = mem.cycles;
    if (mem.ok) {
        result.rd_value = regs_[inst.rd];
    }
    return result;
}

ExecuteResult Executor::execute_store(const DecodedInst& inst)
{
    ExecuteResult result;
    addr_t addr = regs_[inst.rs1] + static_cast<u32>(inst.imm);
    u32 value   = regs_[inst.rs2];
    MemoryResult mem;

    switch (inst.op) {
        case Op::SB: mem = memory_.write8(addr, static_cast<u8>(value)); break;
        case Op::SH: mem = memory_.write16(addr, static_cast<u16>(value)); break;
        case Op::SW: mem = memory_.write32(addr, value); break;
        default: [[unlikely]]
            result.ok = false;
            return result;
    }

    result.ok     = mem.ok;
    result.cycles = mem.cycles;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main Execute Dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

ExecuteResult Executor::execute(const DecodedInst& inst)
{
    ExecuteResult result;
    result.next_pc = pc_ + 4;  // Default to sequential

    u32 rs1_val = regs_[inst.rs1];
    u32 rs2_val = regs_[inst.rs2];

    switch (inst.op) {
        /* ----- Loads & Stores ----- */
        case Op::LB: case Op::LH: case Op::LW:
        case Op::LBU: case Op::LHU:
            result = execute_load(inst);
            result.next_pc = pc_ + 4;
            if (result.ok) stats_.loads++;
            break;

        case Op::SB: case Op::SH: case Op::SW:
            result = execute_store(inst);
            result.next_pc = pc_ + 4;
            if (result.ok) stats_.stores++;
            break;

        /* ----- Branches ----- */
        case Op::BEQ:  result.branch_taken = cond_eq(rs1_val, rs2_val);  goto branch_common;
        case Op::BNE:  result.branch_taken = cond_ne(rs1_val, rs2_val);  goto branch_common;
        case Op::BLT:  result.branch_taken = cond_lt(rs1_val, rs2_val);  goto branch_common;
        case Op::BGE:  result.branch_taken = cond_ge(rs1_val, rs2_val);  goto branch_common;
        case Op::BLTU: result.branch_taken = cond_ltu(rs1_val, rs2_val); goto branch_common;
        case Op::BGEU: result.branch_taken = cond_geu(rs1_val, rs2_val); goto branch_common;
        branch_common:
            result.next_pc = result.branch_taken 
                           ? (pc_ + inst.imm)
                           : (pc_ + 4);
            stats_.branches++;
            if (result.branch_taken) stats_.branches_taken++;
            break;

        /* ----- Jumps ----- */
        case Op::JAL:
            result.rd_value = pc_ + 4;
            set_reg(inst.rd, *result.rd_value);
            result.next_pc  = static_cast<addr_t>(pc_ + inst.imm);
            stats_.jumps++;
            break;

        case Op::JALR:
            result.rd_value = pc_ + 4;
            set_reg(inst.rd, *result.rd_value);
            /* Spec §2.1.5.1: The target address is (rs1 + imm), and the least-significant
             * bit is forced to zero. */
            result.next_pc = (rs1_val + static_cast<u32>(inst.imm)) & ~1u;
            stats_.jumps++;
            break;

        /* ----- Upper Immediates ----- */
        case Op::LUI:
            result.rd_value = static_cast<u32>(inst.imm);
            set_reg(inst.rd, *result.rd_value);
            break;

        case Op::AUIPC:
            result.rd_value = pc_ + static_cast<u32>(inst.imm);
            set_reg(inst.rd, *result.rd_value);
            break;

        /* ----- Arithmetic (Immediate) ----- */
        case Op::ADDI:  set_reg(inst.rd, alu_add(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::SLTI:  set_reg(inst.rd, alu_slt(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::SLTIU: set_reg(inst.rd, alu_sltu(rs1_val, static_cast<u32>(inst.imm))); break;
        case Op::XORI:  set_reg(inst.rd, alu_xor(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::ORI:   set_reg(inst.rd, alu_or(rs1_val, static_cast<u32>(inst.imm)));   break;
        case Op::ANDI:  set_reg(inst.rd, alu_and(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::SLLI:  set_reg(inst.rd, alu_sll(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::SRLI:  set_reg(inst.rd, alu_srl(rs1_val, static_cast<u32>(inst.imm)));  break;
        case Op::SRAI:  set_reg(inst.rd, alu_sra(rs1_val, static_cast<u32>(inst.imm)));  break;

        /* ----- Arithmetic (Register) ----- */
        case Op::ADD:  set_reg(inst.rd, alu_add(rs1_val, rs2_val));  break;
        case Op::SUB:  set_reg(inst.rd, alu_sub(rs1_val, rs2_val));  break;
        case Op::SLL:  set_reg(inst.rd, alu_sll(rs1_val, rs2_val));  break;
        case Op::SLT:  set_reg(inst.rd, alu_slt(rs1_val, rs2_val));  break;
        case Op::SLTU: set_reg(inst.rd, alu_sltu(rs1_val, rs2_val)); break;
        case Op::XOR:  set_reg(inst.rd, alu_xor(rs1_val, rs2_val));  break;
        case Op::SRL:  set_reg(inst.rd, alu_srl(rs1_val, rs2_val));  break;
        case Op::SRA:  set_reg(inst.rd, alu_sra(rs1_val, rs2_val));  break;
        case Op::OR:   set_reg(inst.rd, alu_or(rs1_val, rs2_val));   break;
        case Op::AND:  set_reg(inst.rd, alu_and(rs1_val, rs2_val));  break;

        /* ----- System ----- */
        case Op::ECALL:  result.ecall = true; break;
        case Op::EBREAK: result.ebreak = true; break;
        case Op::FENCE:  break;  // NOP in single-threaded context

        case Op::INVALID:
        default: [[unlikely]]
            result.ok = false;
            break;
    }

    /* Post-execution metadata update */
    if (result.ok) {
        if (inst.writes_rd() && !result.rd_value.has_value())
            result.rd_value = regs_[inst.rd];
        stats_.instructions++;
        stats_.cycles += result.cycles;
    }

    return result;
}

} // namespace riscv
