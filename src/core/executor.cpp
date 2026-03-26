/**
 * @file executor.cpp
 * @brief Implementation of the RV32IMA + RVV instruction executor.
 * * Key architectural behaviors, such as x0 hard-wiring and
 * JALR address alignment, are enforced here.
 */

#include "executor.hpp"

#include <format>
#include <iostream>

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
    reservation_.reset();
    vstate_.reset();
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

    /* Any store to the reserved address invalidates the reservation */
    if (reservation_.has_value()) {
    	addr_t res = reservation_.value();
	addr_t store_end = addr;
	switch(inst.op) {
	    case Op::SB: store_end = addr; break;
        case Op::SH: store_end = addr + 1; break;
        case Op::SW: store_end = addr + 3; break;
        default: break;
	}
    if (addr <= res + 3 && store_end >= res)
        reservation_.reset();
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main Execute Dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

ExecuteResult Executor::execute(const DecodedInst& inst)
{
    ExecuteResult result;
    result.next_pc = pc_ + 4;  // Default to sequential

    u32 rs1 = regs_[inst.rs1];
    u32 rs2 = regs_[inst.rs2];

    switch (inst.op) {
        /* Loads & Stores */
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

        /* Branches */
        case Op::BEQ:  result.branch_taken = cond_eq(rs1, rs2);  goto branch_common;
        case Op::BNE:  result.branch_taken = cond_ne(rs1, rs2);  goto branch_common;
        case Op::BLT:  result.branch_taken = cond_lt(rs1, rs2);  goto branch_common;
        case Op::BGE:  result.branch_taken = cond_ge(rs1, rs2);  goto branch_common;
        case Op::BLTU: result.branch_taken = cond_ltu(rs1, rs2); goto branch_common;
        case Op::BGEU: result.branch_taken = cond_geu(rs1, rs2); goto branch_common;
        branch_common:
            result.next_pc = result.branch_taken 
                           ? (pc_ + inst.imm)
                           : (pc_ + 4);
            stats_.branches++;
            if (result.branch_taken) stats_.branches_taken++;
            break;

        /* Jumps */
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
            result.next_pc = (rs1 + static_cast<u32>(inst.imm)) & ~1u;
            stats_.jumps++;
            break;

        /* Upper Immediates */
        case Op::LUI:
            result.rd_value = static_cast<u32>(inst.imm);
            set_reg(inst.rd, *result.rd_value);
            break;

        case Op::AUIPC:
            result.rd_value = pc_ + static_cast<u32>(inst.imm);
            set_reg(inst.rd, *result.rd_value);
            break;

        /* Arithmetic (Immediate) */
        case Op::ADDI:  set_reg(inst.rd, alu_add(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::SLTI:  set_reg(inst.rd, alu_slt(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::SLTIU: set_reg(inst.rd, alu_sltu(rs1, static_cast<u32>(inst.imm))); break;
        case Op::XORI:  set_reg(inst.rd, alu_xor(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::ORI:   set_reg(inst.rd, alu_or(rs1, static_cast<u32>(inst.imm)));   break;
        case Op::ANDI:  set_reg(inst.rd, alu_and(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::SLLI:  set_reg(inst.rd, alu_sll(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::SRLI:  set_reg(inst.rd, alu_srl(rs1, static_cast<u32>(inst.imm)));  break;
        case Op::SRAI:  set_reg(inst.rd, alu_sra(rs1, static_cast<u32>(inst.imm)));  break;

        /* Arithmetic (Register) */
        case Op::ADD:  set_reg(inst.rd, alu_add(rs1, rs2));  break;
        case Op::SUB:  set_reg(inst.rd, alu_sub(rs1, rs2));  break;
        case Op::SLL:  set_reg(inst.rd, alu_sll(rs1, rs2));  break;
        case Op::SLT:  set_reg(inst.rd, alu_slt(rs1, rs2));  break;
        case Op::SLTU: set_reg(inst.rd, alu_sltu(rs1, rs2)); break;
        case Op::XOR:  set_reg(inst.rd, alu_xor(rs1, rs2));  break;
        case Op::SRL:  set_reg(inst.rd, alu_srl(rs1, rs2));  break;
        case Op::SRA:  set_reg(inst.rd, alu_sra(rs1, rs2));  break;
        case Op::OR:   set_reg(inst.rd, alu_or(rs1, rs2));   break;
        case Op::AND:  set_reg(inst.rd, alu_and(rs1, rs2));  break;

        /* RV32M Multiply / Divide */
        case Op::MUL:    set_reg(inst.rd, alu_mul(rs1, rs2));    break;
        case Op::MULH:   set_reg(inst.rd, alu_mulh(rs1, rs2));   break;
        case Op::MULHSU: set_reg(inst.rd, alu_mulhsu(rs1, rs2)); break;
        case Op::MULHU:  set_reg(inst.rd, alu_mulhu(rs1, rs2));  break;
        case Op::DIV:    set_reg(inst.rd, alu_div(rs1, rs2));    break;
        case Op::DIVU:   set_reg(inst.rd, alu_divu(rs1, rs2));   break;
        case Op::REM:    set_reg(inst.rd, alu_rem(rs1, rs2));    break;
        case Op::REMU:   set_reg(inst.rd, alu_remu(rs1, rs2));   break;
        
        /* RV32A Atomic Operations */
        case Op::LR_W: {
            auto mem = memory_.read32(rs1);
            if (mem.ok) {
                set_reg(inst.rd, mem.value);
                reservation_ = rs1;
                result.rd_value = mem.value;
            }
            result.ok = mem.ok;
            result.cycles = mem.cycles;
            break;
        }
        case Op::SC_W: {
            if (reservation_.has_value() && reservation_.value() == rs1) {
                auto mem = memory_.write32(rs1, rs2);
                set_reg(inst.rd, 0);  /* success */
                result.rd_value = 0;
                result.ok = mem.ok;
                result.cycles = mem.cycles;
            } else {
                set_reg(inst.rd, 1);  /* failure */
                result.rd_value = 1;
            }
            reservation_.reset();
            break;
        }
        case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
        case Op::AMOAND_W:  case Op::AMOOR_W:
        case Op::AMOMIN_W:  case Op::AMOMAX_W:
        case Op::AMOMINU_W: case Op::AMOMAXU_W: {
            auto mem = memory_.read32(rs1);
            if (!mem.ok) { result.ok = false; break; }
            u32 old_val = mem.value;
            u32 new_val;
            switch(inst.op) {
                case Op::AMOSWAP_W: new_val = rs2; break;
                case Op::AMOADD_W:  new_val = old_val + rs2; break;
                case Op::AMOXOR_W:  new_val = old_val ^ rs2; break;
                case Op::AMOAND_W:  new_val = old_val & rs2; break;
                case Op::AMOOR_W:   new_val = old_val | rs2; break;
                case Op::AMOMIN_W:  new_val = (static_cast<i32>(old_val) < static_cast<i32>(rs2)) ? old_val
                                                                                                  : rs2; break;
                case Op::AMOMAX_W:  new_val = (static_cast<i32>(old_val) > static_cast<i32>(rs2)) ? old_val
                                                                                                  : rs2; break;
                case Op::AMOMINU_W: new_val = (old_val < rs2) ? old_val : rs2; break;
                case Op::AMOMAXU_W: new_val = (old_val > rs2) ? old_val : rs2; break;
                default: new_val = old_val; break;
            }
            memory_.write32(rs1, new_val);
            set_reg(inst.rd, old_val);
            result.rd_value = old_val;
            result.cycles = mem.cycles;
            break;
        }

        /* System */
        case Op::ECALL:  result.ecall = true; break;
        case Op::EBREAK: result.ebreak = true; break;
        case Op::FENCE:  break;  // NOP in single-threaded context

        /* RVV vector operations */
        case Op::VSETVLI:  case Op::VSETIVLI:
        case Op::VLE32:    case Op::VSE32:
        case Op::VADD_VV:  case Op::VADD_VX:
        case Op::VSUB_VV:  case Op::VSUB_VX:
        case Op::VAND_VV:  case Op::VAND_VX:
        case Op::VOR_VV:   case Op::VOR_VX:
        case Op::VXOR_VV:  case Op::VXOR_VX:
        case Op::VSLL_VX:  case Op::VSRL_VX:
        case Op::VMSEQ_VV: case Op::VMSEQ_VX:
        case Op::VMSLT_VV: case Op::VMSLTU_VV:
        case Op::VMAND_MM: case Op::VMOR_MM: case Op::VMNOT_M:
        case Op::VREDSUM_VS:
        case Op::VMV_V_X:  case Op::VMV_X_S:
            return execute_vector(inst, rs1, rs2);

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

/* ═══════════════════════════════════════════════════════════════════════
 * Vector execution 
 * ═══════════════════════════════════════════════════════════════════════ */

ExecuteResult Executor::execute_vector(const DecodedInst& inst, u32 rs1, u32 rs2)
{
    ExecuteResult result;
    result.next_pc = pc_ + 4;

    auto& vs = vstate_;
    u32   vl = vs.vl;

    switch (inst.op) {
        /* Configuration */
        case Op::VSETVLI: {
            u32 avl = (inst.rs1 == 0 && inst.rd == 0)
                    ? vs.vl            // keep current vl
                    : (inst.rs1 == 0)
                    ? ~u32{0}          // set vl=VLMAX 
                    : rs1;
            
            u32 new_vl = vs.vsetvli(avl, static_cast<u32>(inst.imm));
            set_reg(inst.rd, new_vl);
            result.rd_value = new_vl;
            break;
        }
        case Op::VSETIVLI: {
            u32 avl = inst.rs1;
            u32 new_vl = vs.vsetvli(avl, static_cast<u32>(inst.imm));
            set_reg(inst.rd, new_vl);
            result.rd_value = new_vl;
            break;
        }
        
        /* Vector load (unit-stride, SEW=32) */
        case Op::VLE32: {
            addr_t base = rs1;
            for (u32 i = 0; i < vl; ++i) {
                auto r = memory_.read32(base + i * 4);
                if (!r.ok) { result.ok = false; return result; }
                vs.regs.set_elem32(inst.rd, i, r.value);
            }
            result.cycles = vl;  /* 1 cycle per element */
            break;
        }

        /* Vector store (unit-stride, SEW=32) */
        case Op::VSE32: {
            addr_t base = rs1;
            for (u32 i = 0; i < vl; ++i) {
                u32 val = vs.regs.get_elem32(inst.rd, i);
                auto r = memory_.write32(base + i * 4, val);
                if (!r.ok) { result.ok = false; return result; }
            }
            result.cycles = vl;
            break;
        }

        /* Arithmetic VV */
        case Op::VADD_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) + vs.regs.get_elem32(inst.rs1, i));
            break;
        case Op::VSUB_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) - vs.regs.get_elem32(inst.rs1, i));
            break;
        case Op::VAND_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) & vs.regs.get_elem32(inst.rs1, i));
            break;
        case Op::VOR_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) | vs.regs.get_elem32(inst.rs1, i));
            break;
        case Op::VXOR_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) ^ vs.regs.get_elem32(inst.rs1, i));
            break;

        /* Arithmetic VX (scalar broadcast) */
        case Op::VADD_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) + rs1);
            break;
        case Op::VSUB_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) - rs1);
            break;
        case Op::VAND_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) & rs1);
            break;
        case Op::VOR_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) | rs1);
            break;
        case Op::VXOR_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) ^ rs1);
            break;
        case Op::VSLL_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) << (rs1 & 0x1F));
            break;
        case Op::VSRL_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i,
                                   vs.regs.get_elem32(inst.rs2, i) >> (rs1 & 0x1F));
            break;

        /* Comparisons (write mask to vd) */
        case Op::VMSEQ_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_mask_bit(i,
                                     vs.regs.get_elem32(inst.rs2, i) == vs.regs.get_elem32(inst.rs1, i));
            break;
        case Op::VMSEQ_VX:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_mask_bit(i,
                                     vs.regs.get_elem32(inst.rs2, i) == rs1);
            break;
        case Op::VMSLT_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_mask_bit(i,
                                     static_cast<i32>(vs.regs.get_elem32(inst.rs2, i)) <
                                     static_cast<i32>(vs.regs.get_elem32(inst.rs1, i)));
            break;
        case Op::VMSLTU_VV:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_mask_bit(i,
                                     vs.regs.get_elem32(inst.rs2, i) < vs.regs.get_elem32(inst.rs1, i));
            break;

        /* Mask operations */
        case Op::VMAND_MM:
            for (u32 i = 0; i < vl; ++i) {
                u32 b2 = vs.regs.get_elem32(inst.rs2, i / 32);
                u32 b1 = vs.regs.get_elem32(inst.rs1, i / 32);
                bool bit2 = (b2 >> (i % 32)) & 1;
                bool bit1 = (b1 >> (i % 32)) & 1;
                vs.regs.set_mask_bit(i, bit2 & bit1);
            }
            break;
        case Op::VMOR_MM:
            for (u32 i = 0; i < vl; ++i) {
                u32 b2 = vs.regs.get_elem32(inst.rs2, i / 32);
                u32 b1 = vs.regs.get_elem32(inst.rs1, i / 32);
                bool bit2 = (b2 >> (i % 32)) & 1;
                bool bit1 = (b1 >> (i % 32)) & 1;
                vs.regs.set_mask_bit(i, bit2 | bit1);
            }
            break;
        case Op::VMNOT_M:
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_mask_bit(i, !vs.regs.get_mask_bit(i));
            break;

        /* Reduction */
        case Op::VREDSUM_VS: {
            /* vd[0] = vs1[0] + sum(vs2[0..vl-1]) */
            u32 acc = vs.regs.get_elem32(inst.rs1, 0);
            for (u32 i = 0; i < vl; ++i)
                acc += vs.regs.get_elem32(inst.rs2, i);
            vs.regs.set_elem32(inst.rd, 0, acc);
            break;
        }

        /* Move */
        case Op::VMV_V_X:
            /* Splat scalar rs1 to all elements of vd */
            for (u32 i = 0; i < vl; ++i)
                vs.regs.set_elem32(inst.rd, i, rs1);
            break;
        case Op::VMV_X_S:
            /* Extract element 0 of vs2 into scalar rd */
            set_reg(inst.rd, vs.regs.get_elem32(inst.rs2, 0));
            result.rd_value = regs_[inst.rd];
            break;

        default:
            result.ok = false;
            break;
    }

    if (result.ok) {
        stats_.instructions++;
        stats_.cycles += result.cycles;
    }

    return result;
}

} // namespace riscv
