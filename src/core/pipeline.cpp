/**
 * @file pipeline.cpp
 * @brief Implementation of the 5-stage synchronous RISC-V datapath.
 *
 * @section timing_model Timing Model: Atomic State Transition
 * This simulator utilizes a double-buffered approach to model clock edges.
 * In each call to tick():
 * 1. A **snapshot** of the current pipeline registers (@p cur) is taken.
 * 2. The **next state** (@p next) is computed based on the snapshot.
 * 3. The state is **committed** back to the member variables at the end
 * of the cycle.
 *
 * This ensures that data dependencies within a single clock cycle are resolved
 * correctly, mimicking the behavior of D-flip-flops in RTL.
 */

#include "pipeline.hpp"

#include <cassert>
#include <format>
#include <iostream>
#include <limits>

namespace riscv {

// ── Construction / lifecycle ───────────────────────────────────────────

PipelinedCPU::PipelinedCPU(std::shared_ptr<Memory> memory, PipelineConfig config)
    : memory_(std::move(memory))
    , config_(config)
    , bht_(config.bht_size, 0)
{
    assert(memory_ != nullptr);
    reset();
}

void PipelinedCPU::reset()
{
    regs_.fill(0);
    pc_ = 0;
    for (auto& s : stages_) s.clear();
    halted_ = false;
    flush_pipeline_ = false;
    reservation_.reset();
    vstate_.reset();
    stats_.reset();
    std::fill(bht_.begin(), bht_.end(), u8{0});
}

void PipelinedCPU::set_config(const PipelineConfig& cfg)
{
    config_ = cfg;
    bht_.assign(cfg.bht_size, 0);
}

void PipelinedCPU::load_program(addr_t addr, std::span<const u8> program)
{
    memory_->load(addr, program);
}

void PipelinedCPU::load_instruction(addr_t addr, u32 instruction)
{
    const std::array<u8, 4> bytes = {
        static_cast<u8>(instruction),
        static_cast<u8>(instruction >> 8),
        static_cast<u8>(instruction >> 16),
        static_cast<u8>(instruction >> 24),
    };
    memory_->load(addr, bytes);
}

bool PipelinedCPU::pipeline_empty() const noexcept
{
    for (const auto& s : stages_)
        if (s.valid) return false;

    return true;
}

// ── Main tick ──────────────────────────────────────────────────────────

bool PipelinedCPU::tick()
{
    if (halted_ && pipeline_empty())
        return false;
    
    stats_.cycles++;
    
    /* 1. Hardware Snapshot */
    auto cur = stages_;
    std::array<PipelineReg, kNumStages> next{};
    
    /* 2. WB: Retire and Commit to Architectural State */
    {
        const auto& wb = cur[static_cast<int>(Stage::WB)];
        if (wb.valid) {
            if (wb.reg_write && wb.inst.rd != 0)
                regs_[wb.inst.rd] = wb.rd_val;

            stats_.instructions_retired++;
        }
    }

    /* 3. MEM: Memory Access and Load Result Generation */
    {
        const auto& mem_in = cur[static_cast<int>(Stage::MEM)];
        auto& wb_out = next[static_cast<int>(Stage::WB)];
        
        if (mem_in.valid) {
            wb_out = mem_in;
            addr_t addr = mem_in.alu_result;
            
            if (mem_in.mem_read) {
                MemoryResult r{};
                switch (mem_in.inst.op) {
                    case Op::LB:
                        r = memory_->read8(addr);
                        wb_out.rd_val = static_cast<u32>(sign_extend<8>(r.value));
                        break;
                    case Op::LH:
                        r = memory_->read16(addr);
                        wb_out.rd_val = static_cast<u32>(sign_extend<16>(r.value));
                        break;
                    case Op::LW:
                        r = memory_->read32(addr);
                        wb_out.rd_val = r.value;
                        break;
                    case Op::LBU:
                        r = memory_->read8(addr);
                        wb_out.rd_val = r.value & 0xFFu;
                        break;
                    case Op::LHU:
                        r = memory_->read16(addr);
                        wb_out.rd_val = r.value & 0xFFFFu;
                        break;

                    /* RVV Vector Load */
                    case Op::VLE32:
                        for (u32 i = 0; i < vstate_.vl; ++i) {
                            r = memory_->read32(addr + i * 4);
                            vstate_.regs.set_elem32(mem_in.inst.rd, i, r.value);
                        }
                        break;

                    /* RV32A Atomics (memory operation) */
                    case Op::LR_W: {
                        r = memory_->read32(addr);
                        wb_out.rd_val = r.value;
                        reservation_ = addr;
                        break;
                    }
                    case Op::SC_W: {
                        if (reservation_.has_value() && reservation_.value() == addr) {
                            memory_->write32(addr, mem_in.rs2_val);
                            wb_out.rd_val = 0;  /* success */
                        } else {
                            wb_out.rd_val = 1;  /* failure */
                        }
                        reservation_.reset();
                        break;
                    }
                    case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
                    case Op::AMOAND_W:  case Op::AMOOR_W:
                    case Op::AMOMIN_W:  case Op::AMOMAX_W:
                    case Op::AMOMINU_W: case Op::AMOMAXU_W: {
                        r = memory_->read32(addr);
                        u32 old_val = r.value;
                        u32 rs2v    = mem_in.rs2_val;
                        u32 new_val;
                        switch (mem_in.inst.op) {
                            case Op::AMOSWAP_W: new_val = rs2v; break;
                            case Op::AMOADD_W:  new_val = old_val + rs2v; break;
                            case Op::AMOXOR_W:  new_val = old_val ^ rs2v; break;
                            case Op::AMOAND_W:  new_val = old_val & rs2v; break;
                            case Op::AMOOR_W:   new_val = old_val | rs2v; break;
                            case Op::AMOMIN_W:  new_val = (static_cast<i32>(old_val) < static_cast<i32>(rs2v)) ? old_val 
                                                                                                               : rs2v; break;
                            case Op::AMOMAX_W:  new_val = (static_cast<i32>(old_val) > static_cast<i32>(rs2v)) ? old_val 
                                                                                                               : rs2v; break;
                            case Op::AMOMINU_W: new_val = (old_val < rs2v) ? old_val : rs2v; break;
                            case Op::AMOMAXU_W: new_val = (old_val > rs2v) ? old_val : rs2v; break;
                            default: new_val = old_val; break;
                        }
                        memory_->write32(addr, new_val);
                        wb_out.rd_val = old_val;
                        break;
                    }

                    default: break;
                }
            } else if (mem_in.mem_write) {
                u32 val = mem_in.rs2_val;
                switch (mem_in.inst.op) {
                    case Op::SB: memory_->write8(addr, static_cast<u8>(val)); break;
                    case Op::SH: memory_->write16(addr, static_cast<u16>(val)); break;
                    case Op::SW: memory_->write32(addr, val); break;

				    case Op::VSE32:
                        for (u32 i = 0; i < vstate_.vl; ++i)
                            memory_->write32(addr + i * 4, vstate_.regs.get_elem32(mem_in.inst.rd, i));
                        break;

                    default: break;
                }
                if (reservation_.has_value()) reservation_.reset();
            } else {
                wb_out.rd_val = mem_in.alu_result;
            }
        }
    }
   
    /* 4. Hazard Detection and Interlock Logic */
    const auto& id_stage = cur[static_cast<int>(Stage::ID)];
    const auto& ex_stage = cur[static_cast<int>(Stage::EX)];
    const auto& mem_stage = cur[static_cast<int>(Stage::MEM)];

    bool stall = false;
    if (id_stage.valid) {
        auto depends_on = [&](reg_idx_t producer_rd) -> bool {
            if (producer_rd == 0) return false;
            bool n1 = id_stage.inst.reads_rs1() && id_stage.inst.rs1 == producer_rd;
            bool n2 = id_stage.inst.reads_rs2() && id_stage.inst.rs2 == producer_rd;
            return n1 || n2;
        };

        /* Load-use: always stalls regardless of forwarding policy */
        if (ex_stage.valid && ex_stage.mem_read && depends_on(ex_stage.inst.rd)) {
            stall = true;
            stats_.stalls_load_use++;
        }

        if (!stall && config_.forwarding == ForwardingPolicy::NONE) {
            if (ex_stage.valid && ex_stage.reg_write && depends_on(ex_stage.inst.rd)) {
                stall = true;
                stats_.stalls_raw++;
            }
            if (!stall && mem_stage.valid && mem_stage.reg_write && depends_on(mem_stage.inst.rd)) {
                stall = true;
                stats_.stalls_raw++;
            }
        }

        if (!stall && config_.forwarding == ForwardingPolicy::PARTIAL) {
            /* Can forward from MEM, but not from EX */
            if (ex_stage.valid && ex_stage.reg_write && !ex_stage.mem_read && depends_on(ex_stage.inst.rd)) {
                stall = true;
                stats_.stalls_raw++;
            }
        }
    }

    /* 5. EX: ALU Execution and Forwarding Resolution */
    {
        const auto& ex_in = cur[static_cast<int>(Stage::EX)];
        auto& mem_out = next[static_cast<int>(Stage::MEM)];
        
        if (ex_in.valid) {
            mem_out = ex_in;
            
            /* Start with register file values */
            u32 rs1 = regs_[ex_in.inst.rs1];
            u32 rs2 = regs_[ex_in.inst.rs2];
            
            /* Forward from MEM stage */
            const auto& fwd_src = next[static_cast<int>(Stage::WB)];
            if (fwd_src.valid && fwd_src.reg_write && fwd_src.inst.rd != 0) {
                if (config_.forwarding != ForwardingPolicy::NONE) {
                    if (ex_in.inst.reads_rs1() && ex_in.inst.rs1 == fwd_src.inst.rd) {
                        rs1 = fwd_src.rd_val;
                        stats_.forwards_mem_ex++;
                    }
                    if (ex_in.inst.reads_rs2() && ex_in.inst.rs2 == fwd_src.inst.rd) {
                        rs2 = fwd_src.rd_val;
                        stats_.forwards_mem_ex++;
                    }
                }
            }

            mem_out.rs2_val = rs2;

            const auto& inst = ex_in.inst;
            
            switch (inst.op) {
                /* Branches */
                case Op::BEQ: case Op::BNE: case Op::BLT:
                case Op::BGE: case Op::BLTU: case Op::BGEU: {
                    bool taken = branch_check(inst.op, rs1, rs2);
                    addr_t target = static_cast<addr_t>(inst.pc + inst.imm);
                    bool predicted_taken = predict_branch(inst.pc, inst.imm);

                    stats_.branches++;
                    if (taken) stats_.branches_taken++;
                    update_predictor(inst.pc, taken);

                    if (taken != predicted_taken) {
                        stats_.branch_mispredicts++;
                        stats_.stalls_control += config_.branch_mispred_penalty;
                        flush_pipeline_ = true;
                        flush_target_ = taken ? target : (inst.pc + 4);
                    }

                    mem_out.branch_taken = taken;
                    mem_out.branch_target = target;
                    break;
                }

                /* Jumps */
                case Op::JAL:
                    mem_out.alu_result    = inst.pc + 4;
                    mem_out.rd_val        = inst.pc + 4;
                    mem_out.branch_taken  = true;
                    mem_out.branch_target = static_cast<addr_t>(inst.pc + inst.imm);

                    /*
                     * JAL target is known at decode in a real CPU
                     * This model resolves at EX - always flush IF/ID
                     */
                    flush_pipeline_ = true;
                    flush_target_   = mem_out.branch_target;
                    stats_.branches++;
                    stats_.branches_taken++;
                    break;
                    
                case Op::JALR:
                    mem_out.alu_result    = inst.pc + 4;
                    mem_out.rd_val        = inst.pc + 4;
                    mem_out.branch_taken  = true;
                    mem_out.branch_target = (rs1 + static_cast<u32>(inst.imm)) & ~1u;
                    
                    flush_pipeline_ = true;
                    flush_target_   = mem_out.branch_target;
                    stats_.branches++;
                    stats_.branches_taken++;
                    break;
                
                /* Upper immediate */
                case Op::LUI:   mem_out.alu_result = static_cast<u32>(inst.imm); break;
                case Op::AUIPC: mem_out.alu_result = inst.pc + static_cast<u32>(inst.imm); break;
                
                /* Load / store address */
                case Op::LB: case Op::LH: case Op::LW:
                case Op::LBU: case Op::LHU:
                case Op::SB: case Op::SH: case Op::SW:
                    mem_out.alu_result = rs1 + static_cast<u32>(inst.imm); break;
                
                /* ALU immediate */
                case Op::ADDI:  mem_out.alu_result = rs1 + static_cast<u32>(inst.imm); break;
                case Op::SLTI:  mem_out.alu_result = (static_cast<i32>(rs1) < inst.imm) ? 1u : 0u; break;
                case Op::SLTIU: mem_out.alu_result = (rs1 < static_cast<u32>(inst.imm)) ? 1u : 0u; break;
                case Op::XORI:  mem_out.alu_result = rs1 ^ static_cast<u32>(inst.imm); break;
                case Op::ORI:   mem_out.alu_result = rs1 | static_cast<u32>(inst.imm); break;
                case Op::ANDI:  mem_out.alu_result = rs1 & static_cast<u32>(inst.imm); break;
                case Op::SLLI:  mem_out.alu_result = rs1 << (inst.imm & 0x1F); break;
                case Op::SRLI:  mem_out.alu_result = rs1 >> (inst.imm & 0x1F); break;
                case Op::SRAI: {
                    u32 shamt = static_cast<u32>(inst.imm) & 0x1Fu;
                    u32 shifted = rs1 >> shamt;
                    if ((rs1 & 0x8000'0000u) && shamt > 0)
                        shifted |= ~u32{0} << (32 - shamt);
                    
                    mem_out.alu_result = shifted;
                    break;
                }

                /* ALU register */
                case Op::ADD:   mem_out.alu_result = rs1 + rs2; break;
                case Op::SUB:   mem_out.alu_result = rs1 - rs2; break;
                case Op::SLL:   mem_out.alu_result = rs1 << (rs2 & 0x1Fu); break;
                case Op::SLT:   mem_out.alu_result = (static_cast<i32>(rs1) < static_cast<i32>(rs2)) ? 1u : 0u; break;
                case Op::SLTU:  mem_out.alu_result = (rs1 < rs2) ? 1u : 0u; break;
                case Op::XOR:   mem_out.alu_result = rs1 ^ rs2; break;
                case Op::SRL:   mem_out.alu_result = rs1 >> (rs2 & 0x1Fu); break;
                case Op::SRA: {
                    u32 shamt = rs2 & 0x1Fu;
                    u32 shifted = rs1 >> shamt;
                    if ((rs1 & 0x8000'0000u) && shamt > 0)
                        shifted |= ~u32{0} << (32 - shamt);

                    mem_out.alu_result = shifted;
                    break;
                }
                case Op::OR:    mem_out.alu_result = rs1 | rs2; break;
                case Op::AND:   mem_out.alu_result = rs1 & rs2; break;
                
                /* RV32M Multiply / Divide */
                case Op::MUL: mem_out.alu_result = rs1 * rs2; break;
                case Op::MULH: {
                    i64 r = static_cast<i64>(static_cast<i32>(rs1))
                          * static_cast<i64>(static_cast<i32>(rs2));

                    mem_out.alu_result = static_cast<u32>(static_cast<u64>(r) >> 32);
                    break;
                }
                case Op::MULHSU: {
                    i64 r = static_cast<i64>(static_cast<i32>(rs1))
                          * static_cast<i64>(static_cast<u64>(rs2));
                    
                    mem_out.alu_result = static_cast<u32>(static_cast<u64>(r) >> 32);
                    break;
                }
                case Op::MULHU: {
                    u64 r = static_cast<u64>(rs1) * static_cast<u64>(rs2);

                    mem_out.alu_result = static_cast<u32>(r >> 32);
                }
                case Op::DIV: {
                    if (rs2 ==0) { mem_out.alu_result = ~u32{0}; break; }
                    auto sa = static_cast<i32>(rs1);
                    auto sb = static_cast<i32>(rs2);

                    if (sa == std::numeric_limits<i32>::min() && sb == -1)
                        mem_out.alu_result = static_cast<u32>(sa);
                    else
                        mem_out.alu_result = static_cast<u32>(sa / sb);
                    break;
                }
                case Op::DIVU: {
                    mem_out.alu_result = (rs2 == 0) ? ~u32{0} : rs1 / rs2;
                    break;
                }
                case Op::REM: {
                    if (rs2 == 0) { mem_out.alu_result = rs1; break; }
                    auto sa = static_cast<i32>(rs1);
                    auto sb = static_cast<i32>(rs2);

                    if (sa == std::numeric_limits<i32>::min() && sb == -1)
                        mem_out.alu_result = 0;
                    else
                        mem_out.alu_result = static_cast<u32>(sa % sb);
                    break;
                }
                case Op::REMU:
                    mem_out.alu_result = (rs2 == 0) ? rs1 : rs1 % rs2;
                    break;

                /* RV32A Atomics (address computation) */
                case Op::LR_W:
                case Op::SC_W:
                case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
                case Op::AMOAND_W:  case Op::AMOOR_W:
                case Op::AMOMIN_W:  case Op::AMOMAX_W:
                case Op::AMOMINU_W: case Op::AMOMAXU_W:
                    mem_out.alu_result = rs1;
                    mem_out.rs2_val    = rs2;
                    mem_out.mem_read   = true;
                    mem_out.reg_write  = true;
                    break;

                /* RVV Vector Operations */
                case Op::VSETVLI: {
                    u32 avl = (inst.rs1 == 0 && inst.rd == 0)
                            ? vstate_.vl
                            : (inst.rs1 == 0) ? ~u32{0} : rs1;
                    u32 new_vl = vstate_.vsetvli(avl, static_cast<u32>(inst.imm));
                    mem_out.alu_result = new_vl;
                    mem_out.rd_val     = new_vl;
                    mem_out.reg_write  = (inst.rd != 0);
                    break;
                }
                case Op::VSETIVLI: {
                    u32 avl = inst.rs1;
                    u32 new_vl = vstate_.vsetvli(avl, static_cast<u32>(inst.imm));
                    mem_out.alu_result = new_vl;
                    mem_out.rd_val     = new_vl;
                    mem_out.reg_write  = (inst.rd != 0);
                    break;
                }

                case Op::VLE32:
                    mem_out.alu_result = rs1;
                    mem_out.mem_read   = true;
                    break;
                case Op::VSE32:
                    mem_out.alu_result = rs1;
                    mem_out.mem_write  = true;
                    break;

                case Op::VADD_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) +
                                                vstate_.regs.get_elem32(inst.rs1, i));
                    break;
                case Op::VSUB_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) -
                                                vstate_.regs.get_elem32(inst.rs1, i));
                    break;
                case Op::VAND_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) &
                                                vstate_.regs.get_elem32(inst.rs1, i));
                    break;
                case Op::VOR_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) |
                                                vstate_.regs.get_elem32(inst.rs1, i));
                    break;
                case Op::VXOR_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) ^
                                                vstate_.regs.get_elem32(inst.rs1, i));
                    break;

                case Op::VADD_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) + rs1);
                    break;
                case Op::VSUB_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) - rs1);
                    break;
                case Op::VAND_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) & rs1);
                    break;
                case Op::VOR_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) | rs1);
                    break;
                case Op::VXOR_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) ^ rs1);
                    break;
                case Op::VSLL_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) << (rs1 & 0x1F));
                    break;
                case Op::VSRL_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_elem32(inst.rd, i,
                                                vstate_.regs.get_elem32(inst.rs2, i) >> (rs1 & 0x1F));
                    break;

                case Op::VMSEQ_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_mask_bit(i,
                                                  vstate_.regs.get_elem32(inst.rs2, i) == vstate_.regs.get_elem32(inst.rs1, i));
                    break;
                case Op::VMSEQ_VX:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_mask_bit(i,
                                                  vstate_.regs.get_elem32(inst.rs2, i) == rs1);
                    break;
                case Op::VMSLT_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_mask_bit(i,
                                                  static_cast<i32>(vstate_.regs.get_elem32(inst.rs2, i)) <
                                                  static_cast<i32>(vstate_.regs.get_elem32(inst.rs1, i)));
                    break;
                case Op::VMSLTU_VV:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_mask_bit(i,
                                                  vstate_.regs.get_elem32(inst.rs2, i) <
                                                  vstate_.regs.get_elem32(inst.rs1, i));
                    break;

                case Op::VMAND_MM:
                    for (u32 i = 0; i < vstate_.vl; ++i) {
                        u32 b2 = vstate_.regs.get_elem32(inst.rs2, i / 32);
                        u32 b1 = vstate_.regs.get_elem32(inst.rs1, i / 32);
                        vstate_.regs.set_mask_bit(i, ((b2 >> (i % 32)) & 1) &
                                                     ((b1 >> (i % 32)) & 1));
                    }
                    break;
                case Op::VMOR_MM:
                    for (u32 i = 0; i < vstate_.vl; ++i) {
                        u32 b2 = vstate_.regs.get_elem32(inst.rs2, i / 32);
                        u32 b1 = vstate_.regs.get_elem32(inst.rs1, i / 32);
                        vstate_.regs.set_mask_bit(i, ((b2 >> (i % 32)) & 1) |
                                                     ((b1 >> (i % 32)) & 1));
                    }
                    break;
                case Op::VMNOT_M:
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        vstate_.regs.set_mask_bit(i, !vstate_.regs.get_mask_bit(i));
                    break;

               case Op::VREDSUM_VS: {
                   u32 acc = vstate_.regs.get_elem32(inst.rs1, 0);
                   for (u32 i = 0; i < vstate_.vl; ++i)
                       acc += vstate_.regs.get_elem32(inst.rs2, i);
                   vstate_.regs.set_elem32(inst.rd, 0, acc);
                   break;
               }

               case Op::VMV_V_X:
                   for (u32 i = 0; i < vstate_.vl; ++i)
                       vstate_.regs.set_elem32(inst.rd, i, rs1);
                   break;
               case Op::VMV_X_S:
                  mem_out.alu_result = vstate_.regs.get_elem32(inst.rs2, 0);
                  mem_out.rd_val     = mem_out.alu_result;
                  mem_out.reg_write  = true;
                  break;

                /* System */
                case Op::FENCE: case Op::ECALL: case Op::EBREAK:
                    break;

                default: break;
            }
            
            /* Default rd_val = ALU result (loads overwrite in MEM stage) */
            if (!mem_out.mem_read)
                mem_out.rd_val = mem_out.alu_result;
        }
    }
    
    /* 6. ID: Decode and Pipeline Register Management */
    if (!stall) {
        if (id_stage.valid)
            next[static_cast<int>(Stage::EX)] = id_stage;
    } else {
        /* Bubble into EX; keep ID and IF frozen */
        next[static_cast<int>(Stage::EX)].clear();
        stats_.bubbles++;
    }
    
    /* 7. Speculative Fetch and PC Update */
    if (!stall) {
        const auto& if_in = cur[static_cast<int>(Stage::IF)];
        auto& id_out = next[static_cast<int>(Stage::ID)];
        
        if (if_in.valid) {
            id_out.valid = true;
            id_out.pc    = if_in.pc;
            id_out.inst  = Decoder::decode(if_in.inst.raw, if_in.pc);
            
            if (id_out.inst.op == Op::INVALID) halted_ = true;
            if (id_out.inst.op == Op::EBREAK)  halted_ = true;
            
            id_out.rs1_val   = regs_[id_out.inst.rs1];
            id_out.rs2_val   = regs_[id_out.inst.rs2];
            id_out.mem_read  = id_out.inst.is_load();
            id_out.mem_write = id_out.inst.is_store();
            id_out.reg_write = id_out.inst.writes_rd();
        }
        
        /* Fetch into IF */
        auto& if_out = next[static_cast<int>(Stage::IF)];
        if (!halted_) {
            auto r = memory_->read32(pc_);
            if (r.ok) {
                if_out.valid    = true;
                if_out.pc       = pc_;
                if_out.inst.raw = r.value;
                
                /*
                 * Simplified speculative PC update based on branch prediction
                 * A real CPU would do this in a pre-decode stage
                 */
                auto opcode = static_cast<u8>(r.value & 0x7F);
                if (opcode == 0b1100011) {  /* BRANCH opcode */
                    auto quick = Decoder::decode(r.value, pc_);
                    if (predict_branch(pc_, quick.imm))
                        pc_ = static_cast<addr_t>(pc_ + quick.imm);
                    else
                        pc_ += 4;
                } else {
                    pc_ += 4;
                }
            } else {
                halted_ = true;
            }
        }
    } else {
        /* Stall: keep IF and ID unchanged */
        next[static_cast<int>(Stage::IF)] = cur[static_cast<int>(Stage::IF)];
        next[static_cast<int>(Stage::ID)] = cur[static_cast<int>(Stage::ID)];
    }
    
    /* 8. Synchronous Flush (Misprediction Recovery) */
    if (flush_pipeline_) {
        next[static_cast<int>(Stage::IF)].clear();
        next[static_cast<int>(Stage::ID)].clear();
        next[static_cast<int>(Stage::EX)].clear();
        pc_ = flush_target_;
        flush_pipeline_ = false;
    	halted_ = false;
    }
    
    /* 9. Committing Next State */
    stages_ = next;
    
    if (trace_) dump_pipeline();
    
    return true;
}

// ── Run helpers ────────────────────────────────────────────────────────

cycle_t PipelinedCPU::run_cycles(cycle_t n)
{
    cycle_t count = 0;
    while (count < n) {
        if (!tick()) break;
        ++count;
    }
    return count;
}

cycle_t PipelinedCPU::run_instructions(u64 n)
{
    u64 start = stats_.instructions_retired;
    while (stats_.instructions_retired - start < n)
        if (!tick()) break;

    return stats_.cycles;
}

cycle_t PipelinedCPU::run_until_pc(addr_t target)
{
    return run_until([this, target]() {
        const auto& wb = stages_[static_cast<int>(Stage::WB)];
        return wb.valid && wb.pc == target;
    });
}

// ── Hazard detection ───────────────────────────────────────────────────

bool PipelinedCPU::detect_data_stall(const std::array<PipelineReg, kNumStages>& cur) const
{
    const auto& id = cur[static_cast<int>(Stage::ID)];
    const auto& ex = cur[static_cast<int>(Stage::EX)];

    if (!id.valid) return false;

    /*
     * Load-use hazard: EX has a load whose rd is needed by ID
     *
     * This always requires a 1-cycle stall - the load result
     * isn't available until the end of MEM
     */
    if (ex.valid && ex.mem_read && ex.inst.rd != 0) {
        bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
        bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

        if (needs_rs1 || needs_rs2) {
            stats_.stalls_load_use++;
            return true;
        }
    }

    /*
     * RAW hazard without forwarding: if forwarding is NONE, stall
     * whenever ID reads a register that EX or MEM will write.
     */
    if (config_.forwarding == ForwardingPolicy::NONE) {
        auto check = [&](reg_idx_t r) -> bool {
            if (r == 0) return false;
            if (ex.valid && ex.reg_write && ex.inst.rd == r) return true;
            const auto& mem = cur[static_cast<int>(Stage::MEM)];
            if (mem.valid && mem.reg_write && mem.inst.rd == r) return true;
            return false;
        };
        bool hazard = false;
        if (id.inst.reads_rs1() && check(id.inst.rs1)) hazard = true;
        if (id.inst.reads_rs2() && check(id.inst.rs2)) hazard = true;
        if (hazard) {
            stats_.stalls_raw++;
            return true;
        }
    }

    /* Partial forwarding: stall on EX->EX hazards (only MEM->EX is available) */
    if (config_.forwarding == ForwardingPolicy::PARTIAL) {
        if (ex.valid && ex.reg_write && !ex.mem_read && ex.inst.rd != 0) {
            bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
            bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

            if (needs_rs1 || needs_rs2) {
                stats_.stalls_raw++;
                return true;
            }
        }
    }

    return false;
}

// ── Forwarding ─────────────────────────────────────────────────────────

PipelinedCPU::ForwardResult PipelinedCPU::try_forward(reg_idx_t reg,
                                        const std::array<PipelineReg, kNumStages>& cur) const
{
    if (reg == 0) return {};

    const auto& mem = cur[static_cast<int>(Stage::MEM)];
    const auto& wb  = cur[static_cast<int>(Stage::WB)];

    /*
     * FULL forwarding: check EX stage (the instruction that just executed)
     * In the "cur" snapshot, MEM holds the result of the previous EX
     */
    if (config_.forwarding == ForwardingPolicy::FULL ||
        config_.forwarding == ForwardingPolicy::PARTIAL) {
        /*
         * MEM->EX: the instruction in MEM has its alu_result (or rd_val
         * for loads that went through MEM stage last cycle)
         */
        if (mem.valid && mem.reg_write && mem.inst.rd == reg) {
            stats_.forwards_mem_ex++;
            return {true, mem.rd_val};
        }
    }

    /*
     * WB->EX: the value is already committed to the register file because
     * WB is processed first. But if the register file hasn't been updated
     * yet in this cycle model, forward from WB.
     */
    if (wb.valid && wb.reg_write && wb.inst.rd == reg)
        return {true, wb.rd_val};

    return {};
}

u32 PipelinedCPU::resolve_rs1(const PipelineReg& id,
                              const std::array<PipelineReg, kNumStages>& cur) const
{
    if (!id.inst.reads_rs1()) return 0;
    auto fwd = try_forward(id.inst.rs1, cur);
    return fwd.available ? fwd.value : id.rs1_val;
}

u32 PipelinedCPU::resolve_rs2(const PipelineReg& id,
                              const std::array<PipelineReg, kNumStages>& cur) const
{
    if (!id.inst.reads_rs2()) return 0;
    auto fwd = try_forward(id.inst.rs2, cur);
    return fwd.available ? fwd.value : id.rs2_val;
}

// ── Branch prediction ──────────────────────────────────────────────────

u32 PipelinedCPU::bht_index(addr_t pc) const noexcept
{
    return (pc >> 2) & (static_cast<u32>(bht_.size()) - 1);
}

bool PipelinedCPU::predict_branch(addr_t pc, i32 offset) const
{
    switch(config_.predictor) {
        case BranchPredictor::NOT_TAKEN:
            return false;

        case BranchPredictor::ALWAYS_TAKEN:
            return true;

        case BranchPredictor::BACKWARD_TAKEN:
            return offset < 0;

        case BranchPredictor::BIMODAL_1BIT:
            return bht_[bht_index(pc)] != 0;

        case BranchPredictor::BIMODAL_2BIT:
            /* Predict taken if counter >= 2 (weakly or strongly taken) */
            return bht_[bht_index(pc)] >= 2;
    }
    return false;
}

void PipelinedCPU::update_predictor(addr_t pc, bool taken)
{
    u32 idx = bht_index(pc);

    switch (config_.predictor) {
        case BranchPredictor::NOT_TAKEN:
        case BranchPredictor::ALWAYS_TAKEN:
        case BranchPredictor::BACKWARD_TAKEN:
            /* Static predictors have no state to update */
            break;

        case BranchPredictor::BIMODAL_1BIT:
            bht_[idx] = taken ? 1 : 0;
            break;

        case BranchPredictor::BIMODAL_2BIT:
            if (taken) {
                if (bht_[idx] < 3) bht_[idx]++;
            } else {
                if (bht_[idx] > 0) bht_[idx]--;
	        }
            break;
    }
}

// ── ALU / branch helpers ───────────────────────────────────────────────

bool PipelinedCPU::branch_check(Op op, u32 rs1, u32 rs2)
{
    switch (op) {
        case Op::BEQ:  return rs1 == rs2;
        case Op::BNE:  return rs1 != rs2;
        case Op::BLT:  return static_cast<i32>(rs1) < static_cast<i32>(rs2);
        case Op::BGE:  return static_cast<i32>(rs1) >= static_cast<i32>(rs2);
        case Op::BLTU: return rs1 < rs2;
        case Op::BGEU: return rs1 >= rs2;
        default:       return false;
    }
}

// ── Debug output ───────────────────────────────────────────────────────

void PipelinedCPU::dump_pipeline() const
{
    static constexpr const char* names[] = {"IF ", "ID ", "EX ", "MEM ", "WB "};
    
    std::cout << std::format("Cycle {}:\n", stats_.cycles);
    for (int i = 0; i < kNumStages; ++i) {
        const auto& s = stages_[i];
        std::cout << std::format("  {}: ", names[i]);
        if (s.valid)
            std::cout << std::format("0x{:08x} {}", s.pc, s.inst.disassemble());
        else
            std::cout << "---";
        std::cout << "\n";
    }
    std::cout << std::format("  PC: 0x{:08x}\n\n", pc_);
}

void PipelinedCPU::dump_regs() const
{
    std::cout << std::format("PC: 0x{:08x}\n", pc_);
    for (int i = 0; i < 32; i += 4) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::format("{:>4s}: 0x{:08x}  ",
                                     reg_name(static_cast<reg_idx_t>(i + j)),
                                     regs_[i + j]);
        }
        std::cout << "\n";
    }
}

} // namespace riscv
