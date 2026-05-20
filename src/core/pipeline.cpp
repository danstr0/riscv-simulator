/**
 * @file pipeline.cpp
 * @brief 5-stage pipeline implementation.
 *
 * Uses double-buffered pipeline registers: each @c tick() snapshots the
 * current state, computes the next state, then commits atomically.
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
    csrs_.reset();
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
    const std::array<u8, 4> bytes =
    {
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
    
    // ── 1. Hardware snapshot ─────────────────────────────────
    auto cur = stages_;
    std::array<PipelineReg, kNumStages> next{};
    
    // 2. WB: Retire and commit to architectural state ─────────
    {
        const auto& wb = cur[static_cast<int>(Stage::WB)];
        if (wb.valid)
        {
            if (wb.reg_write && wb.inst.rd != 0)
                regs_[wb.inst.rd] = wb.rd_val;

            stats_.instructions_retired++;
        }
    }

    // ── 2b. Check for pending interrupts ─────────────────────
    if (csrs_.interrupt_pending() && !flush_pipeline_)
    {
        u32 cause = csrs_.pending_cause();
        if (cause != 0)
        {
            addr_t trap_pc = pc_;

            for (int i = kNumStages - 1; i >= 0; --i)
            {
                if (cur[i].valid)
                {
                    trap_pc = cur[i].pc;
                    break;
                }
            }
            csrs_.enter_trap(trap_pc, cause);
            flush_pipeline_ = true;
            flush_target_ = csrs_.mtvec();
        }
    }

    // ── 3. MEM: Memory access and load result generation ─────
    {
        const auto& mem_in = cur[static_cast<int>(Stage::MEM)];
        auto& wb_out = next[static_cast<int>(Stage::WB)];
        
        if (mem_in.valid)
        {
            wb_out = mem_in;
            addr_t addr = mem_in.alu_result;

            if (mem_in.mem_read)
            {
                MemoryResult r{};
                switch (mem_in.inst.op)
                {
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

                    // ── RVV vector load ─────────────────
                    case Op::VLE32:
                        if (addr & 3)
                        {
                            std::cerr << std::format("[Pipeline] Misaligned vector load "
                                                     "at 0x{:08x} (PC 0x{:08x})\n",
                                                     addr, mem_in.pc);
                            wb_out.valid = false;
                            halted_ = true;
                            return true;
                        }
                        for (u32 i = 0; i < vstate_.vl; ++i)
                        {
                            r = memory_->read32(addr + i * 4);
                            vstate_.regs.set_elem32(mem_in.inst.rd, i, r.value);
                        }
                        break;

                    // ── RV32A memory operation ──────────
                    case Op::LR_W:
                    {
                        r = memory_->read32(addr);
                        wb_out.rd_val = r.value;
                        reservation_ = addr;
                        break;
                    }
                    case Op::SC_W:
                    {
                        if (reservation_.has_value() && reservation_.value() == addr)
                        {
                            memory_->write32(addr, mem_in.rs2_val);
                            wb_out.rd_val = 0; // success
                        }
                        else
                            wb_out.rd_val = 1; // failure
                        reservation_.reset();
                        break;
                    }
                    case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
                    case Op::AMOAND_W:  case Op::AMOOR_W:
                    case Op::AMOMIN_W:  case Op::AMOMAX_W:
                    case Op::AMOMINU_W: case Op::AMOMAXU_W:
                    {
                        r = memory_->read32(addr);
                        u32 old_val = r.value;
                        u32 rs2v    = mem_in.rs2_val;
                        u32 new_val;
                        switch (mem_in.inst.op)
                        {
                        case Op::AMOSWAP_W: new_val = rs2v; break;
                        case Op::AMOADD_W:  new_val = old_val + rs2v; break;
                        case Op::AMOXOR_W:  new_val = old_val ^ rs2v; break;
                        case Op::AMOAND_W:  new_val = old_val & rs2v; break;
                        case Op::AMOOR_W:   new_val = old_val | rs2v; break;
                        case Op::AMOMIN_W:  new_val = (static_cast<i32>(old_val) <
                                                       static_cast<i32>(rs2v)) ? old_val 
                                                                               : rs2v; break;
                        case Op::AMOMAX_W:  new_val = (static_cast<i32>(old_val) >
                                                       static_cast<i32>(rs2v)) ? old_val 
                                                                               : rs2v; break;
                        case Op::AMOMINU_W: new_val = (old_val < rs2v) ? old_val
                                                                       : rs2v; break;
                        case Op::AMOMAXU_W: new_val = (old_val > rs2v) ? old_val
                                                                       : rs2v; break;
                        default: new_val = old_val; break;
                        }
                        memory_->write32(addr, new_val);
                        wb_out.rd_val = old_val;
                        break;
                    }

                    default: break;
                }

                if (!r.ok)
                {
                    std::cerr << std::format("[Pipeline] Memory {} fault at address 0x{:08x} "
                                             "(PC 0x{:08x})\n",
                                             mem_in.mem_read ? "read" : "write", addr, mem_in.pc);
                    wb_out.valid = false;
                    halted_ = true;
                    return true;
                }
            }
            else if (mem_in.mem_write)
            {
                MemoryResult r{};
                u32 val = mem_in.rs2_val;
                switch (mem_in.inst.op)
                {
                case Op::SB: r = memory_->write8(addr, static_cast<u8>(val)); break;
                case Op::SH: r = memory_->write16(addr, static_cast<u16>(val)); break;
                case Op::SW: r = memory_->write32(addr, val); break;

                case Op::VSE32:
                    if (addr & 3)
                    {
                        std::cerr << std::format("[Pipeline] Misaligned vector load "
                                                 "at 0x{:08x} (PC 0x{:08x})\n",
                                                 addr, mem_in.pc);
                        wb_out.valid = false;
                        halted_ = true;
                        return true;
                    }
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        memory_->write32(addr + i * 4,
                                         vstate_.regs.get_elem32(mem_in.inst.rd, i));
                    break;

                default: break;
                }
                if (!r.ok)
                {
                    std::cerr << std::format("[Pipeline] Memory {} fault at address 0x{:08x} "
                                             "(PC 0x{:08x})\n",
                                             mem_in.mem_read ? "read" : "write", addr, mem_in.pc);
                    wb_out.valid = false;
                    halted_ = true;
                    return true;
                }
                if (reservation_.has_value())
                {
                    addr_t res = reservation_.value();
                    u32 store_size;
                    switch(mem_in.inst.op)
                    {
                        case Op::SB: store_size = 1; break;
                        case Op::SH: store_size = 2; break;
                        case Op::SW: store_size = 4; break;
                        default:     store_size = 0; break;
                    }
                    if (addr < res + 4 && addr + store_size > res)
                        reservation_.reset();
                }
            }
            else
                wb_out.rd_val = mem_in.alu_result;
        }
    }
   
    // ── 4. Hazard detection and interlock logic ──────────────
    const auto& id_stage = cur[static_cast<int>(Stage::ID)];
    bool stall = detect_data_stall(cur);

    // ── 5. EX: ALU execution and forwarding resolution ───────
    {
        const auto& ex_in = cur[static_cast<int>(Stage::EX)];
        auto& mem_out = next[static_cast<int>(Stage::MEM)];
        
        if (ex_in.valid)
        {
            mem_out = ex_in;
            
            u32 rs1 = resolve_rs1(ex_in, cur, next);
            u32 rs2 = resolve_rs2(ex_in, cur, next);

            mem_out.rs2_val = rs2;

            const auto& inst = ex_in.inst;
            u32 uimm = static_cast<u32>(inst.imm);

            auto exec_csr = [&](auto compute_new_val, bool do_write)
            {
                const u32 csr_addr = uimm & 0xFFF;
                const u32 old_val  = csrs_.read(csr_addr);

                if (do_write)
                    csrs_.write(csr_addr, compute_new_val(old_val));

                mem_out.alu_result = old_val;
                mem_out.rd_val     = old_val;
                mem_out.reg_write  = (inst.rd != 0);
            };

            auto exec_jump = [&](addr_t target)
            {
                mem_out.alu_result    = inst.pc + 4;
                mem_out.rd_val        = inst.pc + 4;
                mem_out.branch_taken  = true;
                mem_out.branch_target = target;

                /*
                 * Real hardware often resolves JAL/JALR at decode.
                 * This model resolves control flow in EX.
                 */
                flush_pipeline_ = true;
                flush_target_   = target;

                stats_.branches++;
                stats_.branches_taken++;
            };

            switch (inst.op)
            {
                // ── Branches ────────────────────────────
                case Op::BEQ: case Op::BNE: case Op::BLT:
                case Op::BGE: case Op::BLTU: case Op::BGEU:
                {
                    bool taken = branch_check(inst.op, rs1, rs2);
                    addr_t target = static_cast<addr_t>(inst.pc + inst.imm);
                    
                    if (taken && (target & 3))
                    {
                        std::cerr << std::format("[Pipeline] Misaligned branch target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        halted_ = true;
                        return true;
                    }
            
                    bool predicted_taken = predict_branch(inst.pc, inst.imm);

                    stats_.branches++;
                    if (taken) stats_.branches_taken++;
                    update_predictor(inst.pc, taken);

                    if (taken != predicted_taken)
                    {
                        stats_.branch_mispredicts++;
                        stats_.stalls_control += config_.branch_mispred_penalty;
                        flush_pipeline_ = true;
                        flush_target_ = taken ? target : (inst.pc + 4);
                    }

                    mem_out.branch_taken = taken;
                    mem_out.branch_target = target;
                    break;
                }

                // ── Jumps ────────────────────────────────────
                case Op::JAL:
                {
                    addr_t target = static_cast<addr_t>(inst.pc + inst.imm);
                    if (target & 3)
                    {
                        std::cerr << std::format("[Pipeline] Misaligned JAL target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        halted_ = true;
                        return true;
                    }
                    exec_jump(target);
                    break;
                }   
                case Op::JALR:
                {
                    addr_t target = (rs1 + uimm) & ~1u;
                    if (target & 3)
                    {
                        std::cerr << std::format("[Pipeline] Misaligned JALR target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        halted_ = true;
                        return true;
                    }
                    exec_jump(target);
                    break;
                }
                
                // ── Upper immediate ──────────────────────────
                case Op::LUI:   mem_out.alu_result = uimm; break;
                case Op::AUIPC: mem_out.alu_result = inst.pc + uimm; break;
                
                // ── Load / store address ─────────────────────
                case Op::LB: case Op::LH: case Op::LW:
                case Op::LBU: case Op::LHU:
                case Op::SB: case Op::SH: case Op::SW:
                    mem_out.alu_result = rs1 + uimm; break;
                
                // ── ALU ──────────────────────────────────────
                case Op::ADDI: case Op::SLTI: case Op::SLTIU:
                case Op::XORI: case Op::ORI:  case Op::ANDI:
                case Op::SLLI: case Op::SRLI: case Op::SRAI:
                case Op::ADD:  case Op::SUB:  case Op::SLL:
                case Op::SLT:  case Op::SLTU: case Op::XOR:
                case Op::SRL:  case Op::SRA:  case Op::OR:
                case Op::AND:
                case Op::MUL:  case Op::MULH: case Op::MULHSU:
                case Op::MULHU:case Op::DIV:  case Op::DIVU:
                case Op::REM:  case Op::REMU:
                    mem_out.alu_result = alu_execute(inst.op, rs1, rs2, inst.imm);
                    break;

                // ── RV32A instructions ───────────────────────
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

                // ── RVV instructions ─────────────────────────
                case Op::VSETVLI:
                {
                    u32 avl = (inst.rs1 == 0 && inst.rd == 0)
                            ? vstate_.vl
                            : (inst.rs1 == 0) ? ~u32{0} : rs1;
                    u32 new_vl = vstate_.vsetvli(avl, uimm);
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

                case Op::VADD_VV:  case Op::VSUB_VV:
                case Op::VAND_VV:  case Op::VOR_VV:  case Op::VXOR_VV:
                case Op::VADD_VX:  case Op::VSUB_VX:
                case Op::VAND_VX:  case Op::VOR_VX:  case Op::VXOR_VX:
                case Op::VSLL_VX:  case Op::VSRL_VX:
                case Op::VMSEQ_VV: case Op::VMSEQ_VX:
                case Op::VMSLT_VV: case Op::VMSLTU_VV:
                case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
                case Op::VMXOR_MM: case Op::VMOR_MM:   case Op::VMNOR_MM:
                case Op::VMORN_MM: case Op::VMXNOR_MM:
                case Op::VREDSUM_VS:
                case Op::VMV_V_X:
                    execute_vector(inst.op, vstate_,
                                   inst.rd, inst.rs2, inst.rs1,
                                   rs1, rs2);
                    break;
                
                case Op::VMV_X_S:
                    mem_out.alu_result = vstate_.regs.get_elem32(inst.rs2, 0);
                    mem_out.rd_val     = mem_out.alu_result;
                    mem_out.reg_write  = true;
                    break;

                // ── System ───────────────────────────────────
                case Op::FENCE: case Op::ECALL: case Op::EBREAK:
                    break;

                // ── CSR instructions ─────────────────────────
                case Op::CSRRW:
                    exec_csr([&](u32) { return rs1; }, true);
                    break;
                case Op::CSRRS:
                    exec_csr([&](u32 old) { return old | rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRC:
                    exec_csr([&](u32 old) { return old & ~rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRWI:
                    exec_csr([&](u32) { return inst.rs1; }, true);
                    break;
                case Op::CSRRSI:
                    exec_csr([&](u32 old) { return old | inst.rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRCI:
                    exec_csr([&](u32 old) { return old & ~inst.rs1; }, inst.rs1 != 0);
                    break;

                // ── MRET: return from trap ───────────────────
                case Op::MRET:
                {
                    addr_t return_pc = csrs_.mret();
                    flush_pipeline_  = true;
                    flush_target_    = return_pc;
                    break;
                }

                default: break;
            }
            
            // Default: rd_val = ALU result (loads overwrite in MEM stage)
            if (!mem_out.mem_read)
                mem_out.rd_val = mem_out.alu_result;
        }
    }

    // ── 6. ID: Decode and pipeline register management ───────
    if (!stall)
    {
        if (id_stage.valid)
            next[static_cast<int>(Stage::EX)] = id_stage;
    }
    else
    {
        // Bubble into EX; keep ID and IF frozen
        next[static_cast<int>(Stage::EX)].clear();
        stats_.bubbles++;
    }

    // ── 7. Speculative fetch and PC update ───────────────────
    if (!stall)
    {
        const auto& if_in = cur[static_cast<int>(Stage::IF)];
        auto& id_out = next[static_cast<int>(Stage::ID)];

        if (if_in.valid)
        {
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

        // Fetch into IF
        auto& if_out = next[static_cast<int>(Stage::IF)];
        if (!halted_)
        {
            auto r = memory_->read32(pc_);
            if (r.ok)
            {
                if_out.valid    = true;
                if_out.pc       = pc_;
                if_out.inst.raw = r.value;

                // Real hardware does this in a pre-decode stage
                auto opcode = static_cast<u8>(r.value & 0x7F);
                if (opcode == 0b1100011)  // BRANCH opcode
                {
                    auto quick = Decoder::decode(r.value, pc_);
                    if (predict_branch(pc_, quick.imm))
                        pc_ = static_cast<addr_t>(pc_ + quick.imm);
                    else
                        pc_ += 4;
                }
                else
                    pc_ += 4;
            }
            else
                halted_ = true;
        }
    }
    else
    {
        // Stall: keep IF and ID unchanged
        next[static_cast<int>(Stage::IF)] = cur[static_cast<int>(Stage::IF)];
        next[static_cast<int>(Stage::ID)] = cur[static_cast<int>(Stage::ID)];
    }

    // ── 8. Synchronous flush (misprediction recovery) ────────
    if (flush_pipeline_)
    {
        next[static_cast<int>(Stage::IF)].clear();
        next[static_cast<int>(Stage::ID)].clear();
        next[static_cast<int>(Stage::EX)].clear();
        pc_ = flush_target_;
        flush_pipeline_ = false;
    	halted_ = false;
    }
    
    // ── 9. Committing next state ─────────────────────────────
    stages_ = next;
    
    if (trace_) dump_pipeline();
    
    return true;
}

// ── Run helpers ────────────────────────────────────────────────────────

cycle_t PipelinedCPU::run_cycles(cycle_t n)
{
    cycle_t count = 0;
    while (count < n)
    {
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
    return run_until([this, target]()
    {
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
     * Load-use hazard: EX has a load whose rd is needed by ID.
     *
     * This always requires a 1-cycle stall - the load result
     * isn't available until the end of MEM.
     */
    if (ex.valid && ex.mem_read && ex.inst.rd != 0)
    {
        bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
        bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

        if (needs_rs1 || needs_rs2)
        {
            stats_.stalls_load_use++;
            return true;
        }
    }

    /*
     * RAW hazard without forwarding: if forwarding is NONE, stall
     * whenever ID reads a register that EX or MEM will write.
     */
    if (config_.forwarding == ForwardingPolicy::NONE)
    {
        auto check = [&](reg_idx_t r) -> bool
        {
            if (r == 0) return false;
            if (ex.valid && ex.reg_write && ex.inst.rd == r) return true;
            const auto& mem = cur[static_cast<int>(Stage::MEM)];
            if (mem.valid && mem.reg_write && mem.inst.rd == r) return true;
            return false;
        };
        bool hazard = false;
        if (id.inst.reads_rs1() && check(id.inst.rs1)) hazard = true;
        if (id.inst.reads_rs2() && check(id.inst.rs2)) hazard = true;
        if (hazard)
        {
            stats_.stalls_raw++;
            return true;
        }
    }

    // Partial forwarding: stall on EX->EX hazards (only MEM->EX is available)
    if (config_.forwarding == ForwardingPolicy::PARTIAL)
    {
        if (ex.valid && ex.reg_write && !ex.mem_read && ex.inst.rd != 0)
        {
            bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
            bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

            if (needs_rs1 || needs_rs2)
            {
                stats_.stalls_raw++;
                return true;
            }
        }
    }

    return false;
}

// ── Forwarding ─────────────────────────────────────────────────────────

PipelinedCPU::ForwardResult PipelinedCPU::try_forward(
        reg_idx_t reg,
        const std::array<PipelineReg, kNumStages>& cur,
        const std::array<PipelineReg, kNumStages>& next) const
{
    if (reg == 0) return {};

    // MEM->EX: forward from the value MEM just produced this cycle
    if (config_.forwarding != ForwardingPolicy::NONE)
    {
        const auto& wb_out = next[static_cast<int>(Stage::WB)];
        if (wb_out.valid && wb_out.reg_write && wb_out.inst.rd == reg)
        {
            stats_.forwards_mem_ex++;
            return {true, wb_out.rd_val};
        }
    }
    
    return {};
}

u32 PipelinedCPU::resolve_rs1(const PipelineReg& ex,
                              const std::array<PipelineReg, kNumStages>& cur,
                              const std::array<PipelineReg, kNumStages>& next) const
{
    u32 val = regs_[ex.inst.rs1];
    if (!ex.inst.reads_rs1()) return val;
    auto fwd = try_forward(ex.inst.rs1, cur, next);
    return fwd.available ? fwd.value : val;
}

u32 PipelinedCPU::resolve_rs2(const PipelineReg& ex,
                              const std::array<PipelineReg, kNumStages>& cur,
                              const std::array<PipelineReg, kNumStages>& next) const
{
    u32 val = regs_[ex.inst.rs2];
    if (!ex.inst.reads_rs2()) return val;
    auto fwd = try_forward(ex.inst.rs2, cur, next);
    return fwd.available ? fwd.value : val;
}

// ── Branch prediction ──────────────────────────────────────────────────

u32 PipelinedCPU::bht_index(addr_t pc) const noexcept
{
    return (pc >> 2) & (static_cast<u32>(bht_.size()) - 1);
}

bool PipelinedCPU::predict_branch(addr_t pc, i32 offset) const
{
    switch(config_.predictor)
    {
        case BranchPredictor::NOT_TAKEN:
            return false;

        case BranchPredictor::ALWAYS_TAKEN:
            return true;

        case BranchPredictor::BACKWARD_TAKEN:
            return offset < 0;

        case BranchPredictor::BIMODAL_1BIT:
            return bht_[bht_index(pc)] != 0;

        case BranchPredictor::BIMODAL_2BIT:
            // Predict taken if counter >= 2 (weakly or strongly taken)
            return bht_[bht_index(pc)] >= 2;
    }
    return false;
}

void PipelinedCPU::update_predictor(addr_t pc, bool taken)
{
    u32 idx = bht_index(pc);

    switch (config_.predictor)
    {
        case BranchPredictor::NOT_TAKEN:
        case BranchPredictor::ALWAYS_TAKEN:
        case BranchPredictor::BACKWARD_TAKEN:
            // Static predictors have no state to update
            break;

        case BranchPredictor::BIMODAL_1BIT:
            bht_[idx] = taken ? 1 : 0;
            break;

        case BranchPredictor::BIMODAL_2BIT:
            if (taken)
                if (bht_[idx] < 3) bht_[idx]++;
            else
                if (bht_[idx] > 0) bht_[idx]--;
            break;
    }
}

// ── ALU / branch helpers ───────────────────────────────────────────────

u32 PipelinedCPU::alu_execute(Op op, u32 rs1, u32 rs2, i32 imm)
{
    u32 uimm = static_cast<u32>(imm);
    switch(op)
    {
        case Op::ADD:   return rs1 + rs2;
        case Op::SUB:   return rs1 - rs2;
        case Op::SLL:   return rs1 << (rs2 & 0x1Fu);
        case Op::SLT:   return static_cast<i32>(rs1) < static_cast<i32>(rs2) ? 1u : 0u;
        case Op::SLTU:  return rs1 < rs2 ? 1u : 0u;
        case Op::XOR:   return rs1 ^ rs2;
        case Op::SRL:   return rs1 >> (rs2 & 0x1Fu);
        case Op::SRA:
        {
            u32 shamt = rs2 & 0x1Fu;
            u32 shifted = rs1 >> shamt;
            if ((rs1 & 0x8000'0000u) && shamt > 0)
                shifted |= ~u32{0} << (32 - shamt);
            return shifted;
        }
        case Op::OR:    return rs1 | rs2;
        case Op::AND:   return rs1 & rs2;

        case Op::ADDI:  return rs1 + uimm;
        case Op::SLTI:  return static_cast<i32>(rs1) < imm ? 1u : 0u;
        case Op::SLTIU: return rs1 < uimm ? 1u : 0u;
        case Op::XORI:  return rs1 ^ uimm;
        case Op::ORI:   return rs1 | uimm;
        case Op::ANDI:  return rs1 & uimm;
        case Op::SLLI:  return rs1 << (imm & 0x1F);
        case Op::SRLI:  return rs1 >> (imm & 0x1F);
        case Op::SRAI:
        {
            u32 shamt = uimm & 0x1Fu;
            u32 shifted = rs1 >> shamt;
            if ((rs1 & 0x8000'0000u) && shamt > 0)
                shifted |= ~u32{0} << (32 - shamt);
            return shifted;
        }

        case Op::MUL:   return rs1 * rs2;
        case Op::MULH:
        {
            i64 r = static_cast<i64>(static_cast<i32>(rs1))
                  * static_cast<i64>(static_cast<i32>(rs2));
            return static_cast<u32>(static_cast<u64>(r) >> 32);
        }
        case Op::MULHSU:
        {
            i64 r = static_cast<i64>(static_cast<i32>(rs1))
                  * static_cast<i64>(static_cast<u64>(rs2));
            return static_cast<u32>(static_cast<u64>(r) >> 32);
        }
        case Op::MULHU:
        {
            u64 r = static_cast<u64>(rs1) * static_cast<u64>(rs2);
            return static_cast<u32>(r >> 32);
        }
        case Op::DIV:
        {
            if (rs2 == 0) return ~u32{0};
            auto sa = static_cast<i32>(rs1);
            auto sb = static_cast<i32>(rs2);
            if (sa == std::numeric_limits<i32>::min() && sb == -1)
                return static_cast<u32>(sa);
            return static_cast<u32>(sa / sb);
        }
        case Op::DIVU:
            return rs2 == 0 ? ~u32{0} : rs1 / rs2;
        case Op::REM:
        {
            if (rs2 == 0) return rs1;
            auto sa = static_cast<i32>(rs1);
            auto sb = static_cast<i32>(rs2);
            if (sa == std::numeric_limits<i32>::min() && sb == -1)
                return 0;
            return static_cast<u32>(sa % sb);
        }
        case Op::REMU:
            return rs2 == 0 ? rs1 : rs1 % rs2;

        case Op::LUI:   return uimm;
        case Op::AUIPC: return 0;  // caller adds PC
    
        default: return 0;
    }
}

void PipelinedCPU::execute_vector(Op op, VectorState& vs,
                                  u32 vd, u32 vs2, u32 vs1,
                                  u32 rs1_val, u32 rs2_val)
{
    u32 vl      = vs.vl;
    auto& vregs = vs.regs;

    auto exec_vv = [&](auto op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            vregs.set_elem32(vd, i,
                             op(vregs.get_elem32(vs2, i),
                                vregs.get_elem32(vs1, i)));
        }
    };

    auto exec_vx = [&](auto op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            const u32 lhs = vregs.get_elem32(vs2, i);
            vregs.set_elem32(vd, i, op(lhs, rs1_val));
        }
    };

    auto exec_mask = [&](auto pred)
    {
        for (u32 i = 0; i < vl; ++i)
            vregs.set_mask_bit(vd, i, pred(i));
    };

    switch (op)
    {
        // ── VV arithmetic ────────────────────────────────
        case Op::VADD_VV: exec_vv([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VV: exec_vv([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VV: exec_vv([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VV:  exec_vv([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VV: exec_vv([](u32 a, u32 b) { return a ^ b; }); break;
    
        // ── VX arithmetic ────────────────────────────────
        case Op::VADD_VX: exec_vx([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VX: exec_vx([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VX: exec_vx([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VX:  exec_vx([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VX: exec_vx([](u32 a, u32 b) { return a ^ b; }); break;
        case Op::VSLL_VX: exec_vx([](u32 a, u32 b) { return a << (b & 0x1F); }); break;
        case Op::VSRL_VX: exec_vx([](u32 a, u32 b) { return a >> (b & 0x1F); }); break;
    
        // ── Comparisons ─────────────────────────────────
        case Op::VMSEQ_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) ==
                       vregs.get_elem32(vs1, i);
            });
            break;
        case Op::VMSEQ_VX:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) == rs1_val;
            });
            break;
        case Op::VMSLT_VV:
            exec_mask([&](u32 i)
            {
                return static_cast<i32>(vregs.get_elem32(vs2, i)) <
                       static_cast<i32>(vregs.get_elem32(vs1, i));
            });
            break;
        case Op::VMSLTU_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) <
                       vregs.get_elem32(vs1, i);
            });
            break;

        // ── Mask operations ─────────────────────────────
        case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
        case Op::VMXOR_MM: case Op::VMOR_MM:   case Op::VMNOR_MM:
        case Op::VMORN_MM: case Op::VMXNOR_MM:
        {
            for (u32 i = 0; i < vl; ++i)
            {
                u32 b2 = vregs.get_elem32(vs2, i / 32);
                u32 b1 = vregs.get_elem32(vs1, i / 32);
                u32 bit_pos = i % 32;
                bool bit2 = (b2 >> bit_pos) & 1;
                bool bit1 = (b1 >> bit_pos) & 1;
                bool bit_result;

                switch (op)
                {
                    case Op::VMAND_MM:  bit_result = bit2 & bit1;    break;
                    case Op::VMNAND_MM: bit_result = !(bit2 & bit1); break;
                    case Op::VMANDN_MM: bit_result = bit2 & (!bit1); break;
                    case Op::VMXOR_MM:  bit_result = bit2 ^ bit1;    break;
                    case Op::VMOR_MM:   bit_result = bit2 | bit1;    break;
                    case Op::VMNOR_MM:  bit_result = !(bit2 | bit1); break;
                    case Op::VMORN_MM:  bit_result = bit2 | (!bit1); break;
                    case Op::VMXNOR_MM: bit_result = !(bit2 ^ bit1); break;
                    default:            bit_result = false;          break;
                }

                u32 dst_elem = vregs.get_elem32(vd, i / 32);
                if (bit_result) dst_elem |=  (1u << bit_pos);
                else            dst_elem &= ~(1u << bit_pos);
                vregs.set_elem32(vd, i / 32, dst_elem);
            }
            break;
        }

        // ── Reduction ───────────────────────────────────
        case Op::VREDSUM_VS:
        {
            u32 acc = vregs.get_elem32(vs1, 0);
            for (u32 i = 0; i < vl; ++i)
                acc += vregs.get_elem32(vs2, i);
            vregs.set_elem32(vd, 0, acc);
            break;
        }

        // ── Splat ───────────────────────────────────────
        case Op::VMV_V_X:
            for (u32 i = 0; i < vl; ++i)
                vregs.set_elem32(vd, i, rs1_val);
            break;

        default: break;
    }
}

bool PipelinedCPU::branch_check(Op op, u32 rs1, u32 rs2)
{
    switch (op)
    {
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
    for (int i = 0; i < kNumStages; ++i)
    {
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
    for (int i = 0; i < 32; i += 4)
    {
        for (int j = 0; j < 4; ++j)
            std::cout << std::format("{:>4s}: 0x{:08x}  ",
                                     reg_name(static_cast<reg_idx_t>(i + j)),
                                     regs_[i + j]);
        std::cout << "\n";
    }
}

} // namespace riscv
