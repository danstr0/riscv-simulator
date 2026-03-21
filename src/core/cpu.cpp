/**
 * @file cpu.cpp
 * @brief Implementation of the top-level CPU fetch-decode-execute loop.
 */

#include "cpu.hpp"

#include <cassert>
#include <format>
#include <iomanip>
#include <iostream>

namespace riscv {

CPU::CPU(std::shared_ptr<Memory> memory)
    : memory_(std::move(memory))
    , executor_(*memory_)
{
    assert(memory_ != nullptr && "CPU initialized with null memory backend");
}

void CPU::load_program(addr_t addr, std::span<const u8> program)
{
    memory_->load(addr, program);
}

void CPU::load_instruction(addr_t addr, u32 instruction)
{
    const std::array<u8, 4> bytes = {
        static_cast<u8>(instruction),
        static_cast<u8>(instruction >> 8),
        static_cast<u8>(instruction >> 16),
        static_cast<u8>(instruction >> 24),
    };
    memory_->load(addr, bytes);
}

bool CPU::step()
{
    if (halted_) [[unlikely]] {
        return false;
    }
    
    /* ── 1. Fetch ─────────────────────────────────────────────────────── */
    addr_t current_pc = executor_.pc();
    auto fetch_result = memory_->read32(current_pc);

    if (!fetch_result.ok) {
        std::cerr << std::format("[CPU] Fetch error at PC 0x{:08x}\n", current_pc);
        halted_ = true;
        return false;
    }
    
    /* Accumulate fetch latency */
    executor_.stats().cycles += fetch_result.cycles;

    /* ── 2. Decode ────────────────────────────────────────────────────── */
    last_inst_ = Decoder::decode(fetch_result.value, current_pc);
    
    if (trace_) {
        std::cout << std::format("[Trace] 0x{:08x}: {:08x}  {}\n",
                                 current_pc, last_inst_.raw,
                                 last_inst_.disassemble());
    }
    
    if (last_inst_.op == Op::INVALID) {
        std::cerr << std::format("[CPU] Illegal instruction 0x{:08x} at PC 0x{:08x}\n",
                                 fetch_result.value, current_pc); 
        halted_ = true;
        return false;
    }
    
    /* ── 3. Execute ───────────────────────────────────────────────────── */
    last_result_ = executor_.execute(last_inst_);

    if (!last_result_.ok) {
        std::cerr << std::format("[CPU] Execution error at PC {:08x}\n", current_pc);
        halted_ = true;
        return false;
    }
    
    /* Commit architectural state: Update PC */
    executor_.set_pc(last_result_.next_pc);
    
    /* ── 4. Post-Execution State Update ───────────────────────────────── */
    if (last_result_.ebreak) {
        halted_ = true;
        return false;
    }
    
    /* Note: ECALL is handled by the caller checking last_result().ecall */

    return true;
}

void CPU::reset()
{
    executor_.reset();
    last_inst_   = DecodedInst{};
    last_result_ = ExecuteResult{};
    halted_      = false;
}

CpuState CPU::save_state() const
{
    return {
        .regs = executor_.regs(),
        .pc   = executor_.pc(),
        .cycles = executor_.stats().cycles,
        .instructions = executor_.stats().instructions
    };
}

void CPU::restore_state(const CpuState& state)
{
    for (reg_idx_t i = 1; i < 32; ++i) {
        executor_.set_reg(i, state.regs[i]);
    }
    executor_.set_pc(state.pc);

    /* Statistics are cumulative across resets to track total simulation work. */
    halted_ = false;
}

} // namespace riscv
