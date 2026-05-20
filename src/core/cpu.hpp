/**
 * @file cpu.hpp
 * @brief Top-level single-cycle RISC-V hart controller.
 *
 * Orchestrates the fetch-decode-execute loop using a shared memory
 * backend, instruction decoder, and execution engine.
 *
 * @par Halt semantics
 * @c EBREAK, @c INVALID opcodes, and memory faults permanently halt the
 * CPU. Only @c reset() or @c restore_state() clears the halt flag.
 * @c ECALL is non-halting; the caller checks @c last_result().ecall.
 */

#pragma once

#include "decoder.hpp"
#include "executor.hpp"
#include "memory.hpp"

#include <functional>

namespace riscv {

/// Snapshot of architectural and performance state.
struct CpuState
{
    std::array<u32, 32> regs;
    addr_t              pc;
    cycle_t             cycles;
    u64                 instructions;
};

/// Single-cycle RISC-V CPU implementing fetch-decode-execute.
class CPU {
public:
    explicit CPU(std::shared_ptr<Memory> memory);

    /// @name Program loading
    /// @{
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /// @}

    /// @name State access
    /// @{
    void set_pc(addr_t pc)                   noexcept { executor_.set_pc(pc); }
    [[nodiscard]] addr_t pc()          const noexcept { return executor_.pc(); }

    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return executor_.reg(r); }
    void set_reg(reg_idx_t r, u32 value)     noexcept { executor_.set_reg(r, value); }
    /// @}

    /// @name Execution
    /// @{

    /// Executes a single instruction. Returns false if halted.
    bool step();

    /**
     * @brief Run until @p predicate returns true or @p max_instructions is reached.
     * @return Number of instructions retired.
     */
    template<typename Pred>
    u64 run_until(Pred&& predicate, u64 max_instructions = 1'000'000)
    {
        u64 count = 0;
        while (count < max_instructions && !predicate())
        {
            if (!step()) break;
            ++count;
        }
        return count;
    }

    /// Run for up to @p n instruction retirements.
    u64 run(u64 n)
    {
        u64 count = 0;
        while (count < n)
        {
            if (!step()) break;
            ++count;
        }
        return count;
    }

    /// Run until PC equals @p target.
    u64 run_until_pc(addr_t target)
    {
        return run_until([this, target]() { return pc() == target; });
    }

    /// Run until an @c ECALL is retired.
    u64 run_until_ecall()
    {
        return run_until([this]() { return last_result_.ecall; });
    }
    /// @}

    /// @name Status
    /// @{
    [[nodiscard]] bool            halted() const noexcept { return halted_; }
    [[nodiscard]] const CpuStats& stats()  const noexcept { return executor_.stats(); }
    [[nodiscard]] cycle_t         cycles() const noexcept { return executor_.stats().cycles; }
    [[nodiscard]] u64       instructions() const noexcept { return executor_.stats().instructions; }
    /// @}

    /// @name Lifecycle
    /// @{
    void reset();
    [[nodiscard]] CpuState save_state() const;
    void restore_state(const CpuState& state);
    /// @}

    /// @name Inspection
    /// @{
    void dump_regs() const { executor_.dump_regs(); }
    [[nodiscard]] const DecodedInst&   last_instruction() const noexcept { return last_inst_; }
    [[nodiscard]] const ExecuteResult& last_result()      const noexcept { return last_result_; }
    void set_trace(bool enable) noexcept { trace_ = enable; }
    [[nodiscard]] Memory&       memory()       noexcept { return *memory_; }
    [[nodiscard]] const Memory& memory() const noexcept { return *memory_; }
    /// @}

private:
    // Declaration order matters: memory_ must be initialized before executor_
    std::shared_ptr<Memory> memory_;
    Executor                executor_;

    DecodedInst   last_inst_{};
    ExecuteResult last_result_{};
    bool          trace_ = false;
    bool          halted_ = false;
};

} // namespace riscv
