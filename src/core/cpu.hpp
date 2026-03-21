/**
 * @file cpu.hpp
 * @brief Top-level Hart controller for the RV32I simulator.
 *
 * The CPU class implements the main architectural state machine, orchestrating
 * the interaction between memory, instruction decoding, and execution logic.
 *
 * @section PIPELINE Instruction Lifecycle
 * Each call to @ref step() performs the following:
 * 1. **Fetch:** Retrieves a 32-bit word from @ref memory_ at the current PC.
 * 2. **Decode:** Dispatches the word to @ref Decoder to produce a @ref DecodedInst.
 * 3. **Execute:** Passes the decoded instruction to @ref Executor to mutate state.
 *
 * @section HALT_STATES Halt and Trap Semantics
 * - **Deterministic Halt:** An @c EBREAK instruction or an @ref INVALID opcode
 * permanently sets the @ref halted_ flag.
 * - **Exceptions:** Memory access faults during fetch or execute will halt the CPU.
 * - **Recovery:** Only a call to @ref reset() or @ref restore_state() clears the
 * halt flag.
 */

#pragma once

#include "decoder.hpp"
#include "executor.hpp"
#include "memory.hpp"

#include <functional>

namespace riscv {

/** @brief POD snapshot of the architectural and performance state. */
struct CpuState {
    std::array<u32, 32> regs;
    addr_t              pc;
    cycle_t             cycles;
    u64                 instructions;
};

/** @brief The primary interface for the RISC-V CPU Simulator. */
class CPU {
public:
    /**
     * @brief Constructs a CPU instance.
     * @param memory  A shared pointer to the system bus or memory region.
     */
    explicit CPU(std::shared_ptr<Memory> memory);

    /** @name Program Loading
     * Utility methods for initializing the memory state.
     */
    /** @{ */
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /** @} */

    /** @name State Accessors */
    /** @{ */
    void                 set_pc(addr_t pc)   noexcept { executor_.set_pc(pc); }
    [[nodiscard]] addr_t pc()          const noexcept { return executor_.pc(); }

    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return executor_.reg(r); }
    void set_reg(reg_idx_t r, u32 value)     noexcept { executor_.set_reg(r, value); }
    /** @} */

    /** @name Execution Control */
    /** @{ */

    /**
     * @brief Executes a single instruction.
     * @return True if the CPU is still operational; false if a halt/trap occurred.
     */
    bool step();

    /**
     * @brief Executes until a condition is met or a limit is reached.
     * @param predicate  A callable returning bool (true to stop).
     * @param max_instructions  Safety limit to prevent infinite loops.
     * @return Number of instructions successfully retired.
     */
    template<typename Pred>
    u64 run_until(Pred&& predicate, u64 max_instructions = 1'000'000)
    {
        u64 count = 0;
        while (count < max_instructions && !predicate()) {
            if (!step()) break;
            ++count;
        }
        return count;
    }

    /** @brief Runs for a fixed number of instruction retirements. */
    u64 run(u64 n)
    {
        u64 count = 0;
        while (count < n) {
            if (!step()) break;
            ++count;
        }
        return count;
    }

    /** @brief Runs until the Program Counter matches the target address. */
    u64 run_until_pc(addr_t target)
    {
        return run_until([this, target]() { return pc() == target; });
    }

    /** @brief Runs until an ECALL is retired. */
    u64 run_until_ecall()
    {
        return run_until([this]() { return last_result_.ecall; });
    }
    /** @} */

    /** @name Status and Telemetry */
    /** @{ */
    [[nodiscard]] bool halted() const noexcept { return halted_; }

    [[nodiscard]] const CpuStats& stats()  const noexcept { return executor_.stats(); }
    [[nodiscard]] cycle_t cycles()         const noexcept { return executor_.stats().cycles; }
    [[nodiscard]] u64     instructions()   const noexcept { return executor_.stats().instructions; }
    /** @} */

    /** @name Lifecycle Management */
    /** @{ */
    /** @brief Re-initializes architectural state (Registers and PC) to zero. */
    void reset();

    /** @brief Captures the current CPU state. */
    [[nodiscard]] CpuState save_state() const;

    /** @brief Overwrites the current CPU state with a snapshot. */
    void restore_state(const CpuState& state);
    /** @} */

    /** @name Inspection and Debugging */
    /** @{ */
    void dump_regs() const { executor_.dump_regs(); }

    /** @brief Access the metadata of the last retired instruction. */
    [[nodiscard]] const DecodedInst&   last_instruction() const noexcept { return last_inst_; }

    /** @brief Access the result of the last execution (latency, ok status, etc). */
    [[nodiscard]] const ExecuteResult& last_result()      const noexcept { return last_result_; }

    /** @brief Toggle execution tracing (e.g., printing instructions to stderr). */
    void set_trace(bool enable) noexcept { trace_ = enable; }

    [[nodiscard]] Memory&       memory()       noexcept { return *memory_; }
    [[nodiscard]] const Memory& memory() const noexcept { return *memory_; }
    /** @} */

private:
    /* N.B. Declaration order matters: memory_ must be initialized before executor_ */
    std::shared_ptr<Memory> memory_;
    Executor                executor_;
    
    DecodedInst   last_inst_{};
    ExecuteResult last_result_{};
    bool          trace_ = false;
    bool          halted_ = false;
};

} // namespace riscv
