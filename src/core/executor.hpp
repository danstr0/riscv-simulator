/**
 * @file executor.hpp
 * @brief Execution engine for decoded RV32I instructions.
 *
 * The Executor maintains the architectural state of the Hart, including the
 * 31 general-purpose registers (x1-x31), the hardwired zero register (x0),
 * and the Program Counter (PC).
 *
 * @section EXECUTION_CONTRACT Execution Contract
 * - **State Mutability:** The Executor modifies registers and memory.
 * - **PC Management:** The Executor *reads* the current PC for relative
 * calculations (branches/AUIPC) but does not commit the new PC state.
 * The caller is responsible for updating the PC using @ref ExecuteResult.next_pc.
 * - **Register x0:** Rigorously maintained as 0. Writes to index 0 are discarded.
 *
 * @note Reference: RISC-V Unprivileged ISA Specification v20260120, §2.1, §12.1.
 */

#pragma once

#include "decoder.hpp"
#include "memory.hpp"

#include <array>
#include <limits>
#include <optional>

namespace riscv {

/* ═══════════════════════════════════════════════════════════════════════
 * Execution Statistics
 * ═══════════════════════════════════════════════════════════════════════ */

/** @brief Performance and telemetry counters. */
struct CpuStats {
    cycle_t cycles         = 0;
    u64     instructions   = 0;
    u64     loads          = 0;
    u64     stores         = 0;
    u64     branches       = 0;
    u64     branches_taken = 0;
    u64     jumps          = 0;

    /** @brief Returns Instructions Per Cycle (IPC). */
    [[nodiscard]] double ipc() const noexcept
    {
        return cycles > 0 ? static_cast<double>(instructions) / static_cast<double>(cycles) : 0.0;
    }

    /** @brief Returns the ratio of branches that resulted in a PC change. */
    [[nodiscard]] double branch_taken_rate() const noexcept
    {
        return branches > 0 ? static_cast<double>(branches_taken) / static_cast<double>(branches) : 0.0;
    }

    void reset() noexcept { *this = CpuStats{}; }
};

/* ═══════════════════════════════════════════════════════════════════════
 * Single-Instruction Execution Result
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Metadata returned by the executor after processing an instruction.
 * * This structure informs the top-level CPU loop how to transition to the
 * next state (e.g., updating the PC or handling system traps).
 */
struct [[nodiscard]] ExecuteResult {
    bool   ok           = true;   ///< False if an exception (e.g., memory fault) occurred.
    u32    cycles       = 1;      ///< Total latency incurred by this instruction.
    bool   branch_taken = false;  ///< Telemetry: was a branch/jump target taken?
    addr_t next_pc      = 0;      ///< The calculated next instruction address.  

    bool   ecall        = false;  ///< Environment Call trap triggered.
    bool   ebreak       = false;  ///< Breakpoint trap triggered.

    /** @brief Optional record of what was written to the destination register. */
    std::optional<u32> rd_value;
};

/* ═══════════════════════════════════════════════════════════════════════
 * Executor
 * ═══════════════════════════════════════════════════════════════════════ */

class Executor {
public:
    explicit Executor(Memory& memory);

    /**
     * @brief Transforms the Hart state based on a decoded instruction.
     * @param inst  The instruction to execute.
     * @return Result containing the next PC and execution metadata.
     */
    [[nodiscard]] ExecuteResult execute(const DecodedInst& inst);

    /** @name Register Access */
    /** @{ */
    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return regs_[r]; }
    
    void set_reg(reg_idx_t r, u32 value) noexcept
    {
        if (r != 0) regs_[r] = value;
    }

    [[nodiscard]] const std::array<u32, 32>& regs() const noexcept { return regs_; }
    /** @} */

    /** @name Program Counter Access */
    /** @{ */
    [[nodiscard]] addr_t pc() const noexcept { return pc_; }
    void set_pc(addr_t pc) noexcept { pc_ = pc; }
    /** @} */

    /** @name Statistics and Telemetry */
    /** @{ */
    [[nodiscard]] const CpuStats& stats() const noexcept { return stats_; }
    [[nodiscard]] CpuStats&       stats()       noexcept { return stats_; }
    /** @} */

    /** @brief Resets the architectural state and statistics. */
    void reset();

    /** @brief Prints the current register state to stdout for debugging. */
    void dump_regs() const;

private:
    Memory&             memory_;
    std::array<u32, 32> regs_{};
    addr_t              pc_ = 0;
    CpuStats            stats_;

    /**
     * @brief LR/SC reservation address. Set by LR.W, cleared by SC.W
     * or any store to the reserved address.
     */
    std::optional<addr_t> reservation_;

    /** @name Internal ALU operations */
    /** @{ */

    static constexpr u32 alu_add(u32 a, u32 b) noexcept { return a + b; }
    static constexpr u32 alu_sub(u32 a, u32 b) noexcept { return a - b; }
    static constexpr u32 alu_and(u32 a, u32 b) noexcept { return a & b; }
    static constexpr u32 alu_or(u32 a, u32 b)  noexcept { return a | b; }
    static constexpr u32 alu_xor(u32 a, u32 b) noexcept { return a ^ b; }
    static constexpr u32 alu_sll(u32 a, u32 b) noexcept { return a << (b & 0x1Fu); }
    static constexpr u32 alu_srl(u32 a, u32 b) noexcept { return a >> (b & 0x1Fu); }
    
    /**
     * @brief Arithmetic Right Shift.
     * Manually replicates the sign bit to ensure portability across compilers
     * where signed right shifts might be implementation-defined
     */
    static constexpr u32 alu_sra(u32 a, u32 b) noexcept
    { 
        u32 shamt = b & 0x1Fu;
        u32 shifted = a >> shamt;
        if ((a & 0x8000'0000u) && shamt > 0) {
            shifted |= ~u32{0} << (32 - shamt);
        }
        return shifted; 
    }

    static constexpr u32 alu_slt(u32 a, u32 b) noexcept
    { 
        return static_cast<i32>(a) < static_cast<i32>(b) ? 1u : 0u;
    }
    
    static constexpr u32 alu_sltu(u32 a, u32 b) noexcept
    {
        return a < b ? 1u : 0u;
    }

    /** @} */

    /** @name RV32M multiply/divide (spec §12.1) */
    /** @{ */

    static constexpr u32 alu_mul(u32 a, u32 b) noexcept { return a * b; }
    
    static constexpr u32 alu_mulh(u32 a, u32 b) noexcept
    {
        i64 result = static_cast<i64>(static_cast<i32>(a))
                   * static_cast<i64>(static_cast<i32>(b));

        return static_cast<u32>(static_cast<u64>(result) >> 32);
    }

    static constexpr u32 alu_mulhsu(u32 a, u32 b) noexcept
    {
        i64 result = static_cast<i64>(static_cast<i32>(a))
                   * static_cast<i64>(static_cast<u64>(b));
        
        return static_cast<u32>(static_cast<u64>(result) >> 32);
    }

    static constexpr u32 alu_mulhu(u32 a, u32 b) noexcept
    {
        u64 result = static_cast<u64>(a) * static_cast<u64>(b);
        
        return static_cast<u32>(result >> 32);
    }

    static constexpr u32 alu_div(u32 a, u32 b) noexcept
    {
        if (b == 0) return ~u32{0};   /* -1 */
        auto sa = static_cast<i32>(a);
        auto sb = static_cast<i32>(b);

        if (sa == std::numeric_limits<i32>::min() && sb == -1)
            return static_cast<u32>(sa);

        return static_cast<u32>(sa / sb);
    }

    static constexpr u32 alu_divu(u32 a, u32 b) noexcept
    {
        return b == 0 ? ~u32{0} : a / b;
    }

    static constexpr u32 alu_rem(u32 a, u32 b) noexcept
    {
        if (b == 0) return a;
        auto sa = static_cast<i32>(a);
        auto sb = static_cast<i32>(b);

        if (sa == std::numeric_limits<i32>::min() && sb == -1)
            return 0;

        return static_cast<u32>(sa % sb);
    }

    static constexpr u32 alu_remu(u32 a, u32 b) noexcept
    {
        return b == 0 ? a : a % b;
    }

    /** @} */

    /** @name Branch conditions */
    /** @{ */
    static constexpr bool cond_eq(u32 a, u32 b)  noexcept { return a == b; }
    static constexpr bool cond_ne(u32 a, u32 b)  noexcept { return a != b; }
    static constexpr bool cond_lt(u32 a, u32 b)  noexcept { return static_cast<i32>(a) < static_cast<i32>(b); }
    static constexpr bool cond_ge(u32 a, u32 b)  noexcept { return static_cast<i32>(a) >= static_cast<i32>(b); }
    static constexpr bool cond_ltu(u32 a, u32 b) noexcept { return a < b; }
    static constexpr bool cond_geu(u32 a, u32 b) noexcept { return a >= b; }
    /** @} */

    /** @name Memory sub-executors */
    /** @{ */
    ExecuteResult execute_load(const DecodedInst& inst);
    ExecuteResult execute_store(const DecodedInst& inst);
    /** @} */
};

} // namespace riscv
