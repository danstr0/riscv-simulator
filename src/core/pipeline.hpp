/**
 * @file pipeline.hpp
 * @brief Cycle-accurate 5-stage in-order RISC-V pipeline simulator.
 *
 * This implements a classic RISC-V integer pipeline (RV32I) designed
 * for architectural exploration. It features a synchronous "next-state"
 * evaluation model where pipeline registers are updated atomically at
 * the end of every clock cycle. 
 *
 * @section hazard_handling Hazard Resolution
 * - **Data Hazards (RAW)**: Resolved via configurable forwarding paths
 * or interlock stalls.
 * - **Load-Use Hazards**: Enforces a mandatory 1-cycle stall as the load
 * data is not available until the end of the MEM stage.
 * - **Control Hazards**: Handled via configurable branch predictors:
 * mispredictions trigger a pipeline flush and a parameterized penalty.
 *
 * @section policy_sec Parameterization Axes
 * - **Fowarding**: none, partial (MEM->EX), full (EX->EX + MEM->EX).
 * - **Branch Prediction**: not taken, always taken, backward-taken,
 * 1 bit bimodal, 2 bit bimodal.
 * - **Mispredict Penalty**: configurable cycle count.
 *
 * @see Patterson, D. A., & Hennessy, J. L. "Computer Organization and Design: The
 * Hardware/Software Interface (RISC-V Edition)", Sections 4.5-4.8 
 */

#pragma once

#include "decoder.hpp"
#include "memory.hpp"
#include "types.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace riscv {

// ── Pipeline stage identifiers ─────────────────────────────────────────

enum class Stage : u8 {
    IF  = 0,  ///< Instruction Fetch: Fetch from I-Cache/Memory.
    ID  = 1,  ///< Instruction Decode: Decode and Register Read.
    EX  = 2,  ///< Execute: ALU operations and Branch resolution.
    MEM = 3,  ///< Memory: Data Cache/Memory access.
    WB  = 4,  ///< Write Back: Update Register File.
};

inline constexpr int kNumStages = 5;

// ── Inter-stage register ───────────────────────────────────────────────

/**
 * @brief Represents the state held in the latches between stages (e.g., IF/ID, ID/EX).
 */
struct PipelineReg {
    bool        valid  = false;  ///< Bubble control: true if stage contains a valid instruction.
    DecodedInst inst{};          ///< Decoded instruction metadata.
    addr_t      pc     = 0;      ///< Program counter for the contained instruction.
    
    /** @name Datapath Values */
    /** @{ */
    u32 rs1_val    = 0;  ///< Source register 1 value (post-forwarding).
    u32 rs2_val    = 0;  ///< Source register 2 value (post-forwarding).
    u32 alu_result = 0;  ///< Output of the EX stage ALU.
    u32 mem_result = 0;  ///< Data loaded from memory in MEM stage.
    u32 rd_val     = 0;  ///< Final value to be written to rd in WB.
    /** @} */

    /** @name Control Signals (Decoded in ID) */
    /** @{ */
    bool   branch_taken  = false;
    addr_t branch_target = 0;
    bool   mem_read      = false;  ///< Load enable.
    bool   mem_write     = false;  ///< Store enable.
    bool   reg_write     = false;  ///< Register File write enable.
    /** @} */

    void clear() noexcept { *this = PipelineReg{}; }
};

// ── Configuration enums ────────────────────────────────────────────────

/**
 * @brief Defines the availability of internal forwarding paths.
 */
enum class ForwardingPolicy : u8 {
    NONE,     ///< No forwarding: all RAW hazards cause stalls until WB.
    PARTIAL,  ///< Forward from MEM->EX only.
    FULL,     ///< Forward from EX->EX and MEM->EX.
};

/**
 * @brief Dynamic and static branch prediction strategies.
 */
enum class BranchPredictor : u8 {
    NOT_TAKEN,       ///< Static: Predict all branches fall through.
    ALWAYS_TAKEN,    ///< Static: Predict all branches are taken.
    BACKWARD_TAKEN,  ///< Static: Taken if target < current PC.
    BIMODAL_1BIT,    ///< Dynamic: 1-bit saturating counter (Last-Time-Taken).
    BIMODAL_2BIT,    ///< Dynamic: 2-bit saturating counter (Strongly/Weakly Taken).
};

// ── Pipeline configuration ─────────────────────────────────────────────

struct PipelineConfig {
    ForwardingPolicy forwarding = ForwardingPolicy::FULL;
    BranchPredictor  predictor  = BranchPredictor::NOT_TAKEN;

    /// Cycles lost on a mispredicted branch.
    u32 branch_mispred_penalty  = 2;

    /// Number of entries in the Branch History Table (BHT).
    u32 bht_size = 256;
};

// ── Pipeline statistics ────────────────────────────────────────────────

/**
 * @brief Execution metrics for performance analysis.
 */
struct PipelineStats {
    cycle_t cycles               = 0;
    u64     instructions_retired = 0;
   
    /** @name Hazard Accounting */
    /** @{ */ 
    u64 stalls_load_use = 0;  ///< Cycles lost to load-use dependencies.
    u64 stalls_raw      = 0;  ///< Cycles lost to RAW hazards (if forwarding is disabled).
    u64 stalls_control  = 0;  ///< Cycles lost to branch mispredict flushes.
    u64 bubbles         = 0;  ///< Total non-functional cycles in the pipeline.
    /** @} */

    /** @name Branch and Forwarding Metrics */
    /** @{ */
    u64 branches           = 0;
    u64 branches_taken     = 0;
    u64 branch_mispredicts = 0;
    u64 forwards_ex_ex     = 0;
    u64 forwards_mem_ex    = 0;
    /** @} */

    /**
     * @brief Computes Instructions Per Cycle (IPC).
     * $$IPC = \frac{InstructionsRetired}{TotalCycles}$$
     */
    [[nodiscard]] double ipc() const noexcept
    {
        return cycles > 0
            ? static_cast<double>(instructions_retired) / static_cast<double>(cycles)
            : 0.0;
    }
    
    /// Percentage of correctly predicted branches.
    [[nodiscard]] double branch_accuracy() const noexcept
    {
        return branches > 0
            ? 1.0 - static_cast<double>(branch_mispredicts) / static_cast<double>(branches)
            : 1.0;
    }
    
    void reset() noexcept { *this = PipelineStats{}; }
};

// ── Pipelined CPU ──────────────────────────────────────────────────────

class PipelinedCPU {
public:
    explicit PipelinedCPU(std::shared_ptr<Memory> memory, 
                          PipelineConfig config = {});
    
    /** @name Program loading */
    /** @{ */
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /** @} */

    /** @name PC / register access */
    /** @{ */
    void set_pc(addr_t pc) noexcept { pc_ = pc; };
    [[nodiscard]] addr_t pc() const noexcept { return pc_; }
    
    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return regs_[r]; }
    void set_reg(reg_idx_t r, u32 value) noexcept { if (r != 0) regs_[r] = value; }
    /** @} */

    /** @name Execution */
    /** @{ */
    
    /**
     * @brief Advance the pipeline by one clock cycle.
     * Returns false if halted and the pipeline has drained.
     */ 
    bool tick();
    
    /// Run for N cycles
    cycle_t run_cycles(cycle_t n);
    
    /// Run until @p n instructions have retired.
    cycle_t run_instructions(u64 n);
    
    /// Run until PC reaches @p target (checked at WB).
    cycle_t run_until_pc(addr_t target);
    
    /// Run until @p pred returns true.
    template<typename Pred>
    cycle_t run_until(Pred&& pred, cycle_t max_cycles = 10'000'000)
    {
        cycle_t start = stats_.cycles;
        while (stats_.cycles - start < max_cycles && !pred())
            if (!tick()) break;

        return stats_.cycles - start;
    }
    
    /** @} */

    /** @name Inspection */
    /** @{ */
    [[nodiscard]] const PipelineStats&  stats()        const noexcept { return stats_; }
    [[nodiscard]] const PipelineConfig& config()       const noexcept { return config_; }
    [[nodiscard]] const PipelineReg&    stage(Stage s) const noexcept { return stages_[static_cast<int>(s)]; }
    [[nodiscard]] bool  pipeline_empty()               const noexcept;
    [[nodiscard]] bool  halted()                       const noexcept { return halted_; }
    /** @} */

    /** @name Lifecycle */
    /** @{ */
    void reset();
    void set_config(const PipelineConfig& cfg);
    void set_trace(bool enable) noexcept { trace_ = enable; }
    /** @} */

    /** @name Debug */
    /** @{ */
    void dump_pipeline() const;
    void dump_regs() const;
    
    [[nodiscard]] Memory& memory() noexcept { return *memory_; }
    /** @} */

private:
    std::shared_ptr<Memory> memory_;
    PipelineConfig          config_;
    
    std::array<u32, 32>     regs_{};
    addr_t                  pc_ = 0;
    
    std::array<PipelineReg, kNumStages> stages_{};
    
    bool   halted_         = false;
    bool   flush_pipeline_ = false;
    addr_t flush_target_   = 0;
    
    mutable PipelineStats stats_{};
    bool                  trace_ = false;

    /// LR/SC reservation
    std::optional<addr_t> reservation_;

    /** @name Branch prediction state */
    /** @{ */
    
    /**
     * @brief Branch history table (for bimodal predictors).
     * Each entry is a saturating counter: 1-bit (0/1) or 2-bit (0-3).
     */
    std::vector<u8> bht_;

    [[nodiscard]] bool predict_branch(addr_t pc, i32 offset) const;
    void               update_predictor(addr_t pc, bool taken);
    [[nodiscard]] u32  bht_index(addr_t pc) const noexcept;
    /** @} */

    /** @name Forwarding */
    /** @{ */

    struct ForwardResult {
        bool available = false;
        u32  value     = 0;
    };

    /**
     * @brief Try to forward the value of register @p reg from a later pipeline stage.
     * Checks EX->EX then MEM->EX depending on forwarding policy.
     */
    [[nodiscard]] ForwardResult try_forward(reg_idx_t reg,
                                            const std::array<PipelineReg, kNumStages>& cur) const;

    /**
     * @brief Resolve an operand: forward if possible, otherwise use the ID-stage register-file value.
     */
    [[nodiscard]] u32 resolve_rs1(const PipelineReg& id,
                                  const std::array<PipelineReg, kNumStages>& cur) const;
    [[nodiscard]] u32 resolve_rs2(const PipelineReg& id,
                                  const std::array<PipelineReg, kNumStages>& cur) const;

    /** @} */
    
    /** @name Hazard detection */
    /** @{ */
    [[nodiscard]] bool detect_data_stall(const std::array<PipelineReg, kNumStages>& cur) const;
    /** @} */

    /** @name ALU / branch helpers */
    /** @{ */
    static u32  alu_execute(Op op, u32 rs1, u32 rs2, i32 imm);
    static bool branch_check(Op op, u32 rs1, u32 rs2);
    /** @} */
};

} // namespace riscv
