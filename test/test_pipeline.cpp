/**
 * @file test_pipeline.cpp
 * @brief Tests for the 5-stage pipelined CPU.
 *
 * @par Sections
 * @code
 *   1 (line  21) : IPC on straight-line code
 *   2 (line  47) : Data hazards and forwarding
 *   3 (line 128) : Branch prediction
 *   5 (line 236) : Control hazards
 *   7 (line 278) : Pipeline state inspection
 * @endcode
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Pipeline timing — straight-line IPC
// ═══════════════════════════════════════════════════════════════════════

TEST(pipe_straight_line_ipc)
{
    // 5 cycles to fill pipeline; 6 instructions (1 cycle each) -> 11 cycles
    auto cpu = make_pipeline();

    cpu.load_instruction(0,  ADDI(1, 0, 1));
    cpu.load_instruction(4,  ADDI(2, 0, 2));
    cpu.load_instruction(8,  ADDI(3, 0, 3));
    cpu.load_instruction(12, ADDI(4, 0, 4));
    cpu.load_instruction(16, ADDI(5, 0, 5));
    cpu.load_instruction(20, EBREAK);

    cpu.run_instructions(6);
    ASSERT_EQ(cpu.reg(1), 1u);
    ASSERT_EQ(cpu.reg(5), 5u);

    // EBREAK counts as a retired instruction in this model
    ASSERT(cpu.stats().instructions_retired == 6u);
    ASSERT(cpu.stats().ipc() >= 0.5 && cpu.stats().ipc() <= 0.6); // IPC = 6 / 11
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Data hazards and forwarding
// ═══════════════════════════════════════════════════════════════════════

TEST(pipe_raw_hazard_no_forwarding)
{
    // With forwarding disabled, this must stall
    PipelineConfig cfg;
    cfg.forwarding = ForwardingPolicy::NONE;
    auto cpu = make_pipeline(cfg);

    cpu.load_instruction(0, ADDI(1, 0, 10));
    cpu.load_instruction(4, ADDI(2, 1, 20));
    cpu.load_instruction(8, EBREAK);

    cpu.run_instructions(3);
    ASSERT_EQ(cpu.reg(2), 30u);
    ASSERT(cpu.stats().stalls_raw > 0u);
    return true;
}

TEST(pipe_raw_partial_forwarding)
{
    // With PARTIAL forwarding (MEM→EX only), a back-to-back RAW
    // (producer in EX, consumer in ID) stalls 1 cycle, then forwards.
    PipelineConfig cfg;
    cfg.forwarding = ForwardingPolicy::PARTIAL;
    auto cpu = make_pipeline(cfg);

    cpu.load_instruction(0, ADDI(1, 0, 10));
    cpu.load_instruction(4, ADDI(2, 1, 20));
    cpu.load_instruction(8, EBREAK);

    cpu.run_instructions(3);
    ASSERT_EQ(cpu.reg(2), 30u);
    ASSERT(cpu.stats().stalls_raw > 0u);
    return true;
}

TEST(pipe_load_use_hazard)
{
    // lw  x1, 0(x10)   ; load into x1
    // add x2, x1, x3   ; uses x1 immediately -> load-use stall
    // Even with full forwarding, load-use requires 1 stall cycle.
    auto cpu = make_pipeline();

    cpu.set_reg(10, 0x100);
    cpu.set_reg(3, 5);
    cpu.memory().write32(0x100, 42);

    cpu.load_instruction(0, LW(1, 0, 10));
    cpu.load_instruction(4, ADD(2, 1, 3));
    cpu.load_instruction(8, EBREAK);

    cpu.run_instructions(3);
    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT_EQ(cpu.reg(2), 47u);  // 42 + 5
    ASSERT(cpu.stats().stalls_load_use > 0u);
    return true;
}

TEST(pipe_no_load_use_with_gap)
{
    // lw   x1, 0(x10)
    // addi x3, x0, 99   ; independent — gives load time to complete
    // add  x2, x1, x3   ; x1 available via MEM->EX forward, no stall
    auto cpu = make_pipeline();
    cpu.set_reg(10, 0x100);
    cpu.memory().write32(0x100, 42);

    cpu.load_instruction(0,  LW(1, 0, 10));
    cpu.load_instruction(4,  ADDI(3, 0, 99));
    cpu.load_instruction(8,  ADD(2, 1, 3));
    cpu.load_instruction(12, EBREAK);

    cpu.run_instructions(4);
    ASSERT_EQ(cpu.reg(2), 42u + 99u);
    ASSERT_EQ(cpu.stats().stalls_load_use, 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Branch prediction
// ═══════════════════════════════════════════════════════════════════════

TEST(pipe_branch_not_taken_correct)
{
    // BEQ with equal regs -> taken.
    // Predict-not-taken -> mispredict.
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::NOT_TAKEN;
    auto cpu = make_pipeline(cfg);

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);

    cpu.load_instruction(0,  BEQ(1, 2, 8));
    cpu.load_instruction(4,  ADDI(0, 0, 0)); // should get flushed
    cpu.load_instruction(8,  ADDI(3, 0, 42)); // branch target
    cpu.load_instruction(12, EBREAK);

    cpu.run_instructions(4);
    ASSERT_EQ(cpu.reg(3), 42u);
    ASSERT(cpu.stats().branch_mispredicts > 0u);
    return true;
}

TEST(pipe_branch_always_taken_loop)
{
    // A simple loop: predict-always-taken should get every backward
    // branch right, and only mispredict the final fall-through.
    PipelineConfig cfg_taken;
    cfg_taken.predictor = BranchPredictor::ALWAYS_TAKEN;

    PipelineConfig cfg_not_taken;
    cfg_not_taken.predictor = BranchPredictor::NOT_TAKEN;

    auto run_loop = [](PipelineConfig cfg) {
        auto cpu = make_pipeline(cfg);

        cpu.load_instruction(0,  ADDI(1, 0, 0));
        cpu.load_instruction(4,  ADDI(2, 0, 10));
        cpu.load_instruction(8,  ADDI(1, 1, 1));
        cpu.load_instruction(12, BNE(1, 2, -8));
        cpu.load_instruction(20, EBREAK);

        cpu.run_instructions(1000);
        return cpu;
    };

    auto taken     = run_loop(cfg_taken);
    auto not_taken = run_loop(cfg_not_taken);

    // Both must produce the correct result
    ASSERT_EQ(taken.reg(1), 10u);
    ASSERT_EQ(not_taken.reg(1), 10u);

    // ALWAYS_TAKEN should have fewer mispredicts on this loop
    // (only mispredicts the final fall-through).
    ASSERT(taken.stats().branch_mispredicts < not_taken.stats().branch_mispredicts);
    return true;
}

TEST(pipe_backward_taken_predictor)
{
    // Backward-taken predictor should be ideal for loops
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::BACKWARD_TAKEN;
    auto cpu = make_pipeline(cfg);

    cpu.load_instruction(0,  ADDI(1, 0, 0));
    cpu.load_instruction(4,  ADDI(2, 0, 5));
    cpu.load_instruction(8,  ADDI(1, 1, 1));
    cpu.load_instruction(12, BNE(1, 2, -8));
    cpu.load_instruction(20, EBREAK);

    cpu.run_instructions(1000);
    ASSERT_EQ(cpu.reg(1), 5u);
    // Only 1 mispredict: the final iteration where bne falls through
    // (backward branch, predicted taken, but actually not taken)
    ASSERT_EQ(cpu.stats().branch_mispredicts, 1u);
    return true;
}

TEST(pipe_bimodal_2bit_learns)
{
    // The 2-bit bimodal predictor should learn a loop's pattern after
    // a few iterations
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::BIMODAL_2BIT;
    cfg.bht_size  = 64;
    auto cpu = make_pipeline(cfg);

    cpu.load_instruction(0,  ADDI(1, 0, 0));
    cpu.load_instruction(4,  ADDI(2, 0, 10));
    cpu.load_instruction(8,  ADDI(1, 1, 1));
    cpu.load_instruction(12, BNE(1, 2, -8));
    cpu.load_instruction(16, EBREAK);

    cpu.run_instructions(1000);

    // 10 branches total
    ASSERT_EQ(cpu.reg(1), 10u);
    ASSERT(cpu.stats().branch_mispredicts < cpu.stats().branches);
    // Total mispredicts should be 3 or fewer
    ASSERT(cpu.stats().branch_mispredicts <= 3u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Control hazards
// ═══════════════════════════════════════════════════════════════════════

TEST(pipe_jal_flushes)
{
    auto cpu = make_pipeline();

    cpu.load_instruction(0,  ADDI(1, 0, 1));
    cpu.load_instruction(4,  JAL(1, 12));
    cpu.load_instruction(8,  ADDI(1, 0, 2));
    cpu.load_instruction(12, ADDI(1, 0, 3));
    cpu.load_instruction(16, ADDI(1, 0, 4));
    cpu.load_instruction(20, EBREAK);

    cpu.run_instructions(10);
    // x1 should be 4 (the instruction at the jump target)
    ASSERT_EQ(cpu.reg(1), 4u);
    return true;
}

TEST(pipe_jalr_flushes)
{
    auto cpu = make_pipeline();
    cpu.set_reg(5, 0x20);

    cpu.load_instruction(0,  ADDI(1, 0, 1));
    cpu.load_instruction(4,  JALR(0, 0, 5));
    cpu.load_instruction(8,  ADDI(1, 0, 2));
    cpu.load_instruction(12, ADDI(0, 0, 0));
    cpu.load_instruction(16, ADDI(0, 0, 0));
    cpu.load_instruction(20, ADDI(0, 0, 0));
    cpu.load_instruction(24, ADDI(0, 0, 0));
    cpu.load_instruction(28, ADDI(0, 0, 0));
    cpu.load_instruction(32, ADDI(1, 0, 5));
    cpu.load_instruction(36, EBREAK);

    cpu.run_instructions(20);
    ASSERT_EQ(cpu.reg(1), 5u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Pipeline state inspection
// ═══════════════════════════════════════════════════════════════════════

TEST(pipe_halted_after_ebreak)
{
    auto cpu = make_pipeline();
    cpu.load_instruction(0, EBREAK);

    cpu.run_instructions(1);
    ASSERT(cpu.halted());
    return true;
}

TEST(pipe_stats_consistent)
{
    auto cpu = make_pipeline();

    cpu.load_instruction(0,  ADDI(1, 0, 0));
    cpu.load_instruction(4,  ADDI(2, 0, 10));
    cpu.load_instruction(8,  ADD(1, 1, 2));
    cpu.load_instruction(12, EBREAK);

    cpu.run_instructions(4);
    ASSERT(cpu.stats().cycles > 0u);
    ASSERT(cpu.stats().instructions_retired > 0u);
    ASSERT(cpu.stats().ipc() > 0.0);
    ASSERT(cpu.stats().ipc() <= 1.0);
    return true;
}
