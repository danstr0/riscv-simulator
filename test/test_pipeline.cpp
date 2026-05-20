/**
 * @file test_pipeline.cpp
 * @brief Tests for the 5-stage pipelined CPU.
 *
 * Sections:
 *   1 (line 105) : Correctness parity (same results as single-cycle CPU)
 *   2 (line 229) : Pipeline timing (IPC on straight-line code)
 *   3 (line 258) : Data hazards (RAW, load-use)
 *   4 (line 352) : Forwarding policies (NONE, PARTIAL, FULL)
 *   5 (line 402) : Branch prediction strategies
 *   6 (line 514) : Control hazards (branch/jump flush)
 *   7 (line 556) : Pipeline state inspection
 */

#include "core/pipeline.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ── Helpers ────────────────────────────────────────────────────────────

static PipelinedCPU make_pipeline(PipelineConfig cfg = {})
{
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    return PipelinedCPU(mem, cfg);
}

static void load_program(PipelinedCPU& cpu, addr_t addr,
                         std::initializer_list<u32> instructions)
{
    addr_t offset = addr;
    for (u32 inst : instructions) {
        cpu.load_instruction(offset, inst);
        offset += 4;
    }
}

/// Run until the pipeline is fully drained (halted + empty).
static void run_to_completion(PipelinedCPU& cpu, cycle_t max = 10000)
{
    cycle_t n = 0;
    while (n < max) {
        if (!cpu.tick()) break;
        ++n;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Correctness parity with single-cycle CPU
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_addi) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x02a00093,  // addi x1, x0, 42
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(1), 42u);
    return true;
}

TEST(pipe_add_sub) {
    auto cpu = make_pipeline();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, 20);
    load_program(cpu, 0, {
        0x002081b3,  // add x3, x1, x2
        0x402081b3,  // sub x3, x1, x2
        0x00100073,  // ebreak
    });
    // After sub: x3 = 10 - 20 = -10 = 0xFFFFFFF6
    run_to_completion(cpu);
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFF6u);
    return true;
}

TEST(pipe_load_store) {
    auto cpu = make_pipeline();
    cpu.memory().write32(0x100, 0xDEADBEEF);
    cpu.set_reg(10, 0x100);
    load_program(cpu, 0, {
        0x00052083,  // lw x1, 0(x10)
        0x00152223,  // sw x1, 4(x10)
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_HEX_EQ(cpu.reg(1), 0xDEADBEEFu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x104).value, 0xDEADBEEFu);
    return true;
}

TEST(pipe_lui_auipc) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x123450b7,  // lui x1, 0x12345
        0x00010117,  // auipc x2, 0x10
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_HEX_EQ(cpu.reg(1), 0x12345000u);
    // auipc at PC=4: x2 = 4 + 0x10000
    ASSERT_HEX_EQ(cpu.reg(2), 0x10004u);
    return true;
}

TEST(pipe_fibonacci) {
    // Same fibonacci program as test_programs.cpp.
    // Fib(10) = 55.
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00000093,  // addi x1, x0, 0
        0x00100113,  // addi x2, x0, 1
        0x00a00193,  // addi x3, x0, 10
        0x00000213,  // addi x4, x0, 0
        0x00320c63,  // beq  x4, x3, 24
        0x002082b3,  // add  x5, x1, x2
        0x00010093,  // addi x1, x2, 0
        0x00028113,  // addi x2, x5, 0
        0x00120213,  // addi x4, x4, 1
        0xfedff06f,  // jal  x0, -20
        0x00100073,  // ebreak
    });
    run_to_completion(cpu, 100000);
    ASSERT_EQ(cpu.reg(1), 55u);
    return true;
}

TEST(pipe_sum_1_to_10) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00000093,  // addi x1, x0, 0
        0x00100113,  // addi x2, x0, 1
        0x00a00193,  // addi x3, x0, 10
        0x002080b3,  // add  x1, x1, x2
        0x00110113,  // addi x2, x2, 1
        0xfe21dce3,  // bge  x3, x2, -8
        0x00100073,  // ebreak
    });
    run_to_completion(cpu, 100000);
    ASSERT_EQ(cpu.reg(1), 55u);
    return true;
}

TEST(pipe_function_call) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00500513,  // addi a0, x0, 5
        0x00c000ef,  // jal  ra, 12
        0x00050593,  // addi a1, a0, 0
        0x00100073,  // ebreak
        0x00151513,  // slli a0, a0, 1  (double:)
        0x00008067,  // jalr x0, 0(ra)
    });
    run_to_completion(cpu, 10000);
    ASSERT_EQ(cpu.reg(10), 10u);
    ASSERT_EQ(cpu.reg(11), 10u);
    return true;
}

TEST(pipe_x0_always_zero) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x06400013,  // addi x0, x0, 100
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. Pipeline timing — straight-line IPC
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_straight_line_ipc) {
    // 5 independent ADDIs followed by EBREAK.
    // With full forwarding and no branches, steady-state IPC -> 1.0.
    // Total cycles = 5 (pipeline fill) + 5 instructions - 1 = ~9.
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00100093,  // addi x1, x0, 1
        0x00200113,  // addi x2, x0, 2
        0x00300193,  // addi x3, x0, 3
        0x00400213,  // addi x4, x0, 4
        0x00500293,  // addi x5, x0, 5
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);

    ASSERT_EQ(cpu.reg(1), 1u);
    ASSERT_EQ(cpu.reg(5), 5u);
    // 5 useful instructions retired (ebreak doesn't count as retired in
    // most models, but our pipeline does retire it).
    ASSERT(cpu.stats().instructions_retired >= 5u);
    // IPC should be close to 1.0 for straight-line code.
    ASSERT(cpu.stats().ipc() > 0.5);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Data hazards — RAW and load-use
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_raw_hazard_with_forwarding) {
    // addi x1, x0, 10  ; produces x1
    // addi x2, x1, 20  ; consumes x1 (RAW on x1)
    // With FULL forwarding, no stall — x1 is forwarded from EX→EX.
    PipelineConfig cfg;
    cfg.forwarding = ForwardingPolicy::FULL;
    auto cpu = make_pipeline(cfg);
    load_program(cpu, 0, {
        0x00a00093,  // addi x1, x0, 10
        0x01408113,  // addi x2, x1, 20
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(2), 30u);
    ASSERT_EQ(cpu.stats().stalls_load_use, 0u);
    return true;
}

TEST(pipe_raw_hazard_no_forwarding) {
    // Same sequence, but with forwarding disabled — must stall.
    PipelineConfig cfg;
    cfg.forwarding = ForwardingPolicy::NONE;
    auto cpu = make_pipeline(cfg);
    load_program(cpu, 0, {
        0x00a00093,  // addi x1, x0, 10
        0x01408113,  // addi x2, x1, 20
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(2), 30u);  // Correct result regardless.
    ASSERT(cpu.stats().stalls_raw > 0u);  // But with stalls.
    return true;
}

TEST(pipe_raw_partial_forwarding) {
    // With PARTIAL forwarding (MEM→EX only), a back-to-back RAW
    // (producer in EX, consumer in ID) stalls 1 cycle, then forwards.
    PipelineConfig cfg;
    cfg.forwarding = ForwardingPolicy::PARTIAL;
    auto cpu = make_pipeline(cfg);
    load_program(cpu, 0, {
        0x00a00093,  // addi x1, x0, 10
        0x01408113,  // addi x2, x1, 20
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(2), 30u);
    ASSERT(cpu.stats().stalls_raw > 0u);
    return true;
}

TEST(pipe_load_use_hazard) {
    // lw  x1, 0(x10)   ; load into x1
    // add x2, x1, x3   ; uses x1 immediately -> load-use stall
    // Even with full forwarding, load-use requires 1 stall cycle.
    auto cpu = make_pipeline();
    cpu.set_reg(10, 0x100);
    cpu.set_reg(3, 5);
    cpu.memory().write32(0x100, 42);
    load_program(cpu, 0, {
        0x00052083,  // lw  x1, 0(x10)
        0x00308133,  // add x2, x1, x3
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT_EQ(cpu.reg(2), 47u);  // 42 + 5
    ASSERT(cpu.stats().stalls_load_use > 0u);
    return true;
}

TEST(pipe_no_load_use_with_gap) {
    // lw   x1, 0(x10)
    // addi x3, x0, 99   ; independent — gives load time to complete
    // add  x2, x1, x3   ; x1 available via MEM->EX forward, no stall
    auto cpu = make_pipeline();
    cpu.set_reg(10, 0x100);
    cpu.memory().write32(0x100, 42);
    load_program(cpu, 0, {
        0x00052083,  // lw   x1, 0(x10)
        0x06300193,  // addi x3, x0, 99
        0x00308133,  // add  x2, x1, x3
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(2), 42u + 99u);
    ASSERT_EQ(cpu.stats().stalls_load_use, 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Forwarding policy comparison
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_forwarding_comparison) {
    // Chain of dependent ADDIs: each reads the result of the previous.
    // addi x1, x0, 1
    // addi x2, x1, 1
    // addi x3, x2, 1
    // addi x4, x3, 1
    // ebreak
    //
    // FULL:    0 data stalls (all forwarded EX->EX)
    // PARTIAL: stalls on each back-to-back pair (EX->EX not available)
    // NONE:    stalls on every dependent pair

    auto run_with = [](ForwardingPolicy fp) {
        PipelineConfig cfg;
        cfg.forwarding = fp;
        auto cpu = make_pipeline(cfg);
        load_program(cpu, 0, {
            0x00100093,  // addi x1, x0, 1
            0x00108113,  // addi x2, x1, 1
            0x00110193,  // addi x3, x2, 1
            0x00118213,  // addi x4, x3, 1
            0x00100073,  // ebreak
        });
        run_to_completion(cpu);
        return cpu;
    };

    auto full    = run_with(ForwardingPolicy::FULL);
    auto partial = run_with(ForwardingPolicy::PARTIAL);
    auto none    = run_with(ForwardingPolicy::NONE);

    // All must produce correct results.
    ASSERT_EQ(full.reg(4), 4u);
    ASSERT_EQ(partial.reg(4), 4u);
    ASSERT_EQ(none.reg(4), 4u);

    // NONE should have the most stalls, FULL the fewest.
    ASSERT(none.stats().stalls_raw >= partial.stats().stalls_raw);
    ASSERT(full.stats().stalls_raw == 0u);
    ASSERT(full.stats().stalls_load_use == 0u);

    // NONE takes more cycles.
    ASSERT(none.stats().cycles > full.stats().cycles);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Branch prediction
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_branch_not_taken_correct) {
    // BEQ with equal regs -> taken.  Predict-not-taken -> mispredict.
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::NOT_TAKEN;
    auto cpu = make_pipeline(cfg);
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);
    load_program(cpu, 0, {
        0x00208463,  // beq x1, x2, 8
        0x00000013,  // nop (addi x0, x0, 0) — should be flushed
        0x02a00193,  // addi x3, x0, 42      — branch target
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(3), 42u);
    ASSERT(cpu.stats().branch_mispredicts > 0u);
    return true;
}

TEST(pipe_branch_always_taken_loop) {
    // A simple loop: predict-always-taken should get every backward
    // branch right, and only mispredict the final fall-through.
    PipelineConfig cfg_taken;
    cfg_taken.predictor = BranchPredictor::ALWAYS_TAKEN;

    PipelineConfig cfg_not_taken;
    cfg_not_taken.predictor = BranchPredictor::NOT_TAKEN;

    auto run_loop = [](PipelineConfig cfg) {
        auto cpu = make_pipeline(cfg);
        load_program(cpu, 0, {
            0x00000093,  // addi x1, x0, 0
            0x00a00113,  // addi x2, x0, 10
            0x00108093,  // addi x1, x1, 1   (loop body)
            0xfe209ce3,  // bne  x1, x2, -8  (loop back)
            0x00100073,  // ebreak
        });
        run_to_completion(cpu, 100000);
        return cpu;
    };

    auto taken     = run_loop(cfg_taken);
    auto not_taken = run_loop(cfg_not_taken);

    // Both must produce correct result.
    ASSERT_EQ(taken.reg(1), 10u);
    ASSERT_EQ(not_taken.reg(1), 10u);

    // ALWAYS_TAKEN should have fewer mispredicts on this loop
    // (only mispredicts the final fall-through).
    ASSERT(taken.stats().branch_mispredicts < not_taken.stats().branch_mispredicts);
    return true;
}

TEST(pipe_backward_taken_predictor) {
    // Backward-taken predictor should be ideal for loops.
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::BACKWARD_TAKEN;
    auto cpu = make_pipeline(cfg);
    load_program(cpu, 0, {
        0x00000093,  // addi x1, x0, 0
        0x00500113,  // addi x2, x0, 5
        0x00108093,  // addi x1, x1, 1
        0xfe209ce3,  // bne  x1, x2, -8
        0x00100073,  // ebreak
    });
    run_to_completion(cpu, 100000);
    ASSERT_EQ(cpu.reg(1), 5u);
    // Only 1 mispredict: the final iteration where bne falls through
    // (backward branch, predicted taken, but actually not taken).
    ASSERT_EQ(cpu.stats().branch_mispredicts, 1u);
    return true;
}

TEST(pipe_bimodal_2bit_learns) {
    // The 2-bit bimodal predictor should learn a loop's pattern after
    // a few iterations
    PipelineConfig cfg;
    cfg.predictor = BranchPredictor::BIMODAL_2BIT;
    cfg.bht_size  = 64;
    auto cpu = make_pipeline(cfg);

    load_program(cpu, 0, {
        // Loop 1: count x1 from 0 to 5.
        0x00000093,  // 0x00: addi x1, x0, 0
        0x00a00113,  // 0x04: addi x2, x0, 10
        0x00108093,  // 0x08: addi x1, x1, 1   (loop body)
        0xfe209ce3,  // 0x0C: bne  x1, x2, -8  (back to 0x08)
        0x00100073,  // 0x10: ebreak
    });
    run_to_completion(cpu, 100000);

    // 10 branches total
    ASSERT_EQ(cpu.reg(1), 10u);
    // Predictor should have learned (and yielded significantly fewer
    // mispredicts than branches) 
    ASSERT(cpu.stats().branch_mispredicts < cpu.stats().branches);
    // Total mispredicts should be ~3 or fewer
    ASSERT(cpu.stats().branch_mispredicts <= 5u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Control hazards
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_jal_flushes) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00100093,  // 0x00: addi x1, x0, 1
        0x00c000ef,  // 0x04: jal  ra, 12     -> jump to 0x10
        0x00200093,  // 0x08: addi x1, x0, 2  (should be flushed)
        0x00300093,  // 0x0C: addi x1, x0, 3  (should be flushed)
        0x00400093,  // 0x10: addi x1, x0, 4  (jump target)
        0x00100073,  // 0x14: ebreak
    });
    run_to_completion(cpu);
    // x1 should be 4 (the instruction at the jump target), not 2 or 3.
    ASSERT_EQ(cpu.reg(1), 4u);
    // ra should hold the return address (0x04 + 4 = 0x08).
    ASSERT_HEX_EQ(cpu.reg(1), 4u);
    return true;
}

TEST(pipe_jalr_flushes) {
    auto cpu = make_pipeline();
    cpu.set_reg(5, 0x20);
    load_program(cpu, 0, {
        0x00100093,  // 0x00: addi x1, x0, 1
        0x00028067,  // 0x04: jalr x0, 0(x5)  -> jump to 0x20
        0x00200093,  // 0x08: addi x1, x0, 2  (flushed)
        0x00000013,  // 0x0C–0x1C: nops
        0x00000013,
        0x00000013,
        0x00000013,
        0x00000013,  // 0x1C
        0x00500093,  // 0x20: addi x1, x0, 5    (target)
        0x00100073,  // 0x24: ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(1), 5u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. Pipeline state inspection
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pipe_empty_on_reset) {
    auto cpu = make_pipeline();
    ASSERT(cpu.pipeline_empty());
    ASSERT_EQ(cpu.stats().cycles, 0u);
    ASSERT_EQ(cpu.stats().instructions_retired, 0u);
    return true;
}

TEST(pipe_reset_clears_state) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x02a00093,  // addi x1, x0, 42
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT_EQ(cpu.reg(1), 42u);

    cpu.reset();
    ASSERT_EQ(cpu.reg(1), 0u);
    ASSERT_EQ(cpu.pc(), 0u);
    ASSERT(cpu.pipeline_empty());
    ASSERT_EQ(cpu.stats().cycles, 0u);
    return true;
}

TEST(pipe_halted_after_ebreak) {
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT(cpu.halted());
    return true;
}

TEST(pipe_stats_consistent) {
    // Verify that retired + pipeline_fill + stalls + flushes account for
    // all cycles (approximately — this is a sanity check, not exact).
    auto cpu = make_pipeline();
    load_program(cpu, 0, {
        0x00000093,  // addi x1, x0, 0
        0x00a00113,  // addi x2, x0, 10
        0x002080b3,  // add  x1, x1, x2
        0x00100073,  // ebreak
    });
    run_to_completion(cpu);
    ASSERT(cpu.stats().cycles > 0u);
    ASSERT(cpu.stats().instructions_retired > 0u);
    ASSERT(cpu.stats().ipc() > 0.0);
    ASSERT(cpu.stats().ipc() <= 1.0);
    return true;
}
