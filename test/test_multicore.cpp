/**
 * @file test_multicore.cpp
 * @brief Tests for the multi-core CPU system.
 *
 * Sections:
 *   1 (line  89) : Construction — 2-core system builds without errors
 *   2 (line 107) : Independent execution — each core runs its own program
 *   3 (line 151) : Shared memory — core 0 writes, core 1 reads
 *   4 (line 181) : Timer interrupt — fires on all cores
 *   5 (line 205) : Statistics — per-core and aggregate
 *   6 (line 229) : Reset — clears all state
 */

#include "core/multicore.hpp"
#include "test_framework.hpp"

using namespace riscv;

// Helper: smaller caches for faster tests
static MultiCoreConfig test_config(u32 num_cores = 2)
{
    MultiCoreConfig cfg;
    cfg.num_cores = num_cores;
    cfg.main_memory_size = 64 * 1024;  // 64 KB

    cfg.l1d = {
        .size_bytes    = 1024,
        .line_size     = 64,
        .associativity = 4,
        .hit_latency   = 1,
        .miss_penalty  = 10,
    };
    cfg.l2 = {
        .size_bytes    = 4096,
        .line_size     = 64,
        .associativity = 4,
        .hit_latency   = 5,
        .miss_penalty  = 50,
    };
    cfg.l3.size_bytes = 0;  // Disable L3
    return cfg;
}

static constexpr u32 EBREAK = 0x00100073;

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Construction
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_construction_2core) {
    auto sys = MultiCoreCPU(test_config(2));
    ASSERT_EQ(sys.num_cores(), 2u);
    ASSERT_EQ(sys.core(0).pc(), 0u);
    ASSERT_EQ(sys.core(1).pc(), 0u);
    return true;
}

TEST(mc_construction_4core) {
    auto sys = MultiCoreCPU(test_config(4));
    ASSERT_EQ(sys.num_cores(), 4u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. Independent execution
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_independent_programs) {
    auto sys = MultiCoreCPU(test_config(2));

    // Core 0 program at 0x0000
    sys.load_instruction(0x0000, 0x02a00093);  // addi x1, x0, 42
    sys.load_instruction(0x0004, EBREAK);

    // Core 1 program at 0x1000
    sys.load_instruction(0x1000, 0x06300093);  // addi x1, x0, 99
    sys.load_instruction(0x1004, EBREAK);

    sys.set_core_pc(0, 0x0000);
    sys.set_core_pc(1, 0x1000);

    sys.run_until_all_halted(1000);

    ASSERT_EQ(sys.core(0).reg(1), 42u);
    ASSERT_EQ(sys.core(1).reg(1), 99u);
    return true;
}

TEST(mc_core_id_via_register) {
    auto sys = MultiCoreCPU(test_config(2));

    // Both cores run the same program, but a0 = hart_id
    sys.load_instruction(0x0000, 0x00150093);  // addi x1, a0, 1
    sys.load_instruction(0x0004, EBREAK);

    sys.set_core_pc(0, 0x0000);
    sys.set_core_pc(1, 0x0000);
    sys.set_core_reg(0, 10, 0);  // a0 = 0 (hart 0)
    sys.set_core_reg(1, 10, 1);  // a0 = 1 (hart 1)

    sys.run_until_all_halted(1000);

    ASSERT_EQ(sys.core(0).reg(1), 1u);  // 0 + 1
    ASSERT_EQ(sys.core(1).reg(1), 2u);  // 1 + 1
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Shared memory
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_shared_memory_write_read) {
    auto sys = MultiCoreCPU(test_config(2));

    sys.main_memory().write32(0x8000, 42);

    // Core 0: read from 0x8000 into x1
    sys.load_instruction(0x0000, 0x00008137);  // lui  x2, 0x8
    sys.load_instruction(0x0004, 0x00012083);  // lw   x1, 0(x2)
    sys.load_instruction(0x0008, EBREAK);

    // Core 1: same program
    sys.load_instruction(0x1000, 0x00008137);
    sys.load_instruction(0x1004, 0x00012183);  // lw x3, 0(x2)
    sys.load_instruction(0x1008, EBREAK);

    sys.set_core_pc(0, 0x0000);
    sys.set_core_pc(1, 0x1000);

    sys.run_until_all_halted(10000);

    // Both cores should see 42
    ASSERT_EQ(sys.core(0).reg(1), 42u);
    ASSERT_EQ(sys.core(1).reg(3), 42u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Timer interrupt
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_timer_ticks) {
    auto sys = MultiCoreCPU(test_config(2));

    sys.run_cycles(100);
    ASSERT_EQ(sys.timer().current_time(), 100u);
    return true;
}

TEST(mc_timer_sets_mip) {
    auto sys = MultiCoreCPU(test_config(2));

    sys.timer().set_compare(50);
    sys.run_cycles(60);

    // Timer should have set MTIP on both cores
    ASSERT(sys.core(0).csrs().mip() & MInterrupt::MTIE);
    ASSERT(sys.core(1).csrs().mip() & MInterrupt::MTIE);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Statistics
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_stats_per_core) {
    auto sys = MultiCoreCPU(test_config(2));

    sys.load_instruction(0x0000, 0x02a00093);  // addi x1, x0, 42
    sys.load_instruction(0x0004, EBREAK);
    sys.load_instruction(0x1000, 0x06300093);  // addi x1, x0, 99
    sys.load_instruction(0x1004, EBREAK);

    sys.set_core_pc(0, 0x0000);
    sys.set_core_pc(1, 0x1000);

    sys.run_until_all_halted(1000);

    auto stats = sys.get_stats();
    ASSERT_EQ(stats.core_stats.size(), 2u);
    ASSERT(stats.core_stats[0].instructions_retired > 0u);
    ASSERT(stats.core_stats[1].instructions_retired > 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Reset
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mc_reset) {
    auto sys = MultiCoreCPU(test_config(2));

    sys.load_instruction(0x0000, 0x02a00093);
    sys.load_instruction(0x0004, EBREAK);
    sys.set_core_pc(0, 0x0000);
    sys.run_until_all_halted(1000);
    ASSERT_EQ(sys.core(0).reg(1), 42u);

    sys.reset();
    ASSERT_EQ(sys.core(0).reg(1), 0u);
    ASSERT_EQ(sys.core(0).pc(), 0u);
    ASSERT_EQ(sys.cycles(), 0u);
    return true;
}
