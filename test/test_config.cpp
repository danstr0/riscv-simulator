/**
 * @file test_config.cpp
 * @brief Tests for the configuration file parser.
 *
 * @par Sections
 * @code
 *   1 (line  26) : Basic parsing
 *   2 (line  44) : Memory initialization
 *   3 (line 142) : Registers
 *   4 (line 213) : Cache
 *   5 (line 414) : Pipeline
 *   6 (line 475) : System
 *   7 (line 592) : Sweep
 *   8 (line 736) : Multicore configuration
 *   9 (line 906) : apply_config_param
 *  10 (line 924) : Full config file
 * @endcode
 */

#include "core/config.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic parsing
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_empty)
{
    auto r = parse_config("");
    ASSERT(r.ok);
    return true;
}

TEST(cfg_comments_only)
{
    auto r = parse_config("# this is a comment\n; this as well\n");
    ASSERT(r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Memory initialization
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_memory_size)
{
    auto r = parse_config(R"(
        [memory]
        size = 128K
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_size_kb, 128u);
    return true;
}

TEST(cfg_mem_init_words)
{
    auto r = parse_config(R"(
        [memory.init]
        0x8000 = [1, 2, 3, 4]
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_init.size(), 1u);

    auto& e = r.config.mem_init[0];
    ASSERT_EQ(e.address, 0x8000u);
    ASSERT(e.kind == MemInitEntry::Kind::WORDS);
    ASSERT_EQ(e.words.size(), 4u);
    ASSERT_EQ(e.words[0], 1u);
    ASSERT_EQ(e.words[3], 4u);
    return true;
}

TEST(cfg_mem_init_hex_values)
{
    auto r = parse_config(R"(
        [memory.init]
        0x1000 = [0xDEAD, 0xBEEF]
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_init[0].words[0], 0xDEADu);
    ASSERT_EQ(r.config.mem_init[0].words[1], 0xBEEFu);
    return true;
}

TEST(cfg_mem_init_fill)
{
    auto r = parse_config(R"(
        [memory.init]
        0x9000 = fill(0xFF, 256)
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_init.size(), 1u);

    auto& e = r.config.mem_init[0];
    ASSERT(e.kind == MemInitEntry::Kind::FILL);
    ASSERT_EQ(e.fill_val, 0xFFu);
    ASSERT_EQ(e.fill_count, 256u);
    return true;
}

TEST(cfg_mem_init_string)
{
    auto r = parse_config(R"(
        [memory.init]
        0xA000 = "hello"
    )");
    ASSERT(r.ok);

    auto& e = r.config.mem_init[0];
    ASSERT(e.kind == MemInitEntry::Kind::STRING);
    ASSERT(e.str == "hello");
    return true;
}

TEST(cfg_mem_init_single_value)
{
    auto r = parse_config(R"(
        [memory.init]
        0x5000 = 42
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_init[0].words.size(), 1u);
    ASSERT_EQ(r.config.mem_init[0].words[0], 42u);
    return true;
}

TEST(cfg_mem_init_error_bad_address)
{
    auto r = parse_config(R"(
        [memory.init]
        not_an_address = [1, 2, 3]
    )");
    ASSERT(!r.ok);
    ASSERT(r.errors.size() > 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Registers
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_registers)
{
    auto r = parse_config(R"(
        [registers]
        a0 = 8
        a1 = 0x8000
        sp = 0xFFF0
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.reg_init.size(), 3u);

    ASSERT_EQ(r.config.reg_init[0].reg_idx, 10u); // a0
    ASSERT_EQ(r.config.reg_init[0].value, 8u);

    ASSERT_EQ(r.config.reg_init[1].reg_idx, 11u); // a1
    ASSERT_EQ(r.config.reg_init[1].value, 0x8000u);

    ASSERT_EQ(r.config.reg_init[2].reg_idx, 2u);  // sp
    ASSERT_EQ(r.config.reg_init[2].value, 0xFFF0u);
    return true;
}

TEST(cfg_registers_x_names)
{
    auto r = parse_config(R"(
        [registers]
        x1 = 100
        x10 = 0xFF
    )");
    ASSERT(r.ok);

    ASSERT_EQ(r.config.reg_init[0].reg_idx, 1u);
    ASSERT_EQ(r.config.reg_init[0].value, 100u);

    ASSERT_EQ(r.config.reg_init[1].reg_idx, 10u);
    ASSERT_EQ(r.config.reg_init[1].value, 0xFFu);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_registers_error_x0)
{
    auto r1 = parse_config(R"(
        [registers]
        x0 = 4
    )");
    ASSERT(!r1.ok);

    auto r2 = parse_config(R"(
        [registers]
        zero = 4
    )");
    ASSERT(!r2.ok);
    return true;
}

TEST(cfg_registers_error_bad_name)
{
    auto r = parse_config(R"(
        [registers]
        x99 = 42
    )");
    ASSERT(!r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Cache
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_cache_l1)
{
    auto r = parse_config(R"(
        [cache.l1]
        size = 16K
        assoc = 4
        line = 32
        replacement = plru
        write = writethrough
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l1_size, 16u * 1024);
    ASSERT_EQ(r.config.l1_assoc, 4u);
    ASSERT_EQ(r.config.l1_line, 32u);
    ASSERT(r.config.l1_replacement == "plru");
    ASSERT(r.config.l1_write == "writethrough");
    return true;
}

TEST(cfg_cache_l2)
{
    auto r = parse_config(R"(
        [cache.l2]
        size = 32K
        assoc = 16
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l2_size, 32u * 1024);
    ASSERT_EQ(r.config.l2_assoc, 16u);
    return true;
}

TEST(cfg_cache_l3)
{
    // L2 cache disabled by default
    auto r = parse_config(R"(
        [cache.l2]
        size = 32K

        [cache.l3]
        size = 64K
        assoc = 8
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l3_size, 64u * 1024);
    ASSERT_EQ(r.config.l3_assoc, 8u);
    return true;
}

TEST(cfg_cache_disable)
{
    // Cache size 0 is fine - disables cache level
    auto r = parse_config(R"(
        [cache.l1]
        size = 0
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l1_size, 0u);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_cache_error_bad_size)
{
    // Size isn't power of 2 -> error
    auto r1 = parse_config(R"(
        [cache.l1]
        size = 6K
    )");
    ASSERT(!r1.ok);

    auto r2 = parse_config(R"(
        [cache.l2]
        size = 6K
    )");
    ASSERT(!r2.ok);

    auto r3 = parse_config(R"(
        [cache.l2]
        size = 4K

        [cache.l1]
        size = 6K
    )");
    ASSERT(!r3.ok);

    // Size must be >= line_size * assoc
    auto r4 = parse_config(R"(
        [cache.l1]
        size = 2K
        assoc = 32
        line = 128
    )");
    ASSERT(!r4.ok);
    return true;
}

TEST(cfg_cache_error_bad_assoc)
{
    // Can't have assoc = 0
    auto r1 = parse_config(R"(
        [cache.l1]
        assoc = 0
    )");
    ASSERT(!r1.ok);

    // Or assoc not a power-of-2
    auto r2 = parse_config(R"(
        [cache.l1]
        assoc = 6
    )");
    ASSERT(!r2.ok);
    return true;
}

TEST(cfg_cache_error_bad_line)
{
    // Can't have line size = 0
    auto r1 = parse_config(R"(
        [cache.l1]
        line = 0
    )");
    ASSERT(!r1.ok);

    // Or line size not a power-of-2
    auto r2 = parse_config(R"(
        [cache.l1]
        line = 48
    )");
    ASSERT(!r2.ok);
    return true;
}

TEST(cfg_cache_error_unknown_replacement)
{
    auto r = parse_config(R"(
        [cache.l1]
        replacement = pmu
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_cache_error_unknown_write)
{
    // Typo in policy name
    auto r = parse_config(R"(
        [cache.l1]
        write = writetrough
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_cache_error_missing_layer)
{
    // Can't have L2 or L3 without L1
    auto r1 = parse_config(R"(
        [cache.l1]
        size = 0

        [cache.l2]
        size = 16K
        assoc = 8
    )");
    ASSERT(!r1.ok);

    auto r2 = parse_config(R"(
        [cache.l1]
        size = 0

        [cache.l3]
        size = 16K
        assoc = 8
    )");
    ASSERT(!r2.ok);

    // L1 + no L2 -> can't have L3
    auto r3 = parse_config(R"(
        [cache.l1]
        size = 4K

        [cache.l2]
        size = 0
        assoc = 8

        [cache.l3]
        size = 32K
        assoc = 16
    )");
    ASSERT(!r3.ok);
    ASSERT_EQ(r3.config.l1_size, 4u * 1024);
    ASSERT_EQ(r3.config.l2_size, 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Pipeline
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_pipeline)
{
    auto r = parse_config(R"(
        [pipeline]
        forwarding = none
        branch_pred = not_taken
        mispredict_penalty = 5
    )");
    ASSERT(r.ok);
    ASSERT(r.config.forwarding == "none");
    ASSERT(r.config.branch_pred == "not_taken");
    ASSERT_EQ(r.config.mispredict_penalty, 5u);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_pipeline_error_bad_forwarding)
{
    // Forwarding policy that isn't 'none' or 'partial'
    // 'full' (EX->EX) isn't implemented
    auto r = parse_config(R"(
        [pipeline]
        forwarding = full
        branch_pred = always_taken
        mispredict_penalty = 10
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_pipeline_error_bad_predictor)
{
    // 'always' instead of 'always_taken'
    auto r = parse_config(R"(
        [pipeline]
        forwarding = none
        branch_pred = always
        mispredict_penalty = 15
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_pipeline_error_bad_penalty)
{
    // Mispredict penalty can't be 0
    auto r = parse_config(R"(
        [pipeline]
        forwarding = none
        branch_pred = not_taken
        mispredict_penalty = 0
    )");
    ASSERT(!r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  6. System
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_system_simple)
{
    auto r = parse_config(R"(
        [system]
        cpu = simple
        max_cycles = 10000
    )");
    ASSERT(r.ok);
    ASSERT(r.config.cpu_type == "simple");
    ASSERT_EQ(r.config.max_cycles, 10000u);
    return true;
}

TEST(cfg_system_pipeline)
{
    auto r = parse_config(R"(
        [system]
        cpu = pipelined
        max_cycles = 1000
    )");
    ASSERT(r.ok);
    ASSERT(r.config.cpu_type == "pipelined");
    ASSERT_EQ(r.config.max_cycles, 1000u);
    return true;
}

TEST(cfg_system_multicore)
{
    auto r = parse_config(R"(
        [system]
        cpu = multicore
        cores = 4

        [cache.l1]
        ; Disable cache -- L1 enabled by default but requires L2
        size = 0
    )");

    ASSERT(r.ok);
    ASSERT(r.config.cpu_type == "multicore");
    ASSERT_EQ(r.config.num_cores, 4u);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_system_error_zero_cycles)
{
    // Max cycles must be > 0
    auto r = parse_config(R"(
        [system]
        max_cycles = 0
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_system_error_core_mismatch)
{
    // 'simple' and 'pipelined' cannot be assigned > 1 core
    auto r1 = parse_config(R"(
        [system]
        cpu = simple
        cores = 3
    )");
    ASSERT(!r1.ok);

    auto r2 = parse_config(R"(
        [system]
        cpu = pipelined
        cores = 5
    )");
    ASSERT(!r2.ok);
    return true;

    // 'multicore' cannot be assigned 1 core
    auto r3 = parse_config(R"(
        [system]
        cpu = multicore
        cores = 1

        [cache.l1]
        size = 0
    )");
    ASSERT(!r3.ok);
    return true;
}

TEST(cfg_system_error_zero_cores)
{
    auto r = parse_config(R"(
        [system]
        cpu = pipelined
        cores = 0
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_system_error_too_many_cores)
{
    auto r = parse_config(R"(
        [system]
        cpu = multicore
        cores = 20

        [cache.l1]
        size = 0
    )");
    ASSERT(!r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Sweep
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_sweep_cache_axes)
{
    auto r = parse_config(R"(
        [sweep]
        cache.l1.size = [4K, 8K, 16K]
        cache.l1.assoc = [4, 8, 16]
        cache.l1.line = [32, 64, 128]
        cache.l1.replacement = [lru, mru, plru, fifo, random]
        cache.l1.write = [writeback, writethrough]
        cache.l2.size = [8K, 16K, 32K]
        cache.l2.assoc = [4, 8, 16]
        cache.l3.size = [16K, 32K, 64K]
        cache.l3.assoc = [4, 8, 16]
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.sweeps.size(), 9u);

    auto& l1_size = r.config.sweeps[0];
    ASSERT(l1_size.param == "cache.l1.size");
    ASSERT_EQ(l1_size.values.size(), 3u);
    ASSERT(l1_size.values[0] == "4K");
    ASSERT(l1_size.values[1] == "8K");
    ASSERT(l1_size.values[2] == "16K");

    auto& l1_assoc = r.config.sweeps[1];
    ASSERT(l1_assoc.param == "cache.l1.assoc");
    ASSERT_EQ(l1_assoc.values.size(), 3u);
    ASSERT(l1_assoc.values[0] == "4");
    ASSERT(l1_assoc.values[1] == "8");
    ASSERT(l1_assoc.values[2] == "16");

    auto& l1_line = r.config.sweeps[2];
    ASSERT(l1_line.param == "cache.l1.line");
    ASSERT_EQ(l1_line.values.size(), 3u);
    ASSERT(l1_line.values[0] == "32");
    ASSERT(l1_line.values[1] == "64");
    ASSERT(l1_line.values[2] == "128");

    auto& l1_replacement = r.config.sweeps[3];
    ASSERT(l1_replacement.param == "cache.l1.replacement");
    ASSERT_EQ(l1_replacement.values.size(), 5u);
    ASSERT(l1_replacement.values[0] == "lru");
    ASSERT(l1_replacement.values[1] == "mru");
    ASSERT(l1_replacement.values[2] == "plru");
    ASSERT(l1_replacement.values[3] == "fifo");
    ASSERT(l1_replacement.values[4] == "random");

    auto& l1_write = r.config.sweeps[4];
    ASSERT(l1_write.param == "cache.l1.write");
    ASSERT_EQ(l1_write.values.size(), 2u);
    ASSERT(l1_write.values[0] == "writeback");
    ASSERT(l1_write.values[1] == "writethrough");

    auto& l2_size = r.config.sweeps[5];
    ASSERT(l2_size.param == "cache.l2.size");
    ASSERT_EQ(l2_size.values.size(), 3u);
    ASSERT(l2_size.values[0] == "8K");
    ASSERT(l2_size.values[1] == "16K");
    ASSERT(l2_size.values[2] == "32K");

    auto& l2_assoc = r.config.sweeps[6];
    ASSERT(l2_assoc.param == "cache.l2.assoc");
    ASSERT_EQ(l2_assoc.values.size(), 3u);
    ASSERT(l2_assoc.values[0] == "4");
    ASSERT(l2_assoc.values[1] == "8");
    ASSERT(l2_assoc.values[2] == "16");

    auto& l3_size = r.config.sweeps[7];
    ASSERT(l3_size.param == "cache.l3.size");
    ASSERT_EQ(l3_size.values.size(), 3u);
    ASSERT(l3_size.values[0] == "16K");
    ASSERT(l3_size.values[1] == "32K");
    ASSERT(l3_size.values[2] == "64K");

    auto& l3_assoc = r.config.sweeps[8];
    ASSERT(l3_assoc.param == "cache.l3.assoc");
    ASSERT_EQ(l3_assoc.values.size(), 3u);
    ASSERT(l3_assoc.values[0] == "4");
    ASSERT(l3_assoc.values[1] == "8");
    ASSERT(l3_assoc.values[2] == "16");
    return true;
}

TEST(cfg_sweep_pipeline_axes)
{
    auto r = parse_config(R"(
        [sweep]
        pipeline.forwarding = [none, partial]
        pipeline.branch_pred = [not_taken, always_taken, backward_taken, bimodal_1bit, bimodal_2bit]
        pipeline.mispredict_penalty = [3, 5, 10]
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.sweeps.size(), 3u);

    auto& pipe_forwarding = r.config.sweeps[0];
    ASSERT(pipe_forwarding.param == "pipeline.forwarding");
    ASSERT_EQ(pipe_forwarding.values.size(), 2u);
    ASSERT(pipe_forwarding.values[0] == "none");
    ASSERT(pipe_forwarding.values[1] == "partial");

    auto& branch_pred = r.config.sweeps[1];
    ASSERT(branch_pred.param == "pipeline.branch_pred");
    ASSERT_EQ(branch_pred.values.size(), 5u);
    ASSERT(branch_pred.values[0] == "not_taken");
    ASSERT(branch_pred.values[1] == "always_taken");
    ASSERT(branch_pred.values[2] == "backward_taken");
    ASSERT(branch_pred.values[3] == "bimodal_1bit");
    ASSERT(branch_pred.values[4] == "bimodal_2bit");

    auto& mispred_penalty = r.config.sweeps[2];
    ASSERT(mispred_penalty.param == "pipeline.mispredict_penalty");
    ASSERT_EQ(mispred_penalty.values.size(), 3u);
    ASSERT(mispred_penalty.values[0] == "3");
    ASSERT(mispred_penalty.values[1] == "5");
    ASSERT(mispred_penalty.values[2] == "10");
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_sweep_error_bad_member)
{
    auto r = parse_config(R"(
        [sweep]
        cache.l1.replacement = [lru, mru, belady]
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_sweep_error_duplicate)
{
    auto r = parse_config(R"(
        [sweep]
        cache.l1.assoc = [4, 8, 8, 16]
    )");
    ASSERT(!r.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  8. Multicore configuration
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_per_core)
{
    auto r = parse_config(R"(
        [system]
        cpu = multicore
        cores = 3

        [cache.l1]
        size = 0

        [cores.0]
        x1 = 1
        x2 = 4
        pc = 16

        [cores.1]
        x3 = 9
        x4 = 16
        pc = 0x40
        program = fibonacci.s

        [cores.2]
        x5 = 25
        x6 = 36
        pc = 256
        program = sum_10.s
    )");
    ASSERT(r.ok);

    auto& core0 = r.config.cores[0];
    ASSERT_EQ(core0.reg_init[0].reg_idx, 1u);
    ASSERT_EQ(core0.reg_init[0].value, 1u);
    ASSERT_EQ(core0.reg_init[1].reg_idx, 2u);
    ASSERT_EQ(core0.reg_init[1].value, 4u);
    ASSERT_EQ(core0.pc, 16u);

    auto& core1 = r.config.cores[1];
    ASSERT_EQ(core1.reg_init[0].reg_idx, 3u);
    ASSERT_EQ(core1.reg_init[0].value, 9u);
    ASSERT_EQ(core1.reg_init[1].reg_idx, 4u);
    ASSERT_EQ(core1.reg_init[1].value, 16u);
    ASSERT_HEX_EQ(core1.pc, 0x40u);
    ASSERT(core1.program == "fibonacci.s");

    auto& core2 = r.config.cores[2];
    ASSERT_EQ(core2.reg_init[0].reg_idx, 5u);
    ASSERT_EQ(core2.reg_init[0].value, 25u);
    ASSERT_EQ(core2.reg_init[1].reg_idx, 6u);
    ASSERT_EQ(core2.reg_init[1].value, 36u);
    ASSERT_EQ(core2.pc, 256u);
    ASSERT(core2.program == "sum_10.s");
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(cfg_per_core_error_x0)
{
    auto r = parse_config(R"(
        [system]
        cpu = multicore
        cores = 2

        [cache.l1]
        size = 0

        [cores.0]
        x1 = 1

        [cores.1]
        x0 = 2
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_per_core_error_bad_pc)
{
    // Misaligned PC
    auto r1 = parse_config(R"(
        [system]
        cpu = multicore
        cores = 2

        [cache.l1]
        size = 0

        [cores.0]
        pc = 15

        [cores.1]
        pc = 8
    )");
    ASSERT(!r1.ok);

    // PC above main memory
    auto r2 = parse_config(R"(
        [system]
        cpu = multicore
        cores = 2

        [memory]
        size = 1K

        [cache.l1]
        size = 0

        [cores.0]
        pc = 2048
    )");
    ASSERT(!r2.ok);
    return true;
}

TEST(cfg_multicore_error_no_l2)
{
    // Multicore can have no caches, but MUST have L2 if there is L1
    auto r = parse_config(R"(
        [system]
        cpu = multicore
        cores = 2

        [cache.l1]
        ; Non-zero -> enabled
        size = 4K

        [cache.l2]
        size = 0
    )");
    ASSERT(!r.ok);
    return true;
}

TEST(cfg_multicore_error_core_index)
{
    // defining cores.N section with N > num_cores
    auto r1 = parse_config(R"(
        [system]
        cpu = multicore
        cores = 2

        [cache.l1]
        size = 0

        [cores.4]
        x1 = 4
    )");
    ASSERT(!r1.ok);
    return true;

    // N > 15
    auto r2 = parse_config(R"(
        [system]
        cpu = multicore
        cores = 16

        [cache.l1]
        size = 0

        [cores.16]
        x1 = 4
    )");
    ASSERT(!r2.ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  9. apply_config_param
// ═══════════════════════════════════════════════════════════════════════

TEST(cfg_apply_param)
{
    SimConfig cfg;
    apply_config_param(cfg, "cache.l1.size", "64K");
    ASSERT_EQ(cfg.l1_size, 64u * 1024);

    apply_config_param(cfg, "pipeline.forwarding", "none");
    ASSERT(cfg.forwarding == "none");

    apply_config_param(cfg, "pipeline.mispredict_penalty", "7");
    ASSERT_EQ(cfg.mispredict_penalty, 7u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Full config file
// ═══════════════════════════════════════════════════════════════════════════

TEST(cfg_full_file)
{
    auto r = parse_config(R"(
        ; Config for sum-to-N benchmark

        [memory]
        size = 256K

        [memory.init]
        0x8000 = [10, 20, 30, 40, 50, 60, 70, 80]

        [cache.l1]
        size = 32K
        assoc = 8
        line = 64
        replacement = lru
        write = writeback
        write_alloc = no_allocate

        [cache.l2]
        size = 64K

        [pipeline]
        forwarding = partial
        branch_pred = bimodal_2bit
        mispredict_penalty = 3

        [registers]
        a0 = 8       # array length
        a1 = 0x8000  # array pointer
        sp = 0xFFF0

        [sweep]
        cache.l1.size = [16K, 32K, 64K]
        pipeline.forwarding = [none, partial]
    )");

    ASSERT(r.ok);
    ASSERT_EQ(r.config.mem_size_kb, 256u);
    ASSERT_EQ(r.config.mem_init.size(), 1u);
    ASSERT_EQ(r.config.mem_init[0].words.size(), 8u);
    ASSERT_EQ(r.config.l1_size, 32u * 1024);
    ASSERT_EQ(r.config.l2_size, 64u * 1024);
    ASSERT(r.config.forwarding == "partial");
    ASSERT_EQ(r.config.reg_init.size(), 3u);
    ASSERT_EQ(r.config.sweeps.size(), 2u);
    return true;
}
