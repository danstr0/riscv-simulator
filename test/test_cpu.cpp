/**
 * @file test_cpu.cpp
 * @brief Tests for the CPU API.
 *
 * Sections:
 *   1 (line 18) : Basic functions
 *   2 (line 58) : Lifecycle
 *   3 (line 83) : Statistics
 */

#include "test_framework.hpp"
#include "test_utils.hpp"
#include <sstream>

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic functions
// ═══════════════════════════════════════════════════════════════════════

TEST(cpu_run_until_pc)
{
    auto cpu = make_cpu();

    cpu.load_instruction(0,  ADDI(1, 0, 1));
    cpu.load_instruction(4,  ADDI(2, 0, 2));
    cpu.load_instruction(8,  ADDI(3, 0, 3));
    cpu.load_instruction(12, ADDI(4, 0, 4));
    cpu.load_instruction(16, EBREAK);

    cpu.run_until_pc(0x08);

    ASSERT_EQ(cpu.pc(), 0x08u);
    ASSERT_EQ(cpu.reg(1), 1u);
    ASSERT_EQ(cpu.reg(2), 2u);
    ASSERT_EQ(cpu.reg(3), 0u); // not yet executed
    return true;
}

TEST(cpu_run_until_ecall)
{
    // This program terminates with ECALL instead of EBREAK
    auto cpu = make_cpu();

    cpu.load_instruction(0, ADDI(10, 0, 42));
    cpu.load_instruction(4, ECALL);

    u64 count = cpu.run_until_ecall();

    ASSERT_EQ(count, 2u);
    ASSERT_EQ(cpu.reg(10), 42u);
    ASSERT(!cpu.halted());    // ECALL does not halt
    ASSERT(cpu.last_result().ecall);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Lifecycle
// ═══════════════════════════════════════════════════════════════════════

TEST(cpu_invalid_instruction_halts)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0xFFFF'FFFF); // all-ones is invalid
    bool ok = cpu.step();
    ASSERT(!ok);
    ASSERT(cpu.halted());
    return true;
}

TEST(cpu_fetch_fault_halts)
{
    // Memory is 0x0–0xFFFF - set PC past the end
    auto cpu = make_cpu();
    cpu.set_pc(0x20000);
    bool ok = cpu.step();
    ASSERT(!ok);
    ASSERT(cpu.halted());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Statistics
// ═══════════════════════════════════════════════════════════════════════

TEST(cpu_stats_basic)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0,  ADDI(1, 0, 42));
    cpu.load_instruction(4,  ADDI(2, 1, 1));
    cpu.load_instruction(8,  ADDI(3, 2, 2));
    cpu.load_instruction(12, ADDI(4, 3, 3));
    cpu.load_instruction(16, ADDI(5, 4, 4));

    cpu.run(5);
    ASSERT_EQ(cpu.stats().instructions, 5u);
    return true;
}

TEST(cpu_stats_branches)
{
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, BEQ(1, 2, 8)); // taken
    cpu.load_instruction(8, BNE(1, 2, 8)); // not taken

    cpu.step(); 
    ASSERT_EQ(cpu.stats().branches, 1u);
    ASSERT_EQ(cpu.stats().branches_taken, 1u);

    cpu.step();
    ASSERT_EQ(cpu.stats().branches, 2u);
    ASSERT_EQ(cpu.stats().branches_taken, 1u);
    return true;
}

TEST(cpu_stats_loads_stores_jumps)
{
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0);
    cpu.set_reg(10, 0x200);

    cpu.load_instruction(0,  ADDI(1, 0, 42));
    cpu.load_instruction(4,  SW(1, 0, 10));
    cpu.load_instruction(8,  LW(2, 0, 10));
    cpu.load_instruction(12, JAL(1, 100));

    cpu.run(4);
    ASSERT_EQ(cpu.stats().loads, 1u);
    ASSERT_EQ(cpu.stats().stores, 1u);
    ASSERT_EQ(cpu.stats().jumps, 1u);
    ASSERT_EQ(cpu.stats().instructions, 4u);
    return true;
}
