/**
 * @file test_programs.cpp
 * @brief Multi-instruction RV32I program tests.
 *
 * Sections:
 *   1 (line 34) : Sum 1..10, Fibonacci
 *   2 (line ) : memory copy, array swap
 *   3 (line ) : Function call / return, nested calls with stack
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Arithmetic Programs
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_sum_1_to_10()
{
    /*
     * Computes sum = 1 + 2 + … + 10 = 55.
     *
     *   addi x1, x0, 0      # sum = 0
     *   addi x2, x0, 1      # i = 1
     *   addi x3, x0, 10     # limit = 10
     * loop:
     *   add  x1, x1, x2     # sum += i
     *   addi x2, x2, 1      # i++
     *   bge  x3, x2, loop   # if limit >= i goto loop
     *   ebreak
     */
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0,  ADDI(1, 0, 0));
    cpu.load_instruction(4,  ADDI(2, 0, 1));
    cpu.load_instruction(8,  ADDI(3, 0, 10));
    // loop:
    cpu.load_instruction(12, ADD(1, 1, 2));
    cpu.load_instruction(16, ADDI(2, 2, 1));
    cpu.load_instruction(20, BGE(3, 2, -8));
    cpu.load_instruction(24, EBREAK);

    h.run(150);
    ASSERT_EQ(cpu.reg(1), 55u);
    return true;
}

TEST(prog_sum_1_to_10_cpu)  { return run_sum_1_to_10<CPUH>(); }
TEST(prog_sum_1_to_10_pipe) { return run_sum_1_to_10<PipeH>(); }

template <typename Harness>
bool run_fibonacci()
{
    /*
     * Computes Fib(10) = 55.
     *
     *   addi x1, x0, 0      # a = 0  (Fib(0))
     *   addi x2, x0, 1      # b = 1  (Fib(1))
     *   addi x3, x0, 10     # n = 10
     *   addi x4, x0, 0      # counter
     * loop:
     *   beq  x4, x3, done   # if counter == n, done
     *   add  x5, x1, x2     # temp = a + b
     *   addi x1, x2, 0      # a = b
     *   addi x2, x5, 0      # b = temp
     *   addi x4, x4, 1      # counter++
     *   jal  x0, loop
     * done:
     *   ebreak
     */
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0,  ADDI(1, 0, 0));
    cpu.load_instruction(4,  ADDI(2, 0, 1));
    cpu.load_instruction(8,  ADDI(3, 0, 10));
    cpu.load_instruction(12, ADDI(4, 0, 0));
    // loop:
    cpu.load_instruction(16, BEQ(4, 3, 24));
    cpu.load_instruction(20, ADD(5, 1, 2));
    cpu.load_instruction(24, ADDI(1, 2, 0));
    cpu.load_instruction(28, ADDI(2, 5, 0));
    cpu.load_instruction(32, ADDI(4, 4, 1));
    cpu.load_instruction(36, JAL(0, -20));
    // done:
    cpu.load_instruction(40, EBREAK);

    h.run(150);
    ASSERT_EQ(cpu.reg(1), 55u);
    return true;
}

TEST(prog_fibonacci_cpu)  { return run_fibonacci<CPUH>(); }
TEST(prog_fibonacci_pipe) { return run_fibonacci<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  2. Memory Programs
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_memory_copy()
{
    // Copy 4 words from 0x1000 to 0x2000
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x1000, 0xAAAA'AAAA);
    cpu.memory().write32(0x1004, 0xBBBB'BBBB);
    cpu.memory().write32(0x1008, 0xCCCC'CCCC);
    cpu.memory().write32(0x100C, 0xDDDD'DDDD);

    cpu.load_instruction(0,  LUI(1, 0x1000)); // src   = 0x1000
    cpu.load_instruction(4,  LUI(2, 0x2000)); // dest  = 0x2000
    cpu.load_instruction(8,  ADDI(3, 0, 4));  // count = 4
    // loop:
    cpu.load_instruction(12, BEQ(3, 0, 28)); // if count == 0, done
    cpu.load_instruction(16, LW(4, 0, 1));
    cpu.load_instruction(20, SW(4, 0, 2));
    cpu.load_instruction(24, ADDI(1, 1, 4));
    cpu.load_instruction(28, ADDI(2, 2, 4));
    cpu.load_instruction(32, ADDI(3, 3, -1));
    cpu.load_instruction(36, JAL(0, -24));
    // done:
    cpu.load_instruction(40, EBREAK);

    h.run(150);
    ASSERT_HEX_EQ(cpu.memory().read32(0x2000).value, 0xAAAA'AAAAu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x2004).value, 0xBBBB'BBBBu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x2008).value, 0xCCCC'CCCCu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200C).value, 0xDDDD'DDDDu);
    return true;
}

TEST(prog_memory_copy_cpu)  { return run_memory_copy<CPUH>(); }
TEST(prog_memory_copy_pipe) { return run_memory_copy<PipeH>(); }

template <typename Harness>
bool run_array_swap()
{
    // Swap arr[0] and arr[1] in memory
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x1000, 10);
    cpu.memory().write32(0x1004, 20);

    cpu.load_instruction(0,  LUI(1, 0x1000));
    cpu.load_instruction(4,  LW(4, 0, 1));
    cpu.load_instruction(8,  LW(5, 4, 1));
    cpu.load_instruction(12, SW(5, 0, 1));
    cpu.load_instruction(16, SW(4, 4, 1));
    cpu.load_instruction(20, EBREAK);

    h.run(6);
    ASSERT_EQ(cpu.memory().read32(0x1000).value, 20u);
    ASSERT_EQ(cpu.memory().read32(0x1004).value, 10u);
    return true;
}

TEST(prog_array_swap_cpu)  { return run_array_swap<CPUH>(); }
TEST(prog_array_swap_pipe) { return run_array_swap<PipeH>(); }

template <typename Harness>
bool run_load_neg_offset()
{
    /*
     * Use a pointer to the end of a buffer and access elements via
     * negative offsets
     *
     *   # x1 points past the end of a 2-word buffer at 0x1000
     *   lui  x1, 0x1
     *   addi x1, x1, 8      # x1 = 0x1008
     *   lw   x2, -8(x1)     # x2 = mem[0x1000]
     *   lw   x3, -4(x1)     # x3 = mem[0x1004]
     *   ebreak
     */
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x1000, 0x1111'1111);
    cpu.memory().write32(0x1004, 0x2222'2222);

    cpu.load_instruction(0,  LUI(1, 0x1000));
    cpu.load_instruction(4,  ADDI(1, 1, 8));
    cpu.load_instruction(8,  LW(2, -8, 1));
    cpu.load_instruction(12, LW(3, -4, 1));
    cpu.load_instruction(16, EBREAK);

    h.run(5);
    ASSERT_HEX_EQ(cpu.reg(2), 0x1111'1111u);
    ASSERT_HEX_EQ(cpu.reg(3), 0x2222'2222u);
    return true;
}

TEST(prog_load_neg_offset_cpu)  { return run_load_neg_offset<CPUH>(); }
TEST(prog_load_neg_offset_pipe) { return run_load_neg_offset<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  3. Function call programs
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_function_call()
{
    /*
     * Call double(5) -> returns 10.
     *
     * main:
     *   addi x10, x0, 5    # arg = 5
     *   jal  x1, double    # call
     *   addi x11, x10, 0   # save result
     *   ebreak
     * double:
     *   slli x10, x10, 1   # return arg * 2
     *   jalr x0, 0(x1)     # ret
     */
    Harness h;
    auto& cpu = h.get();

    // main:
    cpu.load_instruction(0,  ADDI(10, 0, 5));
    cpu.load_instruction(4,  JAL(1, 12));
    cpu.load_instruction(8,  ADDI(11, 10, 0));
    cpu.load_instruction(12, EBREAK);
    // double:
    cpu.load_instruction(16, SLLI(10, 10, 1));
    cpu.load_instruction(20, JALR(0, 0, 1));

    h.run(15);
    ASSERT_EQ(cpu.reg(10), 10u);
    ASSERT_EQ(cpu.reg(11), 10u);
    return true;
}

TEST(prog_function_call_cpu)  { return run_function_call<CPUH>(); }
TEST(prog_function_call_pipe) { return run_function_call<PipeH>(); }

template <typename Harness>
bool run_nested_calls()
{
    /*
     * triple(3): calls double(3) -> 6, then adds the original arg → 9.
     * Demonstrates caller-save via stack (sp = x2).
     *
     * main:
     *   lui  sp, 0x3      # sp = 0x3000
     *   addi a0, x0, 3    # arg = 3
     *   jal  ra, triple
     *   ebreak
     *
     * triple:
     *   addi sp, sp, -8
     *   sw   ra, 0(sp)
     *   sw   a0, 4(sp)    # save original arg
     *   jal  ra, double   # a0 = double(a0) = 6
     *   lw   a1, 4(sp)    # a1 = original arg (3)
     *   add  a0, a0, a1   # a0 = 6 + 3 = 9
     *   lw   ra, 0(sp)
     *   addi sp, sp, 8
     *   jalr x0, 0(ra)
     *
     * double:
     *   slli a0, a0, 1
     *   jalr x0, 0(ra)
     */
    Harness h;
    auto& cpu = h.get();

    // main:
    cpu.load_instruction(0,  LUI(2, 0x3000));
    cpu.load_instruction(4,  ADDI(10, 0, 3));
    cpu.load_instruction(8,  JAL(1, 12));
    cpu.load_instruction(12, EBREAK);
    // triple:
    cpu.load_instruction(16, ADDI(2, 2, -8));
    cpu.load_instruction(20, SW(1, 0, 2));
    cpu.load_instruction(24, SW(10, 4, 2));
    cpu.load_instruction(28, JAL(1, 24));
    cpu.load_instruction(32, LW(11, 4, 2));
    cpu.load_instruction(36, ADD(10, 10, 11));
    cpu.load_instruction(40, LW(1, 0, 2));
    cpu.load_instruction(44, ADDI(2, 2, 8));
    cpu.load_instruction(48, JALR(0, 0, 1));
    // double:
    cpu.load_instruction(52, SLLI(10, 10, 1));
    cpu.load_instruction(56, JALR(0, 0, 1));

    h.run(40);
    // triple(3) = double(3) + 3 = 6 + 3 = 9
    ASSERT_EQ(cpu.reg(10), 9u);
    return true;
}

TEST(prog_nested_calls_cpu)  { return run_nested_calls<CPUH>(); }
TEST(prog_nested_calls_pipe) { return run_nested_calls<PipeH>(); }
