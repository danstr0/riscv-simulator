/**
 * @file test_programs.cpp
 * @brief Integration tests: multi-instruction programs run on the CPU.
 *
 * Each test loads a small hand-assembled RV32I program, runs it, and
 * checks the final register / memory state.  These tests exercise the
 * full fetch–decode–execute loop and validate that all components work
 * together correctly.
 *
 * Programs:
 *   - Sum 1..10, Fibonacci, memory copy, array swap
 *   - Function call / return (JAL/JALR), nested calls with stack
 *   - run_until_pc, run_until_ecall, save/restore, tracing
 */

#include "core/cpu.hpp"
#include "test_framework.hpp"
#include <sstream>

using namespace riscv;

// ── Helpers ────────────────────────────────────────────────────────────

static CPU make_cpu() {
    auto mem = std::make_shared<FlatMemory>(0x0, 0x10000);
    return CPU(mem);
}

static void load_program(CPU& cpu, addr_t addr,
                         std::initializer_list<u32> instructions) {
    addr_t offset = addr;
    for (u32 inst : instructions) {
        cpu.load_instruction(offset, inst);
        offset += 4;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Arithmetic Programs
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(prog_sum_1_to_10) {
    /* Computes sum = 1 + 2 + … + 10 = 55.
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

    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x00000093,  // addi x1, x0, 0
        0x00100113,  // addi x2, x0, 1
        0x00a00193,  // addi x3, x0, 10
        0x002080b3,  // add  x1, x1, x2
        0x00110113,  // addi x2, x2, 1
        0xfe21dce3,  // bge  x3, x2, -8
        0x00100073,  // ebreak
    });

    cpu.run(1000);
    ASSERT_EQ(cpu.reg(1), 55u);
    ASSERT(cpu.halted());
    return true;
}

TEST(prog_fibonacci) {
    /* Computes Fib(10) = 55.
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

    auto cpu = make_cpu();
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

    cpu.run(1000);
    ASSERT_EQ(cpu.reg(1), 55u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Memory Programs
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(prog_memory_copy) {
    // Copy 4 words from 0x1000 to 0x2000.

    auto cpu = make_cpu();

    cpu.memory().write32(0x1000, 0xAAAAAAAA);
    cpu.memory().write32(0x1004, 0xBBBBBBBB);
    cpu.memory().write32(0x1008, 0xCCCCCCCC);
    cpu.memory().write32(0x100C, 0xDDDDDDDD);

    load_program(cpu, 0, {
        0x000010b7,  // lui  x1, 0x1         # src  = 0x1000
        0x00002137,  // lui  x2, 0x2         # dest = 0x2000
        0x00400193,  // addi x3, x0, 4       # count = 4
        0x00018e63,  // beq  x3, x0, +28     # if count==0, done
        0x0000a203,  // lw   x4, 0(x1)
        0x00412023,  // sw   x4, 0(x2)
        0x00408093,  // addi x1, x1, 4
        0x00410113,  // addi x2, x2, 4
        0xfff18193,  // addi x3, x3, -1
        0xfe9ff06f,  // jal  x0, -24
        0x00100073,  // ebreak
    });

    cpu.run(1000);

    ASSERT_HEX_EQ(cpu.memory().read32(0x2000).value, 0xAAAAAAAAu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x2004).value, 0xBBBBBBBBu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x2008).value, 0xCCCCCCCCu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200C).value, 0xDDDDDDDDu);
    return true;
}

TEST(prog_array_swap) {
    // Swap arr[0] and arr[1] in memory.

    auto cpu = make_cpu();

    cpu.memory().write32(0x1000, 10);
    cpu.memory().write32(0x1004, 20);

    load_program(cpu, 0, {
        0x000010b7,  // lui x1, 0x1
        0x0000a203,  // lw  x4, 0(x1)
        0x0040a283,  // lw  x5, 4(x1)
        0x0050a023,  // sw  x5, 0(x1)
        0x0040a223,  // sw  x4, 4(x1)
        0x00100073,  // ebreak
    });

    cpu.run(100);

    ASSERT_EQ(cpu.memory().read32(0x1000).value, 20u);
    ASSERT_EQ(cpu.memory().read32(0x1004).value, 10u);
    return true;
}

TEST(prog_load_negative_offset) {
    /* Use a pointer to the *end* of a buffer and access elements via
     * negative offsets.  Exercises signed address arithmetic.
     *
     *   # x1 points past the end of a 2-word buffer at 0x1000
     *   lui  x1, 0x1
     *   addi x1, x1, 8      # x1 = 0x1008
     *   lw   x2, -8(x1)     # x2 = mem[0x1000]
     *   lw   x3, -4(x1)     # x3 = mem[0x1004]
     *   ebreak
     */

    auto cpu = make_cpu();
    cpu.memory().write32(0x1000, 0x11111111);
    cpu.memory().write32(0x1004, 0x22222222);

    load_program(cpu, 0, {
        0x000010b7,  // lui  x1, 0x1
        0x00808093,  // addi x1, x1, 8
        0xff80a103,  // lw   x2, -8(x1)
        0xffc0a183,  // lw   x3, -4(x1)
        0x00100073,  // ebreak
    });

    cpu.run(100);

    ASSERT_HEX_EQ(cpu.reg(2), 0x11111111u);
    ASSERT_HEX_EQ(cpu.reg(3), 0x22222222u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Function call programs
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(prog_function_call) {
    /* Call double(5) -> returns 10.
     *
     * main:
     *   addi x10, x0, 5     # arg = 5
     *   jal  x1, double     # call
     *   addi x11, x10, 0    # save result
     *   ebreak
     * double:
     *   slli x10, x10, 1    # return arg * 2
     *   jalr x0, 0(x1)      # ret
     */

    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x00500513,  // addi x10, x0, 5
        0x00c000ef,  // jal  x1, +12
        0x00050593,  // addi x11, x10, 0
        0x00100073,  // ebreak
        0x00151513,  // slli x10, x10, 1
        0x00008067,  // jalr x0, 0(x1)
    });

    cpu.run(100);

    ASSERT_EQ(cpu.reg(10), 10u);
    ASSERT_EQ(cpu.reg(11), 10u);
    return true;
}

TEST(prog_nested_calls) {
    /* triple(3): calls double(3) -> 6, then adds the original arg → 9.
     * Demonstrates caller-save via stack (sp = x2).
     *
     * main:
     *   lui  sp, 0x3          # sp = 0x3000
     *   addi a0, x0, 3        # arg = 3
     *   jal  ra, triple
     *   ebreak
     *
     * triple:
     *   addi sp, sp, -8
     *   sw   ra, 0(sp)
     *   sw   a0, 4(sp)        # save original arg
     *   jal  ra, double       # a0 = double(a0) = 6
     *   lw   a1, 4(sp)        # a1 = original arg (3)
     *   add  a0, a0, a1       # a0 = 6 + 3 = 9
     *   lw   ra, 0(sp)
     *   addi sp, sp, 8
     *   jalr x0, 0(ra)
     *
     * double:
     *   slli a0, a0, 1
     *   jalr x0, 0(ra)
     */

    auto cpu = make_cpu();
    load_program(cpu, 0, {
        // main: 0x00–0x0C
        0x00003137,  // lui  sp, 0x3
        0x00300513,  // addi a0, x0, 3
        0x00c000ef,  // jal  ra, 12   -> triple at 0x10
        0x00100073,  // ebreak

        // triple: 0x10–0x30
        0xff810113,  // addi sp, sp, -8
        0x00112023,  // sw   ra, 0(sp)
        0x00a12223,  // sw   a0, 4(sp)
        0x018000ef,  // jal  ra, 24    -> double at 0x34
        0x00412583,  // lw   a1, 4(sp)
        0x00b50533,  // add  a0, a0, a1
        0x00012083,  // lw   ra, 0(sp)
        0x00810113,  // addi sp, sp, 8
        0x00008067,  // jalr x0, 0(ra)

        // double: 0x34
        0x00151513,  // slli a0, a0, 1
        0x00008067,  // jalr x0, 0(ra)
    });

    cpu.run(100);

    ASSERT_EQ(cpu.reg(10), 9u);  // triple(3) = double(3) + 3 = 6 + 3 = 9
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  CPU Control-flow Tests
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(prog_run_until_pc) {
    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x00100093,  // addi x1, x0, 1
        0x00200113,  // addi x2, x0, 2
        0x00300193,  // addi x3, x0, 3
        0x00400213,  // addi x4, x0, 4
        0x00100073,  // ebreak
    });

    cpu.run_until_pc(0x08);

    ASSERT_EQ(cpu.pc(), 0x08u);
    ASSERT_EQ(cpu.reg(1), 1u);
    ASSERT_EQ(cpu.reg(2), 2u);
    ASSERT_EQ(cpu.reg(3), 0u);  // Not yet executed
    return true;
}

TEST(prog_run_until_ecall) {
    /* A program that terminates with ECALL instead of EBREAK.
     * This is the intended halt mechanism for benchmarking workloads.
     *
     *   addi x10, x0, 42    # "exit code"
     *   ecall               # signal to host
     */

    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x02a00513,  // addi x10, x0, 42
        0x00000073,  // ecall
    });

    u64 count = cpu.run_until_ecall();

    ASSERT_EQ(count, 2u);
    ASSERT_EQ(cpu.reg(10), 42u);
    ASSERT(!cpu.halted());  // ECALL does not halt
    ASSERT(cpu.last_result().ecall);
    return true;
}

TEST(prog_save_restore_state) {
    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x00100093,  // addi x1, x0, 1
        0x00200113,  // addi x2, x0, 2
        0x00300193,  // addi x3, x0, 3
        0x00400213,  // addi x4, x0, 4
        0x00100073,  // ebreak
    });

    cpu.run(2);

    auto state = cpu.save_state();
    ASSERT_EQ(state.pc, 8u);
    ASSERT_EQ(state.regs[1], 1u);
    ASSERT_EQ(state.regs[2], 2u);
    ASSERT_EQ(state.regs[3], 0u);  // Not yet executed when saved

    cpu.run(2);
    ASSERT_EQ(cpu.reg(3), 3u);
    ASSERT_EQ(cpu.reg(4), 4u);

    // Restore: registers and PC go back to the saved values.
    cpu.restore_state(state);
    ASSERT_EQ(cpu.pc(), 8u);
    ASSERT_EQ(cpu.reg(1), 1u);
    ASSERT_EQ(cpu.reg(2), 2u);
    // x3 and x4 are restored to their saved values (0), not their
    // pre-restore values.
    ASSERT_EQ(cpu.reg(3), 0u);
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(prog_trace_output) {
    auto cpu = make_cpu();
    load_program(cpu, 0, {
        0x00100093,  // addi x1, x0, 1
        0x00200113,  // addi x2, x0, 2
        0x00100073,  // ebreak
    });

    cpu.set_trace(true);

    // Capture stdout to verify trace output doesn't crash.
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    cpu.run(3);

    std::cout.rdbuf(old);

    std::string trace = buffer.str();
    ASSERT(trace.find("addi") != std::string::npos);
    return true;
}
