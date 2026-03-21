/**
 * @file test_executor.cpp
 * @brief Tests for instruction execution via the CPU interface.
 *
 * Sections:
 *   1 (line 87)  : ALU register-immediate and register-register operations
 *   2 (line 191) : Load/store at all widths, sign/zero extension
 *   3 (line 368) : Branch taken/not-taken for all six conditions
 *   4 (line 488) : Jumps (JAL, JALR, LSB clearing)
 *   5 (line 623) : Upper immediate (LUI, AUIPC)
 *   6 (line 665) : x0 hardwired to zero
 *   7 (line 686) : System instructions (FENCE, ECALL, EBREAK)
 *   8 (line 698) : CPU lifecycle (halt, reset, save/restore)
 *   9 (line 752) : Execution statistics
 *  10 (line 808) : Edge cases (overflow, shift boundaries, store truncation)
 */

#include "core/cpu.hpp"

#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace riscv;

// ── Shared test infrastructure ─────────────────────────────────────────

struct TestCase {
    std::string             name;
    std::function<bool()>   func;
};
extern std::vector<TestCase> g_tests;

#define TEST(name)                                                            \
    bool test_##name();                                                       \
    static bool reg_##name = (g_tests.push_back({#name, test_##name}), true); \
    bool test_##name()
 
#define ASSERT(cond)                                                        \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "  FAILED: " << #cond << "\n"                      \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define ASSERT_EQ(a, b)                                                      \
    do {                                                                     \
        auto actual_   = (a);                                                \
        auto expected_ = (b);                                                \
        if (actual_ != expected_) {                                          \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"          \
                      << "    got: " << static_cast<std::int64_t>(actual_)   \
                      << " != "      << static_cast<std::int64_t>(expected_) \
                      << "\n"                                                \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";   \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define ASSERT_HEX_EQ(a, b)                                                 \
    do {                                                                    \
        auto actual_   = (a);                                               \
        auto expected_ = (b);                                               \
        if (actual_ != expected_) {                                         \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"         \
                      << std::format("    got: 0x{:x} != 0x{:x}\n",         \
                            static_cast<std::uint64_t>(actual_),            \
                            static_cast<std::uint64_t>(expected_))          \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                   \
        }                                                                   \
    } while (0)

/* Helper: create a CPU backed by 64 KiB of flat RAM at address 0. */
static CPU make_cpu()
{
    auto mem = std::make_shared<FlatMemory>(0x0, 0x10000);
    return CPU(mem);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. ALU — Register-Immediate
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_addi) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x02a00093);  // addi x1, x0, 42
    cpu.step();
    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT_EQ(cpu.pc(), 4u);
    return true;
}

TEST(exec_addi_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 100);
    cpu.load_instruction(0, 0xfe208113);  // addi x2, x1, -30
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 70u);
    return true;
}

TEST(exec_andi) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xABCD);
    cpu.load_instruction(0, 0x0ff0f113);  // andi x2, x1, 0xFF
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xCDu);
    return true;
}

TEST(exec_ori) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xF000);
    cpu.load_instruction(0, 0x00f0e113);  // ori x2, x1, 0x0F
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xF00Fu);
    return true;
}

TEST(exec_xori) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xFFFF);
    cpu.load_instruction(0, 0x0ff0c113);  // xori x2, x1, 0xFF
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFF00u);
    return true;
}

TEST(exec_slti_true) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-10));
    cpu.load_instruction(0, 0x0050a113);  // slti x2, x1, 5
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 1u); // -10 < 5 (signed)
    return true;
}

TEST(exec_slti_false) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.load_instruction(0, 0x0050a113);  // slti x2, x1, 5
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 0u);  // 10 not < 5
    return true;
}

TEST(exec_sltiu) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.load_instruction(0, 0x0140b113);  // sltiu x2, x1, 20
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_slli) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 1);
    cpu.load_instruction(0, 0x00809113);  // slli x2, x1, 8
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 256u);
    return true;
}

TEST(exec_srli) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 256);
    cpu.load_instruction(0, 0x0040d113);  // srli x2, x1, 4
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 16u);
    return true;
}

TEST(exec_srai) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-256));
    cpu.load_instruction(0, 0x4040d113);  // srai x2, x1, 4
    cpu.step();
    ASSERT_EQ(static_cast<i32>(cpu.reg(2)), -16);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. ALU — Register-Register
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_add) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, 20);
    cpu.load_instruction(0, 0x002081b3);  // add x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 30u);
    return true;
}

TEST(exec_add_overflow) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xFFFFFFFF);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, 0x002081b3);  // add x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 0u);  // Wraps to 0
    return true;
}

TEST(exec_sub) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 50);
    cpu.set_reg(2, 20);
    cpu.load_instruction(0, 0x402081b3);  // sub x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 30u);
    return true;
}

TEST(exec_sub_underflow) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, 0x402081b3);  // sub x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);  // Wraps
    return true;
}

TEST(exec_and) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xFF00);
    cpu.set_reg(2, 0x0FF0);
    cpu.load_instruction(0, 0x0020f1b3);  // and x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x0F00u);
    return true;
}

TEST(exec_or) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xFF00);
    cpu.set_reg(2, 0x00FF);
    cpu.load_instruction(0, 0x0020e1b3);  // or x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFu);
    return true;
}

TEST(exec_xor) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xF0F0);
    cpu.set_reg(2, 0xFF00);
    cpu.load_instruction(0, 0x0020c1b3);  // xor x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x0FF0u);
    return true;
}

TEST(exec_sll) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, 0x002091b3);  // sll x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 16u);
    return true;
}

TEST(exec_sll_shift_zero) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xDEADBEEF);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, 0x002091b3);  // sll x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xDEADBEEFu);  // No shift
    return true;
}

TEST(exec_sll_shift_31) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 31);
    cpu.load_instruction(0, 0x002091b3);  // sll x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x80000000u);
    return true;
}

TEST(exec_sll_uses_low_5_bits) {
    // Shift amount should be masked to low 5 bits (§2.4).
    // rs2 = 32 -> effective shift = 0.
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x12345678);
    cpu.set_reg(2, 32);
    cpu.load_instruction(0, 0x002091b3);  // sll x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x12345678u);  // shift by 0
    return true;
}

TEST(exec_srl) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x80);
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, 0x0020d1b3);  // srl x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 8u);
    return true;
}

TEST(exec_sra_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-128));  // 0xFFFFFF80
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, 0x4020d1b3);  // sra x3, x1, x2
    cpu.step();
    ASSERT_EQ(static_cast<i32>(cpu.reg(3)), -8);
    return true;
}

TEST(exec_sra_positive) {
    // SRA on a positive value should zero-fill the upper bits (same as SRL).
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x7FFF0000);  // Positive (bit 31 = 0)
    cpu.set_reg(2, 16);
    cpu.load_instruction(0, 0x4020d1b3);  // sra x3, x1, x2
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x00007FFFu);  // Zero-filled, not sign-extended
    return true;
}

TEST(exec_slt_true) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, 0x0020a1b3);  // slt x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_slt_false) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, static_cast<u32>(-5));
    cpu.load_instruction(0, 0x0020a1b3);  // slt x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(exec_sltu) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, static_cast<u32>(-5));  // 0xFFFFFFFB
    cpu.load_instruction(0, 0x0020b1b3);  // sltu x3, x1, x2
    cpu.step();
    ASSERT_EQ(cpu.reg(3), 1u);  // 10 < 0xFFFFFFFB unsigned
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Load / Store
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_lw) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x100, 0xDEADBEEF);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x0000a103);  // lw x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEADBEEFu);
    return true;
}

TEST(exec_lw_offset) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x108, 0x12345678);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x0080a103);  // lw x2, 8(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x12345678u);
    return true;
}

TEST(exec_lh_sign_extend) {
    auto cpu = make_cpu();
    cpu.memory().write16(0x100, 0xFF00);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x00009103);  // lh x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFFFF00u);  // Sign-extended
    return true;
}

TEST(exec_lhu_zero_extend) {
    auto cpu = make_cpu();
    cpu.memory().write16(0x100, 0xFF00);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x0000d103);  // lhu x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x0000FF00u);
    return true;
}

TEST(exec_lb_sign_extend) {
    auto cpu = make_cpu();
    cpu.memory().write8(0x100, 0x80);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x00008103);  // lb x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFFFF80u);
    return true;
}

TEST(exec_lbu_zero_extend) {
    auto cpu = make_cpu();
    cpu.memory().write8(0x100, 0x80);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, 0x0000c103);  // lbu x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x00000080u);
    return true;
}

TEST(exec_load_to_x0) {
    // Loading to x0 must not change it (x0 is hardwired to zero).
    auto cpu = make_cpu();
    cpu.memory().write32(0x100, 0xDEADBEEF);
    cpu.set_reg(1, 0x100);
    // lw x0, 0(x1) -> opcode with rd=0
    cpu.load_instruction(0, 0x0000a003);
    cpu.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_sw) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xCAFEBABE);
    cpu.load_instruction(0, 0x0020a023);  // sw x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xCAFEBABEu);
    return true;
}

TEST(exec_sh) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x1234);
    cpu.load_instruction(0, 0x00209223);  // sh x2, 4(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.memory().read16(0x104).value, 0x1234u);
    return true;
}

TEST(exec_sb) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xAB);
    cpu.load_instruction(0, 0x00208123);  // sb x2, 2(x1)
    cpu.step();
    ASSERT_EQ(cpu.memory().read8(0x102).value, 0xABu);
    return true;
}

TEST(exec_sb_truncates) {
    // SB should write only the low byte, not the full register.
    auto cpu = make_cpu();
    cpu.memory().write32(0x100, 0x00000000);  // Clear area
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xDEADBEEF);
    cpu.load_instruction(0, 0x00208023);  // sb x2, 0(x1)
    cpu.step();
    // Only byte at 0x100 should be 0xEF; adjacent bytes must be untouched.
    ASSERT_HEX_EQ(cpu.memory().read8(0x100).value, 0xEFu);
    ASSERT_HEX_EQ(cpu.memory().read8(0x101).value, 0x00u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Branches
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_beq_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 42);
    cpu.load_instruction(0, 0x00208863);  // beq x1, x2, 16
    cpu.step();
    ASSERT_EQ(cpu.pc(), 16u);
    return true;
}

TEST(exec_beq_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 43);
    cpu.load_instruction(0, 0x00208863);  // beq x1, x2, 16
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);
    return true;
}

TEST(exec_bne_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, 20);
    cpu.load_instruction(0, 0x00209463);  // bne x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 8u);
    return true;
}

TEST(exec_bne_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, 0x00209463);  // bne x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);
    return true;
}

TEST(exec_blt_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, 0x0020c663);  // blt x1, x2, 12
    cpu.step();
    ASSERT_EQ(cpu.pc(), 12u);
    return true;
}

TEST(exec_blt_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-5));
    cpu.load_instruction(0, 0x0020c663);  // blt x1, x2, 12
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);  // 5 not < -5 (signed)
    return true;
}

TEST(exec_bge_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 10);
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, 0x0020da63);  // bge x1, x2, 20
    cpu.step();
    ASSERT_EQ(cpu.pc(), 20u);
    return true;
}

TEST(exec_bge_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-10));
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, 0x0020da63);  // bge x1, x2, 20
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);  // -10 not >= 10
    return true;
}

TEST(exec_bltu_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, 0x0020e463);  // bltu x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 8u);
    return true;
}

TEST(exec_bltu_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, 0x0020e463);  // bltu x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);  // 0xFFFFFFFF not < 5
    return true;
}

TEST(exec_bgeu_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, 0x0020f463);  // bgeu x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 8u);
    return true;
}

TEST(exec_bgeu_not_taken) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, 0x0020f463);  // bgeu x1, x2, 8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 4u);  // 5 not >= 0xFFFFFFFF
    return true;
}

TEST(exec_branch_backward) {
    auto cpu = make_cpu();
    cpu.set_pc(0x100);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0x100, 0xfe208ce3);  // beq x1, x2, -8
    cpu.step();
    ASSERT_EQ(cpu.pc(), 0xF8u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Jumps
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_jal) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x064000ef);  // jal x1, 100
    cpu.step();
    ASSERT_EQ(cpu.reg(1), 4u);  // Return address = PC + 4
    ASSERT_EQ(cpu.pc(), 100u);
    return true;
}

TEST(exec_jal_backward) {
    auto cpu = make_cpu();
    cpu.set_pc(0x100);
    cpu.load_instruction(0x100, 0xfedff0ef);  // jal x1, -20
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x104u);
    ASSERT_HEX_EQ(cpu.pc(), 0xECu);
    return true;
}

TEST(exec_jalr) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x200);
    cpu.load_instruction(0, 0x00808167);  // jalr x2, 8(x1)
    cpu.step();
    ASSERT_EQ(cpu.reg(2), 4u);
    ASSERT_HEX_EQ(cpu.pc(), 0x208u);
    return true;
}

TEST(exec_jalr_clears_lsb) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x201);  // Odd
    cpu.load_instruction(0, 0x00008167);  // jalr x2, 0(x1)
    cpu.step();
    ASSERT_HEX_EQ(cpu.pc(), 0x200u);  // Bit 0 cleared per spec
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Upper Immediate
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_lui) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x123450b7);  // lui x1, 0x12345
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x12345000u);
    return true;
}

TEST(exec_auipc) {
    auto cpu = make_cpu();
    cpu.set_pc(0x100);
    cpu.load_instruction(0x100, 0x00010097);  // auipc x1, 0x10
    cpu.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x100u + 0x10000u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. x0 hardwired to zero
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_x0_always_zero) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x06400013);  // addi x0, x0, 100
    cpu.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. System Instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_fence_is_nop) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.load_instruction(0, 0x0ff0000f);  // fence iorw, iorw
    cpu.step();
    // FENCE should just advance PC; no register changes.
    ASSERT_EQ(cpu.pc(), 4u);
    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT(!cpu.halted());
    return true;
}

TEST(exec_ebreak_halts) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x00100073);  // ebreak
    bool ran = cpu.step();
    ASSERT(!ran);          // step() returns false
    ASSERT(cpu.halted());  // CPU is halted
    return true;
}

TEST(exec_ecall_does_not_halt) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x00000073);  // ecall
    cpu.load_instruction(4, 0x02a00093);  // addi x1, x0, 42
    bool ran = cpu.step();
    ASSERT(ran);            // ECALL does NOT halt
    ASSERT(!cpu.halted());
    // The ecall flag should be set on last_result.
    ASSERT(cpu.last_result().ecall);
    // Continue execution — next instruction should work.
    cpu.step();
    ASSERT_EQ(cpu.reg(1), 42u);
    return true;
}

TEST(exec_run_until_ecall) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x02a00093);  // addi x1, x0, 42
    cpu.load_instruction(4, 0x00108093);  // addi x1, x1, 1
    cpu.load_instruction(8, 0x00000073);  // ecall
    cpu.load_instruction(12, 0x00108093); // addi x1, x1, 1 (should not run)
 
    u64 count = cpu.run_until_ecall();
    ASSERT_EQ(count, 3u);           // 3 instructions executed (including ecall)
    ASSERT_EQ(cpu.reg(1), 43u);     // 42 + 1
    ASSERT(!cpu.halted());
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  9. CPU Lifecycle — halt, reset, save/restore
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_invalid_instruction_halts) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0xFFFFFFFF);  // All-ones = invalid
    bool ran = cpu.step();
    ASSERT(!ran);
    ASSERT(cpu.halted());
    return true;
}

TEST(exec_fetch_fault_halts) {
    // Memory is 0x0–0xFFFF.  Set PC past the end.
    auto cpu = make_cpu();
    cpu.set_pc(0x20000);
    bool ran = cpu.step();
    ASSERT(!ran);
    ASSERT(cpu.halted());
    return true;
}

TEST(exec_reset_clears_halt) {
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0x00100073);  // ebreak -> halt
    cpu.step();
    ASSERT(cpu.halted());

    cpu.reset();
    ASSERT(!cpu.halted());
    ASSERT_EQ(cpu.pc(), 0u);
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_save_restore_state) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 100);
    cpu.set_reg(2, 200);
    cpu.set_pc(0x400);
 
    auto state = cpu.save_state();

    // Mutate CPU.
    cpu.set_reg(1, 999);
    cpu.set_pc(0x800);

    cpu.restore_state(state);
    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.reg(2), 200u);
    ASSERT_EQ(cpu.pc(), 0x400u);
    ASSERT_EQ(cpu.reg(0), 0u);  // x0 must still be 0
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  10. Statistics
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(exec_stats_basic) {
    auto cpu = make_cpu();
    cpu.load_instruction(0,  0x02a00093);  // addi x1, x0, 42
    cpu.load_instruction(4,  0x00108113);  // addi x2, x1, 1
    cpu.load_instruction(8,  0x00210193);  // addi x3, x2, 2
    cpu.load_instruction(12, 0x00318213);  // addi x4, x3, 3
    cpu.load_instruction(16, 0x00420293);  // addi x5, x4, 4
 
    cpu.run(5);
    ASSERT_EQ(cpu.stats().instructions, 5u);
    return true;
}

TEST(exec_stats_branches) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, 0x00208463);  // beq x1, x2, 8 (taken)
    cpu.step();
 
    ASSERT_EQ(cpu.stats().branches, 1u);
    ASSERT_EQ(cpu.stats().branches_taken, 1u);
    return true;
}

TEST(exec_stats_loads_stores_jumps) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0);
    cpu.set_reg(10, 0x200);
 
    cpu.load_instruction(0,  0x02a00093);  // addi x1, x0, 42
    cpu.load_instruction(4,  0x00152023);  // sw x1, 0(x10)
    cpu.load_instruction(8,  0x00052103);  // lw x2, 0(x10)
    cpu.load_instruction(12, 0x064000ef);  // jal x1, 100
 
    cpu.run(4);
    ASSERT_EQ(cpu.stats().loads, 1u);
    ASSERT_EQ(cpu.stats().stores, 1u);
    ASSERT_EQ(cpu.stats().jumps, 1u);
    ASSERT_EQ(cpu.stats().instructions, 4u);
    return true;
}
