/**
 * @file test_rv32m.cpp
 * @brief Tests for the RV32M integer multiply/divide extension.
 *
 * Sections:
 *   1 (line 106) : Decoder - all 8 M instructions decode correctly
 *   2 (line 128) : MUL - basic, overflow, negative
 *   3 (line 167) : MULH[[S]U] - upper-half products, mixed signs
 *   4 (line 234) : DIV[U] - basic, division by zero, signed overflow
 *   5 (line 308) : REM[U] - basic, division by zero, signed overflow
 *   6 (line 393) : Pipeline correctness
 *
 * Encoding reference (all R-type, opcode = 0110011, funct7 = 0000001):
 *   MUL     funct3=000  MULH    funct3=001
 *   MULHSU  funct3=010  MULHU   funct3=011 
 *   DIV     funct3=100  DIVU    funct3=101
 *   REM     funct3=110  REMU    funct3=111
 */

#include "core/cpu.hpp"
#include "core/pipeline.hpp"

#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
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

// ── Helpers ────────────────────────────────────────────────────────────

static CPU make_cpu() {
    return CPU(std::make_shared<FlatMemory>(0, 0x10000));
}

// Encode an R-type M instruction:
//   funct7=0000001 | rs2 | rs1 | funct3 | rd | opcode=0110011
static constexpr u32 encode_m(u32 funct3, u32 rd, u32 rs1, u32 rs2) {
    return (0b0000001u << 25) | (rs2 << 20) | (rs1 << 15)
         | (funct3 << 12) | (rd << 7) | 0b0110011u;
}

static constexpr u32 MUL(u32 rd, u32 rs1, u32 rs2)    { return encode_m(0b000, rd, rs1, rs2); }
static constexpr u32 MULH(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b001, rd, rs1, rs2); }
static constexpr u32 MULHSU(u32 rd, u32 rs1, u32 rs2) { return encode_m(0b010, rd, rs1, rs2); }
static constexpr u32 MULHU(u32 rd, u32 rs1, u32 rs2)  { return encode_m(0b011, rd, rs1, rs2); }
static constexpr u32 DIV_(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b100, rd, rs1, rs2); }
static constexpr u32 DIVU(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b101, rd, rs1, rs2); }
static constexpr u32 REM_(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b110, rd, rs1, rs2); }
static constexpr u32 REMU(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b111, rd, rs1, rs2); }

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Decoder
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_decode_mul) {
    auto inst = Decoder::decode(MUL(3, 1, 2));
    ASSERT_EQ(inst.op, Op::MUL);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_mulh)   { ASSERT_EQ(Decoder::decode(MULH(3,1,2)).op,   Op::MULH);   return true; }
TEST(m_decode_mulhsu) { ASSERT_EQ(Decoder::decode(MULHSU(3,1,2)).op, Op::MULHSU); return true; }
TEST(m_decode_mulhu)  { ASSERT_EQ(Decoder::decode(MULHU(3,1,2)).op,  Op::MULHU);  return true; }
TEST(m_decode_div)    { ASSERT_EQ(Decoder::decode(DIV_(3,1,2)).op,   Op::DIV);    return true; }
TEST(m_decode_divu)   { ASSERT_EQ(Decoder::decode(DIVU(3,1,2)).op,   Op::DIVU);   return true; }
TEST(m_decode_rem)    { ASSERT_EQ(Decoder::decode(REM_(3,1,2)).op,   Op::REM);    return true; }
TEST(m_decode_remu)   { ASSERT_EQ(Decoder::decode(REMU(3,1,2)).op,   Op::REMU);   return true; }

/* ═══════════════════════════════════════════════════════════════════════
 *  2. MUL
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_mul_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 7);
    cpu.set_reg(2, 6);
    cpu.load_instruction(0, MUL(3, 1, 2));  // x3 = 7 * 6
    cpu.load_instruction(4, 0x00100073);    // ebreak
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_mul_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-3));
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, MUL(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // Lower 32 bits of -21 = 0xFFFF'FFEB.
    ASSERT_HEX_EQ(cpu.reg(3), static_cast<u32>(-21));
    return true;
}

TEST(m_mul_overflow) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x10000);
    cpu.set_reg(2, 0x10000);
    cpu.load_instruction(0, MUL(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // Lower 32 bits of 0x1'0000'0000 = 0
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. MULH[[S]U]
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_mulhu_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 0x10000);
    cpu.set_reg(2, 0x10000);
    cpu.load_instruction(0, MULHU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // Upper 32 bits of 0x1'0000'0000 = 1
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(m_mulh_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, MULH(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // 1 as signed 64-bit: 0x0000'0000'0000'0001 - Upper 32 bits = 0 
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_mulh_large_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULH(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // -2 as signed 64-bit: 0xFFFF'FFFF'FFFF'FFFE - Upper 32 bits = 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);
    return true;
}

TEST(m_mulhsu_mixed) {
    // signed(-1) * unsigned(2).
    // signed(-1) = 0xFFFFFFFF interpreted as -1.
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULHSU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // -2 as signed 64-bit: 0xFFFF'FFFF'FFFF'FFFE - Upper 32 bits = 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);
    return true;
}

TEST(m_mulhu_large) {
    // unsigned(0xFFFFFFFF) * unsigned(0xFFFFFFFF)
    auto cpu = make_cpu();
    cpu.set_reg(1, 0xFFFFFFFF);
    cpu.set_reg(2, 0xFFFFFFFF);
    cpu.load_instruction(0, MULHU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // Upper 32 bits of 0xFFFF'FFFE'0000'0001 = 0xFFFF'FFFE
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFEu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. DIV[U]
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_div_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, DIV_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 6u);
    return true;
}

TEST(m_div_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-20));
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, DIV_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // -20 / 3 = -6 (truncated toward zero)
    ASSERT_EQ(static_cast<i32>(cpu.reg(3)), -6);
    return true;
}

TEST(m_div_by_zero) {
    // Spec: DIV by zero -> -1 (all ones).
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, DIV_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);
    return true;
}

TEST(m_div_overflow) {
    // Spec: INT_MIN / -1 -> INT_MIN (signed overflow wraps).
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(std::numeric_limits<i32>::min()));  // 0x8000'0000
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, DIV_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_HEX_EQ(cpu.reg(3), 0x80000000u);
    return true;
}

TEST(m_divu_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 100);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, DIVU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 14u);  // 100 / 7 = 14
    return true;
}

TEST(m_divu_by_zero) {
    // Spec: DIVU by zero -> 0xFFFF'FFFF.
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, DIVU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. REM[U]
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_rem_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 20);
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, REM_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 2u);  // 20 % 3 = 2
    return true;
}

TEST(m_rem_negative) {
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(-20));
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, REM_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // -20 % 3 = -2 (sign follows dividend per C99/RISC-V spec)
    ASSERT_EQ(static_cast<i32>(cpu.reg(3)), -2);
    return true;
}

TEST(m_rem_negative_divisor) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 20);
    cpu.set_reg(2, static_cast<u32>(-3));
    cpu.load_instruction(0, REM_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    // 20 % -3 = 2 (remainder sign follows the dividend, not the divisor)
    ASSERT_EQ(static_cast<i32>(cpu.reg(3)), 2);
    return true;
}

TEST(m_rem_by_zero) {
    // Spec: REM by zero -> dividend.
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, REM_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_rem_overflow) {
    // Spec: INT_MIN % -1 -> 0.
    auto cpu = make_cpu();
    cpu.set_reg(1, static_cast<u32>(std::numeric_limits<i32>::min()));
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, REM_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_remu_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 100);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, REMU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 2u);  // 100 % 7 = 2
    return true;
}

TEST(m_remu_by_zero) {
    // Spec: REMU by zero -> dividend.
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, REMU(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cpu.run(10);
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Pipeline correctness
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(m_pipe_mul) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 7);
    cpu.set_reg(2, 6);
    cpu.load_instruction(0, MUL(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_pipe_div_by_zero) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 99);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, DIV_(3, 1, 2));
    cpu.load_instruction(4, 0x00100073);
    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFFFFFu);
    return true;
}

TEST(m_pipe_mul_chain) {
    // Test forwarding with M instructions: mul x3, x1, x2 -> add x4, x3, x1
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, MUL(3, 1, 2)); // x3 = 50
    cpu.load_instruction(4, 0x00118233);          // add x4, x3, x1  → x4 = 50 + 5 = 55
    cpu.load_instruction(8, 0x00100073);
    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }
    ASSERT_EQ(cpu.reg(3), 50u);
    ASSERT_EQ(cpu.reg(4), 55u);
    return true;
}

TEST(m_pipe_divrem_program) {
    // Compute quotient and remainder: 47 / 5 = 9 rem 2.
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 47);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, DIV_(3, 1, 2));  // x3 = 47 / 5 = 9
    cpu.load_instruction(4, REM_(4, 1, 2));  // x4 = 47 % 5 = 2
    cpu.load_instruction(8, 0x00100073);
    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }
    ASSERT_EQ(cpu.reg(3), 9u);
    ASSERT_EQ(cpu.reg(4), 2u);
    return true;
}
