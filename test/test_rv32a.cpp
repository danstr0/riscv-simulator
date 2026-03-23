/**
 * @file test_rv32a.cpp
 * @brief Tests for the RV32A atomic memory operations extension.
 *
 * Sections:
 *   1 (line 106) : Decoder - all 11 A instructions decode correctly 
 *   2 (line 156) : LR.W/SC.w - success path, failure paths, reservation clearing
 *   3 (line 253) : AMO operations - each read-modify-write variant
 *   4 (line 332) : AMO - signed/unsigned min/max comparisons
 *   5 (line 397) : CAS pattern - compare-and-swap loop (always succeeds single-core)
 *   6 (line 435) : Pipeline - atomics through the pipelined CPU
 */

#include "core/cpu.hpp"
#include "core/pipeline.hpp"
 
#include <cstdint>
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

// ── Helpers ────────────────────────────────────────────────────────────

static CPU make_cpu() {
    return CPU(std::make_shared<FlatMemory>(0, 0x10000));
}

/* Encode an AMO instruction:
 *   funct5[31:27] | aq=0 | rl=0 | rs2[24:20] | rs1[19:15] |
 *   funct3=010 | rd[11:7] | opcode=0101111
 */
static constexpr u32 encode_amo(u32 funct5, u32 rd, u32 rs1, u32 rs2) {
    return (funct5 << 27) | (rs2 << 20) | (rs1 << 15)
         | (0b010u << 12) | (rd << 7) | 0b0101111u;
}

static constexpr u32 LR_W(u32 rd, u32 rs1)             { return encode_amo(0b00010, rd, rs1, 0); }
static constexpr u32 SC_W(u32 rd, u32 rs1, u32 rs2)    { return encode_amo(0b00011, rd, rs1, rs2); }
static constexpr u32 AMOSWAP(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b00001, rd, rs1, rs2); }
static constexpr u32 AMOADD(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b00000, rd, rs1, rs2); }
static constexpr u32 AMOXOR(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b00100, rd, rs1, rs2); }
static constexpr u32 AMOAND(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b01100, rd, rs1, rs2); }
static constexpr u32 AMOOR(u32 rd, u32 rs1, u32 rs2)   { return encode_amo(0b01000, rd, rs1, rs2); }
static constexpr u32 AMOMIN(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b10000, rd, rs1, rs2); }
static constexpr u32 AMOMAX(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b10100, rd, rs1, rs2); }
static constexpr u32 AMOMINU(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b11000, rd, rs1, rs2); }
static constexpr u32 AMOMAXU(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b11100, rd, rs1, rs2); }

static constexpr u32 EBREAK = 0x00100073;

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Decoder
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_decode_lr_w) {
    auto inst = Decoder::decode(LR_W(3, 1));
    ASSERT_EQ(inst.op, Op::LR_W);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 0);  // LR.W must have rs2=0
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_sc_w) {
    auto inst = Decoder::decode(SC_W(3, 1, 2));
    ASSERT_EQ(inst.op, Op::SC_W);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    return true;
}

TEST(a_decode_amoswap) { ASSERT_EQ(Decoder::decode(AMOSWAP(3,1,2)).op, Op::AMOSWAP_W); return true; }
TEST(a_decode_amoadd)  { ASSERT_EQ(Decoder::decode(AMOADD(3,1,2)).op,  Op::AMOADD_W);  return true; }
TEST(a_decode_amoxor)  { ASSERT_EQ(Decoder::decode(AMOXOR(3,1,2)).op,  Op::AMOXOR_W);  return true; }
TEST(a_decode_amoand)  { ASSERT_EQ(Decoder::decode(AMOAND(3,1,2)).op,  Op::AMOAND_W);  return true; }
TEST(a_decode_amoor)   { ASSERT_EQ(Decoder::decode(AMOOR(3,1,2)).op,   Op::AMOOR_W);   return true; }
TEST(a_decode_amomin)  { ASSERT_EQ(Decoder::decode(AMOMIN(3,1,2)).op,  Op::AMOMIN_W);  return true; }
TEST(a_decode_amomax)  { ASSERT_EQ(Decoder::decode(AMOMAX(3,1,2)).op,  Op::AMOMAX_W);  return true; }
TEST(a_decode_amominu) { ASSERT_EQ(Decoder::decode(AMOMINU(3,1,2)).op, Op::AMOMINU_W); return true; }
TEST(a_decode_amomaxu) { ASSERT_EQ(Decoder::decode(AMOMAXU(3,1,2)).op, Op::AMOMAXU_W); return true; }

TEST(a_decode_lr_w_invalid_rs2) {
    // LR.W with rs2 != 0 is invalid per spec
    u32 bad_lr = encode_amo(0b00010, 3, 1, 5);  // rs2=5
    auto inst = Decoder::decode(bad_lr);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(a_decode_invalid_funct3) {
    // AMO opcode but funct3 != 010 (not .W) should be invalid
    u32 bad = (0b00000u << 27) | (2u << 20) | (1u << 15)
            | (0b011u << 12) | (3u << 7) | 0b0101111u; /* funct3 = 011 */
    auto inst = Decoder::decode(bad);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. LR.W / SC.W
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_lr_sc_success) {
    // LR.W loads and sets reservation; SC.W to same address succeeds
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xAAAAAAAA);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0xBBBBBBBB);

    cpu.load_instruction(0, LR_W(1, 10));       // x1 = mem[0x200], reserve 0x200
    cpu.load_instruction(4, SC_W(2, 10, 11));   // mem[0x200] = x11, x2 = 0 (success)
    cpu.load_instruction(8, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xAAAAAAAAu);  // LR.W loaded old value
    ASSERT_EQ(cpu.reg(2), 0u);               // SC.W succeeded
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xBBBBBBBBu);  // Memory updated
    return true;
}

TEST(a_sc_without_lr_fails) {
    // SC.W without a prior LR.W should fail
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x11111111);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x22222222);

    cpu.load_instruction(0, SC_W(2, 10, 11));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT(cpu.reg(2) != 0u);  // SC.W failed
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0x11111111u);  // Memory unchanged
    return true;
}

TEST(a_sc_wrong_address_fails) {
    // LR.W on one address, SC.W on a different address -> fail
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xAAAA);
    cpu.memory().write32(0x300, 0xBBBB);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0xCCCC);

    cpu.load_instruction(0, LR_W(1, 10));      // Reserve 0x200
    cpu.load_instruction(4, SC_W(2, 11, 12));  // SC to 0x300 -> fail
    cpu.load_instruction(8, EBREAK);
    cpu.run(10);

    ASSERT(cpu.reg(2) != 0u);  // Failed
    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0xBBBBu);  // Unchanged
    return true;
}

TEST(a_sc_double_fails) {
    // Two SC.W after one LR.W — second should fail
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 200);
    cpu.set_reg(12, 300);

    cpu.load_instruction(0,  LR_W(1, 10));      // Reserve
    cpu.load_instruction(4,  SC_W(2, 10, 11));  // First SC -> success
    cpu.load_instruction(8,  SC_W(3, 10, 12));  // Second SC -> fail (reservation cleared)
    cpu.load_instruction(12, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(2), 0u);
    ASSERT(cpu.reg(3) != 0u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u);  // First write stuck
    return true;
}

TEST(a_store_clears_reservation) {
    // A regular SW between LR.W and SC.W should clear the reservation
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 999);
    cpu.set_reg(12, 42);

    cpu.load_instruction(0,  LR_W(1, 10));      // Reserve 0x200
    cpu.load_instruction(4,  0x00B52023);       // sw x11, 0(x10) -> clears reservation
    cpu.load_instruction(8,  SC_W(3, 10, 12));  // SC -> fail
    cpu.load_instruction(12, EBREAK);
    cpu.run(10);

    ASSERT(cpu.reg(3) != 0u);  // SC failed
    // Memory has the SW value, not the SC value
    ASSERT_EQ(cpu.memory().read32(0x200).value, 999u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. AMO operations
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_amoswap) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 200);

    cpu.load_instruction(0, AMOSWAP(1, 10, 11));  // x1 = old(100), mem = 200
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u);
    return true;
}

TEST(a_amoadd) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 50);

    cpu.load_instruction(0, AMOADD(1, 10, 11));  // x1 = old(100), mem = 150
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 150u);
    return true;
}

TEST(a_amoxor) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00FF00);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F0F0F);

    cpu.load_instruction(0, AMOXOR(1, 10, 11));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xFF00FF00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xF00FF00Fu);
    return true;
}

TEST(a_amoand) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00FF00);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F0F0F);

    cpu.load_instruction(0, AMOAND(1, 10, 11));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xFF00FF00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0x0F000F00u);
    return true;
}

TEST(a_amoor) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xF000F000);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F0F0F);

    cpu.load_instruction(0, AMOOR(1, 10, 11));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xF000F000u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xFF0FFF0Fu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. AMO min/max
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_amomin_signed) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMIN(1, 10, 11));  // min(-10, 5) = -10
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(1), static_cast<u32>(-10));  // Old value returned
    ASSERT_EQ(cpu.memory().read32(0x200).value, static_cast<u32>(-10));  // -10 < 5, kept
    return true;
}

TEST(a_amomax_signed) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMAX(1, 10, 11));  // max(-10, 5) = 5
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(1), static_cast<u32>(-10));  // Old value returned
    ASSERT_EQ(cpu.memory().read32(0x200).value, 5u);  // 5 > -10
    return true;
}

TEST(a_amominu_unsigned) {
    // 0xFFFF'FFF6 is -10 signed but a large number unsigned
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFFFFFFF6);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMINU(1, 10, 11));  // unsigned min(0xFFFFFFF6, 5) = 5
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xFFFFFFF6u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 5u);
    return true;
}

TEST(a_amomaxu_unsigned) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFFFFFFF6);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMAXU(1, 10, 11));  // unsigned max(0xFFFFFFF6, 5) = 0xFFFFFFF6
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_HEX_EQ(cpu.reg(1), 0xFFFFFFF6u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xFFFFFFF6u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. CAS (compare-and-swap) pattern
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_cas_pattern) {
    // CAS loop: atomically set mem[0x200] from 100 to 200
    //
    // 0x00 retry: lr.w x13, (x10)
    // 0x04        bne x13, x11, 16      -> fail
    // 0x08        sc.w x14, x12, (x10)
    // 0x0C        bne x14, x0, -12      -> retry
    // 0x10        jal x0, 8
    // 0x14 fail:  addi x15, x0, 1
    // 0x18 done:  ebreak

    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 100);
    cpu.set_reg(12, 200);

    cpu.load_instruction(0x00, LR_W(13, 10));
    cpu.load_instruction(0x04, 0x00b69863);
    cpu.load_instruction(0x08, SC_W(14, 10, 12));
    cpu.load_instruction(0x0C, 0xfe071ae3);
    cpu.load_instruction(0x10, 0x0080006f);
    cpu.load_instruction(0x14, 0x00100793);
    cpu.load_instruction(0x18, EBREAK);

    cpu.run(100);

    // In single-thread, CAS always succeeds on first try
    ASSERT_EQ(cpu.reg(13), 100u);  // LR loaded the expected value
    ASSERT_EQ(cpu.reg(14), 0u);    // SC succeeded
    ASSERT_EQ(cpu.reg(15), 0u);    // Never went to fail path
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Pipeline
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(a_pipe_lr_sc) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x200, 42);
    PipelinedCPU cpu(mem);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 99);

    cpu.load_instruction(0, LR_W(1, 10));
    cpu.load_instruction(4, SC_W(2, 10, 11));
    cpu.load_instruction(8, EBREAK);

    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }

    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT_EQ(cpu.reg(2), 0u);  // SC success
    ASSERT_EQ(mem->read32(0x200).value, 99u);
    return true;
}

TEST(a_pipe_amoadd) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x200, 100);
    PipelinedCPU cpu(mem);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 50);

    cpu.load_instruction(0, AMOADD(1, 10, 11));
    cpu.load_instruction(4, EBREAK);

    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }

    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(mem->read32(0x200).value, 150u);
    return true;
}

TEST(a_pipe_amoswap) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x200, 0xAAAA);
    PipelinedCPU cpu(mem);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0xBBBB);

    cpu.load_instruction(0, AMOSWAP(1, 10, 11));
    cpu.load_instruction(4, EBREAK);

    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }

    ASSERT_HEX_EQ(cpu.reg(1), 0xAAAAu);
    ASSERT_HEX_EQ(mem->read32(0x200).value, 0xBBBBu);
    return true;
}
