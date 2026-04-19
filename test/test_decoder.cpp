/**
 * @file test_decoder.cpp
 * @brief Decoder tests covering all RV32I instruction encodings.
 *
 * Sections:
 *   1 (line  79) : R-type (register–register)
 *   2 (line 183) : I-type (immediate ALU, loads, JALR)
 *   3 (line 441) : S-type (stores)
 *   4 (line 495) : B-type (branches)
 *   5 (line 579) : U-type (LUI, AUIPC)
 *   6 (line 621) : J-type (JAL)
 *   7 (line 661) : System (ECALL, EBREAK, FENCE)
 *   8 (line 689) : DecodedInst query helpers
 *   9 (line 756) : Edge cases and error paths
 */

#include "core/decoder.hpp"

#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace riscv;

// ── Shared test infrastructure ─────────────────────────────────────────

struct TestCase {
    std::string           name;
    std::function<bool()> func;
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

// ═══════════════════════════════════════════════════════════════════════
//  1. R-type instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_add) {
    auto inst = Decoder::decode(0x0031'00b3u); // add x1, x2, x3
    ASSERT_EQ(inst.op, Op::ADD);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_sub) {
    auto inst = Decoder::decode(0x4073'02b3u); // sub x5, x6, x7
    ASSERT_EQ(inst.op, Op::SUB);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.rs2, 7);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_and) {
    auto inst = Decoder::decode(0x00c5'f533u); // and x10, x11, x12
    ASSERT_EQ(inst.op, Op::AND);
    ASSERT_EQ(inst.rd, 10);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.rs2, 12);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_or) {
    auto inst = Decoder::decode(0x00f7'66b3u); // or x13, x14, x15
    ASSERT_EQ(inst.op, Op::OR);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs1, 14);
    ASSERT_EQ(inst.rs2, 15);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_xor) {
    auto inst = Decoder::decode(0x0128'c833u); // xor x16, x17, x18
    ASSERT_EQ(inst.op, Op::XOR);
    ASSERT_EQ(inst.rd, 16);
    ASSERT_EQ(inst.rs1, 17);
    ASSERT_EQ(inst.rs2, 18);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_sll) {
    auto inst = Decoder::decode(0x015a'19b3u); // sll x19, x20, x21
    ASSERT_EQ(inst.op, Op::SLL);
    ASSERT_EQ(inst.rd, 19);
    ASSERT_EQ(inst.rs1, 20);
    ASSERT_EQ(inst.rs2, 21);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_srl) {
    auto inst = Decoder::decode(0x018b'db33u); // srl x22, x23, x24
    ASSERT_EQ(inst.op, Op::SRL);
    ASSERT_EQ(inst.rd, 22);
    ASSERT_EQ(inst.rs1, 23);
    ASSERT_EQ(inst.rs2, 24);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_sra) {
    auto inst = Decoder::decode(0x41bd'5cb3u); // sra x25, x26, x27
    ASSERT_EQ(inst.op, Op::SRA);
    ASSERT_EQ(inst.rd, 25);
    ASSERT_EQ(inst.rs1, 26);
    ASSERT_EQ(inst.rs2, 27);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_slt) {
    auto inst = Decoder::decode(0x01ee'ae33u); // slt x28, x29, x30
    ASSERT_EQ(inst.op, Op::SLT);
    ASSERT_EQ(inst.rd, 28);
    ASSERT_EQ(inst.rs1, 29);
    ASSERT_EQ(inst.rs2, 30);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_sltu) {
    auto inst = Decoder::decode(0x0020'bfb3u); // sltu x31, x1, x2
    ASSERT_EQ(inst.op, Op::SLTU);
    ASSERT_EQ(inst.rd, 31);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. I-type instructions (ALU, loads, JALR)
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_addi) {
    auto inst = Decoder::decode(0x02a0'0093u); // addi x1, x0, 42
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, 42);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_addi_negative) {
    auto inst = Decoder::decode(0xffb1'8113u); // addi x2, x3, -5
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.rs1, 3);
    ASSERT_EQ(inst.imm, -5);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_addi_max_imm) {
    auto inst = Decoder::decode(0x7ff0'0093u); // addi x1, x0, 2047
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, 2047);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_addi_min_imm) {
    auto inst = Decoder::decode(0x8000'0093u); // addi x1, x0, -2048
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, -2048);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_slti) {
    auto inst = Decoder::decode(0x0642'a213u); // slti x4, x5, 100
    ASSERT_EQ(inst.op, Op::SLTI);
    ASSERT_EQ(inst.rd, 4);
    ASSERT_EQ(inst.rs1, 5);
    ASSERT_EQ(inst.imm, 100);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_sltiu) {
    auto inst = Decoder::decode(0x0c83'b313u); // sltiu x6, x7, 200
    ASSERT_EQ(inst.op, Op::SLTIU);
    ASSERT_EQ(inst.rd, 6);
    ASSERT_EQ(inst.rs1, 7);
    ASSERT_EQ(inst.imm, 200);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_xori) {
    auto inst = Decoder::decode(0x0ff4'c413u); // xori x8, x9, 0xFF
    ASSERT_EQ(inst.op, Op::XORI);
    ASSERT_EQ(inst.rd, 8);
    ASSERT_EQ(inst.rs1, 9);
    ASSERT_EQ(inst.imm, 0xFF);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_ori) {
    auto inst = Decoder::decode(0x1235'e513u); // ori x10, x11, 0x123
    ASSERT_EQ(inst.op, Op::ORI);
    ASSERT_EQ(inst.rd, 10);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.imm, 0x123);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_andi) {
    auto inst = Decoder::decode(0x7ff6'f613u); // andi x12, x13, 0x7FF
    ASSERT_EQ(inst.op, Op::ANDI);
    ASSERT_EQ(inst.rd, 12);
    ASSERT_EQ(inst.rs1, 13);
    ASSERT_EQ(inst.imm, 0x7FF);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_slli) {
    auto inst = Decoder::decode(0x0057'9713u); // slli x14, x15, 5
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.rd, 14);
    ASSERT_EQ(inst.rs1, 15);
    ASSERT_EQ(inst.imm, 5);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_slli_zero) {
    auto inst = Decoder::decode(0x0001'1093u); // slli x1, x2, 0
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 0);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_slli_max) {
    auto inst = Decoder::decode(0x01f1'1093u); // slli x1, x2, 31
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 31);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_srli) {
    auto inst = Decoder::decode(0x00a8'd813u); // srli x16, x17, 10
    ASSERT_EQ(inst.op, Op::SRLI);
    ASSERT_EQ(inst.rd, 16);
    ASSERT_EQ(inst.rs1, 17);
    ASSERT_EQ(inst.imm, 10);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_srai) {
    auto inst = Decoder::decode(0x40f9'd913u); // srai x18, x19, 15
    ASSERT_EQ(inst.op, Op::SRAI);
    ASSERT_EQ(inst.rd, 18);
    ASSERT_EQ(inst.rs1, 19);
    ASSERT_EQ(inst.imm, 15);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_srai_max) {
    auto inst = Decoder::decode(0x41f1'5093u); // srai x1, x2, 31
    ASSERT_EQ(inst.op, Op::SRAI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 31);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

// ── Loads ──────────────────────────────────────────────────────────────

TEST(decode_lw) {
    auto inst = Decoder::decode(0x0081'2083u); // lw x1, 8(x2)
    ASSERT_EQ(inst.op, Op::LW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 8);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_lh) {
    auto inst = Decoder::decode(0xffc2'1183u); // lh x3, -4(x4)
    ASSERT_EQ(inst.op, Op::LH);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs1, 4);
    ASSERT_EQ(inst.imm, -4);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_lb) {
    auto inst = Decoder::decode(0x0003'0283u); // lb x5, 0(x6)
    ASSERT_EQ(inst.op, Op::LB);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.imm, 0);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_lhu) {
    auto inst = Decoder::decode(0x0104'5383u); // lhu x7, 16(x8)
    ASSERT_EQ(inst.op, Op::LHU);
    ASSERT_EQ(inst.rd, 7);
    ASSERT_EQ(inst.rs1, 8);
    ASSERT_EQ(inst.imm, 16);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_lbu) {
    auto inst = Decoder::decode(0x0015'4483u); // lbu x9, 1(x10)
    ASSERT_EQ(inst.op, Op::LBU);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs1, 10);
    ASSERT_EQ(inst.imm, 1);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_load_max_imm) {
    auto inst = Decoder::decode(0x7ff1'2083u); // lw x1, 2047(x2)
    ASSERT_EQ(inst.op, Op::LW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 2047);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_load_min_imm) {
    auto inst = Decoder::decode(0x8001'2083u); // lw x1, -2048(x2)
    ASSERT_EQ(inst.op, Op::LW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, -2048);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

// ── JALR ────────────────────────────────────────────────────────────────────

TEST(decode_jalr) {
    auto inst = Decoder::decode(0x0081'00e7u); // jalr x1, 8(x2)
    ASSERT_EQ(inst.op, Op::JALR);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 8);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_jalr_max_imm) {
    auto inst = Decoder::decode(0x7ff1'00e7u); // jalr x1, 2047(x2)
    ASSERT_EQ(inst.op, Op::JALR);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 2047);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_jalr_min_imm) {
    auto inst = Decoder::decode(0x8001'00e7u); // jalr x1, -2048(x2)
    ASSERT_EQ(inst.op, Op::JALR);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, -2048);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. S-type instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_sw) {
    auto inst = Decoder::decode(0x0011'2623u); // sw x1, 12(x2)
    ASSERT_EQ(inst.op, Op::SW);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.imm, 12);
    ASSERT_EQ(inst.format, Format::S);
    return true;
}

TEST(decode_sh) {
    auto inst = Decoder::decode(0xfe32'1c23u); // sh x3, -8(x4)
    ASSERT_EQ(inst.op, Op::SH);
    ASSERT_EQ(inst.rs1, 4);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.imm, -8);
    ASSERT_EQ(inst.format, Format::S);
    return true;
}

TEST(decode_sb) {
    auto inst = Decoder::decode(0x0053'03a3u); // sb x5, 7(x6)
    ASSERT_EQ(inst.op, Op::SB);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.rs2, 5);
    ASSERT_EQ(inst.imm, 7);
    ASSERT_EQ(inst.format, Format::S);
    return true;
}

TEST(decode_store_max_imm) {
    auto inst = Decoder::decode(0x7e11'2fa3u); // sw x1, 2047(x2)
    ASSERT_EQ(inst.op, Op::SW);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.imm, 2047);
    ASSERT_EQ(inst.format, Format::S);
    return true;
}

TEST(decode_store_min_imm) {
    auto inst = Decoder::decode(0x8011'2023u); // sw x1, -2048(x2)
    ASSERT_EQ(inst.op, Op::SW);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.imm, -2048);
    ASSERT_EQ(inst.format, Format::S);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  4. B-type instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_beq) {
    auto inst = Decoder::decode(0x0020'8863u); // beq x1, x2, 16
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.imm, 16);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_bne) {
    auto inst = Decoder::decode(0xfe41'9ce3u); // bne x3, x4, -8
    ASSERT_EQ(inst.op, Op::BNE);
    ASSERT_EQ(inst.rs1, 3);
    ASSERT_EQ(inst.rs2, 4);
    ASSERT_EQ(inst.imm, -8);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_blt) {
    auto inst = Decoder::decode(0x0262'c063u); // blt x5, x6, 32
    ASSERT_EQ(inst.op, Op::BLT);
    ASSERT_EQ(inst.rs1, 5);
    ASSERT_EQ(inst.rs2, 6);
    ASSERT_EQ(inst.imm, 32);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_bge) {
    auto inst = Decoder::decode(0x0083'dc63u); // bge x7, x8, 24
    ASSERT_EQ(inst.op, Op::BGE);
    ASSERT_EQ(inst.rs1, 7);
    ASSERT_EQ(inst.rs2, 8);
    ASSERT_EQ(inst.imm, 24);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_bltu) {
    auto inst = Decoder::decode(0x02a4'e463u); // bltu x9, x10, 40
    ASSERT_EQ(inst.op, Op::BLTU);
    ASSERT_EQ(inst.rs1, 9);
    ASSERT_EQ(inst.rs2, 10);
    ASSERT_EQ(inst.imm, 40);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_bgeu) {
    auto inst = Decoder::decode(0x02c5'f863u); // bgeu x11, x12, 48
    ASSERT_EQ(inst.op, Op::BGEU);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.rs2, 12);
    ASSERT_EQ(inst.imm, 48);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_branch_min_imm) {
    auto inst = Decoder::decode(0x8000'8063u); // beq x1, x0, -4096
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 0);
    ASSERT_EQ(inst.imm, -4096);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

TEST(decode_branch_max_imm) {
    auto inst = Decoder::decode(0x7e00'8fe3u); // beq x1, x0, 4094
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 0);
    ASSERT_EQ(inst.imm, 4094);
    ASSERT_EQ(inst.format, Format::B);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. U-type instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_lui) {
    auto inst = Decoder::decode(0x1234'50b7u); // lui x1, 0x12345
    ASSERT_EQ(inst.op, Op::LUI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.imm, 0x12345000);
    ASSERT_EQ(inst.format, Format::U);
    return true;
}

TEST(decode_auipc) {
    auto inst = Decoder::decode(0xabcde117u); // auipc x2, 0xABCDE
    ASSERT_EQ(inst.op, Op::AUIPC);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.imm, static_cast<i32>(0xABCDE000));
    ASSERT_EQ(inst.format, Format::U);
    return true;
}

TEST(decode_utype_max_imm) {
    auto inst = Decoder::decode(0x7fff'f0b7u); // lui x1, 0x7ffff
    ASSERT_EQ(inst.op, Op::LUI);
    ASSERT_EQ(inst.rd, 1);
    constexpr i32 imm20 = (1 << 19) - 1;
    ASSERT_EQ(inst.imm, imm20 << 12);
    ASSERT_EQ(inst.format, Format::U);
    return true;
}

TEST(decode_utype_min_imm) {
    auto inst = Decoder::decode(0x8000'00b7u); // lui x1, 0x80000
    ASSERT_EQ(inst.op, Op::LUI);
    ASSERT_EQ(inst.rd, 1);
    constexpr i32 imm20 = -(1 << 19);
    ASSERT_EQ(inst.imm, imm20 << 12);
    ASSERT_EQ(inst.format, Format::U);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  6. J-type (JAL)
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_jal) {
    auto inst = Decoder::decode(0x0640'00efu); // jal x1, 100
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.imm, 100);
    ASSERT_EQ(inst.format, Format::J);
    return true;
}

TEST(decode_jal_negative) {
    auto inst = Decoder::decode(0xfedf'f06fu); // jal x0, -20
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.imm, -20);
    ASSERT_EQ(inst.format, Format::J);
    return true;
}

TEST(decode_jal_max_imm) {
    auto inst = Decoder::decode(0x7fff'f06fu); // jal x0, 1048574
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.imm, 1048574);
    ASSERT_EQ(inst.format, Format::J);
    return true;
}

TEST(decode_jal_min_imm) {
    auto inst = Decoder::decode(0x8000'006fu); // jal x0, -1048576
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.imm, -1048576);
    ASSERT_EQ(inst.format, Format::J);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. System instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_ecall) {
    auto inst = Decoder::decode(0x0000'0073u);
    ASSERT_EQ(inst.op, Op::ECALL);
    return true;
}

TEST(decode_ebreak) {
    auto inst = Decoder::decode(0x0010'0073u);
    ASSERT_EQ(inst.op, Op::EBREAK);
    return true;
}

TEST(decode_fence) {
    auto inst = Decoder::decode(0x0ff0'000fu); // fence iorw, iorw
    ASSERT_EQ(inst.op, Op::FENCE);
    return true;
}

TEST(decode_mret) {
    auto inst = Decoder::decode(0x3020'0073u);
    ASSERT_EQ(inst.op, Op::MRET);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  8. DecodedInst query helpers
// ═══════════════════════════════════════════════════════════════════════

TEST(decoded_writes_rd) {
    auto expects_no_rd_write = [](u32 raw)
    {
        auto inst = Decoder::decode(raw);
        ASSERT(!inst.writes_rd());
    };

    auto add = Decoder::decode(0x0031'00b3u); // add x1, x2, x3
    ASSERT(add.writes_rd());

    expects_no_rd_write(0x0031'0033u); // add x0, x2, x3  - x0 is always 0
    expects_no_rd_write(0x0011'2623u); // sw  x1, 12(x2)
    expects_no_rd_write(0x0020'8863u); // beq x1, x2, 16

    return true;
}

TEST(decoded_reads_rs1) {
    auto add = Decoder::decode(0x0031'00b3u); // add x1, x2, x3
    ASSERT(add.reads_rs1());

    auto lui = Decoder::decode(0x1234'50b7u); // lui x1, 0x12345
    ASSERT(!lui.reads_rs1());

    auto jal = Decoder::decode(0x0640'00efu); // jal x1, 100
    ASSERT(!jal.reads_rs1());

    return true;
}

TEST(decoded_reads_rs2) {
    auto add = Decoder::decode(0x0031'00b3u);  // add x1, x2, x3
    ASSERT(add.reads_rs2());

    auto addi = Decoder::decode(0x02a0'0093u); // addi x1, x0, 42
    ASSERT(!addi.reads_rs2());

    auto sw = Decoder::decode(0x0011'2623u);   // sw x1, 12(x2)
    ASSERT(sw.reads_rs2());

    return true;
}

TEST(disassemble) {
    auto inst = Decoder::decode(0x0031'00b3u); // add x1, x2, x3
    auto dis = inst.disassemble();
    ASSERT(dis.find("add") != std::string::npos);
    ASSERT(dis.find("ra") != std::string::npos);  // x1 = ra
    return true;
}

TEST(disassemble_formats) {
    // Smoke-test that disassembly doesn't crash for each format
    Decoder::decode(0x0031'00b3u).disassemble();         // R
    Decoder::decode(0x02a0'0093u).disassemble();         // I (ALU)
    Decoder::decode(0x0081'2083u).disassemble();         // I (load)
    Decoder::decode(0x0011'2623u).disassemble();         // S
    Decoder::decode(0x0020'8863u, 0x1000).disassemble(); // B (with PC)
    Decoder::decode(0x1234'50b7u).disassemble();         // U
    Decoder::decode(0x0640'00efu, 0x2000).disassemble(); // J (with PC)
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  9. Edge cases and error paths
// ═══════════════════════════════════════════════════════════════════════

TEST(decode_invalid_opcode) {
    // All-zeros has opcode 0b0000000 which is not assigned
    auto inst = Decoder::decode(0x0000'0000u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_all_ones) {
    // All-ones (0xffff'ffff) has opcode 0b1111111 which is not assigned
    auto inst = Decoder::decode(0xffff'ffffu);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_r_type_funct7) {
    // An R-type instruction with funct3=000 but funct7=0000010
    // Not assigned to any standard extension - should decode as INVALID
    auto inst = Decoder::decode(0x0420'81b3u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_load_funct3) {
    // LOAD opcode with funct3=011 (not assigned)
    auto inst = Decoder::decode(0x0001'b083u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_jalr_funct3) {
    // JALR opcode (0b1100111) but with funct3=001 instead of 000
    auto inst = Decoder::decode(0x0000'9167u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_store_funct3) {
    // STORE opcode with funct3=011 (not assigned)
    auto inst = Decoder::decode(0x0021'b023u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_branch_funct3) {
    // BRANCH opcode with funct3=010 (not assigned)
    auto inst = Decoder::decode(0x0021'2063u);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_rd_field_is_zero) {
    // addi x0, x1, 5 -> NOP-like
    auto inst = Decoder::decode(0x0050'8013u);
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 0);
    ASSERT(!inst.writes_rd());
    return true;
}

TEST(decode_preserves_raw_and_pc) {
    const u32 encoding = 0x0031'00b3u;
    const addr_t pc    = 0xDEAD'BEE0u;
    auto inst = Decoder::decode(encoding, pc);
    ASSERT_HEX_EQ(inst.raw, encoding);
    ASSERT_HEX_EQ(inst.pc, pc);
    return true;
}

TEST(decode_nop) {
    // The canonical NOP is addi x0, x0, 0
    auto inst = Decoder::decode(0x0000'0013u);
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, 0);
    ASSERT(!inst.writes_rd());
    return true;
}
