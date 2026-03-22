/**
 * @file test_decoder.cpp
 * @brief Decoder tests covering all RV32I instruction encodings.
 *
 * Sections:
 *   1 (line 82)  : R-type (register–register)
 *   2 (line 177) : I-type (immediate ALU, loads, JALR)
 *   3 (line 368) : S-type (stores)
 *   4 (line 401) : B-type (branches)
 *   5 (line 475) : U-type (LUI, AUIPC)
 *   6 (line 505) : J-type (JAL)
 *   7 (line 541) : System (ECALL, EBREAK, FENCE)
 *   8 (line 563) : DecodedInst query helpers
 *   9 (line 647) : Edge cases and error paths
 *
 * Instruction encodings are generated from the canonical RV32I encoding
 * tables and cross-checked with `riscv64-unknown-elf-objdump -d`.
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

/* ═══════════════════════════════════════════════════════════════════════
 *  1. R-type Instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_add) {
    auto inst = Decoder::decode(0x003100b3);  // add x1, x2, x3
    ASSERT_EQ(inst.op, Op::ADD);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(decode_sub) {
    auto inst = Decoder::decode(0x407302b3);  // sub x5, x6, x7
    ASSERT_EQ(inst.op, Op::SUB);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.rs2, 7);
    return true;
}

TEST(decode_and) {
    auto inst = Decoder::decode(0x00c5f533);  // and x10, x11, x12
    ASSERT_EQ(inst.op, Op::AND);
    ASSERT_EQ(inst.rd, 10);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.rs2, 12);
    return true;
}

TEST(decode_or) {
    auto inst = Decoder::decode(0x00f766b3);  // or x13, x14, x15
    ASSERT_EQ(inst.op, Op::OR);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs1, 14);
    ASSERT_EQ(inst.rs2, 15);
    return true;
}

TEST(decode_xor) {
    auto inst = Decoder::decode(0x0128c833);  // xor x16, x17, x18
    ASSERT_EQ(inst.op, Op::XOR);
    ASSERT_EQ(inst.rd, 16);
    ASSERT_EQ(inst.rs1, 17);
    ASSERT_EQ(inst.rs2, 18);
    return true;
}

TEST(decode_sll) {
    auto inst = Decoder::decode(0x015a19b3);  // sll x19, x20, x21
    ASSERT_EQ(inst.op, Op::SLL);
    ASSERT_EQ(inst.rd, 19);
    ASSERT_EQ(inst.rs1, 20);
    ASSERT_EQ(inst.rs2, 21);
    return true;
}

TEST(decode_srl) {
    auto inst = Decoder::decode(0x018bdb33);  // srl x22, x23, x24
    ASSERT_EQ(inst.op, Op::SRL);
    ASSERT_EQ(inst.rd, 22);
    ASSERT_EQ(inst.rs1, 23);
    ASSERT_EQ(inst.rs2, 24);
    return true;
}

TEST(decode_sra) {
    auto inst = Decoder::decode(0x41bd5cb3);  // sra x25, x26, x27
    ASSERT_EQ(inst.op, Op::SRA);
    ASSERT_EQ(inst.rd, 25);
    ASSERT_EQ(inst.rs1, 26);
    ASSERT_EQ(inst.rs2, 27);
    return true;
}

TEST(decode_slt) {
    auto inst = Decoder::decode(0x01eeae33);  // slt x28, x29, x30
    ASSERT_EQ(inst.op, Op::SLT);
    ASSERT_EQ(inst.rd, 28);
    ASSERT_EQ(inst.rs1, 29);
    ASSERT_EQ(inst.rs2, 30);
    return true;
}

TEST(decode_sltu) {
    auto inst = Decoder::decode(0x0020bfb3);  // sltu x31, x1, x2
    ASSERT_EQ(inst.op, Op::SLTU);
    ASSERT_EQ(inst.rd, 31);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. I-type Instructions (ALU, loads, JALR)
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_addi) {
    // addi x1, x0, 42  →  0x02a00093
    auto inst = Decoder::decode(0x02a00093);  // addi x1, x0, 42
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, 42);
    ASSERT_EQ(inst.format, Format::I);
    return true;
}

TEST(decode_addi_negative) {
    auto inst = Decoder::decode(0xffb18113);  // addi x2, x3, -5
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.rs1, 3);
    ASSERT_EQ(inst.imm, -5);
    return true;
}

TEST(decode_addi_max_positive) {
    auto inst = Decoder::decode(0x7ff00093);  // addi x1, x0, 2047
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.imm, 2047);
    return true;
}

TEST(decode_addi_min_negative) {
    auto inst = Decoder::decode(0x80000093);  // addi x1, x0, -2048
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.imm, -2048);
    return true;
}

TEST(decode_slti) {
    auto inst = Decoder::decode(0x0642a213);  // slti x4, x5, 100
    ASSERT_EQ(inst.op, Op::SLTI);
    ASSERT_EQ(inst.rd, 4);
    ASSERT_EQ(inst.rs1, 5);
    ASSERT_EQ(inst.imm, 100);
    return true;
}

TEST(decode_sltiu) {
    auto inst = Decoder::decode(0x0c83b313);  // sltiu x6, x7, 200
    ASSERT_EQ(inst.op, Op::SLTIU);
    ASSERT_EQ(inst.rd, 6);
    ASSERT_EQ(inst.rs1, 7);
    ASSERT_EQ(inst.imm, 200);
    return true;
}

TEST(decode_xori) {
    auto inst = Decoder::decode(0x0ff4c413);  // xori x8, x9, 0xFF
    ASSERT_EQ(inst.op, Op::XORI);
    ASSERT_EQ(inst.rd, 8);
    ASSERT_EQ(inst.rs1, 9);
    ASSERT_EQ(inst.imm, 0xFF);
    return true;
}

TEST(decode_ori) {
    auto inst = Decoder::decode(0x1235e513);  // ori x10, x11, 0x123
    ASSERT_EQ(inst.op, Op::ORI);
    ASSERT_EQ(inst.rd, 10);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.imm, 0x123);
    return true;
}

TEST(decode_andi) {
    auto inst = Decoder::decode(0x7ff6f613);  // andi x12, x13, 0x7FF
    ASSERT_EQ(inst.op, Op::ANDI);
    ASSERT_EQ(inst.rd, 12);
    ASSERT_EQ(inst.rs1, 13);
    ASSERT_EQ(inst.imm, 0x7FF);
    return true;
}

TEST(decode_slli) {
    auto inst = Decoder::decode(0x00579713);  // slli x14, x15, 5
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.rd, 14);
    ASSERT_EQ(inst.rs1, 15);
    ASSERT_EQ(inst.imm, 5);
    return true;
}

TEST(decode_slli_zero) {
    auto inst = Decoder::decode(0x00011093);  // slli x1, x2, 0
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.imm, 0);
    return true;
}

TEST(decode_slli_max) {
    auto inst = Decoder::decode(0x01f11093);  // slli x1, x2, 31
    ASSERT_EQ(inst.op, Op::SLLI);
    ASSERT_EQ(inst.imm, 31);
    return true;
}

TEST(decode_srli) {
    auto inst = Decoder::decode(0x00a8d813);  // srli x16, x17, 10
    ASSERT_EQ(inst.op, Op::SRLI);
    ASSERT_EQ(inst.rd, 16);
    ASSERT_EQ(inst.rs1, 17);
    ASSERT_EQ(inst.imm, 10);
    return true;
}

TEST(decode_srai) {
    auto inst = Decoder::decode(0x40f9d913);  // srai x18, x19, 15
    ASSERT_EQ(inst.op, Op::SRAI);
    ASSERT_EQ(inst.rd, 18);
    ASSERT_EQ(inst.rs1, 19);
    ASSERT_EQ(inst.imm, 15);
    return true;
}

TEST(decode_srai_max) {
    auto inst = Decoder::decode(0x41f15093);  // srai x1, x2, 31
    ASSERT_EQ(inst.op, Op::SRAI);
    ASSERT_EQ(inst.imm, 31);
    return true;
}

// ── Loads ──────────────────────────────────────────────────────────────

TEST(decode_lw) {
    auto inst = Decoder::decode(0x00812083);  // lw x1, 8(x2)
    ASSERT_EQ(inst.op, Op::LW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 8);
    ASSERT(inst.is_load());
    return true;
}

TEST(decode_lh) {
    auto inst = Decoder::decode(0xffc21183);  // lh x3, -4(x4)
    ASSERT_EQ(inst.op, Op::LH);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs1, 4);
    ASSERT_EQ(inst.imm, -4);
    return true;
}

TEST(decode_lb) {
    auto inst = Decoder::decode(0x00030283);  // lb x5, 0(x6)
    ASSERT_EQ(inst.op, Op::LB);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.imm, 0);
    return true;
}

TEST(decode_lhu) {
    auto inst = Decoder::decode(0x01045383);  // lhu x7, 16(x8)
    ASSERT_EQ(inst.op, Op::LHU);
    ASSERT_EQ(inst.rd, 7);
    ASSERT_EQ(inst.rs1, 8);
    ASSERT_EQ(inst.imm, 16);
    return true;
}

TEST(decode_lbu) {
    auto inst = Decoder::decode(0x00154483);  // lbu x9, 1(x10)
    ASSERT_EQ(inst.op, Op::LBU);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs1, 10);
    ASSERT_EQ(inst.imm, 1);
    return true;
}

// ── JALR ────────────────────────────────────────────────────────────────────

TEST(decode_jalr) {
    auto inst = Decoder::decode(0x008100e7);  // jalr x1, 8(x2)
    ASSERT_EQ(inst.op, Op::JALR);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.imm, 8);
    ASSERT(inst.is_jump());
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. S-type Instructions (stores)
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_sw) {
    auto inst = Decoder::decode(0x00112623);  // sw x1, 12(x2)
    ASSERT_EQ(inst.op, Op::SW);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.imm, 12);
    ASSERT_EQ(inst.format, Format::S);
    ASSERT(inst.is_store());
    return true;
}

TEST(decode_sh) {
    auto inst = Decoder::decode(0xfe321c23);  // sh x3, -8(x4)
    ASSERT_EQ(inst.op, Op::SH);
    ASSERT_EQ(inst.rs1, 4);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.imm, -8);
    return true;
}

TEST(decode_sb) {
    auto inst = Decoder::decode(0x005303a3);  // sb x5, 7(x6)
    ASSERT_EQ(inst.op, Op::SB);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.rs2, 5);
    ASSERT_EQ(inst.imm, 7);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. B-type Instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_beq) {
    auto inst = Decoder::decode(0x00208863);  // beq x1, x2, 16
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.imm, 16);
    ASSERT_EQ(inst.format, Format::B);
    ASSERT(inst.is_branch());
    return true;
}

TEST(decode_bne) {
    auto inst = Decoder::decode(0xfe419ce3);  // bne x3, x4, -8
    ASSERT_EQ(inst.op, Op::BNE);
    ASSERT_EQ(inst.rs1, 3);
    ASSERT_EQ(inst.rs2, 4);
    ASSERT_EQ(inst.imm, -8);
    return true;
}

TEST(decode_blt) {
    auto inst = Decoder::decode(0x0262c063);  // blt x5, x6, 32
    ASSERT_EQ(inst.op, Op::BLT);
    ASSERT_EQ(inst.rs1, 5);
    ASSERT_EQ(inst.rs2, 6);
    ASSERT_EQ(inst.imm, 32);
    return true;
}

TEST(decode_bge) {
    auto inst = Decoder::decode(0x0083dc63);  // bge x7, x8, 24
    ASSERT_EQ(inst.op, Op::BGE);
    ASSERT_EQ(inst.rs1, 7);
    ASSERT_EQ(inst.rs2, 8);
    ASSERT_EQ(inst.imm, 24);
    return true;
}

TEST(decode_bltu) {
    auto inst = Decoder::decode(0x02a4e463);  // bltu x9, x10, 40
    ASSERT_EQ(inst.op, Op::BLTU);
    ASSERT_EQ(inst.rs1, 9);
    ASSERT_EQ(inst.rs2, 10);
    ASSERT_EQ(inst.imm, 40);
    return true;
}

TEST(decode_bgeu) {
    auto inst = Decoder::decode(0x02c5f863);  // bgeu x11, x12, 48
    ASSERT_EQ(inst.op, Op::BGEU);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.rs2, 12);
    ASSERT_EQ(inst.imm, 48);
    return true;
}

TEST(decode_branch_max_negative) {
    auto inst = Decoder::decode(0x80008063);  // beq x1, x0, -4096
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.imm, -4096);
    return true;
}

TEST(decode_branch_max_positive) {
    auto inst = Decoder::decode(0x7fe00fe3);  // beq x0, x0, 4094
    ASSERT_EQ(inst.op, Op::BEQ);
    ASSERT_EQ(inst.imm, 4094);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. U-type Instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_lui) {
    auto inst = Decoder::decode(0x123450b7);  // lui x1, 0x12345
    ASSERT_EQ(inst.op, Op::LUI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.imm, 0x12345000);
    ASSERT_EQ(inst.format, Format::U);
    return true;
}

TEST(decode_lui_sign_bit) {
    // imm = 0x80000000 which is negative when viewed as i32.
    auto inst = Decoder::decode(0x800000b7);  // lui x1, 0x80000
    ASSERT_EQ(inst.op, Op::LUI);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.imm, static_cast<i32>(0x80000000u));
    return true;
}

TEST(decode_auipc) {
    auto inst = Decoder::decode(0xabcde117);  // auipc x2, 0xABCDE
    ASSERT_EQ(inst.op, Op::AUIPC);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.imm, static_cast<i32>(0xABCDE000));
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. J-type Instruction
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_jal) {
    auto inst = Decoder::decode(0x064000ef);  // jal x1, 100
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.imm, 100);
    ASSERT_EQ(inst.format, Format::J);
    ASSERT(inst.is_jump());
    return true;
}

TEST(decode_jal_negative) {
    auto inst = Decoder::decode(0xfedff06f);  // jal x0, -20
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.imm, -20);
    return true;
}

TEST(decode_jal_max_positive) {
    auto inst = Decoder::decode(0x7ffff06f);  // jal x0, 1048574
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.imm, 1048574);
    return true;
}

TEST(decode_jal_max_negative) {
    auto inst = Decoder::decode(0x8000006f);  // jal x0, -1048576
    ASSERT_EQ(inst.op, Op::JAL);
    ASSERT_EQ(inst.imm, -1048576);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. System Instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_ecall) {
    auto inst = Decoder::decode(0x00000073);  // ecall
    ASSERT_EQ(inst.op, Op::ECALL);
    return true;
}

TEST(decode_ebreak) {
    auto inst = Decoder::decode(0x00100073);  // ebreak
    ASSERT_EQ(inst.op, Op::EBREAK);
    return true;
}

TEST(decode_fence) {
    auto inst = Decoder::decode(0x0ff0000f);  // fence iorw, iorw
    ASSERT_EQ(inst.op, Op::FENCE);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. DecodedInst query helpers
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decoded_writes_rd) {
    auto add = Decoder::decode(0x003100b3);     // add x1, x2, x3
    ASSERT(add.writes_rd());

    auto add_x0 = Decoder::decode(0x00310033);  // add x0, x2, x3
    ASSERT(!add_x0.writes_rd());                // x0 is always 0

    auto sw = Decoder::decode(0x00112623);      // sw x1, 12(x2)
    ASSERT(!sw.writes_rd());

    auto beq = Decoder::decode(0x00208863);     // beq x1, x2, 16
    ASSERT(!beq.writes_rd());

    return true;
}

TEST(decoded_reads_rs1) {
    auto add = Decoder::decode(0x003100b3);  // add x1, x2, x3
    ASSERT(add.reads_rs1());

    auto lui = Decoder::decode(0x123450b7);  // lui x1, 0x12345
    ASSERT(!lui.reads_rs1());

    auto jal = Decoder::decode(0x064000ef);  // jal x1, 100
    ASSERT(!jal.reads_rs1());

    return true;
}

TEST(decoded_reads_rs2) {
    auto add = Decoder::decode(0x003100b3);   // add x1, x2, x3
    ASSERT(add.reads_rs2());

    auto addi = Decoder::decode(0x02a00093);  // addi x1, x0, 42
    ASSERT(!addi.reads_rs2());

    auto sw = Decoder::decode(0x00112623);    // sw x1, 12(x2)
    ASSERT(sw.reads_rs2());

    return true;
}

TEST(decoded_classification_negative) {
    // An ADD should not be classified as load, store, branch, or jump.
    auto add = Decoder::decode(0x003100b3);
    ASSERT(!add.is_load());
    ASSERT(!add.is_store());
    ASSERT(!add.is_branch());
    ASSERT(!add.is_jump());

    // A LW should not be classified as store, branch, or jump.
    auto lw = Decoder::decode(0x00812083);
    ASSERT(lw.is_load());
    ASSERT(!lw.is_store());
    ASSERT(!lw.is_branch());
    ASSERT(!lw.is_jump());

    return true;
}

TEST(disassemble) {
    auto inst = Decoder::decode(0x003100b3);  // add x1, x2, x3
    auto dis = inst.disassemble();
    ASSERT(dis.find("add") != std::string::npos);
    ASSERT(dis.find("ra") != std::string::npos);  // x1 = ra
    return true;
}

TEST(disassemble_formats) {
    // Smoke-test that disassembly doesn't crash for each format.
    Decoder::decode(0x003100b3).disassemble();          // R
    Decoder::decode(0x02a00093).disassemble();          // I (ALU)
    Decoder::decode(0x00812083).disassemble();          // I (load)
    Decoder::decode(0x00112623).disassemble();          // S
    Decoder::decode(0x00208863, 0x1000).disassemble();  // B (with PC)
    Decoder::decode(0x123450b7).disassemble();          // U
    Decoder::decode(0x064000ef, 0x2000).disassemble();  // J (with PC)
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  9. Edge cases and error paths
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(decode_invalid_opcode) {
    // All-zeros has opcode 0b0000000 which is not assigned.
    auto inst = Decoder::decode(0x00000000);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_all_ones) {
    // All-ones (0xFFFFFFFF) has opcode 0b1111111 which is not assigned.
    auto inst = Decoder::decode(0xFFFFFFFF);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_r_type_funct7) {
    // An R-type instruction with funct3=000 but funct7=0000010
    // Not assigned to any standard extension - should decode as INVALID.
    auto inst = Decoder::decode(0x042081b3);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_jalr_funct3) {
    // JALR opcode (0b1100111) but with funct3=001 instead of 000.
    // This should decode as INVALID per spec §(COME BACK).
    auto inst = Decoder::decode(0x00009167);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_load_funct3) {
    // LOAD opcode with funct3=011 (not assigned in RV32I).
    auto inst = Decoder::decode(0x0001b083);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_branch_funct3) {
    // BRANCH opcode with funct3=010 (not assigned).
    auto inst = Decoder::decode(0x00212063);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_invalid_store_funct3) {
    // STORE opcode with funct3=011 (not assigned in RV32I).
    auto inst = Decoder::decode(0x0021b023);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_csr_as_invalid) {
    // CSR instruction (SYSTEM opcode, funct3 != 0) — not implemented,
    // should decode as INVALID.
    auto inst = Decoder::decode(0x300110f3);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(decode_rd_field_is_zero) {
    // addi x0, x1, 5 -> NOP-like (writes to x0, discarded).
    // Verify the decoder still sets rd=0.
    auto inst = Decoder::decode(0x00508013);
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 0);
    ASSERT(!inst.writes_rd());
    return true;
}

TEST(decode_preserves_raw_and_pc) {
    const u32 encoding = 0x003100b3;
    const addr_t pc    = 0xDEAD'BEE0;
    auto inst = Decoder::decode(encoding, pc);
    ASSERT_HEX_EQ(inst.raw, encoding);
    ASSERT_HEX_EQ(inst.pc, pc);
    return true;
}

TEST(decode_nop) {
    // The canonical NOP is addi x0, x0, 0 -> 0x00000013
    auto inst = Decoder::decode(0x00000013);
    ASSERT_EQ(inst.op, Op::ADDI);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.rs1, 0);
    ASSERT_EQ(inst.imm, 0);
    ASSERT(!inst.writes_rd());
    return true;
}
