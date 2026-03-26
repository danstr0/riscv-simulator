/**
 * @file test_rvv.cpp
 * @brief Tests for the RVV subset implementation.
 *
 * Sections:
 * 1  (line 157) : Decoder - every RVV Op decodes from its encoding helper
 * 2  (line 198) : Decoder - reject invalid encodings
 * 3  (line 216) : Vector state - VSETVLI configuration, VLMAX
 * 4  (line 257) : VLE32/VSE32 - vector load and store
 * 5  (line 286) : Arithmetic - VADD, VSUB
 * 6  (line 363) : Bitwise - VAND, VOR, VXOR, VSLL, VSRL
 * 7  (line 558) : Comparison - VMSEQ, VMSLT, VMSLTU
 * 8  (line 669) : Move and Reduction - VMV.V.X, VMV.X.S, VREDSUM
 * 9  (line 738) : Program - Vectorized checksum program
 * 10 (line 789) : Pipeline - VSETVLI, VLE32/VSE32, VADD, VREDSUM, VMV
 * 11 (line 902) : Pipeline - Vectorized checksum through pipeline
 */

#include "core/cpu.hpp"
#include "core/pipeline.hpp"
#include "core/vector_state.hpp"

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

// ── CPU and encoding helpers ───────────────────────────────────────────

static CPU make_cpu()
{
    return CPU(std::make_shared<FlatMemory>(0, 0x10000));
}

static constexpr u32 VSETVLI(u32 rd, u32 rs1, u32 zimm)
{
    return (0u << 31) | ((zimm & 0x7FF) << 20) | (rs1 << 15)
         | (0b111u << 12) | (rd << 7) | 0b1010111u;
}
static constexpr u32 VSETIVLI(u32 rd, u32 uimm, u32 zimm)
{
    return (0b11u << 30) | ((zimm & 0x3FF) << 20) | (uimm << 15)
         | (0b111u << 12) | (rd << 7) | 0b1010111u;
}
static constexpr u32 VLE32(u32 vd, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vd << 7) | 0b0000111u;
}
static constexpr u32 VSE32(u32 vs3, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vs3 << 7) | 0b0100111u;
}

static constexpr u32 enc_vv(u32 funct6, u32 vd, u32 vs1, u32 vs2)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b000u << 12) | (vd << 7) | 0b1010111u;
}
static constexpr u32 enc_vx(u32 funct6, u32 vd, u32 rs1, u32 vs2)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}
static constexpr u32 enc_mvv(u32 funct6, u32 vd, u32 vs1, u32 vs2)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b010u << 12) | (vd << 7) | 0b1010111u;
}

static constexpr u32 VADD_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b000000, vd, vs1, vs2); }
static constexpr u32 VSUB_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b000010, vd, vs1, vs2); }
static constexpr u32 VAND_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b001001, vd, vs1, vs2); }
static constexpr u32 VOR_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b001010, vd, vs1, vs2); }
static constexpr u32 VXOR_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b001011, vd, vs1, vs2); }
static constexpr u32 VMSEQ_VV(u32 vd, u32 vs2, u32 vs1) { return enc_vv(0b011000, vd, vs1, vs2); }
static constexpr u32 VMSLT_VV(u32 vd, u32 vs2, u32 vs1) { return enc_vv(0b011011, vd, vs1, vs2); }
static constexpr u32 VMSLTU_VV(u32 vd, u32 vs2, u32 vs1){ return enc_vv(0b011010, vd, vs1, vs2); }

static constexpr u32 VADD_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b000000, vd, rs1, vs2); }
static constexpr u32 VSUB_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b000010, vd, rs1, vs2); }
static constexpr u32 VAND_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b001001, vd, rs1, vs2); }
static constexpr u32 VOR_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b001010, vd, rs1, vs2); }
static constexpr u32 VXOR_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b001011, vd, rs1, vs2); }
static constexpr u32 VSLL_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b100101, vd, rs1, vs2); }
static constexpr u32 VSRL_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b101000, vd, rs1, vs2); }
static constexpr u32 VMSEQ_VX(u32 vd, u32 vs2, u32 rs1) { return enc_vx(0b011000, vd, rs1, vs2); }
static constexpr u32 VMV_V_X(u32 vd, u32 rs1)
{
    return (0b010111u << 26) | (1u << 25) | (0u << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}

static constexpr u32 VREDSUM(u32 vd, u32 vs2, u32 vs1)  { return enc_mvv(0b000000, vd, vs1, vs2); }
static constexpr u32 VMAND(u32 vd, u32 vs2, u32 vs1)    { return enc_mvv(0b011001, vd, vs1, vs2); }
static constexpr u32 VMOR(u32 vd, u32 vs2, u32 vs1)     { return enc_mvv(0b011010, vd, vs1, vs2); }
static constexpr u32 VMNOT(u32 vd, u32 vs)              { return enc_mvv(0b011110, vd, vs, vs); }
static constexpr u32 VMV_X_S(u32 rd, u32 vs2)           { return enc_mvv(0b010000, rd, 0, vs2); }

static constexpr u32 EBREAK = 0x00100073;
static constexpr u32 VTYPE_SEW32 = 0b00000010000;

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Decoder — every Op decodes correctly
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_decode_vsetvli)   { ASSERT_EQ(Decoder::decode(VSETVLI(1, 2, VTYPE_SEW32)).op,  Op::VSETVLI);    return true; }
TEST(v_decode_vsetivli)  { ASSERT_EQ(Decoder::decode(VSETIVLI(1, 4, VTYPE_SEW32)).op, Op::VSETIVLI);   return true; }
TEST(v_decode_vle32)     { ASSERT_EQ(Decoder::decode(VLE32(2, 10)).op,                Op::VLE32);      return true; }
TEST(v_decode_vse32)     { ASSERT_EQ(Decoder::decode(VSE32(2, 10)).op,                Op::VSE32);      return true; }
TEST(v_decode_vadd_vv)   { ASSERT_EQ(Decoder::decode(VADD_VV(4, 2, 3)).op,            Op::VADD_VV);    return true; }
TEST(v_decode_vsub_vv)   { ASSERT_EQ(Decoder::decode(VSUB_VV(4, 2, 3)).op,            Op::VSUB_VV);    return true; }
TEST(v_decode_vand_vv)   { ASSERT_EQ(Decoder::decode(VAND_VV(4, 2, 3)).op,            Op::VAND_VV);    return true; }
TEST(v_decode_vor_vv)    { ASSERT_EQ(Decoder::decode(VOR_VV(4, 2, 3)).op,             Op::VOR_VV);     return true; }
TEST(v_decode_vxor_vv)   { ASSERT_EQ(Decoder::decode(VXOR_VV(4, 2, 3)).op,            Op::VXOR_VV);    return true; }
TEST(v_decode_vadd_vx)   { ASSERT_EQ(Decoder::decode(VADD_VX(4, 5, 3)).op,            Op::VADD_VX);    return true; }
TEST(v_decode_vsub_vx)   { ASSERT_EQ(Decoder::decode(VSUB_VX(4, 5, 3)).op,            Op::VSUB_VX);    return true; }
TEST(v_decode_vand_vx)   { ASSERT_EQ(Decoder::decode(VAND_VX(4, 5, 3)).op,            Op::VAND_VX);    return true; }
TEST(v_decode_vor_vx)    { ASSERT_EQ(Decoder::decode(VOR_VX(4, 5, 3)).op,             Op::VOR_VX);     return true; }
TEST(v_decode_vxor_vx)   { ASSERT_EQ(Decoder::decode(VXOR_VX(4, 5, 3)).op,            Op::VXOR_VX);    return true; }
TEST(v_decode_vsll_vx)   { ASSERT_EQ(Decoder::decode(VSLL_VX(4, 5, 3)).op,            Op::VSLL_VX);    return true; }
TEST(v_decode_vsrl_vx)   { ASSERT_EQ(Decoder::decode(VSRL_VX(4, 5, 3)).op,            Op::VSRL_VX);    return true; }
TEST(v_decode_vmseq_vv)  { ASSERT_EQ(Decoder::decode(VMSEQ_VV(0, 2, 3)).op,           Op::VMSEQ_VV);   return true; }
TEST(v_decode_vmseq_vx)  { ASSERT_EQ(Decoder::decode(VMSEQ_VX(0, 5, 3)).op,           Op::VMSEQ_VX);   return true; }
TEST(v_decode_vmslt_vv)  { ASSERT_EQ(Decoder::decode(VMSLT_VV(0, 2, 3)).op,           Op::VMSLT_VV);   return true; }
TEST(v_decode_vmsltu_vv) { ASSERT_EQ(Decoder::decode(VMSLTU_VV(0, 2, 3)).op,          Op::VMSLTU_VV);  return true; }
TEST(v_decode_vmand)     { ASSERT_EQ(Decoder::decode(VMAND(0, 1, 2)).op,              Op::VMAND_MM);   return true; }
TEST(v_decode_vmor)      { ASSERT_EQ(Decoder::decode(VMOR(0, 1, 2)).op,               Op::VMOR_MM);    return true; }
TEST(v_decode_vmnot)     { ASSERT_EQ(Decoder::decode(VMNOT(0, 1)).op,                 Op::VMNOT_M);    return true; }
TEST(v_decode_vredsum)   { ASSERT_EQ(Decoder::decode(VREDSUM(4, 3, 2)).op,            Op::VREDSUM_VS); return true; }
TEST(v_decode_vmv_v_x)   { ASSERT_EQ(Decoder::decode(VMV_V_X(2, 5)).op,               Op::VMV_V_X);    return true; }
TEST(v_decode_vmv_x_s)   { ASSERT_EQ(Decoder::decode(VMV_X_S(3, 2)).op,               Op::VMV_X_S);    return true; }

TEST(v_vsetvli_fields) {
    // Verify VSETVLI decodes rd, rs1, and zimm correctly.
    auto inst = Decoder::decode(VSETVLI(5, 10, VTYPE_SEW32));
    ASSERT_EQ(inst.op, Op::VSETVLI);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 10);
    ASSERT_EQ(inst.imm, static_cast<i32>(VTYPE_SEW32));
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. Decoder — invalid encodings
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_invalid_vl_funct3) {
    // VL opcode (0000111) with funct3 != 110 (not 32-bit width)
    u32 bad = (1u << 25) | (10u << 15) | (0b010u << 12) | (2u << 7) | 0b0000111u;
    ASSERT_EQ(Decoder::decode(bad).op, Op::INVALID);
    return true;
}

TEST(v_invalid_opv_funct6) {
    // OP-V with an unrecognised funct6
    u32 bad = enc_vv(0b111111, 4, 2, 3);
    ASSERT_EQ(Decoder::decode(bad).op, Op::INVALID);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Vector state / VSETVLI
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vsetvli_basic) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 4);  // AVL = 4
    // VSETVLI x2, x1, e32 -> rd gets vl
    cpu.load_instruction(0, VSETVLI(2, 1, VTYPE_SEW32));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(2), 4u);  // vl = min(AVL, VLMAX)
    return true;
}

TEST(v_vsetvli_clamps_to_vlmax) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 1000);  // AVL much larger than VLMAX
    cpu.load_instruction(0, VSETVLI(2, 1, VTYPE_SEW32));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    // Default VLEN=128, SEW=32 -> VLMAX=4.  vl clamped to 4.
    ASSERT_EQ(cpu.reg(2), 4u);
    return true;
}

TEST(v_vsetvli_invalid_sew) {
    auto cpu = make_cpu();
    cpu.set_reg(1, 4);
    // zimm with SEW=64 (vsew=011) — not supported in this implementation's subset
    u32 bad_vtype = 0b00000011000;
    cpu.load_instruction(0, VSETVLI(2, 1, bad_vtype));
    cpu.load_instruction(4, EBREAK);
    cpu.run(10);

    ASSERT_EQ(cpu.reg(2), 0u);  // vill set -> vl = 0
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Vector load / store
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vle32_vse32) {
    auto cpu = make_cpu();

    cpu.memory().write32(0x200, 10);
    cpu.memory().write32(0x204, 20);
    cpu.memory().write32(0x208, 30);
    cpu.memory().write32(0x20C, 40);

    cpu.set_reg(1, 4);       // AVL
    cpu.set_reg(10, 0x200);  // base address for load
    cpu.set_reg(11, 0x300);  // base address for store

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));  // set vl=4
    cpu.load_instruction(4,  VLE32(2, 10));                // v2 = mem[0x200..0x20F]
    cpu.load_instruction(8,  VSE32(2, 11));                // mem[0x300..0x30F] = v2
    cpu.load_instruction(12, EBREAK);
    cpu.run(20);

    ASSERT_EQ(cpu.memory().read32(0x300).value, 10u);
    ASSERT_EQ(cpu.memory().read32(0x304).value, 20u);
    ASSERT_EQ(cpu.memory().read32(0x308).value, 30u);
    ASSERT_EQ(cpu.memory().read32(0x30C).value, 40u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Arithmetic
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vadd_vsub_vv) {
    auto cpu = make_cpu();

    cpu.memory().write32(0x200, 1);
    cpu.memory().write32(0x204, 2);
    cpu.memory().write32(0x208, 3);
    cpu.memory().write32(0x20C, 4);
    cpu.memory().write32(0x300, 10);
    cpu.memory().write32(0x304, 20);
    cpu.memory().write32(0x308, 30);
    cpu.memory().write32(0x30C, 40);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);  // vadd output
    cpu.set_reg(13, 0x500);  // vsub output

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));      // v2 = [1,2,3,4]
    cpu.load_instruction(8,  VLE32(3, 11));      // v3 = [10,20,30,40]
    cpu.load_instruction(12, VADD_VV(4, 3, 2));  // v4 = v3 + v2 = [11,22,33,44]
    cpu.load_instruction(16, VSE32(4, 12));      // store v4
    cpu.load_instruction(20, VSUB_VV(5, 4, 2));  // v5 = v4 - v2 = [10,20,30,40]
    cpu.load_instruction(24, VSE32(5, 13));      // store v5
    cpu.load_instruction(28, EBREAK);
    cpu.run(30);

    ASSERT_EQ(cpu.memory().read32(0x400).value, 11u);
    ASSERT_EQ(cpu.memory().read32(0x404).value, 22u);
    ASSERT_EQ(cpu.memory().read32(0x408).value, 33u);
    ASSERT_EQ(cpu.memory().read32(0x40C).value, 44u);
    ASSERT_EQ(cpu.memory().read32(0x500).value, 10u);
    ASSERT_EQ(cpu.memory().read32(0x504).value, 20u);
    ASSERT_EQ(cpu.memory().read32(0x508).value, 30u);
    ASSERT_EQ(cpu.memory().read32(0x50C).value, 40u);
    return true;
}

TEST(v_vadd_vsub_vx) {
    auto cpu = make_cpu();

    cpu.memory().write32(0x200, 10);
    cpu.memory().write32(0x204, 20);
    cpu.memory().write32(0x208, 30);
    cpu.memory().write32(0x20C, 40);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);
    cpu.set_reg(5, 100);  // scalar addend/subtrahend

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));      // v2 = [10,20,30,40]
    cpu.load_instruction(8,  VADD_VX(3, 2, 5));  // v3 = v2 + x5 = [110,120,130,140]
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, VSUB_VX(4, 3, 5));  // v4 = v3 - x5 = [10,20,30,40]
    cpu.load_instruction(20, VSE32(4, 12));
    cpu.load_instruction(24, EBREAK);
    cpu.run(30);

    ASSERT_EQ(cpu.memory().read32(0x300).value, 110u);
    ASSERT_EQ(cpu.memory().read32(0x304).value, 120u);
    ASSERT_EQ(cpu.memory().read32(0x308).value, 130u);
    ASSERT_EQ(cpu.memory().read32(0x30C).value, 140u);
    ASSERT_EQ(cpu.memory().read32(0x400).value, 10u);
    ASSERT_EQ(cpu.memory().read32(0x404).value, 20u);
    ASSERT_EQ(cpu.memory().read32(0x408).value, 30u);
    ASSERT_EQ(cpu.memory().read32(0x40C).value, 40u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Bitwise — VAND, VOR, VXOR, VSLL, VSRL
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vand_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00FF00);
    cpu.memory().write32(0x204, 0x0F0F0F0F);
    cpu.memory().write32(0x208, 0xAAAAAAAA);
    cpu.memory().write32(0x20C, 0x55555555);
    cpu.memory().write32(0x300, 0x0F0F0F0F);
    cpu.memory().write32(0x304, 0xF0F0F0F0);
    cpu.memory().write32(0x308, 0xFFFFFFFF);
    cpu.memory().write32(0x30C, 0x00000000);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VLE32(3, 11));
    cpu.load_instruction(12, VAND_VV(4, 3, 2));
    cpu.load_instruction(16, VSE32(4, 12));
    cpu.load_instruction(20, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x400).value, 0x0F000F00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x404).value, 0x00000000u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x408).value, 0xAAAAAAAAu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x40C).value, 0x00000000u);
    return true;
}

TEST(v_vor_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xF000F000);
    cpu.memory().write32(0x204, 0x00000000);
    cpu.memory().write32(0x300, 0x0F0F0F0F);
    cpu.memory().write32(0x304, 0x00000000);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VLE32(3, 11));
    cpu.load_instruction(12, VOR_VV(4, 3, 2));
    cpu.load_instruction(16, VSE32(4, 12));
    cpu.load_instruction(20, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x400).value, 0xFF0FFF0Fu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x404).value, 0x00000000u);
    return true;
}

TEST(v_vxor_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00FF00);
    cpu.memory().write32(0x204, 0xAAAAAAAA);
    cpu.memory().write32(0x300, 0x0F0F0F0F);
    cpu.memory().write32(0x304, 0xAAAAAAAA);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VLE32(3, 11));
    cpu.load_instruction(12, VXOR_VV(4, 3, 2));
    cpu.load_instruction(16, VSE32(4, 12));
    cpu.load_instruction(20, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x400).value, 0xF00FF00Fu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x404).value, 0x00000000u);
    return true;
}

TEST(v_vand_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xDEADBEEF);
    cpu.memory().write32(0x204, 0x12345678);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0x0000FFFFu);  // mask low 16 bits

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VAND_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0x0000BEEFu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x00005678u);
    return true;
}

TEST(v_vor_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x00000000);
    cpu.memory().write32(0x204, 0xF0F0F0F0);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0x0F0F0F0Fu);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VOR_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0x0F0F0F0Fu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0xFFFFFFFFu);
    return true;
}

TEST(v_vxor_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x00000000);
    cpu.memory().write32(0x204, 0xF0F0F0F0);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0xF0F0F0F0u);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VXOR_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0xF0F0F0F0u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x00000000u);
    return true;
}

TEST(v_vsll_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 1);
    cpu.memory().write32(0x204, 0x80);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 4);  // shift by 4

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VSLL_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_EQ(cpu.memory().read32(0x300).value, 16u);    // 1 << 4
    ASSERT_EQ(cpu.memory().read32(0x304).value, 0x800u); // 0x80 << 4
    return true;
}

TEST(v_vsrl_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x100);
    cpu.memory().write32(0x204, 0xFFFFFFFF);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 8);  // shift by 8

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VSRL_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_EQ(cpu.memory().read32(0x300).value, 1u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x00FFFFFFu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. Comparison — mask output
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vmseq_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 10);
    cpu.memory().write32(0x204, 20);
    cpu.memory().write32(0x208, 30);
    cpu.memory().write32(0x20C, 40);
    cpu.memory().write32(0x300, 10);
    cpu.memory().write32(0x304, 99);
    cpu.memory().write32(0x308, 30);
    cpu.memory().write32(0x30C, 99);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));
    cpu.load_instruction(0x08, VLE32(3, 11));
    cpu.load_instruction(0x0C, VMSEQ_VV(0, 2, 3));
    cpu.load_instruction(0x10, VMV_X_S(5, 0));  // x5 = v0[0]
    cpu.load_instruction(0x14, EBREAK);
    cpu.run(30);

    // v0[0] should have mask bits: bit0=1, bit1=0, bit2=1, bit3=0 -> 0x5
    ASSERT_HEX_EQ(cpu.reg(5) & 0xF, 0x5u);
    return true;
}

TEST(v_vmseq_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 42);
    cpu.memory().write32(0x204, 99);
    cpu.memory().write32(0x208, 42);
    cpu.memory().write32(0x20C, 0);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(5, 42);  // scalar comparand

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));
    cpu.load_instruction(0x08, VMSEQ_VX(0, 2, 5));
    cpu.load_instruction(0x0C, VMV_X_S(6, 0));
    cpu.load_instruction(0x10, EBREAK);
    cpu.run(30);

    // elem0=42==42->1, elem1=99==42->0, elem2=42==42->1, elem3=0==42->0 -> 0b0101
    ASSERT_HEX_EQ(cpu.reg(6) & 0xF, 0x5u);
    return true;
}

TEST(v_vmslt_vv_signed) {
    auto cpu = make_cpu();
    // Compare [−10, 5, 0, 100] < [0, 5, 1, −1]
    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.memory().write32(0x204, 5);
    cpu.memory().write32(0x208, 0);
    cpu.memory().write32(0x20C, 100);
    cpu.memory().write32(0x300, 0);
    cpu.memory().write32(0x304, 5);
    cpu.memory().write32(0x308, 1);
    cpu.memory().write32(0x30C, static_cast<u32>(-1));

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));   // vs2 = [−10, 5, 0, 100]
    cpu.load_instruction(0x08, VLE32(3, 11));   // vs1 = [0, 5, 1, −1]
    cpu.load_instruction(0x0C, VMSLT_VV(0, 2, 3));
    cpu.load_instruction(0x10, VMV_X_S(5, 0));
    cpu.load_instruction(0x14, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.reg(5) & 0xF, 0x5u);
    return true;
}

TEST(v_vmsltu_vv_unsigned) {
    // Compare [5, 0xFFFFFFF6, 10, 0] < [10, 5, 10, 1]
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 5);
    cpu.memory().write32(0x204, 0xFFFFFFF6);
    cpu.memory().write32(0x208, 10);
    cpu.memory().write32(0x20C, 0);
    cpu.memory().write32(0x300, 10);
    cpu.memory().write32(0x304, 5);
    cpu.memory().write32(0x308, 10);
    cpu.memory().write32(0x30C, 1);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));
    cpu.load_instruction(0x08, VLE32(3, 11));
    cpu.load_instruction(0x0C, VMSLTU_VV(0, 2, 3));
    cpu.load_instruction(0x10, VMV_X_S(5, 0));
    cpu.load_instruction(0x14, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.reg(5) & 0xF, 0x9u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. Move and reduction
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vmv_v_x_splat) {
    auto cpu = make_cpu();

    cpu.set_reg(1, 4);
    cpu.set_reg(5, 42);
    cpu.set_reg(10, 0x200);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32)); 
    cpu.load_instruction(4,  VMV_V_X(2, 5));  // v2 = [42,42,42,42]
    cpu.load_instruction(8,  VSE32(2, 10));
    cpu.load_instruction(12, EBREAK);
    cpu.run(20);

    ASSERT_EQ(cpu.memory().read32(0x200).value, 42u);
    ASSERT_EQ(cpu.memory().read32(0x204).value, 42u);
    ASSERT_EQ(cpu.memory().read32(0x208).value, 42u);
    ASSERT_EQ(cpu.memory().read32(0x20C).value, 42u);
    return true;
}

TEST(v_vmv_x_s_extract) {
    auto cpu = make_cpu();

    cpu.memory().write32(0x200, 99);
    cpu.memory().write32(0x204, 88);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));   // v2 = [99, 88]
    cpu.load_instruction(8,  VMV_X_S(3, 2));  // x3 = v2[0] = 99
    cpu.load_instruction(12, EBREAK);
    cpu.run(20);

    ASSERT_EQ(cpu.reg(3), 99u);
    return true;
}

TEST(v_vredsum) {
    auto cpu = make_cpu();

    // v2 = [1, 2, 3, 4], v3[0] = 0 (initial accumulator)
    cpu.memory().write32(0x200, 1);
    cpu.memory().write32(0x204, 2);
    cpu.memory().write32(0x208, 3);
    cpu.memory().write32(0x20C, 4);
    cpu.memory().write32(0x300, 0);  // scalar init

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));      // v2 = [1,2,3,4]
    cpu.load_instruction(8,  VLE32(3, 11));      // v3[0] = 0
    cpu.load_instruction(12, VREDSUM(4, 2, 3));  // v4[0] = 0 + 1+2+3+4 = 10
    cpu.load_instruction(16, VMV_X_S(5, 4));     // x5 = v4[0] = 10
    cpu.load_instruction(20, EBREAK);
    cpu.run(30);

    ASSERT_EQ(cpu.reg(5), 10u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  9. Program — vectorised sum (checksum-like pattern)
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_checksum_pattern) {
    // Sum 16 words using vector operations.
    // With VLEN=128 (4 elements), this requires 4 iterations of vle32+vadd.
    //
    // Setup: mem[0x400..0x43F] = 1..16
    // Result: sum = 1+2+...+16 = 136

    auto cpu = make_cpu();
    for (u32 i = 0; i < 16; ++i)
        cpu.memory().write32(0x400 + i * 4, i + 1);
    cpu.memory().write32(0x500, 0);

    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x400);
    cpu.set_reg(11, 4);
    cpu.set_reg(12, 0x500);

    // 0x00: vsetvli x0, x1, e32
    // 0x04: vmv.v.x v4, x0         # accumulator = 0
    // 0x08: vle32.v v2, (x10)      # load chunk
    // 0x0C: vadd.vv v4, v2, v4     # accumulate
    // 0x10: addi x10, x10, 16
    // 0x14: addi x11, x11, -1
    // 0x18: bne x11, x0, -16       # -> 0x08
    // 0x1C: vle32.v v6, (x12)      # v6[0] = 0 (reduction init)
    // 0x20: vredsum.vs v5, v6, v4  # v5[0] = sum(v4)
    // 0x24: vmv.x.s x3, v5
    // 0x28: ebreak

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VMV_V_X(4, 0));
    cpu.load_instruction(0x08, VLE32(2, 10));
    cpu.load_instruction(0x0C, VADD_VV(4, 4, 2));
    cpu.load_instruction(0x10, 0x01050513);        // addi x10, x10, 16
    cpu.load_instruction(0x14, 0xfff58593);        // addi x11, x11, -1
    cpu.load_instruction(0x18, 0xfe0598e3);        // bne x11, x0, -16 -> 0x08
    cpu.load_instruction(0x1C, VLE32(6, 12));      // v6[0] = 0
    cpu.load_instruction(0x20, VREDSUM(5, 4, 6));  // v5[0] = sum(v4)
    cpu.load_instruction(0x24, VMV_X_S(3, 5));
    cpu.load_instruction(0x28, EBREAK);

    cpu.run(200);

    ASSERT_EQ(cpu.reg(3), 136u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  10. Pipeline — basic RVV operations
 * ═══════════════════════════════════════════════════════════════════════ */

static void pipe_run(PipelinedCPU& cpu, cycle_t max = 200)
{
    cycle_t n = 0;
    while (n < max) { if (!cpu.tick()) break; ++n; }
}

TEST(v_pipe_vsetvli) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.load_instruction(0, VSETVLI(2, 1, VTYPE_SEW32));
    cpu.load_instruction(4, EBREAK);
    pipe_run(cpu);
    ASSERT_EQ(cpu.reg(2), 4u);
    return true;
}

TEST(v_pipe_vle32_vse32) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->write32(0x200, 10);
    mem->write32(0x204, 20);
    mem->write32(0x208, 30);
    mem->write32(0x20C, 40);

    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));
    cpu.load_instruction(0x08, VSE32(2, 11));
    cpu.load_instruction(0x0C, EBREAK);
    pipe_run(cpu);

    ASSERT_EQ(mem->read32(0x300).value, 10u);
    ASSERT_EQ(mem->read32(0x304).value, 20u);
    ASSERT_EQ(mem->read32(0x308).value, 30u);
    ASSERT_EQ(mem->read32(0x30C).value, 40u);
    return true;
}

TEST(v_pipe_vadd_vv) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    for (u32 i = 0; i < 4; ++i) {
        mem->write32(0x200 + i*4, i + 1);     // [1,2,3,4]
        mem->write32(0x300 + i*4, (i+1)*10);  // [10,20,30,40]
    }
 
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0x400);
 
    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));
    cpu.load_instruction(0x08, VLE32(3, 11));
    cpu.load_instruction(0x0C, VADD_VV(4, 3, 2));
    cpu.load_instruction(0x10, VSE32(4, 12));
    cpu.load_instruction(0x14, EBREAK);
    pipe_run(cpu);
 
    ASSERT_EQ(mem->read32(0x400).value, 11u);
    ASSERT_EQ(mem->read32(0x404).value, 22u);
    ASSERT_EQ(mem->read32(0x408).value, 33u);
    ASSERT_EQ(mem->read32(0x40C).value, 44u);
    return true;
}

TEST(v_pipe_vmv_v_x_and_vmv_x_s) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.set_reg(5, 42);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VMV_V_X(2, 5));  // v2 = [42,42,42,42]
    cpu.load_instruction(0x08, VMV_X_S(3, 2));  // x3 = v2[0] = 42
    cpu.load_instruction(0x0C, EBREAK);
    pipe_run(cpu);

    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(vp_vredsum) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    for (u32 i = 0; i < 4; ++i)
        mem->write32(0x200 + i*4, i + 1);  // [1,2,3,4]
    mem->write32(0x300, 0);  // reduction init

    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VLE32(2, 10));      // v2 = [1,2,3,4]
    cpu.load_instruction(0x08, VLE32(3, 11));      // v3[0] = 0
    cpu.load_instruction(0x0C, VREDSUM(4, 2, 3));  // v4[0] = sum
    cpu.load_instruction(0x10, VMV_X_S(5, 4));     // x5 = v4[0]
    cpu.load_instruction(0x14, EBREAK);
    pipe_run(cpu);

    ASSERT_EQ(cpu.reg(5), 10u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  11. Pipeline — vectorised checksum
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_pipe_checksum) {
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    for (u32 i = 0; i < 16; ++i)
        mem->write32(0x400 + i * 4, i + 1);
    mem->write32(0x500, 0);

    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 4);
    cpu.set_reg(10, 0x400);
    cpu.set_reg(11, 4);
    cpu.set_reg(12, 0x500);

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VMV_V_X(4, 0));     // v4 = [0,0,0,0]
    cpu.load_instruction(0x08, VLE32(2, 10));      // v2 = mem[x10]
    cpu.load_instruction(0x0C, VADD_VV(4, 4, 2));  // v4 += v2
    cpu.load_instruction(0x10, 0x01050513);        // addi x10, x10, 16
    cpu.load_instruction(0x14, 0xfff58593);        // addi x11, x11, -1
    cpu.load_instruction(0x18, 0xfe0598e3);        // bne x11, x0, -16 -> 0x08
    cpu.load_instruction(0x1C, VLE32(6, 12));      // v6[0] = 0
    cpu.load_instruction(0x20, VREDSUM(5, 4, 6));  // v5[0] = sum(v4)
    cpu.load_instruction(0x24, VMV_X_S(3, 5));     // x3 = v5[0]
    cpu.load_instruction(0x28, EBREAK);

    pipe_run(cpu, 5000);

    ASSERT_EQ(cpu.reg(3), 136u);  // sum(1..16)
    return true;
}
