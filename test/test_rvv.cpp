/**
 * @file test_rvv.cpp
 * @brief Tests for the RVV subset.
 *
 * Sections:
 *  1 (line  161) : Decoder
 *  2 (line  466) : Vector state - VSETVLI configuration, VLMAX
 *  3 (line  507) : VLE32/VSE32 - vector load and store
 *  4 (line  536) : Arithmetic - VADD, VSUB
 *  5 (line  613) : Bitwise - VAND, VOR, VXOR, VSLL, VSRL
 *  6 (line  808) : Comparison - VMSEQ, VMSLT, VMSLTU
 *  7 (line  982) : Move and Reduction - VMV.V.X, VMV.X.S, VREDSUM
 *  8 (line 1051) : Program - Vectorized checksum
 *  9 (line 1102) : Pipeline - VSETVLI, VLE32/VSE32, VADD, VREDSUM, VMV
 * 10 (line 1215) : Pipeline - Vectorized checksum
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
static constexpr u32 VLE32(u32 vd, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vd << 7) | 0b0000111u;
}
static constexpr u32 VSE32(u32 vs3, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vs3 << 7) | 0b0100111u;
}

static constexpr u32 enc_vv(u32 funct6, u32 vd, u32 vs2, u32 vs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b000u << 12) | (vd << 7) | 0b1010111u;
}
static constexpr u32 enc_vx(u32 funct6, u32 vd, u32 vs2, u32 rs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}
static constexpr u32 enc_mvv(u32 funct6, u32 vd, u32 vs2, u32 vs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b010u << 12) | (vd << 7) | 0b1010111u;
}

static constexpr u32 VADD_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b000000u, vd, vs2, vs1); }
static constexpr u32 VSUB_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b000010u, vd, vs2, vs1); }
static constexpr u32 VAND_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b001001u, vd, vs2, vs1); }
static constexpr u32 VOR_VV(u32 vd, u32 vs2, u32 vs1)    { return enc_vv(0b001010u, vd, vs2, vs1); }
static constexpr u32 VXOR_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b001011u, vd, vs2, vs1); }
static constexpr u32 VMSEQ_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b011000u, vd, vs2, vs1); }
static constexpr u32 VMSLT_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b011011u, vd, vs2, vs1); }
static constexpr u32 VMSLTU_VV(u32 vd, u32 vs2, u32 vs1) { return enc_vv(0b011010u, vd, vs2, vs1); }

static constexpr u32 VADD_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b000000u, vd, vs2, rs1); }
static constexpr u32 VSUB_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b000010u, vd, vs2, rs1); }
static constexpr u32 VAND_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b001001u, vd, vs2, rs1); }
static constexpr u32 VOR_VX(u32 vd, u32 vs2, u32 rs1)    { return enc_vx(0b001010u, vd, vs2, rs1); }
static constexpr u32 VXOR_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b001011u, vd, vs2, rs1); }
static constexpr u32 VSLL_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b100101u, vd, vs2, rs1); }
static constexpr u32 VSRL_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b101000u, vd, vs2, rs1); }
static constexpr u32 VMSEQ_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b011000u, vd, vs2, rs1); }
static constexpr u32 VMV_V_X(u32 vd, u32 rs1)
{
    return (0b010111u << 26) | (1u << 25) | (0u << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}

static constexpr u32 VREDSUM(u32 vd, u32 vs2, u32 vs1)  { return enc_mvv(0b000000u, vd, vs2, vs1); }
static constexpr u32 VMAND(u32 vd, u32 vs2, u32 vs1)    { return enc_mvv(0b011001u, vd, vs2, vs1); }
static constexpr u32 VMNAND(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011101u, vd, vs2, vs1); }
static constexpr u32 VMANDN(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011000u, vd, vs2, vs1); }
static constexpr u32 VMOR(u32 vd, u32 vs2, u32 vs1)     { return enc_mvv(0b011010u, vd, vs2, vs1); }
static constexpr u32 VMNOR(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011110u, vd, vs2, vs1); }
static constexpr u32 VMORN(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011100u, vd, vs2, vs1); }
static constexpr u32 VMXOR(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011011u, vd, vs2, vs1); }
static constexpr u32 VMXNOR(u32 vd, u32 vs2, u32 vs1)	{ return enc_mvv(0b011111u, vd, vs2, vs1); }
static constexpr u32 VMV_X_S(u32 rd, u32 vs2)           { return enc_mvv(0b010000u, rd, vs2, 0); }

static constexpr u32 EBREAK = 0x0010'0073u;
static constexpr u32 VTYPE_SEW32 = 0b00000010000;

// ═══════════════════════════════════════════════════════════════════════
//  1. Decoder — all instructions decode correctly
// ═══════════════════════════════════════════════════════════════════════

// ── VSETVLI, VLE32, VSE32 ──────────────────────────────────────────────

TEST(v_decode_vsetvli) {
    auto inst = Decoder::decode(0x0100'f157u); // vsetvli x2, x1, 0x010
    ASSERT_EQ(inst.op, Op::VSETVLI);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.imm, 0x010);
    return true;
}

TEST(v_decode_vle32_v) {
    auto inst = Decoder::decode(0x0201'e207u); // vle32.v v4, (x3)
    ASSERT_EQ(inst.op, Op::VLE32);
    ASSERT_EQ(inst.rd, 4);
    ASSERT_EQ(inst.rs1, 3);
    return true;
}

TEST(v_decode_vse32_v) {
    auto inst = Decoder::decode(0x0202'e327u); // vse32.v v6, (x5)
    ASSERT_EQ(inst.op, Op::VSE32);
    ASSERT_EQ(inst.rd, 6);
    ASSERT_EQ(inst.rs1, 5);
    return true;
}

// ── OPIVV instructions ─────────────────────────────────────────────────

TEST(v_decode_vadd_vv) {
    auto inst = Decoder::decode(0x0221'80d7u); // vadd.vv v1, v2, v3
    ASSERT_EQ(inst.op, Op::VADD_VV);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.rs1, 3);
    return true;
}

TEST(v_decode_vsub_vv) {
    auto inst = Decoder::decode(0x0a84'83d7u); // vsub.vv v7, v8, v9
    ASSERT_EQ(inst.op, Op::VSUB_VV);
    ASSERT_EQ(inst.rd, 7);
    ASSERT_EQ(inst.rs2, 8);
    ASSERT_EQ(inst.rs1, 9);
    return true;
}

TEST(v_decode_vand_vv) {
    auto inst = Decoder::decode(0x26e7'86d7u); // vand.vv v13, v14, v15
    ASSERT_EQ(inst.op, Op::VAND_VV);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs2, 14);
    ASSERT_EQ(inst.rs1, 15);
    return true;
}

TEST(v_decode_vor_vv) {
    auto inst = Decoder::decode(0x2b4a'89d7u); // vor.vv v19, v20, v21
    ASSERT_EQ(inst.op, Op::VOR_VV);
    ASSERT_EQ(inst.rd, 19);
    ASSERT_EQ(inst.rs2, 20);
    ASSERT_EQ(inst.rs1, 21);
    return true;
}

TEST(v_decode_vxor_vv) {
    auto inst = Decoder::decode(0x2fad'8cd7u); // vxor.vv v25, v26, v27
    ASSERT_EQ(inst.op, Op::VXOR_VV);
    ASSERT_EQ(inst.rd, 25);
    ASSERT_EQ(inst.rs2, 26);
    ASSERT_EQ(inst.rs1, 27);
    return true;
}

TEST(v_decode_vmseq_vv) {
    auto inst = Decoder::decode(0x6263'82d7u); // vmseq.vv v5, v6, v7
    ASSERT_EQ(inst.op, Op::VMSEQ_VV);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs2, 6);
    ASSERT_EQ(inst.rs1, 7);
    return true;
}

TEST(v_decode_vmslt_vv) {
    auto inst = Decoder::decode(0x6ec6'85d7u); // vmslt.vv v11, v12, v13
    ASSERT_EQ(inst.op, Op::VMSLT_VV);
    ASSERT_EQ(inst.rd, 11);
    ASSERT_EQ(inst.rs2, 12);
    ASSERT_EQ(inst.rs1, 13);
    return true;
}

TEST(v_decode_vmsltu_vv) {
    auto inst = Decoder::decode(0x6af8'0757u); // vmsltu.vv v14, v15, v16
    ASSERT_EQ(inst.op, Op::VMSLTU_VV);
    ASSERT_EQ(inst.rd, 14);
    ASSERT_EQ(inst.rs2, 15);
    ASSERT_EQ(inst.rs1, 16);
    return true;
}

// ── OPIVX instructions ─────────────────────────────────────────────────

TEST(v_decode_vadd_vx) {
    auto inst = Decoder::decode(0x0253'4257u); // vadd.vx v4, v5, v6
    ASSERT_EQ(inst.op, Op::VADD_VX);
    ASSERT_EQ(inst.rd, 4);
    ASSERT_EQ(inst.rs2, 5);
    ASSERT_EQ(inst.rs1, 6);
    return true;
}

TEST(v_decode_vsub_vx) {
    auto inst = Decoder::decode(0x0ab6'4557u); // vsub.vx v10, v11, v12
    ASSERT_EQ(inst.op, Op::VSUB_VX);
    ASSERT_EQ(inst.rd, 10);
    ASSERT_EQ(inst.rs2, 11);
    ASSERT_EQ(inst.rs1, 12);
    return true;
}

TEST(v_decode_vand_vx) {
    auto inst = Decoder::decode(0x2719'4857u); // vand.vx v16, v17, v18
    ASSERT_EQ(inst.op, Op::VAND_VX);
    ASSERT_EQ(inst.rd, 16);
    ASSERT_EQ(inst.rs2, 17);
    ASSERT_EQ(inst.rs1, 18);
    return true;
}

TEST(v_decode_vor_vx) {
    auto inst = Decoder::decode(0x2b7c'4b57u); // vor.vv v22, v23, v24
    ASSERT_EQ(inst.op, Op::VOR_VX);
    ASSERT_EQ(inst.rd, 22);
    ASSERT_EQ(inst.rs2, 23);
    ASSERT_EQ(inst.rs1, 24);
    return true;
}

TEST(v_decode_vxor_vx) {
    auto inst = Decoder::decode(0x2fdf'4e57u); // vxor.vx v28, v29, v30
    ASSERT_EQ(inst.op, Op::VXOR_VX);
    ASSERT_EQ(inst.rd, 28);
    ASSERT_EQ(inst.rs2, 29);
    ASSERT_EQ(inst.rs1, 30);
    return true;
}

TEST(v_decode_vsll_vx) {
    auto inst = Decoder::decode(0x9600'cfd7u); // vsll.vx v31, v0, x1
    ASSERT_EQ(inst.op, Op::VSLL_VX);
    ASSERT_EQ(inst.rd, 31);
    ASSERT_EQ(inst.rs2, 0);
    ASSERT_EQ(inst.rs1, 1);
    return true;
}

TEST(v_decode_vsrl_vx) {
    auto inst = Decoder::decode(0xa232'4157u); // vsrl.vx v2, v3, x4
    ASSERT_EQ(inst.op, Op::VSRL_VX);
    ASSERT_EQ(inst.rd, 2);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.rs1, 4);
    return true;
}

TEST(v_decode_vmseq_vx) {
    auto inst = Decoder::decode(0x6295'4457u); // vmseq.vx v8, v9, v10
    ASSERT_EQ(inst.op, Op::VMSEQ_VX);
    ASSERT_EQ(inst.rd, 8);
    ASSERT_EQ(inst.rs2, 9);
    ASSERT_EQ(inst.rs1, 10);
    return true;
}

TEST(v_decode_vmv_v_x) {
    auto inst = Decoder::decode(0x5e06'c657u); // vmv.v.x v12, x13
    ASSERT_EQ(inst.op, Op::VMV_V_X);
    ASSERT_EQ(inst.rd, 12);
    ASSERT_EQ(inst.rs1, 13);
    return true;
}

TEST(v_decode_vmv_x_s) {
    auto inst = Decoder::decode(0x42f0'2757u); // vmv.x.s x14, v15
    ASSERT_EQ(inst.op, Op::VMV_X_S);
    ASSERT_EQ(inst.rd, 14);
    ASSERT_EQ(inst.rs2, 15);
    return true;
}

// ── OPMVV instructions ─────────────────────────────────────────────────

TEST(v_decode_vmand_mm) {
    auto inst = Decoder::decode(0x6729'a8d7u); // vmand.mm v17, v18, v19
    ASSERT_EQ(inst.op, Op::VMAND_MM);
    ASSERT_EQ(inst.rd, 17);
    ASSERT_EQ(inst.rs2, 18);
    ASSERT_EQ(inst.rs1, 19);
    return true;
}

TEST(v_decode_vmnand_mm) {
    auto inst = Decoder::decode(0x775b'2a57u); // vmnand.mm v20, v21, v22
    ASSERT_EQ(inst.op, Op::VMNAND_MM);
    ASSERT_EQ(inst.rd, 20);
    ASSERT_EQ(inst.rs2, 21);
    ASSERT_EQ(inst.rs1, 22);
    return true;
}

TEST(v_decode_vmandn_mm) {
    auto inst = Decoder::decode(0x638c'abd7u); // vmandn.mm v23, v24, v25
    ASSERT_EQ(inst.op, Op::VMANDN_MM);
    ASSERT_EQ(inst.rd, 23);
    ASSERT_EQ(inst.rs2, 24);
    ASSERT_EQ(inst.rs1, 25);
    return true;
}

TEST(v_decode_vmxor_mm) {
    auto inst = Decoder::decode(0x6fbe'2d57u); // vmxor.mm v26, v27, v28
    ASSERT_EQ(inst.op, Op::VMXOR_MM);
    ASSERT_EQ(inst.rd, 26);
    ASSERT_EQ(inst.rs2, 27);
    ASSERT_EQ(inst.rs1, 28);
    return true;
}

TEST(v_decode_vmor_mm) {
    auto inst = Decoder::decode(0x6bef'aed7u); // vmor.mm v29, v30, v31
    ASSERT_EQ(inst.op, Op::VMOR_MM);
    ASSERT_EQ(inst.rd, 29);
    ASSERT_EQ(inst.rs2, 30);
    ASSERT_EQ(inst.rs1, 31);
    return true;
}

TEST(v_decode_vmnor_mm) {
    auto inst = Decoder::decode(0x7a11'2057u); // vmnor.mm v0, v1, v2
    ASSERT_EQ(inst.op, Op::VMNOR_MM);
    ASSERT_EQ(inst.rd, 0);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.rs1, 2);
    return true;
}

TEST(v_decode_vmorn_mm) {
    auto inst = Decoder::decode(0x7242'a1d7u); // vmorn.mm v3, v4, v5
    ASSERT_EQ(inst.op, Op::VMORN_MM);
    ASSERT_EQ(inst.rd, 3);
    ASSERT_EQ(inst.rs2, 4);
    ASSERT_EQ(inst.rs1, 5);
    return true;
}

TEST(v_decode_vmxnor_mm) {
    auto inst = Decoder::decode(0x7e74'2357u); // vmxnor.mm v6, v7, v8
    ASSERT_EQ(inst.op, Op::VMXNOR_MM);
    ASSERT_EQ(inst.rd, 6);
    ASSERT_EQ(inst.rs2, 7);
    ASSERT_EQ(inst.rs1, 8);
    return true;
}

// ── VREDSUM (OPIVS) ────────────────────────────────────────────────────

TEST(v_decode_vredsum_vs) {
    auto inst = Decoder::decode(0x02a5'a4d7u); // vredsum.vs v9, v10, v11
    ASSERT_EQ(inst.op, Op::VREDSUM_VS);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs2, 10);
    ASSERT_EQ(inst.rs1, 11);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(v_invalid_vl_funct3) {
    // VL opcode (0000111) with funct3 != 110 (not 32-bit width)
    u32 bad_enc = (1u << 25) | (10u << 15) | (0b010u << 12) | (2u << 7) | 0b0000111u;
    ASSERT_EQ(Decoder::decode(bad_enc).op, Op::INVALID);
    return true;
}

TEST(v_invalid_opv_funct6) {
    // OP-V with an unrecognised funct6
    u32 bad_enc = enc_vv(0b111111, 4, 2, 3);
    ASSERT_EQ(Decoder::decode(bad_enc).op, Op::INVALID);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. Vector state / VSETVLI
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
 *  3. Vector load / store
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
 *  4. Arithmetic
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
 *  5. Bitwise — VAND, VOR, VXOR, VSLL, VSRL
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(v_vand_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00'FF00);
    cpu.memory().write32(0x204, 0x0F0F'0F0F);
    cpu.memory().write32(0x208, 0xAAAA'AAAA);
    cpu.memory().write32(0x20C, 0x5555'5555);
    cpu.memory().write32(0x300, 0x0F0F'0F0F);
    cpu.memory().write32(0x304, 0xF0F0'F0F0);
    cpu.memory().write32(0x308, 0xFFFF'FFFF);
    cpu.memory().write32(0x30C, 0x0000'0000);

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

    ASSERT_HEX_EQ(cpu.memory().read32(0x400).value, 0x0F00'0F00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x404).value, 0x0000'0000u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x408).value, 0xAAAA'AAAAu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x40C).value, 0x0000'0000u);
    return true;
}

TEST(v_vor_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xF000'F000);
    cpu.memory().write32(0x204, 0x0000'0000);
    cpu.memory().write32(0x300, 0x0F0F'0F0F);
    cpu.memory().write32(0x304, 0x0000'0000);

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

    ASSERT_HEX_EQ(cpu.memory().read32(0x400).value, 0xFF0F'FF0Fu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x404).value, 0x0000'0000u);
    return true;
}

TEST(v_vxor_vv) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0xFF00'FF00);
    cpu.memory().write32(0x204, 0xAAAA'AAAA);
    cpu.memory().write32(0x300, 0x0F0F'0F0F);
    cpu.memory().write32(0x304, 0xAAAA'AAAA);

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
    cpu.memory().write32(0x200, 0xDEAD'BEEF);
    cpu.memory().write32(0x204, 0x1234'5678);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0x0000'FFFFu);  // mask low 16 bits

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VAND_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0x0000'BEEFu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x0000'5678u);
    return true;
}

TEST(v_vor_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x0000'0000);
    cpu.memory().write32(0x204, 0xF0F0'F0F0);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0x0F0F'0F0Fu);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VOR_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0x0F0F'0F0Fu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0xFFFF'FFFFu);
    return true;
}

TEST(v_vxor_vx) {
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0x0000'0000);
    cpu.memory().write32(0x204, 0xF0F0'F0F0);

    cpu.set_reg(1, 2);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(5, 0xF0F0'F0F0u);

    cpu.load_instruction(0,  VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(4,  VLE32(2, 10));
    cpu.load_instruction(8,  VXOR_VX(3, 2, 5));
    cpu.load_instruction(12, VSE32(3, 11));
    cpu.load_instruction(16, EBREAK);
    cpu.run(30);

    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0xF0F0'F0F0u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x0000'0000u);
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
    cpu.memory().write32(0x204, 0xFFFF'FFFF);

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
    ASSERT_HEX_EQ(cpu.memory().read32(0x304).value, 0x00FF'FFFFu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Comparison — mask output
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
 *  8. Vector mask-register logical instructions
 * ═══════════════════════════════════════════════════════════════════════ */

static u32 run_mask_op(u32 mask_instruction)
{
    auto cpu = make_cpu();
    
    cpu.set_reg(1, 4);
    cpu.set_reg(5, 0xA);  // v1 = 0b1010
    cpu.set_reg(6, 0xC);  // v2 = 0b1100

    cpu.load_instruction(0x00, VSETVLI(0, 1, VTYPE_SEW32));
    cpu.load_instruction(0x04, VMV_V_X(1, 5));     // v1 = splat(0xA)
    cpu.load_instruction(0x08, VMV_V_X(2, 6));     // v2 = splat(0xC)
    cpu.load_instruction(0x0C, mask_instruction);  // v0 = v1 OP v2
    cpu.load_instruction(0x10, VMV_X_S(5, 0));     // x5 = v0[0]
    cpu.load_instruction(0x14, EBREAK);

    cpu.run(20);
    return cpu.reg(5) & 0xF;
}

TEST(v_vmand_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMAND(0, 1, 2)), 0x8u);  // 0b1000
    return true;
}

TEST(v_vmnand_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMNAND(0, 1, 2)), 0x7u);  // 0b0111
    return true;
}

TEST(v_vmandn_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMANDN(0, 1, 2)), 0x2u);  // 0b0010
    return true;
}

TEST(v_vmxor_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMXOR(0, 1, 2)), 0x6u);  // 0b0110
    return true;
}

TEST(v_vmor_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMOR(0, 1, 2)), 0xEu);  // 0b1110
    return true;
}

TEST(v_vmnor_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMNOR(0, 1, 2)), 0x1u); // 0b0001
    return true;
}

TEST(v_vmorn_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMORN(0, 1, 2)), 0xBu); // 0b1011
    return true;
}

TEST(v_vmxnor_mm) {
    ASSERT_HEX_EQ(run_mask_op(VMXNOR(0, 1, 2)), 0x9u);  // 0b1001
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. Move and reduction
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
 *  8. Program — vectorised sum (checksum-like pattern)
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
 *  9. Pipeline — basic RVV operations
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

TEST(v_pipe_vredsum) {
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
 *  10. Pipeline — vectorised checksum
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
