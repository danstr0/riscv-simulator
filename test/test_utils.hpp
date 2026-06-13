/**
 * @file test_utils.hpp
 * @brief Utilities for the test suite.
 */

#pragma once

#include "core/cpu.hpp"
#include "core/pipeline.hpp"
#include "core/memory.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  CPU helpers
// ═══════════════════════════════════════════════════════════════════════

/// Create a CPU backed by 64 KiB of flat RAM at address 0.
inline CPU make_cpu()
{
    auto mem = std::make_shared<FlatMemory>(0x0, 0x10000);
    return CPU(mem);
}

/// Create a PipelinedCPU backed by 64 KiB of flat RAM at address 0.
inline PipelinedCPU make_pipeline(PipelineConfig cfg = {})
{
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    return PipelinedCPU(mem, cfg);
}

template <typename CPUType>
struct TestHarness;

// ── CPU harness ────────────────────────────────────────────────────────

template <>
struct TestHarness<CPU>
{
    CPU cpu = make_cpu();

    CPU& get() { return cpu; }

    auto step() { return cpu.step(); }
    void run(u64 n) { cpu.run(n); }
};

using CPUH = TestHarness<CPU>;

// ── PipelinedCPU harness ───────────────────────────────────────────────

template <>
struct TestHarness<PipelinedCPU>
{
    PipelinedCPU cpu = make_pipeline();

    PipelinedCPU& get() { return cpu; }

    auto step() { return cpu.run_instructions(1); }
    void run(u64 n) { cpu.run_instructions(n); }
};

using PipeH = TestHarness<PipelinedCPU>;

// ═══════════════════════════════════════════════════════════════════════
//  Encoding helpers
// ═══════════════════════════════════════════════════════════════════════

/// @name RV32I I-type encoding
/// @{
constexpr u32 encode_i_type(
    i32 imm,
    u32 rs1,
    u32 funct3,
    u32 rd,
    u32 opcode
)
{
    u32 uimm = static_cast<u32>(imm);

    return (uimm   << 20) |
           (rs1    << 15) |
           (funct3 << 12) |
           (rd     << 7)  |
           opcode;
}

constexpr u32 ADDI(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b000, rd, 0b0010011);
}

constexpr u32 ANDI(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b111, rd, 0b0010011);
}

constexpr u32 ORI(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b110, rd, 0b0010011);
}

constexpr u32 XORI(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b100, rd, 0b0010011);
}

constexpr u32 SLTI(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b010, rd, 0b0010011);
}

constexpr u32 SLTIU(u32 rd, u32 rs1, i32 imm)
{
    return encode_i_type(imm, rs1, 0b011, rd, 0b0010011);
}

constexpr u32 SLLI(u32 rd, u32 rs1, u32 shamt)
{
    return ((0b0000000u)    << 25) |
           ((shamt & 0x1Fu) << 20) |
           ((rs1   & 0x1Fu) << 15) |
           (0b001u          << 12) |
           ((rd    & 0x1Fu) << 7)  |
           0b0010011u;
}

constexpr u32 SRLI(u32 rd, u32 rs1, u32 shamt)
{
    return ((0b0000000u)    << 25) |
           ((shamt & 0x1Fu) << 20) |
           ((rs1   & 0x1Fu) << 15) |
           (0b101u          << 12) |
           ((rd    & 0x1Fu) << 7)  |
           0b0010011u;
}

constexpr u32 SRAI(u32 rd, u32 rs1, u32 shamt)
{
    return ((0b0100000u)    << 25) |
           ((shamt & 0x1Fu) << 20) |
           ((rs1   & 0x1Fu) << 15) |
           (0b101u          << 12) |
           ((rd    & 0x1Fu) << 7)  |
           0b0010011u;
}
/// @}

/// @name RV32I R-type encoding
/// @{
constexpr u32 encode_r_type(
    u32 funct7,
    u32 rs2,
    u32 rs1,
    u32 funct3,
    u32 rd,
    u32 opcode
)
{
    return (funct7 << 25) |
           (rs2    << 20) |
           (rs1    << 15) |
           (funct3 << 12) |
           (rd     << 7)  |
           opcode;
}

constexpr u32 ADD(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b000, rd, 0b0110011);
}

constexpr u32 SUB(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0100000, rs2, rs1, 0b000, rd, 0b0110011);
}

constexpr u32 AND(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b111, rd, 0b0110011);
}

constexpr u32 OR(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b110, rd, 0b0110011);
}

constexpr u32 XOR(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b100, rd, 0b0110011);
}

constexpr u32 SLT(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b010, rd, 0b0110011);
}

constexpr u32 SLTU(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b011, rd, 0b0110011);
}

constexpr u32 SLL(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b001, rd, 0b0110011);
}

constexpr u32 SRL(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0000000, rs2, rs1, 0b101, rd, 0b0110011);
}

constexpr u32 SRA(u32 rd, u32 rs1, u32 rs2)
{
    return encode_r_type(0b0100000, rs2, rs1, 0b101, rd, 0b0110011);
}
/// @}

/// @name RV32I load encoding
/// @{
constexpr u32 encode_rv32i_load(u32 rd, i32 imm, u32 rs1, u32 funct3)
{
    return encode_i_type(imm, rs1, funct3, rd, 0b0000011);
}

constexpr u32 LW(u32 rd, i32 imm, u32 rs1)  { return encode_rv32i_load(rd, imm, rs1, 0b010); }
constexpr u32 LH(u32 rd, i32 imm, u32 rs1)  { return encode_rv32i_load(rd, imm, rs1, 0b001); }
constexpr u32 LHU(u32 rd, i32 imm, u32 rs1) { return encode_rv32i_load(rd, imm, rs1, 0b101); }
constexpr u32 LB(u32 rd, i32 imm, u32 rs1)  { return encode_rv32i_load(rd, imm, rs1, 0b000); }
constexpr u32 LBU(u32 rd, i32 imm, u32 rs1) { return encode_rv32i_load(rd, imm, rs1, 0b100); }
/// @}

/// @name RV32I S-type encoding
/// @{
constexpr u32 encode_s_type(
    i32 imm,
    u32 rs2,
    u32 rs1,
    u32 funct3,
    u32 opcode
)
{
    u32 uimm = static_cast<u32>(imm);

    return (((uimm >> 5) & 0x7Fu) << 25) |
           ((rs2         & 0x1Fu) << 20) |
           ((rs1         & 0x1Fu) << 15) |
           ((funct3      & 0x7u)  << 12) |
           ((uimm        & 0x1Fu) << 7)  |
           (opcode       & 0x7Fu);
}

constexpr u32 encode_rv32i_store(u32 rs2, i32 imm, u32 rs1, u32 funct3)
{
    return encode_s_type(imm, rs2, rs1, funct3, 0b0100011);
}

constexpr u32 SB(u32 rs2, i32 imm, u32 rs1) { return encode_rv32i_store(rs2, imm, rs1, 0b000); }
constexpr u32 SH(u32 rs2, i32 imm, u32 rs1) { return encode_rv32i_store(rs2, imm, rs1, 0b001); }
constexpr u32 SW(u32 rs2, i32 imm, u32 rs1) { return encode_rv32i_store(rs2, imm, rs1, 0b010); }
/// @}

/// @name RV32I B-type encoding
/// @{
constexpr u32 encode_b_type(i32 imm, u32 rs2, u32 rs1, u32 funct3)
{
    u32 uimm = static_cast<u32>(imm);

    return (((uimm >> 12) & 0x1u)  << 31) |
           (((uimm >> 5)  & 0x3Fu) << 25) |
           ((rs2          & 0x1Fu) << 20) |
           ((rs1          & 0x1Fu) << 15) |
           ((funct3       & 0x7u)  << 12) |
           (((uimm >> 1)  & 0xFu)  << 8)  |
           (((uimm >> 11) & 0x1u)  << 7)  |
           (0b1100011     & 0x7Fu);
}

constexpr u32 BEQ(u32 rs1, u32 rs2, i32 imm)  { return encode_b_type(imm, rs2, rs1, 0b000); }
constexpr u32 BNE(u32 rs1, u32 rs2, i32 imm)  { return encode_b_type(imm, rs2, rs1, 0b001); }
constexpr u32 BLT(u32 rs1, u32 rs2, i32 imm)  { return encode_b_type(imm, rs2, rs1, 0b100); }
constexpr u32 BGE(u32 rs1, u32 rs2, i32 imm)  { return encode_b_type(imm, rs2, rs1, 0b101); }
constexpr u32 BLTU(u32 rs1, u32 rs2, i32 imm) { return encode_b_type(imm, rs2, rs1, 0b110); }
constexpr u32 BGEU(u32 rs1, u32 rs2, i32 imm) { return encode_b_type(imm, rs2, rs1, 0b111); }
/// @}

/// @name RV32I J-type encoding
/// @{
constexpr u32 JAL(u32 rd, i32 imm)
{
    const u32 uimm = static_cast<u32>(imm) & 0x1FFFFF;

    return (((uimm >> 20) & 0x1u)   << 31) |
           (((uimm >> 1)  & 0x3FFu) << 21) |
           (((uimm >> 11) & 0x1u)   << 20) |
           (((uimm >> 12) & 0xFFu)  << 12) |
           ((rd           & 0x1Fu)  << 7)  |
           0b1101111;
}

constexpr u32 JALR(u32 rd, i32 imm, u32 rs1)
{
    return encode_i_type(
        imm,
        rs1,
        0b000,
        rd,
        0b1100111
    );
}
/// @}

/// @name RV32I U-type encoding
/// @{ 
constexpr u32 encode_u_type(u32 uimm, u32 rd, u32 opcode)
{
    return (uimm & 0xFFFFF000u) |
           (rd << 7) | opcode;
}

constexpr u32 LUI(u32 rd, u32 imm)   { return encode_u_type(imm, rd, 0b0110111); }
constexpr u32 AUIPC(u32 rd, u32 imm) { return encode_u_type(imm, rd, 0b0010111); }
/// @}

/// @name RV32I system instruction encoding
/// @{
constexpr u32 FENCE  = 0x0ff0'000f;
constexpr u32 EBREAK = 0x0010'0073;
constexpr u32 ECALL  = 0x0000'0073;
/// @}

/// @name RV32M encoding
/// @{
constexpr u32 encode_m(u32 funct3, u32 rd, u32 rs1, u32 rs2)
{
    return (0b0000001u << 25) | (rs2 << 20) | (rs1 << 15)
         | (funct3 << 12) | (rd << 7) | 0b0110011u;
}

constexpr u32 MUL(u32 rd, u32 rs1, u32 rs2)    { return encode_m(0b000, rd, rs1, rs2); }
constexpr u32 MULH(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b001, rd, rs1, rs2); }
constexpr u32 MULHSU(u32 rd, u32 rs1, u32 rs2) { return encode_m(0b010, rd, rs1, rs2); }
constexpr u32 MULHU(u32 rd, u32 rs1, u32 rs2)  { return encode_m(0b011, rd, rs1, rs2); }
constexpr u32 DIV(u32 rd, u32 rs1, u32 rs2)    { return encode_m(0b100, rd, rs1, rs2); }
constexpr u32 DIVU(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b101, rd, rs1, rs2); }
constexpr u32 REM(u32 rd, u32 rs1, u32 rs2)    { return encode_m(0b110, rd, rs1, rs2); }
constexpr u32 REMU(u32 rd, u32 rs1, u32 rs2)   { return encode_m(0b111, rd, rs1, rs2); }
/// @}

/// @name RV32A encoding
/// @{
constexpr u32 encode_amo(u32 funct5, u32 rd, u32 rs1, u32 rs2)
{
    return (funct5 << 27) | (rs2 << 20) | (rs1 << 15)
         | (0b010u << 12) | (rd << 7) | 0b0101111u;
}

constexpr u32 LR_W(u32 rd, u32 rs1)             { return encode_amo(0b00010u, rd, rs1, 0); }
constexpr u32 SC_W(u32 rd, u32 rs1, u32 rs2)    { return encode_amo(0b00011u, rd, rs1, rs2); }
constexpr u32 AMOSWAP(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b00001u, rd, rs1, rs2); }
constexpr u32 AMOADD(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b00000u, rd, rs1, rs2); }
constexpr u32 AMOXOR(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b00100u, rd, rs1, rs2); }
constexpr u32 AMOAND(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b01100u, rd, rs1, rs2); }
constexpr u32 AMOOR(u32 rd, u32 rs1, u32 rs2)   { return encode_amo(0b01000u, rd, rs1, rs2); }
constexpr u32 AMOMIN(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b10000u, rd, rs1, rs2); }
constexpr u32 AMOMAX(u32 rd, u32 rs1, u32 rs2)  { return encode_amo(0b10100u, rd, rs1, rs2); }
constexpr u32 AMOMINU(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b11000u, rd, rs1, rs2); }
constexpr u32 AMOMAXU(u32 rd, u32 rs1, u32 rs2) { return encode_amo(0b11100u, rd, rs1, rs2); }
/// @}

/// @name VSETVLI and V(L|S)E32 encoding
/// @{
constexpr u32 VSETVLI(u32 rd, u32 rs1, u32 zimm)
{
    return (0u << 31) | ((zimm & 0x7FF) << 20) | (rs1 << 15)
         | (0b111u << 12) | (rd << 7) | 0b1010111u;
}
constexpr u32 VLE32(u32 vd, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vd << 7) | 0b0000111u;
}
constexpr u32 VSE32(u32 vs3, u32 rs1)
{
    return (1u << 25) | (rs1 << 15) | (0b110u << 12) | (vs3 << 7) | 0b0100111u;
}

constexpr u32 VTYPE_SEW32 = 0b00000010000;
/// @}

/// @name OPIVV encoding
/// @{
constexpr u32 enc_vv(u32 funct6, u32 vd, u32 vs2, u32 vs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b000u << 12) | (vd << 7) | 0b1010111u;
}

constexpr u32 VADD_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b000000u, vd, vs2, vs1); }
constexpr u32 VSUB_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b000010u, vd, vs2, vs1); }
constexpr u32 VAND_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b001001u, vd, vs2, vs1); }
constexpr u32 VOR_VV(u32 vd, u32 vs2, u32 vs1)   { return enc_vv(0b001010u, vd, vs2, vs1); }
constexpr u32 VXOR_VV(u32 vd, u32 vs2, u32 vs1)  { return enc_vv(0b001011u, vd, vs2, vs1); }
constexpr u32 VMSEQ_VV(u32 vd, u32 vs2, u32 vs1) { return enc_vv(0b011000u, vd, vs2, vs1); }
constexpr u32 VMSLT_VV(u32 vd, u32 vs2, u32 vs1) { return enc_vv(0b011011u, vd, vs2, vs1); }
constexpr u32 VMSLTU_VV(u32 vd, u32 vs2, u32 vs1){ return enc_vv(0b011010u, vd, vs2, vs1); }
/// @}

/// @name OPIVX encoding
/// @{
constexpr u32 enc_vx(u32 funct6, u32 vd, u32 vs2, u32 rs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}

constexpr u32 VADD_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b000000u, vd, vs2, rs1); }
constexpr u32 VSUB_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b000010u, vd, vs2, rs1); }
constexpr u32 VAND_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b001001u, vd, vs2, rs1); }
constexpr u32 VOR_VX(u32 vd, u32 vs2, u32 rs1)   { return enc_vx(0b001010u, vd, vs2, rs1); }
constexpr u32 VXOR_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b001011u, vd, vs2, rs1); }
constexpr u32 VSLL_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b100101u, vd, vs2, rs1); }
constexpr u32 VSRL_VX(u32 vd, u32 vs2, u32 rs1)  { return enc_vx(0b101000u, vd, vs2, rs1); }
constexpr u32 VMSEQ_VX(u32 vd, u32 vs2, u32 rs1) { return enc_vx(0b011000u, vd, vs2, rs1); }
constexpr u32 VMV_V_X(u32 vd, u32 rs1)
{
    return (0b010111u << 26) | (1u << 25) | (0u << 20) | (rs1 << 15)
         | (0b100u << 12) | (vd << 7) | 0b1010111u;
}
/// @}

/// @name OPMVV encoding
/// @{
constexpr u32 enc_mvv(u32 funct6, u32 vd, u32 vs2, u32 vs1)
{
    return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15)
         | (0b010u << 12) | (vd << 7) | 0b1010111u;
}

constexpr u32 VREDSUM(u32 vd, u32 vs2, u32 vs1) { return enc_mvv(0b000000u, vd, vs2, vs1); }
constexpr u32 VMAND(u32 vd, u32 vs2, u32 vs1)   { return enc_mvv(0b011001u, vd, vs2, vs1); }
constexpr u32 VMNAND(u32 vd, u32 vs2, u32 vs1)  { return enc_mvv(0b011101u, vd, vs2, vs1); }
constexpr u32 VMANDN(u32 vd, u32 vs2, u32 vs1)  { return enc_mvv(0b011000u, vd, vs2, vs1); }
constexpr u32 VMOR(u32 vd, u32 vs2, u32 vs1)    { return enc_mvv(0b011010u, vd, vs2, vs1); }
constexpr u32 VMNOR(u32 vd, u32 vs2, u32 vs1)   { return enc_mvv(0b011110u, vd, vs2, vs1); }
constexpr u32 VMORN(u32 vd, u32 vs2, u32 vs1)   { return enc_mvv(0b011100u, vd, vs2, vs1); }
constexpr u32 VMXOR(u32 vd, u32 vs2, u32 vs1)   { return enc_mvv(0b011011u, vd, vs2, vs1); }
constexpr u32 VMXNOR(u32 vd, u32 vs2, u32 vs1)  { return enc_mvv(0b011111u, vd, vs2, vs1); }
constexpr u32 VMV_X_S(u32 rd, u32 vs2)          { return enc_mvv(0b010000u, rd, vs2, 0); }
/// @}
