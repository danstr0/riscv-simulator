/**
 * @file test_csr.cpp
 * @brief Tests for CSR instructions.
 *
 * @par Sections
 * @code
 *   1 (line 17) : Decoder
 * @endcode
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Decoder
// ═══════════════════════════════════════════════════════════════════════

TEST(csr_decode_csrrw)
{
    auto inst = Decoder::decode(0x3001'10f3u); // csrrw x1, mstatus, x2
    ASSERT_EQ(inst.op, Op::CSRRW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    constexpr u32 mstatus = 0x300u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mstatus);
    return true;
}

TEST(csr_decode_csrrs)
{
    auto inst = Decoder::decode(0x3043'22f3u); // csrrs x5, mie, x6
    ASSERT_EQ(inst.op, Op::CSRRS);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    constexpr u32 mie = 0x304u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mie);
    return true;
}

TEST(csr_decode_csrrc)
{
    auto inst = Decoder::decode(0x3055'34f3u); // csrrc x9, mtvec, x10
    ASSERT_EQ(inst.op, Op::CSRRC);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs1, 10);
    constexpr u32 mtvec = 0x305u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mtvec);
    return true;
}

TEST(csr_decode_csrrwi)
{
    auto inst = Decoder::decode(0x3412'd6f3u); // csrrwi x13, mepc, 5
    ASSERT_EQ(inst.op, Op::CSRRWI);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs1, 5);
    constexpr u32 mepc = 0x341u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mepc);
    return true;
}

TEST(csr_decode_csrrsi)
{
    auto inst = Decoder::decode(0x3421'e7f3u); // csrrsi x15, mcause, 3
    ASSERT_EQ(inst.op, Op::CSRRSI);
    ASSERT_EQ(inst.rd, 15);
    ASSERT_EQ(inst.rs1,3);
    constexpr u32 mcause = 0x342u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mcause);
    return true;
}

TEST(csr_decode_csrrci)
{
    auto inst = Decoder::decode(0x3443'f8f3u); // csrrci x17, mip, 7
    ASSERT_EQ(inst.op, Op::CSRRCI);
    ASSERT_EQ(inst.rd, 17);
    ASSERT_EQ(inst.rs1, 7);
    constexpr u32 mip = 0x344u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mip);
    return true;
}
