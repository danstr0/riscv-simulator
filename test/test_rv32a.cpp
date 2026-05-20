/**
 * @file test_rv32a.cpp
 * @brief Tests for the RV32A extension.
 *
 * Sections:
 *   1 (line  22) : Decoder
 *   2 (line 169) : LR.W
 *   3 (line 235) : SC.W
 *   4 (line 420) : AMO* shared execution logic
 *   5 (line 541) : AMO(SWAP|ADD)
 *   6 (line 608) : AMO(XOR|AND|OR)
 *   7 (line 675) : AMO(MIN|MAX)U?
 *   8 (line 764) : CAS program
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Decoder - all instructions decode correctly
// ═══════════════════════════════════════════════════════════════════════

TEST(a_decode_lr_w)
{
    auto inst = Decoder::decode(0x1000'a1afu); // lr.w x3, (x1)
    ASSERT_EQ(inst.op, Op::LR_W);
    ASSERT_EQ(inst.rd,  3);
    ASSERT_EQ(inst.rs2, 0); // LR.W must have rs2=0
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_sc_w)
{
    auto inst = Decoder::decode(0x1811'21afu); // sc.w x3, x1, (x2)
    ASSERT_EQ(inst.op, Op::SC_W);
    ASSERT_EQ(inst.rd,  3);
    ASSERT_EQ(inst.rs2, 1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amoswap)
{
    auto inst = Decoder::decode(0x0821'a0afu); // amoswap.w x1, x2, (x3)
    ASSERT_EQ(inst.op, Op::AMOSWAP_W);
    ASSERT_EQ(inst.rd,  1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.rs1, 3);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amoadd)
{
    auto inst = Decoder::decode(0x0053'222fu); // amoadd.w x4, x5, (x6)
    ASSERT_EQ(inst.op, Op::AMOADD_W);
    ASSERT_EQ(inst.rd,  4);
    ASSERT_EQ(inst.rs2, 5);
    ASSERT_EQ(inst.rs1, 6);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amoxor)
{
    auto inst = Decoder::decode(0x2084'a3afu); // amoxor.w x7, x8, (x9)
    ASSERT_EQ(inst.op, Op::AMOXOR_W);
    ASSERT_EQ(inst.rd,  7);
    ASSERT_EQ(inst.rs2, 8);
    ASSERT_EQ(inst.rs1, 9);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amoand)
{
    auto inst = Decoder::decode(0x60b6'252fu); // amoand.w x10, x11, (x12)
    ASSERT_EQ(inst.op, Op::AMOAND_W);
    ASSERT_EQ(inst.rd,  10);
    ASSERT_EQ(inst.rs2, 11);
    ASSERT_EQ(inst.rs1, 12);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amoor)
{
    auto inst = Decoder::decode(0x40e7'a6afu); // amoor.w x13, x14, (x15)
    ASSERT_EQ(inst.op, Op::AMOOR_W);
    ASSERT_EQ(inst.rd,  13);
    ASSERT_EQ(inst.rs2, 14);
    ASSERT_EQ(inst.rs1, 15);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amomin)
{
    auto inst = Decoder::decode(0x8119'282fu); // amomin.w x16, x17, (x18)
    ASSERT_EQ(inst.op, Op::AMOMIN_W);
    ASSERT_EQ(inst.rd,  16);
    ASSERT_EQ(inst.rs2, 17);
    ASSERT_EQ(inst.rs1, 18);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amomax)
{
    auto inst = Decoder::decode(0xa14a'a9afu); // amomax.w x19, x20, (x21)
    ASSERT_EQ(inst.op, Op::AMOMAX_W);
    ASSERT_EQ(inst.rd,  19);
    ASSERT_EQ(inst.rs2, 20);
    ASSERT_EQ(inst.rs1, 21);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amominu)
{
    auto inst = Decoder::decode(0xc17c'2b2fu); // amominu.w x22, x23, (x24)
    ASSERT_EQ(inst.op, Op::AMOMINU_W);
    ASSERT_EQ(inst.rd,  22);
    ASSERT_EQ(inst.rs2, 23);
    ASSERT_EQ(inst.rs1, 24);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(a_decode_amomaxu)
{
    auto inst = Decoder::decode(0xe1ad'acafu); // amomaxu.w x25, x26, (x27)
    ASSERT_EQ(inst.op, Op::AMOMAXU_W);
    ASSERT_EQ(inst.rd,  25);
    ASSERT_EQ(inst.rs2, 26);
    ASSERT_EQ(inst.rs1, 27);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(a_decode_lr_w_invalid_rs2)
{
    // LR.W with rs2 != 0 is invalid per spec
    u32 bad_lr = encode_amo(0b00010u, 3, 1, 5); // rs2=5
    auto inst = Decoder::decode(bad_lr);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

TEST(a_decode_invalid_funct3)
{
    // AMO opcode but funct3 != 010 (not .W) should be invalid
    // Here, funct3 = 011
    u32 bad_enc = (0b00000u << 27) | (2u << 20) | (1u << 15)
                | (0b011u << 12) | (3u << 7) | 0b0101111u;
    auto inst = Decoder::decode(bad_enc);
    ASSERT_EQ(inst.op, Op::INVALID);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. LR.W
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_lr_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0xAABB'CCDD);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LR_W(2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xAABB'CCDDu);
    return true;
}

TEST(a_exec_lr_w_cpu)  { return run_lr_w<CPUH>(); }
TEST(a_exec_lr_w_pipe) { return run_lr_w<PipeH>(); }

template <typename Harness>
bool run_lr_w_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0x1234'5678);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LR_W(0, 1));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(a_exec_lr_w_x0_cpu)  { return run_lr_w_x0<CPUH>(); }
TEST(a_exec_lr_w_x0_pipe) { return run_lr_w_x0<PipeH>(); }

template <typename Harness>
bool run_lr_w_misaligned()
{
    for (u32 addr : {0x101u, 0x102u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.set_reg(1, addr);
        cpu.load_instruction(0,  LR_W(2, 1));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

         cpu.run_until([&]() { return cpu.halted(); }, 20);
         ASSERT(cpu.halted());
         ASSERT_EQ(cpu.reg(3), 0);
    }
    return true;
}

TEST(a_exec_lr_w_misaligned_cpu)  { return run_lr_w_misaligned<CPUH>(); }
TEST(a_exec_lr_w_misaligned_pipe) { return run_lr_w_misaligned<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  3. SC.W
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_sc_w_success()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 0xAAAA'AAAA);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0xBBBB'BBBB);

    cpu.load_instruction(0, LR_W(1, 10));
    cpu.load_instruction(4, SC_W(2, 10, 11));

    h.step();
    h.step();

    ASSERT_HEX_EQ(cpu.reg(1), 0xAAAA'AAAAu); // LR.W loaded old value
    ASSERT_EQ(cpu.reg(2), 0u);               // SC.W succeeded
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xBBBBBBBBu);
    return true;
}

TEST(a_exec_sc_w_success_cpu)  { return run_sc_w_success<CPUH>(); }
TEST(a_exec_sc_w_success_pipe) { return run_sc_w_success<PipeH>(); }

template <typename Harness>
bool run_sc_w_no_reservation()
{
    // SC.W without reservation fails
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 0x1111'1111);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x2222'2222);

    cpu.load_instruction(0, SC_W(2, 10, 11));

    h.step();
    ASSERT(cpu.reg(2) != 0u); // SC.W failed
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0x1111'1111u);
    return true;
}

TEST(a_exec_sc_w_no_reservation_cpu)  { return run_sc_w_no_reservation<CPUH>(); }
TEST(a_exec_sc_w_no_reservation_pipe) { return run_sc_w_no_reservation<PipeH>(); }

template <typename Harness>
bool run_sc_w_wrong_address()
{
    // LR.W on one address, SC.W on another -> fail
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 0xAAAA);
    cpu.memory().write32(0x300, 0xBBBB);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x300);
    cpu.set_reg(12, 0xCCCC);

    cpu.load_instruction(0, LR_W(1, 10));     // reserve 0x200
    cpu.load_instruction(4, SC_W(2, 11, 12)); // SC to 0x300 -> fail
    
    h.step();
    h.step();
    ASSERT(cpu.reg(2) != 0u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x300).value, 0xBBBBu);
    return true;
}

TEST(a_exec_sc_w_wrong_address_cpu)  { return run_sc_w_wrong_address<CPUH>(); }
TEST(a_exec_sc_w_wrong_address_pipe) { return run_sc_w_wrong_address<PipeH>(); }

template <typename Harness>
bool run_sc_w_clears_1()
{
    // SC clears reservation upon success
    // Two SC.W after one LR.W — second should fail
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 200);
    cpu.set_reg(12, 300);

    cpu.load_instruction(0,  LR_W(1, 10));     // reserve 0x200
    cpu.load_instruction(4,  SC_W(2, 10, 11)); // succeeds
    cpu.load_instruction(8,  SC_W(3, 10, 12)); // fails

    h.step();
    h.step();
    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    ASSERT(cpu.reg(3) != 0u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u); // first write stuck
    return true;
}

TEST(a_exec_sc_w_clears_reservation_1_cpu)  { return run_sc_w_clears_1<CPUH>(); }
TEST(a_exec_sc_w_clears_reservation_1_pipe) { return run_sc_w_clears_1<PipeH>(); }

template <typename Harness>
bool run_sc_w_clears_2()
{
    // SC.W clears reservation on failure as well
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 5);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(3, 0x104);
    cpu.set_reg(4, 42);
    cpu.set_reg(6, 0x100);

    cpu.load_instruction(0, LR_W(2, 1));
    cpu.load_instruction(4, SC_W(5, 3, 4)); // fails
    cpu.load_instruction(8, SC_W(7, 6, 4)); // should also fail

    h.step();
    h.step();
    h.step();
    ASSERT_EQ(cpu.reg(5), 1u);
    ASSERT_EQ(cpu.reg(7), 1u);
    return true;
}

TEST(a_exec_sc_w_clears_reservation_2_cpu)  { return run_sc_w_clears_2<CPUH>(); }
TEST(a_exec_sc_w_clears_reservation_2_pipe) { return run_sc_w_clears_2<PipeH>(); }

template <typename Harness>
bool run_store_clears()
{
    // A regular store between LR.W and SC.W should clear the reservation
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 999);
    cpu.set_reg(12, 42);

    cpu.load_instruction(0,  LR_W(1, 10));     // reserve 0x200
    cpu.load_instruction(4,  SW(11, 0, 10));   // clears reservation
    cpu.load_instruction(8,  SC_W(3, 10, 12)); // SC -> fail

    h.step();
    h.step();
    h.step();
    ASSERT(cpu.reg(3) != 0u);
    // Memory has the SW value, not the SC value
    ASSERT_EQ(cpu.memory().read32(0x200).value, 999u);
    return true;
}

TEST(a_store_clears_reservation_cpu)  { return run_store_clears<CPUH>(); }
TEST(a_store_clears_reservation_pipe) { return run_store_clears<PipeH>(); }

template <typename Harness>
bool run_sc_w_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 5);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 42);

    cpu.load_instruction(0, LR_W(3, 1));
    cpu.load_instruction(4, SC_W(0, 1, 2));
    
    h.step();
    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    ASSERT_EQ(cpu.memory().read32(0x100).value, 42u);
    return true;
}

TEST(a_exec_sc_w_x0_cpu)  { return run_sc_w_x0<CPUH>(); }
TEST(a_exec_sc_w_x0_pipe) { return run_sc_w_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  4. Shared AMO* instruction logic
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_amo_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.memory().write32(0x100, 5);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 99);

    cpu.load_instruction(0, AMOSWAP(0, 1, 2));
    
    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    ASSERT_EQ(cpu.memory().read32(0x100).value, 99u);
    return true;
}

TEST(a_exec_amo_x0_cpu)  { return run_amo_x0<CPUH>(); }
TEST(a_exec_amo_x0_pipe) { return run_amo_x0<PipeH>(); }

template <typename Harness>
bool run_amo_rd_eq_rs1()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 55);
    cpu.set_reg(1, 0x100); // rs1 AND rd
    cpu.set_reg(2, 99);

    cpu.load_instruction(0, AMOSWAP(1, 1, 2));

    h.step();

    // rd gets old memory value
    ASSERT_EQ(cpu.reg(1), 55u);
    // memory gets rs2
    ASSERT_EQ(cpu.memory().read32(0x100).value, 99u);
    return true;
}

TEST(a_exec_amo_rd_eq_rs1_cpu)  { return run_amo_rd_eq_rs1<CPUH>(); }
TEST(a_exec_amo_rd_eq_rs1_pipe) { return run_amo_rd_eq_rs1<PipeH>(); }

template <typename Harness>
bool run_amo_rd_eq_rs2()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 10);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 5); // rs2 AND rd

    cpu.load_instruction(0, AMOADD(2, 1, 2));

    h.step();
    // rd gets old mem
    ASSERT_EQ(cpu.reg(2), 10u);
    // memory gets old + original rs2
    ASSERT_EQ(cpu.memory().read32(0x100).value, 15u);
    return true;
}

TEST(a_exec_amo_rd_eq_rs2_cpu)  { return run_amo_rd_eq_rs2<CPUH>(); }
TEST(a_exec_amo_rd_eq_rs2_pipe) { return run_amo_rd_eq_rs2<PipeH>(); }

template <typename Harness>
bool run_amo_all_overlap()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 7);
    cpu.set_reg(1, 0x100); // rd, rs1, rs2 all x1

    cpu.load_instruction(0, AMOADD(1, 1, 1));

    h.step();
    // rd gets old mem
    ASSERT_EQ(cpu.reg(1), 7u);
    // memory should be old + original rs2 (=0x100)
    ASSERT_EQ(cpu.memory().read32(0x100).value, 7u + 0x100u);
    return true;
}

TEST(a_exec_amo_all_overlap_cpu)  { return run_amo_all_overlap<CPUH>(); }
TEST(a_exec_amo_all_overlap_pipe) { return run_amo_all_overlap<PipeH>(); }

template <typename Harness>
bool run_amo_misaligned()
{
    for (u32 addr : {0x101u, 0x102u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.set_reg(1, addr);
        cpu.set_reg(2, 99);
        cpu.load_instruction(0,  AMOSWAP(3, 1, 2));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

         cpu.run_until([&]() { return cpu.halted(); }, 20);
         ASSERT(cpu.halted());
         ASSERT_EQ(cpu.reg(3), 0);
    }
    return true;
}

TEST(a_exec_amo_misaligned_cpu)  { return run_amo_misaligned<CPUH>(); }
TEST(a_exec_amo_misaligned_pipe) { return run_amo_misaligned<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  4. AMO(SWAP|ADD)
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_amoswap_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 200);

    cpu.load_instruction(0, AMOSWAP(1, 10, 11));

    h.step();    
    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u);
    return true;
}

TEST(a_exec_amoswap_w_cpu)  { return run_amoswap_w<CPUH>(); }
TEST(a_exec_amoswap_w_pipe) { return run_amoswap_w<PipeH>(); }

template <typename Harness>
bool run_amoadd_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 50);

    cpu.load_instruction(0, AMOADD(1, 10, 11));

    h.step();
    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 150u);
    return true;
}

TEST(a_exec_amoadd_w_cpu)  { return run_amoadd_w<CPUH>(); }
TEST(a_exec_amoadd_w_pipe) { return run_amoadd_w<PipeH>(); }

template <typename Harness>
bool run_amoadd_w_wrap()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0xFFFF'FFFF);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 1);

    cpu.load_instruction(0, AMOADD(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    ASSERT_EQ(cpu.memory().read32(0x100).value, 0u);
    return true;
}

TEST(a_exec_amoadd_w_wrap_cpu)  { return run_amoadd_w_wrap<CPUH>(); }
TEST(a_exec_amoadd_w_wrap_pipe) { return run_amoadd_w_wrap<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  5. AMO(XOR|AND|OR)
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_amoxor_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 0xFF00'FF00);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F'0F0F);

    cpu.load_instruction(0, AMOXOR(1, 10, 11));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xFF00'FF00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xF00F'F00Fu);
    return true;
}

TEST(a_exec_amoxor_w_cpu)  { return run_amoxor_w<CPUH>(); }
TEST(a_exec_amoxor_w_pipe) { return run_amoxor_w<PipeH>(); }

template <typename Harness>
bool run_amoand_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 0xFF00'FF00);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F'0F0F);

    cpu.load_instruction(0, AMOAND(1, 10, 11));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xFF00'FF00u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0x0F00'0F00u);
    return true;
}

TEST(a_exec_amoand_w_cpu)  { return run_amoand_w<CPUH>(); }
TEST(a_exec_amoand_w_pipe) { return run_amoand_w<PipeH>(); }

template <typename Harness>
bool run_amoor_w()
{
    Harness h;
    auto& cpu = h.get ();

    cpu.memory().write32(0x200, 0xF000'F000);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 0x0F0F'0F0F);

    cpu.load_instruction(0, AMOOR(1, 10, 11));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xF000'F000u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xFF0F'FF0Fu);
    return true;
}

TEST(a_exec_amoor_w_cpu)  { return run_amoor_w<CPUH>(); }
TEST(a_exec_amoor_w_pipe) { return run_amoor_w<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  6. AMO(MIN|MAX)U?
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_amomin_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMIN(1, 10, 11)); // min(-10, 5)

    h.step();
    ASSERT_EQ(cpu.reg(1), static_cast<u32>(-10));
    ASSERT_EQ(cpu.memory().read32(0x200).value, static_cast<u32>(-10));
    return true;
}

TEST(a_exec_amomin_w_cpu)  { return run_amomin_w<CPUH>(); }
TEST(a_exec_amomin_w_pipe) { return run_amomin_w<PipeH>(); }

template <typename Harness>
bool run_amomax_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMAX(1, 10, 11)); // max(-10, 5)

    h.step();
    ASSERT_EQ(cpu.reg(1), static_cast<u32>(-10));
    ASSERT_EQ(cpu.memory().read32(0x200).value, 5u);
    return true;
}

TEST(a_exec_amomax_w_cpu)  { return run_amomax_w<CPUH>(); }
TEST(a_exec_amomax_w_pipe) { return run_amomax_w<PipeH>(); }

template <typename Harness>
bool run_amominu_w()
{
    // -10 signed -> 0xFFFF'FFF6 unsigned
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMINU(1, 10, 11));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xFFFF'FFF6u);
    ASSERT_EQ(cpu.memory().read32(0x200).value, 5u); // 5 < u32(-10)
    return true;
}

TEST(a_exec_amominu_w_cpu)  { return run_amominu_w<CPUH>(); }
TEST(a_exec_amominu_w_pipe) { return run_amominu_w<PipeH>(); }

template <typename Harness>
bool run_amomaxu_w()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, static_cast<u32>(-10));
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 5);

    cpu.load_instruction(0, AMOMAXU(1, 10, 11));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xFFFF'FFF6u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0xFFFF'FFF6u);
    return true;
}

TEST(a_exec_amomaxu_w_cpu)  { return run_amomaxu_w<CPUH>(); }
TEST(a_exec_amomaxu_w_pipe) { return run_amomaxu_w<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  5. CAS (compare-and-swap) pattern
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_prog_cas_pattern()
{
    // CAS loop: atomically set mem[0x200] from 100 to 200
    //
    // 0x00 retry: lr.w x13, (x10)
    // 0x04        bne x13, x11, 16      -> fail
    // 0x08        sc.w x14, x12, (x10)
    // 0x0C        bne x14, x0, -12      -> retry
    // 0x10        jal x0, 8
    // 0x14 fail:  addi x15, x0, 1
    // 0x18 done:  ebreak

    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x200, 100);
    cpu.set_reg(10, 0x200);
    cpu.set_reg(11, 100);
    cpu.set_reg(12, 200);

    cpu.load_instruction(0x00, LR_W(13, 10));
    cpu.load_instruction(0x04, BNE(13, 11 ,16));
    cpu.load_instruction(0x08, SC_W(14, 10, 12));
    cpu.load_instruction(0x0C, BNE(14, 0, -12));
    cpu.load_instruction(0x10, JAL(0, 8));
    cpu.load_instruction(0x14, ADDI(15, 0, 1));
    cpu.load_instruction(0x18, EBREAK);

    h.run(100);
    // In single-thread, CAS always succeeds on first try
    ASSERT_EQ(cpu.reg(13), 100u); // LR loaded the expected value
    ASSERT_EQ(cpu.reg(14), 0u);   // SC succeeded
    ASSERT_EQ(cpu.reg(15), 0u);   // never went to fail path
    ASSERT_EQ(cpu.memory().read32(0x200).value, 200u);
    return true;
}

TEST(a_prog_cas_pattern_cpu)  { return run_prog_cas_pattern<CPUH>(); }
TEST(a_prog_cas_pattern_pipe) { return run_prog_cas_pattern<PipeH>(); }
