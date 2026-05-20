/**
 * @file test_executor.cpp
 * @brief Tests for RV32I instruction execution.
 *
 * Sections:
 *   1 (line   25) : I-type instructions
 *   2 (line  869) : R-type instructions
 *   3 (line 1887) : Loads
 *   4 (line 2455) : Stores
 *   5 (line 2818) : Branches
 *   6 (line 3388) : x0 hardwired to zero
 *   7 (line 3596) : Upper immediates
 *   8 (line 3735) : System instructions
 *   9 (line 3793) : CPU lifecycle
 *  10 (line 3853) : Execution statistics
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. I-type instructions
// ═══════════════════════════════════════════════════════════════════════

// ── ADDI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_exec_addi()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(2, 10);
    cpu.load_instruction(0, ADDI(1, 2, 5));

    h.step();
    ASSERT_EQ(cpu.reg(1), 15u);
    return true;
}

TEST(exec_addi_cpu)  { return run_exec_addi<CPUH>(); }
TEST(exec_addi_pipe) { return run_exec_addi<PipeH>(); }

template <typename Harness>
bool run_exec_addi_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 100);
    cpu.load_instruction(0, ADDI(2, 1, -30)); 

    h.step();
    ASSERT_EQ(cpu.reg(2), 70u);
    return true;
}

TEST(exec_addi_neg_cpu)  { return run_exec_addi_neg<CPUH>(); }
TEST(exec_addi_neg_pipe) { return run_exec_addi_neg<PipeH>(); }

template <typename Harness>
bool run_addi_wraparound()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 0xFFFF'FFFF);
    cpu.load_instruction(0, ADDI(1, 1, 1));
    
    h.step();
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_addi_wraparound_cpu)  { return run_addi_wraparound<CPUH>(); }
TEST(exec_addi_wraparound_pipe) { return run_addi_wraparound<PipeH>(); }

template <typename Harness>
bool run_exec_addi_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.load_instruction(0, ADDI(1, 0, 42));
    cpu.load_instruction(4, ADDI(0, 0, 123));

    h.step(); 
    ASSERT_EQ(cpu.reg(1), 42u);

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_addi_x0_cpu)  { return run_exec_addi_x0<CPUH>(); }
TEST(exec_addi_x0_pipe) { return run_exec_addi_x0<PipeH>(); }

// ── ANDI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_andi()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xABCD);
    cpu.load_instruction(0, ANDI(2, 1, 0xFF));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xCDu);
    return true;
}

TEST(exec_andi_cpu)  { return run_andi<CPUH>(); }
TEST(exec_andi_pipe) { return run_andi<PipeH>(); }

template <typename Harness>
bool run_andi_zero()
{
    // AND with 0 -> clears all bits
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234);
    cpu.load_instruction(0, ANDI(2, 1, 0));

    h.step();    
    ASSERT_HEX_EQ(cpu.reg(2), 0x0u);
    return true;
}

TEST(exec_andi_zero_cpu)  { return run_andi_zero<CPUH>(); }
TEST(exec_andi_zero_pipe) { return run_andi_zero<PipeH>(); }

template <typename Harness>
bool run_andi_neg1()
{
    // AND with -1 -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xBEEF);
    cpu.load_instruction(0, ANDI(2, 1, -1));

    h.step();    
    ASSERT_HEX_EQ(cpu.reg(2), 0xBEEFu);
    return true;
}

TEST(exec_andi_neg1_cpu)  { return run_andi_neg1<CPUH>(); }
TEST(exec_andi_neg1_pipe) { return run_andi_neg1<PipeH>(); }

template <typename Harness>
bool run_andi_neg_imm()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    // -256 sign-extends to 0xFFFF'FF00
    cpu.load_instruction(0, ANDI(2, 1, -256));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234'5600u);
    return true;
}

TEST(exec_andi_neg_imm_cpu)  { return run_andi_neg_imm<CPUH>(); }
TEST(exec_andi_neg_imm_pipe) { return run_andi_neg_imm<PipeH>(); }

template <typename Harness>
bool run_andi_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 0xABCD);
    cpu.load_instruction(0, ANDI(0, 1, 0xFF));
    cpu.load_instruction(4, ANDI(1, 0, 0xFF));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();    
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_andi_x0_cpu)  { return run_andi_x0<CPUH>(); }
TEST(exec_andi_x0_pipe) { return run_andi_x0<PipeH>(); }

// ── ORI ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_ori()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xF000);
    cpu.load_instruction(0, ORI(2, 1, 0x0F));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xF00Fu);
    return true;
}

TEST(exec_ori_cpu)  { return run_ori<CPUH>(); }
TEST(exec_ori_pipe) { return run_ori<PipeH>(); }

template <typename Harness>
bool run_ori_zero()
{
    // OR with zero -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234);
    cpu.load_instruction(0, ORI(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234u);
    return true;
}

TEST(exec_ori_zero_cpu)  { return run_ori_zero<CPUH>(); }
TEST(exec_ori_zero_pipe) { return run_ori_zero<PipeH>(); }

template <typename Harness>
bool run_ori_neg1()
{
    // OR with -1 -> sets all bits
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, ORI(2, 1, -1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FFFFu);
    return true;
}

TEST(exec_ori_neg1_cpu)  { return run_ori_neg1<CPUH>(); }
TEST(exec_ori_neg1_pipe) { return run_ori_neg1<PipeH>(); }

template <typename Harness>
bool run_ori_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xF00F);
    cpu.load_instruction(0, ORI(0, 1, 0xFF));
    cpu.load_instruction(4, ORI(1, 0, 0xFF));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(0), 0x0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0xFFu);
    return true;
}

TEST(exec_ori_x0_cpu)  { return run_ori_x0<CPUH>(); }
TEST(exec_ori_x0_pipe) { return run_ori_x0<PipeH>(); }

// ── XORI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_xori()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF);
    cpu.load_instruction(0, XORI(2, 1, 0xFF));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFF00u);
    return true;
}

TEST(exec_xori_cpu)  { return run_xori<CPUH>(); }
TEST(exec_xori_pipe) { return run_xori<PipeH>(); }

template <typename Harness>
bool run_xori_zero()
{
    // XORI with zero -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1EE7);
    cpu.load_instruction(0, XORI(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1EE7u);
    return true;
}

TEST(exec_xori_zero_cpu)  { return run_xori_zero<CPUH>(); }
TEST(exec_xori_zero_pipe) { return run_xori_zero<PipeH>(); }

template <typename Harness>
bool run_xori_neg1()
{
    // XORI with -1 -> bitwise inverse
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, XORI(2, 1, -1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), ~0x1234'5678u);
    return true;
}

TEST(exec_xori_neg1_cpu)  { return run_xori_neg1<CPUH>(); }
TEST(exec_xori_neg1_pipe) { return run_xori_neg1<PipeH>(); }

template <typename Harness>
bool run_xori_neg_imm()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 0x1234'5678);
    // -256 sign-extends to 0xFFFF'FF00
    cpu.load_instruction(0, XORI(2, 1, -256));
    
    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xEDCB'A978u);
    return true;
}

TEST(exec_xori_neg_imm_cpu)  { return run_xori_neg_imm<CPUH>(); }
TEST(exec_xori_neg_imm_pipe) { return run_xori_neg_imm<PipeH>(); }

template <typename Harness>
bool run_xori_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xBEEF);
    cpu.load_instruction(0, XORI(0, 1, 0xF00F));
    cpu.load_instruction(4, XORI(1, 0, 0x123));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(0), 0x0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x123u);
    return true;
}

TEST(exec_xori_x0_cpu)  { return run_xori_x0<CPUH>(); }
TEST(exec_xori_x0_pipe) { return run_xori_x0<PipeH>(); }

// ── SLTI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_slti_true()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, static_cast<u32>(-10));
    cpu.load_instruction(0, SLTI(2, 1, 5)); // -10 < 5
    
    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_slti_true_cpu)  { return run_slti_true<CPUH>(); }
TEST(exec_slti_true_pipe) { return run_slti_true<PipeH>(); }

template <typename Harness>
bool run_slti_false()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.load_instruction(0, SLTI(2, 1, 5)); // 10 < 5

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_slti_false_cpu)  { return run_slti_false<CPUH>(); }
TEST(exec_slti_false_pipe) { return run_slti_false<PipeH>(); }

template <typename Harness>
bool run_slti_signed_neg()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.load_instruction(0, SLTI(2, 1, -1)); // -5 < -1
    cpu.load_instruction(4, SLTI(2, 1, -7)); // -6 < -7
    
    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_slti_signed_neg_cpu)  { return run_slti_signed_neg<CPUH>(); }
TEST(exec_slti_signed_neg_pipe) { return run_slti_signed_neg<PipeH>(); }

template <typename Harness>
bool run_slti_equal()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.load_instruction(0, SLTI(2, 1, 42));  // 42 < 42
    cpu.load_instruction(4, SLTI(4, 3, -42)); // -42 < -42 (after x3 reassignment)

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);

    cpu.set_reg(3, static_cast<u32>(-42));

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_slti_equal_cpu)  { return run_slti_equal<CPUH>(); }
TEST(exec_slti_equal_pipe) { return run_slti_equal<PipeH>(); }

template <typename Harness>
bool run_slti_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.load_instruction(0,  SLTI(0, 1, 43)); // 42 < 43
    cpu.load_instruction(4,  SLTI(2, 0, 1));  // 0 < 1
    cpu.load_instruction(8,  SLTI(3, 0, 0));  // 0 < 0
    cpu.load_instruction(12, SLTI(4, 0, -1)); // 0 < -1

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    
    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_slti_x0_cpu)  { return run_slti_x0<CPUH>(); }
TEST(exec_slti_x0_pipe) { return run_slti_x0<PipeH>(); }

// ── SLTIU ──────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sltiu_true()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.load_instruction(0, SLTIU(2, 1, 20)); // 10 < 20

    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_sltiu_true_cpu)  { return run_sltiu_true<CPUH>(); }
TEST(exec_sltiu_true_pipe) { return run_sltiu_true<PipeH>(); }

template <typename Harness>
bool run_sltiu_false()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.load_instruction(0, SLTIU(2, 1, 5)); // 10 < 5

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_sltiu_false_cpu)  { return run_sltiu_false<CPUH>(); }
TEST(exec_sltiu_false_pipe) { return run_sltiu_false<PipeH>(); }

template <typename Harness>
bool run_sltiu_neg_reg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.load_instruction(0, SLTIU(2, 1, 1)); // -1 < 1

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u); // -1 sign-extends to 0xFFFF'FFFF
    return true;
}

TEST(exec_sltiu_neg_reg_cpu)  { return run_sltiu_neg_reg<CPUH>(); }
TEST(exec_sltiu_neg_reg_pipe) { return run_sltiu_neg_reg<PipeH>(); }

template <typename Harness>
bool run_sltiu_neg_imm()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0);
    cpu.load_instruction(0, SLTIU(2, 1, -1)); // 1 < u32(-1)

    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_sltiu_neg_imm_cpu)  { return run_sltiu_neg_imm<CPUH>(); }
TEST(exec_sltiu_neg_imm_pipe) { return run_sltiu_neg_imm<PipeH>(); }

template <typename Harness>
bool run_sltiu_equal()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.load_instruction(0, SLTIU(2, 1, -1));
    
    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_sltiu_equal_cpu)  { return run_sltiu_equal<CPUH>(); }
TEST(exec_sltiu_equal_pipe) { return run_sltiu_equal<PipeH>(); }

template <typename Harness>
bool run_sltiu_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 1);
    cpu.load_instruction(0,  SLTIU(0, 1, 2));  // 1 < 2
    cpu.load_instruction(4,  SLTIU(2, 0, 0));  // 0 < 0
    cpu.load_instruction(8,  SLTIU(3, 0, 1));  // 0 < 1
    cpu.load_instruction(12, SLTIU(4, 0, -1)); // 0 < u32(-1)
    
    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 1u);
    return true;
}

TEST(exec_sltiu_x0_cpu)  { return run_sltiu_x0<CPUH>(); }
TEST(exec_sltiu_x0_pipe) { return run_sltiu_x0<PipeH>(); }

// ── SLLI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_slli()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 1);
    cpu.load_instruction(0, SLLI(2, 1, 8)); // 1 << 8
    
    h.step();
    ASSERT_EQ(cpu.reg(2), 256u);
    return true;
}

TEST(exec_slli_cpu)  { return run_slli<CPUH>(); }
TEST(exec_slli_pipe) { return run_slli<PipeH>(); }

template <typename Harness>
bool run_slli_zero_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, SLLI(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234'5678u);
    return true;
}

TEST(exec_slli_zero_shift_cpu)  { return run_slli_zero_shift<CPUH>(); }
TEST(exec_slli_zero_shift_pipe) { return run_slli_zero_shift<PipeH>(); }

template <typename Harness>
bool run_slli_max_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.load_instruction(0, SLLI(2, 1, 31));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x8000'0000u);
    return true;
}

TEST(exec_slli_max_shift_cpu)  { return run_slli_max_shift<CPUH>(); }
TEST(exec_slli_max_shift_pipe) { return run_slli_max_shift<PipeH>(); }

template <typename Harness>
bool run_slli_high_bits()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, SLLI(2, 1, 16));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x5678'0000u);
    return true;
}

TEST(exec_slli_high_bits_cpu)  { return run_slli_high_bits<CPUH>(); }
TEST(exec_slli_high_bits_pipe) { return run_slli_high_bits<PipeH>(); }

template <typename Harness>
bool run_slli_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.load_instruction(0, SLLI(0, 1, 5));
    cpu.load_instruction(4, SLLI(2, 0, 7));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_slli_x0_cpu)  { return run_slli_x0<CPUH>(); }
TEST(exec_slli_x0_pipe) { return run_slli_x0<PipeH>(); }

// ── SRLI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_srli()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 256);
    cpu.load_instruction(0, SRLI(2, 1, 4)); // 256 >> 4

    h.step();
    ASSERT_EQ(cpu.reg(2), 16u);
    return true;
}

TEST(exec_srli_cpu)  { return run_srli<CPUH>(); }
TEST(exec_srli_pipe) { return run_srli<PipeH>(); }

template <typename Harness>
bool run_srli_zero_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.load_instruction(0, SRLI(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEAD'BEEFu);
    return true;
}

TEST(exec_srli_zero_shift_cpu)  { return run_srli_zero_shift<CPUH>(); }
TEST(exec_srli_zero_shift_pipe) { return run_srli_zero_shift<PipeH>(); }

template <typename Harness>
bool run_srli_zero_fill()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.load_instruction(0, SRLI(2, 1, 4));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x0800'0000u);
    return true;
}

TEST(exec_srli_zero_fill_cpu)  { return run_srli_zero_fill<CPUH>(); }
TEST(exec_srli_zero_fill_pipe) { return run_srli_zero_fill<PipeH>(); }

template <typename Harness>
bool run_srli_max_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000u);
    cpu.load_instruction(0, SRLI(2, 1, 31));

    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_srli_max_shift_cpu)  { return run_srli_max_shift<CPUH>(); }
TEST(exec_srli_max_shift_pipe) { return run_srli_max_shift<PipeH>(); }

template <typename Harness>
bool run_srli_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 256);
    cpu.load_instruction(0, SRLI(0, 1, 4));
    cpu.load_instruction(4, SRLI(2, 0, 4));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_srli_x0_cpu)  { return run_srli_x0<CPUH>(); }
TEST(exec_srli_x0_pipe) { return run_srli_x0<PipeH>(); }

// ── SRAI ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_srai()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 8);
    cpu.load_instruction(0, SRAI(2, 1, 3)); // 8 >> 3

    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);
    return true;
}

TEST(exec_srai_cpu)  { return run_srai<CPUH>(); }
TEST(exec_srai_pipe) { return run_srai<PipeH>(); }

template <typename Harness>
bool run_srai_sign_extend()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-256));
    cpu.load_instruction(0, SRAI(2, 1, 4));

    h.step();
    ASSERT_EQ(cpu.reg(2), static_cast<u32>(-16));
    return true;
}

TEST(exec_srai_sign_extend_cpu)  { return run_srai_sign_extend<CPUH>(); }
TEST(exec_srai_sign_extend_pipe) { return run_srai_sign_extend<PipeH>(); }

template <typename Harness>
bool run_srai_neg1()
{
    // -1 stays -1, regardless of shamt
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.load_instruction(0, SRAI(2, 1, 12));

    h.step();
    ASSERT_EQ(cpu.reg(2), static_cast<u32>(-1));
    return true;
}

TEST(exec_srai_neg1_cpu)  { return run_srai_neg1<CPUH>(); }
TEST(exec_srai_neg1_pipe) { return run_srai_neg1<PipeH>(); }

template <typename Harness>
bool run_srai_max_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(3, 0x7000'0000);
    cpu.load_instruction(0, SRAI(2, 1, 31));
    cpu.load_instruction(4, SRAI(4, 3, 31));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FFFFu);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_srai_max_shift_cpu)  { return run_srai_max_shift<CPUH>(); }
TEST(exec_srai_max_shift_pipe) { return run_srai_max_shift<PipeH>(); }

template <typename Harness>
bool run_srai_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 256);
    cpu.load_instruction(0, SRAI(0, 1, 4));
    cpu.load_instruction(4, SRAI(2, 0, 4));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_srai_x0_cpu)  { return run_srai_x0<CPUH>(); }
TEST(exec_srai_x0_pipe) { return run_srai_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  2. ALU — Register-Register
// ═══════════════════════════════════════════════════════════════════════

// ── ADD ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_add()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, 20);
    cpu.load_instruction(0, ADD(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 30u);
    return true;
}

TEST(exec_add_cpu)  { return run_add<CPUH>(); }
TEST(exec_add_pipe) { return run_add<PipeH>(); }

template <typename Harness>
bool run_add_wraparound()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFF);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, ADD(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(exec_add_wraparound_cpu)  { return run_add_wraparound<CPUH>(); }
TEST(exec_add_wraparound_pipe) { return run_add_wraparound<PipeH>(); }

template <typename Harness>
bool run_add_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 22);

    cpu.load_instruction(0, ADD(2, 1, 0));
    cpu.load_instruction(4, ADD(3, 0, 1));
    cpu.load_instruction(8, ADD(0, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(2), 42u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 42u);

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_add_x0_cpu)  { return run_add_x0<CPUH>(); }
TEST(exec_add_x0_pipe) { return run_add_x0<PipeH>(); }

// ── SUB ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sub()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 50);
    cpu.set_reg(2, 20);

    cpu.load_instruction(0, SUB(3, 1, 2));
    cpu.load_instruction(4, SUB(4, 2, 1)); // commutative property does not hold

    h.step();
    ASSERT_EQ(cpu.reg(3), 30u);

    h.step();
    ASSERT_EQ(cpu.reg(4), static_cast<u32>(-30));
    return true;
}

TEST(exec_sub_cpu)  { return run_sub<CPUH>(); }
TEST(exec_sub_pipe) { return run_sub<PipeH>(); }

template <typename Harness>
bool run_sub_zero()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0);
    cpu.set_reg(2, 10);

    cpu.load_instruction(0, SUB(3, 1, 2));
    cpu.load_instruction(4, SUB(4, 2, 1));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-10));

    h.step();
    ASSERT_EQ(cpu.reg(4), 10u);
    return true;
}

TEST(exec_sub_zero_cpu)  { return run_sub_zero<CPUH>(); }
TEST(exec_sub_zero_pipe) { return run_sub_zero<PipeH>(); }

template <typename Harness>
bool run_sub_self()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(3, static_cast<u32>(-10));

    cpu.load_instruction(0, SUB(2, 1, 1));
    cpu.load_instruction(4, SUB(4, 3, 3));

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_sub_self_cpu)  { return run_sub_self<CPUH>(); }
TEST(exec_sub_self_pipe) { return run_sub_self<PipeH>(); }

template <typename Harness>
bool run_sub_underflow()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, SUB(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(exec_sub_underflow_cpu)  { return run_sub_underflow<CPUH>(); }
TEST(exec_sub_underflow_pipe) { return run_sub_underflow<PipeH>(); }

template <typename Harness>
bool run_sub_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(3, 5);

    cpu.load_instruction(0, SUB(2, 0, 1));
    cpu.load_instruction(4, SUB(0, 1, 3));

    h.step();
    ASSERT_EQ(cpu.reg(2), UINT_MAX - 9);

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_sub_x0_cpu)  { return run_sub_x0<CPUH>(); }
TEST(exec_sub_x0_pipe) { return run_sub_x0<PipeH>(); }

// ── AND ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_and()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFF00);
    cpu.set_reg(2, 0x0FF0);
    cpu.load_instruction(0, AND(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x0F00u);
    return true;
}

TEST(exec_and_cpu)  { return run_and<CPUH>(); }
TEST(exec_and_pipe) { return run_and<PipeH>(); }

template <typename Harness>
bool run_and_zero()
{
    // AND with zero -> clears all bits
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, AND(2, 1, 0));

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_and_zero_cpu)  { return run_and_zero<CPUH>(); }
TEST(exec_and_zero_pipe) { return run_and_zero<PipeH>(); }

template <typename Harness>
bool run_and_identity()
{
    // AND with self -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.set_reg(3, static_cast<u32>(-1));

    cpu.load_instruction(0, AND(2, 1, 1));
    cpu.load_instruction(4, AND(4, 1, 3));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEAD'BEEFu); // AND with self -> identity

    h.step();
    ASSERT_HEX_EQ(cpu.reg(4), 0xDEAD'BEEFu); // AND with -1 -> identity
    return true;
}

TEST(exec_and_identity_cpu)  { return run_and_identity<CPUH>(); }
TEST(exec_and_identity_pipe) { return run_and_identity<PipeH>(); }

template <typename Harness>
bool run_and_x0_dest()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xBEEF);
    cpu.set_reg(2, 0xFF);
    cpu.load_instruction(0, AND(0, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_and_x0_dest_cpu)  { return run_and_x0_dest<CPUH>(); }
TEST(exec_and_x0_dest_pipe) { return run_and_x0_dest<PipeH>(); }

// ── OR ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_or()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFF00);
    cpu.set_reg(2, 0x00FF);
    cpu.load_instruction(0, OR(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFFu);
    return true;
}

TEST(exec_or_cpu)  { return run_or<CPUH>(); }
TEST(exec_or_pipe) { return run_or<PipeH>(); }

template <typename Harness>
bool run_or_zero()
{
    // OR with zero -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.load_instruction(0, OR(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEAD'BEEFu);
    return true;
}

TEST(exec_or_zero_cpu)  { return run_or_zero<CPUH>(); }
TEST(exec_or_zero_pipe) { return run_or_zero<PipeH>(); }

template <typename Harness>
bool run_or_neg1()
{
    // OR with -1 -> sets all bits
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, OR(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(exec_or_neg1_cpu)  { return run_or_neg1<CPUH>(); }
TEST(exec_or_neg1_pipe) { return run_or_neg1<PipeH>(); }

template <typename Harness>
bool run_or_x0_dest()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1EE7);
    cpu.set_reg(2, 0xF00F);
    cpu.load_instruction(0, OR(0, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_or_x0_dest_cpu)  { return run_or_x0_dest<CPUH>(); }
TEST(exec_or_x0_dest_pipe) { return run_or_x0_dest<PipeH>(); }

// ── XOR ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_xor()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xF0F0);
    cpu.set_reg(2, 0xFF00);
    cpu.load_instruction(0, XOR(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x0FF0u);
    return true;
}

TEST(exec_xor_cpu)  { return run_xor<CPUH>(); }
TEST(exec_xor_pipe) { return run_xor<PipeH>(); }

template <typename Harness>
bool run_xor_zero()
{
    // XOR with 0 -> identity
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD);
    cpu.load_instruction(0, XOR(2, 1, 0));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEADu);
    return true;
}

TEST(exec_xor_zero_cpu)  { return run_xor_zero<CPUH>(); }
TEST(exec_xor_zero_pipe) { return run_xor_zero<PipeH>(); }

template <typename Harness>
bool run_xor_self()
{
    // XOR with self -> clears all bits
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.load_instruction(0, XOR(2, 1, 1));

    h.step();
    ASSERT_EQ(cpu.reg(2), 0u);
    return true;
}

TEST(exec_xor_self_cpu)  { return run_xor_self<CPUH>(); }
TEST(exec_xor_self_pipe) { return run_xor_self<PipeH>(); }

template <typename Harness>
bool run_xor_neg1()
{
    // XOR with -1 -> bitwise inverse
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.load_instruction(0, XOR(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), ~0xDEAD'BEEFu);
    return true;
}

TEST(exec_xor_neg1_cpu)  { return run_xor_neg1<CPUH>(); }
TEST(exec_xor_neg1_pipe) { return run_xor_neg1<PipeH>(); }

template <typename Harness>
bool run_xor_x0_dest()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFF00);
    cpu.set_reg(2, 0x00FF);
    cpu.load_instruction(0, XOR(0, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_xor_x0_dest_cpu)  { return run_xor_x0_dest<CPUH>(); }
TEST(exec_xor_x0_dest_pipe) { return run_xor_x0_dest<PipeH>(); }

// ── SLT ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_slt_true()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, SLT(3, 1, 2)); // -5 < 10

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_slt_true_cpu)  { return run_slt_true<CPUH>(); }
TEST(exec_slt_true_pipe) { return run_slt_true<PipeH>(); }

template <typename Harness>
bool run_slt_false()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, static_cast<u32>(-5));
    cpu.load_instruction(0, SLT(3, 1, 2)); // 10 < -5

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(exec_slt_false_cpu)  { return run_slt_false<CPUH>(); }
TEST(exec_slt_false_pipe) { return run_slt_false<PipeH>(); }

template <typename Harness>
bool run_slt_equal()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 42);
    cpu.load_instruction(0, SLT(3, 1, 2));
    cpu.load_instruction(4, SLT(4, 1, 1));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_slt_equal_cpu)  { return run_slt_equal<CPUH>(); }
TEST(exec_slt_equal_pipe) { return run_slt_equal<PipeH>(); }

template <typename Harness>
bool run_slt_both_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.set_reg(2, static_cast<u32>(-10));

    cpu.load_instruction(0, SLT(3, 1, 2)); // -5 < -10
    cpu.load_instruction(4, SLT(4, 2, 1)); // -10 < -5

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 1u);
    return true;
}

TEST(exec_slt_both_neg_cpu)  { return run_slt_both_neg<CPUH>(); }
TEST(exec_slt_both_neg_pipe) { return run_slt_both_neg<PipeH>(); }

template <typename Harness>
bool run_slt_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(10, 1);
    cpu.set_reg(11, static_cast<u32>(-1));

    cpu.load_instruction(0,  SLT(1, 0, 0));  // 0 < 0
    cpu.load_instruction(4,  SLT(2, 0, 10)); // 0 < 1
    cpu.load_instruction(8,  SLT(3, 0, 11)); // 0 < -1
    cpu.load_instruction(12, SLT(0, 4, 5));  // 4 < 5

    h.step();
    ASSERT_EQ(cpu.reg(1), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 1u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    cpu.set_reg(3, 4);
    cpu.set_reg(4, 5);

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_slt_x0_cpu)  { return run_slt_x0<CPUH>(); }
TEST(exec_slt_x0_pipe) { return run_slt_x0<PipeH>(); }

// ── SLTU ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sltu_true()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 10);
    cpu.load_instruction(0, SLTU(3, 1, 2)); // 5 < 10

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_sltu_true_cpu)  { return run_sltu_true<CPUH>(); }
TEST(exec_sltu_true_pipe) { return run_sltu_true<PipeH>(); }

template <typename Harness>
bool run_sltu_false()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, SLTU(3, 1, 2)); // 10 < 5

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(exec_sltu_false_cpu)  { return run_sltu_false<CPUH>(); }
TEST(exec_sltu_false_pipe) { return run_sltu_false<PipeH>(); }

template <typename Harness>
bool run_sltu_unsigned_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, static_cast<u32>(-5)); // sign-extended to 0xFFFF'FFFB
    cpu.load_instruction(0, SLTU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_sltu_unsigned_neg_cpu)  { return run_sltu_unsigned_neg<CPUH>(); }
TEST(exec_sltu_unsigned_neg_pipe) { return run_sltu_unsigned_neg<PipeH>(); }

template <typename Harness>
bool run_sltu_both_neg()
{
    // Both get sign-extended; order remains the same
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-6)); // 0xFFFF'FFFA
    cpu.set_reg(2, static_cast<u32>(-5)); // 0xFFFF'FFFB
    cpu.load_instruction(0, SLTU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_sltu_both_neg_cpu)  { return run_sltu_both_neg<CPUH>(); }
TEST(exec_sltu_both_neg_pipe) { return run_sltu_both_neg<PipeH>(); }

template <typename Harness>
bool run_sltu_equal()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(4, static_cast<u32>(-42));
    cpu.set_reg(1, 42);
    cpu.set_reg(2, 42);

    cpu.load_instruction(0, SLTU(3, 1, 2));
    cpu.load_instruction(4, SLTU(5, 4, 4));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(5), 0u);
    return true;
}

TEST(exec_sltu_equal_cpu)  { return run_sltu_equal<CPUH>(); }
TEST(exec_sltu_equal_pipe) { return run_sltu_equal<PipeH>(); }

template <typename Harness>
bool run_sltu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 10);
    cpu.set_reg(5, static_cast<u32>(-5)); // 0xFFFF'FFFB

    cpu.load_instruction(0,  SLTU(0, 1, 2)); // 5 < 10
    cpu.load_instruction(4,  SLTU(3, 0, 0)); // 0 < 0
    cpu.load_instruction(8,  SLTU(4, 0, 1)); // 0 < 5
    cpu.load_instruction(12, SLTU(6, 0, 5)); // 0 < u32(-5)

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 1u);

    h.step();
    ASSERT_EQ(cpu.reg(6), 1u);
    return true;
}

TEST(exec_sltu_x0_cpu)  { return run_sltu_x0<CPUH>(); }
TEST(exec_sltu_x0_pipe) { return run_sltu_x0<PipeH>(); }

// ── SLL ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sll()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, SLL(3, 1, 2)); // 4 << 1

    h.step();
    ASSERT_EQ(cpu.reg(3), 16u);
    return true;
}

TEST(exec_sll_cpu)  { return run_sll<CPUH>(); }
TEST(exec_sll_pipe) { return run_sll<PipeH>(); }

template <typename Harness>
bool run_sll_zero_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, SLL(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xDEAD'BEEFu);
    return true;
}

TEST(exec_sll_zero_shift_cpu)  { return run_sll_zero_shift<CPUH>(); }
TEST(exec_sll_zero_shift_pipe) { return run_sll_zero_shift<PipeH>(); }

template <typename Harness>
bool run_sll_max_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.set_reg(2, 31);
    cpu.load_instruction(0, SLL(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x8000'0000u);
    return true;
}

TEST(exec_sll_max_shift_cpu)  { return run_sll_max_shift<CPUH>(); }
TEST(exec_sll_max_shift_pipe) { return run_sll_max_shift<PipeH>(); }

template <typename Harness>
bool run_sll_masks_shamt()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.set_reg(2, 32);
    cpu.set_reg(4, 40);

    cpu.load_instruction(0, SLL(3, 1, 2)); // rs2 = 32 -> effective shift = 0
    cpu.load_instruction(4, SLL(5, 1, 4)); // rs2 = 40 -> effective shift = 0

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x1234'5678u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(5), 0x3456'7800u); // upper bits discarded
    return true;
}

TEST(exec_sll_masks_shamt_cpu)  { return run_sll_masks_shamt<CPUH>(); }
TEST(exec_sll_masks_shamt_pipe) { return run_sll_masks_shamt<PipeH>(); }

template <typename Harness>
bool run_sll_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xCAFE'BABE);
    cpu.set_reg(2, 8);

    cpu.load_instruction(0, SLL(0, 1, 2));
    cpu.load_instruction(4, SLL(3, 1, 0));
    cpu.load_instruction(8, SLL(4, 0, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xCAFE'BABEu);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_sll_x0_cpu)  { return run_sll_x0<CPUH>(); }
TEST(exec_sll_x0_pipe) { return run_sll_x0<PipeH>(); }

// ── SRL ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_srl()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x80);
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, SRL(3, 1, 2)); // 0x80 >> 4

    h.step();
    ASSERT_EQ(cpu.reg(3), 8u);
    return true;
}

TEST(exec_srl_cpu)  { return run_srl<CPUH>(); }
TEST(exec_srl_pipe) { return run_srl<PipeH>(); }

template <typename Harness>
bool run_srl_zero_fill()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, SRL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0x4000'0000u);
    return true;
}

TEST(exec_srl_zero_fill_cpu)  { return run_srl_zero_fill<CPUH>(); }
TEST(exec_srl_zero_fill_pipe) { return run_srl_zero_fill<PipeH>(); }

template <typename Harness>
bool run_srl_zero_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xDEAD'BEEF);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, SRL(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xDEAD'BEEFu);
    return true;
}

TEST(exec_srl_zero_shift_cpu)  { return run_srl_zero_shift<CPUH>(); }
TEST(exec_srl_zero_shift_pipe) { return run_srl_zero_shift<PipeH>(); }

template <typename Harness>
bool run_srl_max_shift()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(2, 31);
    cpu.load_instruction(0, SRL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(exec_srl_max_shift_cpu)  { return run_srl_max_shift<CPUH>(); }
TEST(exec_srl_max_shift_pipe) { return run_srl_max_shift<PipeH>(); }

template <typename Harness>
bool run_srl_masks_shamt()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x1234'5678);
    cpu.set_reg(2, 32);
    cpu.set_reg(4, 40);

    cpu.load_instruction(0, SRL(3, 1, 2)); // rs2 = 32 -> shamt = 0
    cpu.load_instruction(4, SRL(5, 1, 4)); // rs2 = 40 -> shamt = 8

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x1234'5678u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(5), 0x0012'3456u);
    return true;
}

TEST(exec_srl_masks_shamt_cpu)  { return run_srl_masks_shamt<CPUH>(); }
TEST(exec_srl_masks_shamt_pipe) { return run_srl_masks_shamt<PipeH>(); }

template <typename Harness>
bool run_srl_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xA110'CA7E);
    cpu.set_reg(2, 12);

    cpu.load_instruction(0, SRL(0, 1, 2));
    cpu.load_instruction(4, SRL(3, 1, 0));
    cpu.load_instruction(8, SRL(4, 0, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xA110'CA7Eu);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_srl_x0_cpu)  { return run_srl_x0<CPUH>(); }
TEST(exec_srl_x0_pipe) { return run_srl_x0<PipeH>(); }

// ── SRA ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sra_positive()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x7FFF'0000);
    cpu.set_reg(2, 16);
    cpu.load_instruction(0, SRA(3, 1, 2)); // 0x7FFF'000 >> 16

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x0000'7FFFu);
    return true;
}

TEST(exec_sra_positive_cpu)  { return run_sra_positive<CPUH>(); }
TEST(exec_sra_positive_pipe) { return run_sra_positive<PipeH>(); }

template <typename Harness>
bool run_sra_negative()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-128)); // 0xFFFF'FF80
    cpu.set_reg(2, 4);
    cpu.load_instruction(0, SRA(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-8)); // 0xFFFF'FFF8
    return true;
}

TEST(exec_sra_negative_cpu)  { return run_sra_negative<CPUH>(); }
TEST(exec_sra_negative_pipe) { return run_sra_negative<PipeH>(); }

template <typename Harness>
bool run_sra_neg1()
{
    // -1 stays the same
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 8);
    cpu.load_instruction(0, SRA(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-1));
    return true;
}

TEST(exec_sra_neg1_cpu)  { return run_sra_neg1<CPUH>(); }
TEST(exec_sra_neg1_pipe) { return run_sra_neg1<PipeH>(); }

template <typename Harness>
bool run_sra_masks_shamt()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(2, 33);
    cpu.set_reg(4, 0x7000'0000);
    cpu.set_reg(5, 40);

    cpu.load_instruction(0, SRA(3, 1, 2)); // rs2 = 33 -> shamt = 1
    cpu.load_instruction(4, SRA(6, 4, 5)); // rs2 = 40 -> shamt = 8

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xC000'0000u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(6), 0x0070'0000u);
    return true;
}

TEST(exec_sra_masks_shamt_cpu)  { return run_sra_masks_shamt<CPUH>(); }
TEST(exec_sra_masks_shamt_pipe) { return run_sra_masks_shamt<PipeH>(); }

template <typename Harness>
bool run_sra_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xA110'CA7E);
    cpu.set_reg(2, 12);

    cpu.load_instruction(0, SRA(0, 1, 2));
    cpu.load_instruction(4, SRA(3, 1, 0));
    cpu.load_instruction(8, SRA(4, 0, 2));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xA110'CA7Eu);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_sra_x0_cpu)  { return run_sra_x0<CPUH>(); }
TEST(exec_sra_x0_pipe) { return run_sra_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  3. Load
// ═══════════════════════════════════════════════════════════════════════

// ── LW ─────────────────────────────────────────────────────────────────

template<typename Harness>
bool run_lw()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0xDEAD'BEEF);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LW(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xDEAD'BEEFu);
    return true;
}

TEST(exec_lw_cpu)  { return run_lw<CPUH>(); }
TEST(exec_lw_pipe) { return run_lw<PipeH>(); }

template <typename Harness>
bool run_lw_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x108, 0x1234'5678);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LW(2, 8, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234'5678u);
    return true;
}

TEST(exec_lw_offset_cpu)  { return run_lw_offset<CPUH>(); }
TEST(exec_lw_offset_pipe) { return run_lw_offset<PipeH>(); }

template <typename Harness>
bool run_lw_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(96, 0xAAAA'5555);
    cpu.set_reg(1, 100);
    cpu.load_instruction(0, LW(2, -4, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xAAAA'5555u);
    return true;
}

TEST(exec_lw_neg_offset_cpu)  { return run_lw_neg_offset<CPUH>(); }
TEST(exec_lw_neg_offset_pipe) { return run_lw_neg_offset<PipeH>(); }

template <typename Harness>
bool run_lw_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0x1234'5678);
    cpu.set_reg(1, 100);

    cpu.load_instruction(0, LW(0,     0, 1));
    cpu.load_instruction(4, LW(2, 0x100, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(2), 0x1234'5678u);
    return true;
}

TEST(exec_lw_x0_cpu)  { return run_lw_x0<CPUH>(); }
TEST(exec_lw_x0_pipe) { return run_lw_x0<PipeH>(); }

template <typename Harness>
bool run_lw_misaligned()
{
    for (u32 addr : {0x101u, 0x102u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.memory().write32(0x100, 0xDEAD'BEEF);
        cpu.set_reg(1, addr);
        cpu.set_reg(3, 0);

        cpu.load_instruction(0, LW(2, 0, 1));
        // This padding prevents the pipeline from halting by fetching
        // uninitialized memory before the faulting load reaches the
        // MEM stage
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

        // WARNING TO SELF: DON'T USE step() FOR THIS KIND OF TEST
        //                  DON'T TRY TO CHANGE step()
        //                  DON'T TRY TO CHANGE run_instructions()
        //
        // step() works perfectly for non-faulting tests, but WILL loop
        // indefinitely on faulting tests. I tried to add a "if (halted_)"
        // condition to run_instructions() -- this will make every other
        // test (without padding instructions) fail because the loop breaks
        // before WB (from fetch to uninitialized memory), so nothing retires.
        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
    }
    return true;
}

TEST(exec_lw_misaligned_cpu)  { return run_lw_misaligned<CPUH>(); }
TEST(exec_lw_misaligned_pipe) { return run_lw_misaligned<PipeH>(); }

// ── LH ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_lh()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0x1234);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LH(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234u);
    return true;
}

TEST(exec_lh_cpu)  { return run_lh<CPUH>(); }
TEST(exec_lh_pipe) { return run_lh<PipeH>(); }

template <typename Harness>
bool run_lh_sign_extend()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xFF00);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LH(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FF00u);
    return true;
}

TEST(exec_lh_sign_extend_cpu)  { return run_lh_sign_extend<CPUH>(); }
TEST(exec_lh_sign_extend_pipe) { return run_lh_sign_extend<PipeH>(); }

template <typename Harness>
bool run_lh_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x102, 0x5678);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LH(2, 2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x5678u);
    return true;
}

TEST(exec_lh_offset_cpu)  { return run_lh_sign_extend<CPUH>(); }
TEST(exec_lh_offset_pipe) { return run_lh_sign_extend<PipeH>(); }

template <typename Harness>
bool run_lh_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xBEEF);
    cpu.set_reg(1, 0x102);
    cpu.load_instruction(0, LH(2, -2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'BEEFu);
    return true;
}

TEST(exec_lh_neg_offset_cpu)  { return run_lh_neg_offset<CPUH>(); }
TEST(exec_lh_neg_offset_pipe) { return run_lh_neg_offset<PipeH>(); }

template <typename Harness>
bool run_lh_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xCAFE);
    cpu.set_reg(1, 0x100);

    cpu.load_instruction(0, LH(0, 0, 1));
    cpu.load_instruction(4, LH(2, 0x100, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'CAFEu);
    return true;
}

TEST(exec_lh_x0_cpu)  { return run_lh_x0<CPUH>(); }
TEST(exec_lh_x0_pipe) { return run_lh_x0<PipeH>(); }

template <typename Harness>
bool run_lh_misaligned()
{
    for (u32 addr : {0x101u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.memory().write16(0x100, 0x1EE7);
        cpu.set_reg(1, addr);

        cpu.load_instruction(0,  LH(2, 0, 1));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
    }
    return true;
}

TEST(exec_lh_misaligned_cpu)  { return run_lh_misaligned<CPUH>(); }
TEST(exec_lh_misaligned_pipe) { return run_lh_misaligned<PipeH>(); }

// ── LHU ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_lhu()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0x1234);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LHU(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x1234u);
    return true;
}

TEST(exec_lhu_cpu)  { return run_lhu<CPUH>(); }
TEST(exec_lhu_pipe) { return run_lhu<PipeH>(); }

template <typename Harness>
bool run_lhu_no_sign_extend()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xFF00);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LHU(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFF00u);
    return true;
}

TEST(exec_lhu_no_sign_extend_cpu)  { return run_lhu_no_sign_extend<CPUH>(); }
TEST(exec_lhu_no_sign_extend_pipe) { return run_lhu_no_sign_extend<PipeH>(); }

template <typename Harness>
bool run_lhu_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x102, 0x5678);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LHU(2, 2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x5678u);
    return true;
}

TEST(exec_lhu_offset_cpu)  { return run_lhu_offset<CPUH>(); }
TEST(exec_lhu_offset_pipe) { return run_lhu_offset<PipeH>(); }

template <typename Harness>
bool run_lhu_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xBEEF);
    cpu.set_reg(1, 0x102);
    cpu.load_instruction(0, LHU(2, -2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xBEEFu);
    return true;
}

TEST(exec_lhu_neg_offset_cpu)  { return run_lhu_neg_offset<CPUH>(); }
TEST(exec_lhu_neg_offset_pipe) { return run_lhu_neg_offset<PipeH>(); }

template <typename Harness>
bool run_lhu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write16(0x100, 0xCAFE);
    cpu.set_reg(1, 0x100);

    cpu.load_instruction(0, LHU(0, 0, 1));
    cpu.load_instruction(4, LHU(2, 0x100, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xCAFEu);
    return true;
}

TEST(exec_lhu_x0_cpu)  { return run_lhu_x0<CPUH>(); }
TEST(exec_lhu_x0_pipe) { return run_lhu_x0<PipeH>(); }

template <typename Harness>
bool run_lhu_misaligned()
{
    for (u32 addr : {0x101u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.memory().write16(0x100, 0x1EE7);
        cpu.set_reg(1, addr);

        cpu.load_instruction(0,  LHU(2, 0, 1));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
    }
    return true;
}

TEST(exec_lhu_misaligned_cpu)  { return run_lhu_misaligned<CPUH>(); }
TEST(exec_lhu_misaligned_pipe) { return run_lhu_misaligned<PipeH>(); }

// ── LB ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_lb()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x100, 0x12);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LB(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x12u);
    return true;
}

TEST(exec_lb_cpu)  { return run_lb<CPUH>(); }
TEST(exec_lb_pipe) { return run_lb<PipeH>(); }

template <typename Harness>
bool run_lb_sign_extend()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x101, 0xFF);
    cpu.set_reg(1, 0x101);
    cpu.load_instruction(0, LB(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FFFFu);
    return true;
}

TEST(exec_lb_sign_extend_cpu)  { return run_lb_sign_extend<CPUH>(); }
TEST(exec_lb_sign_extend_pipe) { return run_lb_sign_extend<PipeH>(); }

template <typename Harness>
bool run_lb_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x102, 0x56);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LB(2, 2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x56u);
    return true;
}

TEST(exec_lb_offset_cpu)  { return run_lb_offset<CPUH>(); }
TEST(exec_lb_offset_pipe) { return run_lb_offset<PipeH>(); }

template <typename Harness>
bool run_lb_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x103, 0xBE);
    cpu.set_reg(1, 0x105);
    cpu.load_instruction(0, LB(2, -2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FFBEu);
    return true;
}

TEST(exec_lb_neg_offset_cpu)  { return run_lb_neg_offset<CPUH>(); }
TEST(exec_lb_neg_offset_pipe) { return run_lb_neg_offset<PipeH>(); }

template <typename Harness>
bool run_lb_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x100, 0xEF);
    cpu.set_reg(1, 0x100);

    cpu.load_instruction(0, LB(0, 0, 1));
    cpu.load_instruction(4, LB(2, 0x100, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFFF'FFEFu);
    return true;
}

TEST(exec_lb_x0_cpu)  { return run_lb_x0<CPUH>(); }
TEST(exec_lb_x0_pipe) { return run_lb_x0<PipeH>(); }

// ── LBU ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_lbu()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x100, 0x12);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LBU(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x12u);
    return true;
}

TEST(exec_lbu_cpu)  { return run_lbu<CPUH>(); }
TEST(exec_lbu_pipe) { return run_lbu<PipeH>(); }

template <typename Harness>
bool run_lbu_no_sign_extend()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x101, 0xFF);
    cpu.set_reg(1, 0x101);
    cpu.load_instruction(0, LBU(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xFFu);
    return true;
}

TEST(exec_lbu_no_sign_extend_cpu)  { return run_lbu_no_sign_extend<CPUH>(); }
TEST(exec_lbu_no_sign_extend_pipe) { return run_lbu_no_sign_extend<PipeH>(); }

template <typename Harness>
bool run_lbu_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x102, 0x56);
    cpu.set_reg(1, 0x100);
    cpu.load_instruction(0, LBU(2, 2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0x56u);
    return true;
}

TEST(exec_lbu_offset_cpu)  { return run_lbu_offset<CPUH>(); }
TEST(exec_lbu_offset_pipe) { return run_lbu_offset<PipeH>(); }

template <typename Harness>
bool run_lbu_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x103, 0xBE);
    cpu.set_reg(1, 0x105);
    cpu.load_instruction(0, LBU(2, -2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xBEu);
    return true;
}

TEST(exec_lbu_neg_offset_cpu)  { return run_lbu_neg_offset<CPUH>(); }
TEST(exec_lbu_neg_offset_pipe) { return run_lbu_neg_offset<PipeH>(); }

template <typename Harness>
bool run_lbu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write8(0x100, 0xCA);
    cpu.set_reg(1, 0x100);

    cpu.load_instruction(0, LBU(0, 0, 1));
    cpu.load_instruction(4, LBU(2, 0x100, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(2), 0xCAu);
    return true;
}

TEST(exec_lbu_x0_cpu)  { return run_lbu_x0<CPUH>(); }
TEST(exec_lbu_x0_pipe) { return run_lbu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  4. Stores
// ═══════════════════════════════════════════════════════════════════════

// ── SW ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sw()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xCAFE'BABE);
    cpu.load_instruction(0, SW(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xCAFE'BABEu);
    return true;
}

TEST(exec_sw_cpu)  { return run_sw<CPUH>(); }
TEST(exec_sw_pipe) { return run_sw<PipeH>(); }

template <typename Harness>
bool run_sw_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x1234'5678);
    cpu.load_instruction(0, SW(2, 4, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x104).value, 0x1234'5678u);
    return true;
}

TEST(exec_sw_offset_cpu)  { return run_sw_offset<CPUH>(); }
TEST(exec_sw_offset_pipe) { return run_sw_offset<PipeH>(); }

template <typename Harness>
bool run_sw_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x108);
    cpu.set_reg(2, 0xDEAD'BEEF);
    cpu.load_instruction(0, SW(2, -8, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xDEAD'BEEFu);
    return true;
}

TEST(exec_sw_neg_offset_cpu)  { return run_sw_neg_offset<CPUH>(); }
TEST(exec_sw_neg_offset_pipe) { return run_sw_neg_offset<PipeH>(); }

template <typename Harness>
bool run_sw_overwrites()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0x1234'5678);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xABCD'EF01u);
    cpu.load_instruction(0, SW(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xABCD'EF01u);
    return true;
}

TEST(exec_sw_overwrites_cpu)  { return run_sw_overwrites<CPUH>(); }
TEST(exec_sw_overwrites_pipe) { return run_sw_overwrites<PipeH>(); }

template <typename Harness>
bool run_sw_no_corruption()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0xFC,  0xCAFE'BABE);
    cpu.memory().write32(0x104, 0xDEAD'BEEF);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x1234'5678);
    cpu.load_instruction(0, SW(2, 0, 1));
    
    h.step();
    // Neighbors don't get corrupted
    ASSERT_HEX_EQ(cpu.memory().read32(0xFC).value,  0xCAFE'BABEu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0x1234'5678u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x104).value, 0xDEAD'BEEFu);
    return true;
}

TEST(exec_sw_no_corruption_cpu)  { return run_sw_no_corruption<CPUH>(); }
TEST(exec_sw_no_corruption_pipe) { return run_sw_no_corruption<PipeH>(); }

template <typename Harness>
bool run_sw_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0x1234'5678);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x5678'1234u);
    cpu.load_instruction(0, SW(0, 0, 1));
    cpu.load_instruction(4, SW(2, 0x200, 0));

    h.step();
    ASSERT_EQ(cpu.memory().read32(0x100).value, 0u);
    
    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0x200).value, 0x5678'1234u);
    return true;
}

TEST(exec_sw_x0_cpu)  { return run_sw_x0<CPUH>(); }
TEST(exec_sw_x0_pipe) { return run_sw_x0<PipeH>(); }

template <typename Harness>
bool run_sw_misaligned()
{
    for (u32 addr : {0x101u, 0x102u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.set_reg(1, addr);
        cpu.set_reg(2, 0x8BAD'F00D);

        cpu.load_instruction(0,  SW(2, 0, 1));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
    }
    return true;
}

TEST(exec_sw_misaligned_cpu)  { return run_sw_misaligned<CPUH>(); }
TEST(exec_sw_misaligned_pipe) { return run_sw_misaligned<PipeH>(); }

// ── SH ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sh()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x1234);
    cpu.load_instruction(0, SH(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read16(0x100).value, 0x1234u);
    return true;
}

TEST(exec_sh_cpu)  { return run_sh<CPUH>(); }
TEST(exec_sh_pipe) { return run_sh<PipeH>(); }

template <typename Harness>
bool run_sh_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0x5678);
    cpu.load_instruction(0, SH(2, 4, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read16(0x104).value, 0x5678u);
    return true;
}

TEST(exec_sh_offset_cpu)  { return run_sh_offset<CPUH>(); }
TEST(exec_sh_offset_pipe) { return run_sh_offset<PipeH>(); }

template <typename Harness>
bool run_sh_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x108);
    cpu.set_reg(2, 0xABCD);
    cpu.load_instruction(0, SH(2, -8, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read16(0x100).value, 0xABCDu);
    return true;
}

TEST(exec_sh_neg_offset_cpu)  { return run_sh_neg_offset<CPUH>(); }
TEST(exec_sh_neg_offset_pipe) { return run_sh_neg_offset<PipeH>(); }

template <typename Harness>
bool run_sh_truncates()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0x100, 0x0000'0000);
    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xDEAD'BEEF);
    cpu.load_instruction(0, SH(2, 0, 1));

    h.step();
    // SH should write only the low two bytes
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0x0000'BEEFu);
    return true;
}

TEST(exec_sh_truncates_cpu)  { return run_sh_truncates<CPUH>(); }
TEST(exec_sh_truncates_pipe) { return run_sh_truncates<PipeH>(); }

template <typename Harness>
bool run_sh_overwrites()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0xFC,  0x1234'ABCD);
    cpu.memory().write32(0x100, 0xDEAD'BEEF);
    cpu.memory().write32(0x104, 0xABCD'1234);

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xCAFE);

    // offset = 2 -> writes upper 2 bits of 0x100
    cpu.load_instruction(0, SH(2, 2, 1)); 

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0xFC).value,  0x1234'ABCDu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xCAFE'BEEFu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x104).value, 0xABCD'1234u);
    return true;
}

TEST(exec_sh_overwrites_cpu)  { return run_sh_overwrites<CPUH>(); }
TEST(exec_sh_overwrites_pipe) { return run_sh_overwrites<PipeH>(); }

template <typename Harness>
bool run_sh_misaligned()
{
    for (u32 addr : {0x101u, 0x103u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.set_reg(1, addr);
        cpu.set_reg(2, 0xF00D);

        cpu.load_instruction(0,  SH(2, 0, 1));
        cpu.load_instruction(4,  ADDI(3, 0, 10));
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));

        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
    }
    return true;
}

TEST(exec_sh_misaligned_cpu)  { return run_sh_misaligned<CPUH>(); }
TEST(exec_sh_misaligned_pipe) { return run_sh_misaligned<PipeH>(); }

// ── SB ─────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_sb()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xAB);
    cpu.load_instruction(0, SB(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read8(0x100).value, 0xABu);
    return true;
}

TEST(exec_sb_cpu)  { return run_sb<CPUH>(); }
TEST(exec_sb_pipe) { return run_sb<PipeH>(); }

template <typename Harness>
bool run_sb_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x100);
    cpu.set_reg(2, 0xCD);
    cpu.load_instruction(0, SB(2, 1, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read8(0x101).value, 0xCDu);
    return true;
}

TEST(exec_sb_offset_cpu)  { return run_sb_offset<CPUH>(); }
TEST(exec_sb_offset_pipe) { return run_sb_offset<PipeH>(); }

template <typename Harness>
bool run_sb_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x104);
    cpu.set_reg(2, 0xEF);
    cpu.load_instruction(0, SB(2, -2, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read8(0x102).value, 0xEFu);
    return true;
}

TEST(exec_sb_neg_offset_cpu)  { return run_sb_neg_offset<CPUH>(); }
TEST(exec_sb_neg_offset_pipe) { return run_sb_neg_offset<PipeH>(); }

template <typename Harness>
bool run_sb_truncates()
{
    Harness h;
    auto& cpu = h.get();

    cpu.memory().write32(0xFC,  0xDEAD'C0DE);
    cpu.memory().write32(0x100, 0x1234'5678);
    cpu.memory().write32(0x104, 0xFEED'FACE);

    cpu.set_reg(1, 0x103);
    cpu.set_reg(2, 0xDEAD'BEEF);
    cpu.load_instruction(0, SB(2, 0, 1));

    h.step();
    ASSERT_HEX_EQ(cpu.memory().read32(0xFC).value,  0xDEAD'C0DEu);
    ASSERT_HEX_EQ(cpu.memory().read32(0x100).value, 0xEF34'5678u);
    ASSERT_HEX_EQ(cpu.memory().read32(0x104).value, 0xFEED'FACEu);
    return true;
}

TEST(exec_sb_truncates_cpu)  { return run_sb_truncates<CPUH>(); }
TEST(exec_sb_truncates_pipe) { return run_sb_truncates<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  4. Branches
// ═══════════════════════════════════════════════════════════════════════

// ── Common branch logic ────────────────────────────────────────────────

template <typename Harness>
bool run_branch_forward()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);

    cpu.load_instruction(0, BEQ(1, 2, 8));
    cpu.load_instruction(4, ADDI(10, 0, 10));
    cpu.load_instruction(8, ADDI(11, 0, 11));

    h.run(2);
    ASSERT_EQ(cpu.reg(10), 0u);
    ASSERT_EQ(cpu.reg(11), 11u);
    return true;
}

TEST(exec_branch_taken_forward_cpu)  { return run_branch_forward<CPUH>(); }
TEST(exec_branch_taken_forward_pipe) { return run_branch_forward<PipeH>(); }

template <typename Harness>
bool run_branch_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(1, 10);

    cpu.load_instruction(0,  BEQ(1, 2, 16));
    cpu.load_instruction(4,  ADDI(10, 0, 10));
    cpu.load_instruction(16, ADDI(11, 0, 11));

    h.run(2);
    ASSERT_EQ(cpu.reg(10), 10u);
    ASSERT_EQ(cpu.reg(11), 0u);
    return true;
}

TEST(exec_branch_not_taken_cpu)  { return run_branch_not_taken<CPUH>(); }
TEST(exec_branch_not_taken_pipe) { return run_branch_not_taken<PipeH>(); }

template <typename Harness>
bool run_branch_backward()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_pc(0x100);
    cpu.set_reg(1, 1);
    cpu.set_reg(2, 1);

    cpu.load_instruction(0x100, BEQ(1, 2, -8));
    cpu.load_instruction(0xF8,  ADDI(10, 0, 10));
    cpu.load_instruction(0x104, ADDI(11, 0, 11));
    
    h.run(2);
    ASSERT_EQ(cpu.reg(10), 10u);
    ASSERT_EQ(cpu.reg(11), 0u);
    return true;
}

TEST(exec_branch_taken_backward_cpu)  { return run_branch_backward<CPUH>(); }
TEST(exec_branch_taken_backward_pipe) { return run_branch_backward<PipeH>(); }

template <typename Harness>
bool run_branch_zero_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.set_reg(2, 1);
    cpu.load_instruction(0, BEQ(1, 2, 0)); // branches to self (PC 0)

    h.run(2);
    ASSERT_EQ(cpu.stats().branches_taken, 2u);
    return true;
}

TEST(exec_branch_zero_offset_cpu)  { return run_branch_zero_offset<CPUH>(); }
TEST(exec_branch_zero_offset_pipe) { return run_branch_zero_offset<PipeH>(); }

template <typename Harness>
bool run_branch_misaligned()
{
    for (u32 offset : {10u, 11u})
    {
        Harness h;
        auto& cpu = h.get();

        cpu.set_reg(1, 10);
        cpu.set_reg(2, 10);

        cpu.load_instruction(0,  BEQ(1, 2, 20 + static_cast<i32>(offset)));
        cpu.load_instruction(4,  ADDI(3, 0, 10)); // fallthrough — shouldn't execute
        cpu.load_instruction(8,  ADDI(3, 3, 10));
        cpu.load_instruction(12, ADDI(3, 3, 10));
        cpu.load_instruction(16, ADDI(3, 3, 10));
        cpu.load_instruction(20, ADDI(3, 3, 10));
        // Misaligned target — shouldn't execute
        cpu.load_instruction(20 + offset, ADDI(4, 0, 42));

        cpu.run_until([&]() { return cpu.halted(); }, 20);
        ASSERT(cpu.halted());
        ASSERT_EQ(cpu.reg(3), 0u);
        ASSERT_EQ(cpu.reg(4), 0u);
    }
    return true;
}

TEST(exec_branch_misaligned_cpu)  { return run_branch_misaligned<CPUH>(); }
TEST(exec_branch_misaligned_pipe) { return run_branch_misaligned<PipeH>(); }

template <typename Harness>
bool run_branch_clears_lsb()
{
    // offset = 9 -> effective offset = 8
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 20);
    cpu.set_reg(2, 20);

    cpu.load_instruction(0, BEQ(1, 2, 9));
    cpu.load_instruction(4, ADDI(3, 0, 42));
    cpu.load_instruction(8, ADDI(4, 0, 42));

    h.run(2);
    ASSERT_EQ(cpu.reg(3), 0u);
    ASSERT_EQ(cpu.reg(4), 42u);
    return true;
}

TEST(exec_branch_clears_lsb_cpu)  { return run_branch_clears_lsb<CPUH>(); }
TEST(exec_branch_clears_lsb_pipe) { return run_branch_clears_lsb<PipeH>(); }

// ── BEQ ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_beq_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 42);
    cpu.set_reg(3, static_cast<u32>(-42));
    cpu.set_reg(4, static_cast<u32>(-42));

    cpu.load_instruction(0,  BEQ(1, 2, 16));
    cpu.load_instruction(16, BEQ(3, 4, 16));
    cpu.load_instruction(32, ADDI(5, 0, 42));

    h.run(3);
    ASSERT_EQ(cpu.reg(5), 42u);
    return true;
}

TEST(exec_beq_taken_cpu)  { return run_beq_taken<CPUH>(); }
TEST(exec_beq_taken_pipe) { return run_beq_taken<PipeH>(); }

template <typename Harness>
bool run_beq_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 43);
    cpu.set_reg(3, static_cast<u32>(-42));
    cpu.set_reg(4, static_cast<u32>(-43));

    cpu.load_instruction(0,  BEQ(1, 2, 16));
    cpu.load_instruction(16, BEQ(3, 4, 16));
    cpu.load_instruction(32, ADDI(6, 0, 42));
    cpu.load_instruction(4,  ADDI(5, 0, 10));
    cpu.load_instruction(8,  ADDI(5, 5, 10));

    h.run(3);
    ASSERT_EQ(cpu.reg(6), 0u);
    ASSERT_EQ(cpu.reg(5), 20u);
    return true;
}

TEST(exec_beq_not_taken_cpu)  { return run_beq_not_taken<CPUH>(); }
TEST(exec_beq_not_taken_pipe) { return run_beq_not_taken<PipeH>(); }

template <typename Harness>
bool run_beq_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, BEQ(0, 0, 8));
    cpu.load_instruction(4, ADDI(1, 0, 42));
    cpu.load_instruction(8, ADDI(2, 0, 42));

    h.run(2);
    ASSERT_EQ(cpu.reg(1), 0u);
    ASSERT_EQ(cpu.reg(2), 42u);
    return true;
}

TEST(exec_beq_x0_cpu)  { return run_beq_x0<CPUH>(); }
TEST(exec_beq_x0_pipe) { return run_beq_x0<PipeH>(); }

// ── BNE ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_bne_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, 20);

    cpu.load_instruction(0, BNE(1, 2, 8));
    cpu.load_instruction(4, ADDI(3, 0, 42));
    cpu.load_instruction(8, ADDI(4, 0, 42));

    h.run(2);
    ASSERT_EQ(cpu.reg(3), 0u);
    ASSERT_EQ(cpu.reg(4), 42u);
    return true;
}

TEST(exec_bne_taken_cpu)  { return run_bne_taken<CPUH>(); }
TEST(exec_bne_taken_pipe) { return run_bne_taken<PipeH>(); }

template <typename Harness>
bool run_bne_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, 10);

    cpu.load_instruction(0, BNE(1, 2, 8));
    cpu.load_instruction(4, ADDI(3, 0, 42));
    cpu.load_instruction(8, ADDI(4, 0, 42));

    h.run(2);
    ASSERT_EQ(cpu.reg(3), 42u);
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_bne_not_taken_cpu)  { return run_bne_not_taken<CPUH>(); }
TEST(exec_bne_not_taken_pipe) { return run_bne_not_taken<PipeH>(); }

template <typename Harness>
bool run_bne_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-5));

    cpu.load_instruction(0,  BNE(0, 0, 16));
    cpu.load_instruction(4,  BNE(0, 1, 16));
    cpu.load_instruction(20, BNE(0, 2, 16));
    cpu.load_instruction(36, ADDI(3, 0, 42));

    h.run(4);
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(exec_bne_x0_cpu)  { return run_bne_x0<CPUH>(); }
TEST(exec_bne_x0_pipe) { return run_bne_x0<PipeH>(); }

// ── BLT ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_blt_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 3);
    cpu.set_reg(2, 5);
    cpu.set_reg(3, static_cast<u32>(-5));
    cpu.set_reg(4, static_cast<u32>(-3));

    cpu.load_instruction(0,  BLT(1, 2, 12));
    cpu.load_instruction(12, BLT(1, 2, 8));
    cpu.load_instruction(20, BLT(1, 2, 12));
    cpu.load_instruction(32, ADDI(5, 0, 42));

    h.run(4);
    ASSERT_EQ(cpu.reg(5), 42u);
    return true;
}

TEST(exec_blt_taken_cpu)  { return run_blt_taken<CPUH>(); }
TEST(exec_blt_taken_pipe) { return run_blt_taken<PipeH>(); }

template <typename Harness>
bool run_blt_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 3);
    cpu.set_reg(3, static_cast<u32>(-5));
    cpu.set_reg(4, static_cast<u32>(-3));

    cpu.load_instruction(0,  BLT(1, 2, 12));
    cpu.load_instruction(4,  BLT(1, 4, 12));
    cpu.load_instruction(8,  BLT(4, 3, 12));
    cpu.load_instruction(12, ADDI(5, 0, 10));

    h.run(4);
    ASSERT_EQ(cpu.reg(5), 10u);
    return true;
}

TEST(exec_blt_not_taken_cpu)  { return run_blt_not_taken<CPUH>(); }
TEST(exec_blt_not_taken_pipe) { return run_blt_not_taken<PipeH>(); }

template <typename Harness>
bool run_blt_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-5));

    cpu.load_instruction(0,  BLT(0, 0, 8));
    cpu.load_instruction(4,  BLT(0, 1, 12));
    cpu.load_instruction(16, BLT(0, 2, 8));
    cpu.load_instruction(20, ADDI(3, 0, 5));

    h.run(4);
    ASSERT_EQ(cpu.reg(3), 5u);
    return true;
}

TEST(exec_blt_x0_cpu)  { return run_blt_x0<CPUH>(); }
TEST(exec_blt_x0_pipe) { return run_blt_x0<PipeH>(); }

// ── BGE ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_bge_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 10);
    cpu.set_reg(2, 10);
    cpu.set_reg(3, 5);

    cpu.load_instruction(0,  BGE(1, 2, 20));
    cpu.load_instruction(20, BGE(1, 3, 8));
    cpu.load_instruction(28, ADDI(4, 0, 4));

    h.run(3);
    ASSERT_EQ(cpu.reg(4), 4u);
    return true;
}

TEST(exec_bge_taken_cpu)  { return run_bge_taken<CPUH>(); }
TEST(exec_bge_taken_pipe) { return run_bge_taken<PipeH>(); }

template <typename Harness>
bool run_bge_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-10));
    cpu.set_reg(2, static_cast<u32>(-5));
    cpu.set_reg(3, 5);

    cpu.load_instruction(0,  BGE(1, 2, 20));
    cpu.load_instruction(4,  BGE(1, 3, 20));
    cpu.load_instruction(8,  ADDI(4, 0, 4));
    cpu.load_instruction(20, ADDI(5, 0, 5));
    cpu.load_instruction(24, ADDI(5, 5, 5));

    h.run(3);
    ASSERT_EQ(cpu.reg(4), 4u);
    ASSERT_EQ(cpu.reg(5), 0u);
    return true;
}

TEST(exec_bge_not_taken_cpu)  { return run_bge_not_taken<CPUH>(); }
TEST(exec_bge_not_taken_pipe) { return run_bge_not_taken<PipeH>(); }

template <typename Harness>
bool run_bge_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-5));
    cpu.set_reg(2, 5);

    cpu.load_instruction(0,  BGE(0, 0, 8));
    cpu.load_instruction(8,  BGE(0, 1, 8));
    cpu.load_instruction(16, BGE(0, 2, 8));
    cpu.load_instruction(20, ADDI(6, 0, 6));

    h.run(4);
    ASSERT_EQ(cpu.reg(6), 6u);
    return true;
}

TEST(exec_bge_x0_cpu)  { return run_bge_x0<CPUH>(); }
TEST(exec_bge_x0_pipe) { return run_bge_x0<PipeH>(); }

// ── BLTU ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_bltu_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.set_reg(3, 7);

    cpu.load_instruction(0,  BLTU(1, 2, 8));
    cpu.load_instruction(8,  BLTU(1, 2, 12));
    cpu.load_instruction(20, ADDI(7, 0, 7));

    h.run(3);
    ASSERT_EQ(cpu.reg(7), 7u);
    return true;
}

TEST(exec_bltu_taken_cpu)  { return run_bltu_taken<CPUH>(); }
TEST(exec_bltu_taken_pipe) { return run_bltu_taken<PipeH>(); }

template <typename Harness>
bool run_bltu_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1));
    cpu.set_reg(2, 5);
    cpu.set_reg(3, static_cast<u32>(-3));

    cpu.load_instruction(0,  BLTU(1, 2, 16));
    cpu.load_instruction(4,  BLTU(1, 3, 16));
    cpu.load_instruction(8,  ADDI(8, 0, 8));
    cpu.load_instruction(16, ADDI(9, 0, 9));
    cpu.load_instruction(20, ADDI(9, 9, 9));

    h.run(3);
    ASSERT_EQ(cpu.reg(8), 8u);
    ASSERT_EQ(cpu.reg(9), 0u);
    return true;
}

TEST(exec_bltu_not_taken_cpu)  { return run_bltu_not_taken<CPUH>(); }
TEST(exec_bltu_not_taken_pipe) { return run_bltu_not_taken<PipeH>(); }

template <typename Harness>
bool run_bltu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.set_reg(2, static_cast<u32>(-1));

    cpu.load_instruction(0,  BLTU(0, 0, 8));
    cpu.load_instruction(4,  BLTU(0, 1, 8));
    cpu.load_instruction(12, BLTU(0, 2, 8));
    cpu.load_instruction(20, ADDI(10, 0, 10));

    h.run(4);
    ASSERT_EQ(cpu.reg(10), 10u);
    return true;
}

TEST(exec_bltu_x0_cpu)  { return run_bltu_x0<CPUH>(); }
TEST(exec_bltu_x0_pipe) { return run_bltu_x0<PipeH>(); }

// ── BGEU ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_bgeu_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);
    cpu.set_reg(3, static_cast<u32>(-1));
    cpu.set_reg(4, static_cast<u32>(-1));

    cpu.load_instruction(0,  BGEU(1, 2, 8));
    cpu.load_instruction(8,  BGEU(3, 2, 8));
    cpu.load_instruction(16, BGEU(4, 2, 8));
    cpu.load_instruction(24, ADDI(11, 0, 11));

    h.run(4);
    ASSERT_EQ(cpu.reg(11), 11u);
    return true;
}

TEST(exec_bgeu_taken_cpu)  { return run_bgeu_taken<CPUH>(); }
TEST(exec_bgeu_taken_pipe) { return run_bgeu_taken<PipeH>(); }

template <typename Harness>
bool run_bgeu_not_taken()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 5);
    cpu.set_reg(2, static_cast<u32>(-1));
    cpu.set_reg(3, 7);

    cpu.load_instruction(0,  BGEU(1, 2, 16));
    cpu.load_instruction(4,  BGEU(1, 3, 16));
    cpu.load_instruction(8,  ADDI(12, 0, 12));
    cpu.load_instruction(16, ADDI(4, 0, 4));
    cpu.load_instruction(20, ADDI(4, 4, 4));

    h.run(3);
    ASSERT_EQ(cpu.reg(12), 12u);
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_bgeu_not_taken_cpu)  { return run_bgeu_not_taken<CPUH>(); }
TEST(exec_bgeu_not_taken_pipe) { return run_bgeu_not_taken<PipeH>(); }

template <typename Harness>
bool run_bgeu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 1);
    cpu.set_reg(2, static_cast<u32>(-1));

    cpu.load_instruction(0,  BGEU(0, 0, 16));
    cpu.load_instruction(16, BGEU(0, 1, 16));
    cpu.load_instruction(20, BGEU(0, 2, 16));
    cpu.load_instruction(24, ADDI(13, 0, 13));
    
    h.run(4);
    ASSERT_EQ(cpu.reg(13), 13u);
    return true;
}

TEST(exec_bgeu_x0_cpu)  { return run_bgeu_x0<CPUH>(); }
TEST(exec_bgeu_x0_pipe) { return run_bgeu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  5. Jumps
// ═══════════════════════════════════════════════════════════════════════

// ── JAL ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_jal()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0,   JAL(1, 100));
    cpu.load_instruction(100, ADDI(14, 0, 14));

    h.run(2);
    // Return address = PC + 4
    ASSERT_EQ(cpu.reg(1), 4u);
    ASSERT_EQ(cpu.reg(14), 14u);
    return true;
}

TEST(exec_jal_cpu)  { return run_jal<CPUH>(); }
TEST(exec_jal_pipe) { return run_jal<PipeH>(); }

template <typename Harness>
bool run_jal_backward()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_pc(0x100);
    cpu.load_instruction(0x100, JAL(1, -20));
    cpu.load_instruction(0xEC,  ADDI(15, 0, 15));

    h.run(2);
    ASSERT_HEX_EQ(cpu.reg(1), 0x104u);
    ASSERT_EQ(cpu.reg(15), 15u);
    return true;
}

TEST(exec_jal_backward_cpu)  { return run_jal_backward<CPUH>(); }
TEST(exec_jal_backward_pipe) { return run_jal_backward<PipeH>(); }

template <typename Harness>
bool run_jal_zero_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, JAL(1, 0));
    cpu.load_instruction(4, ADDI(4, 0, 4)); // never reached

    h.run(2);
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(exec_jal_zero_offset_cpu)  { return run_jal_zero_offset<CPUH>(); }
TEST(exec_jal_zero_offset_pipe) { return run_jal_zero_offset<PipeH>(); }

template <typename Harness>
bool run_jal_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, JAL(0, 8));
    cpu.load_instruction(8, ADDI(16, 0, 16));

    h.run(2);
    ASSERT_EQ(cpu.reg(0), 0u);
    ASSERT_EQ(cpu.reg(16), 16u);
    return true;
}

TEST(exec_jal_x0_cpu)  { return run_jal_x0<CPUH>(); }
TEST(exec_jal_x0_pipe) { return run_jal_x0<PipeH>(); }

template <typename Harness>
bool run_jal_misaligned()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0,  JAL(1, 66));
    cpu.load_instruction(66, ADDI(4, 0, 4));
    cpu.load_instruction(4,  ADDI(5, 0, 5));
    cpu.load_instruction(8,  ADDI(5, 5, 5));
    cpu.load_instruction(12, ADDI(5, 5, 5));
    cpu.load_instruction(20, ADDI(5, 5, 5));

    cpu.run_until([&]() { return cpu.halted(); }, 20);
    ASSERT(cpu.halted());
    ASSERT_EQ(cpu.reg(1), 0u);
    ASSERT_EQ(cpu.reg(4), 0u);
    ASSERT_EQ(cpu.reg(5), 0u);
    return true;
}

TEST(exec_jal_misaligned_cpu)  { return run_jal_misaligned<CPUH>(); }
TEST(exec_jal_misaligned_pipe) { return run_jal_misaligned<PipeH>(); }

// ── JALR ───────────────────────────────────────────────────────────────

template <typename Harness>
bool run_jalr()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x200);
    cpu.load_instruction(0,     JALR(2, 8, 1));
    cpu.load_instruction(0x208, ADDI(17, 0, 17));

    h.run(2);
    ASSERT_EQ(cpu.reg(2), 4u);
    ASSERT_EQ(cpu.reg(17), 17u);
    return true;
}

TEST(exec_jalr_cpu)  { return run_jalr<CPUH>(); }
TEST(exec_jalr_pipe) { return run_jalr<PipeH>(); }

template <typename Harness>
bool run_jalr_clears_lsb()
{
    Harness h;
    auto& cpu = h.get();

    // Target bit 0 cleared -> effectively 0x200
    cpu.set_reg(1, 0x201);
    cpu.load_instruction(0,     JALR(2, 0, 1));
    cpu.load_instruction(0x200, ADDI(18, 0, 18));

    h.run(2);
    ASSERT_EQ(cpu.reg(2), 4u);
    ASSERT_EQ(cpu.reg(18), 18u);
    return true;
}

TEST(exec_jalr_clears_lsb_cpu)  { return run_jalr_clears_lsb<CPUH>(); }
TEST(exec_jalr_clears_lsb_pipe) { return run_jalr_clears_lsb<PipeH>(); }

template <typename Harness>
bool run_jalr_neg_offset()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 100);
    cpu.load_instruction(0,  JALR(2, -4, 1));
    cpu.load_instruction(96, ADDI(19, 0, 19));

    h.step();
    h.step();
    ASSERT_EQ(cpu.reg(2), 4u);
    ASSERT_EQ(cpu.reg(19), 19u);
    return true;
}

TEST(exec_jalr_neg_offset_cpu)  { return run_jalr_neg_offset<CPUH>(); }
TEST(exec_jalr_neg_offset_pipe) { return run_jalr_neg_offset<PipeH>(); }

template <typename Harness>
bool run_jalr_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 100);
    cpu.load_instruction(0,   JALR(0, 0, 1));
    cpu.load_instruction(100, ADDI(20, 0, 20));

    h.run(2);
    ASSERT_EQ(cpu.reg(0), 0u);
    ASSERT_EQ(cpu.reg(20), 20u);
    return true;
}

TEST(exec_jalr_x0_cpu)  { return run_jalr_x0<CPUH>(); }
TEST(exec_jalr_x0_pipe) { return run_jalr_x0<PipeH>(); }

template <typename Harness>
bool run_jalr_misaligned()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 2);
    cpu.load_instruction(0,  JALR(2, 40, 1));
    cpu.load_instruction(42, ADDI(4, 0, 4));
    cpu.load_instruction(4,  ADDI(5, 0, 5));
    cpu.load_instruction(8,  ADDI(5, 5, 5));
    cpu.load_instruction(12, ADDI(5, 5, 5));
    cpu.load_instruction(20, ADDI(5, 5, 5));

    cpu.run_until([&]() { return cpu.halted(); }, 20);
    ASSERT(cpu.halted());
    ASSERT_EQ(cpu.reg(2), 0u);
    ASSERT_EQ(cpu.reg(4), 0u);
    ASSERT_EQ(cpu.reg(5), 0u);
    return true;
}

TEST(exec_jalr_misaligned_cpu)  { return run_jalr_misaligned<CPUH>(); }
TEST(exec_jalr_misaligned_pipe) { return run_jalr_misaligned<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  6. Upper Immediate
// ═══════════════════════════════════════════════════════════════════════

// ── LUI ────────────────────────────────────────────────────────────────

template <typename Harness>
bool run_lui()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, LUI(1, 0x1234'5000)); // 0x12345

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x1234'5000u);
    return true;
}

TEST(exec_lui_cpu)  { return run_lui<CPUH>(); }
TEST(exec_lui_pipe) { return run_lui<PipeH>(); }

template <typename Harness>
bool run_lui_zero()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, LUI(1, 0));

    h.step();
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_lui_zero_cpu)  { return run_lui_zero<CPUH>(); }
TEST(exec_lui_zero_pipe) { return run_lui_zero<PipeH>(); }

template <typename Harness>
bool run_lui_high_bit()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, LUI(1, 0x8000'0000)); // 0x80000

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x8000'0000u);
    return true;
}

TEST(exec_lui_high_bit_cpu)  { return run_lui_high_bit<CPUH>(); }
TEST(exec_lui_high_bit_pipe) { return run_lui_high_bit<PipeH>(); }

template <typename Harness>
bool run_lui_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, LUI(0, 0xABCD'E000)); // 0xABCDE

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_lui_x0_cpu)  { return run_lui_x0<CPUH>(); }
TEST(exec_lui_x0_pipe) { return run_lui_x0<PipeH>(); }

// ── AUIPC ──────────────────────────────────────────────────────────────

template <typename Harness>
bool run_auipc()
{
    Harness h;
    auto& cpu = h.get();

    cpu.load_instruction(0, AUIPC(1, 0x1000));

    h.step();
    ASSERT_EQ(cpu.reg(1), 0x1000u);
    return true;
}

TEST(exec_auipc_cpu)  { return run_auipc<CPUH>(); }
TEST(exec_auipc_pipe) { return run_auipc<PipeH>(); }

template <typename Harness>
bool run_auipc_nonzero_pc()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_pc(0x100);
    cpu.load_instruction(0x100, AUIPC(1, 0x0001'0000)); // 0x10

    h.step();
    ASSERT_HEX_EQ(cpu.reg(1), 0x100u + 0x10000u);
    return true;
}

TEST(exec_auipc_nonzero_pc_cpu)  { return run_auipc_nonzero_pc<CPUH>(); }
TEST(exec_auipc_nonzero_pc_pipe) { return run_auipc_nonzero_pc<PipeH>(); }

template <typename Harness>
bool run_auipc_negative()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_pc(0x1000);
    cpu.load_instruction(0x1000, AUIPC(1, 0xFFFF'F000)); // 0xFFFFF

    h.step();
    // 0x1000 + 0xFFFF'F000 = 0x0 (wraparound)
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_auipc_negative_cpu)  { return run_auipc_negative<CPUH>(); }
TEST(exec_auipc_negative_pipe) { return run_auipc_negative<PipeH>(); }

template <typename Harness>
bool run_auipc_x0()
{
    Harness h;
    auto& cpu = h.get();
    
    cpu.load_instruction(0, AUIPC(0, 0x1000));
    
    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);
    return true;
}

TEST(exec_auipc_x0_cpu)  { return run_auipc_x0<CPUH>(); }
TEST(exec_auipc_x0_pipe) { return run_auipc_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  8. System instructions
// ═══════════════════════════════════════════════════════════════════════

TEST(exec_fence_is_nop)
{
    auto cpu = make_cpu();
    cpu.set_reg(1, 42);
    cpu.load_instruction(0, FENCE); // fence iorw, iorw
    cpu.step();
    // FENCE should just advance PC; no register changes.
    ASSERT_EQ(cpu.pc(), 4u);
    ASSERT_EQ(cpu.reg(1), 42u);
    ASSERT(!cpu.halted());
    return true;
}

TEST(exec_ebreak_halts)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, EBREAK);
    bool ran = cpu.step();
    ASSERT(!ran);
    ASSERT(cpu.halted());
    return true;
}

TEST(exec_ecall_does_not_halt)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, ECALL);
    cpu.load_instruction(4, ADDI(1, 0, 42));
    bool ran = cpu.step();
    ASSERT(ran);
    ASSERT(!cpu.halted());
    // The ecall flag should be set on last_result
    ASSERT(cpu.last_result().ecall);
    // The next instruction should work
    cpu.step();
    ASSERT_EQ(cpu.reg(1), 42u);
    return true;
}

TEST(exec_run_until_ecall)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, ADDI(1, 0, 42));
    cpu.load_instruction(4, ADDI(1, 1, 1));
    cpu.load_instruction(8, ECALL);
    cpu.load_instruction(12, ADDI(1, 1, 1)); // should not run
 
    u64 count = cpu.run_until_ecall();
    ASSERT_EQ(count, 3u);       // 3 instructions executed (including ecall)
    ASSERT_EQ(cpu.reg(1), 43u); // 42 + 1
    ASSERT(!cpu.halted());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  9. CPU lifecycle
// ═══════════════════════════════════════════════════════════════════════

TEST(exec_invalid_instruction_halts)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, 0xFFFF'FFFF); // All-ones = invalid
    bool ran = cpu.step();
    ASSERT(!ran);
    ASSERT(cpu.halted());
    return true;
}

TEST(exec_fetch_fault_halts)
{
    // Memory is 0x0–0xFFFF - set PC past the end
    auto cpu = make_cpu();
    cpu.set_pc(0x20000);
    bool ran = cpu.step();
    ASSERT(!ran);
    ASSERT(cpu.halted());
    return true;
}

TEST(exec_reset_clears_halt)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0, EBREAK); // ebreak -> halt
    cpu.step();
    ASSERT(cpu.halted());

    cpu.reset();
    ASSERT(!cpu.halted());
    ASSERT_EQ(cpu.pc(), 0u);
    ASSERT_EQ(cpu.reg(1), 0u);
    return true;
}

TEST(exec_save_restore_state)
{
    auto cpu = make_cpu();
    cpu.set_reg(1, 100);
    cpu.set_reg(2, 200);
    cpu.set_pc(0x400);
 
    auto state = cpu.save_state();

    // Mutate CPU
    cpu.set_reg(1, 999);
    cpu.set_pc(0x800);

    cpu.restore_state(state);
    ASSERT_EQ(cpu.reg(1), 100u);
    ASSERT_EQ(cpu.reg(2), 200u);
    ASSERT_EQ(cpu.pc(), 0x400u);
    ASSERT_EQ(cpu.reg(0), 0u);  // x0 must still be 0
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  10. Statistics
// ═══════════════════════════════════════════════════════════════════════

TEST(exec_stats_basic)
{
    auto cpu = make_cpu();
    cpu.load_instruction(0,  ADDI(1, 0, 42));
    cpu.load_instruction(4,  ADDI(2, 1, 1));
    cpu.load_instruction(8,  ADDI(3, 2, 2));
    cpu.load_instruction(12, ADDI(4, 3, 3));
    cpu.load_instruction(16, ADDI(5, 4, 4));
 
    cpu.run(5);
    ASSERT_EQ(cpu.stats().instructions, 5u);
    return true;
}

TEST(exec_stats_branches)
{
    auto cpu = make_cpu();
    cpu.set_reg(1, 5);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, BEQ(1, 2, 8)); // taken
    cpu.step();
 
    ASSERT_EQ(cpu.stats().branches, 1u);
    ASSERT_EQ(cpu.stats().branches_taken, 1u);

    cpu.load_instruction(8, BNE(1, 2, 8)); // not taken
    cpu.step();
    ASSERT_EQ(cpu.stats().branches, 2u);
    ASSERT_EQ(cpu.stats().branches_taken, 1u);
    return true;
}

TEST(exec_stats_loads_stores_jumps)
{
    auto cpu = make_cpu();
    cpu.memory().write32(0x200, 0);
    cpu.set_reg(10, 0x200);
 
    cpu.load_instruction(0,  ADDI(1, 0, 42));
    cpu.load_instruction(4,  SW(1, 0, 10));
    cpu.load_instruction(8,  LW(2, 0, 10));
    cpu.load_instruction(12, JAL(1, 100));
 
    cpu.run(4);
    ASSERT_EQ(cpu.stats().loads, 1u);
    ASSERT_EQ(cpu.stats().stores, 1u);
    ASSERT_EQ(cpu.stats().jumps, 1u);
    ASSERT_EQ(cpu.stats().instructions, 4u);
    return true;
}
