/**
 * @file test_rv32m.cpp
 * @brief Tests for the RV32M extension.
 *
 * Sections:
 *    1 (line  24) : Decoder
 *    2 (line 116) : MUL execution
 *    3 (line 257) : MULH execution
 *    4 (line 364) : MULHU execution
 *    5 (line 433) : MULHSU execution
 *    6 (line 524) : DIV execution
 *    7 (line 648) : DIVU execution
 *    8 (line 734) : REM execution
 *    9 (line 858) : REMU execution
 *   10 (line 944) : Pipeline correctness
 */

#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. Decoder - all 8 M instructions decode correctly
// ═══════════════════════════════════════════════════════════════════════

TEST(m_decode_mul)
{
    auto inst = Decoder::decode(0x0220'81b3u); // mul x3, x1, x2
    ASSERT_EQ(inst.op, Op::MUL);
    ASSERT_EQ(inst.rd,  3);
    ASSERT_EQ(inst.rs1, 1);
    ASSERT_EQ(inst.rs2, 2);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_mulh)
{
    auto inst = Decoder::decode(0x0231'10b3u); // mulh x1, x2, x3
    ASSERT_EQ(inst.op, Op::MULH);
    ASSERT_EQ(inst.rd,  1);
    ASSERT_EQ(inst.rs1, 2);
    ASSERT_EQ(inst.rs2, 3);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_mulhsu)
{
    auto inst = Decoder::decode(0x0262'a233u); // mulhsu x4, x5, x6
    ASSERT_EQ(inst.op, Op::MULHSU);
    ASSERT_EQ(inst.rd,  4);
    ASSERT_EQ(inst.rs1, 5);
    ASSERT_EQ(inst.rs2, 6);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_mulhu)
{
    auto inst = Decoder::decode(0x0294'33b3u); // mulhu x7, x8, x9
    ASSERT_EQ(inst.op, Op::MULHU);
    ASSERT_EQ(inst.rd,  7);
    ASSERT_EQ(inst.rs1, 8);
    ASSERT_EQ(inst.rs2, 9);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_div)
{
    auto inst = Decoder::decode(0x02c5'c533u); // div x10, x11, x12
    ASSERT_EQ(inst.op, Op::DIV);
    ASSERT_EQ(inst.rd,  10);
    ASSERT_EQ(inst.rs1, 11);
    ASSERT_EQ(inst.rs2, 12);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_divu)
{
    auto inst = Decoder::decode(0x02f7'56b3u); // divu x13, x14, x15
    ASSERT_EQ(inst.op, Op::DIVU);
    ASSERT_EQ(inst.rd,  13);
    ASSERT_EQ(inst.rs1, 14);
    ASSERT_EQ(inst.rs2, 15);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_rem)
{
    auto inst = Decoder::decode(0x0328'e833u); // rem x16, x17, x18
    ASSERT_EQ(inst.op, Op::REM);
    ASSERT_EQ(inst.rd,  16);
    ASSERT_EQ(inst.rs1, 17);
    ASSERT_EQ(inst.rs2, 18);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

TEST(m_decode_remu)
{
    auto inst = Decoder::decode(0x035a'79b3u); // remu x19, x20, x21
    ASSERT_EQ(inst.op, Op::REMU);
    ASSERT_EQ(inst.rd,  19);
    ASSERT_EQ(inst.rs1, 20);
    ASSERT_EQ(inst.rs2, 21);
    ASSERT_EQ(inst.format, Format::R);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. MUL
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_mul()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 7);
    cpu.set_reg(2, 6);
    cpu.load_instruction(0, MUL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_exec_mul_cpu)  { return run_mul<CPUH>(); }
TEST(m_exec_mul_pipe) { return run_mul<PipeH>(); }

template <typename Harness>
bool run_mul_zero()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 12345);
    cpu.set_reg(2, 0);

    cpu.load_instruction(0, MUL(3, 1, 2));
    cpu.load_instruction(4, MUL(4, 2, 1));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(m_exec_mul_zero_cpu)  { return run_mul_zero<CPUH>(); }
TEST(m_exec_mul_zero_pipe) { return run_mul_zero<PipeH>(); }

template <typename Harness>
bool run_mul_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-3));
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, MUL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-21));
    return true;
}

TEST(m_exec_mul_neg_cpu)  { return run_mul_neg<CPUH>(); }
TEST(m_exec_mul_neg_pipe) { return run_mul_neg<PipeH>(); }

template <typename Harness>
bool run_mul_both_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-3));
    cpu.set_reg(2, static_cast<u32>(-7));
    cpu.load_instruction(0, MUL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 21u);
    return true;
}

TEST(m_exec_mul_both_neg_cpu)  { return run_mul_both_neg<CPUH>(); }
TEST(m_exec_mul_both_neg_pipe) { return run_mul_both_neg<PipeH>(); }

template <typename Harness>
bool run_mul_max()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFF); // max unsigned; -1 signed
    cpu.set_reg(2, 0xFFFF'FFFF);
    cpu.load_instruction(0, MUL(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(m_exec_mul_max_cpu)  { return run_mul_max<CPUH>(); }
TEST(m_exec_mul_max_pipe) { return run_mul_max<PipeH>(); }

template <typename Harness>
bool run_mul_overflow()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x10000);
    cpu.set_reg(2, 0x10000);
    cpu.load_instruction(0, MUL(3, 1, 2));

    h.step();
    // Lower 32 bits of 0x1'0000'0000 = 0
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_exec_mul_overflow_cpu)  { return run_mul_overflow<CPUH>(); }
TEST(m_exec_mul_overflow_pipe) { return run_mul_overflow<PipeH>(); }

template <typename Harness>
bool run_mul_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 4);
    cpu.set_reg(2, 10);

    cpu.load_instruction(0, MUL(0, 1, 2));
    cpu.load_instruction(4, MUL(3, 0, 1));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_exec_mul_x0_cpu)  { return run_mul_x0<CPUH>(); }
TEST(m_exec_mul_x0_pipe) { return run_mul_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  3. MULH
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_mulh()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x10000);
    cpu.set_reg(2, 0x10000);
    cpu.load_instruction(0, MULH(3, 1, 2));

    h.step();
    // Upper 32 bits of 0x1'0000'0000 -> 1
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(m_exec_mulh_cpu)  { return run_mulh<CPUH>(); }
TEST(m_exec_mulh_pipe) { return run_mulh<PipeH>(); }

template <typename Harness>
bool run_mulh_mixed()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1)); // 0xFFFF'FFFF
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULH(3, 1, 2));

    h.step();
    // 0xFFFF'FFFF'FFFF'FFFE -> 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_mulh_mixed_cpu)  { return run_mulh_mixed<CPUH>(); }
TEST(m_exec_mulh_mixed_pipe) { return run_mulh_mixed<PipeH>(); }

template <typename Harness>
bool run_mulh_both_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-2));
    cpu.set_reg(2, static_cast<u32>(-3));
    cpu.load_instruction(0, MULH(3, 1, 2));

    h.step();
    // 0x0000'0000'0000'0006 -> 0
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_exec_mulh_both_neg_cpu)  { return run_mulh_both_neg<CPUH>(); }
TEST(m_exec_mulh_both_neg_pipe) { return run_mulh_both_neg<PipeH>(); }

template <typename Harness>
bool run_mulh_int_min()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000u);
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULH(3, 1, 2));

    h.step();
    // 0xFFFF'FFFF'0000'0000 -> 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_mulh_int_min_cpu)  { return run_mulh_int_min<CPUH>(); }
TEST(m_exec_mulh_int_min_pipe) { return run_mulh_int_min<PipeH>(); }

template <typename Harness>
bool run_mulh_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-1)); // 0xFFFF'FFFF
    cpu.set_reg(2, 2);

    cpu.load_instruction(0, MULH(0, 1, 2));
    cpu.load_instruction(4, MULH(3, 0, 1));
    cpu.load_instruction(8, MULH(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(m_exec_mulh_x0_cpu)  { return run_mulh_x0<CPUH>(); }
TEST(m_exec_mulh_x0_pipe) { return run_mulh_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  4. MULHU
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_mulhu()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFFu);
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULHU(3, 1, 2));

    h.step();
    // 0x1'FFFF'FFFE -> 1
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(m_exec_mulhu_cpu)  { return run_mulhu<CPUH>(); }
TEST(m_exec_mulhu_pipe) { return run_mulhu<PipeH>(); }

template <typename Harness>
bool run_mulhu_max()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFFu);
    cpu.set_reg(2, 0xFFFF'FFFFu);
    cpu.load_instruction(0, MULHU(3, 1, 2));

    h.step();
    // 0xFFFF'FFFE'0000'0001 -> 0xFFFF'FFFE
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFEu);
    return true;
}

TEST(m_exec_mulhu_max_cpu)  { return run_mulhu_max<CPUH>(); }
TEST(m_exec_mulhu_max_pipe) { return run_mulhu_max<PipeH>(); }

template <typename Harness>
bool run_mulhu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFFu);
    cpu.set_reg(2, 2);

    cpu.load_instruction(0, MULHU(0, 1, 2));
    cpu.load_instruction(4, MULHU(3, 0, 1));
    cpu.load_instruction(8, MULHU(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(m_exec_mulhu_x0_cpu)  { return run_mulhu_x0<CPUH>(); }
TEST(m_exec_mulhu_x0_pipe) { return run_mulhu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  5. MULHSU
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_mulhsu()
{
    // Negative signed * unsigned
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFFu); // -1
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULHSU(3, 1, 2));

    h.step();
    // 0xFFFF'FFFF'FFFF'FFFE -> 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_mulhsu_cpu)  { return run_mulhsu<CPUH>(); }
TEST(m_exec_mulhsu_pipe) { return run_mulhsu<PipeH>(); }

template <typename Harness>
bool run_mulhsu_pos()
{
    // Positive signed * unsigned
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x10000);
    cpu.set_reg(2, 0x10000);
    cpu.load_instruction(0, MULHSU(3, 1, 2));

    h.step();
    // 0x1'0000'0000 -> 1
    ASSERT_EQ(cpu.reg(3), 1u);
    return true;
}

TEST(m_exec_mulhsu_pos_cpu)  { return run_mulhsu_pos<CPUH>(); }
TEST(m_exec_mulhsu_pos_pipe) { return run_mulhsu_pos<PipeH>(); }

template <typename Harness>
bool run_mulhsu_int_min()
{
    // INT32_MIN * unsigned
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000u);
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, MULHSU(3, 1, 2));

    h.step();
    // 0xFFFF'FFFF'0000'0000 -> 0xFFFF'FFFF
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_mulhsu_int_min_cpu)  { return run_mulhsu_int_min<CPUH>(); }
TEST(m_exec_mulhsu_int_min_pipe) { return run_mulhsu_int_min<PipeH>(); }

template <typename Harness>
bool run_mulhsu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000u);
    cpu.set_reg(2, 2);

    cpu.load_instruction(0, MULHSU(0, 1, 2));
    cpu.load_instruction(4, MULHSU(3, 0, 1));
    cpu.load_instruction(8, MULHSU(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 0u);
    return true;
}

TEST(m_exec_mulhsu_x0_cpu)  { return run_mulhsu_x0<CPUH>(); }
TEST(m_exec_mulhsu_x0_pipe) { return run_mulhsu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  6. DIV
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_div()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, DIV(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 6u);
    return true;
}

TEST(m_exec_div_cpu)  { return run_div<CPUH>(); }
TEST(m_exec_div_pipe) { return run_div<PipeH>(); }

template <typename Harness>
bool run_div_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-20));
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, DIV(3, 1, 2));

    h.step();
    // -20 / 3 = -6 (truncated)
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-6));
    return true;
}

TEST(m_exec_div_neg_cpu)  { return run_div_neg<CPUH>(); }
TEST(m_exec_div_neg_pipe) { return run_div_neg<PipeH>(); }

template <typename Harness>
bool run_div_neg_divisor()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 20);
    cpu.set_reg(2, static_cast<u32>(-3));
    cpu.load_instruction(0, DIV(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-6));
    return true;
}

TEST(m_exec_div_neg_divisor_cpu)  { return run_div_neg_divisor<CPUH>(); }
TEST(m_exec_div_neg_divisor_pipe) { return run_div_neg_divisor<PipeH>(); }

template <typename Harness>
bool run_div_by_zero()
{
    // Spec: DIV by zero -> -1
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, DIV(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_div_by_zero_cpu)  { return run_div_by_zero<CPUH>(); }
TEST(m_exec_div_by_zero_pipe) { return run_div_by_zero<PipeH>(); }

template <typename Harness>
bool run_div_overflow()
{
    // Spec: INT_MIN / -1 -> INT_MIN
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(2, 0xFFFF'FFFF);
    cpu.load_instruction(0, DIV(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x8000'0000u);
    return true;
}

TEST(m_exec_div_overflow_cpu)  { return run_div_overflow<CPUH>(); }
TEST(m_exec_div_overflow_pipe) { return run_div_overflow<PipeH>(); }

template <typename Harness>
bool run_div_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 7);

    cpu.load_instruction(0, DIV(0, 1, 2));
    cpu.load_instruction(4, DIV(3, 0, 1));
    cpu.load_instruction(8, DIV(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(4), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_div_x0_cpu)  { return run_div_x0<CPUH>(); }
TEST(m_exec_div_x0_pipe) { return run_div_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  7. DIVU
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_divu()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 100);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, DIVU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 14u);
    return true;
}

TEST(m_exec_divu_cpu)  { return run_divu<CPUH>(); }
TEST(m_exec_divu_pipe) { return run_divu<PipeH>(); }

template <typename Harness>
bool run_divu_large()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFF);
    cpu.set_reg(2, 2);
    cpu.load_instruction(0, DIVU(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0x7FFF'FFFFu);
    return true;
}

TEST(m_exec_divu_large_cpu)  { return run_divu_large<CPUH>(); }
TEST(m_exec_divu_large_pipe) { return run_divu_large<PipeH>(); }

template <typename Harness>
bool run_divu_by_zero()
{
    // Spec: DIVU by zero -> 0xFFFF'FFFF
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, DIVU(3, 1, 2));

    h.step();
    ASSERT_HEX_EQ(cpu.reg(3), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_divu_by_zero_cpu)  { return run_divu_by_zero<CPUH>(); }
TEST(m_exec_divu_by_zero_pipe) { return run_divu_by_zero<PipeH>(); }

template <typename Harness>
bool run_divu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 6);

    cpu.load_instruction(0, DIVU(0, 1, 2));
    cpu.load_instruction(4, DIVU(3, 0, 1));
    cpu.load_instruction(8, DIVU(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_HEX_EQ(cpu.reg(4), 0xFFFF'FFFFu);
    return true;
}

TEST(m_exec_divu_x0_cpu)  { return run_divu_x0<CPUH>(); }
TEST(m_exec_divu_x0_pipe) { return run_divu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  8. REM
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_rem()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 20);
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, REM(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 2u);
    return true;
}

TEST(m_exec_rem_cpu)  { return run_rem<CPUH>(); }
TEST(m_exec_rem_pipe) { return run_rem<PipeH>(); }

template <typename Harness>
bool run_rem_neg()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, static_cast<u32>(-20));
    cpu.set_reg(2, 3);
    cpu.load_instruction(0, REM(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), static_cast<u32>(-2));
    return true;
}

TEST(m_exec_rem_neg_cpu)  { return run_rem_neg<CPUH>(); }
TEST(m_exec_rem_neg_pipe) { return run_rem_neg<PipeH>(); }

template <typename Harness>
bool run_rem_neg_divisor()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 20);
    cpu.set_reg(2, static_cast<u32>(-3));
    cpu.load_instruction(0, REM(3, 1, 2));

    h.step();
    // Remainder sign follows the dividend, not the divisor
    ASSERT_EQ(cpu.reg(3), 2u);
    return true;
}

TEST(m_exec_rem_neg_divisor_cpu)  { return run_rem_neg_divisor<CPUH>(); }
TEST(m_exec_rem_neg_divisor_pipe) { return run_rem_neg_divisor<PipeH>(); }

template <typename Harness>
bool run_rem_by_zero()
{
    // Spec: REM by zero -> dividend.
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, REM(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_exec_rem_by_zero_cpu)  { return run_rem_by_zero<CPUH>(); }
TEST(m_exec_rem_by_zero_pipe) { return run_rem_by_zero<PipeH>(); }

template <typename Harness>
bool run_rem_overflow()
{
    // Spec: INT_MIN % -1 -> 0.
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0x8000'0000);
    cpu.set_reg(2, 0xFFFF'FFFF);
    cpu.load_instruction(0, REM(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);
    return true;
}

TEST(m_exec_rem_overflow_cpu)  { return run_rem_overflow<CPUH>(); }
TEST(m_exec_rem_overflow_pipe) { return run_rem_overflow<PipeH>(); }

template <typename Harness>
bool run_rem_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 4);

    cpu.load_instruction(0, REM(0, 1, 2));
    cpu.load_instruction(4, REM(3, 0, 1));
    cpu.load_instruction(8, REM(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 4u);
    return true;
}

TEST(m_exec_rem_x0_cpu)  { return run_rem_x0<CPUH>(); }
TEST(m_exec_rem_x0_pipe) { return run_rem_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  9. REMU
// ═══════════════════════════════════════════════════════════════════════

template <typename Harness>
bool run_remu()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 100);
    cpu.set_reg(2, 7);
    cpu.load_instruction(0, REMU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 2u);
    return true;
}

TEST(m_exec_remu_cpu)  { return run_remu<CPUH>(); }
TEST(m_exec_remu_pipe) { return run_remu<PipeH>(); }

template <typename Harness>
bool run_remu_large()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 0xFFFF'FFFF);
    cpu.set_reg(2, 16);
    cpu.load_instruction(0, REMU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 15u);
    return true;
}

TEST(m_exec_remu_large_cpu)  { return run_remu_large<CPUH>(); }
TEST(m_exec_remu_large_pipe) { return run_remu_large<PipeH>(); }

template <typename Harness>
bool run_remu_by_zero()
{
    // Spec: REMU by zero -> dividend
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 0);
    cpu.load_instruction(0, REMU(3, 1, 2));

    h.step();
    ASSERT_EQ(cpu.reg(3), 42u);
    return true;
}

TEST(m_exec_remu_by_zero_cpu)  { return run_remu_by_zero<CPUH>(); }
TEST(m_exec_remu_by_zero_pipe) { return run_remu_by_zero<PipeH>(); }

template <typename Harness>
bool run_remu_x0()
{
    Harness h;
    auto& cpu = h.get();

    cpu.set_reg(1, 42);
    cpu.set_reg(2, 8);

    cpu.load_instruction(0, REMU(0, 1, 2));
    cpu.load_instruction(4, REMU(3, 0, 1));
    cpu.load_instruction(8, REMU(4, 2, 0));

    h.step();
    ASSERT_EQ(cpu.reg(0), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(3), 0u);

    h.step();
    ASSERT_EQ(cpu.reg(4), 8u);
    return true;
}

TEST(m_exec_remu_x0_cpu)  { return run_remu_x0<CPUH>(); }
TEST(m_exec_remu_x0_pipe) { return run_remu_x0<PipeH>(); }

// ═══════════════════════════════════════════════════════════════════════
//  10. Pipeline correctness
// ═══════════════════════════════════════════════════════════════════════

TEST(m_pipe_mul_chain)
{
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

TEST(m_pipe_divrem_program)
{
    // Compute quotient and remainder: 47 / 5 = 9 rem 2.
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    PipelinedCPU cpu(mem);
    cpu.set_reg(1, 47);
    cpu.set_reg(2, 5);
    cpu.load_instruction(0, DIV(3, 1, 2));  // x3 = 47 / 5 = 9
    cpu.load_instruction(4, REM(4, 1, 2));  // x4 = 47 % 5 = 2
    cpu.load_instruction(8, 0x00100073);
    cycle_t n = 0;
    while (n < 100) { if (!cpu.tick()) break; ++n; }
    ASSERT_EQ(cpu.reg(3), 9u);
    ASSERT_EQ(cpu.reg(4), 2u);
    return true;
}
