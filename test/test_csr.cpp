/**
 * @file test_csr.cpp
 * @brief Tests for CSR instructions.
 *
 * Sections:
 *   1 (line 62) : Decoder
 */

#include "core/cpu.hpp"
#include "core/pipeline.hpp"

#include <cstdint>
#include <format>
#include <iostream>

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
//  1. Decoder
// ═══════════════════════════════════════════════════════════════════════

TEST(csr_decode_csrrw) {
    auto inst = Decoder::decode(0x3001'10f3u); // csrrw x1, mstatus, x2
    ASSERT_EQ(inst.op, Op::CSRRW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    constexpr u32 mstatus = 0x300u;
    ASSERT_EQ(inst.imm, mstatus);
    return true;
}

TEST(csr_decode_csrrs) {
    auto inst = Decoder::decode(0x3043'22f3u); // csrrs x5, mie, x6
    ASSERT_EQ(inst.op, Op::CSRRS);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    constexpr u32 mie = 0x304u;
    ASSERT_EQ(inst.imm, mie);
    return true;
}

TEST(csr_decode_csrrc) {
    auto inst = Decoder::decode(0x3055'34f3u); // csrrc x9, mtvec, x10
    ASSERT_EQ(inst.op, Op::CSRRC);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs1, 10);
    constexpr u32 mtvec = 0x305u;
    ASSERT_EQ(inst.imm, mtvec);
    return true;
}

TEST(csr_decode_csrrwi) {
    auto inst = Decoder::decode(0x3412'd6f3u); // csrrwi x13, mepc, 5
    ASSERT_EQ(inst.op, Op::CSRRWI);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs1, 5);
    constexpr u32 mepc = 0x341u;
    ASSERT_EQ(inst.imm, mepc);
    return true;
}

TEST(csr_decode_csrrsi) {
    auto inst = Decoder::decode(0x3421'e7f3u); // csrrsi x15, mcause, 3
    ASSERT_EQ(inst.op, Op::CSRRSI);
    ASSERT_EQ(inst.rd, 15);
    ASSERT_EQ(inst.rs1,3);
    constexpr u32 mcause = 0x342u;
    ASSERT_EQ(inst.imm, mcause);
    return true;
}

TEST(csr_decode_csrrci) {
    auto inst = Decoder::decode(0x3443'f8f3u); // csrrci x17, mip, 7
    ASSERT_EQ(inst.op, Op::CSRRCI);
    ASSERT_EQ(inst.rd, 17);
    ASSERT_EQ(inst.rs1, 7);
    constexpr u32 mip = 0x344u;
    ASSERT_EQ(inst.imm, mip);
    return true;
}
