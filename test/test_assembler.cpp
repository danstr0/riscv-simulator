/**
 * @file test_assembler.cpp
 * @brief Tests for the two-pass assembler.
 *
 * Sections:
 *   1 (line 112) : Parsing
 *   2 (line 147) : RV32I encoding
 *   3 (line 426) : RV32I Pseudo-instructions
 *   4 (line 458) : Labels
 *   5 (line 515) : M extension
 *   6 (line 554) : A extension
 *   7 (line 613) : V extension
 *   8 (line 779) : CSR and system instructions
 *   9 (line 826) : Integration
 *  10 (line 911) : Misc. edge cases
 */

#include "core/assembler.hpp"
#include "core/cpu.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ── Helpers ────────────────────────────────────────────────────────────

/// Assemble one line and return the first word.
static u32 asm1(const char* line)
{
    Assembler as;
    auto r = as.assemble(line);
    if (!r.ok || r.code.size() < 4) return 0xDEAD;
    
    return static_cast<u32>(r.code[0])
         | (static_cast<u32>(r.code[1]) << 8)
         | (static_cast<u32>(r.code[2]) << 16)
         | (static_cast<u32>(r.code[3]) << 24);
}

/// Run an assembled program and return a register value.
static u32 run_asm(const char* source, u32 result_reg = 1)
{
    Assembler as;
    auto r = as.assemble(source);
    if (!r.ok)
    {
        std::cerr << "    ASM ERROR: " << r.errors[0].message
                  << " (line " << r.errors[0].line << ")\n";
        return 0xDEADDEAD;
    }
    auto mem = std::make_shared<FlatMemory>(0, 0x10000);
    mem->load(0, r.code);
    CPU cpu(mem);
    cpu.run(10000);
    return cpu.reg(result_reg);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. Parsing
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_blank_line) {
    Assembler as;
    auto r = as.assemble("   \n\n");
    ASSERT(r.ok);
    ASSERT_EQ(r.code.size(), 0u);
    return true;
}

TEST(asm_comments_only) {
    Assembler as;
    auto r = as.assemble("# comment 1\n; comment 2\n// comment 3");
    ASSERT(r.ok);
    ASSERT_EQ(r.code.size(), 0u);
    return true;
}

TEST(asm_label_only) {
    Assembler as;
    auto r = as.assemble("start:");
    ASSERT(r.ok);
    ASSERT(r.labels.count("start"));
    ASSERT_EQ(r.labels["start"], 0u);
    return true;
}

TEST(asm_inline_comment) {
    auto w = asm1("addi x1, x0, 42  # set x1 to 42");
    ASSERT_HEX_EQ(w, 0x02a0'0093u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. RV32I encoding
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_add) {
    ASSERT_HEX_EQ(asm1("add x1, x2, x3"), 0x0031'00b3u);
    return true;
}

TEST(asm_sub) {
    ASSERT_HEX_EQ(asm1("sub x5, x6, x7"), 0x4073'02b3u);
    return true;
}

TEST(asm_and) {
    ASSERT_HEX_EQ(asm1("and x10, x11, x12"), 0x00c5'f533u);
    return true;
}

TEST(asm_or) {
    ASSERT_HEX_EQ(asm1("or x13, x14, x15"), 0x00f7'66b3u);
    return true;
}

TEST(asm_xor) {
    ASSERT_HEX_EQ(asm1("xor x16, x17, x18"), 0x0128'c833u);
    return true;
}

TEST(asm_sll) {
    ASSERT_HEX_EQ(asm1("sll x19, x20, x21"), 0x015a'19b3u);
    return true;
}

TEST(asm_srl) {
    ASSERT_HEX_EQ(asm1("srl x22, x23, x24"), 0x018b'db33u);
    return true;
}

TEST(asm_sra) {
    ASSERT_HEX_EQ(asm1("sra x25, x26, x27"), 0x41bd'5cb3u);
    return true;
}

TEST(asm_slt) {
    ASSERT_HEX_EQ(asm1("slt x28, x29, x30"), 0x01ee'ae33u);
    return true;
}

TEST(asm_sltu) {
    ASSERT_HEX_EQ(asm1("sltu x31, x1, x2"), 0x0020'bfb3u);
    return true;
}

// ── I-type ALU ─────────────────────────────────────────────────────────

TEST(asm_addi) {
    ASSERT_HEX_EQ(asm1("addi x1, x0, 42"), 0x02a0'0093u);
    return true;
}

TEST(asm_addi_negative) {
    ASSERT_HEX_EQ(asm1("addi x2, x3, -5"), 0xffb1'8113u);
    return true;
}

TEST(asm_addi_max_positive) {
    ASSERT_HEX_EQ(asm1("addi x1, x0, 2047"), 0x7ff0'0093u);
    return true;
}

TEST(asm_addi_min_negative) {
    ASSERT_HEX_EQ(asm1("addi x1, x0, -2048"), 0x8000'0093u);
    return true;
}

TEST(asm_slti) {
    ASSERT_HEX_EQ(asm1("slti x4, x5, 100"), 0x0642'a213u);
    return true;
}

TEST(asm_sltiu) {
    ASSERT_HEX_EQ(asm1("sltiu x6, x7, 200"), 0x0c83'b313u);
    return true;
}

TEST(asm_xori) {
    ASSERT_HEX_EQ(asm1("xori x8, x9, 0xFF"), 0x0ff4'c413u);
    return true;
}

TEST(asm_ori) {
    ASSERT_HEX_EQ(asm1("ori x10, x11, 0x123"), 0x1235'e513u);
    return true;
}

TEST(asm_andi) {
    ASSERT_HEX_EQ(asm1("andi x12, x13, 0x7FF"), 0x7ff6'f613u);
    return true;
}

TEST(asm_slli) {
    ASSERT_HEX_EQ(asm1("slli x14, x15, 5"), 0x0057'9713u);
    return true;
}

TEST(asm_slli_zero) {
    ASSERT_HEX_EQ(asm1("slli x1, x2, 0"), 0x0001'1093u);
    return true;
}

TEST(asm_slli_max) {
    ASSERT_HEX_EQ(asm1("slli x1, x2, 31"), 0x01f1'1093u);
    return true;
}

TEST(asm_srli) {
    ASSERT_HEX_EQ(asm1("srli x16, x17, 10"), 0x00a8'd813u);
    return true;
}

TEST(asm_srai) {
    ASSERT_HEX_EQ(asm1("srai x18, x19, 15"), 0x40f9'd913u);
    return true;
}

TEST(asm_srai_max) {
    ASSERT_HEX_EQ(asm1("srai x1, x2, 31"), 0x41f1'5093u);
    return true;
}

// ── I-type Loads and JALR ──────────────────────────────────────────────

TEST(asm_lw) {
    ASSERT_HEX_EQ(asm1("lw x1, 8(x2)"), 0x0081'2083u);
    return true;
}

TEST(asm_lw_zero) {
    ASSERT_HEX_EQ(asm1("lw x1, 0(x2)"), 0x0001'2083u);
    return true;
}

TEST(asm_lh) {
    ASSERT_HEX_EQ(asm1("lh x3, -4(x4)"), 0xffc2'1183u);
    return true;
}

TEST(asm_lb) {
    ASSERT_HEX_EQ(asm1("lb x5, 4(x6)"), 0x0043'0283u);
    return true;
}

TEST(asm_lhu) {
    ASSERT_HEX_EQ(asm1("lhu x7, 16(x8)"), 0x0104'5383u);
    return true;
}

TEST(asm_lbu) {
    ASSERT_HEX_EQ(asm1("lbu x9, 1(x10)"), 0x0015'4483u);
    return true;
}

TEST(asm_jalr) {
    ASSERT_HEX_EQ(asm1("jalr x1, 8(x2)"), 0x0081'00e7u);
    return true;
}

// ── S-type instructions (stores) ───────────────────────────────────────

TEST(asm_sw) {
    ASSERT_HEX_EQ(asm1("sw x1, 12(x2)"), 0x0011'2623u);
    return true;
}

TEST(asm_sw_zero) {
    ASSERT_HEX_EQ(asm1("sw x1, 0(x2)"), 0x0011'2023u);
    return true;
}

TEST(asm_sh) {
    ASSERT_HEX_EQ(asm1("sh x3, -8(x4)"), 0xfe32'1c23u);
    return true;
}

TEST(asm_sb) {
    ASSERT_HEX_EQ(asm1("sb x5, 7(x6)"), 0x0053'03a3u);
    return true;
}

// ── B-type instructions (branches) ─────────────────────────────────────

TEST(asm_beq) {
    ASSERT_HEX_EQ(asm1("beq x1, x2, 16"), 0x0020'8863u);
    return true;
}

TEST(asm_bne) {
    ASSERT_HEX_EQ(asm1("bne x3, x4, -8"), 0xfe41'9ce3u);
    return true;
}

TEST(asm_blt) {
    ASSERT_HEX_EQ(asm1("blt x5, x6, 32"), 0x0262'c063u);
    return true;
}

TEST(asm_bge) {
    ASSERT_HEX_EQ(asm1("bge x7, x8, 24"), 0x0083'dc63u);
    return true;
}

TEST(asm_bltu) {
    ASSERT_HEX_EQ(asm1("bltu x9, x10, 40"), 0x02a4'e463u);
    return true;
}

TEST(asm_bgeu) {
    ASSERT_HEX_EQ(asm1("bgeu x11, x12, 48"), 0x02c5'f863u);
    return true;
}

TEST(asm_branch_max_negative) {
    ASSERT_HEX_EQ(asm1("beq x1, x0, -4096"), 0x8000'8063u);
    return true;
}

TEST(asm_branch_max_positive) {
    ASSERT_HEX_EQ(asm1("beq x0, x0, 4094"), 0x7e00'0fe3u);
    return true;
}

// ── U-type instructions ────────────────────────────────────────────────

TEST(asm_lui) {
    ASSERT_HEX_EQ(asm1("lui x1, 0x12345"), 0x1234'50b7u);
    return true;
}

TEST(asm_auipc) {
    ASSERT_HEX_EQ(asm1("auipc x2, 0xABCDE"), 0xabcd'e117u);
    return true;
}

// ── JAL ────────────────────────────────────────────────────────────────

TEST(asm_jal) {
    ASSERT_HEX_EQ(asm1("jal x1, 100"), 0x0640'00efu);
    return true;
}

TEST(asm_jal_negative) {
    ASSERT_HEX_EQ(asm1("jal x0, -20"), 0xfedf'f06fu);
    return true;
}

TEST(asm_jal_max_positive) {
    ASSERT_HEX_EQ(asm1("jal x0, 1048574"), 0x7fff'f06fu);
    return true;
}

TEST(asm_jal_max_negative) {
    ASSERT_HEX_EQ(asm1("jal x0, -1048576"), 0x8000'006fu);
    return true;
}

// ── Misc. ──────────────────────────────────────────────────────────────

TEST(asm_abi_names) {
    ASSERT_HEX_EQ(asm1("addi a0, zero, 5"), asm1("addi x10, x0, 5"));
    ASSERT_HEX_EQ(asm1("add t0, s0, ra"), asm1("add x5, x8, x1"));
    return true;
}

TEST(asm_hex_immediate) {
    ASSERT_HEX_EQ(asm1("addi x1, x0, 0xFF"), asm1("addi x1, x0, 255"));
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. Pseudo-instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_nop) {
    ASSERT_HEX_EQ(asm1("nop"), 0x0000'0013u);
    return true;
}

TEST(asm_mv) {
    ASSERT_HEX_EQ(asm1("mv x1, x2"), asm1("addi x1, x2, 0"));
    return true;
}

TEST(asm_ret) {
    ASSERT_HEX_EQ(asm1("ret"), asm1("jalr x0, (x1)"));
    return true;
}

TEST(asm_li_small) {
    ASSERT_HEX_EQ(asm1("li x1, 42"), asm1("addi x1, x0, 42"));
    return true;
}

TEST(asm_li_large) {
    Assembler as;
    auto r = as.assemble("li x1, 0x12345");
    ASSERT(r.ok);
    ASSERT_EQ(r.code.size(), 8u);  // lui + addi
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Labels
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_label_address) {
    Assembler as;
    auto r = as.assemble(
        "start:\n"
        "    nop\n"
        "end:\n"
        "    nop\n"
    );
    ASSERT(r.ok);
    ASSERT_EQ(r.labels["start"], 0u);
    ASSERT_EQ(r.labels["end"], 4u);
    return true;
}

TEST(asm_branch_forward) {
    Assembler as;
    auto r = as.assemble(
        "    beq x1, x0, skip\n"
        "    addi x2, x0, 1\n"
        "skip:\n"
        "    addi x3, x0, 2\n"
    );
    ASSERT(r.ok);
    ASSERT_EQ(r.code.size(), 12u);
    return true;
}

TEST(asm_branch_backward) {
    Assembler as;
    auto r = as.assemble(
        "loop:\n"
        "    addi x1, x1, -1\n"
        "    bne x1, x0, loop\n"
        "    ebreak\n"
    );
    ASSERT(r.ok);
    ASSERT_EQ(r.code.size(), 12u);
    return true;
}

TEST(asm_j_label) {
    Assembler as;
    auto r = as.assemble(
        "    j skip\n"
        "    addi x1, x0, 99\n"
        "skip:\n"
        "    addi x1, x0, 42\n"
        "    ebreak\n"
    );
    ASSERT(r.ok);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. M extension
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_mulh) {
    ASSERT_HEX_EQ(asm1("mulh x1, x2, x3"), 0x0231'10b3u);
    return true;
}

TEST(asm_mulhsu) {
    ASSERT_HEX_EQ(asm1("mulhsu x4, x5, x6"), 0x0262'a233u);
    return true;
}

TEST(asm_mulhu) {
    ASSERT_HEX_EQ(asm1("mulhu x7, x8, x9"), 0x0294'33b3u);
    return true;
}

TEST(asm_div) {
    ASSERT_HEX_EQ(asm1("div x10, x11, x12"), 0x02c5'c533u);
    return true;
}

TEST(asm_divu) {
    ASSERT_HEX_EQ(asm1("divu x13, x14, x15"), 0x02f7'56b3u);
    return true;
}

TEST(asm_rem) {
    ASSERT_HEX_EQ(asm1("rem x16, x17, x18"), 0x0328'e833u);
    return true;
}

TEST(asm_remu) {
    ASSERT_HEX_EQ(asm1("remu x19, x20, x21"), 0x035a'79b3u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. A extension 
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_lr_w) {
    ASSERT_HEX_EQ(asm1("lr.w x3, (x1)"), 0x1000'a1afu);
    return true;
}

TEST(asm_sc_w) {
    ASSERT_HEX_EQ(asm1("sc.w x3, x1, (x2)"), 0x1811'21afu);
    return true;
}

TEST(asm_amoswap) {
    ASSERT_HEX_EQ(asm1("amoswap.w x1, x2, (x3)"), 0x0821'a0afu);
    return true;
}

TEST(asm_amoadd) {
    ASSERT_HEX_EQ(asm1("amoadd.w x4, x5, (x6)"), 0x0053'222fu);
    return true;
}

TEST(asm_amoxor) {
    ASSERT_HEX_EQ(asm1("amoxor.w x7, x8, (x9)"), 0x2084'a3afu);
    return true;
}

TEST(asm_amoand) {
    ASSERT_HEX_EQ(asm1("amoand.w x10, x11, (x12)"), 0x60b6'252fu);
    return true;
}

TEST(asm_amoor) {
    ASSERT_HEX_EQ(asm1("amoor.w x13, x14, (x15)"), 0x40e7'a6afu);
    return true;
}

TEST(asm_amomin) {
    ASSERT_HEX_EQ(asm1("amomin.w x16, x17, (x18)"), 0x8119'282fu);
    return true;
}

TEST(asm_amomax) {
    ASSERT_HEX_EQ(asm1("amomax.w x19, x20, (x21)"), 0xa14a'a9afu);
    return true;
}

TEST(asm_amominu) {
    ASSERT_HEX_EQ(asm1("amominu.w x22, x23, (x24)"), 0xc17c'2b2fu);
    return true;
}

TEST(asm_amomaxu) {
    ASSERT_HEX_EQ(asm1("amomaxu.w x25, x26, (x27)"), 0xe1ad'acafu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. RVV subset
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_vsetvli) {
    ASSERT_HEX_EQ(asm1("vsetvli x2, x1, 0x010"), 0x0100'f157u);
    return true;
}

TEST(asm_vle32_v) {
    ASSERT_HEX_EQ(asm1("vle32.v v4, (x3)"), 0x0201'e207u);
    return true;
}

TEST(asm_vse32_v) {
    ASSERT_HEX_EQ(asm1("vse32.v v6, (x5)"), 0x0202'e327u);
    return true;
}

TEST(asm_vadd_vv) {
    ASSERT_HEX_EQ(asm1("vadd.vv v1, v2, v3"), 0x0221'80d7u);
    return true;
}

TEST(asm_vadd_vx) {
    ASSERT_HEX_EQ(asm1("vadd.vx v4, v5, x6"), 0x0253'4257u);
    return true;
}

TEST(asm_vsub_vv) {
    ASSERT_HEX_EQ(asm1("vsub.vv v7, v8, v9"), 0x0a84'83d7u);
    return true;
}

TEST(asm_vsub_vx) {
    ASSERT_HEX_EQ(asm1("vsub.vx v10, v11, x12"), 0x0ab6'4557u);
    return true;
}

TEST(asm_vand_vv) {
    ASSERT_HEX_EQ(asm1("vand.vv v13, v14, v15"), 0x26e7'86d7u);
    return true;
}

TEST(asm_vand_vx) {
    ASSERT_HEX_EQ(asm1("vand.vx v16, v17, x18"), 0x2719'4857u);
    return true;
}

TEST(asm_vor_vv) {
    ASSERT_HEX_EQ(asm1("vor.vv v19, v20, v21"), 0x2b4a'89d7u);
    return true;
}

TEST(asm_vor_vx) {
    ASSERT_HEX_EQ(asm1("vor.vx v22, v23, x24"), 0x2b7c'4b57u);
    return true;
}

TEST(asm_vxor_vv) {
    ASSERT_HEX_EQ(asm1("vxor.vv v25, v26, v27"), 0x2fad'8cd7u);
    return true;
}

TEST(asm_vxor_vx) {
    ASSERT_HEX_EQ(asm1("vxor.vx v28, v29, x30"), 0x2fdf'4e57u);
    return true;
}

TEST(asm_vsll_vx) {
    ASSERT_HEX_EQ(asm1("vsll.vx v31, v0, x1"), 0x9600'cfd7u);
    return true;
}

TEST(asm_vsrl_vx) {
    ASSERT_HEX_EQ(asm1("vsrl.vx v2, v3, x4"), 0xa232'4157u);
    return true;
}

TEST(asm_vmseq_vv) {
    ASSERT_HEX_EQ(asm1("vmseq.vv v5, v6, v7"), 0x6263'82d7u);
    return true;
}

TEST(asm_vmseq_vx) {
    ASSERT_HEX_EQ(asm1("vmseq.vx v8, v9, x10"), 0x6295'4457u);
    return true;
}

TEST(asm_vmslt_vv) {
    ASSERT_HEX_EQ(asm1("vmslt.vv v11, v12, v13"), 0x6ec6'85d7u);
    return true;
}

TEST(asm_vmsltu_vv) {
    ASSERT_HEX_EQ(asm1("vmsltu.vv v14, v15, v16"), 0x6af8'0757u);
    return true;
}

// ── Mask instructions ──────────────────────────────────────────────────

TEST(asm_vmand_mm) {
    ASSERT_HEX_EQ(asm1("vmand.mm v17, v18, v19"), 0x6729'a8d7u);
    return true;
}

TEST(asm_vmnand_mm) {
    ASSERT_HEX_EQ(asm1("vmnand.mm v20, v21, v22"), 0x775b'2a57u);
    return true;
}

TEST(asm_vmandn_mm) {
    ASSERT_HEX_EQ(asm1("vmandn.mm v23, v24, v25"), 0x638c'abd7u);
    return true;
}

TEST(asm_vmxor_mm) {
    ASSERT_HEX_EQ(asm1("vmxor.mm v26, v27, v28"), 0x6fbe'2d57u);
    return true;
}

TEST(asm_vmor_mm) {
    ASSERT_HEX_EQ(asm1("vmor.mm v29, v30, v31"), 0x6bef'aed7u);
    return true;
}

TEST(asm_vmnor_mm) {
    ASSERT_HEX_EQ(asm1("vmnor.mm v0, v1, v2"), 0x7a11'2057u);
    return true;
}

TEST(asm_vmorn_mm) {
    ASSERT_HEX_EQ(asm1("vmorn.mm v3, v4, v5"), 0x7242'a1d7u);
    return true;
}

TEST(asm_vmxnor_mm) {
    ASSERT_HEX_EQ(asm1("vmxnor.mm v6, v7, v8"), 0x7e74'2357u);
    return true;
}

// ── Pseudo-instructions ────────────────────────────────────────────────

TEST(asm_vmmv_m) {
    ASSERT_HEX_EQ(asm1("vmmv.m v9, v10"), asm1("vmand.mm v9, v10, v10"));
    ASSERT_HEX_EQ(asm1("vmmv.m v9, v10"), 0x66a5'24d7u);
    return true;
}
TEST(asm_vmclr_m) {
    ASSERT_HEX_EQ(asm1("vmclr.m v11"), asm1("vmxor.mm v11, v11, v11"));
    ASSERT_HEX_EQ(asm1("vmclr.m v11"), 0x6eb5'a5d7u);
    return true;
}

TEST(asm_vmset_m) {
    ASSERT_HEX_EQ(asm1("vmset.m v12"), asm1("vmxnor.mm v12, v12, v12"));
    ASSERT_HEX_EQ(asm1("vmset.m v12"), 0x7ec6'2657u);
    return true;
}

TEST(asm_vmnot_m) {
    ASSERT_HEX_EQ(asm1("vmnot.m v13, v14"), asm1("vmnand.mm v13, v14, v14"));
    ASSERT_HEX_EQ(asm1("vmnot.m v13, v14"), 0x76e7'26d7u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. CSR and system instructions
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_csrrw) {
    ASSERT_HEX_EQ(asm1("csrrw x1, mstatus, x2"), asm1("csrrw x1, 0x300, x2"));
    ASSERT_HEX_EQ(asm1("csrrw x1, mstatus, x2"), 0x3001'10f3u);
    return true;
}
 
TEST(asm_csrrs) {
    ASSERT_HEX_EQ(asm1("csrrs x5, mie, x6"), asm1("csrrs x5, 0x304, x6"));
    ASSERT_HEX_EQ(asm1("csrrs x5, mie, x6"), 0x3043'22f3u);
    return true;
}

TEST(asm_csrrc) {
    ASSERT_HEX_EQ(asm1("csrrc x9, mtvec, x10"), asm1("csrrc x9, 0x305, x10"));
    ASSERT_HEX_EQ(asm1("csrrc x9, mtvec, x10"), 0x3055'34f3u);
    return true;
}

TEST(asm_csrrwi) {
    ASSERT_HEX_EQ(asm1("csrrwi x13, mepc, 5"), asm1("csrrwi x13, 0x341, 5"));
    ASSERT_HEX_EQ(asm1("csrrwi x13, mepc, 5"), 0x3412'd6f3u);
    return true;
}

TEST(asm_csrrsi) {
    ASSERT_HEX_EQ(asm1("csrrsi x15, mcause, 3"), asm1("csrrsi x15, 0x342, 3"));
    ASSERT_HEX_EQ(asm1("csrrsi x15, mcause, 3"), 0x3421'e7f3u);
    return true;
}

TEST(asm_csrrci) {
    ASSERT_HEX_EQ(asm1("csrrci x17, mip, 7"), asm1("csrrci x17, 0x344, 7"));
    ASSERT_HEX_EQ(asm1("csrrci x17, mip, 7"), 0x3443'f8f3u);
    return true;
}

TEST(asm_system_instructions) {
    ASSERT_HEX_EQ(asm1("ecall"),  0x0000'0073u);
    ASSERT_HEX_EQ(asm1("ebreak"), 0x0010'0073u);
    ASSERT_HEX_EQ(asm1("mret"),   0x3020'0073u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. Integration — assemble and run through CPU
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_run_sum_1_to_10) {
    const char* src = R"(
        addi x1, x0, 0     # sum = 0
        addi x2, x0, 1     # i = 1
        addi x3, x0, 11    # limit = 11
    loop:
        add  x1, x1, x2    # sum += i
        addi x2, x2, 1     # i++
        bne  x2, x3, loop  # if i != 11, goto loop
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 55u);
    return true;
}

TEST(asm_run_fibonacci) {
    const char* src = R"(
        addi x1, x0, 0   ; fib_prev = 0
        addi x2, x0, 1   ; fib_curr = 0
        addi x3, x0, 9   ; 9 iterations
    loop:
        add  x4, x1, x2  ; next = prev + curr
        mv   x1, x2      ; prev = curr
        mv   x2, x4      ; curr = next
        addi x3, x3, -1
        bnez x3, loop
        mv   x1, x2      ; result in x1
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 55u);
    return true;
}

TEST(asm_run_memory_store_load) {
    const char* src = R"(
        addi x1, x0, 42
        lui  x2, 0x8    // x2 = 0x8000
        sw   x1, 0(x2)  // mem[0x8000] = 42
        lw   x3, 0(x2)  // x3 = mem[0x8000]
        mv   x1, x3     // result in x1
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 42u);
    return true;
}

TEST(asm_run_j_skip) {
    const char* src = R"(
        j skip
        addi x1, x0, 99  # should be skipped
    skip:
        addi x1, x0, 42
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 42u);
    return true;
}

TEST(asm_run_j_skip_with_blank_line) {
    // Ensure code can be written with lines of whitespace
    const char* src = R"(
        j skip
        addi x1, x0, 99  # should be skipped

    skip:
        addi x1, x0, 42
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 42u);
    return true;
}

TEST(asm_run_li_large) {
    const char* src = R"(
        li x1, 0x12345
        ebreak
    )";
    ASSERT_EQ(run_asm(src, 1), 0x12345u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  10. Misc. edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(asm_error_unknown_instruction) {
    Assembler as;
    auto r = as.assemble("foobar x1, x2, x3");
    ASSERT(!r.ok);
    ASSERT(r.errors.size() > 0u);
    return true;
}

TEST(asm_error_duplicate_label) {
    Assembler as;
    auto r = as.assemble("foo:\nfoo:\n");
    ASSERT(!r.ok);
    return true;
}

TEST(asm_error_four_operands) {
    Assembler as;
    auto r = as.assemble("add x1, x2, x3, x4");
    ASSERT(!r.ok);
    return true;
}

TEST(asm_error_junk_after_parentheses) {
    Assembler as;
    auto r = as.assemble("jalr x0, (x1)foo");
    ASSERT(!r.ok);
    return true;
}
