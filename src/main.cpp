/**
 * @file main.cpp
 * @brief CLI runner for the RV32I CPU simulator.
 *
 * Loads a raw binary RISC-V machine code file into simulated memory,
 * executes it, and optionally prints a register dump and execution
 * statistics. Execution stops on EBREAK, an invalid instruction, or
 * after reaching the instruction limit.
 *
 * The program's exit code is taken from register a0 (x10), clamped to
 * 0-255, matching the RISC-V Linux syscall convention.
 *
 * Usage:
 *   rvsim [options] <binary>
 *   rvsim --help
 */

#include "core/cpu.hpp"

#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

using namespace riscv;

static constexpr const char* kVersion = "0.1.0";

static void print_usage(const char* prog)
{
    std::cerr
        << std::format("rvsim v{} — RV32I CPU simulator\n\n", kVersion)
        << std::format("Usage: {} [options] <binary>\n\n", prog)
        << "Options:\n"
        << "  -t, --trace      Enable instruction tracing\n"
        << "  -n <count>       Maximum instructions to execute (default: 10M)\n"
        << "  -b <addr>        Binary load address, hex (default: 0)\n"
        << "  -e <addr>        Entry point, hex (default: load address)\n"
        << "  -m <size>        Memory size in KiB (default: 64)\n"
        << "  -d               Dump registers after execution\n"
        << "  -s               Print execution statistics\n"
        << "  -v, --version    Print version and exit\n"
        << "  -h, --help       Show this help\n"
        << "\n"
        << "The binary should be a flat (headerless) RISC-V machine code file.\n"
        << "Execution halts on EBREAK or error.  Exit code = a0 & 0xFF.\n";
}

int main(int argc, char** argv)
{
    bool        trace            = false;
    bool        dump_regs        = false;
    bool        print_stats      = false;
    u64         max_instructions = 10'000'000;
    addr_t      base_addr        = 0;
    addr_t      entry_point      = 0;
    bool        entry_set        = false;
    size_t      mem_size_kb      = 64;
    const char* binary_path      = nullptr;
    
    /* ── Argument Parsing ─────────────────────────────────────────────── */

        for (int i = 1; i < argc; ++i) {
        const auto match = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
 
        try {
            if (match("-t") || match("--trace")) {
                trace = true;
            } else if (match("-d")) {
                dump_regs = true;
            } else if (match("-s")) {
                print_stats = true;
            } else if (match("-n") && i + 1 < argc) {
                max_instructions = std::stoull(argv[++i]);
            } else if (match("-b") && i + 1 < argc) {
                base_addr = static_cast<addr_t>(std::stoul(argv[++i], nullptr, 16));
            } else if (match("-e") && i + 1 < argc) {
                entry_point = static_cast<addr_t>(std::stoul(argv[++i], nullptr, 16));
                entry_set = true;
            } else if (match("-m") && i + 1 < argc) {
                mem_size_kb = std::stoul(argv[++i]);
            } else if (match("-v") || match("--version")) {
                std::cout << std::format("rvsim v{}\n", kVersion);
                return 0;
            } else if (match("-h") || match("--help")) {
                print_usage(argv[0]);
                return 0;
            } else if (argv[i][0] != '-') {
                binary_path = argv[i];
            } else {
                std::cerr << std::format("Unknown option: {}\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << std::format("Bad argument for {}: {}\n", argv[i - 1], e.what());
            return 1;
        }
    }
 
    if (!binary_path) {
        print_usage(argv[0]);
        return 1;
    }
 
    if (!entry_set) {
        entry_point = base_addr;
    }
    
    /* ── Load Binary ──────────────────────────────────────────────────── */

    std::ifstream file(binary_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << std::format("Error: cannot open file: {}\n", binary_path);
        return 1;
    }
    
    auto file_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    
    std::vector<u8> program(file_size);
    file.read(reinterpret_cast<char*>(program.data()), 
              static_cast<std::streamsize>(file_size));
    
    if (!file) {
        std::cerr << "Error: Failed to read file\n";
        return 1;
    }

    std::cerr << std::format("Loaded {} bytes from {}\n", file_size, binary_path)
              << std::format("Base address: 0x{:08x}\n", base_addr)
              << std::format("Entry point:  0x{:08x}\n", entry_point);

    /* ── Create CPU ───────────────────────────────────────────────────── */

    const size_t mem_bytes = mem_size_kb * 1024;
    auto memory = std::make_shared<FlatMemory>(0, mem_bytes);
    CPU cpu(memory);
    
    cpu.load_program(base_addr, program);
    cpu.set_pc(entry_point);
    
    /* Initialize stack pointer (x2 / sp) to the top of memory, 16-byte aligned. */
    constexpr reg_idx_t kSP = 2;
    cpu.set_reg(kSP, static_cast<u32>(mem_bytes - 16));
    
    if (trace) {
        cpu.set_trace(true);
    }

    /* ── Execute ──────────────────────────────────────────────────────── */    

    std::cerr << "Starting execution...\n";
    u64 executed = cpu.run(max_instructions);
    std::cerr << std::format("Execution stopped after {} instructions\n", executed);
    
    if (dump_regs) {
        std::cerr << "\nRegister state:\n";
        cpu.dump_regs();
    }
    
    if (print_stats) {
        const auto& s = cpu.stats();
        std::cerr << std::format(
            "\nStatistics:\n"
            "  Instructions: {}\n"
            "  Cycles:       {}\n"
            "  IPC:          {:.3f}\n"
            "  Loads:        {}\n"
            "  Stores:       {}\n"
            "  Branches:     {} ({:.1f}% taken)\n"
            "  Jumps:        {}\n",
            s.instructions, s.cycles, s.ipc(),
            s.loads, s.stores,
            s.branches, s.branch_taken_rate() * 100.0,
            s.jumps); 
    }
    
    int exit_code = static_cast<int>(cpu.reg(10) & 0xFF);
    return exit_code;
}
