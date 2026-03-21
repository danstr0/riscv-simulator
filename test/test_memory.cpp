/**
 * @file test_memory.cpp
 * @brief Tests for FlatMemory and MMIOBus.
 *
 * Sections:
 *   FlatMemory  — read/write at all widths, little-endian layout,
 *    (line 75)    out-of-bounds rejection, bulk load, valid_address.
 *   MMIOBus     — single/multiple region routing, offset translation,
 *    (line 243)   default memory fallback, unmap, region boundaries,
 *                 8/16-bit forwarding, custom MMIO device with cycle cost.
 */

#include "core/memory.hpp"

#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace riscv;

// ── Shared test infrastructure ──────────────────────────────────────────────

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

/* ═══════════════════════════════════════════════════════════════════════
 *  FlatMemory
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(flat_mem_basic_read_write) {
    FlatMemory mem(0x1000, 0x1000);

    auto wr = mem.write32(0x1000, 0xDEADBEEF);
    ASSERT(wr.ok);

    auto rd = mem.read32(0x1000);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xDEADBEEFu);
    ASSERT_EQ(rd.cycles, 1u);
    return true;
}

TEST(flat_mem_byte_access) {
    FlatMemory mem(0x0, 0x100);

    auto wr = mem.write8(0x10, 0xAB);
    ASSERT(wr.ok);

    auto rd = mem.read8(0x10);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xABu);
    return true;
}

TEST(flat_mem_half_access) {
    FlatMemory mem(0x0, 0x100);

    auto wr = mem.write16(0x20, 0x1234);
    ASSERT(wr.ok);

    auto rd = mem.read16(0x20);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0x1234u);
    return true;
}

TEST(flat_mem_little_endian_32) {
    FlatMemory mem(0x0, 0x100);

    mem.write32(0x0, 0x04030201);

    // Individual bytes in memory should be little-endian.
    ASSERT_HEX_EQ(mem.read8(0x0).value, 0x01u);
    ASSERT_HEX_EQ(mem.read8(0x1).value, 0x02u);
    ASSERT_HEX_EQ(mem.read8(0x2).value, 0x03u);
    ASSERT_HEX_EQ(mem.read8(0x3).value, 0x04u);
    return true;
}

TEST(flat_mem_little_endian_16) {
    FlatMemory mem(0x0, 0x100);

    mem.write16(0x0, 0xBEEF);

    ASSERT_HEX_EQ(mem.read8(0x0).value, 0xEFu);
    ASSERT_HEX_EQ(mem.read8(0x1).value, 0xBEu);
    return true;
}

TEST(flat_mem_cross_width_read) {
    // Write bytes individually, read back as wider types.
    FlatMemory mem(0x0, 0x100);

    mem.write8(0x0, 0x78);
    mem.write8(0x1, 0x56);
    mem.write8(0x2, 0x34);
    mem.write8(0x3, 0x12);

    ASSERT_HEX_EQ(mem.read16(0x0).value, 0x5678u);
    ASSERT_HEX_EQ(mem.read32(0x0).value, 0x12345678u);
    return true;
}

TEST(flat_mem_read8_no_sign_extend) {
    // read8 must return zero-extended values — upper bits must be 0.
    // This matters for LBU execution.
    FlatMemory mem(0x0, 0x100);

    mem.write8(0x0, 0xFF);  // All bits set (would be -1 if sign-extended)

    auto rd = mem.read8(0x0);
    ASSERT_EQ(rd.value, 0x000000FFu);  // Must be zero-extended, not 0xFFFFFFFF
    return true;
}

TEST(flat_mem_read16_no_sign_extend) {
    FlatMemory mem(0x0, 0x100);

    mem.write16(0x0, 0xFFFF);

    auto rd = mem.read16(0x0);
    ASSERT_EQ(rd.value, 0x0000FFFFu);
    return true;
}

TEST(flat_mem_out_of_bounds) {
    FlatMemory mem(0x1000, 0x100);  // 0x1000–0x10FF

    // Before range
    ASSERT(!mem.read32(0x0FFC).ok);
    // After range
    ASSERT(!mem.read32(0x1100).ok);
    // Partially out (last byte would be at 0x1100)
    ASSERT(!mem.read32(0x10FD).ok);

    // Writes should also fail
    ASSERT(!mem.write32(0x0FFC, 0).ok);
    ASSERT(!mem.write32(0x1100, 0).ok);
    return true;
}

TEST(flat_mem_out_of_bounds_sub_word) {
    FlatMemory mem(0x0, 0x10);  // 16 bytes: 0x0–0xF

    // read16 at 0xF: needs 2 bytes but only 1 left
    ASSERT(!mem.read16(0xF).ok);
    // write16 at 0xF
    ASSERT(!mem.write16(0xF, 0).ok);
    // read8 at 0xF: exactly at last byte — should succeed
    ASSERT(mem.read8(0xF).ok);
    // read8 at 0x10: one past end — should fail
    ASSERT(!mem.read8(0x10).ok);
    return true;
}

TEST(flat_mem_load_bulk) {
    FlatMemory mem(0x0, 0x100);

    u8 data[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    mem.load(0x10, std::span<const u8>(data, sizeof(data)));

    ASSERT_HEX_EQ(mem.read8(0x10).value, 0x11u);
    ASSERT_HEX_EQ(mem.read8(0x17).value, 0x88u);
    ASSERT_HEX_EQ(mem.read32(0x10).value, 0x44332211u);
    return true;
}

TEST(flat_mem_load_bulk_out_of_range) {
    FlatMemory mem(0x0, 0x10);

    u8 data[32] = {};
    bool threw = false;
    try {
        mem.load(0x0, std::span<const u8>(data, sizeof(data)));
    } catch (const MemoryAccessException&) {
        threw = true;
    }
    ASSERT(threw);
    return true;
}

TEST(flat_mem_valid_address) {
    FlatMemory mem(0x1000, 0x100);

    ASSERT(mem.valid_address(0x1000));
    ASSERT(mem.valid_address(0x10FF));
    ASSERT(!mem.valid_address(0x0FFF));
    ASSERT(!mem.valid_address(0x1100));
    ASSERT(mem.valid_address(0x1000, 4));
    ASSERT(!mem.valid_address(0x10FD, 4));  // Would span past end
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  MMIOBus
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(mmio_bus_single_region) {
    auto bus = std::make_shared<MMIOBus>();
    auto ram = std::make_shared<FlatMemory>(0x0, 0x1000);

    bus->map(0x0, 0x1000, ram, "RAM");

    bus->write32(0x100, 0xCAFEBABE);
    auto rd = bus->read32(0x100);

    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xCAFEBABEu);
    return true;
}

TEST(mmio_bus_multiple_regions) {
    auto bus = std::make_shared<MMIOBus>();
    auto ram = std::make_shared<FlatMemory>(0x0, 0x1000);
    auto rom = std::make_shared<FlatMemory>(0x0, 0x1000);

    rom->write32(0x0, 0x12345678);

    bus->map(0x00000000, 0x1000, ram, "RAM");
    bus->map(0x10000000, 0x1000, rom, "ROM");

    bus->write32(0x00000100, 0xAAAAAAAA);

    ASSERT_HEX_EQ(bus->read32(0x00000100).value, 0xAAAAAAAAu);
    ASSERT_HEX_EQ(bus->read32(0x10000000).value, 0x12345678u);
    return true;
}

TEST(mmio_bus_region_offset) {
    auto bus    = std::make_shared<MMIOBus>();
    auto device = std::make_shared<FlatMemory>(0x0, 0x100);

    bus->map(0x80000000, 0x100, device, "Device");

    bus->write32(0x80000010, 0xDEADC0DE);

    // Device sees the write at offset 0x10 (not absolute 0x80000010).
    ASSERT_HEX_EQ(device->read32(0x10).value, 0xDEADC0DEu);
    return true;
}

TEST(mmio_bus_region_boundary) {
    // Verify that the last byte of a region is accessible and the first
    // byte after it falls through to default (or fails).
    auto bus     = std::make_shared<MMIOBus>();
    auto device  = std::make_shared<FlatMemory>(0x0, 0x100);

    bus->map(0x2000, 0x100, device, "Dev");

    // Last valid byte write/read.
    auto wr = bus->write8(0x20FF, 0x42);
    ASSERT(wr.ok);
    ASSERT_EQ(bus->read8(0x20FF).value, 0x42u);

    // First byte past the region — no default set, should fail.
    ASSERT(!bus->read8(0x2100).ok);
    return true;
}

TEST(mmio_bus_byte_and_half_forwarding) {
    // Verify 8-bit and 16-bit operations are forwarded correctly through
    // the bus (not just 32-bit).
    auto bus    = std::make_shared<MMIOBus>();
    auto device = std::make_shared<FlatMemory>(0x0, 0x100);

    bus->map(0x3000, 0x100, device, "Dev");

    bus->write8(0x3010, 0xAB);
    ASSERT_HEX_EQ(bus->read8(0x3010).value, 0xABu);

    bus->write16(0x3020, 0xCDEF);
    ASSERT_HEX_EQ(bus->read16(0x3020).value, 0xCDEFu);

    // Cross-width: write 16-bit via bus, read bytes from device.
    ASSERT_HEX_EQ(device->read8(0x20).value, 0xEFu);
    ASSERT_HEX_EQ(device->read8(0x21).value, 0xCDu);
    return true;
}

TEST(mmio_bus_default_memory) {
    auto bus         = std::make_shared<MMIOBus>();
    auto default_mem = std::make_shared<FlatMemory>(0x0, 0x10000);
    auto device      = std::make_shared<FlatMemory>(0x0, 0x100);

    bus->set_default(default_mem);
    bus->map(0x1000, 0x100, device, "Device");

    // Unmapped region -> default
    bus->write32(0x5000, 0x11111111);
    ASSERT_HEX_EQ(default_mem->read32(0x5000).value, 0x11111111u);

    // Mapped region → device
    bus->write32(0x1020, 0x22222222);
    ASSERT_HEX_EQ(device->read32(0x20).value, 0x22222222u);
    return true;
}

TEST(mmio_bus_unmap) {
    auto bus         = std::make_shared<MMIOBus>();
    auto default_mem = std::make_shared<FlatMemory>(0x0, 0x10000);
    auto device      = std::make_shared<FlatMemory>(0x0, 0x100);

    bus->set_default(default_mem);
    bus->map(0x1000, 0x100, device, "Device");

    // Goes to device
    bus->write32(0x1000, 0xAAAAAAAA);
    ASSERT_HEX_EQ(device->read32(0x0).value, 0xAAAAAAAAu);

    bus->unmap(0x1000);

    // Now goes to default
    bus->write32(0x1000, 0xBBBBBBBB);
    ASSERT_HEX_EQ(default_mem->read32(0x1000).value, 0xBBBBBBBBu);
    return true;
}

TEST(mmio_bus_no_region_fails) {
    MMIOBus bus;  // No default, no regions

    ASSERT(!bus.read32(0x1000).ok);
    ASSERT(!bus.write32(0x1000, 0).ok);
    ASSERT(!bus.read8(0x0).ok);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Custom MMIO device
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Counter device: read32 at offset 0 returns (and increments) a counter.
 * Demonstrates mutable state behind the const-read interface.
 */
class CounterDevice : public Memory {
public:
    mutable u32 counter = 0;  // mutable: reads have side effects (MMIO)
    u32 access_cycles   = 5;

    MemoryResult read32(addr_t addr) const override
    {
        if (addr == 0) return {counter++, access_cycles, true};
        return {0, 1, false};
    }
    MemoryResult read16(addr_t) const override { return {0, 1, false}; }
    MemoryResult read8(addr_t)  const override { return {0, 1, false}; }

    MemoryResult write32(addr_t addr, u32 value) override
    {
        if (addr == 0) { counter = value; return {value, access_cycles, true}; }
        return {0, 1, false};
    }
    MemoryResult write16(addr_t, u16) override { return {0, 1, false}; }
    MemoryResult write8(addr_t, u8)   override { return {0, 1, false}; }

    void load(addr_t, std::span<const u8>) override {}
    bool valid_address(addr_t addr, size_t) const override { return addr == 0; }
};

TEST(mmio_custom_device) {
    auto bus     = std::make_shared<MMIOBus>();
    auto counter = std::make_shared<CounterDevice>();

    bus->map(0x10000000, 0x4, counter, "Counter");

    // Each read returns an incrementing value.
    ASSERT_EQ(bus->read32(0x10000000).value, 0u);
    ASSERT_EQ(bus->read32(0x10000000).value, 1u);
    ASSERT_EQ(bus->read32(0x10000000).value, 2u);

    // Verify cycle cost propagates through the bus.
    ASSERT_EQ(bus->read32(0x10000000).cycles, 5u);

    // Reset counter via write.
    bus->write32(0x10000000, 100);
    ASSERT_EQ(bus->read32(0x10000000).value, 100u);
    return true;
}
