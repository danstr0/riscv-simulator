/**
 * @file test_memory.cpp
 * @brief Tests for FlatMemory and MMIOBus.
 *
 * Sections:
 *   1 (line  16) : FlatMemory
 *   2 (line 186) : MMIOBus
 */

#include "core/memory.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
//  1. FlatMemory
// ═══════════════════════════════════════════════════════════════════════

TEST(flat_mem_basic_read_write) {
    FlatMemory mem(0x1000u, 0x1000u);

    auto wr = mem.write32(0x1000u, 0xDEAD'BEEFu);
    ASSERT(wr.ok);

    auto rd = mem.read32(0x1000u);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xDEAD'BEEFu);
    ASSERT_EQ(rd.cycles, 1u);
    return true;
}

TEST(flat_mem_byte_access) {
    FlatMemory mem(0x0u, 0x100u);

    auto wr = mem.write8(0x10u, 0xABu);
    ASSERT(wr.ok);

    auto rd = mem.read8(0x10u);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xABu);
    return true;
}

TEST(flat_mem_half_access) {
    FlatMemory mem(0x0u, 0x100u);

    auto wr = mem.write16(0x20u, 0x1234u);
    ASSERT(wr.ok);

    auto rd = mem.read16(0x20u);
    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0x1234u);
    return true;
}

TEST(flat_mem_little_endian_32) {
    FlatMemory mem(0x0u, 0x100u);

    mem.write32(0x0u, 0x0403'0201u);

    // Individual bytes in memory should be little-endian
    ASSERT_HEX_EQ(mem.read8(0x0u).value, 0x01u);
    ASSERT_HEX_EQ(mem.read8(0x1u).value, 0x02u);
    ASSERT_HEX_EQ(mem.read8(0x2u).value, 0x03u);
    ASSERT_HEX_EQ(mem.read8(0x3u).value, 0x04u);
    return true;
}

TEST(flat_mem_little_endian_16) {
    FlatMemory mem(0x0u, 0x100u);

    mem.write16(0x0u, 0xBEEFu);

    ASSERT_HEX_EQ(mem.read8(0x0u).value, 0xEFu);
    ASSERT_HEX_EQ(mem.read8(0x1u).value, 0xBEu);
    return true;
}

TEST(flat_mem_cross_width_read) {
    // Write bytes individually, read back as wider types
    FlatMemory mem(0x0u, 0x100u);

    mem.write8(0x0u, 0x78u);
    mem.write8(0x1u, 0x56u);
    mem.write8(0x2u, 0x34u);
    mem.write8(0x3u, 0x12u);

    ASSERT_HEX_EQ(mem.read16(0x0u).value, 0x5678u);
    ASSERT_HEX_EQ(mem.read32(0x0u).value, 0x1234'5678u);
    return true;
}

TEST(flat_mem_read8_no_sign_extend) {
    // read8 must return zero-extended values — upper bits must be 0
    FlatMemory mem(0x0u, 0x100u);

    mem.write8(0x0u, 0xFFu); // all bits set (would be -1 if sign-extended)

    auto rd = mem.read8(0x0u);
    ASSERT_EQ(rd.value, 0x0000'00FFu); // must be zero-extended, not 0xFFFF'FFFF
    return true;
}

TEST(flat_mem_read16_no_sign_extend) {
    FlatMemory mem(0x0u, 0x100u);

    mem.write16(0x0u, 0xFFFFu);

    auto rd = mem.read16(0x0u);
    ASSERT_EQ(rd.value, 0x0000'FFFFu);
    return true;
}

TEST(flat_mem_out_of_bounds) {
    FlatMemory mem(0x1000u, 0x100u); // 0x1000–0x10FF

    // Before range
    ASSERT(!mem.read32(0x0FFCu).ok);
    // After range
    ASSERT(!mem.read32(0x1100u).ok);
    // Partially out (last byte would be at 0x1100)
    ASSERT(!mem.read32(0x10FDu).ok);

    // Writes should also fail
    ASSERT(!mem.write32(0x0FFCu, 0).ok);
    ASSERT(!mem.write32(0x1100u, 0).ok);
    return true;
}

TEST(flat_mem_out_of_bounds_sub_word) {
    FlatMemory mem(0x0u, 0x10u); // 16 bytes: 0x0–0xF

    // read16 at 0xF: needs 2 bytes but only 1 left
    ASSERT(!mem.read16(0xFu).ok);
    // write16 at 0xF
    ASSERT(!mem.write16(0xFu, 0).ok);
    // read8 at 0xF: exactly at last byte — should succeed
    ASSERT(mem.read8(0xFu).ok);
    // read8 at 0x10: one past end — should fail
    ASSERT(!mem.read8(0x10).ok);
    return true;
}

TEST(flat_mem_load_bulk) {
    FlatMemory mem(0x0u, 0x100u);

    u8 data[] = {0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u};
    mem.load(0x10u, std::span<const u8>(data, sizeof(data)));

    ASSERT_HEX_EQ(mem.read8(0x10u).value, 0x11u);
    ASSERT_HEX_EQ(mem.read8(0x17u).value, 0x88u);
    ASSERT_HEX_EQ(mem.read32(0x10u).value, 0x4433'2211u);
    return true;
}

TEST(flat_mem_load_bulk_out_of_range) {
    FlatMemory mem(0x0u, 0x10u);

    u8 data[32] = {};
    bool threw = false;
    try
    {
        mem.load(0x0u, std::span<const u8>(data, sizeof(data)));
    }
    catch (const MemoryAccessException&) 
    {
        threw = true;
    }
    ASSERT(threw);
    return true;
}

TEST(flat_mem_valid_address) {
    FlatMemory mem(0x1000u, 0x100u);

    ASSERT(mem.valid_address(0x1000u));
    ASSERT(mem.valid_address(0x10FFu));
    ASSERT(!mem.valid_address(0x0FFFu));
    ASSERT(!mem.valid_address(0x1100u));
    ASSERT(mem.valid_address(0x1000u, 4));
    ASSERT(!mem.valid_address(0x10FDu, 4)); // would span past end
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. MMIOBus
// ═══════════════════════════════════════════════════════════════════════

TEST(mmio_bus_single_region) {
    auto bus = std::make_shared<MMIOBus>();
    auto ram = std::make_shared<FlatMemory>(0x0u, 0x1000u);

    bus->map(0x0u, 0x1000u, ram, "RAM");

    bus->write32(0x100u, 0xCAFE'BABEu);
    auto rd = bus->read32(0x100u);

    ASSERT(rd.ok);
    ASSERT_HEX_EQ(rd.value, 0xCAFE'BABEu);
    return true;
}

TEST(mmio_bus_multiple_regions) {
    auto bus = std::make_shared<MMIOBus>();
    auto ram = std::make_shared<FlatMemory>(0x0u, 0x1000u);
    auto rom = std::make_shared<FlatMemory>(0x0u, 0x1000u);

    rom->write32(0x0u, 0x1234'5678u);

    bus->map(0x0000'0000u, 0x1000u, ram, "RAM");
    bus->map(0x1000'0000u, 0x1000u, rom, "ROM");

    bus->write32(0x0000'0100u, 0xAAAA'AAAAu);

    ASSERT_HEX_EQ(bus->read32(0x0000'0100u).value, 0xAAAA'AAAAu);
    ASSERT_HEX_EQ(bus->read32(0x1000'0000u).value, 0x1234'5678u);
    return true;
}

TEST(mmio_bus_region_offset) {
    auto bus    = std::make_shared<MMIOBus>();
    auto device = std::make_shared<FlatMemory>(0x0u, 0x100u);

    bus->map(0x8000'0000u, 0x100u, device, "Device");

    bus->write32(0x8000'0010u, 0xDEAD'C0DEu);

    // Device sees the write at offset 0x10 (not absolute 0x8000'0010)
    ASSERT_HEX_EQ(device->read32(0x10u).value, 0xDEAD'C0DEu);
    return true;
}

TEST(mmio_bus_region_boundary) {
    // Verify that the last byte of a region is accessible and the first
    // byte after it falls through to default (or fails)
    auto bus     = std::make_shared<MMIOBus>();
    auto device  = std::make_shared<FlatMemory>(0x0u, 0x100u);

    bus->map(0x2000u, 0x100u, device, "Dev");

    // Last valid byte write/read
    auto wr = bus->write8(0x20FFu, 0x42u);
    ASSERT(wr.ok);
    ASSERT_EQ(bus->read8(0x20FFu).value, 0x42u);

    // First byte past the region — no default set, should fail
    ASSERT(!bus->read8(0x2100u).ok);
    return true;
}

TEST(mmio_bus_byte_and_half_forwarding) {
    // Verify 8-bit and 16-bit operations are forwarded correctly through
    // the bus (not just 32-bit)
    auto bus    = std::make_shared<MMIOBus>();
    auto device = std::make_shared<FlatMemory>(0x0u, 0x100u);

    bus->map(0x3000u, 0x100u, device, "Dev");

    bus->write8(0x3010u, 0xABu);
    ASSERT_HEX_EQ(bus->read8(0x3010u).value, 0xABu);

    // Cross-width: write 16-bit via bus, read bytes from device
    bus->write16(0x3020u, 0xCDEFu);
    ASSERT_HEX_EQ(bus->read16(0x3020u).value, 0xCDEFu);

    ASSERT_HEX_EQ(device->read8(0x20u).value, 0xEFu);
    ASSERT_HEX_EQ(device->read8(0x21u).value, 0xCDu);
    return true;
}

TEST(mmio_bus_default_memory) {
    auto bus         = std::make_shared<MMIOBus>();
    auto default_mem = std::make_shared<FlatMemory>(0x0u, 0x10000u);
    auto device      = std::make_shared<FlatMemory>(0x0u, 0x100u);

    bus->set_default(default_mem);
    bus->map(0x1000u, 0x100u, device, "Device");

    // Unmapped region → default
    bus->write32(0x5000u, 0x1111'1111u);
    ASSERT_HEX_EQ(default_mem->read32(0x5000u).value, 0x1111'1111u);

    // Mapped region → device
    bus->write32(0x1020u, 0x2222'2222u);
    ASSERT_HEX_EQ(device->read32(0x20u).value, 0x2222'2222u);
    return true;
}

TEST(mmio_bus_unmap) {
    auto bus         = std::make_shared<MMIOBus>();
    auto default_mem = std::make_shared<FlatMemory>(0x0u, 0x10000u);
    auto device      = std::make_shared<FlatMemory>(0x0u, 0x100u);

    bus->set_default(default_mem);
    bus->map(0x1000u, 0x100u, device, "Device");

    // Goes to device
    bus->write32(0x1000u, 0xAAAA'AAAAu);
    ASSERT_HEX_EQ(device->read32(0x0u).value, 0xAAAA'AAAAu);

    bus->unmap(0x1000u);

    // Now goes to default
    bus->write32(0x1000u, 0xBBBB'BBBBu);
    ASSERT_HEX_EQ(default_mem->read32(0x1000u).value, 0xBBBB'BBBBu);
    return true;
}

TEST(mmio_bus_no_region_fails) {
    MMIOBus bus; // No default, no regions

    ASSERT(!bus.read32(0x1000u).ok);
    ASSERT(!bus.write32(0x1000u, 0).ok);
    ASSERT(!bus.read8(0x0u).ok);
    return true;
}

// ── Custom MMIO device ─────────────────────────────────────────────────

/**
 * Counter device: read32 at offset 0 returns (and increments) a counter.
 * Demonstrates mutable state behind the const-read interface.
 */
class CounterDevice : public Memory {
public:
    mutable u32 counter = 0; // mutable: reads have side effects (MMIO)
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

    bus->map(0x1000'0000u, 0x4u, counter, "Counter");

    // Each read returns an incrementing value
    ASSERT_EQ(bus->read32(0x1000'0000u).value, 0u);
    ASSERT_EQ(bus->read32(0x1000'0000u).value, 1u);
    ASSERT_EQ(bus->read32(0x1000'0000u).value, 2u);

    // Verify cycle cost propagates through the bus
    ASSERT_EQ(bus->read32(0x1000'0000u).cycles, 5u);

    // Reset counter via write
    bus->write32(0x1000'0000u, 100u);
    ASSERT_EQ(bus->read32(0x1000'0000u).value, 100u);
    return true;
}
