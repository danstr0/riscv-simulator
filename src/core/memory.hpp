/**
 * @file memory.hpp
 * @brief Memory subsystem and bus routing for the RV32I CPU simulator.
 *
 * This header defines a unified memory interface allowing the CPU to interact
 * with various memory-backed resources (RAM, ROM, MMIO) transparently.
 *
 * @section ADDRESS_TRANSLATION Address Translation
 * The system utilizes a tiered addressing model:
 * 1. **Absolute Addresses:** Used by the @ref MMIOBus to route requests.
 * 2. **Relative Offsets:** Used by concrete devices (like @ref FlatMemory).
 * When a request is routed through the bus, the base address is subtracted,
 * presenting the device with a zero-based offset.
 *
 * @section LATENCY_MODELING Latency Modeling
 * While currently functional as a functional simulator (1 cycle/access),
 * @ref MemoryResult includes a @c cycles field to support future cycle-accurate
 * timing or cache-miss penalty injection.
 */

#pragma once

#include "types.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace riscv {

/* ═══════════════════════════════════════════════════════════════════════
 * Memory Operation Result
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Encapsulates the outcome of a memory transaction.
 * * Callers must verify @ref ok before consuming @ref value. A failure (@c ok == false)
 * typically indicates a @ref MemoryAccessException or @ref MisalignedAccessException.
 */
struct [[nodiscard]] MemoryResult {
    u32 value  = 0;     ///< Data retrieved on reads; undefined for writes.
    u32 cycles = 1;     ///< Latency incurred by this specific transaction.
    bool ok    = true;  ///< Transaction status; false if a fault occurred.
};

/* ═══════════════════════════════════════════════════════════════════════
 * Abstract Memory Interface
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Pure virtual interface for all memory-mapped entities.
 *
 * All implementations must be byte-addressable and follow the host's native
 * endianness (assumed little-endian).
 */
class Memory {
public:
    virtual ~Memory() = default;

    /**
     * @name Read Interface
     * @{
     */
    [[nodiscard]] virtual MemoryResult read32(addr_t addr) const = 0;
    [[nodiscard]] virtual MemoryResult read16(addr_t addr) const = 0;
    [[nodiscard]] virtual MemoryResult read8(addr_t addr)  const = 0;
    /** @} */

    /**
     * @name Write Interface
     * @{
     */
    virtual MemoryResult write32(addr_t addr, u32 value) = 0;
    virtual MemoryResult write16(addr_t addr, u16 value) = 0;
    virtual MemoryResult write8(addr_t addr, u8 value)   = 0;
    /** @} */

    /**
     * @brief Performs a bulk binary load into memory.
     * @param addr  Starting address for the load.
     * @param data  Span of bytes to copy into the memory region.
     * @throw std::out_of_range If the data exceeds region boundaries.
     */
    virtual void load(addr_t addr, std::span<const u8> data) = 0;

    /** @brief Validates if a memory range is accessible. */
    [[nodiscard]] virtual bool valid_address(addr_t addr, size_t size = 1) const = 0;
};

/* ═══════════════════════════════════════════════════════════════════════
 * Flat (contiguous) RAM
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Contiguous, byte-addressable RAM region.
 *
 * Backed by a @c std::vector, this class represents a physical block of memory.
 * @note When used as a device behind @ref MMIOBus, @c base_addr is typically @c 0.
 */
class FlatMemory : public Memory {
public:
    /**
     * @brief Constructs a RAM region.
     * @param base_addr  The logical start address (often 0 for relative devices).
     * @param size  Size of the region in bytes.
     */
    FlatMemory(addr_t base_addr, size_t size);

    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    
    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

    /** @name Debug/Instrospection Helpers */
    /** @{ */
    [[nodiscard]] addr_t    base() const noexcept { return base_addr_; }
    [[nodiscard]] size_t    size() const noexcept { return ram_.size(); }
    [[nodiscard]] const u8* data() const noexcept { return ram_.data(); }
    [[nodiscard]] u8*       data()       noexcept { return ram_.data(); }
    /** @} */

private:
    addr_t          base_addr_;
    std::vector<u8> ram_;

    [[nodiscard]] bool in_range(addr_t addr, size_t len) const noexcept
    {
        if (addr < base_addr_) return false;
        auto offset = static_cast<size_t>(addr - base_addr_);
        return len <= ram_.size() && offset <= ram_.size() - len;
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * MMIO Address-Space Bus
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Central interconnect for the CPU address space.
 *
 * The @c MMIOBus acts as a router. It contains a collection of @ref Region
 * objects. For every access, it:
 * 1. Identifies the target @ref Region based on the absolute @c addr.
 * 2. Subtracts the region's @c base to create a relative offset.
 * 3. Forwards the request to the underlying @ref Memory device.
 */
class MMIOBus : public Memory {
public:
    /** Defines a mapping between an address range and a device. */
    struct Region {
        addr_t                  base;    ///< Absolute start address.
        addr_t                  size;    ///< Size of the region in bytes.
        std::shared_ptr<Memory> device;  ///< The memory-mapped device.
        std::string             name;    ///< Symbolic name (e.g., "UART", "RAM").
    };

    /** @brief Sets a fallback memory for unmapped addresses (e.g., an "Open Bus"). */
    void set_default(std::shared_ptr<Memory> mem);

    /**
     * @brief Maps a device into the bus.
     * @param base  Absolute base address.
     * @param size  Size in bytes.
     * @param device  Pointer to the Memory implementation.
     * @param name  Optional label for logging/debugging.
     */
    void map(addr_t base, addr_t size, std::shared_ptr<Memory> device,
             std::string name = "");

    /** @brief Removes a mapping at the specified base address. */
    void unmap(addr_t base);

    /* Memory interface implementations */
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

    /** @brief Returns all current mappings. */
    [[nodiscard]] const std::vector<Region>& regions() const noexcept { return regions_; }

private:
    std::vector<Region>     regions_;
    std::shared_ptr<Memory> default_mem_;

    /**
     * @brief Internal routing logic.
     * @param addr  Absolute address to look up.
     * @param[out] offset  Pointer to receive the translated relative offset.
     * @return Raw pointer to the target device, or the default device.
     */
    [[nodiscard]] Memory* find_region(addr_t addr, addr_t* offset = nullptr) const;
};

} // namespace riscv
