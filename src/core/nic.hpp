/**
 * @file nic.hpp
 * @brief Cycle-accurate NIC simulation model.
 *
 * This module implements a simplified version of the Intel 82599
 * controller architecture. It is designed for evaluating the overhead
 * of interrupt-driven vs. polling-based I/O.
 *
 * @section descriptors Descriptor Ring Semantics
 * The NIC uses a circular ring of 16-byte descriptors.
 * - **Head Pointer (RDH/TDH)**: Managed by hardware; points to the next
 * descriptor the NIC will process.
 * - **Tail Pointer (RDT/TDT)**: Managed by software; points to one beyond
 * the last valid descriptor posted by the driver.
 *
 * @section dma DMA and Timing Model
 * DMA transfers are not instantaneous - the model calculates latency as:
 * $$T_{dma} = L_{fixed} + \lceil \frac{Bytes}{64} \rceil \times L_{burst}$$
 * where $L_{fixed}$ is the setup overhead and $L_{burst}$ is the per-cacheline 
 * transfer cost over the simulated bus.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"
 
#include <array>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace riscv {

/**
 * @brief Register Map Offsets.
 *
 * All registers are 32-bit aligned and should be accessed via 32-bit loads/stores.
 */
namespace NicReg {
    /** @name Control and Status */
    /** @{ */
    constexpr addr_t CTRL     = 0x00;  ///< Device Control (Bit 0: RST, 1: RXEN, 2: TXEN)
    constexpr addr_t STATUS   = 0x04;  ///< Device Status (Read-Only)
    /** @} */

    /** @name Interrupt Management */
    /** @{ */
    constexpr addr_t IMC      = 0x08;  ///< Interrupt Mask Clear (Write bits to disable)
    constexpr addr_t IMS      = 0x0C;  ///< Interrupt Mask Set (Write bits to enable)
    constexpr addr_t ICR      = 0x10;  ///< Interrupt Cause Read (Read-Clear)
    constexpr addr_t ITR      = 0x14;  ///< Interrupt Throttling Rate (Coalescing)
    /** @} */
    
    /** @name Receive (RX) Descriptor */
    /** @{ */
    constexpr addr_t RDBAL    = 0x20;  ///< RX Descriptor Base Address (Low 32 bits)
    constexpr addr_t RDLEN    = 0x28;  ///< RX Descriptor Ring Length (in bytes)
    constexpr addr_t RDH      = 0x2C;  ///< RX Descriptor Head (Hardware owned)
    constexpr addr_t RDT      = 0x30;  ///< RX Descriptor Tail (Software owned)
    /** @} */
    
    /** @name Transmit (TX) Descriptor */
    /** @{ */
    constexpr addr_t TDBAL    = 0x40;  ///< TX Descriptor Base Address
    constexpr addr_t TDLEN    = 0x48;  ///< TX Descriptor Ring Length
    constexpr addr_t TDH      = 0x4C;  ///< TX Descriptor Head
    constexpr addr_t TDT      = 0x50;  ///< TX Descriptor Tail
    /** @} */

    /** @name Packet/Byte Counters */
    /** @{ */
    constexpr addr_t RXPKT    = 0x60;  ///< RX packet counter
    constexpr addr_t TXPKT    = 0x64;  ///< TX packet counter
    constexpr addr_t RXBYTES  = 0x68;  ///< RX byte counter
    constexpr addr_t TXBYTES  = 0x6C;  ///< TX byte counter
    /** @} */

    constexpr addr_t REG_SIZE = 0x80;
}

/**
 * @brief Bits for the Device Control Register (CTRL).
 *
 * These bits govern the global state machine of the NIC.
 */
namespace NicCtrl {
    /**
     * @brief Software Reset (Self-Clearing).
     * Writing a 1 to this bit resets the internal state machines,
     * clears the RX/TX FIFOs, and resets Head/Tail pointers to 0.
     */
    constexpr u32 RST     = 1u << 0;

    /**
     * @brief Receive Enable.
     * When cleared, the NIC drops all incoming packets at the wire.
     * When set, the DMA engine begins polling the RX descriptor ring.
     */
    constexpr u32 RXEN    = 1u << 1;

    /**
     * @brief Transmit Enable.
     * Enables the TX DMA engine to process descriptors between TDH and TDT.
     */
    constexpr u32 TXEN    = 1u << 2;

    /**
     * @brief Physical Link Status (Read-Only).
     * Indicates whether the virtual "cable" is connected.
     */
    constexpr u32 LINK_UP = 1u << 3;
}

namespace NicInt {
    constexpr u32 RXQ0  = 1u << 0;  ///< RX packet received
    constexpr u32 TXQ0  = 1u << 1;  ///< TX completed
    constexpr u32 TIMER = 1u << 2;  ///< Coalescing timer expired
}

/**
 * @brief RX descriptor format (16 bytes).
 *
 * Mapping based on Intel 82599 Legacy Receive Descriptor Layout.
 *
 * @see Intel 82599 10 Gigabit Ethernet Controller Datasheet, Table 7-11.
 */
struct RxDescriptor {
    u32 buffer_addr;     ///< Physical address of the data buffer.
    u32 buffer_addr_hi;  ///< High 32 bits (unused in RV32).
    u16 length;          ///< Bytes written by hardware.
    u16 checksum;        ///< Receive checksum offload result.
    u8  status;          ///< Descriptor status (Bit 0: Descriptor Done/DD).
    u8  errors;          ///< Error indications.
    u16 vlan;            ///< VLAN tag if present.

    static constexpr u8 STATUS_DD  = 0x01;  ///< Descriptor Done: NIC has finished writing data.
    static constexpr u8 STATUS_EOP = 0x02;  ///< End of packet.
};
static_assert(sizeof(RxDescriptor) == 16);

/**
 * @brief TX descriptor format (16 bytes)
 *
 * Mapping based on Intel 82599 Legacy Transmit Descriptor Layout.
 *
 * @see Intel 82599 10 Gigabit Ethernet Controller Datasheet, Table 7-29.
 */
struct TxDescriptor {
    u32 buffer_addr;     ///< Physical address of the data buffer.
    u32 buffer_addr_hi;  ///< High 32 bits.
    u16 length;          ///< Length of the data buffer (in bytes).
    u8  cso;             ///< Offset in bytes to begin checksum insertion.
    u8  cmd;             ///< Written by software to control how NIC handles this descriptor.
    u8  status;          ///< Descriptor Status
    u8  css;             ///< Offset in bytes to the start of the checksum
    u16 vlan;            ///< VLAN tag if present

    static constexpr u8 CMD_EOP   = 0x01;
    static constexpr u8 CMD_RS    = 0x08;
    static constexpr u8 STATUS_DD = 0x01;
};
static_assert(sizeof(TxDescriptor) == 16);

/**
 * @brief Representation of a packet in flight within the simulation.
 */
struct Packet {
    std::vector<u8> data;          ///< Raw Ethernet frame data.
    cycle_t arrival_cycle    = 0;  ///< Cycle when the first bit hit the wire.
    cycle_t dma_start_cycle  = 0;  ///< Cycle when DMA transfer initiated.
    cycle_t dma_done_cycle   = 0;  ///< Cycle when DMA transfer completed.
    cycle_t sw_receive_cycle = 0;
};

/**
 * @brief Configuration for DMA and bus latency.
 */
struct NicTiming {
    u32 dma_latency_cycles       = 100;  ///< Fixed overhead for DMA arbitration.
    u32 dma_cycles_per_cacheline = 10;   ///< Per-cacheline transfer cost.
    u32 cacheline_size           = 64;
    u32 mmio_read_cycles         = 5;    ///< MMIO register read latency (PCIe round-trip).
    u32 mmio_write_cycles        = 5;    ///< MMIO register write latency.

    /** @brief Calculate total cycles required for a DMA transfer of @p bytes. */
    [[nodiscard]] u32 dma_cycles(u32 bytes) const noexcept
    {
        u32 lines = (bytes + cacheline_size - 1) / cacheline_size;

        return dma_latency_cycles + lines * dma_cycles_per_cacheline;
    }
};

/**
 * @brief Configuration for interrupt coalescing.
 */
struct CoalesceConfig {
    bool enabled          = false;
    u32  max_packets      = 1;  ///< Fire interrupt after this many packets.
    u32  max_delay_cycles = 0;  ///< Fire interrupt after this many cycles (0 = no timer).
};

/**
 * @brief NIC statistics.
 */
struct NicStats {
    u64 rx_packets        = 0;
    u64 tx_packets        = 0;
    u64 rx_bytes          = 0;
    u64 tx_bytes          = 0;
    u64 rx_dropped        = 0;
    u64 tx_dropped        = 0;
    u64 interrupts_raised = 0;
    u64 coalesced_packets = 0;  ///< Packets batched into coalesced interrupts.

    u64 total_rx_latency  = 0;
    u64 min_rx_latency    = ~u64{0};
    u64 max_rx_latency    = 0;

    void reset() noexcept { *this = NicStats{}; }

    [[nodiscard]] double avg_rx_latency() const noexcept
    {
        return rx_packets > 0
            ? static_cast<double>(total_rx_latency) / static_cast<double>(rx_packets)
            : 0.0;
    }
};

/**
 * @brief Cycle-accurate Network Interface Controller.
 *
 * Handles MMIO register access, coordinates DMA transfers with
 * system memory, and manages interrupt coalescing logic.
 */
class NIC : public Memory {
public:
    explicit NIC(std::shared_ptr<Memory> system_memory,
                 NicTiming timing = {});

    /** @name MMIO Interface */
    /** @{ */
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /** @} */

    /** @name Simulation interface */
    /** @{ */
    
    /**
     * @brief Steps the NIC internal state machine by one cycle.
     *
     * Manages pending DMAs, interrupt timers, and ring processing.
     */
    void tick(cycle_t current_cycle);

    /** @brief Simulates a packet arriving from the network medium. */
    void inject_packet(const Packet& pkt);

    /** @brief Pulls a transmitted packet from the TX FIFO. */
    [[nodiscard]] std::optional<Packet> poll_tx();
    /** @} */

    /** @name Interrupt Interface */
    /** @{ */

    [[nodiscard]] bool interrupt_pending() const noexcept { return interrupt_pending_; }
    void clear_interrupt() noexcept { interrupt_pending_ = false; }

    /** @brief PLIC interrupt source ID (set by the system builder). */
    u32 plic_source = 0;

    using InterruptCallback = std::function<void()>;
    
    /** @brief Binds the NIC interrupt line to a system-level handler (e.g., PLIC). */
    void set_interrupt_callback(InterruptCallback cb) { interrupt_cb_ = std::move(cb); }
    /** @} */

    /** @name Configuration */
    /** @{ */

    [[nodiscard]] const NicTiming& timing() const noexcept { return timing_; }
    void set_timing(const NicTiming& t) noexcept { timing_ = t; }

    void set_coalescing(const CoalesceConfig& cfg) noexcept { coalesce_ = cfg; }
    [[nodiscard]] const CoalesceConfig& coalescing() const noexcept { return coalesce_; }
    /** @} */

    /** @name Statistics */
    /** @{ */
    [[nodiscard]] const NicStats& stats() const noexcept { return stats_; }
    /** @} */

    /** @name State inspection */
    /** @{ */
    [[nodiscard]] u32 reg(addr_t offset) const noexcept;
    [[nodiscard]] bool rx_enabled() const noexcept { return regs_[NicReg::CTRL / 4] & NicCtrl::RXEN; }
    [[nodiscard]] bool tx_enabled() const noexcept { return regs_[NicReg::CTRL / 4] & NicCtrl::TXEN; }
    [[nodiscard]] u32 rx_ring_size() const noexcept { return regs_[NicReg::RDLEN / 4] / 16; }
    [[nodiscard]] u32 tx_ring_size() const noexcept { return regs_[NicReg::TDLEN / 4] / 16; }

    void reset();

private:
    std::shared_ptr<Memory> sys_mem_;
    NicTiming      timing_;
    CoalesceConfig coalesce_;
    NicStats       stats_;

    mutable std::array<u32, NicReg::REG_SIZE / 4> regs_{};

    /** @name Interrupt state */
    /** @{ */
    u32 interrupt_mask_     = 0;
    bool interrupt_pending_ = false;
    InterruptCallback interrupt_cb_;
    /** @} */

    /** @name Coalescing state */
    /** @{ */
    u32     coalesce_pending_pkts_ = 0;
    cycle_t coalesce_first_cycle_  = 0;
    /** @} */

    /** @name Pending DMA operations */
    /** @{ */
    struct PendingDma {
        enum class Type { RX, TX } type;
        u32     desc_idx;
        Packet  packet;
        cycle_t complete_cycle;
    };
    std::deque<PendingDma> pending_dma_;
    /** @} */

    /** @name Queues */
    /** @{ */
    std::deque<Packet> rx_queue_;
    std::deque<Packet> tx_complete_;
    /** @} */

    cycle_t current_cycle_ = 0;

    /** @name Internal helpers */
    /** @{ */

    void raise_interrupt(u32 cause);
    void check_coalescing(u32 cause);
    void process_rx_queue();           ///< Processes descriptors when RDT is updated.
    void process_tx_ring();            ///< Processes descriptors when TDT is updated.
    void complete_dma();

    RxDescriptor read_rx_desc(u32 idx) const;
    void write_rx_desc(u32 idx, const RxDescriptor& desc);
    TxDescriptor read_tx_desc(u32 idx) const;
    void write_tx_desc(u32 idx, const TxDescriptor& desc);
    /** @} */
};

/**
 * @brief Loopback helper
 */
class NicLoopback {
public:
    explicit NicLoopback(NIC& nic, u32 delay_cycles = 50)
        : nic_(nic), delay_(delay_cycles) {}

    void tick(cycle_t current_cycle);

private:
    NIC& nic_;
    u32  delay_;

    struct DelayedPacket {
        Packet  pkt;
        cycle_t deliver_cycle;
    };
    std::deque<DelayedPacket> pending_;
};

} // namespace riscv
