/**
 * @file nic.hpp
 * @brief Cycle-accurate NIC simulation modeled after the Intel 82599.
 *
 * Implements descriptor ring DMA, interrupt coalescing, and Receive Side
 * Scaling (RSS) for evaluating interrupt-driven vs. polling-based I/O.
 *
 * @par Descriptor ring semantics
 * Head pointer (RDH/TDH) is hardware-managed; tail pointer (RDT/TDT) is
 * software-managed and points one past the last valid descriptor.
 *
 * @par DMA timing model
 * Transfer latency = fixed setup cost + ceil(bytes / cacheline_size) * per-line cost.
 *
 * @see Intel 82599 10 GbE Controller Datasheet, Tables 7-11 and 7-29.
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

/// NIC MMIO register offsets - all 32-bit aligned.
namespace NicReg
{
    /// @name Control and status
    /// @{
    constexpr addr_t CTRL     = 0x00;  ///< Device control (bit 0: RST, 1: RXEN, 2: TXEN).
    constexpr addr_t STATUS   = 0x04;  ///< Device status (read-only).
    /// @}

    /// @name Interrupt management
    /// @{
    constexpr addr_t IMC      = 0x08;  ///< Interrupt mask clear.
    constexpr addr_t IMS      = 0x0C;  ///< Interrupt mask set.
    constexpr addr_t ICR      = 0x10;  ///< Interrupt cause read (read-clear).
    constexpr addr_t ITR      = 0x14;  ///< Interrupt throttling rate.
    /// @}

    /// @name RX descriptor ring
    /// @{
    constexpr addr_t RDBAL    = 0x20;  ///< RX descriptor base address (low 32).
    constexpr addr_t RDLEN    = 0x28;  ///< RX descriptor ring length (bytes).
    constexpr addr_t RDH      = 0x2C;  ///< RX descriptor head (hardware-owned)
    constexpr addr_t RDT      = 0x30;  ///< RX descriptor tail (software-owned)
    /// @}

    /// @name TX descriptor ring
    /// @{
    constexpr addr_t TDBAL    = 0x40;  ///< TX descriptor base address.
    constexpr addr_t TDLEN    = 0x48;  ///< TX descriptor ring length (bytes).
    constexpr addr_t TDH      = 0x4C;  ///< TX descriptor head.
    constexpr addr_t TDT      = 0x50;  ///< TX descriptor tail.
    /// @}

    /// @name Packet/byte counters
    /// @{
    constexpr addr_t RXPKT    = 0x60;
    constexpr addr_t TXPKT    = 0x64;
    constexpr addr_t RXBYTES  = 0x68;
    constexpr addr_t TXBYTES  = 0x6C;
    /// @}

    constexpr addr_t REG_SIZE = 0x80;
}

/// Device control register bits.
namespace NicCtrl
{
    constexpr u32 RST     = 1u << 0;  ///< Software reset (self-clearing).
    constexpr u32 RXEN    = 1u << 1;  ///< Receive enable.
    constexpr u32 TXEN    = 1u << 2;  ///< Transmit enable.
    constexpr u32 LINK_UP = 1u << 3;  ///< Physical link status (read-only).
}

namespace NicInt
{
    constexpr u32 RXQ0  = 1u << 0;  ///< RX packet received.
    constexpr u32 TXQ0  = 1u << 1;  ///< TX completed.
    constexpr u32 TIMER = 1u << 2;  ///< Coalescing timer expired.
}

/// Legacy RX descriptor (16 bytes, 82599-compatible layout).
struct RxDescriptor
{
    u32 buffer_addr;
    u32 buffer_addr_hi;
    u16 length;
    u16 checksum;
    u8  status;
    u8  errors;
    u16 vlan;

    static constexpr u8 STATUS_DD  = 0x01;  ///< Descriptor done.
    static constexpr u8 STATUS_EOP = 0x02;  ///< End of packet.
};
static_assert(sizeof(RxDescriptor) == 16);

/// Legacy TX descriptor (16 bytes, 82599-compatible layout).
struct TxDescriptor
{
    u32 buffer_addr;
    u32 buffer_addr_hi;
    u16 length;
    u8  cso;
    u8  cmd;
    u8  status;
    u8  css;
    u16 vlan;

    static constexpr u8 CMD_EOP   = 0x01;
    static constexpr u8 CMD_RS    = 0x08;
    static constexpr u8 STATUS_DD = 0x01;
};
static_assert(sizeof(TxDescriptor) == 16);

/// A packet in flight within the simulation.
struct Packet
{
    std::vector<u8> data;
    cycle_t arrival_cycle    = 0;
    cycle_t dma_start_cycle  = 0;
    cycle_t dma_done_cycle   = 0;
    cycle_t sw_receive_cycle = 0;
};

/// DMA and bus timing parameters.
struct NicTiming
{
    u32 dma_latency_cycles       = 100;  ///< Fixed DMA arbitration overhead.
    u32 dma_cycles_per_cacheline = 10;   ///< Per-cacheline transfer cost.
    u32 cacheline_size           = 64;
    u32 mmio_read_cycles         = 5;    ///< PCIe round-trip for MMIO read.
    u32 mmio_write_cycles        = 5;    ///< PCIe round-trip for MMIO write.

    /// Total cycles for a DMA transfer of @p bytes.
    [[nodiscard]] u32 dma_cycles(u32 bytes) const noexcept
    {
        u32 lines = (bytes + cacheline_size - 1) / cacheline_size;
        return dma_latency_cycles + lines * dma_cycles_per_cacheline;
    }
};

/// Interrupt coalescing configuration.
struct CoalesceConfig
{
    bool enabled          = false;
    u32  max_packets      = 1;  ///< Fire after this many packets.
    u32  max_delay_cycles = 0;  ///< Fire after this many cycles (0 = disabled).
};

/// Receive Side Scaling configuration.
struct RSSConfig
{
    bool enabled    = false;
    u32  num_queues = 1;

    std::array<u8, 128> indirection_table{};

    bool hash_ipv4 = true;
    bool hash_tcp4 = true;

    void init_round_robin(u32 queues)
    {
        num_queues = queues;
        enabled = (queues > 1);

        for (u32 i = 0; i < indirection_table.size(); ++i)
            indirection_table[i] = static_cast<u8>(i % queues);
    }
};

/// Per-queue descriptor ring state.
struct DescriptorRing
{
    addr_t base_addr = 0;
    u32    ring_size = 0;
    u32    head      = 0;
    u32    tail      = 0;

    u32     coalesce_pending = 0;
    cycle_t coalesce_first   = 0;
};

/// NIC performance counters.
struct NicStats
{
    u64 rx_packets        = 0;
    u64 tx_packets        = 0;
    u64 rx_bytes          = 0;
    u64 tx_bytes          = 0;
    u64 rx_dropped        = 0;
    u64 tx_dropped        = 0;
    u64 interrupts_raised = 0;
    u64 coalesced_packets = 0;

    u64 total_rx_latency  = 0;
    u64 min_rx_latency    = ~u64{0};
    u64 max_rx_latency    = 0;

    void reset() noexcept { *this = NicStats{}; }

    [[nodiscard]] double avg_rx_latency() const noexcept
    {
        return rx_packets > 0
            ? static_cast<double>(total_rx_latency) 
             / static_cast<double>(rx_packets)
            : 0.0;
    }
};

/**
 * @brief Cycle-accurate network interface controller.
 *
 * Handles MMIO register access, DMA transfers with system memory,
 * and interrupt coalescing. Supports multi-queue RX via RSS.
 */
class NIC : public Memory {
public:
    explicit NIC(std::shared_ptr<Memory> system_memory,
                 NicTiming timing = {});

    /// @name Memory-mapped interface
    /// @{
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /// @}

    /// @name Simulation interface
    /// @{

    /// Advance internal state by one cycle.
    void tick(cycle_t current_cycle);

    /// Inject a packet from the network medium into the RX path.
    void inject_packet(const Packet& pkt);

    /// Pull a transmitted packet from the TX completion queue.
    [[nodiscard]] std::optional<Packet> poll_tx(u32 qid = 0);
    /// @}

    /// @name Interrupt interface
    /// @{
    [[nodiscard]] bool interrupt_pending() const noexcept { return interrupt_pending_; }
    void clear_interrupt() noexcept { interrupt_pending_ = false; }

    u32 plic_source = 0;  ///< PLIC source ID (set by system builder).

    using InterruptCallback = std::function<void()>;
    void set_interrupt_callback(InterruptCallback cb) { interrupt_cb_ = std::move(cb); }
    /// @}

    /// @name Configuration
    /// @{
    void set_timing(const NicTiming& t) noexcept { timing_ = t; }
    [[nodiscard]] const NicTiming& timing() const noexcept { return timing_; }

    void set_coalescing(const CoalesceConfig& cfg) noexcept { coalesce_ = cfg; }
    [[nodiscard]] const CoalesceConfig& coalescing() const noexcept { return coalesce_; }

    void set_rss(const RSSConfig& cfg);
    [[nodiscard]] const RSSConfig& rss() const noexcept { return rss_; }

    void configure_rx_queue(u32 qid, addr_t base, u32 ring_size, u32 tail);
    void configure_tx_queue(u32 qid, addr_t base, u32 ring_size);

    static constexpr u32 MAX_QUEUES = 8;
    /// @}

    /// @name Statistics
    /// @{
    [[nodiscard]] const NicStats& stats() const noexcept { return stats_; }
    /// @}

    /// @name State inspection
    /// @{
    [[nodiscard]] u32 reg(addr_t offset) const noexcept;
    [[nodiscard]] bool rx_enabled()   const noexcept { return regs_[NicReg::CTRL / 4] & NicCtrl::RXEN; }
    [[nodiscard]] bool tx_enabled()   const noexcept { return regs_[NicReg::CTRL / 4] & NicCtrl::TXEN; }
    [[nodiscard]] u32  rx_ring_size() const noexcept { return regs_[NicReg::RDLEN / 4] / 16; }
    [[nodiscard]] u32  tx_ring_size() const noexcept { return regs_[NicReg::TDLEN / 4] / 16; }
    void set_trace(bool enable) { trace_ = enable; }
    /// @}

    void reset();

private:
    std::shared_ptr<Memory> sys_mem_;
    NicTiming      timing_;
    CoalesceConfig coalesce_;
    RSSConfig      rss_;
    NicStats       stats_;

    bool trace_ = false;

    mutable std::array<u32, NicReg::REG_SIZE / 4> regs_{};

    u32  interrupt_mask_    = 0;
    bool interrupt_pending_ = false;
    InterruptCallback interrupt_cb_;

    std::array<DescriptorRing, MAX_QUEUES> rx_rings_{};
    std::array<DescriptorRing, MAX_QUEUES> tx_rings_{};

    struct PendingDma
    {
        enum class Type { RX, TX } type;
        u32     queue_id;
        u32     desc_idx;
        Packet  packet;
        cycle_t complete_cycle;
    };
    std::deque<PendingDma> pending_dma_;

    std::deque<Packet>                         rx_queue_;
    std::array<std::deque<Packet>, MAX_QUEUES> tx_complete_;

    cycle_t current_cycle_ = 0;

    /// @name Internal helpers
    /// @{
    void raise_interrupt(u32 cause);
    void check_coalescing(u32 qid, u32 cause);
    void process_rx_queue();
    void process_tx_ring(u32 qid);
    void complete_dma();

    [[nodiscard]] u32 compute_rss_hash(const Packet& pkt) const;
    [[nodiscard]] u32 select_rx_queue(const Packet& pkt)  const;

    RxDescriptor read_rx_desc(u32 qid, u32 idx) const;
    TxDescriptor read_tx_desc(u32 qid, u32 idx) const;
    void write_rx_desc(u32 qid, u32 idx, const RxDescriptor& desc);
    void write_tx_desc(u32 qid, u32 idx, const TxDescriptor& desc);
    /// @}
};

/// Loopback adapter: routes TX packets back to RX after a configurable delay.
class NicLoopback {
public:
    explicit NicLoopback(NIC& nic, u32 delay_cycles = 50)
        : nic_(nic), delay_(delay_cycles) {}

    void tick(cycle_t current_cycle);

private:
    NIC& nic_;
    u32  delay_;

    struct DelayedPacket
    {
        Packet  pkt;
        cycle_t deliver_cycle;
    };
    std::deque<DelayedPacket> pending_;
};

} // namespace riscv
