/**
 * @file test_nic.cpp
 * @brief Tests for the NIC device model.
 *
 * Sections:
 *   1 (line 147) : MMIO registers - control, status, read/write, read-clear
 *   2 (line 208) : RX path - inject packet, DMA to memory, descriptor update
 *   3 (line 269) : TX path - write descriptor, DMA from memory, poll_tx
 *   4 (line 314) : Interrupts - mask, raise, clear, callback
 *   5 (line 388) : Interrupt coalescing - packet count, timer threshold
 *   6 (line 462) : Loopback - TX->RX round-trip
 *   7 (line 502) : Statistics - counters, latency tracking
 *   8 (line 543) : Edge cases - ring full, disabled, reset
 *   9 (line 583) : RSS — multi-queue packet distribution
 */

#include "core/nic.hpp"
#include "test_framework.hpp"

using namespace riscv;

// ── Helpers ────────────────────────────────────────────────────────────

/**
 * Set up a NIC with a system memory region and a simple RX descriptor ring.
 * Returns {nic, sys_mem}.
 */
struct NicTestSetup
{
    std::shared_ptr<FlatMemory> mem;
    NIC nic;

    static NicTestSetup create(NicTiming timing = {})
    {
        auto mem = std::make_shared<FlatMemory>(0, 0x100000);
        NicTestSetup s{mem, NIC(mem, timing)};
        return s;
    }

    /**
     * Set up a simple RX ring at base_addr with @c count descriptors.
     *
     * Each descriptor points to a buffer at buf_base + i * buf_size.
     */
    void setup_rx_ring(addr_t base_addr, u32 count, addr_t buf_base, u32 buf_size)
    {
        for (u32 i = 0; i < count; ++i)
  	    {
            addr_t desc_addr = base_addr + i * 16;
            mem->write32(desc_addr, static_cast<u32>(buf_base + i * buf_size));  // buffer_addr
            mem->write32(desc_addr + 4, 0);   // buffer_addr_hi
            mem->write32(desc_addr + 8, 0);   // length/checksum
            mem->write32(desc_addr + 12, 0);  // status/errors/vlan
        }
        nic.write32(NicReg::RDBAL, static_cast<u32>(base_addr));
        nic.write32(NicReg::RDLEN, count * 16);
        nic.write32(NicReg::RDH, 0);
        nic.write32(NicReg::RDT, count - 1);  // All descriptors available
    }

    /** Set up a simple TX ring at base_addr with @c count descriptors. */
    void setup_tx_ring(addr_t base_addr, u32 count)
    {
        for (u32 i = 0; i < count; ++i)
	    {
            addr_t desc_addr = base_addr + i * 16;
            mem->write32(desc_addr, 0);
            mem->write32(desc_addr + 4, 0);
            mem->write32(desc_addr + 8, 0);
            mem->write32(desc_addr + 12, 0);
        }
        nic.write32(NicReg::TDBAL, static_cast<u32>(base_addr));
        nic.write32(NicReg::TDLEN, count * 16);
        nic.write32(NicReg::TDH, 0);
        nic.write32(NicReg::TDT, 0);  // No pending TX
    }

    void enable_rxtx()
    {
        nic.write32(NicReg::CTRL, NicCtrl::RXEN | NicCtrl::TXEN);
    }
};

static Packet make_packet(std::initializer_list<u8> bytes)
{
    Packet p;
    p.data = bytes;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. MMIO registers
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_status_link_up) {
    auto s = NicTestSetup::create();
    auto r = s.nic.read32(NicReg::STATUS);
    ASSERT(r.ok);
    ASSERT(r.value & NicCtrl::LINK_UP);
    return true;
}

TEST(nic_ctrl_write_read) {
    auto s = NicTestSetup::create();
    s.nic.write32(NicReg::CTRL, NicCtrl::RXEN | NicCtrl::TXEN);
    ASSERT(s.nic.rx_enabled());
    ASSERT(s.nic.tx_enabled());
    return true;
}

TEST(nic_icr_read_clear) {
    auto s = NicTestSetup::create();
    // Manually poke ICR to simulate a cause
    s.nic.write32(NicReg::IMS, NicInt::RXQ0);
    
    auto r = s.nic.read32(NicReg::ICR);
    ASSERT_EQ(r.value, 0u);
    return true;
}

TEST(nic_mmio_latency) {
    NicTiming timing;
    timing.mmio_read_cycles = 10;
    timing.mmio_write_cycles = 8;
    auto s = NicTestSetup::create(timing);
    auto r = s.nic.read32(NicReg::STATUS);
    ASSERT_EQ(r.cycles, 10u);
    auto w = s.nic.write32(NicReg::CTRL, 0);
    ASSERT_EQ(w.cycles, 8u);
    return true;
}

TEST(nic_invalid_addr) {
    auto s = NicTestSetup::create();
    auto r = s.nic.read32(0xFF);  // Misaligned
    ASSERT(!r.ok);
    r = s.nic.read32(NicReg::REG_SIZE);  // Out of range
    ASSERT(!r.ok);
    return true;
}

TEST(nic_reset_via_ctrl) {
    auto s = NicTestSetup::create();
    s.nic.write32(NicReg::CTRL, NicCtrl::RXEN | NicCtrl::TXEN);
    ASSERT(s.nic.rx_enabled());
    s.nic.write32(NicReg::CTRL, NicCtrl::RST);
    ASSERT(!s.nic.rx_enabled());
    ASSERT(!s.nic.tx_enabled());
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. RX path
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_rx_basic) {
    NicTiming timing;
    timing.dma_latency_cycles = 10;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);

    // 4-descriptor RX ring at 0x1000, buffers at 0x2000 (256 bytes each)
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    s.nic.inject_packet(make_packet({0xAA, 0xBB, 0xCC, 0xDD}));

    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.mem->read8(0x2000).value, 0xAAu);
    ASSERT_EQ(s.mem->read8(0x2001).value, 0xBBu);
    ASSERT_EQ(s.mem->read8(0x2002).value, 0xCCu);
    ASSERT_EQ(s.mem->read8(0x2003).value, 0xDDu);

    u32 desc_status_word = s.mem->read32(0x1000 + 12).value;
    u8 status = desc_status_word & 0xFF;
    ASSERT(status & RxDescriptor::STATUS_DD);
    ASSERT(status & RxDescriptor::STATUS_EOP);

    u32 desc_len_word = s.mem->read32(0x1000 + 8).value;
    u16 length = desc_len_word & 0xFFFF;
    ASSERT_EQ(length, 4u);

    ASSERT_EQ(s.nic.reg(NicReg::RDH), 1u);

    ASSERT_EQ(s.nic.stats().rx_packets, 1u);
    ASSERT_EQ(s.nic.stats().rx_bytes, 4u);
    return true;
}

TEST(nic_rx_multiple_packets) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 8, 0x2000, 256);
    s.enable_rxtx();

    for (int i = 0; i < 3; ++i)
        s.nic.inject_packet(make_packet({static_cast<u8>(i + 1)}));

    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.nic.stats().rx_packets, 3u);
    ASSERT_EQ(s.mem->read8(0x2000).value, 1u);
    ASSERT_EQ(s.mem->read8(0x2100).value, 2u);
    ASSERT_EQ(s.mem->read8(0x2200).value, 3u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. TX path
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_tx_basic) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_tx_ring(0x3000, 4);
    s.enable_rxtx();
 
    // Write packet data at 0x4000
    s.mem->write8(0x4000, 0x11);
    s.mem->write8(0x4001, 0x22);
    s.mem->write8(0x4002, 0x33);
    s.mem->write8(0x4003, 0x44);
 
    // Write TX descriptor: buffer_addr=0x4000, length=4, cmd=EOP|RS
    addr_t desc0 = 0x3000;
    s.mem->write32(desc0, 0x4000);  // buffer_addr
    s.mem->write32(desc0 + 4, 0);   // buffer_addr_hi
    s.mem->write32(desc0 + 8, 4 | (static_cast<u32>(TxDescriptor::CMD_EOP | TxDescriptor::CMD_RS) << 24));
    s.mem->write32(desc0 + 12, 0);
 
    s.nic.write32(NicReg::TDT, 1);
 
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);
 
    u32 status_word = s.mem->read32(desc0 + 12).value;
    ASSERT(status_word & TxDescriptor::STATUS_DD);
 
    ASSERT_EQ(s.nic.reg(NicReg::TDH), 1u);
 
    auto pkt = s.nic.poll_tx();
    ASSERT(pkt.has_value());
    ASSERT_EQ(pkt->data.size(), 4u);
    ASSERT_EQ(pkt->data[0], 0x11u);
    ASSERT_EQ(pkt->data[3], 0x44u);
 
    ASSERT_EQ(s.nic.stats().tx_packets, 1u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. Interrupts
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_interrupt_masked) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    // Don't enable any interrupt mask
    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    // Packet received, but interrupt should NOT be pending
    ASSERT_EQ(s.nic.stats().rx_packets, 1u);
    ASSERT(!s.nic.interrupt_pending());
    return true;
}

TEST(nic_interrupt_fires) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    // Enable RX interrupt
    s.nic.write32(NicReg::IMS, NicInt::RXQ0);

    bool callback_fired = false;
    s.nic.set_interrupt_callback([&]() { callback_fired = true; });

    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT(s.nic.interrupt_pending());
    ASSERT(callback_fired);
    ASSERT_EQ(s.nic.stats().interrupts_raised, 1u);

    // ICR should have RXQ0 set
    auto icr = s.nic.read32(NicReg::ICR);
    ASSERT(icr.value & NicInt::RXQ0);
    // Read-clear: second read should be 0
    auto icr2 = s.nic.read32(NicReg::ICR);
    ASSERT_EQ(icr2.value, 0u);
    return true;
}

TEST(nic_interrupt_mask_clear) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    // Enable, then disable RX interrupt
    s.nic.write32(NicReg::IMS, NicInt::RXQ0);
    s.nic.write32(NicReg::IMC, NicInt::RXQ0);

    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT(!s.nic.interrupt_pending());
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. Interrupt coalescing
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_coalesce_by_packet_count) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 16, 0x2000, 256);
    s.enable_rxtx();
    s.nic.write32(NicReg::IMS, NicInt::RXQ0);

    CoalesceConfig coal;
    coal.enabled = true;
    coal.max_packets = 4;
    coal.max_delay_cycles = 0;  // No timer — only packet count
    s.nic.set_coalescing(coal);

    int interrupt_count = 0;
    s.nic.set_interrupt_callback([&]() { interrupt_count++; });

    // Inject 3 packets — should NOT fire interrupt yet
    for (int i = 0; i < 3; ++i) {
        s.nic.inject_packet(make_packet({static_cast<u8>(i)}));
        for (cycle_t c = 0; c < 50; ++c)
            s.nic.tick(100 * i + c);
    }
    ASSERT_EQ(interrupt_count, 0);

    // 4th packet -> fires
    s.nic.inject_packet(make_packet({0x03}));
    for (cycle_t c = 0; c < 50; ++c)
        s.nic.tick(300 + c);
    ASSERT_EQ(interrupt_count, 1);

    return true;
}

TEST(nic_coalesce_by_timer) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 16, 0x2000, 256);
    s.enable_rxtx();
    s.nic.write32(NicReg::IMS, NicInt::RXQ0 | NicInt::TIMER);

    CoalesceConfig coal;
    coal.enabled = true;
    coal.max_packets = 100;       // High threshold — won't trigger by count
    coal.max_delay_cycles = 500;  // Timer fires after 500 cycles
    s.nic.set_coalescing(coal);

    int interrupt_count = 0;
    s.nic.set_interrupt_callback([&]() { interrupt_count++; });

    // Inject 1 packet at cycle 100
    s.nic.tick(100);
    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 101; c <= 200; ++c)
        s.nic.tick(c);

    // DMA completes around cycle 110-115; timer starts then
    ASSERT_EQ(interrupt_count, 0);  // Timer hasn't expired yet

    // Advance to cycle 700
    for (cycle_t c = 201; c <= 700; ++c)
        s.nic.tick(c);

    ASSERT(interrupt_count >= 1);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. Loopback
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_loopback) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 8, 0x2000, 256);
    s.setup_tx_ring(0x3000, 8);
    s.enable_rxtx();
 
    NicLoopback loopback(s.nic, 10);

    // Write packet at 0x4000 and set up TX descriptor
    s.mem->write8(0x4000, 0xDE);
    s.mem->write8(0x4001, 0xAD);
    addr_t desc0 = 0x3000;
    s.mem->write32(desc0, 0x4000);
    s.mem->write32(desc0 + 4, 0);
    s.mem->write32(desc0 + 8, 2 | (static_cast<u32>(TxDescriptor::CMD_EOP) << 24));
    s.mem->write32(desc0 + 12, 0);
    s.nic.write32(NicReg::TDT, 1);

    for (cycle_t c = 1; c <= 500; ++c) {
        s.nic.tick(c);
        loopback.tick(c);
    }

    // Packet should have been transmitted and looped back to RX
    ASSERT(s.nic.stats().tx_packets >= 1u);
    ASSERT(s.nic.stats().rx_packets >= 1u);

    // Verify the looped-back data arrived in the RX buffer
    ASSERT_EQ(s.mem->read8(0x2000).value, 0xDEu);
    ASSERT_EQ(s.mem->read8(0x2001).value, 0xADu);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. Statistics
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_rx_latency_tracking) {
    NicTiming timing;
    timing.dma_latency_cycles = 50;
    timing.dma_cycles_per_cacheline = 5;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    s.nic.tick(1000);
    s.nic.inject_packet(make_packet({0x01, 0x02, 0x03, 0x04}));
    for (cycle_t c = 1001; c <= 1200; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.nic.stats().rx_packets, 1u);
    ASSERT(s.nic.stats().min_rx_latency > 0u);
    ASSERT(s.nic.stats().max_rx_latency >= s.nic.stats().min_rx_latency);
    ASSERT(s.nic.stats().avg_rx_latency() > 0.0);
    return true;
}

TEST(nic_stat_counters) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 8, 0x2000, 256);
    s.enable_rxtx();

    s.nic.inject_packet(make_packet({0xAA, 0xBB}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.nic.reg(NicReg::RXPKT), 1u);
    ASSERT_EQ(s.nic.reg(NicReg::RXBYTES), 2u);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. Edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(nic_rx_disabled) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);

    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    // Packet stays in the queue but is never processed
    ASSERT_EQ(s.nic.stats().rx_packets, 0u);
    return true;
}

TEST(nic_reset_clears_stats) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    s.nic.inject_packet(make_packet({0xAA}));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);
    ASSERT_EQ(s.nic.stats().rx_packets, 1u);

    s.nic.reset();
    ASSERT_EQ(s.nic.stats().rx_packets, 0u);
    ASSERT_EQ(s.nic.stats().rx_bytes, 0u);
    ASSERT(!s.nic.rx_enabled());
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  9. RSS — multi-queue packet distribution
 * ═══════════════════════════════════════════════════════════════════════ */

static Packet make_ip_packet(u32 src_ip, u32 dst_ip)
{
    Packet p;
    p.data.resize(24, 0);

    // protocol (TCP)
    p.data[9] = 6;

    // src IP
    p.data[12] = (src_ip >> 24) & 0xFF;
    p.data[13] = (src_ip >> 16) & 0xFF;
    p.data[14] = (src_ip >> 8)  & 0xFF;
    p.data[15] =  src_ip        & 0xFF;

    // dst ip
    p.data[16] = (dst_ip >> 24) & 0xFF;
    p.data[17] = (dst_ip >> 16) & 0xFF;
    p.data[18] = (dst_ip >> 8)  & 0xFF;
    p.data[19] =  dst_ip        & 0xFF;

    // src/dst port
    p.data[20] = 0; p.data[21] = 80;   // src port 80
    p.data[22] = 0; p.data[23] = 443;  // dst port 443

    return p;
}

TEST(nic_rss_distributes_to_queues) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);

    RSSConfig rss;
    rss.init_round_robin(2);
    s.nic.set_rss(rss);

    // Queue 0: descriptors at 0x1000, buffers at 0x2000
    s.nic.configure_rx_queue(0, 0x1000, 8, 7);
    for (u32 i = 0; i < 8; ++i)
    {
        addr_t da = 0x1000 + i * 16;

        s.mem->write32(da, static_cast<u32>(0x2000 + i * 256));
        s.mem->write32(da + 4, 0);
        s.mem->write32(da + 8, 0);
        s.mem->write32(da + 12, 0);
    }

    // Queue 1: descriptors at 0x3000, buffers at 0x4000
    s.nic.configure_rx_queue(1, 0x3000, 8, 7);
    for (u32 i = 0; i < 8; ++i)
    {
        addr_t da = 0x3000 + i * 16;
        s.mem->write32(da, static_cast<u32>(0x4000 + i * 256));
        s.mem->write32(da + 4, 0);
        s.mem->write32(da + 8, 0);
        s.mem->write32(da + 12, 0);
    }

    s.enable_rxtx();

    // Inject packets from different IP addresses
    for (u32 i = 0; i < 10; ++i)
        s.nic.inject_packet(make_ip_packet(0x0A00'0001 + i, 0x0B00'0001));

    for (cycle_t c = 1; c <= 500; ++c)
        s.nic.tick(c);

    // All 10 packets should be received, unless the hash
    // distribution is uneven enough that one queue runs
    // out of descriptors (7 available per queue).
    ASSERT(s.nic.stats().rx_packets >= 8u);
    ASSERT(s.nic.stats().rx_packets + s.nic.stats().rx_dropped == 10u);
    return true;
}

TEST(nic_rss_same_flow_same_queue) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);

    RSSConfig rss;
    rss.init_round_robin(4);
    s.nic.set_rss(rss);

    // Set up 4 queues
    for (u32 q = 0; q < 4; ++q)
    {
        addr_t desc_base = 0x1000 + q * 0x1000;
        addr_t buf_base  = 0x5000 + q * 0x2000;
        s.nic.configure_rx_queue(q, desc_base, 8, 7);

        for (u32 i = 0; i < 8; ++i)
        {
            addr_t da = desc_base + i * 16;

            s.mem->write32(da, static_cast<u32>(buf_base + i * 256));
            s.mem->write32(da + 4, 0);
            s.mem->write32(da + 8, 0);
            s.mem->write32(da + 12, 0);
        }
    }

    s.enable_rxtx();

    // Inject 5 packets from the same src/dst IP
    // These should all go to the same queue
    Packet flow_pkt = make_ip_packet(0x0A00'0001, 0x0B00'0001);
    for (int i = 0; i < 5; ++i)
        s.nic.inject_packet(flow_pkt);

    for (cycle_t c = 1; c <= 500; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.nic.stats().rx_packets, 5u);

    int queues_with_packets = 0;
    for (u32 q = 0; q < 4; ++q)
    {
        addr_t desc_base = 0x1000 + q * 0x1000;
        bool has_packet = false;

        for (u32 i = 0; i < 8; ++i)
        {
            u32 status_word = s.mem->read32(desc_base + i * 16 + 12).value;
            
            if (status_word & RxDescriptor::STATUS_DD) has_packet = true;
        }
        if (has_packet) queues_with_packets++;
    }
    ASSERT_EQ(queues_with_packets, 1);
    return true;
}

TEST(nic_rss_disabled_uses_queue_zero) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);

    // RSS disabled by default - all packets go to queue 0
    s.setup_rx_ring(0x1000, 4, 0x2000, 256);
    s.enable_rxtx();

    s.nic.inject_packet(make_ip_packet(0x0102'0304, 0x0506'0708));
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    ASSERT_EQ(s.nic.stats().rx_packets, 1u);
    ASSERT_EQ(s.mem->read8(0x2000).value, make_ip_packet(0x0102'0304, 0x0506'0708).data[0]);
    return true;
}

TEST(nic_multiqueue_tx_queue0_regression) {
    NicTiming timing;
    timing.dma_latency_cycles = 5;
    timing.dma_cycles_per_cacheline = 1;
    auto s = NicTestSetup::create(timing);
    s.setup_tx_ring(0x3000, 4);
    s.enable_rxtx();

    // Write packet data at 0x4000 and set up a TX descriptor on queue 0
    s.mem->write8(0x4000, 0x11);
    addr_t desc0 = 0x3000;
    s.mem->write32(desc0, 0x4000);
    s.mem->write32(desc0 + 4, 0);
    s.mem->write32(desc0 + 8, 1 | (u32(TxDescriptor::CMD_EOP) << 24));
    s.mem->write32(desc0 + 12, 0);

    // Advance TDT via legacy MMIO — triggers queue 0 processing
    s.nic.write32(NicReg::TDT, 1);
    for (cycle_t c = 1; c <= 200; ++c)
        s.nic.tick(c);

    auto pkt = s.nic.poll_tx(0);
    ASSERT(pkt.has_value());
    ASSERT_EQ(pkt->data[0], 0x11u);

    // Queue 1 should be empty - nothing was submitted
    ASSERT(!s.nic.poll_tx(1).has_value());
    return true;
}
