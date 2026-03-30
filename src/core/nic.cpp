/**
 * @file nic.cpp
 * @brief NIC implementation with DMA timing and interrupt coalescing.
 */

#include "nic.hpp"

#include <algorithm>

namespace riscv {

// ── Constructor / reset ────────────────────────────────────────────────

NIC::NIC(std::shared_ptr<Memory> system_memory, NicTiming timing)
    : sys_mem_(std::move(system_memory))
    , timing_(timing)
{
    reset();
}

void NIC::reset()
{
    regs_.fill(0);
    regs_[NicReg::STATUS / 4] = NicCtrl::LINK_UP;

    interrupt_mask_        = 0;
    interrupt_pending_     = false;
    coalesce_pending_pkts_ = 0;
    coalesce_first_cycle_  = 0;
    pending_dma_.clear();
    rx_queue_.clear();
    stats_.reset();
    current_cycle_ = 0;
}

// ── MMIO register access ───────────────────────────────────────────────

u32 NIC::reg(addr_t offset) const noexcept
{
    if (offset < NicReg::REG_SIZE && (offset & 3) == 0)
        return regs_[offset / 4];
    return 0;

}

MemoryResult NIC::read32(addr_t addr) const
{
    if (addr >= NicReg::REG_SIZE || (addr & 3) != 0)
        return {0, 1, false};

    u32 value = regs_[addr / 4];

    if (addr == NicReg::ICR)
        regs_[NicReg::ICR / 4] = 0;

    return {value, timing_.mmio_read_cycles, true};
}

MemoryResult NIC::write32(addr_t addr, u32 value)
{
    if (addr >= NicReg::REG_SIZE || (addr & 3) != 0)
        return {0, 1, false};

    switch (addr) {
        case NicReg::CTRL:
            if (value & NicCtrl::RST)
                reset();
            else
                regs_[NicReg::CTRL / 4] = value & ~NicCtrl::LINK_UP;
            break;

        case NicReg::STATUS:
            break;

        case NicReg::IMC:
            interrupt_mask_ &= ~value;
            break;

        case NicReg::IMS:
            interrupt_mask_ |= value;
            break;

        case NicReg::ICR:
            regs_[NicReg::ICR / 4] &= ~value;
            break;

        case NicReg::ITR:
            coalesce_.max_delay_cycles = value & 0xFFFF;
            coalesce_.max_packets      = (value >> 16) & 0xFF;
            coalesce_.enabled          = (coalesce_.max_delay_cycles > 0
                                       || coalesce_.max_packets > 1);
            regs_[addr / 4] = value;
            break;

        case NicReg::RDT:
            regs_[addr / 4] = value;
            process_rx_queue();
            break;

        case NicReg::TDT:
            regs_[addr / 4] = value;
            process_tx_ring();
            break;

        case NicReg::RXPKT: case NicReg::TXPKT:
        case NicReg::RXBYTES: case NicReg::TXBYTES:
            break;

        default:
            regs_[addr / 4] = value;
            break;
    }

    return {value, timing_.mmio_write_cycles, true};
}

MemoryResult NIC::read16(addr_t addr) const
{
    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 2) * 8;
    return {(r.value >> shift) & 0xFFFF, r.cycles, true};
}

MemoryResult NIC::write16(addr_t addr, u16 value)
{
    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 2) * 8;
    u32 mask = 0xFFFF << shift;
    return write32(addr & ~3u, (r.value & ~mask) | (static_cast<u32>(value) << shift));
}

MemoryResult NIC::read8(addr_t addr) const
{
    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 3) * 8;
    return {(r.value >> shift) & 0xFF, r.cycles, true};
}

MemoryResult NIC::write8(addr_t addr, u8 value)
{
    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 3) * 8;
    u32 mask = 0xFF << shift;
    return write32(addr & ~3u, (r.value & ~mask) | (static_cast<u32>(value) << shift));
}

bool NIC::valid_address(addr_t addr, size_t size) const
{
    return addr + size <= NicReg::REG_SIZE;
}

// ── Simulation tick ────────────────────────────────────────────────────

void NIC::tick(cycle_t current_cycle)
{
    current_cycle_ = current_cycle;

    complete_dma();

    if (rx_enabled())
        process_rx_queue();

    if (coalesce_.enabled && coalesce_pending_pkts_ > 0
        && coalesce_.max_delay_cycles > 0
        && (current_cycle_ - coalesce_first_cycle_) >= coalesce_.max_delay_cycles)
    {
        stats_.coalesced_packets += coalesce_pending_pkts_ - 1;
        coalesce_pending_pkts_ = 0;
        raise_interrupt(NicInt::TIMER);
    }
}

// ── Packet injection / retrieval ───────────────────────────────────────

void NIC::inject_packet(const Packet& pkt)
{
    Packet p = pkt;
    p.arrival_cycle = current_cycle_;
    rx_queue_.push_back(std::move(p));

    if (rx_enabled())
        process_rx_queue();
}


std::optional<Packet> NIC::poll_tx()
{
    if (tx_complete_.empty())
        return std::nullopt;

    Packet pkt = std::move(tx_complete_.front());
    tx_complete_.pop_front();
    return pkt;
}

// ── Interrupt handling ─────────────────────────────────────────────────

void NIC::raise_interrupt(u32 cause)
{
    regs_[NicReg::ICR / 4] |= cause;

    if (cause & interrupt_mask_) {
        if (!interrupt_pending_) {
            interrupt_pending_ = true;
            stats_.interrupts_raised++;
            
            if (interrupt_cb_)
                interrupt_cb_();
        }
    }
}

void NIC::check_coalescing(u32 cause)
{
    coalesce_pending_pkts_++;

    if (!coalesce_.enabled) {
        raise_interrupt(cause);
        coalesce_pending_pkts_ = 0;
        return;
    }

    if (coalesce_pending_pkts_ == 1)
        coalesce_first_cycle_ = current_cycle_;

    if (coalesce_pending_pkts_ >= coalesce_.max_packets) {
        stats_.coalesced_packets += coalesce_pending_pkts_ - 1;
        coalesce_pending_pkts_ = 0;
        raise_interrupt(cause);
    }
}

// ── RX processing ──────────────────────────────────────────────────────


void NIC::process_rx_queue()
{
    if (!rx_enabled() || rx_queue_.empty())
        return;

    u32 ring_size = rx_ring_size();
    if (ring_size == 0) return;

    u32 pending_rx = 0;
    for (const auto& dma : pending_dma_)
        if (dma.type == PendingDma::Type::RX) ++pending_rx;

    u32 head = regs_[NicReg::RDH / 4];
    u32 pending_head = (head + pending_rx) % ring_size;
    u32 tail = regs_[NicReg::RDT / 4];

    while (!rx_queue_.empty()) {
        if (pending_head == tail)
            break;

        Packet pkt = std::move(rx_queue_.front());
        rx_queue_.pop_front();

        pkt.dma_start_cycle = current_cycle_;
        u32 dma_time = timing_.dma_cycles(static_cast<u32>(pkt.data.size()));

        PendingDma dma;
        dma.type           = PendingDma::Type::RX;
        dma.desc_idx       = pending_head;
        dma.packet         = std::move(pkt);
        dma.complete_cycle = current_cycle_ + dma_time;
        pending_dma_.push_back(std::move(dma));

        pending_head = (pending_head + 1) % ring_size;
    }
}

// ── TX processing ──────────────────────────────────────────────────────

void NIC::process_tx_ring()
{
    if (!tx_enabled()) return;

    u32 ring_size = tx_ring_size();
    if (ring_size == 0) return;

    u32 head = regs_[NicReg::TDH / 4];
    u32 tail = regs_[NicReg::TDT / 4];

    while (head != tail) {
        TxDescriptor desc = read_tx_desc(head);

        Packet pkt;
        pkt.data.resize(desc.length);
        pkt.dma_start_cycle = current_cycle_;

        u32 dma_time = timing_.dma_cycles(desc.length);

        PendingDma dma;
        dma.type           = PendingDma::Type::TX;
        dma.desc_idx       = head;
        dma.packet         = std::move(pkt);
        dma.complete_cycle = current_cycle_ + dma_time;
        pending_dma_.push_back(std::move(dma));

        head = (head + 1) % ring_size;
    }
}

// ── DMA completion ─────────────────────────────────────────────────────

void NIC::complete_dma()
{
    while (!pending_dma_.empty() && pending_dma_.front().complete_cycle <= current_cycle_) {
        PendingDma dma = std::move(pending_dma_.front());
        pending_dma_.pop_front();

        if (dma.type == PendingDma::Type::RX) {
            /* Write packet data to buffer */
            RxDescriptor desc = read_rx_desc(dma.desc_idx);
            for (size_t i = 0; i < dma.packet.data.size(); ++i)
                sys_mem_->write8(desc.buffer_addr + static_cast<addr_t>(i),
                                 dma.packet.data[i]);

            /* Update descriptor status */
            desc.length = static_cast<u16>(dma.packet.data.size());
            desc.status = RxDescriptor::STATUS_DD | RxDescriptor::STATUS_EOP;
            desc.errors = 0;
            write_rx_desc(dma.desc_idx, desc);

            /* Advance head */
            u32 ring_size = rx_ring_size();
            regs_[NicReg::RDH / 4] = (dma.desc_idx + 1) % ring_size;

            /* Statistics */
            stats_.rx_packets++;
            stats_.rx_bytes += dma.packet.data.size();
            regs_[NicReg::RXPKT / 4]++;
            regs_[NicReg::RXBYTES / 4] += static_cast<u32>(dma.packet.data.size());

            /* Latency tracking */
            dma.packet.dma_done_cycle = current_cycle_;
            cycle_t latency = current_cycle_ - dma.packet.arrival_cycle;
            stats_.total_rx_latency += latency;
            stats_.min_rx_latency = std::min(stats_.min_rx_latency, latency);
            stats_.max_rx_latency = std::max(stats_.max_rx_latency, latency);

            /* Interrupt */
            check_coalescing(NicInt::RXQ0);
        
        } else { /* TX */
            /* Read packet data from buffer */
            TxDescriptor desc = read_tx_desc(dma.desc_idx);
            for (u32 i = 0; i < desc.length; ++i) {
                auto r = sys_mem_->read8(desc.buffer_addr + i);

                if (i < dma.packet.data.size())
                    dma.packet.data[i] = r.ok ? static_cast<u8>(r.value) : 0;
            }

            /* Mark descriptor done */
            desc.status = TxDescriptor::STATUS_DD;
            write_tx_desc(dma.desc_idx, desc);

            /* Advance head */
            u32 ring_size = tx_ring_size();
            regs_[NicReg::TDH / 4] = (dma.desc_idx + 1) % ring_size;

            /* Statistics */
            stats_.tx_packets++;
            stats_.tx_bytes += dma.packet.data.size();
            regs_[NicReg::TXPKT / 4]++;
            regs_[NicReg::TXBYTES / 4] += static_cast<u32>(dma.packet.data.size());

            tx_complete_.push_back(std::move(dma.packet));

            /* Interrupt if report-status requested */
            if (desc.cmd & TxDescriptor::CMD_RS)
                check_coalescing(NicInt::TXQ0);
        }
    }
}

// ── Descriptor ring access (via system memory) ─────────────────────────

RxDescriptor NIC::read_rx_desc(u32 idx) const
{
    addr_t base = regs_[NicReg::RDBAL / 4];
    addr_t addr = base + idx * 16;

    RxDescriptor d{};
    d.buffer_addr    = sys_mem_->read32(addr).value;
    d.buffer_addr_hi = sys_mem_->read32(addr + 4).value;
    
    u32 w2 = sys_mem_->read32(addr + 8).value;
    u32 w3 = sys_mem_->read32(addr + 12).value;

    d.length   = static_cast<u16>(w2 & 0xFFFF);
    d.checksum = static_cast<u16>((w2 >> 16) & 0xFFFF);
    d.status   = static_cast<u8>(w3 & 0xFF);
    d.errors   = static_cast<u8>((w3 >> 8) & 0xFF);
    d.vlan     = static_cast<u16>((w3 >> 16) & 0xFFFF);
    return d;
}

void NIC::write_rx_desc(u32 idx, const RxDescriptor& d)
{
    addr_t base = regs_[NicReg::RDBAL / 4];
    addr_t addr = base + idx * 16;

    sys_mem_->write32(addr,      d.buffer_addr);
    sys_mem_->write32(addr + 4,  d.buffer_addr_hi);
    sys_mem_->write32(addr + 8,  d.length | (static_cast<u32>(d.checksum) << 16));
    sys_mem_->write32(addr + 12, d.status | (static_cast<u32>(d.errors) << 8)
                                          | (static_cast<u32>(d.vlan) << 16));
}

TxDescriptor NIC::read_tx_desc(u32 idx) const
{
    addr_t base = regs_[NicReg::TDBAL / 4];
    addr_t addr = base + idx * 16;

    TxDescriptor d{};
    d.buffer_addr    = sys_mem_->read32(addr).value;
    d.buffer_addr_hi = sys_mem_->read32(addr + 4).value;

    u32 w2 = sys_mem_->read32(addr + 8).value;
    u32 w3 = sys_mem_->read32(addr + 12).value;

    d.length = static_cast<u16>(w2 & 0xFFFF);
    d.cso    = static_cast<u8>((w2 >> 16) & 0xFF);
    d.cmd    = static_cast<u8>((w2 >> 24) & 0xFF);
    d.status = static_cast<u8>(w3 & 0xFF);
    d.css    = static_cast<u8>((w3 >> 8) & 0xFF);
    d.vlan   = static_cast<u16>((w3 >> 16) & 0xFFFF);
    return d;
}

void NIC::write_tx_desc(u32 idx, const TxDescriptor& d)
{
    addr_t base = regs_[NicReg::TDBAL / 4];
    addr_t addr = base + idx * 16;

    sys_mem_->write32(addr,      d.buffer_addr);
    sys_mem_->write32(addr + 4,  d.buffer_addr_hi);
    sys_mem_->write32(addr + 8,  d.length | (static_cast<u32>(d.cso) << 16)
                                          | (static_cast<u32>(d.cmd) << 24));
    sys_mem_->write32(addr + 12, d.status | (static_cast<u32>(d.css) << 8)
                                          | (static_cast<u32>(d.vlan) << 16));
}

// ── Loopback ───────────────────────────────────────────────────────────

void NicLoopback::tick(cycle_t current_cycle)
{
    while (auto pkt = nic_.poll_tx()) {
        DelayedPacket dp;
        dp.pkt           = std::move(*pkt);
        dp.deliver_cycle = current_cycle + delay_;
        pending_.push_back(std::move(dp));
    }

    while (!pending_.empty() && pending_.front().deliver_cycle <= current_cycle) {
        nic_.inject_packet(pending_.front().pkt);
        pending_.pop_front();
    }
}

} // namespace riscv
