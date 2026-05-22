/**
 * @file plic.cpp
 * @brief Simplified PLIC implementation for single-hart RISC-V.
 */

#include "plic.hpp"

namespace riscv {

void PLIC::reset()
{
    priorities_.fill(0);
    pending_bits_ = 0;
    enable_bits_  = 0;
    threshold_    = 0;
    claimed_      = 0;
}

// ── MMIO reads ─────────────────────────────────────────────────────────

MemoryResult PLIC::read32(addr_t addr) const
{
    // Source priorities: 0x000 + source * 4
    if (addr >= PRIORITY_BASE
     && addr  < PRIORITY_BASE + PLICConfig::MAX_SOURCES * 4)
    {
        u32 source = (addr - PRIORITY_BASE) / 4;
        return {priorities_[source], 1, true};
    }

    // Pending bits
    if (addr == PENDING_BASE)
        return {pending_bits_, 1, true};

    // Enable bits
    if (addr == ENABLE_BASE)
        return {enable_bits_, 1, true};

    // Threshold
    if (addr == THRESHOLD_ADDR)
        return {threshold_, 1, true};

    // Claim: returns highest-priority pending+enabled source, clears pending
    if (addr == CLAIM_ADDR)
        return { const_cast<PLIC*>(this)->claim(), 1, true };

    return {0, 1, false};
}

bool PLIC::valid_address(addr_t addr, size_t size) const
{
    return size > 0 && addr + size <= REG_SIZE;
}

// ── MMIO writes ────────────────────────────────────────────────────────

MemoryResult PLIC::write32(addr_t addr, u32 value)
{
    // Source priorities
    if (addr >= PRIORITY_BASE
     && addr  < PRIORITY_BASE + PLICConfig::MAX_SOURCES * 4)
    {
        u32 source = (addr - PRIORITY_BASE) / 4;
        if (source > 0)
            priorities_[source] = value & 0x7;

        return {value, 1, true};
    }

    // Pending bits: read-only
    if (addr == PENDING_BASE)
        return {value, 1 , true};

    // Enable bits
    if (addr == ENABLE_BASE)
    {
        enable_bits_ = value & ~1u;
        notify();
        return {value, 1, true};
    }

    // Threshold
    if (addr == THRESHOLD_ADDR)
    {
        threshold_ = value & 0x7;
        notify();
        return {value, 1, true};
    }

    // Complete: write the source ID that was claimed.
    if (addr == CLAIM_ADDR)
    {
        if (value > 0
         && value < PLICConfig::MAX_SOURCES
         && value == claimed_)
        {
            claimed_ = 0;
            notify();
        }
        return {value, 1, true};
    }

    return {0, 1, false};
}

// ── Device API ─────────────────────────────────────────────────────────

void PLIC::set_pending(u32 source)
{
    if (source == 0
     || source >= PLICConfig::MAX_SOURCES) return;

    pending_bits_ |= (1u << source);
    notify();
}

void PLIC::clear_pending(u32 source)
{
    if (source == 0
     || source >= PLICConfig::MAX_SOURCES) return;

    pending_bits_ &= ~(1u << source);
    notify();
}

bool PLIC::interrupt_pending() const
{
    for (u32 i = 1; i < PLICConfig::MAX_SOURCES; ++i)
    {
        if ((pending_bits_ & (1u << i))
         && (enable_bits_  & (1u << i))
         && priorities_[i] > threshold_)
            return true;
    }
    return false;
}

u32 PLIC::claim()
{
    u32 best = 0;
    u32 best_prio = 0;
    for (u32 i = 1; i < PLICConfig::MAX_SOURCES; ++i)
    {
        if ((pending_bits_ & (1u << i))
         && (enable_bits_  & (1u << i)))
        {
            if (priorities_[i] > threshold_
             && priorities_[i] > best_prio)
            {
                best = i;
                best_prio = priorities_[i];
            }
        }
    }
    if (best != 0)
    {
        pending_bits_ &= ~(1u << best);
        claimed_ = best;
        notify();
    }
    return best;
}

void PLIC::complete(u32 source)
{
    if (source > 0
     && source < PLICConfig::MAX_SOURCES
     && source == claimed_)
    {
        claimed_ = 0;
        notify();
    }
}

void PLIC::set_priority(u32 source, u32 prio)
{
    if (source > 0
     && source < PLICConfig::MAX_SOURCES)
    {
        priorities_[source] = prio & 0x7;
        notify();
    }
}

void PLIC::set_enable(u32 source, bool enable)
{
    if (source > 0
     && source < PLICConfig::MAX_SOURCES)
    {
        if (enable)
            enable_bits_ |= (1u << source);
        else
            enable_bits_ &= ~(1u << source);
        notify();
    }
}

u32 PLIC::priority(u32 source) const
{
    return source < PLICConfig::MAX_SOURCES ? priorities_[source] : 0;
}

bool PLIC::enabled(u32 source) const
{
    return source < PLICConfig::MAX_SOURCES && (enable_bits_ & (1u << source));
}

void PLIC::notify() const
{
    if (notify_cb_)
        notify_cb_(interrupt_pending());
}

} // namespace riscv
