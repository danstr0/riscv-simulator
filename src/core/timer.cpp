/**
 * @file timer.cpp
 * @brief CLINT-style machine timer implementation.
 */

#include "timer.hpp"

namespace riscv {

void Timer::reset()
{
    mtime_    = 0;
    mtimecmp_ = ~u64{0};
    pending_  = false;
}

bool Timer::tick(cycle_t cycle)
{
    mtime_ = cycle;
    bool was_pending = pending_;
    check();
    return !was_pending && pending_;
}

void Timer::check()
{
    bool new_pending = (mtime_ >= mtimecmp_);
    if (new_pending != pending_)
    {
        pending_ = new_pending;
        if (notify_cb_) notify_cb_(pending_);
    }
}

// ── MMIO ───────────────────────────────────────────────────────────────

MemoryResult Timer::read32(addr_t addr) const
{
    switch(addr)
    {
        case MTIME_LO:    return {static_cast<u32>(mtime_),          1, true};
        case MTIME_HI:    return {static_cast<u32>(mtime_ >> 32),    1, true};
        case MTIMECMP_LO: return {static_cast<u32>(mtimecmp_),       1, true};
        case MTIMECMP_HI: return {static_cast<u32>(mtimecmp_ >> 32), 1, true};
        default:          return {0, 1, false};
    }
}

MemoryResult Timer::write32(addr_t addr, u32 value)
{
    switch(addr)
    {
        case MTIME_LO:
        case MTIME_HI:
            // Silently ignored; mtime advances only via tick()
            return {value, 1, true};

        case MTIMECMP_LO:
            mtimecmp_ = (mtimecmp_ & 0xFFFF'FFFF'0000'0000ULL) | value;
            check();
            return {value, 1, true};

        case MTIMECMP_HI:
            mtimecmp_ = (mtimecmp_ & 0x0000'0000'FFFF'FFFFULL)
                      | (static_cast<u64>(value) << 32);
            check();
            return {value, 1, true};

        default:
            return {0, 1, false};
    }
}

bool Timer::valid_address(addr_t addr, size_t size) const
{
    return size > 0 && addr + size <= REG_SIZE;
}

} // namespace riscv
