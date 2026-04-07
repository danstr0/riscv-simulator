/**
 * @file multicore.cpp
 * @brief Multi-core CPU system implementation.
 */

#include "multicore.hpp"

#include <cassert>

namespace riscv {

// ── Construction ───────────────────────────────────────────────────────

MultiCoreCPU::MultiCoreCPU(MultiCoreConfig config)
    : config_(config)
{
    assert(config_.num_cores > 0 && config_.num_cores <= 16);

    /* Build memory hierarchy bottom-up */
    main_mem_ = std::make_shared<FlatMemory>(0, config_.main_memory_size);

    std::shared_ptr<Memory> l2_backing;
    if (config_.l3.size_bytes > 0){
        l3_cache_ = std::make_shared<Cache>(config_.l3, main_mem_);
        l2_backing = l3_cache_;
    } else {
        l2_backing = main_mem_;
    }

    l2_cache_ = std::make_shared<Cache>(config_.l2, l2_backing);

    l1d_caches_.reserve(config_.num_cores);
    for (u32 i = 0; i < config_.num_cores; ++i)
        l1d_caches_.push_back(std::make_shared<Cache>(config_.l1d, l2_cache_));

    /* Coherence controller */

    std::vector<Cache*> l1_ptrs;
    l1_ptrs.reserve(config_.num_cores);
    for (auto& c : l1d_caches_) l1_ptrs.push_back(c.get());

    coherence_ = std::make_unique<CoherenceController>(std::move(l1_ptrs), 
                                                       l2_cache_, 
                                                       config_.l1d.line_size);

    for (u32 i = 0; i < config_.num_cores; ++i) {
        auto* ctrl = coherence_.get();
        u32 core_id = i;

        l1d_caches_[i]->set_on_read_miss([ctrl, core_id](addr_t line_addr) -> u32
        {
            return ctrl->handle_read_miss(core_id, line_addr);
        });
    
        l1d_caches_[i]->set_on_write([ctrl, core_id](addr_t line_addr) -> u32
        {
            return ctrl->handle_write_miss(core_id, line_addr);
        });
    }

    /* MMIO bus */

    auto plic_mem =  std::shared_ptr<Memory>(std::shared_ptr<void>{}, &plic_);
    auto timer_mem = std::shared_ptr<Memory>(std::shared_ptr<void>{}, &timer_);

    /* Create cores */

    cores_.reserve(config_.num_cores);
    for (u32 i = 0; i < config_.num_cores; ++i) {
        auto bus = std::make_shared<MMIOBus>();
        bus->set_default(l1d_caches_[i]);
        bus->map(0x1000'0000, PLIC::REG_SIZE, plic_mem, "plic");
        bus->map(0x1000'1000, Timer::REG_SIZE, timer_mem, "timer");

        cores_.push_back(std::make_unique<PipelinedCPU>(bus, config_.pipeline));
    }

    wire_interrupts();
}

// ── Interrupt wiring ───────────────────────────────────────────────────

void MultiCoreCPU::wire_interrupts()
{
    /* Timer -> all cores' mip.MTIP */
    timer_.set_notify([this](bool pending)
    {
        for (auto& core : cores_) {
            if (pending)
                core->csrs().set_mip_bit(MInterrupt::MTIE);
            else
                core->csrs().clear_mip_bit(MInterrupt::MTIE);
        }
    });

    /* PLIC -> all cores' mip.MEIP */
    plic_.set_notify([this](bool pending)
    {
        for (auto& core : cores_) {
            if (pending)
                core->csrs().set_mip_bit(MInterrupt::MEIE);
            else
                core->csrs().clear_mip_bit(MInterrupt::MEIE);
        }
    });
}

// ── Program loading ────────────────────────────────────────────────────

void MultiCoreCPU::load_program(addr_t addr, std::span<const u8> program)
{
    main_mem_->load(addr, program);
}

void MultiCoreCPU::load_instruction(addr_t addr, u32 instruction)
{
    const std::array<u8, 4> bytes = {
        static_cast<u8>(instruction),
        static_cast<u8>(instruction >> 8),
        static_cast<u8>(instruction >> 16),
        static_cast<u8>(instruction >> 24),
    };
    main_mem_->load(addr, bytes);
}

// ── Execution ──────────────────────────────────────────────────────────

bool MultiCoreCPU::tick()
{
    cycle_count_++;

    timer_.tick(cycle_count_);

    if (nic_) nic_->tick(cycle_count_);

    bool any_running = false;
    for (auto& core : cores_) {
        if (core->tick())
            any_running = true;
    }

    return any_running;
}

cycle_t MultiCoreCPU::run_cycles(cycle_t n)
{
    cycle_t count = 0;
    while (count < n) {
        cycle_count_++;
        timer_.tick(cycle_count_);

        if (nic_) nic_->tick(cycle_count_);
        for (auto& core : cores_) core->tick();
        ++count;
    }
    return count;
}

cycle_t MultiCoreCPU::run_until_all_halted(cycle_t max_cycles)
{
    cycle_t count = 0;
    while (count < max_cycles) {
        if (!tick()) break;
        ++count;
    }
    return count;
}

// ── NIC attachment ─────────────────────────────────────────────────────

void MultiCoreCPU::attach_nic(std::shared_ptr<NIC> nic, addr_t mmio_base)
{
    nic_ = std::move(nic);
    nic_->plic_source = config_.nic_plic_source;

    u32 source = config_.nic_plic_source;
    nic_->set_interrupt_callback([this, source]()
    {
        plic_.set_pending(source);
    });

    // TODO: map NIC on per-core buses.
}

// ── Core configuration ─────────────────────────────────────────────────

void MultiCoreCPU::set_core_pc(u32 core_id, addr_t pc)
{
    assert(core_id < config_.num_cores);
    cores_[core_id]->set_pc(pc);
}

void MultiCoreCPU::set_core_reg(u32 core_id, reg_idx_t r, u32 value)
{
    assert(core_id < config_.num_cores);
    cores_[core_id]->set_reg(r, value);
}

// ── Statistics ─────────────────────────────────────────────────────────

MultiCoreStats MultiCoreCPU::get_stats() const
{
    MultiCoreStats s;
    s.core_stats.resize(config_.num_cores);
    s.l1d_stats.resize(config_.num_cores);

    for (u32 i = 0; i < config_.num_cores; ++i) {
        s.core_stats[i] = cores_[i]->stats();
        s.l1d_stats[i]  = l1d_caches_[i]->stats();
    }

    s.l2_stats  = l2_cache_->stats();
    s.l3_stats  = l3_cache_ ? l3_cache_->stats() : CacheStats{};
    s.coherence = coherence_->stats();

    return s;
}

void MultiCoreCPU::reset_stats()
{
    for (auto& l1 : l1d_caches_) l1->stats().reset();
    l2_cache_->stats().reset();
    if (l3_cache_) l3_cache_->stats().reset();
    coherence_->reset_stats();
}

// ── Reset ──────────────────────────────────────────────────────────────

void MultiCoreCPU::reset()
{
    for (auto& core : cores_) core->reset();
    for (auto& l1 : l1d_caches_) {
        l1->flush_all();
        l1->stats().reset();
    }

    l2_cache_->flush_all();
    l2_cache_->stats().reset();

    if (l3_cache_) {
        l3_cache_->flush_all();
        l3_cache_->stats().reset();
    }

    coherence_->reset_stats();
    plic_.reset();
    timer_.reset();
    cycle_count_ = 0;
}

} // namespace riscv
