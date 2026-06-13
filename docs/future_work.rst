===========
Future Work
===========
 
This document lists known limitations, planned improvements, and
architectural extensions that would meaningfully expand the
simulator's capabilities.


Pipeline Model Refinement
-------------------------

**True EX→EX forwarding**

The current forwarding implementation covers MEM→EX bypass, but does
not fully model the timing of multi-cycle operations.  In real
hardware, a multiply instruction occupies the EX stage for multiple
cycles, and the forwarding path from EX to the next instruction's EX
input must account for this latency. The current model treats all ALU
operations as single-cycle in EX, which overestimates IPC for
multiply-heavy workloads.

Fixing this requires modeling functional unit latency in the EX
stage: MUL/MULH take 3–5 cycles, DIV/REM take 10–30 cycles, and
the pipeline stalls or uses a scoreboard to track when results are
ready for forwarding.


ISA Extension to RV32G
-----------------------

The simulator currently implements RV32IMAV (integer, multiply,
atomics, vector subset) + Zicsr. RV32G adds:

- **F extension** — single-precision floating-point (32 registers,
  fadd, fmul, fdiv, fmadd, fcvt, fmv, comparison, classification).
- **D extension** — double-precision floating-point (reuses the F
  register file widened to 64 bits).
- **Zifencei** — FENCE.I instruction (instruction cache flush).

The F and D extensions would require adding a floating-point
register file, floating-point ALU operations in the executor, and
FP-aware forwarding in the pipeline.  The vector extension would
also benefit from SEW=64 support for double-precision vector
arithmetic.


Device Memory Region and MultiCoreCPU integration
-------------------------------------------------

**Device support for single-core CPU types**

The PLIC, timer, and NIC are currently only available in multi-core
mode (``MultiCoreCPU``).  The ``PipelinedCPU`` and ``CPU`` types
have no MMIO bus and cannot access devices via load/store
instructions.  Adding device support to these CPU types would
require wrapping them in an ``MMIOBus`` (which ``MultiCoreCPU``
already does internally) and computing a device base address.

This would enable interrupt-driven programs on single-core systems,
which is useful for benchmarking interrupt latency without the
added complexity of cache coherence and multi-core contention.


Multicore Research Platform
---------------------------

The research platform (``rvsim``) currently uses only ``PipelinedCPU``
for sweep benchmarking.  Adding multicore sweep support would enable
questions like "how does L1 size affect contention on a shared
counter?" or "what's the cache coherence overhead at 2 vs 4 vs 8
cores?"

This is blocked on two issues:

1. **Device region sizing** — the PLIC's spec-compliant address space
   is too large for the current compact device region, making
   realistic interrupt-driven multicore benchmarks impractical.

2. **Heterogeneous multicore** — sweeping multicore configurations
   is most interesting when cores can have different parameters
   (e.g., big.LITTLE with fast/slow cores, or asymmetric cache
   sizes).  The current ``MultiCoreCPU`` applies the same pipeline
   and L1 config to all cores.  Supporting heterogeneous configs
   would require per-core pipeline parameters in the config file
   and independent ``PipelineConfig`` objects per core.

Without heterogeneous support, multicore sweeps would only vary
shared resources (L2 size, coherence parameters, core count),
which is too narrow a parameter space.


Out-of-Order Execution
-----------------------

The simulator implements an in-order pipeline, which is the standard
model for educational and embedded RISC-V cores (SiFive E-series,
BOOM educational core).  Adding out-of-order execution would model
high-performance cores (SiFive P-series, Ventana Veyron) and enable
benchmarking of:

- **Instruction-level parallelism (ILP):** how much parallelism
  exists in a workload beyond what in-order issue can exploit.
- **Reorder buffer sizing:** how many in-flight instructions are
  needed to hide memory latency.
- **Issue width:** superscalar execution (2-wide, 4-wide) and its
  interaction with branch prediction accuracy.

An OoO implementation would require:

- A reorder buffer (ROB) for in-order commit.
- A reservation station or issue queue for dynamic scheduling.
- Register renaming (physical register file + RAT).
- A load/store queue for memory disambiguation.
- Speculative execution with rollback on mispredict.

This is a substantial architectural change, but would make the simulator
applicable to a much broader class of architecture research questions.


Other Improvements
------------------

- **Branch predictor enhancements:** tournament predictor (local +
  global history), TAGE, return address stack (RAS) for function
  calls.  The current bimodal predictor is the baseline; more
  sophisticated predictors would better model modern cores.
- **TLB and virtual memory:** page table walks, TLB hit/miss
  modeling, page fault handling.  Currently the simulator uses
  physical addresses only.
- **Non-blocking caches:** the current cache blocks on miss (no
  MSHRs).  Adding miss-status holding registers would allow the
  pipeline to continue executing independent instructions while
  a cache miss is being serviced.
- **Prefetching:** hardware stride prefetcher or next-line
  prefetcher to reduce compulsory misses on sequential access
  patterns.
- **Power/area estimation:** augment cycle counts with rough
  energy-per-operation estimates to enable power-performance
  Pareto analysis, not just performance-only.

