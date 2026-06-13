============
Architecture
============

This document describes the simulated RISC-V architecture: which
instructions are supported, how the pipeline works, how the cache
hierarchy is organized, and how the multi-core system ties everything
together.


Instruction Set
---------------

The simulator implements a subset of the RV32 ISA sufficient for
general-purpose computation, SIMD processing, multi-core
synchronization, and system-level interrupt handling.

RV32I — Base Integer (37 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 20 50 30
   :header-rows: 1

   * - Category
     - Instructions
     - Encoding
   * - Loads
     - ``LB``, ``LH``, ``LW``, ``LBU``, ``LHU``
     - I-type
   * - Stores
     - ``SB``, ``SH``, ``SW``
     - S-type
   * - Branches
     - ``BEQ``, ``BNE``, ``BLT``, ``BGE``, ``BLTU``, ``BGEU``
     - B-type
   * - Jumps
     - ``JAL``, ``JALR``
     - J-type / I-type
   * - Upper imm
     - ``LUI``, ``AUIPC``
     - U-type
   * - Reg-imm
     - ``ADDI``, ``SLTI``, ``SLTIU``, ``XORI``, ``ORI``, ``ANDI``
     - I-type
   * - Shifts
     - ``SLLI``, ``SRLI``, ``SRAI``
     - I-type (funct7)
   * - Reg-reg
     - ``ADD``, ``SUB``, ``SLL``, ``SLT``, ``SLTU``, ``XOR``, ``SRL``, ``SRA``, ``OR``, ``AND``
     - R-type
   * - System
     - ``ECALL``, ``EBREAK``, ``FENCE``
     - I-type

RV32M — Integer Multiply/Divide (8 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All R-type with funct7 = 0000001:

- ``MUL``, ``MULH``, ``MULHSU``, ``MULHU``
- ``DIV``, ``DIVU``, ``REM``, ``REMU``

RV32A — Atomics (11 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All R-type with opcode = 0101111 (AMO):

- ``LR.W`` — load-reserved (places a reservation on the cache line)
- ``SC.W`` — store-conditional (succeeds only if reservation is intact)
- ``AMOSWAP.W``, ``AMOADD.W``, ``AMOXOR.W``, ``AMOAND.W``, ``AMOOR.W``
- ``AMOMIN.W``, ``AMOMAX.W``, ``AMOMINU.W``, ``AMOMAXU.W``

``LR``/``SC`` is the foundation for lock-free synchronization — ``SC`` returns
``0`` on success (reservation held) and ``1`` on failure (another core wrote
to the line between ``LR`` and ``SC``).

RVV — Vector Extension Subset (25 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VLEN = 128 bits (configurable), SEW = 32 only, LMUL = 1.  With
SEW=32 and VLEN=128, each vector register holds 4 elements.

.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Category
     - Instructions
   * - Configuration
     - ``VSETVLI``
   * - Load/store
     - ``VLE32.V``, ``VSE32.V`` (unit-stride, 32-bit elements)
   * - Arithmetic
     - ``VADD``, ``VSUB``, ``VAND``, ``VOR``, ``VXOR``
       (``.VV`` and ``.VX`` variants)
   * - Shifts
     - ``VSLL.VX``, ``VSRL.VX``
   * - Comparison
     - ``VMSEQ(.VV|.VX)``, ``VMSLT.VV``, ``VMSLTU.VV``
   * - Mask-mask
     - ``VMAND``, ``VMNAND``, ``VMANDN``, ``VMXOR``, ``VMOR``,
       ``VMNOR``, ``VMORN``, ``VMXNOR`` (``.MM``)
   * - Reduction
     - ``VREDSUM.VS``
   * - Move
     - ``VMV.V.X`` (splat scalar → vector),
       ``VMV.X.S`` (extract element 0 → scalar)

CSR Access (6 instructions + MRET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CSRRW``, ``CSRRS``, ``CSRRC``, ``CSRRWI``, ``CSRRSI``, ``CSRRCI`` — atomic read-modify-write
of control/status registers. ``MRET`` returns from a machine-mode trap.

.. list-table:: **Supported CSRs** 
   :widths: 15 15 70
   :header-rows: 1

   * - Name
     - Address
     - Purpose
   * - ``mstatus``
     - 0x300
     - Global interrupt enable (MIE, MPIE bits)
   * - ``mie``
     - 0x304
     - Per-source interrupt enable (MEIE, MTIE, MSIE)
   * - ``mtvec``
     - 0x305
     - Trap vector base address
   * - ``mepc``
     - 0x341
     - Exception PC (saved on trap entry)
   * - ``mcause``
     - 0x342
     - Trap cause code
   * - ``mip``
     - 0x344
     - Pending interrupt bits

Interrupt priority: external (MEIE) > timer (MTIE) > software (MSIE).


Pipeline
--------

The simulator offers two CPU modes:

- **Simple CPU** — single-cycle execution, one instruction per call to
  ``step()``.
- **Pipelined CPU** — classic 5-stage in-order pipeline:
  ::

      IF → ID → EX → MEM → WB

  Each stage advances once per ``tick()`` call. The pipeline models
  structural, data, and control hazards with configurable mitigation.

Pipeline stages
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Stage
     - Name
     - Function
   * - ``IF``
     - Fetch
     - Read instruction from memory at PC. Branch predictor selects
       next PC (fall-through or predicted target).
   * - ``ID``
     - Decode
     - Decode instruction, read register file, detect hazards.
       Insert stall bubble if needed.
   * - ``EX``
     - Execute
     - ALU operation, branch resolution, address calculation.
       Forwarding paths inject bypassed values here.
   * - ``MEM``
     - Memory
     - Data cache access for loads and stores.
   * - ``WB``
     - Writeback
     - Write result to register file.

Forwarding (data hazard mitigation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``none``
     - Stall on every RAW dependency until the producing instruction reaches WB.
       Worst case: 2-cycle stall per dependency.
   * - ``partial``
     - Forward from MEM→EX only. Reduces stalls but still stalls
       on EX→EX dependencies.

Branch prediction
^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Predictor
     - Strategy
   * - ``not_taken``
     - Always predict fall-through. Mispredicts on every taken branch.
   * - ``always_taken``
     - Always predict taken. Good for tight backward loops, bad for
       forward branches (if-then).
   * - ``backward_taken``
     - Predict taken if target < PC (backward branch), not-taken otherwise.
   * - ``bimodal_1bit``
     - 1-bit saturating counter per PC. Adapts to branch history but "ping-pongs"
       on alternating outcomes.
   * - ``bimodal_2bit``
     - 2-bit saturating counter per PC. Requires two consecutive mispredicts to
       change predictor.

Misprediction penalty is configurable (default: 3 cycles). On a
mispredict, the pipeline flushes all stages after EX and restarts
fetch at the correct target.


Cache Hierarchy
---------------

The cache hierarchy is fully configurable at each level. All levels
are optional: the CPU can run directly against main memory (useful
for isolating pipeline effects in benchmarks).
::

    CPU → [L1] → [L2] → [L3] → Main Memory

L1 cache (per-core in multi-core mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Size**: configurable (e.g., 4K, 8K, 16K, 32K, 64K)
- **Associativity**: configurable (must be power of 2)
- **Line size**: configurable (e.g., 32, 64, 128 bytes)
- **Replacement policy**: LRU, MRU, Pseudo-LRU, FIFO, Random
- **Write policy**: write-back or write-through
- **Write-allocate**: allocate on write miss, or no-allocate

L2 cache (shared in multi-core mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Required when L1 is present in multi-core configurations
- Configurable size and associativity

L3 cache (shared, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Configurable size and associativity

Geometry constraints (enforced by the config validator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- All sizes, associativities, and line sizes must be powers of 2
- ``size % (line_size × associativity)`` must equal 0
- Cache sizes must not decrease with distance from CPU (L2 ≥ L1, L3 ≥ L2)
- L2 requires L1; L3 requires L2

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Policy
     - Name
     - Description
   * - ``lru``
     - Least Recently Used
     - Evicts the line accessed longest ago.
   * - ``mru``
     - Most Recently Used
     - Evicts the most recently accessed line.
   * - ``plru``
     - Pseudo-LRU
     - Tree-based approximation of LRU using one bit per internal node.
   * - ``fifo``
     - First-In First-Out
     -  Evicts the oldest line regardless of access pattern.
   * - ``random``
     - Random eviction
     -


Multi-Core System
-----------------

The multi-core system connects N pipelined cores through a shared
memory hierarchy with hardware cache coherence.
::

    Core 0 → MMIOBus → L1d[0] ──┐
                                ├── Coherence Controller → L2 → L3 → Main Memory
    Core 1 → MMIOBus → L1d[1] ──┘

Cache modes
^^^^^^^^^^^

1. **No caches** — all cores access main memory directly through
   their MMIO bus. Useful for testing device interactions without
   cache noise.

2. **L1 + L2** — per-core L1 data caches with a shared L2.  A
   directory-based MESI coherence controller manages cross-core
   consistency.

3. **L1 + L2 + L3** — same as above with an additional shared L3
   between L2 and main memory.

MESI coherence protocol
^^^^^^^^^^^^^^^^^^^^^^^

The coherence controller is directory-based (not snoopy-bus).  Each
cache line can be in one of four states:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - State
     - Meaning
   * - Modified
     - Line is dirty, only this core has it. Must write back before sharing.
   * - Exclusive
     - Line is clean, only this core has it. Can transition to Modified on write
       without bus traffic.
   * - Shared
     - Line is clean, multiple cores may have it. Must invalidate other copies
       before writing.
   * - Invalid
     - Line is not present in this cache.

**Coherence latency model:**

- snoop = 2 cycles
- cache-to-cache transfer = 5 cycles
- invalidation = 2 cycles
 
Device MMIO
^^^^^^^^^^^

Devices are memory-mapped immediately above main memory at
page-aligned addresses:

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Offset
     - Device
     - Size
   * - +0x0000
     - PLIC (interrupt controller)
     - 8 KiB
   * - +0x1000
     - Timer (CLINT-style)
     - 16 bytes
   * - +0x2000
     - NIC (network interface)
     - 256 bytes

The device base address is computed as
``align_up(main_memory_size, 0x1000)``. For 64 KiB memory, devices
start at 0x10000.  MMIO accesses bypass the cache entirely — the
per-core MMIO bus routes device addresses to the device objects
directly, and only non-device addresses pass through to the L1 cache.

Devices
^^^^^^^

- **PLIC (Platform-Level Interrupt Controller)** — routes external
  interrupt sources to cores.  Supports up to 31 sources with
  priority-based arbitration.  A source becomes pending, the core
  claims it (masking further interrupts from that source), services
  it, then completes the claim.
- **Timer (CLINT-style)** — 64-bit free-running ``mtime`` counter
  incremented every cycle.  When ``mtime ≥ mtimecmp``, a timer
  interrupt is raised on all cores (via ``mip.MTIP``).
- **NIC (Network Interface Controller)** — cycle-accurate DMA-based
  NIC with descriptor rings, interrupt coalescing, and RSS
  (Receive Side Scaling) for multi-queue packet distribution.
  Supports up to 8 RX/TX queues with configurable coalescing
  thresholds (packet count and timer-based).


Assembler
---------

The two-pass assembler translates ``.s`` source files into machine
code loaded directly into the simulator's memory.

- **Pass 1:** Scan lines, record label addresses, compute instruction
  sizes (LI and LA expand to 2 instructions = 8 bytes).
- **Pass 2:** Encode instructions using the label table for branch and
  jump target resolution.

Syntax
^^^^^^

- **Labels:** ``name:`` at the start of a line
- **Comments:** ``#``, ``;``, or ``//`` to end of line
- **Register names:** ``x0``–``x31`` and ABI names (``zero``, ``ra``,
  ``sp``, ``a0``–``a7``, ``s0``–``s11``, ``t0``–``t6``, ``fp``)
- **Vector registers:** ``v0``–``v31``
- **Immediates:** decimal (``42``, ``-10``), hex (``0xFF``)
- **Memory operands:** ``offset(reg)`` or ``(reg)``
- **Directives:** ``.word``, ``.byte``, ``.zero``, ``.align``

Pseudo-instructions
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Pseudo
     - Expansion
   * - ``nop``
     - ``addi x0, x0, 0``
   * - ``mv``
     - ``addi rd, rs, 0``
   * - ``li``
     - ``addi`` (small) or ``lui+addi``
   * - ``j``
     - ``jal x0, offset``
   * - ``jr``
     - ``jalr x0, rs, 0``
   * - ``ret``
     - ``jalr x0, ra, 0``
   * - ``not``
     - ``xori rd, rs, -1``
   * - ``neg``
     - ``sub rd, x0, rs``
   * - ``beqz``
     - ``beq rs, x0, offset``
   * - ``bnez``
     - ``bne rs, x0, offset``
   * - ``seqz``
     - ``sltiu rd, rs, 1``
   * - ``snez``
     - ``sltu rd, x0, rs``
   * - ``vmnot.m``
     - ``vmnand.mm vd, vs, vs``
   * - ``vmclr.m``
     - ``vmxor.mm vd, vd, vd``
   * - ``vmset.m``
     - ``vmxnor.mm vd, vd, vd``
   * - ``vmcpy.m``
     - ``vmand.mm vd, vs, vs``

