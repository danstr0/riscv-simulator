# RISC-V Architecture Engine

A parameterized, cycle-accurate RISC-V CPU simulator in C++20 for
exploring microarchitectural tradeoffs. Write a program in RISC-V
assembly, define architecture parameters in a config file, sweep across
configurations, and visualize the results.

## What It Models

- **ISA:** RV32I, M (multiply/divide), A (atomics with LR/SC), V (25-instruction
  vector subset), CSR access, machine-mode traps (MRET)
- **Pipeline:** 5-stage in-order (IF→ID→EX→MEM→WB) with 2 forwarding modes,
  5 branch predictors, and configurable mispredict penalty
- **Cache:** Up to 3 levels (L1/L2/L3), with configurable size, associativity,
  line size, 5 replacement policies, write-back/write-through, and (no-)write-allocate
- **Multi-core:** N pipelined cores, directory-based MESI coherence, private L1
  with shared L2/L3
- **Devices:** PLIC interrupt controller, CLINT timer, DMA-based NIC with
  interrupt coalescing and RSS

## Building

```bash
# Requires a C++20 compiler.
mkdir build
cmake --build build -j
cd build
```

Produces:
- `rvsim` — research platform (assemble → run → sweep → CSV)

## Usage

**Run a program with default architecture:**
```bash
./rvsim examples/sum_to_n.s
```

**Run with a config file:**
```bash
./rvsim examples/matmul.s --config examples/matmul.cfg
```

**Sweep and visualize:**
```bash
./rvsim examples/bubble_sort.s --config examples/bubble_sort.cfg > results/bubble_sort.csv
pip install -r tools/requirements.txt
python3 tools/visualize.py results/bubble_sort.csv --output plots/
```

## Tests

```bash
cd build
./rvsim_tests
```

## Documentation

See the [docs/](docs/) directory:

- [Architecture](docs/architecture.rst) — what the simulator models
- [Configuration](docs/config.rst) — config file reference
- [Research Platform](docs/research_platform.rst) — how to run sweeps
- [Future Work](docs/future_work.rst) — planned improvements

