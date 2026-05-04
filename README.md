# leanDAS

A minimal Rust implementation of the **STARS** construction — a Reed–Solomon proximity-test-free DAS scheme that runs both the random linear combination (RLC) across blobs **and** FRI folding *inside* the leanVM trace, so the ZKVM verifies every fold step rather than spot-checking. This upgrades FRI's proximity guarantee to **exact** RS membership.

For the construction itself, the threat model, and the soundness argument, see [`paper/short-note.md`](paper/short-note.md).

## What's in this repo

A single Rust workspace under [`rust/`](rust/) with four crates:

| Crate | Role |
|---|---|
| `das-core` | RLC, fold step, FRI fold chain + constant assertion, polynomial / Merkle utilities |
| `das-prover` | Prover entry, batch + aggregation, leanVM circuit compilation (Python DSL → bytecode) |
| `das-verifier` | Verifier |
| `das-cli` | CLI driver + benchmark harness |

The leanVM toolchain itself comes from upstream [`leanEthereum/leanMultisig`](https://github.com/leanEthereum/leanMultisig), pinned by SHA in [`rust/Cargo.toml`](rust/Cargo.toml).

## Build

```sh
cd rust
cargo build --release --workspace
```

A Python interpreter must be available on `PATH` — the prover compiles leanVM bytecode from a Python DSL at runtime via files in `rust/crates/das-prover/`.

## Run

Reference mode (no STARK proof, validates the construction):

```sh
cargo run --release --bin leandas -- -m 4 -n 32
```

Full zkVM mode (STARK proof + verification):

```sh
cargo run --release --bin leandas -- -m 4 -n 32 --zkvm
```

CLI flags:

```
-m, --codewords <N>     Number of codewords (default: 4)
-n, --length <N>        Codeword length, must be power of 2 (default: 32)
-b, --batch-size <N>    Codewords per batch (default: all in one batch)
    --zkvm              Enable ZKVM proving (default: reference mode)
    --hybrid            Run hybrid recursive aggregation demo
    --spot-check        Use RLC hint + spot-check mode (faster proving)
-h, --help              Print this help
```

## Reproduce the paper benchmark

The headline number from `paper/short-note.md` § Benchmarks is **5.28 s prove time, 364 KB/s message throughput, 356 KB proof** on a Apple M2 Max, at the sweet-spot configuration `n = 4096`, `m = 240`, half-rate:

```sh
cargo run --release --bin leandas -- -m 240 -n 4096 --zkvm
```

## Provenance

This repo is a minimal extract of the leanDAS research monorepo at commit **`bc2c418`** ("add rlc hints", 2026-03-28) — the last commit before a leanVM-fork dependency and a parallel Plonky3 path were introduced. Everything stripped (the Lean 4 formal spec, the LaTeX paper sources, peripheral docs) lives in the source repo.

The leanMultisig dependency is pinned to **`df0807f7b1d670f9fca44fc0a7431cb5dfe9c8b7`** — the SHA the `bc2c418` `Cargo.lock` was already resolved against, so the build matches what the paper benchmarks were measured on.

## Known issue at the base commit

`cargo test --workspace` reports 13 passes and 4 failures. The four failing tests (`test_compile_unified_circuit`, `test_prepare_private_input`, `test_prove_unified_leaf_mode`, `test_prove_unified_recursion_end_to_end`) all reference a `das_unified_main.py` file that isn't present in the `bc2c418` tree. This is a pre-existing inconsistency in the source code at the base commit, not caused by the MVP extraction. The "unified" code path is independent of the MVP critical path; the **hybrid** path (which the paper benchmarks measure and the CLI uses by default) is exercised by passing tests and runs cleanly end-to-end.
