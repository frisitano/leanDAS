//! leanDAS benchmark suite.
//!
//! Measures compile, prove, and verify times across a matrix of parameters.
//! Run with: cargo run --release --bin leandas-bench

use std::collections::HashMap;
use std::time::Instant;

use backend::PrimeCharacteristicRing;
use das_core::commitment::{commit_codeword_with_tree_epl, open_position_epl};
use das_core::polynomial::rs_encode;
use das_core::types::{Codeword, F, compute_evals_per_leaf};
use das_prover::circuit::{compile_das_circuit, compile_hybrid_das_circuit, compile_unified_das_circuit, prove_circuit, prove_unified_leaf};
use das_prover::pipeline::{prove_das_with_bytecode, DasConfig, DasProof};
use das_prover::recursion::{prove_hybrid_recursion, prove_unified_recursion};
use das_verifier::verify_das_with_bytecode;

/// Cache key: (batch_size, codeword_len, epl, degree).
type CircuitKey = (usize, usize, usize, usize);

/// A single benchmark scenario.
struct Scenario {
    name: &'static str,
    num_codewords: usize,
    codeword_len: usize,
    batch_size: usize,
    zkvm: bool,
    degree: usize, // 0 = constant polynomials
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let scenarios = vec![
        // --- Scaling codeword length (single codeword, half-rate RS) ---
        Scenario { name: "1×16 deg4",     num_codewords: 1, codeword_len: 16,   batch_size: 1, zkvm: true, degree: 4 },
        Scenario { name: "1×32 deg8",     num_codewords: 1, codeword_len: 32,   batch_size: 1, zkvm: true, degree: 8 },
        Scenario { name: "1×64 deg16",    num_codewords: 1, codeword_len: 64,   batch_size: 1, zkvm: true, degree: 16 },
        Scenario { name: "1×128 deg32",   num_codewords: 1, codeword_len: 128,  batch_size: 1, zkvm: true, degree: 32 },
        Scenario { name: "1×256 deg64",   num_codewords: 1, codeword_len: 256,  batch_size: 1, zkvm: true, degree: 64 },
        Scenario { name: "1×512 deg128",  num_codewords: 1, codeword_len: 512,  batch_size: 1, zkvm: true, degree: 128 },
        Scenario { name: "1×1024 deg256",  num_codewords: 1, codeword_len: 1024,  batch_size: 1, zkvm: true, degree: 256 },
        Scenario { name: "1×4096 deg1024", num_codewords: 1, codeword_len: 4096,  batch_size: 1, zkvm: true, degree: 1024 },
        Scenario { name: "1×16384 deg4096", num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 4096 },
        Scenario { name: "1×65536 deg16384", num_codewords: 1, codeword_len: 65536, batch_size: 1, zkvm: true, degree: 16384 },

        // --- Scaling batch size (n=64, deg16) ---
        Scenario { name: "1×64 b1 deg16", num_codewords: 1, codeword_len: 64, batch_size: 1, zkvm: true, degree: 16 },
        Scenario { name: "2×64 b2 deg16", num_codewords: 2, codeword_len: 64, batch_size: 2, zkvm: true, degree: 16 },
        Scenario { name: "4×64 b4 deg16", num_codewords: 4, codeword_len: 64, batch_size: 4, zkvm: true, degree: 16 },
        Scenario { name: "8×64 b8 deg16", num_codewords: 8, codeword_len: 64, batch_size: 8, zkvm: true, degree: 16 },

        // --- Scaling batch size (n=128, deg32) ---
        Scenario { name: "1×128 b1 deg32",  num_codewords: 1,  codeword_len: 128, batch_size: 1,  zkvm: true, degree: 32 },
        Scenario { name: "2×128 b2 deg32",  num_codewords: 2,  codeword_len: 128, batch_size: 2,  zkvm: true, degree: 32 },
        Scenario { name: "4×128 b4 deg32",  num_codewords: 4,  codeword_len: 128, batch_size: 4,  zkvm: true, degree: 32 },
        Scenario { name: "8×128 b8 deg32",  num_codewords: 8,  codeword_len: 128, batch_size: 8,  zkvm: true, degree: 32 },

        // --- Scaling batch size (n=256, deg64) ---
        Scenario { name: "1×256 b1 deg64",  num_codewords: 1,  codeword_len: 256, batch_size: 1,  zkvm: true, degree: 64 },
        Scenario { name: "4×256 b4 deg64",  num_codewords: 4,  codeword_len: 256, batch_size: 4,  zkvm: true, degree: 64 },
        Scenario { name: "8×256 b4 deg64",  num_codewords: 8,  codeword_len: 256, batch_size: 4,  zkvm: true, degree: 64 },

        // --- Degree scaling: same n, varying degree ---
        Scenario { name: "1×16384 deg4",    num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 4 },
        Scenario { name: "1×16384 deg16",   num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 16 },
        Scenario { name: "1×16384 deg64",   num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 64 },
        Scenario { name: "1×16384 deg256",  num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 256 },
        Scenario { name: "1×16384 deg1024", num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 1024 },
        Scenario { name: "1×16384 deg4096", num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 4096 },
        Scenario { name: "1×16384 deg8192", num_codewords: 1, codeword_len: 16384, batch_size: 1, zkvm: true, degree: 8192 },

        // --- Multi-batch (batch_size < num_codewords) ---
        Scenario { name: "8×64 b2 deg16",  num_codewords: 8, codeword_len: 64,  batch_size: 2, zkvm: true, degree: 16 },
        Scenario { name: "8×128 b2 deg32", num_codewords: 8, codeword_len: 128, batch_size: 2, zkvm: true, degree: 32 },
    ];

    // --- Deep recursion comparison benchmarks ---
    let deep_scenarios = vec![
        // -- 1 level of recursion, scaling codeword size --
        DeepScenario { name: "L1 2×(1×128 d32)",    num_leaves: 2, batch_size: 1, codeword_len: 128,  degree: 32,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(1×256 d64)",    num_leaves: 2, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(1×512 d128)",   num_leaves: 2, batch_size: 1, codeword_len: 512,  degree: 128, rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(1×1024 d256)",  num_leaves: 2, batch_size: 1, codeword_len: 1024, degree: 256, rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(1×4096 d1024)", num_leaves: 2, batch_size: 1, codeword_len: 4096, degree: 1024, rec_fan_in: 2 },

        // -- 1 level, bigger batches --
        DeepScenario { name: "L1 2×(2×256 d64)",    num_leaves: 2, batch_size: 2, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(4×256 d64)",    num_leaves: 2, batch_size: 4, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },

        // -- 2 levels of recursion (4 leaves -> 2 rec -> 1 rec) --
        DeepScenario { name: "L2 4×(1×128 d32)",    num_leaves: 4, batch_size: 1, codeword_len: 128,  degree: 32,  rec_fan_in: 2 },
        DeepScenario { name: "L2 4×(1×256 d64)",    num_leaves: 4, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },
        DeepScenario { name: "L2 4×(1×1024 d256)",  num_leaves: 4, batch_size: 1, codeword_len: 1024, degree: 256, rec_fan_in: 2 },

        // -- 3 levels (8 leaves -> 4 -> 2 -> 1) --
        DeepScenario { name: "L3 8×(1×128 d32)",    num_leaves: 8, batch_size: 1, codeword_len: 128,  degree: 32,  rec_fan_in: 2 },
        DeepScenario { name: "L3 8×(1×256 d64)",    num_leaves: 8, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },

        // -- Wide fan-in (4-way recursion, fewer levels) --
        DeepScenario { name: "L1 4×(1×256 d64) f4", num_leaves: 4, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 4 },
        DeepScenario { name: "L1 4×(1×1024 d256) f4", num_leaves: 4, batch_size: 1, codeword_len: 1024, degree: 256, rec_fan_in: 4 },
    ];

    // --- Hybrid-specific extended benchmarks ---
    let hybrid_extra_scenarios = vec![
        // -- Bigger batch sizes (more codewords per leaf proof) --
        DeepScenario { name: "L1 2×(4×128 d32)",       num_leaves: 2, batch_size: 4, codeword_len: 128,  degree: 32,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(8×128 d32)",       num_leaves: 2, batch_size: 8, codeword_len: 128,  degree: 32,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(4×256 d64)",       num_leaves: 2, batch_size: 4, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(8×256 d64)",       num_leaves: 2, batch_size: 8, codeword_len: 256,  degree: 64,  rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(4×512 d128)",      num_leaves: 2, batch_size: 4, codeword_len: 512,  degree: 128, rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(8×512 d128)",      num_leaves: 2, batch_size: 8, codeword_len: 512,  degree: 128, rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(4×1024 d256)",     num_leaves: 2, batch_size: 4, codeword_len: 1024, degree: 256, rec_fan_in: 2 },
        DeepScenario { name: "L1 2×(8×1024 d256)",     num_leaves: 2, batch_size: 8, codeword_len: 1024, degree: 256, rec_fan_in: 2 },

        // -- Wide fan-in (4-way and 8-way recursion) --
        DeepScenario { name: "L1 4×(1×128 d32) f4",    num_leaves: 4, batch_size: 1, codeword_len: 128,  degree: 32,  rec_fan_in: 4 },
        DeepScenario { name: "L1 8×(1×128 d32) f8",    num_leaves: 8, batch_size: 1, codeword_len: 128,  degree: 32,  rec_fan_in: 8 },
        DeepScenario { name: "L1 4×(1×512 d128) f4",   num_leaves: 4, batch_size: 1, codeword_len: 512,  degree: 128, rec_fan_in: 4 },
        DeepScenario { name: "L1 8×(1×512 d128) f8",   num_leaves: 8, batch_size: 1, codeword_len: 512,  degree: 128, rec_fan_in: 8 },
        DeepScenario { name: "L1 8×(1×256 d64) f8",    num_leaves: 8, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 8 },
        DeepScenario { name: "L1 8×(1×1024 d256) f8",  num_leaves: 8, batch_size: 1, codeword_len: 1024, degree: 256, rec_fan_in: 8 },

        // -- Wide fan-in + bigger batches --
        DeepScenario { name: "L1 4×(4×256 d64) f4",    num_leaves: 4, batch_size: 4, codeword_len: 256,  degree: 64,  rec_fan_in: 4 },
        DeepScenario { name: "L1 8×(4×256 d64) f8",    num_leaves: 8, batch_size: 4, codeword_len: 256,  degree: 64,  rec_fan_in: 8 },
        DeepScenario { name: "L1 4×(4×1024 d256) f4",  num_leaves: 4, batch_size: 4, codeword_len: 1024, degree: 256, rec_fan_in: 4 },

        // -- Multi-level with wide fan-in --
        DeepScenario { name: "L2 16×(1×256 d64) f4",   num_leaves: 16, batch_size: 1, codeword_len: 256,  degree: 64,  rec_fan_in: 4 },
        DeepScenario { name: "L2 8×(4×256 d64) f4",    num_leaves: 8,  batch_size: 4, codeword_len: 256,  degree: 64,  rec_fan_in: 4 },
        DeepScenario { name: "L2 16×(1×1024 d256) f4", num_leaves: 16, batch_size: 1, codeword_len: 1024, degree: 256, rec_fan_in: 4 },

        // -- 4MB rate-1/2 (n=2d) sweep: find optimal codeword size --
        // 4MB = 1,048,576 FE of data in degree, cl=2*degree
        // bs=8: data/batch = 8*d FE

        // d=512 cl=1024: N=256 batches (16KB/batch)
        DeepScenario { name: "4M 8×(8×1024 d512) f8",    num_leaves: 256, batch_size: 8, codeword_len: 1024,  degree: 512,   rec_fan_in: 8 },
        DeepScenario { name: "4M 8×(8×1024 d512) f4",    num_leaves: 256, batch_size: 8, codeword_len: 1024,  degree: 512,   rec_fan_in: 4 },

        // d=1024 cl=2048: N=128 batches (32KB/batch)
        DeepScenario { name: "4M 8×(8×2048 d1024) f8",   num_leaves: 128, batch_size: 8, codeword_len: 2048,  degree: 1024,  rec_fan_in: 8 },
        DeepScenario { name: "4M 8×(8×2048 d1024) f4",   num_leaves: 128, batch_size: 8, codeword_len: 2048,  degree: 1024,  rec_fan_in: 4 },

        // d=2048 cl=4096: N=64 batches (64KB/batch)
        DeepScenario { name: "4M 8×(8×4096 d2048) f8",   num_leaves: 64,  batch_size: 8, codeword_len: 4096,  degree: 2048,  rec_fan_in: 8 },
        DeepScenario { name: "4M 8×(8×4096 d2048) f4",   num_leaves: 64,  batch_size: 8, codeword_len: 4096,  degree: 2048,  rec_fan_in: 4 },

        // d=4096 cl=8192: N=32 batches (128KB/batch)
        DeepScenario { name: "4M 8×(8×8192 d4096) f8",   num_leaves: 32,  batch_size: 8, codeword_len: 8192,  degree: 4096,  rec_fan_in: 8 },
        DeepScenario { name: "4M 8×(8×8192 d4096) f4",   num_leaves: 32,  batch_size: 8, codeword_len: 8192,  degree: 4096,  rec_fan_in: 4 },

        // d=8192 cl=16384: N=16 batches (256KB/batch)
        DeepScenario { name: "4M 8×(8×16384 d8192) f8",  num_leaves: 16,  batch_size: 8, codeword_len: 16384, degree: 8192,  rec_fan_in: 8 },
        DeepScenario { name: "4M 8×(8×16384 d8192) f4",  num_leaves: 16,  batch_size: 8, codeword_len: 16384, degree: 8192,  rec_fan_in: 4 },

        // d=16384 cl=32768: N=8 batches (512KB/batch)
        DeepScenario { name: "4M 8×(8×32768 d16384) f8", num_leaves: 8,   batch_size: 8, codeword_len: 32768, degree: 16384, rec_fan_in: 8 },

        // Also try bs=4 for the larger codewords (less inner circuit overhead)
        DeepScenario { name: "4M 4×(4×4096 d2048) f8",   num_leaves: 128, batch_size: 4, codeword_len: 4096,  degree: 2048,  rec_fan_in: 8 },
        DeepScenario { name: "4M 4×(4×8192 d4096) f8",   num_leaves: 64,  batch_size: 4, codeword_len: 8192,  degree: 4096,  rec_fan_in: 8 },
        DeepScenario { name: "4M 4×(4×16384 d8192) f8",  num_leaves: 32,  batch_size: 4, codeword_len: 16384, degree: 8192,  rec_fan_in: 8 },
    ];

    // ── Pre-compile all circuits on a single large-stack thread ──────────

    // Collect unique keys for each circuit type.
    let mut das_keys: Vec<CircuitKey> = Vec::new();
    let mut unified_keys: Vec<CircuitKey> = Vec::new();
    let mut hybrid_keys: Vec<CircuitKey> = Vec::new();

    // Basic scenarios need DAS circuit bytecodes.
    for s in &scenarios {
        if !s.zkvm {
            continue;
        }
        let epl = compute_evals_per_leaf(s.codeword_len);
        let circuit_degree = if s.degree == 0 { s.codeword_len / 2 } else { s.degree.next_power_of_two() };
        let actual_batch_size = s.batch_size.min(s.num_codewords);
        let key = (actual_batch_size, s.codeword_len, epl, circuit_degree);
        if !das_keys.contains(&key) {
            das_keys.push(key);
        }
    }

    // Deep scenarios need DAS + unified + hybrid circuit bytecodes.
    for s in &deep_scenarios {
        let epl = compute_evals_per_leaf(s.codeword_len);
        let circuit_degree = s.degree.next_power_of_two();
        let key = (s.batch_size, s.codeword_len, epl, circuit_degree);
        if !das_keys.contains(&key) {
            das_keys.push(key);
        }
        if !unified_keys.contains(&key) {
            unified_keys.push(key);
        }
        if !hybrid_keys.contains(&key) {
            hybrid_keys.push(key);
        }
    }

    // Hybrid-extra scenarios need DAS (inner) + hybrid circuit bytecodes.
    for s in &hybrid_extra_scenarios {
        let epl = compute_evals_per_leaf(s.codeword_len);
        let circuit_degree = s.degree.next_power_of_two();
        let key = (s.batch_size, s.codeword_len, epl, circuit_degree);
        if !hybrid_keys.contains(&key) {
            hybrid_keys.push(key);
        }
    }

    println!("Pre-compiling circuits: {} DAS, {} unified, {} hybrid ...",
             das_keys.len(), unified_keys.len(), hybrid_keys.len());

    let das_keys_clone = das_keys.clone();
    let unified_keys_clone = unified_keys.clone();
    let hybrid_keys_clone = hybrid_keys.clone();

    let compile_t0 = Instant::now();
    let (das_cache, unified_cache, hybrid_cache) = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut das_map: HashMap<CircuitKey, lean_vm::Bytecode> = HashMap::new();
            for key in &das_keys_clone {
                let (bs, cl, ep, cd) = *key;
                println!("  compiling DAS circuit       bs={} cl={} epl={} deg={}", bs, cl, ep, cd);
                das_map.insert(*key, compile_das_circuit(bs, cl, ep, cd));
            }
            let mut uni_map: HashMap<CircuitKey, lean_vm::Bytecode> = HashMap::new();
            for key in &unified_keys_clone {
                let (bs, cl, ep, cd) = *key;
                println!("  compiling unified circuit    bs={} cl={} epl={} deg={}", bs, cl, ep, cd);
                uni_map.insert(*key, compile_unified_das_circuit(bs, cl, ep, cd));
            }
            let mut hyb_map: HashMap<CircuitKey, (lean_vm::Bytecode, lean_vm::Bytecode)> = HashMap::new();
            for key in &hybrid_keys_clone {
                let (bs, cl, ep, cd) = *key;
                println!("  compiling hybrid circuit     bs={} cl={} epl={} deg={}", bs, cl, ep, cd);
                hyb_map.insert(*key, compile_hybrid_das_circuit(bs, cl, ep, cd));
            }
            (das_map, uni_map, hyb_map)
        })
        .expect("failed to spawn compilation thread")
        .join()
        .expect("compilation thread panicked");
    let compile_total_ms = compile_t0.elapsed().as_secs_f64() * 1000.0;
    println!("All circuits compiled in {}", format_duration(compile_total_ms));

    // ── Basic benchmark scenarios ────────────────────────────────────────

    println!("╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              leanDAS benchmark suite                                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
             "scenario", "compile", "prove", "verify", "total", "proof FE", "instrs");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in &scenarios {
        let result = run_scenario(s, &das_cache);
        println!(
            "║ {:28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
            s.name,
            format_duration(result.compile_ms),
            format_duration(result.prove_ms),
            format_duration(result.verify_ms),
            format_duration(result.compile_ms + result.prove_ms + result.verify_ms),
            result.proof_size_fe,
            result.num_instructions,
        );
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ── Two-circuit deep recursion benchmarks ────────────────────────────

    println!();
    // ── Unified deep recursion benchmarks ────────────────────────────────

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         leanDAS unified circuit recursion (deep)                                                       ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
             "scenario", "compile", "leaf pv", "rec pv", "levels", "total", "proof FE", "instrs");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in &deep_scenarios {
        match run_deep_unified(s, &unified_cache) {
            Some(r) => {
                let total = r.compile_ms + r.leaf_prove_ms + r.rec_prove_ms;
                println!(
                    "║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
                    s.name,
                    format_duration(r.compile_ms),
                    format_duration(r.leaf_prove_ms),
                    format_duration(r.rec_prove_ms),
                    r.rec_levels,
                    format_duration(total),
                    r.final_proof_size_fe,
                    r.num_instructions,
                );
            }
            None => {
                println!("║ {:30} {:>87} ║", s.name, "SKIPPED");
            }
        }
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ── Hybrid deep recursion benchmarks ──────────────────────────────────

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         leanDAS hybrid circuit recursion (deep)                                                        ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
             "scenario", "leaf pv", "rec pv", "levels", "total", "proof FE", "inn instr", "hyb instr");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in &deep_scenarios {
        match run_deep_hybrid(s, &hybrid_cache) {
            Some(r) => {
                let total = r.leaf_prove_ms + r.rec_prove_ms;
                println!(
                    "║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
                    s.name,
                    format_duration(r.leaf_prove_ms),
                    format_duration(r.rec_prove_ms),
                    r.rec_levels,
                    format_duration(total),
                    r.final_proof_size_fe,
                    r.inner_num_instructions,
                    r.hybrid_num_instructions,
                );
            }
            None => {
                println!("║ {:30} {:>95} ║", s.name, "SKIPPED");
            }
        }
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ── Hybrid extended benchmarks (bigger batches + wider fan-in) ──────

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                 leanDAS hybrid extended (big batches + wide fan-in)                                                    ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
             "scenario", "leaf pv", "rec pv", "levels", "total", "proof FE", "inn instr", "hyb instr");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in &hybrid_extra_scenarios {
        match run_deep_hybrid(s, &hybrid_cache) {
            Some(r) => {
                let total = r.leaf_prove_ms + r.rec_prove_ms;
                println!(
                    "║ {:30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} ║",
                    s.name,
                    format_duration(r.leaf_prove_ms),
                    format_duration(r.rec_prove_ms),
                    r.rec_levels,
                    format_duration(total),
                    r.final_proof_size_fe,
                    r.inner_num_instructions,
                    r.hybrid_num_instructions,
                );
            }
            None => {
                println!("║ {:30} {:>95} ║", s.name, "SKIPPED");
            }
        }
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
}

struct BenchResult {
    compile_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    proof_size_fe: usize,
    num_instructions: usize,
}

fn run_scenario(s: &Scenario, das_cache: &HashMap<CircuitKey, lean_vm::Bytecode>) -> BenchResult {
    let epl = compute_evals_per_leaf(s.codeword_len);

    // Generate codewords.
    let codewords: Vec<Codeword> = if s.degree == 0 {
        (0..s.num_codewords)
            .map(|i| {
                let c = F::from_usize(i + 1);
                Codeword::new(vec![c; s.codeword_len])
            })
            .collect()
    } else {
        (0..s.num_codewords)
            .map(|i| {
                let msg: Vec<F> = (0..s.degree)
                    .map(|j| F::from_usize((i + 1) * (j + 1) + i * 7 + 3))
                    .collect();
                rs_encode(&msg, s.codeword_len)
            })
            .collect()
    };

    // Use next power of 2 for degree (circuit requires it).
    let circuit_degree = if s.degree == 0 { s.codeword_len / 2 } else { s.degree.next_power_of_two() };

    let config = DasConfig {
        batch_size: s.batch_size,
        zkvm_mode: s.zkvm,
        evals_per_leaf: Some(epl),
        degree: circuit_degree,
        parallel: 0,
    };

    let actual_batch_size = s.batch_size.min(s.num_codewords);

    // Look up pre-compiled bytecode from cache (compile time is 0 — already paid).
    let bytecode = if s.zkvm {
        let key = (actual_batch_size, s.codeword_len, epl, circuit_degree);
        Some(das_cache[&key].clone())
    } else {
        None
    };
    let compile_ms = 0.0;
    let num_instructions = bytecode.as_ref().map_or(0, |bc| bc.instructions.len());

    // Prove.
    let t1 = Instant::now();
    let verifier_bytecode = bytecode.clone();
    let das_proof: DasProof = prove_das_with_bytecode(codewords.clone(), &config, bytecode);
    let prove_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Proof size.
    let proof_size_fe: usize = das_proof
        .aggregated_proof
        .batch_proofs
        .iter()
        .filter_map(|bp| bp.proof.as_ref())
        .map(|p| p.proof_size_fe())
        .sum();

    // Generate openings for verification.
    let trees: Vec<_> = codewords
        .iter()
        .map(|cw| commit_codeword_with_tree_epl(cw, epl))
        .collect();
    let num_openings = 2.min(s.num_codewords * (s.codeword_len / epl));
    let openings: Vec<_> = (0..num_openings)
        .map(|i| {
            let cw_idx = i % s.num_codewords;
            let pos = (i * 7) % s.codeword_len;
            (cw_idx, open_position_epl(&codewords[cw_idx], &trees[cw_idx].1, pos, epl))
        })
        .collect();

    // Verify.
    let t2 = Instant::now();
    let result = verify_das_with_bytecode(&das_proof, &openings, verifier_bytecode);
    let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // Skip assertion — circuit may be modified for profiling.
    let _ = result;

    BenchResult {
        compile_ms,
        prove_ms,
        verify_ms,
        proof_size_fe,
        num_instructions,
    }
}

fn format_duration(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.1}µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.0}ms", ms)
    } else {
        format!("{:.2}s", ms / 1000.0)
    }
}

// --- Deep recursion benchmarks ---

struct DeepScenario {
    name: &'static str,
    num_leaves: usize,     // total leaf batches
    batch_size: usize,     // codewords per batch
    codeword_len: usize,
    degree: usize,
    rec_fan_in: usize,     // proofs aggregated per recursion step
}

struct DeepUnifiedResult {
    compile_ms: f64,
    leaf_prove_ms: f64,
    rec_prove_ms: f64,
    rec_levels: usize,
    final_proof_size_fe: usize,
    num_instructions: usize,
}

/// Generate test codewords for a scenario.
fn make_codewords(total: usize, codeword_len: usize, degree: usize) -> Vec<Codeword> {
    (0..total)
        .map(|i| {
            let msg: Vec<F> = (0..degree)
                .map(|j| F::from_usize((i + 1) * (j + 1) + i * 7 + 3))
                .collect();
            rs_encode(&msg, codeword_len)
        })
        .collect()
}

/// Build a Batch from a slice of codewords with a specific evals-per-leaf.
fn make_batch(codewords: &[Codeword], start: usize, epl: usize) -> das_core::types::Batch {
    let commitments: Vec<_> = codewords.iter()
        .map(|cw| das_core::commitment::commit_codeword_epl(cw, epl))
        .collect();
    let challenges = commitments.iter().map(|c| das_core::derive_challenge(c)).collect();
    das_core::types::Batch {
        indices: (start..start + codewords.len()).collect(),
        codewords: codewords.to_vec(),
        commitments,
        challenges,
    }
}

fn run_deep_unified(
    s: &DeepScenario,
    unified_cache: &HashMap<CircuitKey, lean_vm::Bytecode>,
) -> Option<DeepUnifiedResult> {
    let epl = compute_evals_per_leaf(s.codeword_len);
    let circuit_degree = s.degree.next_power_of_two();
    let total_codewords = s.num_leaves * s.batch_size;
    let codewords = make_codewords(total_codewords, s.codeword_len, s.degree);

    // Look up pre-compiled unified circuit.
    let key = (s.batch_size, s.codeword_len, epl, circuit_degree);
    let bytecode = unified_cache.get(&key)?.clone();
    let compile_ms = 0.0;
    let num_instructions = bytecode.instructions.len();

    // Prove leaf batches.
    let log_inv_rate = 1;
    let t1 = Instant::now();
    let mut leaf_proofs = Vec::new();
    for batch_idx in 0..s.num_leaves {
        let start = batch_idx * s.batch_size;
        let end = start + s.batch_size;
        let batch = make_batch(&codewords[start..end], start, epl);
        let (proof, pub_input) = prove_unified_leaf(&bytecode, &batch, log_inv_rate);
        leaf_proofs.push((pub_input, proof));
    }
    let leaf_prove_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Multi-level recursion — unified supports this natively.
    let t2 = Instant::now();
    let mut current_layer = leaf_proofs;
    let mut rec_levels = 0;

    while current_layer.len() > 1 {
        let mut next_layer = Vec::new();
        for chunk in current_layer.chunks(s.rec_fan_in) {
            let (proof, pub_input) = prove_unified_recursion(
                chunk,
                &bytecode,
                log_inv_rate,
            );
            next_layer.push((pub_input, proof));
        }
        current_layer = next_layer;
        rec_levels += 1;
    }
    let rec_prove_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let final_proof_size_fe = current_layer.last().map_or(0, |(_, p)| p.proof_size_fe());

    Some(DeepUnifiedResult {
        compile_ms,
        leaf_prove_ms,
        rec_prove_ms,
        rec_levels,
        final_proof_size_fe,
        num_instructions,
    })
}

struct DeepHybridResult {
    leaf_prove_ms: f64,
    rec_prove_ms: f64,
    rec_levels: usize,
    final_proof_size_fe: usize,
    inner_num_instructions: usize,
    hybrid_num_instructions: usize,
}

fn run_deep_hybrid(
    s: &DeepScenario,
    hybrid_cache: &HashMap<CircuitKey, (lean_vm::Bytecode, lean_vm::Bytecode)>,
) -> Option<DeepHybridResult> {
    let epl = compute_evals_per_leaf(s.codeword_len);
    let circuit_degree = s.degree.next_power_of_two();
    let total_codewords = s.num_leaves * s.batch_size;
    let codewords = make_codewords(total_codewords, s.codeword_len, s.degree);

    // Look up pre-compiled hybrid circuits (inner + hybrid).
    let key = (s.batch_size, s.codeword_len, epl, circuit_degree);
    let (inner_bytecode, hybrid_bytecode) = hybrid_cache.get(&key)?;
    let inner_num_instructions = inner_bytecode.instructions.len();
    let hybrid_num_instructions = hybrid_bytecode.instructions.len();

    // Prove leaf batches with the inner circuit.
    let log_inv_rate = 1;
    let t1 = Instant::now();
    let mut leaf_proofs = Vec::new();
    for batch_idx in 0..s.num_leaves {
        let start = batch_idx * s.batch_size;
        let end = start + s.batch_size;
        let batch = make_batch(&codewords[start..end], start, epl);
        let (proof, pub_input) = prove_circuit(inner_bytecode, &batch, circuit_degree);
        leaf_proofs.push((pub_input, proof));
    }
    let leaf_prove_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Multi-level recursion with hybrid circuit.
    // Level 1: hybrid verifies inner batch proofs.
    // Level 2+: hybrid verifies its own proofs (self-referential).
    let t2 = Instant::now();
    let mut current_layer: Vec<(Vec<F>, backend::Proof<F>)> = Vec::new();
    let mut rec_levels = 0;

    // First level: verify inner proofs.
    for chunk in leaf_proofs.chunks(s.rec_fan_in) {
        let (proof, pub_input) = prove_hybrid_recursion(
            &[],    // no self proofs
            chunk,  // inner batch proofs
            hybrid_bytecode,
            inner_bytecode,
            log_inv_rate,
        );
        current_layer.push((pub_input, proof));
    }
    rec_levels += 1;

    // Subsequent levels: hybrid verifies its own proofs (self-referential).
    while current_layer.len() > 1 {
        let mut next_layer = Vec::new();
        for chunk in current_layer.chunks(s.rec_fan_in) {
            let (proof, pub_input) = prove_hybrid_recursion(
                chunk,  // self proofs
                &[],    // no inner proofs
                hybrid_bytecode,
                inner_bytecode,
                log_inv_rate,
            );
            next_layer.push((pub_input, proof));
        }
        current_layer = next_layer;
        rec_levels += 1;
    }
    let rec_prove_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let final_proof_size_fe = current_layer.last().map_or(0, |(_, p)| p.proof_size_fe());

    Some(DeepHybridResult {
        leaf_prove_ms,
        rec_prove_ms,
        rec_levels,
        final_proof_size_fe,
        inner_num_instructions,
        hybrid_num_instructions,
    })
}
