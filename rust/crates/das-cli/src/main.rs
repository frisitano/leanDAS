use std::time::Instant;

use backend::PrimeCharacteristicRing;
use das_core::commitment::{commit_codeword_with_tree_epl, open_position_epl};
use das_core::types::{Codeword, F, MIN_EVALS_PER_LEAF, compute_evals_per_leaf};
use das_prover::pipeline::{prove_das_with_bytecode, DasConfig};
use das_prover::circuit::{compile_hybrid_das_circuit, prove_circuit, compile_spotcheck_circuit, prove_spotcheck_circuit, compute_num_spot_checks};
use das_prover::recursion::prove_hybrid_recursion;
use das_verifier::{verify_das_with_bytecode, VerifyResult};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mut num_codewords: usize = 4;
    let mut codeword_len: usize = 32;
    let mut batch_size: Option<usize> = None;
    let mut zkvm = false;
    let mut num_openings: usize = 4;
    let mut epl_override: Option<usize> = None;
    let mut parallel: usize = 0;
    let mut hybrid = false;
    let mut num_inner_proofs: usize = 4;
    let mut spot_check = false;
    let mut spot_check_k: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--codewords" | "-m" => {
                i += 1;
                num_codewords = args[i].parse().expect("invalid --codewords value");
            }
            "--length" | "-n" => {
                i += 1;
                codeword_len = args[i].parse().expect("invalid --length value");
            }
            "--batch-size" | "-b" => {
                i += 1;
                batch_size = Some(args[i].parse().expect("invalid --batch-size value"));
            }
            "--epl" => {
                i += 1;
                epl_override = Some(args[i].parse().expect("invalid --epl value"));
            }
            "--zkvm" => {
                zkvm = true;
            }
            "--parallel" => {
                i += 1;
                parallel = args[i].parse().expect("invalid --parallel value");
            }
            "--openings" => {
                i += 1;
                num_openings = args[i].parse().expect("invalid --openings value");
            }
            "--hybrid" => {
                hybrid = true;
            }
            "--inner-proofs" => {
                i += 1;
                num_inner_proofs = args[i].parse().expect("invalid --inner-proofs value");
            }
            "--spot-check" => {
                spot_check = true;
            }
            "-k" => {
                i += 1;
                spot_check_k = Some(args[i].parse().expect("invalid -k value"));
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if hybrid {
        let epl = epl_override.unwrap_or_else(|| compute_evals_per_leaf(codeword_len));
        let bs = batch_size.unwrap_or(1);
        run_hybrid(num_inner_proofs, codeword_len, epl, bs);
        return;
    }

    if spot_check {
        let epl = epl_override.unwrap_or_else(|| compute_evals_per_leaf(codeword_len));
        let bs = batch_size.unwrap_or(num_codewords);
        run_spot_check(num_codewords, codeword_len, epl, bs, spot_check_k);
        return;
    }

    assert!(
        codeword_len.is_power_of_two(),
        "codeword length must be a power of 2"
    );
    assert!(
        codeword_len >= MIN_EVALS_PER_LEAF,
        "codeword length must be >= {MIN_EVALS_PER_LEAF}"
    );

    let batch_size = batch_size.unwrap_or(num_codewords);
    let epl = epl_override.unwrap_or_else(|| compute_evals_per_leaf(codeword_len));

    println!("╔══════════════════════════════════════╗");
    println!("║           leanDAS pipeline           ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  codewords       : {num_codewords:<17} ║");
    println!("║  codeword length : {codeword_len:<17} ║");
    println!("║  batch size      : {batch_size:<17} ║");
    println!("║  evals per leaf  : {epl:<17} ║");
    println!("║  mode            : {:<17} ║", if zkvm { "zkvm" } else { "reference" });
    println!("║  openings        : {num_openings:<17} ║");
    println!("╚══════════════════════════════════════╝");
    println!();

    // Generate codewords.
    let t0 = Instant::now();
    // Use constant codewords (degree 0 polynomials) — always valid RS.
    let codewords: Vec<Codeword> = (0..num_codewords)
        .map(|i| {
            let c = F::from_usize(i + 1);
            let evals = vec![c; codeword_len];
            Codeword::new(evals)
        })
        .collect();
    let gen_time = t0.elapsed();
    println!("  codeword generation    : {:>10.2?}", gen_time);

    // Build Merkle trees (needed for openings).
    let t1 = Instant::now();
    let trees: Vec<_> = codewords
        .iter()
        .map(|cw| commit_codeword_with_tree_epl(cw, epl))
        .collect();
    let commit_time = t1.elapsed();
    println!("  merkle commitment      : {:>10.2?}", commit_time);

    // Compile circuit (if ZKVM mode).
    let degree = codeword_len / 2; // default: half-rate RS
    let config = DasConfig {
        batch_size,
        zkvm_mode: zkvm,
        evals_per_leaf: Some(epl),
        degree,
        parallel,
    };
    let num_batches = (num_codewords + batch_size - 1) / batch_size;
    let actual_batch_size = batch_size.min(num_codewords);

    let (bytecode, compile_time) = if zkvm {
        let tc = Instant::now();
        let bc = das_prover::circuit::compile_das_circuit(actual_batch_size, codeword_len, epl, degree);
        let elapsed = tc.elapsed();
        println!("  circuit compilation    : {:>10.2?}  ({} instructions)", elapsed, bc.instructions.len());
        (Some(bc), elapsed)
    } else {
        (None, std::time::Duration::ZERO)
    };

    // Prove.
    let t2 = Instant::now();
    let verifier_bytecode = bytecode.clone();
    let das_proof = prove_das_with_bytecode(codewords.clone(), &config, bytecode);
    let prove_time = t2.elapsed();
    println!("  proving ({num_batches} batches)     : {:>10.2?}", prove_time);

    // Stats on the proof.
    let num_batch_proofs = das_proof.aggregated_proof.batch_proofs.len();
    let has_stark = das_proof
        .aggregated_proof
        .batch_proofs
        .iter()
        .any(|bp| bp.proof.is_some());

    // Proof size stats.
    let proof_size_fe: usize = das_proof
        .aggregated_proof
        .batch_proofs
        .iter()
        .filter_map(|bp| bp.proof.as_ref())
        .map(|p| p.proof_size_fe())
        .sum();
    if proof_size_fe > 0 {
        let proof_size_kb = (proof_size_fe * 4) as f64 / 1024.0;
        println!("  proof size             : {:>7} FE ({:.1} KB)", proof_size_fe, proof_size_kb);
    }

    // Generate openings.
    let actual_openings = num_openings.min(num_codewords * (codeword_len / epl));
    let t3 = Instant::now();
    let openings: Vec<_> = (0..actual_openings)
        .map(|i| {
            let cw_idx = i % num_codewords;
            let pos = (i * 7) % codeword_len; // spread positions
            (cw_idx, open_position_epl(&codewords[cw_idx], &trees[cw_idx].1, pos, epl))
        })
        .collect();
    let open_time = t3.elapsed();
    println!("  opening generation     : {:>10.2?}", open_time);

    // Verify.
    let t4 = Instant::now();
    let result = verify_das_with_bytecode(&das_proof, &openings, verifier_bytecode);
    let verify_time = t4.elapsed();
    println!("  verification           : {:>10.2?}", verify_time);

    // Summary.
    let total_time = gen_time + commit_time + compile_time + prove_time + open_time + verify_time;
    println!();
    println!("┌──────────────────────────────────────┐");
    println!("│              summary                 │");
    println!("├──────────────────────────────────────┤");
    println!("│  total time      : {:>10.2?}       │", total_time);
    println!(
        "│  result          : {:<17} │",
        match &result {
            VerifyResult::Accept => "ACCEPT".to_string(),
            VerifyResult::RejectProof { batch_index } =>
                format!("REJECT (batch {batch_index})"),
            VerifyResult::RejectOpening {
                codeword_index,
                position,
            } => format!("REJECT (cw {codeword_index} pos {position})"),
        }
    );
    println!("│  batch proofs    : {:<17} │", num_batch_proofs);
    println!(
        "│  stark proofs    : {:<17} │",
        if has_stark { "yes" } else { "no" }
    );
    if proof_size_fe > 0 {
        let proof_size_kb = (proof_size_fe * 4) as f64 / 1024.0;
        println!("│  proof size      : {:<17} │", format!("{:.1} KB", proof_size_kb));
    }
    println!(
        "│  openings checked: {:<17} │",
        actual_openings
    );
    if zkvm && prove_time.as_secs_f64() > 0.0 {
        let degree = codeword_len / 2;
        let systematic_bytes = (num_codewords * degree * 4) as f64;
        let full_bytes = (num_codewords * codeword_len * 4) as f64;
        let prove_secs = prove_time.as_secs_f64();
        println!("│  throughput (sys): {:>7.1} KB/s │", systematic_bytes / 1024.0 / prove_secs);
        println!("│  throughput (full):{:>7.1} KB/s │", full_bytes / 1024.0 / prove_secs);
    }
    println!("└──────────────────────────────────────┘");

    if !matches!(result, VerifyResult::Accept) {
        std::process::exit(1);
    }
}

fn run_hybrid(num_inner_proofs: usize, codeword_len: usize, epl: usize, batch_size: usize) {
    use das_core::{commit_codeword_epl, derive_challenge};
    use lean_prover::verify_execution::verify_execution;

    let degree = codeword_len / 2;
    let log_inv_rate = 1;

    println!("╔══════════════════════════════════════════════╗");
    println!("║         leanDAS hybrid aggregation           ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  inner proofs    : {:<25} ║", num_inner_proofs);
    println!("║  codeword length : {:<25} ║", codeword_len);
    println!("║  batch size      : {:<25} ║", batch_size);
    println!("║  degree          : {:<25} ║", degree);
    println!("║  evals per leaf  : {:<25} ║", epl);
    println!("╚══════════════════════════════════════════════╝");
    println!();

    // Step 1: Compile both circuits.
    let t0 = Instant::now();
    let (inner_bytecode, hybrid_bytecode) =
        compile_hybrid_das_circuit(batch_size, codeword_len, epl, degree);
    let compile_time = t0.elapsed();
    println!("  compilation            : {:>10.2?}", compile_time);
    println!("    inner circuit        : {:>6} instructions (log_size={})",
        inner_bytecode.instructions.len(), inner_bytecode.log_size());
    println!("    hybrid circuit       : {:>6} instructions (log_size={})",
        hybrid_bytecode.instructions.len(), hybrid_bytecode.log_size());
    println!();

    // Step 2: Prove inner batches.
    println!("  --- inner batch proving ({} proofs) ---", num_inner_proofs);
    let mut inner_proofs = Vec::new();
    let t_inner_total = Instant::now();

    for batch_idx in 0..num_inner_proofs {
        let codewords: Vec<Codeword> = (0..batch_size)
            .map(|i| {
                let c = F::from_usize(batch_idx * batch_size + i + 1);
                Codeword::new(vec![c; codeword_len])
            })
            .collect();
        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword_epl(cw, epl)).collect();
        let challenges = commitments.iter().map(derive_challenge).collect();
        let batch = das_core::types::Batch {
            indices: (0..batch_size).collect(),
            codewords,
            commitments,
            challenges,
        };

        let t_prove = Instant::now();
        let (proof, pub_input) = prove_circuit(&inner_bytecode, &batch, degree);
        let prove_time = t_prove.elapsed();

        let t_verify = Instant::now();
        verify_execution(&inner_bytecode, &pub_input, proof.clone())
            .expect("inner proof must verify");
        let verify_time = t_verify.elapsed();

        let proof_kb = (proof.proof_size_fe() * 4) as f64 / 1024.0;
        println!("    batch {:>2}: prove={:>8.2?}  verify={:>8.2?}  proof={:.1} KB",
            batch_idx, prove_time, verify_time, proof_kb);

        inner_proofs.push((pub_input, proof));
    }
    let inner_total = t_inner_total.elapsed();
    println!("    total                : {:>10.2?}", inner_total);
    println!();

    // Step 3: Hybrid aggregation.
    println!("  --- hybrid aggregation ({} inner proofs -> 1 proof) ---", num_inner_proofs);
    let t_agg = Instant::now();
    let (rec_proof, rec_pub_input) = prove_hybrid_recursion(
        &[],
        &inner_proofs,
        &hybrid_bytecode,
        &inner_bytecode,
        log_inv_rate,
    );
    let agg_prove_time = t_agg.elapsed();

    let t_agg_verify = Instant::now();
    verify_execution(&hybrid_bytecode, &rec_pub_input, rec_proof.clone())
        .expect("hybrid proof must verify");
    let agg_verify_time = t_agg_verify.elapsed();

    let agg_proof_kb = (rec_proof.proof_size_fe() * 4) as f64 / 1024.0;
    println!("    prove                : {:>10.2?}", agg_prove_time);
    println!("    verify               : {:>10.2?}", agg_verify_time);
    println!("    proof size           : {:>7} FE ({:.1} KB)", rec_proof.proof_size_fe(), agg_proof_kb);
    println!("    pub input            : {:>7} elements", rec_pub_input.len());
    println!();

    // Throughput calculation.
    // Total codewords proved = num_inner_proofs * batch_size
    // Each codeword has codeword_len base-field elements (4 bytes each).
    // Systematic part = degree elements per codeword (half-rate: degree = codeword_len/2).
    let total_codewords = num_inner_proofs * batch_size;
    let prove_secs = (inner_total + agg_prove_time).as_secs_f64();
    let systematic_bytes = (total_codewords * degree * 4) as f64;
    let full_bytes = (total_codewords * codeword_len * 4) as f64;
    let systematic_kbps = systematic_bytes / 1024.0 / prove_secs;
    let full_kbps = full_bytes / 1024.0 / prove_secs;

    // Summary.
    let total = compile_time + inner_total + agg_prove_time + agg_verify_time;
    println!("┌──────────────────────────────────────────────┐");
    println!("│                  summary                     │");
    println!("├──────────────────────────────────────────────┤");
    println!("│  total time      : {:>10.2?}               │", total);
    println!("│  compilation     : {:>10.2?}               │", compile_time);
    println!("│  inner proving   : {:>10.2?}               │", inner_total);
    println!("│  aggregation     : {:>10.2?}               │", agg_prove_time);
    println!("│  agg verify      : {:>10.2?}               │", agg_verify_time);
    println!("│  agg proof size  : {:>10.1} KB             │", agg_proof_kb);
    println!("│  throughput (sys): {:>7.1} KB/s             │", systematic_kbps);
    println!("│  throughput (full):{:>7.1} KB/s             │", full_kbps);
    println!("│  status          : VERIFIED                  │");
    println!("└──────────────────────────────────────────────┘");
}

fn run_spot_check(num_codewords: usize, codeword_len: usize, epl: usize, batch_size: usize, k_override: Option<usize>) {
    use das_core::{commit_codeword_epl, derive_challenge};
    use lean_prover::verify_execution::verify_execution;

    let degree = codeword_len / 2;
    let k = k_override.unwrap_or_else(|| compute_num_spot_checks(codeword_len, degree, 128));

    println!("╔══════════════════════════════════════╗");
    println!("║       leanDAS spot-check mode        ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  codewords       : {:<17} ║", num_codewords);
    println!("║  codeword length : {:<17} ║", codeword_len);
    println!("║  batch size      : {:<17} ║", batch_size);
    println!("║  degree          : {:<17} ║", degree);
    println!("║  evals per leaf  : {:<17} ║", epl);
    println!("║  spot-checks (k) : {:<17} ║", k);
    println!("║  rate            : 1/{:<14} ║", codeword_len / degree);
    println!("╚══════════════════════════════════════╝");
    println!();

    // Generate codewords (at least batch_size).
    let actual_codewords = num_codewords.max(batch_size);
    let t0 = Instant::now();
    let codewords: Vec<Codeword> = (0..actual_codewords)
        .map(|i| {
            let c = F::from_usize(i + 1);
            Codeword::new(vec![c; codeword_len])
        })
        .collect();
    let gen_time = t0.elapsed();
    println!("  codeword generation    : {:>10.2?}", gen_time);

    // Commit.
    let t1 = Instant::now();
    let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword_epl(cw, epl)).collect();
    let challenges: Vec<_> = commitments.iter().map(derive_challenge).collect();
    let commit_time = t1.elapsed();
    println!("  merkle commitment      : {:>10.2?}", commit_time);

    // Compile spot-check circuit.
    let tc = Instant::now();
    let bytecode = compile_spotcheck_circuit(batch_size, codeword_len, epl, degree, k);
    let compile_time = tc.elapsed();
    println!("  circuit compilation    : {:>10.2?}  ({} instructions)", compile_time, bytecode.instructions.len());

    // Build batch and prove.
    let batch = das_core::types::Batch {
        indices: (0..batch_size).collect(),
        codewords: codewords[..batch_size].to_vec(),
        commitments: commitments[..batch_size].to_vec(),
        challenges: challenges[..batch_size].to_vec(),
    };

    let t2 = Instant::now();
    let (proof, pub_input) = prove_spotcheck_circuit(&bytecode, &batch, degree);
    let prove_time = t2.elapsed();
    println!("  proving                : {:>10.2?}", prove_time);

    let proof_kb = (proof.proof_size_fe() * 4) as f64 / 1024.0;
    println!("  proof size             : {:>7} FE ({:.1} KB)", proof.proof_size_fe(), proof_kb);

    // Verify.
    let t3 = Instant::now();
    verify_execution(&bytecode, &pub_input, proof).expect("spot-check proof must verify");
    let verify_time = t3.elapsed();
    println!("  verification           : {:>10.2?}", verify_time);

    // Summary.
    let total = gen_time + commit_time + compile_time + prove_time + verify_time;
    let prove_secs = prove_time.as_secs_f64();
    let systematic_bytes = (batch_size * degree * 4) as f64;
    let full_bytes = (batch_size * codeword_len * 4) as f64;

    println!();
    println!("┌──────────────────────────────────────┐");
    println!("│              summary                 │");
    println!("├──────────────────────────────────────┤");
    println!("│  total time      : {:>10.2?}       │", total);
    println!("│  compilation     : {:>10.2?}       │", compile_time);
    println!("│  proving         : {:>10.2?}       │", prove_time);
    println!("│  verification    : {:>10.2?}       │", verify_time);
    println!("│  proof size      : {:>10.1} KB    │", proof_kb);
    println!("│  spot-checks     : {:>10}         │", k);
    if prove_secs > 0.0 {
        println!("│  throughput (sys): {:>7.1} KB/s │", systematic_bytes / 1024.0 / prove_secs);
        println!("│  throughput (full):{:>7.1} KB/s │", full_bytes / 1024.0 / prove_secs);
    }
    println!("│  status          : VERIFIED          │");
    println!("└──────────────────────────────────────┘");
}

fn print_usage() {
    println!("Usage: leandas [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -m, --codewords <N>   Number of codewords (default: 4)");
    println!("  -n, --length <N>      Codeword length, must be power of 2 (default: 32)");
    println!("  -b, --batch-size <N>  Codewords per batch (default: all in one batch)");
    println!("      --zkvm            Enable ZKVM proving (default: reference mode)");
    println!("      --epl <N>         Evaluations per Merkle leaf (default: auto)");
    println!("      --openings <N>    Number of openings to verify (default: 4)");
    println!("      --parallel <N>    Prove N batches concurrently (0 = sequential)");
    println!("      --hybrid          Run hybrid recursive aggregation demo");
    println!("      --inner-proofs <N> Number of inner proofs to aggregate (default: 4)");
    println!("      --spot-check      Use RLC hint + spot-check mode (faster proving)");
    println!("  -k <N>                Number of spot-checks (default: auto for 128-bit security)");
    println!("  -h, --help            Print this help");
}
