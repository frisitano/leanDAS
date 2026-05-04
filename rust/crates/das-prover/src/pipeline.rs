//! End-to-end DAS proving pipeline.
//!
//! Orchestrates the two-phase architecture:
//! 1. Commit all codewords
//! 2. Partition into batches
//! 3. Prove each batch in parallel (RLC + FRI)
//! 4. Aggregate proofs into a single proof

use das_core::commitment::commit_codeword_epl;
use das_core::types::{Codeword, Commitment, compute_evals_per_leaf};
use rayon::prelude::*;

use crate::aggregation::{aggregate_proofs, AggregatedProof};
use crate::batch::{prepare_batch, prove_batch, prove_batch_zkvm, BatchProof};
use crate::circuit;

/// Configuration for the DAS proving pipeline.
pub struct DasConfig {
    /// Number of codewords per batch.
    pub batch_size: usize,
    /// Whether to generate ZKVM proofs (true) or run in reference mode (false).
    pub zkvm_mode: bool,
    /// Number of evaluations packed into each Merkle leaf.
    /// Must be >= 16, a multiple of 8, and divide the codeword length.
    /// If `None`, computed automatically from codeword length.
    pub evals_per_leaf: Option<usize>,
    /// Polynomial degree (number of coefficients / message length).
    /// Must be a power of 2 and <= codeword_len / 2.
    /// Controls the number of FRI fold rounds: log2(degree).
    pub degree: usize,
    /// Number of batches to prove concurrently (0 = sequential).
    /// Each concurrent batch gets `total_threads / parallel` threads.
    pub parallel: usize,
}

impl Default for DasConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            zkvm_mode: false,
            evals_per_leaf: None,
            degree: 1, // default: constant polynomials
            parallel: 0,
        }
    }
}

/// Result of the DAS proving pipeline.
pub struct DasProof {
    /// Commitments to all m input codewords.
    pub commitments: Vec<Commitment>,
    /// The single aggregated proof certifying all codewords are valid RS.
    pub aggregated_proof: AggregatedProof,
    /// Batch size used during proving (needed for verification).
    pub batch_size: usize,
    /// Codeword length (needed for verification).
    pub codeword_length: usize,
    /// Evaluations per Merkle leaf used during proving (needed for verification).
    pub evals_per_leaf: usize,
    /// Polynomial degree used during proving.
    pub degree: usize,
}

/// Run the full DAS proving pipeline.
///
/// Given m input codewords, produces commitments and a single
/// aggregated proof certifying that every codeword is a valid
/// RS codeword.
pub fn prove_das(codewords: Vec<Codeword>, config: &DasConfig) -> DasProof {
    let n = codewords[0].len();
    let epl = config.evals_per_leaf.unwrap_or_else(|| compute_evals_per_leaf(n));
    let bytecode = if config.zkvm_mode {
        let batch_size = config.batch_size.min(codewords.len());
        Some(circuit::compile_das_circuit(batch_size, n, epl, config.degree))
    } else {
        None
    };
    prove_das_with_bytecode(codewords, config, bytecode)
}

/// Run the DAS proving pipeline with a pre-compiled circuit.
///
/// If `bytecode` is `Some`, uses it for ZKVM proving (skipping compilation).
/// If `None` and `config.zkvm_mode` is true, compiles the circuit internally.
pub fn prove_das_with_bytecode(
    codewords: Vec<Codeword>,
    config: &DasConfig,
    bytecode: Option<lean_vm::Bytecode>,
) -> DasProof {
    let m = codewords.len();
    assert!(m > 0, "must have at least one codeword");

    let n = codewords[0].len();
    let epl = config.evals_per_leaf.unwrap_or_else(|| compute_evals_per_leaf(n));
    let batch_size = config.batch_size.min(m);

    tracing::info!(
        m,
        batch_size,
        evals_per_leaf = epl,
        zkvm_mode = config.zkvm_mode,
        "starting DAS proof"
    );

    // Phase 0: Commit all codewords.
    let commitments: Vec<Commitment> = codewords
        .par_iter()
        .map(|cw| commit_codeword_epl(cw, epl))
        .collect();

    // Compile the circuit if needed and not pre-compiled.
    let bytecode = if bytecode.is_some() {
        bytecode
    } else if config.zkvm_mode {
        Some(circuit::compile_das_circuit(batch_size, n, epl, config.degree))
    } else {
        None
    };

    // Phase 1: Partition into batches and prove each.
    // In ZKVM mode, each proof is memory-intensive (STARK trace + FRI
    // commitments), so we prove sequentially to avoid OOM. In reference
    // mode, batches are cheap and can run in parallel.
    let num_batches = (m + batch_size - 1) / batch_size;
    tracing::info!(num_batches, "partitioning into batches");

    let make_batch = |batch_idx: usize| {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(m);
        let indices: Vec<usize> = (start..end).collect();
        let batch_codewords: Vec<Codeword> = codewords[start..end].to_vec();
        let batch_commitments: Vec<Commitment> = commitments[start..end].to_vec();
        prepare_batch(indices, batch_codewords, batch_commitments)
    };

    let batch_proofs: Vec<BatchProof> = if config.zkvm_mode && num_batches > 1 && config.parallel > 0 {
        // Parallel ZKVM proving with controlled concurrency.
        // Process batches in groups of `parallel`, each group member
        // gets its own rayon pool with `total_threads / parallel` threads.
        let concurrency = config.parallel.min(num_batches);
        let total_threads = rayon::current_num_threads();
        let threads_per_batch = (total_threads / concurrency).max(1);
        tracing::info!(concurrency, total_threads, threads_per_batch, num_batches, "parallel proving");

        let mut all_proofs = Vec::with_capacity(num_batches);
        for chunk_start in (0..num_batches).step_by(concurrency) {
            let chunk_end = (chunk_start + concurrency).min(num_batches);
            let chunk_proofs: Vec<BatchProof> = std::thread::scope(|s| {
                let handles: Vec<_> = (chunk_start..chunk_end)
                    .map(|batch_idx| {
                        let batch = make_batch(batch_idx);
                        let bc = bytecode.as_ref().unwrap();
                        s.spawn(move || {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(threads_per_batch)
                                .build()
                                .expect("failed to build thread pool");
                            pool.install(|| prove_batch_zkvm(&batch, bc, config.degree))
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            all_proofs.extend(chunk_proofs);
        }
        all_proofs
    } else if config.zkvm_mode {
        // Sequential proving: one STARK proof at a time to limit memory.
        (0..num_batches)
            .map(|batch_idx| {
                let batch = make_batch(batch_idx);
                prove_batch_zkvm(&batch, bytecode.as_ref().unwrap(), config.degree)
            })
            .collect()
    } else {
        // Parallel proving: reference mode is lightweight.
        (0..num_batches)
            .into_par_iter()
            .map(|batch_idx| {
                let batch = make_batch(batch_idx);
                prove_batch(&batch)
            })
            .collect()
    };

    tracing::info!(
        num_batch_proofs = batch_proofs.len(),
        "phase 1 complete: all batches proved"
    );

    // Phase 2: Aggregate proofs.
    let agg_proofs: Vec<AggregatedProof> =
        batch_proofs.into_iter().map(AggregatedProof::from).collect();

    let aggregated_proof = aggregate_proofs(agg_proofs);

    tracing::info!(
        total_commitments = aggregated_proof.commitments.len(),
        "phase 2 complete: proofs aggregated"
    );

    DasProof {
        commitments,
        aggregated_proof,
        batch_size,
        codeword_length: n,
        evals_per_leaf: epl,
        degree: config.degree,
    }
}
