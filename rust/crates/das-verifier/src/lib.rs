//! # das-verifier
//!
//! DAS verification logic. The verifier receives:
//! - Commitments com_1, ..., com_m to all m input codewords
//! - A DAS proof (batch STARK proofs + aggregation metadata)
//!
//! Verification:
//! 1. For each batch, verify the STARK proof certifying that every
//!    committed codeword in the batch is a valid RS codeword
//!    (via FRI folding inside the ZKVM).
//! 2. For each queried position, verify the commitment opening
//!    against com_i (position-binding).

use das_core::commitment;
use das_core::types::Opening;
use das_prover::circuit::{
    compile_das_circuit, compile_unified_das_circuit,
    build_verifier_public_input, verify_circuit_with_public_input,
};
use das_prover::pipeline::DasProof;


/// Result of DAS verification.
#[derive(Debug)]
pub enum VerifyResult {
    /// All checks passed.
    Accept,
    /// A batch STARK proof failed verification.
    RejectProof { batch_index: usize },
    /// A commitment opening failed verification.
    RejectOpening { codeword_index: usize, position: usize },
}

/// Verify a DAS proof: check all batch STARK proofs and any symbol openings.
///
/// `openings` contains (codeword_index, Opening) pairs for the
/// positions the verifier has sampled.
pub fn verify_das(
    das_proof: &DasProof,
    openings: &[(usize, Opening)],
) -> VerifyResult {
    let bytecode = if das_proof.aggregated_proof.batch_proofs.iter().any(|bp| bp.proof.is_some()) {
        Some(compile_das_circuit(
            das_proof.batch_size,
            das_proof.codeword_length,
            das_proof.evals_per_leaf,
            das_proof.degree,
        ))
    } else {
        None
    };
    verify_das_with_bytecode(das_proof, openings, bytecode)
}

/// Verify a DAS proof with a pre-compiled circuit bytecode.
///
/// If `bytecode` is `None` and STARK proofs are present, compiles internally.
pub fn verify_das_with_bytecode(
    das_proof: &DasProof,
    openings: &[(usize, Opening)],
    bytecode: Option<lean_vm::Bytecode>,
) -> VerifyResult {
    // Step 1: Verify each batch's STARK proof.
    let batch_proofs = &das_proof.aggregated_proof.batch_proofs;

    if !batch_proofs.is_empty() {
        let bytecode = bytecode.unwrap_or_else(|| compile_das_circuit(
            das_proof.batch_size,
            das_proof.codeword_length,
            das_proof.evals_per_leaf,
            das_proof.degree,
        ));

        for (batch_idx, bp) in batch_proofs.iter().enumerate() {
            match &bp.proof {
                Some(proof) => {
                    // Build public input: [chain_hash(commitments), hash(bytecode, domain_sep)]
                    let public_input = build_verifier_public_input(
                        &bp.commitments,
                        &bytecode,
                    );

                    let result = verify_circuit_with_public_input(
                        &bytecode,
                        &public_input,
                        proof.clone(),
                    );
                    if let Err(e) = result {
                        tracing::error!(
                            batch_index = batch_idx,
                            error = %e,
                            "STARK proof verification failed"
                        );
                        return VerifyResult::RejectProof { batch_index: batch_idx };
                    }
                    tracing::info!(batch_index = batch_idx, "STARK proof verified");
                }
                None => {
                    tracing::warn!(
                        batch_index = batch_idx,
                        "no STARK proof present (reference mode)"
                    );
                }
            }
        }
    }

    // Step 2: Verify commitment openings (position-binding).
    for (cw_idx, opening) in openings {
        let com = &das_proof.commitments[*cw_idx];
        if !commitment::verify_opening_epl(com, opening, das_proof.codeword_length, das_proof.evals_per_leaf) {
            tracing::error!(
                codeword = cw_idx,
                leaf_index = opening.leaf_index,
                "commitment opening verification failed"
            );
            return VerifyResult::RejectOpening {
                codeword_index: *cw_idx,
                position: opening.leaf_index,
            };
        }
    }

    tracing::info!(
        num_commitments = das_proof.commitments.len(),
        num_openings = openings.len(),
        num_batch_proofs = batch_proofs.len(),
        "DAS verification passed"
    );

    VerifyResult::Accept
}

/// Verify a unified (self-referential) DAS proof.
///
/// The aggregated proof must have a single recursive proof and public input.
/// This compiles the unified circuit and verifies the proof against it.
pub fn verify_das_unified(
    das_proof: &DasProof,
    openings: &[(usize, Opening)],
) -> VerifyResult {
    // Verify the single recursive proof if present.
    let agg = &das_proof.aggregated_proof;
    if let (Some(proof), Some(public_input)) = (&agg.proof, &agg.public_input) {
        let bytecode = compile_unified_das_circuit(
            das_proof.batch_size,
            das_proof.codeword_length,
            das_proof.evals_per_leaf,
            das_proof.degree,
        );

        let result = verify_circuit_with_public_input(
            &bytecode,
            public_input,
            proof.clone(),
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "unified proof verification failed");
            return VerifyResult::RejectProof { batch_index: 0 };
        }
        tracing::info!("unified recursive proof verified");
    }

    // Verify commitment openings.
    for (cw_idx, opening) in openings {
        let com = &das_proof.commitments[*cw_idx];
        if !commitment::verify_opening_epl(com, opening, das_proof.codeword_length, das_proof.evals_per_leaf) {
            tracing::error!(
                codeword = cw_idx,
                leaf_index = opening.leaf_index,
                "commitment opening verification failed"
            );
            return VerifyResult::RejectOpening {
                codeword_index: *cw_idx,
                position: opening.leaf_index,
            };
        }
    }

    VerifyResult::Accept
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::PrimeCharacteristicRing;
    use das_core::commitment::{commit_codeword_with_tree, open_position};
    use das_core::types::{Codeword, F, EVALS_PER_LEAF};
    use das_prover::pipeline::{prove_das, DasConfig};

    /// Create a valid RS test codeword (constant polynomial, degree 0).
    fn make_codeword(n: usize, seed: usize) -> Codeword {
        let c = F::from_usize(seed + 1);
        let evals = vec![c; n];
        Codeword::new(evals)
    }

    fn default_config(batch_size: usize, zkvm_mode: bool, codeword_len: usize) -> DasConfig {
        DasConfig {
            batch_size,
            zkvm_mode,
            evals_per_leaf: Some(EVALS_PER_LEAF),
            degree: codeword_len / 2,
            parallel: 0,
        }
    }

    #[test]
    fn test_end_to_end_reference_mode() {
        let n = 32;
        let m = 4;
        let codewords: Vec<Codeword> = (0..m).map(|i| make_codeword(n, i)).collect();

        let trees: Vec<_> = codewords
            .iter()
            .map(|cw| commit_codeword_with_tree(cw))
            .collect();

        let config = default_config(2, false, n);
        let das_proof = prove_das(codewords.clone(), &config);

        assert_eq!(das_proof.commitments.len(), m);

        let mut openings = Vec::new();
        openings.push((0, open_position(&codewords[0], &trees[0].1, 3)));
        openings.push((2, open_position(&codewords[2], &trees[2].1, 17)));
        openings.push((3, open_position(&codewords[3], &trees[3].1, 0)));

        let result = verify_das(&das_proof, &openings);
        assert!(matches!(result, VerifyResult::Accept));
    }

    #[test]
    fn test_tampered_opening_rejected() {
        let n = 32;
        let cw = make_codeword(n, 42);
        let (_, tree) = commit_codeword_with_tree(&cw);

        let config = default_config(1, false, n);
        let das_proof = prove_das(vec![cw.clone()], &config);

        let mut opening = open_position(&cw, &tree, 5);
        opening.values[0] = F::from_usize(999_999);

        let result = verify_das(&das_proof, &[(0, opening)]);
        assert!(matches!(result, VerifyResult::RejectOpening { .. }));
    }

    #[test]
    fn test_end_to_end_zkvm_mode() {
        let n = 16;
        let m = 2;
        let codewords: Vec<Codeword> = (0..m).map(|i| make_codeword(n, i)).collect();

        let config = default_config(2, true, n);
        let das_proof = prove_das(codewords.clone(), &config);

        let result = verify_das(&das_proof, &[]);
        assert!(
            matches!(result, VerifyResult::Accept),
            "ZKVM proof should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_end_to_end_rs_codewords() {
        use das_core::polynomial::rs_encode;

        let n = 32;
        let m = 4;
        let degree = 8;

        let codewords: Vec<Codeword> = (0..m)
            .map(|i| {
                let msg: Vec<F> = (0..degree)
                    .map(|j| F::from_usize((i + 1) * (j + 1) + i * 7 + 3))
                    .collect();
                rs_encode(&msg, n)
            })
            .collect();

        let trees: Vec<_> = codewords
            .iter()
            .map(|cw| commit_codeword_with_tree(cw))
            .collect();

        let config = DasConfig {
            batch_size: 2,
            zkvm_mode: false,
            evals_per_leaf: Some(EVALS_PER_LEAF),
            degree,
            parallel: 0,
        };
        let das_proof = prove_das(codewords.clone(), &config);

        let openings = vec![
            (0, open_position(&codewords[0], &trees[0].1, 3)),
            (2, open_position(&codewords[2], &trees[2].1, 17)),
        ];

        let result = verify_das(&das_proof, &openings);
        assert!(matches!(result, VerifyResult::Accept));
    }

    #[test]
    fn test_end_to_end_rs_zkvm() {
        use das_core::polynomial::rs_encode;

        let n = 32;
        let m = 2;
        let degree = 4;

        let codewords: Vec<Codeword> = (0..m)
            .map(|i| {
                let msg: Vec<F> = (0..degree)
                    .map(|j| F::from_usize((i + 1) * (j + 1)))
                    .collect();
                rs_encode(&msg, n)
            })
            .collect();

        let config = DasConfig {
            batch_size: 2,
            zkvm_mode: true,
            evals_per_leaf: Some(EVALS_PER_LEAF),
            degree,
            parallel: 0,
        };
        let das_proof = prove_das(codewords.clone(), &config);

        let result = verify_das(&das_proof, &[]);
        assert!(
            matches!(result, VerifyResult::Accept),
            "RS codewords + ZKVM should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_end_to_end_large_epl() {
        // Test with EPL=32 (sequential compress with 4 chunks).
        use das_core::commitment::{commit_codeword_with_tree_epl, open_position_epl};

        let n = 64;
        let m = 2;
        let epl = 32;
        let codewords: Vec<Codeword> = (0..m).map(|i| make_codeword(n, i)).collect();

        let trees: Vec<_> = codewords
            .iter()
            .map(|cw| commit_codeword_with_tree_epl(cw, epl))
            .collect();

        let config = DasConfig {
            batch_size: 2,
            zkvm_mode: false,
            evals_per_leaf: Some(epl),
            degree: n / 2,
            parallel: 0,
        };
        let das_proof = prove_das(codewords.clone(), &config);

        assert_eq!(das_proof.evals_per_leaf, epl);

        let openings = vec![
            (0, open_position_epl(&codewords[0], &trees[0].1, 3, epl)),
            (1, open_position_epl(&codewords[1], &trees[1].1, 40, epl)),
        ];

        let result = verify_das(&das_proof, &openings);
        assert!(matches!(result, VerifyResult::Accept));
    }
}
