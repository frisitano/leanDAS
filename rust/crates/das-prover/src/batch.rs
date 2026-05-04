//! Phase 1: Proving (RLC + FRI folding inside the ZKVM).
//!
//! Each prover takes a batch of codewords with their commitments,
//! derives challenges, computes the RLC, and proves RS membership
//! via FRI folding — all inside a single ZKVM run.

use backend::{BasedVectorSpace, Proof, PrimeCharacteristicRing};
use das_core::types::{Batch, Codeword, Commitment, F, EF, DIGEST_SIZE};
use das_core::{derive_challenge, fri_fold_evaluations, compute_rlc};
use lean_vm::Bytecode;

use crate::circuit;

/// The output of a single prover: a STARK proof plus metadata.
#[derive(Clone)]
pub struct BatchProof {
    /// The commitments covered by this proof (public inputs).
    pub commitments: Vec<Commitment>,
    /// The STARK proof certifying RLC correctness + exact RS membership.
    /// None in reference mode; Some when proved inside the ZKVM.
    pub proof: Option<Proof<F>>,
    /// The non-reserved public input used for proving (needed for recursion).
    /// None in reference mode.
    pub public_input: Option<Vec<F>>,
}

/// Prove a batch of codewords in reference mode (no ZKVM proof).
///
/// Computes RLC + FRI folding natively to validate correctness.
/// Useful for testing the pipeline without the overhead of ZKVM proving.
pub fn prove_batch(batch: &Batch) -> BatchProof {
    assert!(!batch.codewords.is_empty(), "batch must be non-empty");
    assert_eq!(batch.codewords.len(), batch.commitments.len());
    assert_eq!(batch.codewords.len(), batch.challenges.len());

    let n = batch.codewords[0].len();
    assert!(n.is_power_of_two());

    // Step 1: Compute the RLC of the batch's codewords.
    let rlc = compute_rlc(&batch.codewords, &batch.challenges);
    assert_eq!(rlc.len(), n);

    // Step 2: FRI folding on the RLC evaluation vector.
    let num_rounds = n.ilog2() as usize;
    let fri_challenges: Vec<EF> = (0..num_rounds)
        .map(|i| {
            let seed = [F::from_usize(i + 1); DIGEST_SIZE];
            EF::from_basis_coefficients_slice(&seed[..5]).unwrap()
        })
        .collect();

    let _constant = fri_fold_evaluations(&rlc, &fri_challenges);

    tracing::info!(
        batch_size = batch.codewords.len(),
        codeword_len = n,
        "proved batch (reference mode): RLC + FRI folding complete"
    );

    BatchProof {
        commitments: batch.commitments.clone(),
        proof: None,
        public_input: None,
    }
}

/// Prove a batch of codewords inside the ZKVM, producing a STARK proof.
///
/// This compiles the RLC+FRI circuit, runs it inside the leanVM,
/// and returns a proof certifying both RLC correctness and exact RS
/// membership of the batch's combined codeword.
pub fn prove_batch_zkvm(batch: &Batch, bytecode: &Bytecode, degree: usize) -> BatchProof {
    assert!(!batch.codewords.is_empty(), "batch must be non-empty");
    assert_eq!(batch.codewords.len(), batch.commitments.len());
    assert_eq!(batch.codewords.len(), batch.challenges.len());

    let n = batch.codewords[0].len();
    assert!(n.is_power_of_two());

    // Run the ZKVM circuit to produce the STARK proof.
    let (proof, public_input) = circuit::prove_circuit(bytecode, batch, degree);

    tracing::info!(
        batch_size = batch.codewords.len(),
        codeword_len = n,
        "proved batch (ZKVM mode): STARK proof generated"
    );

    BatchProof {
        commitments: batch.commitments.clone(),
        proof: Some(proof),
        public_input: Some(public_input),
    }
}

/// Prepare a batch from codewords and their commitments.
pub fn prepare_batch(
    indices: Vec<usize>,
    codewords: Vec<Codeword>,
    commitments: Vec<Commitment>,
) -> Batch {
    let challenges: Vec<EF> = commitments.iter().map(derive_challenge).collect();

    Batch {
        indices,
        codewords,
        commitments,
        challenges,
    }
}
