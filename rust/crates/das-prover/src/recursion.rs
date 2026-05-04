//! Recursive proof aggregation for DAS.
//!
//! Provides unified and hybrid recursive proving. The unified circuit is
//! self-referential (verifies its own proofs). The hybrid approach uses a
//! fast inner batch circuit for leaves and a self-referential outer circuit
//! for aggregation. Both reuse the generic STARK verifier from
//! rec_aggregation (recursion.py + whir.py).

use backend::*;
use das_core::types::{F, DIGEST_SIZE};
use lean_prover::prove_execution::prove_execution;
use lean_prover::verify_execution::verify_execution;
use lean_prover::default_whir_config;
use lean_vm::*;
use rec_aggregation::hash_bytecode_claims;
use utils::{build_prover_state, get_poseidon16};

/// Maximum number of inner proofs that can be aggregated in a single recursion.
const MAX_INNER_PROOFS: usize = 16;

/// Prove recursive aggregation using the unified self-referential circuit.
///
/// All inner proofs were produced by the same unified circuit. The function
/// verifies each, computes the bytecode sumcheck, and proves via the unified
/// circuit in recursion mode (n_recursions > 0, n_raw_batches = 0).
///
/// Returns `(proof, public_input)` so the result can be recursively aggregated further.
pub fn prove_unified_recursion(
    inner_proofs: &[(Vec<F>, Proof<F>)],
    bytecode: &Bytecode,
    log_inv_rate: usize,
) -> (Proof<F>, Vec<F>) {
    let n_inner = inner_proofs.len();
    assert!(n_inner > 0 && n_inner <= MAX_INNER_PROOFS);

    let whir_config = default_whir_config(log_inv_rate);
    let bytecode_point_n_vars =
        bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let bytecode_claim_size = (bytecode_point_n_vars + 1) * DIMENSION;
    let bytecode_claim_size_padded = bytecode_claim_size.next_multiple_of(DIGEST_LEN);

    // Step 1: Verify each inner proof to extract RawProof and bytecode evaluation.
    let mut raw_proofs = Vec::new();
    let mut bytecode_evals = Vec::new();
    for (pub_input, proof) in inner_proofs {
        let (details, raw_proof) =
            verify_execution(bytecode, pub_input, proof.clone())
                .expect("inner proof must verify");
        bytecode_evals.push(details.bytecode_evaluation);
        raw_proofs.push(raw_proof);
    }

    // Step 2: Build bytecode claims and compute sumcheck reduction.
    // For each inner proof, we have two claims:
    //   claim[2*i]   = bytecode claim from child's public input (self-referential)
    //   claim[2*i+1] = the actual bytecode evaluation from verification
    let mut claims: Vec<Evaluation<EF>> = Vec::new();
    for (i, eval) in bytecode_evals.iter().enumerate() {
        // First claim: read from child's public input at bytecode claim offset.
        let child_pub = &inner_proofs[i].0;
        let claim_offset = DIGEST_SIZE; // bytecode claim starts after aggregated_data_hash
        let claim_data = &child_pub[claim_offset..claim_offset + bytecode_claim_size];
        // Parse the flattened claim back into (point, value).
        let point_data = &claim_data[..bytecode_point_n_vars * DIMENSION];
        let value_data = &claim_data[bytecode_point_n_vars * DIMENSION..];
        let point: Vec<EF> = point_data
            .chunks_exact(DIMENSION)
            .map(|c| EF::from_basis_coefficients_slice(c).unwrap())
            .collect();
        let value = EF::from_basis_coefficients_slice(value_data).unwrap();
        claims.push(Evaluation::new(MultilinearPoint(point), value));

        // Second claim: from proof verification.
        claims.push(eval.clone());
    }

    let n_claims = claims.len();
    let claims_hash = hash_bytecode_claims(&claims);

    let mut reduction_prover = build_prover_state();
    reduction_prover.add_base_scalars(&claims_hash);
    let alpha: EF = reduction_prover.sample();

    let alpha_powers: Vec<EF> = alpha.powers().take(n_claims).collect();

    let weights_packed = claims
        .par_iter()
        .zip(&alpha_powers)
        .map(|(eval, &alpha_i)| eval_eq_packed_scaled(&eval.point.0, alpha_i))
        .reduce_with(|mut acc, eq_i| {
            acc.par_iter_mut().zip(&eq_i).for_each(|(w, e)| *w += *e);
            acc
        })
        .unwrap();

    let claimed_sum: EF = dot_product(
        claims.iter().map(|c| c.value),
        alpha_powers.iter().copied(),
    );

    let witness = MleGroupOwned::ExtensionPacked(vec![
        bytecode.instructions_multilinear_packed.clone(),
        weights_packed,
    ]);

    let (challenges, final_evals, _) = sumcheck_prove::<EF, _, _>(
        witness,
        &ProductComputation {},
        &vec![],
        None,
        &mut reduction_prover,
        claimed_sum,
        false,
    );

    let reduced_point = challenges;
    let reduced_value = final_evals[0];

    // Build the bytecode claim output (point + value, flattened to base field).
    let mut ef_claim: Vec<EF> = reduced_point.0.clone();
    ef_claim.push(reduced_value);
    let bytecode_claim_output = flatten_scalars_to_base::<F, EF>(&ef_claim);
    assert_eq!(bytecode_claim_output.len(), bytecode_claim_size);

    // Recover the sumcheck transcript for the circuit.
    let final_sumcheck_transcript = {
        let mut vs = VerifierState::<EF, _>::new(
            reduction_prover.into_proof(),
            get_poseidon16().clone(),
        )
        .unwrap();
        vs.next_base_scalars_vec(claims_hash.len()).unwrap();
        let _: EF = vs.sample();
        sumcheck_verify(&mut vs, bytecode_point_n_vars, 2, claimed_sum, None).unwrap();
        vs.into_raw_proof().transcript
    };

    // Step 3: Build public input (unified format).
    // Chain-hash each child's aggregated_data_hash.
    let mut aggregated_hash = [F::ZERO; DIGEST_SIZE];
    for (pub_input, _) in inner_proofs {
        let child_agg_hash: [F; DIGEST_SIZE] = pub_input[..DIGEST_SIZE].try_into().unwrap();
        aggregated_hash = utils::poseidon16_compress_pair(&aggregated_hash, &child_agg_hash);
    }

    let pub_input_size = DIGEST_LEN + bytecode_claim_size_padded + DIGEST_LEN;
    let mut non_reserved_public_input = vec![F::ZERO; pub_input_size];
    non_reserved_public_input[..DIGEST_SIZE].copy_from_slice(&aggregated_hash);

    // Bytecode claim (padded).
    non_reserved_public_input[DIGEST_SIZE..DIGEST_SIZE + bytecode_claim_size]
        .copy_from_slice(&bytecode_claim_output);

    // Bytecode hash.
    let bh_offset = DIGEST_SIZE + bytecode_claim_size_padded;
    let hash = utils::poseidon16_compress_pair(&bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    non_reserved_public_input[bh_offset..bh_offset + DIGEST_SIZE].copy_from_slice(&hash);

    let public_memory = build_public_memory(&non_reserved_public_input);

    // Step 4: Build private input with unified header.
    // Header: [n_recursions, n_raw_batches=0, ptr_src_0, ..., ptr_src_{N-1}, ptr_bytecode_sumcheck]
    let n_raw_batches = 0usize;
    let header_size = 2 + n_inner + n_raw_batches + 1;
    let private_base = public_memory.len();

    // Build source blocks for each inner proof.
    let inner_pub_memories: Vec<Vec<F>> = inner_proofs
        .iter()
        .map(|(pub_input, _)| build_public_memory(pub_input))
        .collect();

    let mut source_blocks: Vec<Vec<F>> = Vec::new();
    for i in 0..n_inner {
        let mut block = Vec::new();
        // bytecode_value_hint (DIM elements)
        block.extend_from_slice(
            bytecode_evals[i].value.as_basis_coefficients_slice(),
        );
        // inner_pub_mem (padded public memory of inner proof)
        block.extend_from_slice(&inner_pub_memories[i]);
        // proof_transcript
        block.extend_from_slice(&raw_proofs[i].transcript);
        source_blocks.push(block);
    }

    // Compute absolute addresses for source pointers.
    let sources_start = private_base + header_size;
    let mut offset = sources_start;
    let mut source_ptrs: Vec<usize> = Vec::new();
    for block in &source_blocks {
        source_ptrs.push(offset);
        offset += block.len();
    }
    let bytecode_sumcheck_ptr = offset;

    let mut private_input = Vec::new();
    private_input.push(F::from_usize(n_inner));        // n_recursions
    private_input.push(F::from_usize(n_raw_batches));   // n_raw_batches
    for &ptr in &source_ptrs {
        private_input.push(F::from_usize(ptr));
    }
    private_input.push(F::from_usize(bytecode_sumcheck_ptr));
    assert_eq!(private_input.len(), header_size);

    for block in &source_blocks {
        private_input.extend_from_slice(block);
    }
    private_input.extend_from_slice(&final_sumcheck_transcript);

    // Build merkle paths from raw proofs.
    let merkle_paths: Vec<Vec<F>> = raw_proofs
        .iter()
        .flat_map(|p| p.merkle_openings.iter())
        .flat_map(|o| {
            let leaf = o.leaf_data.clone();
            let path: Vec<F> = o.path.iter().flat_map(|d| d.iter().copied()).collect();
            [leaf, path]
        })
        .collect();

    let xmss_signatures: Vec<Vec<F>> = vec![];

    let witness = ExecutionWitness {
        private_input: &private_input,
        xmss_signatures: &xmss_signatures,
        merkle_paths: &merkle_paths,
    };

    tracing::info!(
        n_inner,
        private_input_len = private_input.len(),
        merkle_paths_count = merkle_paths.len(),
        "proving unified recursion"
    );

    let execution_proof = prove_execution(
        bytecode,
        &non_reserved_public_input,
        &witness,
        &whir_config,
        false,
    );

    (execution_proof.proof, non_reserved_public_input)
}

/// Prove hybrid recursion: verify a mix of self proofs and inner batch proofs.
///
/// - `self_proofs`: proofs from the hybrid circuit itself (self-referential).
///   Each has the hybrid public input layout: [agg_hash, inner_claim, self_claim, inner_bc_hash, self_bc_hash].
/// - `inner_proofs`: proofs from the fast-path inner batch circuit.
///   Each has the inner public input layout: [batch_hash, bytecode_hash].
/// - `hybrid_bytecode`: the hybrid circuit's bytecode (used for self-proof verification and proving).
/// - `inner_bytecode`: the inner batch circuit's bytecode (used for inner proof verification).
/// - `log_inv_rate`: WHIR log inverse rate.
///
/// Returns `(proof, non_reserved_public_input)`.
pub fn prove_hybrid_recursion(
    self_proofs: &[(Vec<F>, Proof<F>)],
    inner_proofs: &[(Vec<F>, Proof<F>)],
    hybrid_bytecode: &Bytecode,
    inner_bytecode: &Bytecode,
    log_inv_rate: usize,
) -> (Proof<F>, Vec<F>) {
    let n_recursions = self_proofs.len();
    let n_inner_proofs = inner_proofs.len();
    assert!(n_recursions + n_inner_proofs > 0);
    assert!(n_recursions <= MAX_INNER_PROOFS);
    assert!(n_inner_proofs <= MAX_INNER_PROOFS);

    let whir_config = default_whir_config(log_inv_rate);

    // Self circuit claim dimensions.
    let self_bpnv = hybrid_bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let self_claim_size = (self_bpnv + 1) * DIMENSION;
    let self_claim_size_padded = self_claim_size.next_multiple_of(DIGEST_LEN);

    // Inner circuit claim dimensions.
    let inner_bpnv = inner_bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let inner_claim_size = (inner_bpnv + 1) * DIMENSION;
    let inner_claim_size_padded = inner_claim_size.next_multiple_of(DIGEST_LEN);

    // Public input layout offsets.
    let inner_claim_pub_offset = DIGEST_SIZE;
    let self_claim_pub_offset = DIGEST_SIZE + inner_claim_size_padded;

    // Step 1: Verify self proofs.
    let mut self_raw_proofs = Vec::new();
    let mut self_bytecode_evals = Vec::new();
    for (pub_input, proof) in self_proofs {
        let (details, raw_proof) =
            verify_execution(hybrid_bytecode, pub_input, proof.clone())
                .expect("self proof must verify");
        self_bytecode_evals.push(details.bytecode_evaluation);
        self_raw_proofs.push(raw_proof);
    }

    // Step 2: Verify inner proofs.
    let mut inner_raw_proofs = Vec::new();
    let mut inner_bytecode_evals = Vec::new();
    for (pub_input, proof) in inner_proofs {
        let (details, raw_proof) =
            verify_execution(inner_bytecode, pub_input, proof.clone())
                .expect("inner proof must verify");
        inner_bytecode_evals.push(details.bytecode_evaluation);
        inner_raw_proofs.push(raw_proof);
    }

    // Step 3: Build claims.
    // inner_claims: n_recursions (propagated from self children) + 2 * n_inner_proofs (default + verification)
    let mut inner_claims: Vec<Evaluation<EF>> = Vec::new();
    for (i, _) in self_proofs.iter().enumerate() {
        // Propagate child's inner bytecode claim.
        let child_pub = &self_proofs[i].0;
        let claim_data = &child_pub[inner_claim_pub_offset..inner_claim_pub_offset + inner_claim_size];
        let point: Vec<EF> = claim_data[..inner_bpnv * DIMENSION]
            .chunks_exact(DIMENSION)
            .map(|c| EF::from_basis_coefficients_slice(c).unwrap())
            .collect();
        let value = EF::from_basis_coefficients_slice(
            &claim_data[inner_bpnv * DIMENSION..inner_claim_size],
        )
        .unwrap();
        inner_claims.push(Evaluation::new(MultilinearPoint(point), value));
    }
    for eval in &inner_bytecode_evals {
        // Default claim (zero point, inner_bytecode_zero_eval).
        let zero_point = vec![EF::ZERO; inner_bpnv];
        let zero_eval = EF::from_basis_coefficients_slice(
            &[inner_bytecode.instructions_multilinear[0], F::ZERO, F::ZERO, F::ZERO, F::ZERO][..DIMENSION],
        )
        .unwrap();
        inner_claims.push(Evaluation::new(MultilinearPoint(zero_point), zero_eval));
        // Verification claim.
        inner_claims.push(eval.clone());
    }

    // self_claims: 2 * n_recursions (propagated + verification)
    let mut self_claims: Vec<Evaluation<EF>> = Vec::new();
    for (i, eval) in self_bytecode_evals.iter().enumerate() {
        // Propagate child's self bytecode claim.
        let child_pub = &self_proofs[i].0;
        let claim_data = &child_pub[self_claim_pub_offset..self_claim_pub_offset + self_claim_size];
        let point: Vec<EF> = claim_data[..self_bpnv * DIMENSION]
            .chunks_exact(DIMENSION)
            .map(|c| EF::from_basis_coefficients_slice(c).unwrap())
            .collect();
        let value = EF::from_basis_coefficients_slice(
            &claim_data[self_bpnv * DIMENSION..self_claim_size],
        )
        .unwrap();
        self_claims.push(Evaluation::new(MultilinearPoint(point), value));
        // Verification claim.
        self_claims.push(eval.clone());
    }

    // Step 4: Sumcheck reductions.
    let inner_sumcheck_transcript = if inner_claims.is_empty() {
        vec![]
    } else {
        do_bytecode_sumcheck_reduction(
            &inner_claims,
            inner_bpnv,
            &inner_bytecode.instructions_multilinear_packed,
        )
    };

    let self_sumcheck_transcript = if self_claims.is_empty() {
        vec![]
    } else {
        do_bytecode_sumcheck_reduction(
            &self_claims,
            self_bpnv,
            &hybrid_bytecode.instructions_multilinear_packed,
        )
    };

    // Step 5: Compute reduced claims for public input.
    let inner_claim_output = if inner_claims.is_empty() {
        // Default claim: zero point, inner_bytecode_zero_eval.
        let mut claim = vec![F::ZERO; inner_claim_size];
        claim[inner_bpnv * DIMENSION] = inner_bytecode.instructions_multilinear[0];
        claim
    } else {
        compute_reduced_claim(&inner_claims, inner_bpnv, &inner_bytecode.instructions_multilinear_packed)
    };

    let self_claim_output = if self_claims.is_empty() {
        // Default claim: zero point, self_bytecode_zero_eval.
        let mut claim = vec![F::ZERO; self_claim_size];
        claim[self_bpnv * DIMENSION] = hybrid_bytecode.instructions_multilinear[0];
        claim
    } else {
        compute_reduced_claim(&self_claims, self_bpnv, &hybrid_bytecode.instructions_multilinear_packed)
    };

    // Step 6: Build public input.
    let mut aggregated_hash = [F::ZERO; DIGEST_SIZE];
    // Chain self children's aggregated hashes.
    for (pub_input, _) in self_proofs {
        let child_agg_hash: [F; DIGEST_SIZE] = pub_input[..DIGEST_SIZE].try_into().unwrap();
        aggregated_hash = utils::poseidon16_compress_pair(&aggregated_hash, &child_agg_hash);
    }
    // Chain inner children's batch hashes.
    for (pub_input, _) in inner_proofs {
        let batch_hash: [F; DIGEST_SIZE] = pub_input[..DIGEST_SIZE].try_into().unwrap();
        aggregated_hash = utils::poseidon16_compress_pair(&aggregated_hash, &batch_hash);
    }

    let hybrid_pub_input_size = DIGEST_LEN
        + inner_claim_size_padded
        + self_claim_size_padded
        + DIGEST_LEN
        + DIGEST_LEN;
    let mut non_reserved_public_input = vec![F::ZERO; hybrid_pub_input_size];
    non_reserved_public_input[..DIGEST_SIZE].copy_from_slice(&aggregated_hash);

    // Inner bytecode claim.
    non_reserved_public_input[DIGEST_SIZE..DIGEST_SIZE + inner_claim_size]
        .copy_from_slice(&inner_claim_output);

    // Self bytecode claim.
    let self_claim_start = DIGEST_SIZE + inner_claim_size_padded;
    non_reserved_public_input[self_claim_start..self_claim_start + self_claim_size]
        .copy_from_slice(&self_claim_output);

    // Inner bytecode hash.
    let inner_bh_offset = DIGEST_SIZE + inner_claim_size_padded + self_claim_size_padded;
    let inner_hash =
        utils::poseidon16_compress_pair(&inner_bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    non_reserved_public_input[inner_bh_offset..inner_bh_offset + DIGEST_SIZE]
        .copy_from_slice(&inner_hash);

    // Self bytecode hash.
    let self_bh_offset = inner_bh_offset + DIGEST_SIZE;
    let self_hash =
        utils::poseidon16_compress_pair(&hybrid_bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    non_reserved_public_input[self_bh_offset..self_bh_offset + DIGEST_SIZE]
        .copy_from_slice(&self_hash);

    let public_memory = build_public_memory(&non_reserved_public_input);

    // Step 7: Build private input.
    // Header: [n_recursions, n_inner_proofs,
    //          ptr_src_0, ..., ptr_src_{N-1},
    //          ptr_inner_sumcheck, ptr_self_sumcheck]
    let total_sources = n_recursions + n_inner_proofs;
    let header_size = 2 + total_sources + 2; // 2 counts + ptrs + 2 sumcheck ptrs
    let private_base = public_memory.len();

    // Build inner public memories for both proof types.
    let self_pub_memories: Vec<Vec<F>> = self_proofs
        .iter()
        .map(|(pub_input, _)| build_public_memory(pub_input))
        .collect();
    let inner_pub_memories: Vec<Vec<F>> = inner_proofs
        .iter()
        .map(|(pub_input, _)| build_public_memory(pub_input))
        .collect();

    // Source blocks: first self proofs, then inner proofs.
    let mut source_blocks: Vec<Vec<F>> = Vec::new();
    for i in 0..n_recursions {
        let mut block = Vec::new();
        block.extend_from_slice(
            self_bytecode_evals[i].value.as_basis_coefficients_slice(),
        );
        block.extend_from_slice(&self_pub_memories[i]);
        block.extend_from_slice(&self_raw_proofs[i].transcript);
        source_blocks.push(block);
    }
    for i in 0..n_inner_proofs {
        let mut block = Vec::new();
        block.extend_from_slice(
            inner_bytecode_evals[i].value.as_basis_coefficients_slice(),
        );
        block.extend_from_slice(&inner_pub_memories[i]);
        block.extend_from_slice(&inner_raw_proofs[i].transcript);
        source_blocks.push(block);
    }

    // Compute absolute addresses.
    let sources_start = private_base + header_size;
    let mut offset = sources_start;
    let mut source_ptrs: Vec<usize> = Vec::new();
    for block in &source_blocks {
        source_ptrs.push(offset);
        offset += block.len();
    }
    let inner_sumcheck_ptr = offset;
    offset += inner_sumcheck_transcript.len();
    let self_sumcheck_ptr = offset;

    let mut private_input = Vec::new();
    private_input.push(F::from_usize(n_recursions));
    private_input.push(F::from_usize(n_inner_proofs));
    for &ptr in &source_ptrs {
        private_input.push(F::from_usize(ptr));
    }
    private_input.push(F::from_usize(inner_sumcheck_ptr));
    private_input.push(F::from_usize(self_sumcheck_ptr));
    assert_eq!(private_input.len(), header_size);

    for block in &source_blocks {
        private_input.extend_from_slice(block);
    }
    private_input.extend_from_slice(&inner_sumcheck_transcript);
    private_input.extend_from_slice(&self_sumcheck_transcript);

    // Build merkle paths from all raw proofs.
    let merkle_paths: Vec<Vec<F>> = self_raw_proofs
        .iter()
        .chain(inner_raw_proofs.iter())
        .flat_map(|p| p.merkle_openings.iter())
        .flat_map(|o| {
            let leaf = o.leaf_data.clone();
            let path: Vec<F> = o.path.iter().flat_map(|d| d.iter().copied()).collect();
            [leaf, path]
        })
        .collect();

    let xmss_signatures: Vec<Vec<F>> = vec![];

    let witness = ExecutionWitness {
        private_input: &private_input,
        xmss_signatures: &xmss_signatures,
        merkle_paths: &merkle_paths,
    };

    tracing::info!(
        n_recursions,
        n_inner_proofs,
        private_input_len = private_input.len(),
        merkle_paths_count = merkle_paths.len(),
        "proving hybrid recursion"
    );

    let execution_proof = prove_execution(
        hybrid_bytecode,
        &non_reserved_public_input,
        &witness,
        &whir_config,
        false,
    );

    (execution_proof.proof, non_reserved_public_input)
}

/// Perform sumcheck reduction on a set of bytecode claims.
/// Returns the raw transcript bytes of the sumcheck proof.
fn do_bytecode_sumcheck_reduction(
    claims: &[Evaluation<EF>],
    point_n_vars: usize,
    bytecode_packed: &[EFPacking<EF>],
) -> Vec<F> {
    let n_claims = claims.len();
    let claims_hash = hash_bytecode_claims(claims);

    let mut reduction_prover = build_prover_state();
    reduction_prover.add_base_scalars(&claims_hash);
    let alpha: EF = reduction_prover.sample();

    let alpha_powers: Vec<EF> = alpha.powers().take(n_claims).collect();

    let weights_packed = claims
        .par_iter()
        .zip(&alpha_powers)
        .map(|(eval, &alpha_i)| eval_eq_packed_scaled(&eval.point.0, alpha_i))
        .reduce_with(|mut acc, eq_i| {
            acc.par_iter_mut().zip(&eq_i).for_each(|(w, e)| *w += *e);
            acc
        })
        .unwrap();

    let claimed_sum: EF = dot_product(
        claims.iter().map(|c| c.value),
        alpha_powers.iter().copied(),
    );

    let witness = MleGroupOwned::ExtensionPacked(vec![
        bytecode_packed.to_vec(),
        weights_packed,
    ]);

    let (_, _, _) = sumcheck_prove::<EF, _, _>(
        witness,
        &ProductComputation {},
        &vec![],
        None,
        &mut reduction_prover,
        claimed_sum,
        false,
    );

    // Recover transcript.
    let mut vs = VerifierState::<EF, _>::new(
        reduction_prover.into_proof(),
        get_poseidon16().clone(),
    )
    .unwrap();
    vs.next_base_scalars_vec(claims_hash.len()).unwrap();
    let _: EF = vs.sample();
    sumcheck_verify(&mut vs, point_n_vars, 2, claimed_sum, None).unwrap();
    vs.into_raw_proof().transcript
}

/// Compute the reduced bytecode claim from a set of claims after sumcheck.
/// Returns flattened base-field claim data (point + value).
fn compute_reduced_claim(
    claims: &[Evaluation<EF>],
    point_n_vars: usize,
    bytecode_packed: &[EFPacking<EF>],
) -> Vec<F> {
    let n_claims = claims.len();
    let claims_hash = hash_bytecode_claims(claims);

    let mut reduction_prover = build_prover_state();
    reduction_prover.add_base_scalars(&claims_hash);
    let alpha: EF = reduction_prover.sample();

    let alpha_powers: Vec<EF> = alpha.powers().take(n_claims).collect();

    let weights_packed = claims
        .par_iter()
        .zip(&alpha_powers)
        .map(|(eval, &alpha_i)| eval_eq_packed_scaled(&eval.point.0, alpha_i))
        .reduce_with(|mut acc, eq_i| {
            acc.par_iter_mut().zip(&eq_i).for_each(|(w, e)| *w += *e);
            acc
        })
        .unwrap();

    let claimed_sum: EF = dot_product(
        claims.iter().map(|c| c.value),
        alpha_powers.iter().copied(),
    );

    let witness = MleGroupOwned::ExtensionPacked(vec![
        bytecode_packed.to_vec(),
        weights_packed,
    ]);

    let (challenges, final_evals, _) = sumcheck_prove::<EF, _, _>(
        witness,
        &ProductComputation {},
        &vec![],
        None,
        &mut reduction_prover,
        claimed_sum,
        false,
    );

    let reduced_point = challenges;
    let reduced_value = final_evals[0];

    let mut ef_claim: Vec<EF> = reduced_point.0;
    ef_claim.push(reduced_value);
    let claim_size = (point_n_vars + 1) * DIMENSION;
    let claim_output = flatten_scalars_to_base::<F, EF>(&ef_claim);
    assert_eq!(claim_output.len(), claim_size);
    claim_output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit;

    #[test]
    fn test_prove_unified_recursion_end_to_end() {
        use das_core::types::{Codeword, EVALS_PER_LEAF};
        use das_core::{commit_codeword, derive_challenge};

        let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
        let handle = builder
            .spawn(|| {
                let batch_size = 1;
                let codeword_len = 32;
                let epl = EVALS_PER_LEAF;
                let degree = 16;

                // Step 1: Compile unified circuit (self-referential).
                let bytecode = circuit::compile_unified_das_circuit(
                    batch_size, codeword_len, epl, degree,
                );
                println!(
                    "unified circuit: {} instructions, log_size={}",
                    bytecode.instructions.len(),
                    bytecode.log_size()
                );

                // Step 2: Prove 2 leaf batches with unified circuit.
                let num_batches = 2;
                let mut leaf_proofs = Vec::new();
                let log_inv_rate = 1;

                for batch_idx in 0..num_batches {
                    let codewords: Vec<Codeword> = (0..batch_size)
                        .map(|i| {
                            let c = F::from_usize(batch_idx * batch_size + i + 1);
                            Codeword::new(vec![c; codeword_len])
                        })
                        .collect();

                    let commitments: Vec<_> =
                        codewords.iter().map(|cw| commit_codeword(cw)).collect();
                    let challenges = commitments.iter().map(derive_challenge).collect();

                    let batch = das_core::types::Batch {
                        indices: (0..batch_size).collect(),
                        codewords,
                        commitments,
                        challenges,
                    };

                    let (proof, pub_input) =
                        circuit::prove_unified_leaf(&bytecode, &batch, log_inv_rate);
                    println!(
                        "leaf batch {}: proved, pub_input len={}, proof size={}",
                        batch_idx,
                        pub_input.len(),
                        proof.proof_size_fe()
                    );

                    // Verify the leaf proof.
                    verify_execution(&bytecode, &pub_input, proof.clone())
                        .expect("leaf proof must verify");
                    println!("leaf batch {}: verified", batch_idx);

                    leaf_proofs.push((pub_input, proof));
                }

                // Step 3: Self-referential recursion — same bytecode for inner and outer.
                println!(
                    "proving unified recursion over {} leaf proofs...",
                    leaf_proofs.len()
                );
                let (rec_proof, rec_pub_input) = prove_unified_recursion(
                    &leaf_proofs,
                    &bytecode,
                    log_inv_rate,
                );
                println!(
                    "unified recursion proof: size={}, pub_input len={}",
                    rec_proof.proof_size_fe(),
                    rec_pub_input.len()
                );

                // Step 4: Verify the recursive proof.
                verify_execution(&bytecode, &rec_pub_input, rec_proof)
                    .expect("recursive proof must verify");
                println!("unified recursion proof: verified!");
            })
            .expect("failed to spawn thread");
        handle.join().expect("unified recursion thread panicked");
    }

    #[test]
    fn test_compile_hybrid_circuit() {
        use das_core::types::EVALS_PER_LEAF;

        let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
        let handle = builder
            .spawn(|| {
                let (inner_bytecode, hybrid_bytecode) =
                    circuit::compile_hybrid_das_circuit(1, 32, EVALS_PER_LEAF, 16);
                println!(
                    "inner circuit: {} instructions, log_size={}",
                    inner_bytecode.instructions.len(),
                    inner_bytecode.log_size()
                );
                println!(
                    "hybrid circuit: {} instructions, log_size={}",
                    hybrid_bytecode.instructions.len(),
                    hybrid_bytecode.log_size()
                );
                assert!(
                    !inner_bytecode.instructions.is_empty(),
                    "inner circuit should have instructions"
                );
                assert!(
                    !hybrid_bytecode.instructions.is_empty(),
                    "hybrid circuit should have instructions"
                );
                assert!(
                    hybrid_bytecode.log_size() > inner_bytecode.log_size(),
                    "hybrid circuit should be larger than inner"
                );
            })
            .expect("failed to spawn thread");
        handle.join().expect("hybrid compilation thread panicked");
    }

    #[test]
    fn test_prove_hybrid_recursion_end_to_end() {
        use das_core::types::{Codeword, EVALS_PER_LEAF};
        use das_core::{commit_codeword, derive_challenge};
        use std::time::Instant;

        let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
        let handle = builder
            .spawn(|| {
                let batch_size = 1;
                let codeword_len = 32;
                let epl = EVALS_PER_LEAF;
                let degree = 16;
                let log_inv_rate = 1;

                // Step 1: Compile both circuits.
                let t0 = Instant::now();
                let (inner_bytecode, hybrid_bytecode) =
                    circuit::compile_hybrid_das_circuit(batch_size, codeword_len, epl, degree);
                let compile_time = t0.elapsed();
                println!(
                    "\n=== Compilation ({:.2?}) ===",
                    compile_time
                );
                println!(
                    "  inner circuit:  {:>6} instructions, log_size={}",
                    inner_bytecode.instructions.len(),
                    inner_bytecode.log_size(),
                );
                println!(
                    "  hybrid circuit: {:>6} instructions, log_size={}",
                    hybrid_bytecode.instructions.len(),
                    hybrid_bytecode.log_size(),
                );

                // Step 2: Prove 4 batches with the inner circuit.
                let num_batches = 4;
                let mut inner_proofs = Vec::new();

                println!("\n=== Inner Batch Proving ({} batches, batch_size={}, N={}, deg={}) ===",
                    num_batches, batch_size, codeword_len, degree);

                let t_inner_total = Instant::now();
                for batch_idx in 0..num_batches {
                    let codewords: Vec<Codeword> = (0..batch_size)
                        .map(|i| {
                            let c = F::from_usize(batch_idx * batch_size + i + 1);
                            Codeword::new(vec![c; codeword_len])
                        })
                        .collect();

                    let commitments: Vec<_> =
                        codewords.iter().map(|cw| commit_codeword(cw)).collect();
                    let challenges = commitments.iter().map(derive_challenge).collect();

                    let batch = das_core::types::Batch {
                        indices: (0..batch_size).collect(),
                        codewords,
                        commitments,
                        challenges,
                    };

                    let t_prove = Instant::now();
                    let (proof, pub_input) = circuit::prove_circuit(&inner_bytecode, &batch, degree);
                    let prove_time = t_prove.elapsed();

                    let t_verify = Instant::now();
                    verify_execution(&inner_bytecode, &pub_input, proof.clone())
                        .expect("inner batch proof must verify");
                    let verify_time = t_verify.elapsed();

                    println!(
                        "  batch {}: prove={:.2?}, verify={:.2?}, proof_size={} FE, pub_input={}",
                        batch_idx, prove_time, verify_time,
                        proof.proof_size_fe(), pub_input.len()
                    );

                    inner_proofs.push((pub_input, proof));
                }
                let inner_total = t_inner_total.elapsed();
                println!("  total inner proving: {:.2?}", inner_total);

                // Step 3: Hybrid recursion — verify inner proofs with hybrid circuit.
                println!("\n=== Hybrid Aggregation ({} inner proofs → 1 proof) ===",
                    inner_proofs.len());

                let t_agg = Instant::now();
                let (rec_proof, rec_pub_input) = prove_hybrid_recursion(
                    &[],            // no self proofs
                    &inner_proofs,  // inner batch proofs
                    &hybrid_bytecode,
                    &inner_bytecode,
                    log_inv_rate,
                );
                let agg_prove_time = t_agg.elapsed();

                let t_agg_verify = Instant::now();
                verify_execution(&hybrid_bytecode, &rec_pub_input, rec_proof.clone())
                    .expect("hybrid recursive proof must verify");
                let agg_verify_time = t_agg_verify.elapsed();

                println!(
                    "  prove:      {:.2?}",
                    agg_prove_time
                );
                println!(
                    "  verify:     {:.2?}",
                    agg_verify_time
                );
                println!(
                    "  proof_size: {} FE",
                    rec_proof.proof_size_fe()
                );
                println!(
                    "  pub_input:  {} elements",
                    rec_pub_input.len()
                );
                println!("\n=== All done ===");
            })
            .expect("failed to spawn thread");
        handle.join().expect("hybrid recursion thread panicked");
    }
}
