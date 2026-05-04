//! Phase 2: Recursive proof aggregation.
//!
//! Proofs from Phase 1 are recursively aggregated in a binary tree.
//! Each aggregation step verifies N child proofs inside the ZKVM
//! and produces a new proof whose public inputs are the union of
//! the children's commitments.

use backend::Proof;
use das_core::types::{Commitment, F};
use lean_vm::Bytecode;

use crate::batch::BatchProof;
use crate::recursion::prove_unified_recursion;

/// An aggregated proof: covers all commitments in its subtree.
#[derive(Clone)]
pub struct AggregatedProof {
    /// All commitments covered by this proof (public inputs).
    pub commitments: Vec<Commitment>,
    /// The single recursive STARK proof.
    /// None in reference mode or when recursion is not yet wired up.
    pub proof: Option<Proof<F>>,
    /// Individual batch proofs (used for verification until recursive
    /// aggregation replaces them with a single proof).
    pub batch_proofs: Vec<BatchProof>,
    /// The public input used for this proof (needed for recursive aggregation
    /// of this proof into a higher-level proof).
    pub public_input: Option<Vec<F>>,
}

impl From<BatchProof> for AggregatedProof {
    fn from(bp: BatchProof) -> Self {
        let public_input = bp.public_input.clone();
        AggregatedProof {
            commitments: bp.commitments.clone(),
            proof: bp.proof.clone(),
            batch_proofs: vec![bp],
            public_input,
        }
    }
}

/// Context for unified self-referential aggregation.
///
/// Uses a single bytecode for both leaf proving and recursive aggregation.
pub struct UnifiedAggregationContext {
    /// The unified (self-referential) bytecode.
    pub bytecode: Bytecode,
    /// WHIR log inverse rate.
    pub log_inv_rate: usize,
}

impl UnifiedAggregationContext {
    /// Compile the unified self-referential circuit.
    pub fn new(
        batch_size: usize,
        codeword_len: usize,
        evals_per_leaf: usize,
        degree: usize,
        log_inv_rate: usize,
    ) -> Self {
        let bytecode = crate::circuit::compile_unified_das_circuit(
            batch_size, codeword_len, evals_per_leaf, degree,
        );
        Self {
            bytecode,
            log_inv_rate,
        }
    }
}

/// Aggregate multiple proofs using the unified self-referential circuit.
///
/// All children must have been proved with the same unified bytecode.
pub fn aggregate_with_unified_recursion(
    children: &[&AggregatedProof],
    ctx: &UnifiedAggregationContext,
) -> AggregatedProof {
    let mut commitments = Vec::new();
    for child in children {
        commitments.extend_from_slice(&child.commitments);
    }

    let inner_proofs: Vec<(Vec<F>, Proof<F>)> = children
        .iter()
        .map(|child| {
            let pub_input = child
                .public_input
                .as_ref()
                .expect("child must have public_input for unified recursion");
            let proof = child
                .proof
                .as_ref()
                .expect("child must have a proof for unified recursion");
            (pub_input.clone(), proof.clone())
        })
        .collect();

    tracing::info!(
        n_children = children.len(),
        total_coms = commitments.len(),
        "aggregating proofs via unified recursion circuit"
    );

    let (proof, public_input) = prove_unified_recursion(
        &inner_proofs,
        &ctx.bytecode,
        ctx.log_inv_rate,
    );

    let mut batch_proofs = Vec::new();
    for child in children {
        batch_proofs.extend_from_slice(&child.batch_proofs);
    }

    AggregatedProof {
        commitments,
        proof: Some(proof),
        batch_proofs,
        public_input: Some(public_input),
    }
}

/// Recursively aggregate proofs using the unified self-referential circuit.
pub fn aggregate_proofs_unified(
    proofs: Vec<AggregatedProof>,
    ctx: &UnifiedAggregationContext,
    max_per_step: usize,
) -> AggregatedProof {
    assert!(!proofs.is_empty(), "must have at least one proof");
    assert!(max_per_step >= 2, "need at least 2 proofs per step");

    if proofs.len() == 1 {
        return proofs.into_iter().next().unwrap();
    }

    let mut current_layer = proofs;
    let mut level = 0;

    while current_layer.len() > 1 {
        tracing::info!(
            level,
            proofs = current_layer.len(),
            "unified recursive aggregation level"
        );

        let mut next_layer = Vec::new();
        for chunk in current_layer.chunks(max_per_step) {
            let refs: Vec<&AggregatedProof> = chunk.iter().collect();
            next_layer.push(aggregate_with_unified_recursion(&refs, ctx));
        }

        current_layer = next_layer;
        level += 1;
    }

    tracing::info!(
        total_levels = level,
        total_commitments = current_layer[0].commitments.len(),
        "unified recursive aggregation complete"
    );

    current_layer.into_iter().next().unwrap()
}

/// Aggregate two proofs into one (reference mode — no ZKVM proof).
pub fn aggregate_pair(left: &AggregatedProof, right: &AggregatedProof) -> AggregatedProof {
    // Union of commitments.
    let mut commitments = left.commitments.clone();
    commitments.extend_from_slice(&right.commitments);

    tracing::info!(
        left_coms = left.commitments.len(),
        right_coms = right.commitments.len(),
        total_coms = commitments.len(),
        "aggregating proof pair (reference mode)"
    );

    // Collect batch proofs from children.
    let mut batch_proofs = left.batch_proofs.clone();
    batch_proofs.extend_from_slice(&right.batch_proofs);

    AggregatedProof {
        commitments,
        proof: None,
        batch_proofs,
        public_input: None,
    }
}

/// Recursively aggregate a list of proofs into a single proof
/// using a binary tree structure.
///
/// Tree depth is O(log B) for B proofs. All nodes at the same
/// level can be proved in parallel.
pub fn aggregate_proofs(proofs: Vec<AggregatedProof>) -> AggregatedProof {
    assert!(!proofs.is_empty(), "must have at least one proof");

    if proofs.len() == 1 {
        return proofs.into_iter().next().unwrap();
    }

    let mut current_layer = proofs;
    let mut level = 0;

    while current_layer.len() > 1 {
        tracing::info!(
            level,
            proofs = current_layer.len(),
            "aggregation level"
        );

        let mut next_layer = Vec::new();
        let mut iter = current_layer.into_iter();

        while let Some(left) = iter.next() {
            match iter.next() {
                Some(right) => {
                    next_layer.push(aggregate_pair(&left, &right));
                }
                None => {
                    // Odd one out — carry forward to the next level.
                    next_layer.push(left);
                }
            }
        }

        current_layer = next_layer;
        level += 1;
    }

    tracing::info!(
        total_levels = level,
        total_commitments = current_layer[0].commitments.len(),
        "aggregation complete"
    );

    current_layer.into_iter().next().unwrap()
}

