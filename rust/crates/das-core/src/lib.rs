//! # das-core
//!
//! Core types and algorithms for leanDAS: RLC accumulation,
//! Merkle commitments, and FRI folding on evaluation vectors.

pub mod commitment;
pub mod fold;
pub mod fri;
pub mod polynomial;
pub mod rlc;
pub mod types;

pub use commitment::{
    commit_codeword, commit_codeword_epl, commit_codeword_with_tree, commit_codeword_with_tree_epl,
    hash_leaf_ltr, open_position, open_position_epl, verify_opening, verify_opening_epl,
};
pub use fold::{FoldingPlan, FoldedLayer, precompute_folding_plan, apply_folding_plan, check_rs_precomputed, challenge_from_digest};
pub use fri::{fri_fold_evaluations, check_rs_membership_iterative};
pub use polynomial::{rs_encode, Domain, Polynomial};
pub use rlc::{compute_rlc, derive_challenge};
pub use types::{Codeword, Commitment, Opening, F, EF, DIGEST_SIZE, EVALS_PER_LEAF, MIN_EVALS_PER_LEAF, compute_evals_per_leaf};
