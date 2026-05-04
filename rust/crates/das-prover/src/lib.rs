//! # das-prover
//!
//! Proof generation for leanDAS. Two-phase architecture:
//!
//! **Phase 1 — Proving (RLC + FRI):** Each prover takes a batch of
//! codewords, computes the RLC, performs FRI folding inside the ZKVM,
//! and produces a STARK proof certifying both RLC correctness and
//! exact RS membership.
//!
//! **Phase 2 — Proof aggregation:** Proofs are recursively aggregated
//! in a binary tree. Each aggregation step verifies two child proofs
//! inside the ZKVM and produces a new proof. At the root, a single
//! proof certifies all m input codewords.

pub mod aggregation;
pub mod batch;
pub mod circuit;
pub mod pipeline;
pub mod recursion;

pub use batch::prove_batch;
pub use aggregation::{
    aggregate_proofs, aggregate_proofs_unified,
    UnifiedAggregationContext,
};
pub use circuit::compile_unified_das_circuit;
pub use pipeline::prove_das;
pub use recursion::{prove_unified_recursion, prove_hybrid_recursion};
