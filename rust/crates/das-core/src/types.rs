//! Core types for leanDAS.

use backend::KoalaBear;
use backend::QuinticExtensionFieldKB;

/// Base field: KoalaBear (p = 2^31 - 2^24 + 1).
pub type F = KoalaBear;

/// Extension field: GF(p^5), |E| ≈ 2^155.
pub type EF = QuinticExtensionFieldKB;

/// Poseidon2 digest size in base-field elements.
pub const DIGEST_SIZE: usize = 8;

/// A codeword: evaluations of a degree-< k polynomial over a domain of size n.
/// Each element is a base-field element in F.
#[derive(Clone, Debug)]
pub struct Codeword {
    /// Evaluation vector: v[j] = p(ω^j) for j = 0..n-1.
    pub evals: Vec<F>,
}

impl Codeword {
    pub fn new(evals: Vec<F>) -> Self {
        Self { evals }
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }
}

/// A Merkle commitment to a codeword: the Poseidon2 Merkle root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub root: [F; DIGEST_SIZE],
}

/// Minimum evaluations per leaf (matches Poseidon2 sponge minimum: 2 × RATE=8).
pub const MIN_EVALS_PER_LEAF: usize = 16;

/// Default evaluations per leaf.
/// For small codewords (≤128), use the minimum (16).
/// For larger codewords, use 64 to reduce tree depth.
pub const DEFAULT_EVALS_PER_LEAF: usize = 16;

/// Compute a reasonable evals-per-leaf for a given codeword length.
///
/// Returns a power-of-2 value that is at least `MIN_EVALS_PER_LEAF` (16),
/// at most `codeword_len`, and a multiple of 8.
pub fn compute_evals_per_leaf(codeword_len: usize) -> usize {
    if codeword_len <= 128 {
        MIN_EVALS_PER_LEAF
    } else {
        // For large codewords, use 64 evals per leaf.
        // This reduces tree depth by 2 levels compared to 16.
        64.min(codeword_len)
    }
}

/// Legacy constant for backward compatibility.
pub const EVALS_PER_LEAF: usize = DEFAULT_EVALS_PER_LEAF;

/// A Merkle opening proof for a leaf (chunk of evaluations).
#[derive(Clone, Debug)]
pub struct Opening {
    /// The leaf index (evaluation indices leaf_index * EVALS_PER_LEAF .. +EVALS_PER_LEAF).
    pub leaf_index: usize,
    /// The evaluation values in this leaf.
    pub values: Vec<F>,
    /// Merkle authentication path (sibling digests from leaf to root).
    pub siblings: Vec<[F; DIGEST_SIZE]>,
}

/// A batch of codewords assigned to a single prover.
#[derive(Clone, Debug)]
pub struct Batch {
    /// Indices into the global codeword array.
    pub indices: Vec<usize>,
    /// The codewords in this batch.
    pub codewords: Vec<Codeword>,
    /// Commitments for each codeword.
    pub commitments: Vec<Commitment>,
    /// RLC challenges derived from commitments.
    pub challenges: Vec<EF>,
}
