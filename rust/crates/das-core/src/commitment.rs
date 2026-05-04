//! Merkle tree commitment scheme for codewords using Poseidon2.
//!
//! Each leaf packs `EVALS_PER_LEAF` consecutive evaluations, which are
//! hashed via the Poseidon2 sponge into a digest. The leaves form a
//! binary Merkle tree whose root serves as the commitment.

use backend::merkle::MerkleTree;
use utils::{get_poseidon16, poseidon16_compress_pair, Poseidon16};

use crate::types::{Codeword, Commitment, Opening, F, DIGEST_SIZE, EVALS_PER_LEAF, MIN_EVALS_PER_LEAF};

/// Left-to-right sponge hash of a leaf's evaluations.
///
/// Splits `data` into chunks of `DIGEST_SIZE` (=8) and chains:
///   compress(compress(compress(chunk0, chunk1), chunk2), chunk3)
///
/// This matches the `poseidon_chain` precompile semantics used in the circuit.
pub fn hash_leaf_ltr(data: &[F]) -> [F; DIGEST_SIZE] {
    assert!(data.len() >= 2 * DIGEST_SIZE, "leaf must have at least 2 chunks");
    assert!(data.len() % DIGEST_SIZE == 0, "leaf size must be a multiple of DIGEST_SIZE");

    let chunks: Vec<&[F]> = data.chunks(DIGEST_SIZE).collect();
    let mut acc: [F; DIGEST_SIZE] = poseidon16_compress_pair(
        chunks[0].try_into().unwrap(),
        chunks[1].try_into().unwrap(),
    );
    for chunk in &chunks[2..] {
        acc = poseidon16_compress_pair(&acc, (*chunk).try_into().unwrap());
    }
    acc
}

/// Build a Merkle tree from a codeword's evaluations with a given evals-per-leaf.
///
/// Uses left-to-right sponge hashing to match the circuit's `poseidon_chain`.
fn build_merkle_tree_with_epl(codeword: &Codeword, epl: usize) -> MerkleTree<F, DIGEST_SIZE> {
    let n = codeword.evals.len();
    assert!(n > 0, "codeword must be non-empty");
    assert!(n.is_power_of_two(), "codeword length must be a power of 2");
    assert!(epl >= MIN_EVALS_PER_LEAF, "epl must be at least {MIN_EVALS_PER_LEAF}");
    assert!(epl % 8 == 0, "epl must be a multiple of 8");
    assert!(n % epl == 0, "codeword length must be divisible by epl");

    let comp = get_poseidon16();

    let first_layer: Vec<[F; DIGEST_SIZE]> = codeword
        .evals
        .chunks(epl)
        .map(|chunk| hash_leaf_ltr(chunk))
        .collect();

    MerkleTree::from_first_layer::<F, Poseidon16, 16>(comp, first_layer)
}

/// Build a Merkle tree using the default EVALS_PER_LEAF.
fn build_merkle_tree(codeword: &Codeword) -> MerkleTree<F, DIGEST_SIZE> {
    build_merkle_tree_with_epl(codeword, EVALS_PER_LEAF)
}

/// Commit to a codeword by building a Poseidon2 Merkle tree (default EPL).
pub fn commit_codeword(codeword: &Codeword) -> Commitment {
    let tree = build_merkle_tree(codeword);
    Commitment { root: tree.root() }
}

/// Commit to a codeword with a specific evals-per-leaf.
pub fn commit_codeword_epl(codeword: &Codeword, epl: usize) -> Commitment {
    let tree = build_merkle_tree_with_epl(codeword, epl);
    Commitment { root: tree.root() }
}

/// Commit to a codeword and return both the commitment and the tree
/// (needed for opening positions later). Uses default EPL.
pub fn commit_codeword_with_tree(codeword: &Codeword) -> (Commitment, MerkleTree<F, DIGEST_SIZE>) {
    commit_codeword_with_tree_epl(codeword, EVALS_PER_LEAF)
}

/// Commit to a codeword and return both the commitment and the tree,
/// with a specific evals-per-leaf.
pub fn commit_codeword_with_tree_epl(codeword: &Codeword, epl: usize) -> (Commitment, MerkleTree<F, DIGEST_SIZE>) {
    let tree = build_merkle_tree_with_epl(codeword, epl);
    let commitment = Commitment { root: tree.root() };
    (commitment, tree)
}

/// Open a codeword at a specific evaluation position, given its Merkle tree.
/// Uses the default EVALS_PER_LEAF.
pub fn open_position(
    codeword: &Codeword,
    tree: &MerkleTree<F, DIGEST_SIZE>,
    position: usize,
) -> Opening {
    open_position_epl(codeword, tree, position, EVALS_PER_LEAF)
}

/// Open a codeword at a specific evaluation position with a specific evals-per-leaf.
///
/// Returns the leaf containing that position (all `epl` values
/// in the chunk) plus the authentication path.
pub fn open_position_epl(
    codeword: &Codeword,
    tree: &MerkleTree<F, DIGEST_SIZE>,
    position: usize,
    epl: usize,
) -> Opening {
    let n = codeword.evals.len();
    assert!(position < n, "position out of bounds");

    let leaf_index = position / epl;
    let num_leaves = n / epl;
    let log_height = num_leaves.ilog2() as usize;
    let siblings = tree.open_siblings(leaf_index, log_height);

    let start = leaf_index * epl;
    let values = codeword.evals[start..start + epl].to_vec();

    Opening {
        leaf_index,
        values,
        siblings,
    }
}

/// Verify an opening against a commitment (default EPL).
pub fn verify_opening(commitment: &Commitment, opening: &Opening, codeword_length: usize) -> bool {
    verify_opening_epl(commitment, opening, codeword_length, EVALS_PER_LEAF)
}

/// Verify an opening against a commitment with a specific evals-per-leaf.
///
/// Recomputes the leaf hash (left-to-right sponge, matching circuit) and walks
/// the authentication path to check it matches the commitment root.
pub fn verify_opening_epl(
    commitment: &Commitment,
    opening: &Opening,
    codeword_length: usize,
    epl: usize,
) -> bool {
    let num_leaves = codeword_length / epl;
    let log_height = num_leaves.ilog2() as usize;

    if opening.siblings.len() != log_height {
        return false;
    }

    // Leaf hash using LTR sponge (matches poseidon_chain in circuit)
    let mut node = hash_leaf_ltr(&opening.values);

    // Walk the Merkle proof
    let mut index = opening.leaf_index;
    for sibling in &opening.siblings {
        let (left, right) = if index & 1 == 0 {
            (node, *sibling)
        } else {
            (*sibling, node)
        };
        node = poseidon16_compress_pair(&left, &right);
        index >>= 1;
    }

    node == commitment.root
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::PrimeCharacteristicRing;

    #[test]
    fn test_commit_open_verify_roundtrip() {
        // Codeword of length 32 → 2 leaves of 16 evals each.
        let n = 32;
        let evals: Vec<F> = (0..n).map(|i| F::from_usize(i + 1)).collect();
        let cw = Codeword::new(evals);

        let (commitment, tree) = commit_codeword_with_tree(&cw);

        // Open and verify every position.
        for pos in 0..n {
            let opening = open_position(&cw, &tree, pos);
            assert_eq!(opening.values.len(), EVALS_PER_LEAF);
            assert!(opening.values.contains(&cw.evals[pos]));
            assert!(
                verify_opening(&commitment, &opening, n),
                "opening at position {pos} should verify"
            );
        }
    }

    #[test]
    fn test_tampered_opening_fails() {
        let n = 32;
        let evals: Vec<F> = (0..n).map(|i| F::from_usize(i + 10)).collect();
        let cw = Codeword::new(evals);

        let (commitment, tree) = commit_codeword_with_tree(&cw);
        let mut opening = open_position(&cw, &tree, 2);

        // Tamper with one value in the leaf.
        opening.values[0] = F::from_usize(999);
        assert!(
            !verify_opening(&commitment, &opening, n),
            "tampered opening should not verify"
        );
    }

    #[test]
    fn test_different_codewords_different_commitments() {
        let cw1 = Codeword::new(vec![F::ONE; 16]);
        let cw2 = Codeword::new(vec![F::TWO; 16]);

        let com1 = commit_codeword(&cw1);
        let com2 = commit_codeword(&cw2);

        assert_ne!(com1, com2);
    }

    #[test]
    fn test_commit_open_verify_epl32() {
        // Test with EPL=32: sequential compress (4 chunks of 8).
        let n = 128;
        let epl = 32;
        let evals: Vec<F> = (0..n).map(|i| F::from_usize(i + 1)).collect();
        let cw = Codeword::new(evals);

        let (commitment, tree) = commit_codeword_with_tree_epl(&cw, epl);

        // Open and verify several positions.
        for pos in [0, 15, 31, 32, 63, 96, 127] {
            let opening = open_position_epl(&cw, &tree, pos, epl);
            assert_eq!(opening.values.len(), epl);
            assert!(opening.values.contains(&cw.evals[pos]));
            assert!(
                verify_opening_epl(&commitment, &opening, n, epl),
                "opening at position {pos} with EPL={epl} should verify"
            );
        }
    }

    #[test]
    fn test_commit_open_verify_epl64() {
        // Test with EPL=64: sequential compress (8 chunks of 8).
        let n = 256;
        let epl = 64;
        let evals: Vec<F> = (0..n).map(|i| F::from_usize(i + 1)).collect();
        let cw = Codeword::new(evals);

        let (commitment, tree) = commit_codeword_with_tree_epl(&cw, epl);

        for pos in [0, 33, 63, 64, 128, 200, 255] {
            let opening = open_position_epl(&cw, &tree, pos, epl);
            assert_eq!(opening.values.len(), epl);
            assert!(
                verify_opening_epl(&commitment, &opening, n, epl),
                "opening at position {pos} with EPL={epl} should verify"
            );
        }
    }

    #[test]
    fn test_tampered_opening_epl32_fails() {
        let n = 128;
        let epl = 32;
        let evals: Vec<F> = (0..n).map(|i| F::from_usize(i + 10)).collect();
        let cw = Codeword::new(evals);

        let (commitment, tree) = commit_codeword_with_tree_epl(&cw, epl);
        let mut opening = open_position_epl(&cw, &tree, 2, epl);
        opening.values[0] = F::from_usize(999);
        assert!(!verify_opening_epl(&commitment, &opening, n, epl));
    }
}
