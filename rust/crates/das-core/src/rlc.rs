//! Random Linear Combination (RLC) for codeword accumulation.
//!
//! Given codewords b_1, ..., b_s and challenges r_1, ..., r_s ∈ E,
//! the RLC is c*[j] = Σ r_i · b_i[j], where each b_i[j] is embedded
//! from F into E via the canonical embedding.

use backend::{BasedVectorSpace, PrimeCharacteristicRing};

use crate::types::{Codeword, Commitment, EF};

/// Derive an RLC challenge r ∈ E from a commitment.
///
/// Interprets 5 of the 8 digest elements as a single extension-field element.
/// Since |digest| = 8 ≥ 5, no additional hashing is needed.
pub fn derive_challenge(commitment: &Commitment) -> EF {
    let root = &commitment.root;
    // Interpret the first 5 elements of the digest as an EF element.
    // QuinticExtensionFieldKB is represented as [F; 5].
    EF::from_basis_coefficients_slice(&root[..5]).expect("digest has >= 5 elements")
}

/// Compute the RLC of a set of codewords with the given challenges.
///
/// Returns c*[j] = Σ_i r_i · embed(b_i[j]) for each position j.
/// All arithmetic is over the extension field E.
pub fn compute_rlc(codewords: &[Codeword], challenges: &[EF]) -> Vec<EF> {
    assert_eq!(codewords.len(), challenges.len());
    assert!(!codewords.is_empty());

    let n = codewords[0].len();
    for cw in codewords {
        assert_eq!(cw.len(), n, "all codewords must have the same length");
    }

    let mut result = vec![EF::ZERO; n];

    for (cw, &r) in codewords.iter().zip(challenges.iter()) {
        for j in 0..n {
            // Embed b_i[j] ∈ F into E, then multiply by r_i ∈ E.
            let embedded = EF::from(cw.evals[j]);
            result[j] = result[j] + r * embedded;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::F;

    #[test]
    fn test_rlc_single_codeword() {
        // With a single codeword and challenge r=1, RLC should equal
        // the embedded codeword.
        let cw = Codeword::new(vec![F::ONE, F::TWO, F::from_u8(3)]);
        let r = EF::ONE;
        let result = compute_rlc(&[cw.clone()], &[r]);

        for j in 0..3 {
            assert_eq!(result[j], EF::from(cw.evals[j]));
        }
    }

    #[test]
    fn test_rlc_linearity() {
        // RLC of two identical codewords with challenges r1, r2
        // should equal (r1 + r2) * embed(codeword).
        let vals = vec![F::ONE, F::TWO];
        let cw = Codeword::new(vals);
        let r1 = EF::from(F::TWO);
        let r2 = EF::from(F::from_u8(3));

        let result = compute_rlc(&[cw.clone(), cw.clone()], &[r1, r2]);
        let expected_coeff = r1 + r2;

        for j in 0..2 {
            let expected = expected_coeff * EF::from(cw.evals[j]);
            assert_eq!(result[j], expected);
        }
    }
}
