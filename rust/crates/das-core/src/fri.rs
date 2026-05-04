//! FRI folding on evaluation vectors.
//!
//! All folding operates entirely on evaluation vectors — no polynomial
//! interpolation or evaluation at arbitrary points. Given evaluations on
//! a multiplicative domain D, the even/odd parts are extracted by pairing
//! evaluations at ω^i and -ω^i, and the folded evaluations are computed
//! by a linear combination with proper twiddle factor division.

use crate::types::{EF, F};
use backend::{Field, PrimeCharacteristicRing, TwoAdicField};

/// Perform one round of FRI folding on an evaluation vector.
///
/// Given evaluations `evals` of a polynomial p on a domain D of size n
/// (where n is even) with generator `domain_gen`, and a folding challenge `alpha`:
///
///   p_even(x²) = (p(x) + p(-x)) / 2
///   p_odd(x²)  = (p(x) - p(-x)) / (2x)
///   p_fold(x²) = p_even(x²) + alpha * p_odd(x²)
///
/// The evaluations are paired: evals[i] = p(ω^i) and evals[i + n/2] = p(-ω^i)
/// (since ω^(n/2) = -1 in a multiplicative group of order n).
///
/// `inv_twiddles` must contain [(2·ω^0)^{-1}, (2·ω^1)^{-1}, ..., (2·ω^{n/2-1})^{-1}]
/// in the extension field. These combine the /2 and /ω^i into one multiplication.
///
/// Returns the folded evaluation vector of length n/2.
pub fn fri_fold_round(evals: &[EF], alpha: EF, inv_twiddles: &[EF]) -> Vec<EF> {
    let n = evals.len();
    assert!(n >= 2, "need at least 2 evaluations to fold");
    assert!(n % 2 == 0, "evaluation count must be even");

    let half = n / 2;
    assert_eq!(inv_twiddles.len(), half);

    let two_inv = EF::TWO.inverse();
    let mut folded = Vec::with_capacity(half);

    for i in 0..half {
        let p_pos = evals[i]; // p(ω^i)
        let p_neg = evals[i + half]; // p(-ω^i) = p(ω^(i + n/2))

        // Even part: (p(ω^i) + p(-ω^i)) / 2
        let p_even = (p_pos + p_neg) * two_inv;

        // Odd part: (p(ω^i) - p(-ω^i)) / (2 · ω^i)
        let p_odd = (p_pos - p_neg) * inv_twiddles[i];

        folded.push(p_even + alpha * p_odd);
    }

    folded
}

/// Compute the inverse twiddle factors for a domain of the given size.
///
/// Returns [(2·ω^0)^{-1}, (2·ω^1)^{-1}, ..., (2·ω^{n/2-1})^{-1}] as EF elements.
fn compute_inv_twiddles(domain_gen: F, half: usize) -> Vec<EF> {
    let two = EF::TWO;
    let mut twiddles = Vec::with_capacity(half);
    let mut omega_i = F::ONE;
    for _ in 0..half {
        // (2 · ω^i)^{-1} in EF
        let tw = two * EF::from(omega_i);
        twiddles.push(tw.inverse());
        omega_i = omega_i * domain_gen;
    }
    twiddles
}

/// Perform complete FRI folding: repeatedly halve the evaluation vector
/// until a single constant remains.
///
/// `domain_size` is the size of the evaluation domain (must equal evals.len()).
/// `challenges` must have length log2(n). Returns the final constant.
///
/// If the original polynomial had degree < n/2 (RS rate 1/2), after
/// log2(n) rounds the result is a single value. For a degree-0 result,
/// the polynomial was in the RS code.
pub fn fri_fold_evaluations(evals: &[EF], challenges: &[EF]) -> EF {
    let n = evals.len();
    assert!(n.is_power_of_two(), "evaluation count must be a power of 2");

    let num_rounds = n.ilog2() as usize;
    assert_eq!(
        challenges.len(),
        num_rounds,
        "need exactly log2(n) folding challenges"
    );

    let bits = num_rounds;
    let mut current = evals.to_vec();
    let mut current_gen = F::two_adic_generator(bits);

    for (round, &alpha) in challenges.iter().enumerate() {
        let half = current.len() / 2;
        let inv_twiddles = compute_inv_twiddles(current_gen, half);
        current = fri_fold_round(&current, alpha, &inv_twiddles);
        // Next round's domain generator is the square of the current one.
        current_gen = current_gen * current_gen;
        debug_assert_eq!(
            current.len(),
            n >> (round + 1),
            "folded size mismatch at round {round}"
        );
    }

    assert_eq!(current.len(), 1);
    current[0]
}

/// Check whether a polynomial (given as evaluations on a domain of size n)
/// is of degree < n/2 by performing FRI folding to size 2 and verifying
/// the two remaining values are equal (constant polynomial check).
///
/// Uses `log2(n) - 1` challenges to fold from n to 2. Returns true if
/// the two remaining values match.
pub fn check_rs_membership(evals: &[EF], challenges: &[EF]) -> bool {
    let n = evals.len();
    assert!(n.is_power_of_two() && n >= 2);

    let num_rounds = n.ilog2() as usize - 1; // fold to size 2
    assert!(
        challenges.len() >= num_rounds,
        "need at least {} challenges, got {}",
        num_rounds,
        challenges.len()
    );

    let bits = n.ilog2() as usize;
    let mut current = evals.to_vec();
    let mut current_gen = F::two_adic_generator(bits);

    for round in 0..num_rounds {
        let half = current.len() / 2;
        let inv_twiddles = compute_inv_twiddles(current_gen, half);
        current = fri_fold_round(&current, challenges[round], &inv_twiddles);
        current_gen = current_gen * current_gen;
    }

    assert_eq!(current.len(), 2);
    current[0] == current[1]
}

/// Perform one round of FRI folding with iterative twiddle computation.
///
/// Computes the standard FRI fold but derives inverse twiddles on the fly:
///   inv_tw_0 = half = (1/2)
///   inv_tw_{i+1} = inv_tw_i · ω^{-1}
///
/// This matches the leanDAS circuit: 6 ops per element
/// (add_ee sum, add_ee diff, dot_product_be half*sum, dot_product_be inv_tw*diff,
///  dot_product_ee alpha*odd, add_ee combine) plus 1 dot_product_be for the
/// twiddle update.
///
/// Public input only needs `half` and one `ω^{-1}` per round.
pub fn fri_fold_round_iterative(evals: &[EF], alpha: EF, omega_inv: EF) -> Vec<EF> {
    let n = evals.len();
    assert!(n >= 2 && n % 2 == 0);

    let half = n / 2;
    let half_val = EF::TWO.inverse();
    let mut inv_tw = half_val; // inv_tw_0 = 1/2 (since ω^0 = 1)
    let mut folded = Vec::with_capacity(half);

    for i in 0..half {
        let p_pos = evals[i];
        let p_neg = evals[i + half];
        let p_sum = p_pos + p_neg;
        let p_diff = p_pos - p_neg;
        let p_even = half_val * p_sum;
        let p_odd = inv_tw * p_diff;
        folded.push(p_even + alpha * p_odd);
        inv_tw = inv_tw * omega_inv; // next twiddle
    }

    folded
}

/// Check RS membership using iterative-twiddle FRI folding.
///
/// Folds from n evaluations down to 2 using `log2(n) - 1` rounds,
/// computing twiddle factors on the fly from the domain generator inverse.
/// Then checks the two remaining values are equal (constant polynomial).
///
/// This is the reference implementation of what the leanDAS circuit proves.
pub fn check_rs_membership_iterative(evals: &[EF], challenges: &[EF]) -> bool {
    let n = evals.len();
    assert!(n.is_power_of_two() && n >= 2);

    let num_rounds = n.ilog2() as usize - 1;
    assert!(
        challenges.len() >= num_rounds,
        "need at least {} challenges, got {}",
        num_rounds,
        challenges.len()
    );

    let bits = n.ilog2() as usize;
    let mut current = evals.to_vec();
    let mut domain_gen = F::two_adic_generator(bits);

    for round in 0..num_rounds {
        let omega_inv = EF::from(domain_gen.inverse());
        current = fri_fold_round_iterative(&current, challenges[round], omega_inv);
        domain_gen = domain_gen * domain_gen;
    }

    assert_eq!(current.len(), 2);
    current[0] == current[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_constant_polynomial() {
        // A constant polynomial p(x) = c has evals [c, c, c, c].
        // After folding, the result should be c regardless of challenge.
        let c = EF::from(F::from_u8(42));
        let evals = vec![c; 4];
        let challenges = vec![EF::from(F::from_u8(7)), EF::from(F::from_u8(13))];

        let result = fri_fold_evaluations(&evals, &challenges);
        assert_eq!(result, c);
    }

    #[test]
    fn test_fold_round_halves_size() {
        let n = 8;
        let omega = F::two_adic_generator(3); // 8th root
        let inv_tw = compute_inv_twiddles(omega, 4);
        let evals = vec![EF::ONE; n];
        let folded = fri_fold_round(&evals, EF::ONE, &inv_tw);
        assert_eq!(folded.len(), 4);
    }

    #[test]
    fn test_check_rs_membership_valid() {
        // A valid RS codeword (from a degree-< n/2 polynomial) should pass.
        use crate::polynomial::rs_encode;
        let msg: Vec<F> = (0..8).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, 32); // degree < 8 < 32/2 = 16 ✓
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();

        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership(&evals, &challenges));
    }

    #[test]
    fn test_check_rs_membership_valid_max_degree() {
        // Degree exactly n/2 - 1 (the maximum for rate 1/2 RS).
        use crate::polynomial::rs_encode;
        let k = 16; // degree < 16
        let n = 32; // domain size
        let msg: Vec<F> = (0..k).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, n);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();

        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership(&evals, &challenges));
    }

    #[test]
    fn test_check_rs_membership_invalid() {
        // An arbitrary non-RS vector should fail with high probability.
        let evals: Vec<EF> = (0..16)
            .map(|i| EF::from(F::from_usize(i * 17 + 3)))
            .collect();

        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 13 + 7)))
            .collect();
        assert!(!check_rs_membership(&evals, &challenges));
    }

    #[test]
    fn test_fri_full_fold_valid_polynomial() {
        // Full fold of a valid codeword should return a well-defined value.
        use crate::polynomial::rs_encode;
        let msg: Vec<F> = (0..4).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, 16);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();

        let challenges: Vec<EF> = (0..4)
            .map(|i| EF::from(F::from_usize(i * 3 + 1)))
            .collect();
        let result = fri_fold_evaluations(&evals, &challenges);

        // The result should be deterministic.
        let result2 = fri_fold_evaluations(&evals, &challenges);
        assert_eq!(result, result2);
    }

    // ── Twiddle-free fold tests ──────────────────────────────────────

    #[test]
    fn test_iterative_constant_polynomial() {
        let c = EF::from(F::from_u8(42));
        let evals = vec![c; 8];
        let challenges: Vec<EF> = (0..5).map(|i| EF::from(F::from_usize(i + 1))).collect();
        assert!(check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_valid_rs_degree4() {
        use crate::polynomial::rs_encode;
        let msg: Vec<F> = (0..4).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, 32);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_valid_rs_degree8() {
        use crate::polynomial::rs_encode;
        let msg: Vec<F> = (0..8).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, 32);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_valid_rs_max_degree() {
        // degree exactly n/2 - 1 (maximum for rate 1/2 RS)
        use crate::polynomial::rs_encode;
        let k = 16;
        let n = 32;
        let msg: Vec<F> = (0..k).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, n);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_invalid_rs_rejected() {
        let n = 16;
        let evals: Vec<EF> = (0..n)
            .map(|i| EF::from(F::from_usize(i * 17 + 3)))
            .collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 13 + 7)))
            .collect();
        assert!(!check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_tampered_codeword_rejected() {
        use crate::polynomial::rs_encode;
        let msg: Vec<F> = (0..8).map(|i| F::from_usize(i + 1)).collect();
        let mut cw = rs_encode(&msg, 32);
        cw.evals[5] = cw.evals[5] + F::ONE; // tamper one evaluation
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(!check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_agrees_with_standard_on_validity() {
        // Both fold variants should agree on RS membership for valid codewords.
        use crate::polynomial::rs_encode;
        for degree in [1, 4, 8, 15] {
            let n = 32;
            let msg: Vec<F> = (0..degree).map(|i| F::from_usize(i + 1)).collect();
            let cw = rs_encode(&msg, n);
            let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
            let challenges: Vec<EF> = (0..10)
                .map(|i| EF::from(F::from_usize(i * 7 + 3)))
                .collect();
            assert!(
                check_rs_membership(&evals, &challenges),
                "standard fold should accept degree {degree}"
            );
            assert!(
                check_rs_membership_iterative(&evals, &challenges),
                "twiddle-free fold should accept degree {degree}"
            );
        }
    }

    #[test]
    fn test_iterative_agrees_with_standard_on_invalidity() {
        // Both fold variants should agree on rejection.
        for seed in [3, 17, 99, 1234] {
            let n = 16;
            let evals: Vec<EF> = (0..n)
                .map(|i| EF::from(F::from_usize(i * seed + 3)))
                .collect();
            let challenges: Vec<EF> = (0..10)
                .map(|i| EF::from(F::from_usize(i * 13 + 7)))
                .collect();
            let std_result = check_rs_membership(&evals, &challenges);
            let free_result = check_rs_membership_iterative(&evals, &challenges);
            assert_eq!(
                std_result, free_result,
                "both variants should agree for seed {seed}"
            );
        }
    }

    #[test]
    fn test_iterative_large_domain() {
        // Test at a larger scale: n=1024, degree=256
        use crate::polynomial::rs_encode;
        let k = 256;
        let n = 1024;
        let msg: Vec<F> = (0..k).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, n);
        let evals: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();
        let challenges: Vec<EF> = (0..20)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();
        assert!(check_rs_membership_iterative(&evals, &challenges));
    }

    #[test]
    fn test_iterative_with_rlc() {
        // End-to-end: RLC of multiple codewords, then twiddle-free fold.
        // This is exactly what the circuit does.
        use crate::polynomial::rs_encode;
        use crate::rlc::{compute_rlc, derive_challenge};
        use crate::commitment::commit_codeword;

        let n = 32;
        let m = 4;
        let degree = 8;

        let codewords: Vec<crate::types::Codeword> = (0..m)
            .map(|i| {
                let msg: Vec<F> = (0..degree)
                    .map(|j| F::from_usize((i + 1) * (j + 1) + i * 7 + 3))
                    .collect();
                rs_encode(&msg, n)
            })
            .collect();

        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword(cw)).collect();
        let challenges: Vec<EF> = commitments.iter().map(|c| derive_challenge(c)).collect();
        let rlc = compute_rlc(&codewords, &challenges);

        // RLC of valid RS codewords is still a valid RS codeword.
        let fold_challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 11 + 5)))
            .collect();
        assert!(check_rs_membership_iterative(&rlc, &fold_challenges));
    }
}
