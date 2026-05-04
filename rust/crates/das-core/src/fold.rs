//! Precomputed FRI-style folding with compile-time weight vectors.
//!
//! # Overview
//!
//! Given a codeword of length N and degree bound d, we precompute weight
//! coefficient vectors at compile time (depending only on the domain structure),
//! then combine them with a runtime challenge β = r to compute the final
//! folded layer via dot products.
//!
//! # Mathematical Foundation
//!
//! ## Standard FRI fold with constant challenge β
//!
//! Each round folds domain size N_r to N_r/2:
//!
//! ```text
//! v^{(r+1)}_i = ((1 + β·ω^{-2^r·i})/2) · v^{(r)}_i
//!              + ((1 - β·ω^{-2^r·i})/2) · v^{(r)}_{i + N_r/2}
//! ```
//!
//! After t = log2(d) rounds, output i depends on d ancestors:
//!
//! ```text
//! v^{(t)}_i = Σ_{s=0}^{d-1} w_{i,s}(β) · codeword[i + s·(N/d)]
//! ```
//!
//! ## Weight decomposition
//!
//! Each weight is a degree-t polynomial in β with base-field coefficients:
//!
//! ```text
//! w_{i,s}(β) = Σ_{k=0}^{t} c_{i,s,k} · β^k     (c_{i,s,k} ∈ F)
//! ```
//!
//! The coefficients c_{i,s,k} depend only on twiddle factors ω^{-j} ∈ F
//! and can be precomputed at compile time. At runtime, given β = r ∈ EF:
//!
//! ```text
//! v^{(t)}_i = Σ_{k=0}^{t} r^k · Σ_{s=0}^{d-1} c_{k,i,s} · codeword[i + s·stride]
//! ```
//!
//! Each inner sum is a `dot_product_be(F_weights, EF_data)` — one precompile
//! instruction in the leanVM circuit.
//!
//! ## Efficiency
//!
//! Main trace instructions per output: (t+1) dot_products + t multiplications.
//! Total: (2t+1)*(N/d) main trace rows. Precompile trace: (t+1)*N rows.
//! For N=4096, d=2048, t=11: 46 main trace instructions vs ~16K for butterfly.

use crate::types::{EF, F, DIGEST_SIZE};
use backend::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};

// ─── Data Structures ─────────────────────────────────────────────────────

/// A precomputed folding plan with compile-time weight coefficient vectors.
///
/// `coeffs[k][i]` is a Vec<F> of length `degree`, containing the base-field
/// coefficients of β^k in the weight polynomial for output position i.
#[derive(Clone, Debug)]
pub struct FoldingPlan {
    /// Codeword length N (power of 2).
    pub codeword_len: usize,
    /// Degree bound d (power of 2, d ≤ N).
    pub degree: usize,
    /// Number of fold rounds: t = log2(d).
    pub num_rounds: usize,
    /// Number of outputs in the final layer: N / d.
    pub num_outputs: usize,
    /// Stride between ancestor elements: N / d.
    pub stride: usize,
    /// Coefficient matrices: coeffs[k][i][s] = c_{i,s,k} ∈ F.
    /// k ranges over 0..=num_rounds (t+1 layers).
    /// i ranges over 0..num_outputs.
    /// s ranges over 0..degree (the d ancestors).
    pub coeffs: Vec<Vec<Vec<F>>>,
}

/// The result of applying a folding plan to a codeword.
#[derive(Clone, Debug)]
pub struct FoldedLayer {
    /// The final-layer values: one per output position.
    pub values: Vec<EF>,
}

// ─── Precomputation ──────────────────────────────────────────────────────

/// Precompute a folding plan for the given parameters.
///
/// The plan's coefficient vectors depend only on N, d, and the domain
/// structure — not on the challenge β. This makes them suitable for
/// embedding as compile-time constants in a circuit.
///
/// # Complexity
/// - Time: O(d² · N/d) = O(N·d) — polynomial multiplication per ancestor.
/// - Memory: O((t+1) · N) — the coefficient matrices.
pub fn precompute_folding_plan(codeword_len: usize, degree: usize) -> FoldingPlan {
    assert!(
        codeword_len.is_power_of_two(),
        "codeword_len must be a power of 2, got {codeword_len}"
    );
    assert!(
        degree.is_power_of_two(),
        "degree must be a power of 2, got {degree}"
    );
    assert!(
        degree > 0 && degree <= codeword_len,
        "degree must be in (0, codeword_len], got degree={degree}, N={codeword_len}"
    );

    let num_rounds = degree.ilog2() as usize;
    let num_outputs = codeword_len / degree;
    let stride = num_outputs;

    let log_n = codeword_len.ilog2() as usize;
    let omega = F::two_adic_generator(log_n);
    let omega_inv = omega.inverse();

    // For each output position i, compute the weight polynomials for all d ancestors.
    // We track each ancestor's weight as a polynomial in β: Vec<F> of degree ≤ t.
    //
    // IMPORTANT: We traverse rounds BACKWARD (from last round to first) because
    // the twiddle factor at each round depends on the position in that round's
    // OUTPUT, not the original position. Starting from the final output position i,
    // we expand backward through each round to reach the original input positions.
    //
    // At round r (output → input):
    //   - h = N / 2^{r+1}  (half-size of round-r input)
    //   - twiddle for output position j: tw = ω^{-j·2^r}
    //   - output[j] = (1 + β·tw)/2 · input[j] + (1 - β·tw)/2 · input[j+h]

    let half_f = F::TWO.inverse();

    // Initialize: for each output i, one entry at position i in the final output.
    let mut all_ancestor_polys: Vec<Vec<(usize, Vec<F>)>> = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        all_ancestor_polys.push(vec![(i, vec![F::ONE])]);
    }

    // Traverse rounds backward: last round first.
    for round in (0..num_rounds).rev() {
        let half = codeword_len >> (round + 1);
        // twiddle_base = ω^{-2^round}
        let twiddle_base = omega_inv.exp_u64(1u64 << round);

        for ancestors in all_ancestor_polys.iter_mut() {
            let mut new_ancestors = Vec::with_capacity(ancestors.len() * 2);
            for (pos, poly) in ancestors.drain(..) {
                // tw = ω^{-pos·2^round} — based on the output position at this round.
                let tw = twiddle_base.exp_u64(pos as u64);
                // left factor: [half_f, half_f * tw]
                let left_poly = poly_mul_linear(&poly, half_f, half_f * tw);
                // right factor: [half_f, -half_f * tw]
                let right_poly = poly_mul_linear(&poly, half_f, -(half_f * tw));
                new_ancestors.push((pos, left_poly));
                new_ancestors.push((pos + half, right_poly));
            }
            *ancestors = new_ancestors;
        }
    }

    // Extract coefficient matrices.
    // coeffs[k][i][s] where k = power of β, i = output index, s = ancestor index within block.
    // Sort ancestors by original position first, since backward traversal may produce
    // them in a different order than the canonical stride ordering.
    let t = num_rounds;
    let d = degree;
    let mut coeffs = vec![vec![vec![F::ZERO; d]; num_outputs]; t + 1];

    for (i, ancestors) in all_ancestor_polys.iter_mut().enumerate() {
        assert_eq!(ancestors.len(), d);
        ancestors.sort_by_key(|&(pos, _)| pos);
        for (s, (pos, poly)) in ancestors.iter().enumerate() {
            debug_assert_eq!(*pos, i + s * stride);
            for (k, &coeff) in poly.iter().enumerate() {
                coeffs[k][i][s] = coeff;
            }
        }
    }

    FoldingPlan {
        codeword_len,
        degree,
        num_rounds,
        num_outputs,
        stride,
        coeffs,
    }
}

/// Multiply a polynomial (as Vec<F>) by the linear factor (a + b·β).
/// Result has degree one higher.
fn poly_mul_linear(poly: &[F], a: F, b: F) -> Vec<F> {
    let mut result = vec![F::ZERO; poly.len() + 1];
    for (i, &c) in poly.iter().enumerate() {
        result[i] = result[i] + a * c;
        result[i + 1] = result[i + 1] + b * c;
    }
    result
}

// ─── Evaluation ──────────────────────────────────────────────────────────

/// Apply a precomputed folding plan to a codeword with runtime challenge β.
///
/// For each output position i:
/// `v_i = Σ_{k=0}^{t} β^k · Σ_{s} coeffs[k][i][s] · codeword[i + s·stride]`
///
/// # Complexity
/// - Time: O((t+1) · N) — (t+1) passes over the N ancestor values.
/// - Memory: O(N/d) — the output layer.
pub fn apply_folding_plan(plan: &FoldingPlan, codeword: &[EF], beta: EF) -> FoldedLayer {
    assert_eq!(
        codeword.len(),
        plan.codeword_len,
        "codeword length mismatch: expected {}, got {}",
        plan.codeword_len,
        codeword.len()
    );

    // Precompute powers of beta: [1, β, β², ..., β^t]
    let t = plan.num_rounds;
    let mut beta_powers = Vec::with_capacity(t + 1);
    let mut bp = EF::ONE;
    for _ in 0..=t {
        beta_powers.push(bp);
        bp = bp * beta;
    }

    let mut values = Vec::with_capacity(plan.num_outputs);

    for i in 0..plan.num_outputs {
        let mut acc = EF::ZERO;
        for k in 0..=t {
            // Inner dot product: Σ_s coeffs[k][i][s] · codeword[i + s·stride]
            let mut dot = EF::ZERO;
            for s in 0..plan.degree {
                dot = dot + EF::from(plan.coeffs[k][i][s]) * codeword[i + s * plan.stride];
            }
            acc = acc + beta_powers[k] * dot;
        }
        values.push(acc);
    }

    FoldedLayer { values }
}

/// Apply a folding plan to base-field evaluations (auto-embeds into EF).
pub fn apply_folding_plan_base(plan: &FoldingPlan, codeword: &[F], beta: EF) -> FoldedLayer {
    assert_eq!(codeword.len(), plan.codeword_len);
    let ef_codeword: Vec<EF> = codeword.iter().map(|&v| EF::from(v)).collect();
    apply_folding_plan(plan, &ef_codeword, beta)
}

// ─── Constantness Check ──────────────────────────────────────────────────

/// Check whether the folded layer is constant (all values equal).
pub fn is_constant(layer: &FoldedLayer) -> bool {
    if layer.values.len() <= 1 {
        return true;
    }
    let first = layer.values[0];
    layer.values[1..].iter().all(|&v| v == first)
}

/// Check RS membership via precomputed folding with challenge β.
///
/// Returns true if the codeword, after t = log2(degree) rounds of FRI
/// folding with constant challenge β, produces a constant final layer.
pub fn check_rs_precomputed(codeword: &[EF], degree: usize, beta: EF) -> bool {
    let plan = precompute_folding_plan(codeword.len(), degree);
    let layer = apply_folding_plan(&plan, codeword, beta);
    is_constant(&layer)
}

// ─── Challenge Derivation ────────────────────────────────────────────────

/// Derive a folding challenge from a Poseidon digest.
///
/// Interprets the first 5 elements of the digest as an extension field element.
/// This is the same encoding used by `derive_challenge` in rlc.rs.
pub fn challenge_from_digest(digest: &[F; DIGEST_SIZE]) -> EF {
    EF::from_basis_coefficients_slice(&digest[..5]).expect("digest has >= 5 elements")
}

// ─── Iterative Reference Implementation ──────────────────────────────────

/// Perform t rounds of iterative FRI folding with constant challenge β.
///
/// This is the standard FRI butterfly with proper twiddle factors,
/// used as a reference for testing the precomputed approach.
pub fn fold_iterative(codeword: &[EF], num_rounds: usize, beta: EF) -> Vec<EF> {
    let n = codeword.len();
    assert!(n.is_power_of_two());

    let log_n = n.ilog2() as usize;
    let omega = F::two_adic_generator(log_n);
    let half = EF::TWO.inverse();

    let mut current = codeword.to_vec();

    for round in 0..num_rounds {
        let cur_n = current.len();
        assert!(cur_n >= 2 && cur_n % 2 == 0);
        let h = cur_n / 2;

        // twiddle: ω^{-2^round}
        let omega_inv_round = EF::from(omega.inverse().exp_u64(1u64 << round));

        let mut inv_tw = half; // starts at (1/2) · ω^{-2^round · 0} = 1/2
        let mut next = Vec::with_capacity(h);

        for i in 0..h {
            let p_pos = current[i];
            let p_neg = current[i + h];
            let p_sum = p_pos + p_neg;
            let p_diff = p_pos - p_neg;
            let p_even = half * p_sum;
            let p_odd = inv_tw * p_diff;
            next.push(p_even + beta * p_odd);
            inv_tw = inv_tw * omega_inv_round;
        }

        current = next;
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::{rs_encode, Domain, Polynomial};

    fn embed(cw: &[F]) -> Vec<EF> {
        cw.iter().map(|&v| EF::from(v)).collect()
    }

    /// A fixed nonzero beta for tests.
    fn test_beta() -> EF {
        EF::from_basis_coefficients_slice(&[
            F::from_u8(7),
            F::from_u8(13),
            F::from_u8(3),
            F::from_u8(19),
            F::from_u8(11),
        ])
        .unwrap()
    }

    // ── Plan structure tests ─────────────────────────────────────────

    #[test]
    fn test_plan_n8_d2() {
        let plan = precompute_folding_plan(8, 2);
        assert_eq!(plan.num_rounds, 1);
        assert_eq!(plan.num_outputs, 4);
        assert_eq!(plan.stride, 4);
        assert_eq!(plan.coeffs.len(), 2); // t+1 = 2 layers
        assert_eq!(plan.coeffs[0].len(), 4); // 4 outputs
        assert_eq!(plan.coeffs[0][0].len(), 2); // d=2 ancestors
    }

    #[test]
    fn test_plan_n16_d4() {
        let plan = precompute_folding_plan(16, 4);
        assert_eq!(plan.num_rounds, 2);
        assert_eq!(plan.num_outputs, 4);
        assert_eq!(plan.coeffs.len(), 3); // t+1 = 3
        assert_eq!(plan.coeffs[0][0].len(), 4); // d=4
    }

    #[test]
    fn test_plan_n32_d8() {
        let plan = precompute_folding_plan(32, 8);
        assert_eq!(plan.num_rounds, 3);
        assert_eq!(plan.num_outputs, 4);
        assert_eq!(plan.coeffs.len(), 4); // t+1 = 4
        assert_eq!(plan.coeffs[0][0].len(), 8); // d=8
    }

    // ── Direct vs iterative agreement ────────────────────────────────

    #[test]
    fn test_direct_vs_iterative_n8_d2() {
        let cw: Vec<EF> = (0..8).map(|i| EF::from(F::from_usize(i * 17 + 3))).collect();
        let beta = test_beta();
        let plan = precompute_folding_plan(8, 2);
        let direct = apply_folding_plan(&plan, &cw, beta);
        let iterative = fold_iterative(&cw, 1, beta);
        assert_eq!(direct.values, iterative);
    }

    #[test]
    fn test_direct_vs_iterative_n16_d4() {
        let cw: Vec<EF> = (0..16).map(|i| EF::from(F::from_usize(i * 13 + 7))).collect();
        let beta = test_beta();
        let plan = precompute_folding_plan(16, 4);
        let direct = apply_folding_plan(&plan, &cw, beta);
        let iterative = fold_iterative(&cw, 2, beta);
        assert_eq!(direct.values, iterative);
    }

    #[test]
    fn test_direct_vs_iterative_n32_d8() {
        let cw: Vec<EF> = (0..32).map(|i| EF::from(F::from_usize(i * 11 + 5))).collect();
        let beta = test_beta();
        let plan = precompute_folding_plan(32, 8);
        let direct = apply_folding_plan(&plan, &cw, beta);
        let iterative = fold_iterative(&cw, 3, beta);
        assert_eq!(direct.values, iterative);
    }

    #[test]
    fn test_direct_vs_iterative_n64_d16() {
        let cw: Vec<EF> = (0..64).map(|i| EF::from(F::from_usize(i * 7 + 1))).collect();
        let beta = test_beta();
        let plan = precompute_folding_plan(64, 16);
        let direct = apply_folding_plan(&plan, &cw, beta);
        let iterative = fold_iterative(&cw, 4, beta);
        assert_eq!(direct.values, iterative);
    }

    // ── RS membership (valid codewords) ──────────────────────────────

    #[test]
    fn test_constant_polynomial_passes() {
        let cw = rs_encode(&[F::from_u8(42)], 8);
        assert!(check_rs_precomputed(&embed(&cw.evals), 2, test_beta()));
    }

    #[test]
    fn test_linear_polynomial_passes_d2() {
        let cw = rs_encode(&[F::from_u8(1), F::from_u8(2)], 8);
        assert!(check_rs_precomputed(&embed(&cw.evals), 2, test_beta()));
    }

    #[test]
    fn test_degree3_passes_d4() {
        let msg: Vec<F> = (1..=4).map(|i| F::from_u8(i)).collect();
        let cw = rs_encode(&msg, 16);
        assert!(check_rs_precomputed(&embed(&cw.evals), 4, test_beta()));
    }

    #[test]
    fn test_degree7_passes_d8() {
        let msg: Vec<F> = (1..=8).map(|i| F::from_usize(i)).collect();
        let cw = rs_encode(&msg, 32);
        assert!(check_rs_precomputed(&embed(&cw.evals), 8, test_beta()));
    }

    #[test]
    fn test_max_degree_passes() {
        let beta = test_beta();
        for (n, d) in [(8, 2), (16, 4), (32, 8), (64, 16)] {
            let msg: Vec<F> = (1..=d).map(|i| F::from_usize(i)).collect();
            let cw = rs_encode(&msg, n);
            assert!(
                check_rs_precomputed(&embed(&cw.evals), d, beta),
                "degree-{} polynomial should pass d={d} check on N={n}",
                d - 1
            );
        }
    }

    // ── RS membership (invalid codewords) ────────────────────────────

    #[test]
    fn test_random_vector_rejected() {
        let cw: Vec<EF> = (0..32).map(|i| EF::from(F::from_usize(i * 17 + 3))).collect();
        assert!(!check_rs_precomputed(&cw, 8, test_beta()));
    }

    #[test]
    fn test_tampered_codeword_rejected() {
        let msg: Vec<F> = (1..=4).map(|i| F::from_usize(i)).collect();
        let mut cw = rs_encode(&msg, 16);
        cw.evals[3] = cw.evals[3] + F::ONE;
        assert!(!check_rs_precomputed(&embed(&cw.evals), 4, test_beta()));
    }

    // ── Soundness: β≠0 fixes the gap ────────────────────────────────

    #[test]
    fn test_soundness_x5_rejected_with_random_beta() {
        // f(x) = x^5 on N=8, d=4. With β=0 this was a false positive.
        // With random β, the odd part is properly tested and this is rejected.
        let domain = Domain::new(8);
        let poly = Polynomial::new(vec![
            F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE,
        ]);
        let evals: Vec<EF> = domain
            .elements
            .iter()
            .map(|&x| EF::from(poly.evaluate(x)))
            .collect();

        assert!(
            !check_rs_precomputed(&evals, 4, test_beta()),
            "x^5 should be rejected with random beta"
        );
    }

    #[test]
    fn test_soundness_x1_rejected_d1_n8() {
        // f(x) = x has degree 1 ≥ 1, fold with d=1 means 0 rounds.
        // d=1 is a degenerate case: no folding, just check if the codeword
        // itself is constant. x evaluations are not constant.
        let cw = rs_encode(&[F::ZERO, F::ONE], 8);
        let evals = embed(&cw.evals);
        assert!(!is_constant(&FoldedLayer { values: evals }));
    }

    // ── β=0 regression: should degenerate to uniform averaging ───────

    #[test]
    fn test_beta_zero_gives_uniform_average() {
        let cw: Vec<EF> = (0..16).map(|i| EF::from(F::from_usize(i + 1))).collect();
        let plan = precompute_folding_plan(16, 4);
        let layer = apply_folding_plan(&plan, &cw, EF::ZERO);

        // With β=0, only coeffs[0] contributes: uniform 1/d average.
        let d = 4usize;
        let stride = 4;
        let weight = EF::from(F::from_usize(d)).inverse();
        for i in 0..plan.num_outputs {
            let mut expected = EF::ZERO;
            for s in 0..d {
                expected = expected + cw[i + s * stride];
            }
            expected = weight * expected;
            assert_eq!(layer.values[i], expected, "β=0 output[{i}] mismatch");
        }
    }

    // ── Algebraic identity: degree < d folds to constant a_0 ────────

    #[test]
    fn test_low_degree_folds_to_constant() {
        // For degree < d, the folded layer is constant regardless of β.
        // With β≠0, the constant value depends on β (not just a_0).
        let beta = test_beta();
        let msg = vec![F::from_u8(7), F::from_u8(11), F::from_u8(3), F::from_u8(5)];
        let cw = rs_encode(&msg, 16);
        let plan = precompute_folding_plan(16, 4);
        let layer = apply_folding_plan_base(&plan, &cw.evals, beta);

        assert!(is_constant(&layer), "degree < d should fold to constant");

        // With β=0, the constant IS a_0.
        let layer_beta0 = apply_folding_plan_base(&plan, &cw.evals, EF::ZERO);
        let a0 = EF::from(F::from_u8(7));
        for (i, &v) in layer_beta0.values.iter().enumerate() {
            assert_eq!(v, a0, "β=0 output[{i}] should be a_0");
        }
    }

    // ── Coefficient sanity: coeffs[0] rows sum to 1/d ────────────────

    #[test]
    fn test_coeffs_k0_rows_sum_to_one() {
        // The k=0 coefficients are the β=0 weights. Since β=0 folding is
        // averaging (each round divides by 2), the total weight sums to 1.
        for (n, d) in [(8, 2), (16, 4), (32, 8)] {
            let plan = precompute_folding_plan(n, d);
            for i in 0..plan.num_outputs {
                let row_sum: F = plan.coeffs[0][i].iter().copied().sum();
                assert_eq!(
                    row_sum, F::ONE,
                    "coeffs[0][{i}] should sum to 1 for N={n}, d={d}"
                );
            }
        }
    }

    // ── Property: direct and iterative always agree ──────────────────

    #[test]
    fn test_agreement_on_rs_codewords() {
        let beta = test_beta();
        for (n, d) in [(8, 2), (8, 4), (16, 4), (16, 8), (32, 8), (32, 16), (64, 16)] {
            let msg: Vec<F> = (0..d).map(|i| F::from_usize(i * 7 + 3)).collect();
            let cw = rs_encode(&msg, n);
            let evals = embed(&cw.evals);

            let plan = precompute_folding_plan(n, d);
            let direct = apply_folding_plan(&plan, &evals, beta);
            let iterative = fold_iterative(&evals, plan.num_rounds, beta);

            assert_eq!(
                direct.values, iterative,
                "direct vs iterative mismatch for N={n}, d={d}"
            );
        }
    }

    #[test]
    fn test_agreement_on_arbitrary_vectors() {
        let beta = test_beta();
        for n in [8, 16, 32] {
            for d in [2, 4, 8] {
                if d > n {
                    continue;
                }
                let cw: Vec<EF> = (0..n)
                    .map(|i| EF::from(F::from_usize(i * 31 + 17)))
                    .collect();
                let plan = precompute_folding_plan(n, d);
                let direct = apply_folding_plan(&plan, &cw, beta);
                let iterative = fold_iterative(&cw, plan.num_rounds, beta);
                assert_eq!(
                    direct.values, iterative,
                    "mismatch for N={n}, d={d} on arbitrary vector"
                );
            }
        }
    }

    #[test]
    fn test_agreement_with_multiple_betas() {
        // Verify agreement for several different beta values.
        let cw: Vec<EF> = (0..16).map(|i| EF::from(F::from_usize(i * 13 + 7))).collect();
        let plan = precompute_folding_plan(16, 4);

        let betas = [
            EF::ZERO,
            EF::ONE,
            EF::TWO,
            test_beta(),
            EF::from(F::from_usize(12345)),
        ];

        for beta in betas {
            let direct = apply_folding_plan(&plan, &cw, beta);
            let iterative = fold_iterative(&cw, 2, beta);
            assert_eq!(direct.values, iterative);
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_degree_1_is_identity() {
        // d=1 means 0 fold rounds: the output IS the input.
        let cw: Vec<EF> = (0..8).map(|i| EF::from(F::from_usize(i))).collect();
        let plan = precompute_folding_plan(8, 1);
        assert_eq!(plan.num_rounds, 0);
        assert_eq!(plan.num_outputs, 8);
        assert_eq!(plan.coeffs.len(), 1); // only coeffs[0]
        let layer = apply_folding_plan(&plan, &cw, test_beta());
        assert_eq!(layer.values, cw);
    }

    #[test]
    fn test_degree_equals_n_folds_to_one() {
        // d=N means log2(N) folds, single output.
        let n = 8;
        let cw: Vec<EF> = (0..n).map(|i| EF::from(F::from_usize(i + 1))).collect();
        let beta = test_beta();
        let plan = precompute_folding_plan(n, n);
        assert_eq!(plan.num_rounds, 3);
        assert_eq!(plan.num_outputs, 1);
        let layer = apply_folding_plan(&plan, &cw, beta);

        // Compare with iterative.
        let iterative = fold_iterative(&cw, 3, beta);
        assert_eq!(layer.values, iterative);
        assert!(is_constant(&layer)); // trivially: 1 element
    }
}
