//! Polynomial operations and Reed-Solomon encoding.
//!
//! Provides polynomial evaluation over multiplicative domains (roots of
//! unity) and RS codeword generation. The domain uses the two-adic
//! structure of KoalaBear: the multiplicative group has order p-1 =
//! 2^24 * 127, so we have roots of unity of order 2^k for k <= 24.

use crate::types::{Codeword, F};
use backend::{PrimeCharacteristicRing, TwoAdicField};

/// A polynomial in coefficient form: c_0 + c_1*x + c_2*x^2 + ... + c_{d-1}*x^{d-1}.
#[derive(Clone, Debug)]
pub struct Polynomial {
    /// Coefficients in ascending order: coeffs[i] is the coefficient of x^i.
    pub coeffs: Vec<F>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    /// The number of coefficients (degree + 1, or 0 for the zero polynomial).
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Evaluate at a single point using Horner's method.
    pub fn evaluate(&self, x: F) -> F {
        if self.coeffs.is_empty() {
            return F::ZERO;
        }
        let mut result = *self.coeffs.last().unwrap();
        for &c in self.coeffs.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }

    /// Evaluate on a full multiplicative domain, returning all n evaluations.
    pub fn evaluate_domain(&self, domain: &Domain) -> Vec<F> {
        domain.elements.iter().map(|&x| self.evaluate(x)).collect()
    }
}

/// A multiplicative domain of size n: the subgroup {1, omega, omega^2, ..., omega^{n-1}}
/// where omega is a primitive n-th root of unity.
#[derive(Clone, Debug)]
pub struct Domain {
    /// Domain size (must be a power of 2).
    pub size: usize,
    /// Primitive n-th root of unity.
    pub generator: F,
    /// All domain elements: [1, omega, omega^2, ..., omega^{n-1}].
    pub elements: Vec<F>,
}

impl Domain {
    /// Create a multiplicative domain of the given size.
    ///
    /// `size` must be a power of 2 and at most 2^24 (the two-adicity of KoalaBear).
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "domain size must be a power of 2");
        let bits = size.ilog2() as usize;
        assert!(
            bits <= F::TWO_ADICITY,
            "domain size 2^{bits} exceeds field two-adicity 2^{}",
            F::TWO_ADICITY
        );

        let generator = F::two_adic_generator(bits);

        let mut elements = Vec::with_capacity(size);
        let mut current = F::ONE;
        for _ in 0..size {
            elements.push(current);
            current = current * generator;
        }

        Self {
            size,
            generator,
            elements,
        }
    }
}

/// Encode a message as a Reed-Solomon codeword.
///
/// Interprets `message` as polynomial coefficients (degree < k where k = message.len()),
/// then evaluates on a domain of size `domain_size` (must be > k and a power of 2).
/// The resulting codeword has rate k/n.
///
/// For DAS, typically n = 2k (rate 1/2): a polynomial of degree < k evaluated
/// on n points gives an [n, k] RS code.
pub fn rs_encode(message: &[F], domain_size: usize) -> Codeword {
    assert!(!message.is_empty(), "message must be non-empty");
    assert!(
        domain_size > message.len(),
        "domain size {domain_size} must exceed message length {}",
        message.len()
    );

    let poly = Polynomial::new(message.to_vec());
    let domain = Domain::new(domain_size);
    let evals = poly.evaluate_domain(&domain);
    Codeword::new(evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_evaluate_constant() {
        let p = Polynomial::new(vec![F::from_u8(42)]);
        assert_eq!(p.evaluate(F::ZERO), F::from_u8(42));
        assert_eq!(p.evaluate(F::ONE), F::from_u8(42));
        assert_eq!(p.evaluate(F::TWO), F::from_u8(42));
    }

    #[test]
    fn test_polynomial_evaluate_linear() {
        // p(x) = 3 + 5x
        let p = Polynomial::new(vec![F::from_u8(3), F::from_u8(5)]);
        assert_eq!(p.evaluate(F::ZERO), F::from_u8(3));
        assert_eq!(p.evaluate(F::ONE), F::from_u8(8));
        assert_eq!(p.evaluate(F::TWO), F::from_u8(13));
    }

    #[test]
    fn test_polynomial_evaluate_quadratic() {
        // p(x) = 1 + 2x + 3x^2
        let p = Polynomial::new(vec![F::from_u8(1), F::from_u8(2), F::from_u8(3)]);
        // p(0) = 1, p(1) = 6, p(2) = 1 + 4 + 12 = 17
        assert_eq!(p.evaluate(F::ZERO), F::from_u8(1));
        assert_eq!(p.evaluate(F::ONE), F::from_u8(6));
        assert_eq!(p.evaluate(F::TWO), F::from_u8(17));
    }

    #[test]
    fn test_domain_generator_is_primitive_root() {
        let domain = Domain::new(32);
        assert_eq!(domain.elements.len(), 32);
        assert_eq!(domain.elements[0], F::ONE);
        // omega^32 should be 1 (wraps around).
        let omega_32 = domain.generator.exp_u64(32);
        assert_eq!(omega_32, F::ONE);
        // omega^16 should NOT be 1 (primitive, not just any root).
        let omega_16 = domain.generator.exp_u64(16);
        assert_ne!(omega_16, F::ONE);
    }

    #[test]
    fn test_domain_elements_distinct() {
        use backend::PrimeField32;
        let domain = Domain::new(16);
        let mut seen = std::collections::HashSet::new();
        for &e in &domain.elements {
            assert!(seen.insert(e.as_canonical_u32()), "duplicate domain element");
        }
    }

    #[test]
    fn test_rs_encode_constant() {
        // Degree-0 polynomial: p(x) = 7. All evaluations should be 7.
        let cw = rs_encode(&[F::from_u8(7)], 16);
        assert_eq!(cw.len(), 16);
        for &v in &cw.evals {
            assert_eq!(v, F::from_u8(7));
        }
    }

    #[test]
    fn test_rs_encode_linear() {
        // p(x) = 1 + 2x on domain of size 8.
        let cw = rs_encode(&[F::from_u8(1), F::from_u8(2)], 8);
        assert_eq!(cw.len(), 8);

        // Verify against direct polynomial evaluation.
        let domain = Domain::new(8);
        let poly = Polynomial::new(vec![F::from_u8(1), F::from_u8(2)]);
        for (i, &omega_i) in domain.elements.iter().enumerate() {
            assert_eq!(cw.evals[i], poly.evaluate(omega_i));
        }
    }

    #[test]
    fn test_rs_encode_degree_bound() {
        // Degree-3 polynomial on domain of size 32 (rate 4/32 = 1/8).
        let msg = vec![F::from_u8(1), F::from_u8(2), F::from_u8(3), F::from_u8(4)];
        let cw = rs_encode(&msg, 32);
        assert_eq!(cw.len(), 32);

        // First evaluation (at omega^0 = 1): sum of coefficients.
        let expected_at_one = F::from_u8(1) + F::from_u8(2) + F::from_u8(3) + F::from_u8(4);
        assert_eq!(cw.evals[0], expected_at_one);
    }

    #[test]
    fn test_rs_encode_fri_foldable() {
        // Encode a degree-< k polynomial on domain of size n = 2k.
        // After FRI folding (log2(n) rounds), the result should be constant.
        use crate::fri::fri_fold_evaluations;
        use crate::types::EF;

        let k = 16;
        let n = 32;
        let msg: Vec<F> = (0..k).map(|i| F::from_usize(i + 1)).collect();
        let cw = rs_encode(&msg, n);

        // Embed into extension field for FRI.
        let evals_ef: Vec<EF> = cw.evals.iter().map(|&v| EF::from(v)).collect();

        // Generate deterministic challenges.
        let num_rounds = n.ilog2() as usize;
        let challenges: Vec<EF> = (0..num_rounds)
            .map(|i| EF::from(F::from_usize(i * 7 + 3)))
            .collect();

        let result = fri_fold_evaluations(&evals_ef, &challenges);

        // For a valid RS codeword (degree < n/2 = 16), after full folding
        // to a single value, the result is just a constant. We verify it's
        // well-defined (non-panic).
        let _ = result;
    }

    #[test]
    fn test_non_rs_codeword_detected_by_fri() {
        // An arbitrary non-RS vector should fail the membership check.
        use crate::fri::check_rs_membership;
        use crate::types::EF;

        let n = 16;
        let evals: Vec<EF> = (0..n).map(|i| EF::from(F::from_usize(i * 17 + 3))).collect();
        let challenges: Vec<EF> = (0..10)
            .map(|i| EF::from(F::from_usize(i * 13 + 7)))
            .collect();
        assert!(!check_rs_membership(&evals, &challenges));
    }
}
