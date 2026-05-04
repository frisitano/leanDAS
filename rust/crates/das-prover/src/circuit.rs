//! ZKVM circuit compilation and execution for leanDAS proving.
//!
//! Compiles the Merkle + RLC + FRI circuit for the leanVM, prepares
//! public inputs and private witness, and invokes the prover.
//!
//! Three compilation modes:
//! - **Fast path** (`compile_das_circuit`): Standalone batch circuit (~400 instructions).
//!   Used when recursion is not needed.
//! - **Unified** (`compile_unified_das_circuit`): Self-referential circuit (~143K instructions)
//!   that handles both raw batch proving and recursive proof verification.
//!   Used when recursive aggregation is needed.
//! - **Hybrid** (`compile_hybrid_das_circuit`): Self-referential recursion circuit that also
//!   verifies proofs from the fast-path inner batch circuit. Combines fast leaf proving
//!   (~50ms) with multi-level recursive aggregation (~1.1s per level).

use backend::*;
use das_core::types::{Batch, Commitment, F, DIGEST_SIZE};
use lean_compiler::{compile_program_with_flags, CompilationFlags, ProgramSource};
use lean_prover::prove_execution::prove_execution;
use lean_prover::verify_execution::verify_execution;
use lean_prover::{
    default_whir_config, MAX_NUM_VARIABLES_TO_SEND_COEFFS,
    RS_DOMAIN_INITIAL_REDUCTION_FACTOR, WHIR_INITIAL_FOLDING_FACTOR,
    WHIR_SUBSEQUENT_FOLDING_FACTOR,
};
use lean_vm::*;
use sub_protocols::{min_stacked_n_vars, total_whir_statements};
use utils::Counter;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The rec_aggregation crate directory (set by build.rs).
const REC_AGG_DIR: &str = env!("REC_AGGREGATION_DIR");

/// Maximum number of inner proofs that can be aggregated in a single recursion.
pub const MAX_INNER_PROOFS: usize = 16;

/// FRI inner loop threshold: unroll when half_len <= this, use range() above.

// ─── Fast-path compilation (standalone batch circuit) ───────────────────

/// Compile the standalone RLC+FRI circuit for a given batch size, codeword length,
/// evals-per-leaf, and polynomial degree.
///
/// This is the "fast path" (~400 instructions). Use when recursion is not needed.
pub fn compile_das_circuit(
    batch_size: usize,
    codeword_len: usize,
    evals_per_leaf: usize,
    degree: usize,
) -> Bytecode {
    assert!(codeword_len.is_power_of_two());
    assert!(evals_per_leaf >= 16 && evals_per_leaf % 8 == 0);
    assert!(codeword_len % evals_per_leaf == 0);
    assert!(degree > 0 && degree.is_power_of_two(), "degree must be a power of 2");
    assert!(degree <= codeword_len / 2, "degree must be <= codeword_len / 2");

    let replacements = build_das_replacements(batch_size, codeword_len, evals_per_leaf, degree);

    let filepath = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("circuit.py")
        .to_str()
        .unwrap()
        .to_string();
    compile_program_with_flags(&ProgramSource::Filepath(filepath), CompilationFlags { replacements })
}

/// Build the replacement map for circuit.py template parameters.
fn build_das_replacements(batch_size: usize, codeword_len: usize, evals_per_leaf: usize, degree: usize) -> BTreeMap<String, String> {
    use backend::PrimeField32;
    let num_leaves = codeword_len / evals_per_leaf;
    let tree_height = if num_leaves > 1 { num_leaves.ilog2() as usize } else { 0 };
    let num_fold_rounds = degree.ilog2() as usize;
    let n_chunks = evals_per_leaf / DIGEST_SIZE;

    // Compute 1/2 in the base field.
    let half = F::TWO.inverse();

    // Compute omega_inv_round for each FRI round:
    // omega_inv_round[r] = omega^{-2^r} where omega is the N-th root of unity.
    let log_n = codeword_len.ilog2() as usize;
    let omega: F = F::two_adic_generator(log_n);
    let omega_inv = omega.inverse();
    let omega_inv_rounds: Vec<u32> = (0..num_fold_rounds)
        .map(|r| omega_inv.exp_u64(1u64 << r).as_canonical_u32())
        .collect();
    let omega_inv_rounds_str = format!(
        "[{}]",
        omega_inv_rounds.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
    );

    let mut r = BTreeMap::new();
    r.insert("__NRPIS__".into(), NONRESERVED_PROGRAM_INPUT_START.to_string());
    r.insert("__BATCH_SIZE__".into(), batch_size.to_string());
    r.insert("__CODEWORD_LEN__".into(), codeword_len.to_string());
    r.insert("__DIM__".into(), DIMENSION.to_string());
    r.insert("__DIGEST_LEN__".into(), DIGEST_SIZE.to_string());
    r.insert("__EPL__".into(), evals_per_leaf.to_string());
    r.insert("__NUM_LEAVES__".into(), num_leaves.to_string());
    r.insert("__TREE_HEIGHT__".into(), tree_height.to_string());
    r.insert("__NUM_FOLD_ROUNDS__".into(), num_fold_rounds.to_string());
    r.insert("__N_CHUNKS__".into(), n_chunks.to_string());
    r.insert("__HALF__".into(), half.as_canonical_u32().to_string());
    r.insert("__OMEGA_INV_ROUNDS__".into(), omega_inv_rounds_str);
    r
}

// ─── Spot-check compilation (RLC hint + spot-check circuit) ──────────

/// Compute the number of spot-checks needed for a target security level.
///
/// Each spot-check catches an incorrect RLC with probability ≥ 1 - rate,
/// where rate = degree / codeword_len. For k checks the failure probability
/// is ≤ rate^k, so we need k ≥ security_bits / log₂(1/rate).
pub fn compute_num_spot_checks(codeword_len: usize, degree: usize, security_bits: usize) -> usize {
    assert!(degree < codeword_len);
    let log_inv_rate = (codeword_len / degree).ilog2() as usize;
    assert!(log_inv_rate > 0, "rate must be < 1 (degree < codeword_len)");
    (security_bits + log_inv_rate - 1) / log_inv_rate
}

/// Compile the spot-check DAS circuit.
///
/// This circuit replaces the full RLC computation with a prover-provided RLC
/// hint verified at `num_spot_checks` random positions. For rate 1/2 with
/// 128-bit security, k=128 spot-checks reduce the extension_op precompile
/// trace from ~983K to ~31K rows (32x reduction at m=240, N=4096).
pub fn compile_spotcheck_circuit(
    batch_size: usize,
    codeword_len: usize,
    evals_per_leaf: usize,
    degree: usize,
    num_spot_checks: usize,
) -> Bytecode {
    assert!(codeword_len.is_power_of_two());
    assert!(evals_per_leaf >= 16 && evals_per_leaf % 8 == 0);
    assert!(codeword_len % evals_per_leaf == 0);
    assert!(degree > 0 && degree.is_power_of_two(), "degree must be a power of 2");
    assert!(degree <= codeword_len / 2, "degree must be <= codeword_len / 2");
    assert!(num_spot_checks > 0, "need at least 1 spot-check");

    let mut replacements = build_das_replacements(batch_size, codeword_len, evals_per_leaf, degree);

    // Spot-check specific constants.
    let log_codeword_len = codeword_len.ilog2() as usize;
    let positions_per_element = 31 / log_codeword_len;
    let positions_per_hash = positions_per_element * DIGEST_SIZE;
    let num_hash_rounds = (num_spot_checks + positions_per_hash - 1) / positions_per_hash;
    let n_rlc_chunks = (codeword_len * DIMENSION + DIGEST_SIZE - 1) / DIGEST_SIZE;

    let num_rand_elements = (num_spot_checks + positions_per_element - 1) / positions_per_element;
    let num_full_rand_elements = num_spot_checks / positions_per_element;
    let remainder_positions = num_spot_checks % positions_per_element;

    replacements.insert("__NUM_SPOT_CHECKS__".into(), num_spot_checks.to_string());
    replacements.insert("__LOG_CODEWORD_LEN__".into(), log_codeword_len.to_string());
    replacements.insert("__POSITIONS_PER_ELEMENT__".into(), positions_per_element.to_string());
    replacements.insert("__NUM_RAND_ELEMENTS__".into(), num_rand_elements.to_string());
    replacements.insert("__NUM_FULL_RAND_ELEMENTS__".into(), num_full_rand_elements.to_string());
    replacements.insert("__REMAINDER_POSITIONS__".into(), remainder_positions.to_string());
    replacements.insert("__NUM_HASH_ROUNDS__".into(), num_hash_rounds.to_string());
    replacements.insert("__N_RLC_CHUNKS__".into(), n_rlc_chunks.to_string());

    let filepath = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("circuit_spotcheck.py")
        .to_str()
        .unwrap()
        .to_string();
    compile_program_with_flags(&ProgramSource::Filepath(filepath), CompilationFlags { replacements })
}

/// Prepare the private input buffer for the spot-check DAS circuit.
///
/// Layout: four regions:
///
/// Region 0 — commitments: `[com_0[0..8], ..., com_{B-1}[0..8]]`
/// Region 1 — row-major BF (for Merkle): `[cw_0[0..N], ..., cw_{B-1}[0..N]]`
/// Region 2 — column-major BF (for spot-check dot_product): `[col_0[0..B], ..., col_{N-1}[0..B]]`
/// Region 3 — RLC hint (N extension field elements): `[rlc_0[0..DIM], ..., rlc_{N-1}[0..DIM]]`
pub fn prepare_spotcheck_private_input(batch: &Batch, _degree: usize) -> Vec<F> {
    let b = batch.codewords.len();
    let n = batch.codewords[0].len();

    let mut buf = Vec::with_capacity(b * DIGEST_SIZE + b * n + n * b + n * DIMENSION);

    // Region 0: commitments
    for com in &batch.commitments {
        buf.extend_from_slice(&com.root);
    }

    // Region 1: row-major BF (all B codewords, for Merkle hashing)
    for cw in &batch.codewords {
        buf.extend_from_slice(&cw.evals);
    }

    // Region 2: column-major BF (all B codewords, for spot-check dot_product_be)
    for j in 0..n {
        for cw in &batch.codewords {
            buf.push(cw.evals[j]);
        }
    }

    // Region 3: RLC hint values (N * DIM base field elements)
    let rlc_values = das_core::rlc::compute_rlc(&batch.codewords, &batch.challenges);
    for ef_val in &rlc_values {
        buf.extend_from_slice(ef_val.as_basis_coefficients_slice());
    }

    buf
}

/// Run the spot-check DAS circuit inside the ZKVM and produce a STARK proof.
pub fn prove_spotcheck_circuit(
    bytecode: &Bytecode,
    batch: &Batch,
    degree: usize,
) -> (backend::Proof<F>, Vec<F>) {
    let public_input = prepare_public_input_with_hash(batch, bytecode);
    let private_input = prepare_spotcheck_private_input(batch, degree);

    let witness = ExecutionWitness {
        private_input: &private_input,
        xmss_signatures: &[],
        merkle_paths: &[],
    };

    let whir_config = default_whir_config(1);

    tracing::info!(
        batch_size = batch.codewords.len(),
        codeword_len = batch.codewords[0].len(),
        "proving spot-check DAS circuit in ZKVM"
    );

    let execution_proof = prove_execution(
        bytecode,
        &public_input,
        &witness,
        &whir_config,
        false,
    );

    (execution_proof.proof, public_input)
}

// ─── Unified compilation (self-referential batch + recursion circuit) ───

/// Compile the unified self-referential DAS circuit.
///
/// This circuit handles both raw batch proving (Merkle+RLC+FRI) and recursive
/// proof verification in a single circuit. It verifies proofs of **itself**,
/// requiring a self-referential compilation loop that converges on bytecode size.
///
/// The resulting circuit is ~143K instructions (log_size=18).
pub fn compile_unified_das_circuit(
    batch_size: usize,
    codeword_len: usize,
    evals_per_leaf: usize,
    degree: usize,
) -> Bytecode {
    assert!(codeword_len.is_power_of_two());
    assert!(evals_per_leaf >= 16 && evals_per_leaf % 8 == 0);
    assert!(codeword_len % evals_per_leaf == 0);
    assert!(degree > 0 && degree.is_power_of_two(), "degree must be a power of 2");
    assert!(degree <= codeword_len / 2, "degree must be <= codeword_len / 2");

    let mut log_size_guess = 18; // initial guess (typical recursion circuit size)
    let bytecode_zero_eval = F::ONE;

    loop {
        let bytecode = compile_unified_inner(
            batch_size,
            codeword_len,
            evals_per_leaf,
            degree,
            log_size_guess,
            bytecode_zero_eval,
        );
        assert_eq!(bytecode_zero_eval, bytecode.instructions_multilinear[0]);
        let actual_log_size = bytecode.log_size();
        if actual_log_size == log_size_guess {
            tracing::info!(
                log_size = actual_log_size,
                instructions = bytecode.instructions.len(),
                "unified DAS circuit compiled"
            );
            return bytecode;
        }
        tracing::info!(
            actual_log_size,
            log_size_guess,
            "self-referential loop: recompiling"
        );
        log_size_guess = actual_log_size;
    }
}

/// Inner compilation for the unified circuit with a given log_size guess.
fn compile_unified_inner(
    batch_size: usize,
    codeword_len: usize,
    evals_per_leaf: usize,
    degree: usize,
    inner_program_log_size: usize,
    bytecode_zero_eval: F,
) -> Bytecode {
    // Start with DAS-specific replacements.
    let mut r = build_das_replacements(batch_size, codeword_len, evals_per_leaf, degree);

    // Add recursion replacements.
    let log_inner_bytecode = inner_program_log_size;
    let bytecode_point_n_vars = log_inner_bytecode + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let claim_data_size = (bytecode_point_n_vars + 1) * DIMENSION;
    let claim_data_size_padded = claim_data_size.next_multiple_of(DIGEST_LEN);

    // Public input size: [aggregated_hash(DIGEST_LEN) + bytecode_claim(padded) + bytecode_hash(DIGEST_LEN)]
    let pub_input_size = DIGEST_LEN + claim_data_size_padded + DIGEST_LEN;
    let inner_public_memory_log_size =
        log2_ceil_usize(NONRESERVED_PROGRAM_INPUT_START + pub_input_size);

    let min_stacked = min_stacked_n_vars(log_inner_bytecode);

    // WHIR configuration for all possible (log_inv_rate, n_vars) combinations.
    let mut all_potential_num_queries = vec![];
    let mut all_potential_query_grinding = vec![];
    let mut all_potential_num_oods = vec![];
    let mut all_potential_folding_grinding = vec![];

    for log_inv_rate in MIN_WHIR_LOG_INV_RATE..=MAX_WHIR_LOG_INV_RATE {
        let max_n_vars = F::TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate;
        let whir_config_builder = default_whir_config(log_inv_rate);

        let mut queries_for_rate = vec![];
        let mut query_grinding_for_rate = vec![];
        let mut oods_for_rate = vec![];
        let mut folding_grinding_for_rate = vec![];

        for n_vars in min_stacked..=max_n_vars {
            let cfg = WhirConfig::<EF>::new(&whir_config_builder, n_vars);

            let mut num_queries = vec![];
            let mut query_grinding_bits = vec![];
            let mut oods = vec![cfg.commitment_ood_samples];
            let mut folding_grinding = vec![cfg.starting_folding_pow_bits];

            for round in &cfg.round_parameters {
                num_queries.push(round.num_queries);
                query_grinding_bits.push(round.query_pow_bits);
                oods.push(round.ood_samples);
                folding_grinding.push(round.folding_pow_bits);
            }
            num_queries.push(cfg.final_queries);
            query_grinding_bits.push(cfg.final_query_pow_bits);

            queries_for_rate.push(format!(
                "[{}]",
                num_queries.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(", ")
            ));
            query_grinding_for_rate.push(format!(
                "[{}]",
                query_grinding_bits.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(", ")
            ));
            oods_for_rate.push(format!(
                "[{}]",
                oods.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(", ")
            ));
            folding_grinding_for_rate.push(format!(
                "[{}]",
                folding_grinding.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", ")
            ));
        }
        all_potential_num_queries.push(format!("[{}]", queries_for_rate.join(", ")));
        all_potential_query_grinding.push(format!("[{}]", query_grinding_for_rate.join(", ")));
        all_potential_num_oods.push(format!("[{}]", oods_for_rate.join(", ")));
        all_potential_folding_grinding.push(format!("[{}]", folding_grinding_for_rate.join(", ")));
    }

    // WHIR replacements
    r.insert("WHIR_FIRST_RS_REDUCTION_FACTOR_PLACEHOLDER".into(), RS_DOMAIN_INITIAL_REDUCTION_FACTOR.to_string());
    r.insert("WHIR_ALL_POTENTIAL_NUM_QUERIES_PLACEHOLDER".into(), format!("[{}]", all_potential_num_queries.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_QUERY_GRINDING_PLACEHOLDER".into(), format!("[{}]", all_potential_query_grinding.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_NUM_OODS_PLACEHOLDER".into(), format!("[{}]", all_potential_num_oods.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_FOLDING_GRINDING_PLACEHOLDER".into(), format!("[{}]", all_potential_folding_grinding.join(", ")));
    r.insert("MIN_STACKED_N_VARS_PLACEHOLDER".into(), min_stacked.to_string());

    // VM recursion parameters
    r.insert("N_TABLES_PLACEHOLDER".into(), N_TABLES.to_string());
    r.insert("MIN_LOG_N_ROWS_PER_TABLE_PLACEHOLDER".into(), MIN_LOG_N_ROWS_PER_TABLE.to_string());

    let mut max_log_n_rows_per_table = MAX_LOG_N_ROWS_PER_TABLE.to_vec();
    max_log_n_rows_per_table.sort_by_key(|(table, _)| table.index());
    max_log_n_rows_per_table.dedup();
    r.insert(
        "MAX_LOG_N_ROWS_PER_TABLE_PLACEHOLDER".into(),
        format!(
            "[{}]",
            max_log_n_rows_per_table.iter().map(|(_, v)| v.to_string()).collect::<Vec<_>>().join(", ")
        ),
    );

    r.insert("MIN_WHIR_LOG_INV_RATE_PLACEHOLDER".into(), MIN_WHIR_LOG_INV_RATE.to_string());
    r.insert("MAX_WHIR_LOG_INV_RATE_PLACEHOLDER".into(), MAX_WHIR_LOG_INV_RATE.to_string());
    r.insert("MIN_LOG_MEMORY_SIZE_PLACEHOLDER".into(), MIN_LOG_MEMORY_SIZE.to_string());
    r.insert("MAX_LOG_MEMORY_SIZE_PLACEHOLDER".into(), MAX_LOG_MEMORY_SIZE.to_string());
    r.insert("MAX_BUS_WIDTH_PLACEHOLDER".into(), max_bus_width_including_domainsep().to_string());
    r.insert("LOGUP_MEMORY_DOMAINSEP_PLACEHOLDER".into(), LOGUP_MEMORY_DOMAINSEP.to_string());
    r.insert("LOGUP_PRECOMPILE_DOMAINSEP_PLACEHOLDER".into(), LOGUP_PRECOMPILE_DOMAINSEP.to_string());
    r.insert("LOGUP_BYTECODE_DOMAINSEP_PLACEHOLDER".into(), LOGUP_BYTECODE_DOMAINSEP.to_string());
    r.insert("LOG_GUEST_BYTECODE_LEN_PLACEHOLDER".into(), log_inner_bytecode.to_string());
    r.insert("COL_PC_PLACEHOLDER".into(), COL_PC.to_string());
    r.insert("NONRESERVED_PROGRAM_INPUT_START_PLACEHOLDER".into(), NONRESERVED_PROGRAM_INPUT_START.to_string());
    r.insert("INNER_PUBLIC_MEMORY_LOG_SIZE_PLACEHOLDER".into(), inner_public_memory_log_size.to_string());
    r.insert("PUB_INPUT_SIZE_PLACEHOLDER".into(), pub_input_size.to_string());
    r.insert("MAX_NUM_VARIABLES_TO_SEND_COEFFS_PLACEHOLDER".into(), MAX_NUM_VARIABLES_TO_SEND_COEFFS.to_string());
    r.insert("WHIR_INITIAL_FOLDING_FACTOR_PLACEHOLDER".into(), WHIR_INITIAL_FOLDING_FACTOR.to_string());
    r.insert("WHIR_SUBSEQUENT_FOLDING_FACTOR_PLACEHOLDER".into(), WHIR_SUBSEQUENT_FOLDING_FACTOR.to_string());
    r.insert("EXECUTION_TABLE_INDEX_PLACEHOLDER".into(), Table::execution().index().to_string());
    r.insert("MAX_NUM_AIR_CONSTRAINTS_PLACEHOLDER".into(), max_air_constraints().to_string());
    r.insert("N_INSTRUCTION_COLUMNS_PLACEHOLDER".into(), N_INSTRUCTION_COLUMNS.to_string());
    r.insert("N_COMMITTED_EXEC_COLUMNS_PLACEHOLDER".into(), N_RUNTIME_COLUMNS.to_string());
    r.insert("TOTAL_WHIR_STATEMENTS_PLACEHOLDER".into(), total_whir_statements().to_string());
    r.insert("STARTING_PC_PLACEHOLDER".into(), STARTING_PC.to_string());
    r.insert("ENDING_PC_PLACEHOLDER".into(), ENDING_PC.to_string());
    r.insert("BYTECODE_ZERO_EVAL_PLACEHOLDER".into(), bytecode_zero_eval.as_canonical_u64().to_string());

    // DAS-specific recursion replacements
    r.insert("__MAX_INNER_PROOFS__".into(), MAX_INNER_PROOFS.to_string());

    // Table-specific replacements (lookups, AIR columns, degrees, etc.)
    let mut lookup_indexes_str = vec![];
    let mut lookup_values_str = vec![];
    let mut num_cols_air = vec![];
    let mut air_degrees = vec![];
    let mut n_air_columns = vec![];
    let mut air_down_columns = vec![];

    for table in ALL_TABLES {
        let this_look_f_indexes_str = table
            .lookups()
            .iter()
            .map(|lookup_f| lookup_f.index.to_string())
            .collect::<Vec<_>>();
        lookup_indexes_str.push(format!("[{}]", this_look_f_indexes_str.join(", ")));
        num_cols_air.push(table.n_columns().to_string());
        let this_lookup_f_values_str = table
            .lookups()
            .iter()
            .map(|lookup_f| {
                format!(
                    "[{}]",
                    lookup_f.values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>();
        lookup_values_str.push(format!("[{}]", this_lookup_f_values_str.join(", ")));
        air_degrees.push(table.degree_air().to_string());
        n_air_columns.push(table.n_columns().to_string());
        air_down_columns.push(format!(
            "[{}]",
            table.down_column_indexes().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        ));
    }

    r.insert("LOOKUPS_INDEXES_PLACEHOLDER".into(), format!("[{}]", lookup_indexes_str.join(", ")));
    r.insert("LOOKUPS_VALUES_PLACEHOLDER".into(), format!("[{}]", lookup_values_str.join(", ")));
    r.insert("NUM_COLS_AIR_PLACEHOLDER".into(), format!("[{}]", num_cols_air.join(", ")));
    r.insert("AIR_DEGREES_PLACEHOLDER".into(), format!("[{}]", air_degrees.join(", ")));
    r.insert("N_AIR_COLUMNS_PLACEHOLDER".into(), format!("[{}]", n_air_columns.join(", ")));
    r.insert("AIR_DOWN_COLUMNS_PLACEHOLDER".into(), format!("[{}]", air_down_columns.join(", ")));

    // AIR constraint evaluation functions
    r.insert("EVALUATE_AIR_FUNCTIONS_PLACEHOLDER".into(), generate_all_air_evals_dsl());

    // Set up temp directory with symlinks and compile.
    let temp_dir = setup_unified_circuit_dir();
    let main_path = temp_dir.join("das_unified_main.py");
    let filepath = main_path.to_str().unwrap().to_string();

    compile_program_with_flags(
        &ProgramSource::Filepath(filepath),
        CompilationFlags { replacements: r },
    )
}

/// Set up a temporary directory with symlinks to rec_aggregation's Python files
/// and our unified DAS circuit.
fn setup_unified_circuit_dir() -> PathBuf {
    let dir_name = format!(
        "das_unified_circuit_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    );
    let temp_dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

    let rec_agg = Path::new(REC_AGG_DIR);

    // Symlink the Python files we need from rec_aggregation.
    for file in [
        "recursion.py",
        "whir.py",
        "fiat_shamir.py",
        "hashing.py",
        "utils.py",
    ] {
        let target = temp_dir.join(file);
        let source = rec_agg.join(file);
        let _ = std::fs::remove_file(&target);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&source, &target)
            .unwrap_or_else(|e| panic!("failed to symlink {}: {}", file, e));
        #[cfg(not(unix))]
        std::fs::copy(&source, &target)
            .unwrap_or_else(|e| panic!("failed to copy {}: {}", file, e));
    }

    // Copy our unified circuit.
    let our_main = Path::new(env!("CARGO_MANIFEST_DIR")).join("das_unified_main.py");
    let target_main = temp_dir.join("das_unified_main.py");
    let _ = std::fs::remove_file(&target_main);
    std::fs::copy(&our_main, &target_main).expect("failed to copy das_unified_main.py");

    temp_dir
}

/// Generate DSL code for evaluating AIR constraints for all tables.
fn generate_all_air_evals_dsl() -> String {
    let mut res = String::new();
    res += &air_eval_in_zk_dsl(ExecutionTable::<false> {});
    res += &air_eval_in_zk_dsl(ExtensionOpPrecompile::<false> {});
    res += &air_eval_in_zk_dsl(Poseidon16Precompile::<false> {});
    res
}

// ─── Hybrid compilation (inner batch + self-referential recursion) ───

/// Compile both circuits for the hybrid DAS approach:
/// - An inner batch circuit (fast path, ~400 instructions)
/// - A self-referential recursion circuit that can verify both inner batch proofs
///   and its own proofs
///
/// Returns `(inner_bytecode, hybrid_bytecode)`.
pub fn compile_hybrid_das_circuit(
    batch_size: usize,
    codeword_len: usize,
    evals_per_leaf: usize,
    degree: usize,
) -> (Bytecode, Bytecode) {
    // Step 1: Compile the inner batch circuit.
    let inner_bytecode = compile_das_circuit(batch_size, codeword_len, evals_per_leaf, degree);
    let inner_log_size = inner_bytecode.log_size();
    tracing::info!(inner_log_size, "inner batch circuit compiled");

    // Step 2: Self-referential loop for the hybrid recursion circuit.
    let mut log_size_guess = 18;
    let self_bytecode_zero_eval = F::ONE;

    let hybrid_bytecode = loop {
        let bytecode = compile_hybrid_inner(
            log_size_guess,
            self_bytecode_zero_eval,
            &inner_bytecode,
        );
        assert_eq!(self_bytecode_zero_eval, bytecode.instructions_multilinear[0]);
        let actual_log_size = bytecode.log_size();
        if actual_log_size == log_size_guess {
            tracing::info!(
                log_size = actual_log_size,
                instructions = bytecode.instructions.len(),
                "hybrid DAS circuit compiled"
            );
            break bytecode;
        }
        tracing::info!(
            actual_log_size,
            log_size_guess,
            "hybrid self-referential loop: recompiling"
        );
        log_size_guess = actual_log_size;
    };

    (inner_bytecode, hybrid_bytecode)
}

/// Inner compilation for the hybrid circuit with a given log_size guess.
fn compile_hybrid_inner(
    self_program_log_size: usize,
    self_bytecode_zero_eval: F,
    inner_bytecode: &Bytecode,
) -> Bytecode {
    // No DAS-specific batch/FRI constants needed — hybrid only verifies STARK proofs.
    let mut r = BTreeMap::new();

    // Self-referential recursion replacements (same as compile_unified_inner).
    let log_self_bytecode = self_program_log_size;
    let self_bytecode_point_n_vars = log_self_bytecode + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let self_claim_data_size = (self_bytecode_point_n_vars + 1) * DIMENSION;
    let self_claim_data_size_padded = self_claim_data_size.next_multiple_of(DIGEST_LEN);

    // Inner batch circuit parameters.
    let inner_log_size = inner_bytecode.log_size();
    let inner_bytecode_point_n_vars = inner_log_size + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let inner_claim_data_size = (inner_bytecode_point_n_vars + 1) * DIMENSION;
    let inner_claim_data_size_padded = inner_claim_data_size.next_multiple_of(DIGEST_LEN);

    // Inner batch circuit's PUB_INPUT_SIZE: [batch_hash(DIGEST_LEN) + bytecode_hash(DIGEST_LEN)]
    let inner_pub_input_size = 2 * DIGEST_LEN;
    let inner_pm_log = log2_ceil_usize(NONRESERVED_PROGRAM_INPUT_START + inner_pub_input_size);

    // Hybrid public input: [agg_hash + inner_claim + self_claim + inner_bc_hash + self_bc_hash]
    let hybrid_pub_input_size = DIGEST_LEN
        + inner_claim_data_size_padded
        + self_claim_data_size_padded
        + DIGEST_LEN
        + DIGEST_LEN;
    let self_pm_log = log2_ceil_usize(NONRESERVED_PROGRAM_INPUT_START + hybrid_pub_input_size);

    // The hybrid circuit verifies proofs from both itself (log_self_bytecode) and the
    // inner batch circuit (inner_log_size). The WHIR tables must cover both ranges.
    let min_stacked = min_stacked_n_vars(log_self_bytecode)
        .min(min_stacked_n_vars(inner_log_size));

    // WHIR configuration for all possible (log_inv_rate, n_vars) combinations.
    let mut all_potential_num_queries = vec![];
    let mut all_potential_query_grinding = vec![];
    let mut all_potential_num_oods = vec![];
    let mut all_potential_folding_grinding = vec![];

    for log_inv_rate in MIN_WHIR_LOG_INV_RATE..=MAX_WHIR_LOG_INV_RATE {
        let max_n_vars = F::TWO_ADICITY + WHIR_INITIAL_FOLDING_FACTOR - log_inv_rate;
        let whir_config_builder = default_whir_config(log_inv_rate);

        let mut queries_for_rate = vec![];
        let mut query_grinding_for_rate = vec![];
        let mut oods_for_rate = vec![];
        let mut folding_grinding_for_rate = vec![];

        for n_vars in min_stacked..=max_n_vars {
            let cfg = WhirConfig::<EF>::new(&whir_config_builder, n_vars);

            let mut num_queries = vec![];
            let mut query_grinding_bits = vec![];
            let mut oods = vec![cfg.commitment_ood_samples];
            let mut folding_grinding = vec![cfg.starting_folding_pow_bits];

            for round in &cfg.round_parameters {
                num_queries.push(round.num_queries);
                query_grinding_bits.push(round.query_pow_bits);
                oods.push(round.ood_samples);
                folding_grinding.push(round.folding_pow_bits);
            }
            num_queries.push(cfg.final_queries);
            query_grinding_bits.push(cfg.final_query_pow_bits);

            queries_for_rate.push(format!(
                "[{}]",
                num_queries.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(", ")
            ));
            query_grinding_for_rate.push(format!(
                "[{}]",
                query_grinding_bits.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(", ")
            ));
            oods_for_rate.push(format!(
                "[{}]",
                oods.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(", ")
            ));
            folding_grinding_for_rate.push(format!(
                "[{}]",
                folding_grinding.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", ")
            ));
        }
        all_potential_num_queries.push(format!("[{}]", queries_for_rate.join(", ")));
        all_potential_query_grinding.push(format!("[{}]", query_grinding_for_rate.join(", ")));
        all_potential_num_oods.push(format!("[{}]", oods_for_rate.join(", ")));
        all_potential_folding_grinding.push(format!("[{}]", folding_grinding_for_rate.join(", ")));
    }

    // WHIR replacements
    r.insert("WHIR_FIRST_RS_REDUCTION_FACTOR_PLACEHOLDER".into(), RS_DOMAIN_INITIAL_REDUCTION_FACTOR.to_string());
    r.insert("WHIR_ALL_POTENTIAL_NUM_QUERIES_PLACEHOLDER".into(), format!("[{}]", all_potential_num_queries.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_QUERY_GRINDING_PLACEHOLDER".into(), format!("[{}]", all_potential_query_grinding.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_NUM_OODS_PLACEHOLDER".into(), format!("[{}]", all_potential_num_oods.join(", ")));
    r.insert("WHIR_ALL_POTENTIAL_FOLDING_GRINDING_PLACEHOLDER".into(), format!("[{}]", all_potential_folding_grinding.join(", ")));
    r.insert(
        "MIN_STACKED_N_VARS_PLACEHOLDER".into(),
        min_stacked.to_string(),
    );

    // VM recursion parameters
    r.insert("N_TABLES_PLACEHOLDER".into(), N_TABLES.to_string());
    r.insert("MIN_LOG_N_ROWS_PER_TABLE_PLACEHOLDER".into(), MIN_LOG_N_ROWS_PER_TABLE.to_string());

    let mut max_log_n_rows_per_table = MAX_LOG_N_ROWS_PER_TABLE.to_vec();
    max_log_n_rows_per_table.sort_by_key(|(table, _)| table.index());
    max_log_n_rows_per_table.dedup();
    r.insert(
        "MAX_LOG_N_ROWS_PER_TABLE_PLACEHOLDER".into(),
        format!(
            "[{}]",
            max_log_n_rows_per_table.iter().map(|(_, v)| v.to_string()).collect::<Vec<_>>().join(", ")
        ),
    );

    r.insert("MIN_WHIR_LOG_INV_RATE_PLACEHOLDER".into(), MIN_WHIR_LOG_INV_RATE.to_string());
    r.insert("MAX_WHIR_LOG_INV_RATE_PLACEHOLDER".into(), MAX_WHIR_LOG_INV_RATE.to_string());
    r.insert("MIN_LOG_MEMORY_SIZE_PLACEHOLDER".into(), MIN_LOG_MEMORY_SIZE.to_string());
    r.insert("MAX_LOG_MEMORY_SIZE_PLACEHOLDER".into(), MAX_LOG_MEMORY_SIZE.to_string());
    r.insert("MAX_BUS_WIDTH_PLACEHOLDER".into(), max_bus_width_including_domainsep().to_string());
    r.insert("LOGUP_MEMORY_DOMAINSEP_PLACEHOLDER".into(), LOGUP_MEMORY_DOMAINSEP.to_string());
    r.insert("LOGUP_PRECOMPILE_DOMAINSEP_PLACEHOLDER".into(), LOGUP_PRECOMPILE_DOMAINSEP.to_string());
    r.insert("LOGUP_BYTECODE_DOMAINSEP_PLACEHOLDER".into(), LOGUP_BYTECODE_DOMAINSEP.to_string());
    // Self-referential: LOG_GUEST_BYTECODE_LEN is the hybrid circuit's own log_size
    r.insert("LOG_GUEST_BYTECODE_LEN_PLACEHOLDER".into(), log_self_bytecode.to_string());
    r.insert("COL_PC_PLACEHOLDER".into(), COL_PC.to_string());
    r.insert("NONRESERVED_PROGRAM_INPUT_START_PLACEHOLDER".into(), NONRESERVED_PROGRAM_INPUT_START.to_string());
    r.insert("INNER_PUBLIC_MEMORY_LOG_SIZE_PLACEHOLDER".into(), self_pm_log.to_string());
    r.insert("PUB_INPUT_SIZE_PLACEHOLDER".into(), hybrid_pub_input_size.to_string());
    r.insert("MAX_NUM_VARIABLES_TO_SEND_COEFFS_PLACEHOLDER".into(), MAX_NUM_VARIABLES_TO_SEND_COEFFS.to_string());
    r.insert("WHIR_INITIAL_FOLDING_FACTOR_PLACEHOLDER".into(), WHIR_INITIAL_FOLDING_FACTOR.to_string());
    r.insert("WHIR_SUBSEQUENT_FOLDING_FACTOR_PLACEHOLDER".into(), WHIR_SUBSEQUENT_FOLDING_FACTOR.to_string());
    r.insert("EXECUTION_TABLE_INDEX_PLACEHOLDER".into(), Table::execution().index().to_string());
    r.insert("MAX_NUM_AIR_CONSTRAINTS_PLACEHOLDER".into(), max_air_constraints().to_string());
    r.insert("N_INSTRUCTION_COLUMNS_PLACEHOLDER".into(), N_INSTRUCTION_COLUMNS.to_string());
    r.insert("N_COMMITTED_EXEC_COLUMNS_PLACEHOLDER".into(), N_RUNTIME_COLUMNS.to_string());
    r.insert("TOTAL_WHIR_STATEMENTS_PLACEHOLDER".into(), total_whir_statements().to_string());
    r.insert("STARTING_PC_PLACEHOLDER".into(), STARTING_PC.to_string());
    r.insert("ENDING_PC_PLACEHOLDER".into(), ENDING_PC.to_string());
    r.insert("BYTECODE_ZERO_EVAL_PLACEHOLDER".into(), self_bytecode_zero_eval.as_canonical_u64().to_string());

    // DAS-specific recursion replacements
    r.insert("__MAX_INNER_PROOFS__".into(), MAX_INNER_PROOFS.to_string());

    // Inner batch circuit parameters (for recursion_parameterized)
    r.insert("__INNER_LOG_GUEST_BL__".into(), inner_log_size.to_string());
    r.insert("__INNER_PUB_INPUT_SIZE__".into(), inner_pub_input_size.to_string());
    r.insert("__INNER_PM_LOG__".into(), inner_pm_log.to_string());
    r.insert(
        "__INNER_BYTECODE_ZERO_EVAL__".into(),
        inner_bytecode.instructions_multilinear[0].as_canonical_u64().to_string(),
    );

    // Table-specific replacements
    let mut lookup_indexes_str = vec![];
    let mut lookup_values_str = vec![];
    let mut num_cols_air = vec![];
    let mut air_degrees = vec![];
    let mut n_air_columns = vec![];
    let mut air_down_columns = vec![];

    for table in ALL_TABLES {
        let this_look_f_indexes_str = table
            .lookups()
            .iter()
            .map(|lookup_f| lookup_f.index.to_string())
            .collect::<Vec<_>>();
        lookup_indexes_str.push(format!("[{}]", this_look_f_indexes_str.join(", ")));
        num_cols_air.push(table.n_columns().to_string());
        let this_lookup_f_values_str = table
            .lookups()
            .iter()
            .map(|lookup_f| {
                format!(
                    "[{}]",
                    lookup_f.values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>();
        lookup_values_str.push(format!("[{}]", this_lookup_f_values_str.join(", ")));
        air_degrees.push(table.degree_air().to_string());
        n_air_columns.push(table.n_columns().to_string());
        air_down_columns.push(format!(
            "[{}]",
            table.down_column_indexes().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        ));
    }

    r.insert("LOOKUPS_INDEXES_PLACEHOLDER".into(), format!("[{}]", lookup_indexes_str.join(", ")));
    r.insert("LOOKUPS_VALUES_PLACEHOLDER".into(), format!("[{}]", lookup_values_str.join(", ")));
    r.insert("NUM_COLS_AIR_PLACEHOLDER".into(), format!("[{}]", num_cols_air.join(", ")));
    r.insert("AIR_DEGREES_PLACEHOLDER".into(), format!("[{}]", air_degrees.join(", ")));
    r.insert("N_AIR_COLUMNS_PLACEHOLDER".into(), format!("[{}]", n_air_columns.join(", ")));
    r.insert("AIR_DOWN_COLUMNS_PLACEHOLDER".into(), format!("[{}]", air_down_columns.join(", ")));

    // AIR constraint evaluation functions
    r.insert("EVALUATE_AIR_FUNCTIONS_PLACEHOLDER".into(), generate_all_air_evals_dsl());

    // Set up temp directory and compile.
    let temp_dir = setup_hybrid_circuit_dir();
    let main_path = temp_dir.join("das_hybrid_main.py");
    let filepath = main_path.to_str().unwrap().to_string();

    compile_program_with_flags(
        &ProgramSource::Filepath(filepath),
        CompilationFlags { replacements: r },
    )
}

/// Set up a temporary directory for hybrid circuit compilation.
/// Symlinks rec_aggregation files and copies both das_unified_main.py and recursion_param.py.
fn setup_hybrid_circuit_dir() -> PathBuf {
    let dir_name = format!(
        "das_hybrid_circuit_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    );
    let temp_dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

    let rec_agg = Path::new(REC_AGG_DIR);

    // Symlink the Python files we need from rec_aggregation.
    for file in [
        "recursion.py",
        "whir.py",
        "fiat_shamir.py",
        "hashing.py",
        "utils.py",
    ] {
        let target = temp_dir.join(file);
        let source = rec_agg.join(file);
        let _ = std::fs::remove_file(&target);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&source, &target)
            .unwrap_or_else(|e| panic!("failed to symlink {}: {}", file, e));
        #[cfg(not(unix))]
        std::fs::copy(&source, &target)
            .unwrap_or_else(|e| panic!("failed to copy {}: {}", file, e));
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Copy our hybrid main circuit.
    let our_main = manifest_dir.join("das_hybrid_main.py");
    let target_main = temp_dir.join("das_hybrid_main.py");
    let _ = std::fs::remove_file(&target_main);
    std::fs::copy(&our_main, &target_main).expect("failed to copy das_hybrid_main.py");

    // Copy the parameterized recursion module (needed by das_hybrid_main.py).
    let our_rec_param = manifest_dir.join("recursion_param.py");
    let target_rec_param = temp_dir.join("recursion_param.py");
    let _ = std::fs::remove_file(&target_rec_param);
    std::fs::copy(&our_rec_param, &target_rec_param).expect("failed to copy recursion_param.py");

    temp_dir
}

/// Build the public input for the hybrid circuit.
///
/// Layout: [aggregated_data_hash(DIGEST_SIZE),
///          inner_bytecode_claim(padded),
///          self_bytecode_claim(padded),
///          inner_bytecode_hash(DIGEST_SIZE),
///          self_bytecode_hash(DIGEST_SIZE)]
///
/// For leaf mode, pass batch_hashes (from commitment chain-hashing).
/// For recursion mode, pass `&[]`.
pub fn build_hybrid_public_input(
    batch_hashes: &[[F; DIGEST_SIZE]],
    inner_bytecode: &Bytecode,
    self_bytecode: &Bytecode,
) -> Vec<F> {
    let mut aggregated_data_hash = [F::ZERO; DIGEST_SIZE];
    for bh in batch_hashes {
        aggregated_data_hash = utils::poseidon16_compress_pair(&aggregated_data_hash, bh);
    }

    // Inner claim dimensions.
    let inner_bpnv = inner_bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let inner_claim_padded = ((inner_bpnv + 1) * DIMENSION).next_multiple_of(DIGEST_LEN);

    // Self claim dimensions.
    let self_bpnv = self_bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let self_claim_padded = ((self_bpnv + 1) * DIMENSION).next_multiple_of(DIGEST_LEN);

    let pub_input_size =
        DIGEST_LEN + inner_claim_padded + self_claim_padded + DIGEST_LEN + DIGEST_LEN;
    let mut public_input = vec![F::ZERO; pub_input_size];

    // Aggregated data hash.
    public_input[..DIGEST_SIZE].copy_from_slice(&aggregated_data_hash);

    // Default inner bytecode claim (zero point, inner_bytecode_zero_eval).
    let inner_value_offset = DIGEST_SIZE + inner_bpnv * DIMENSION;
    public_input[inner_value_offset] = inner_bytecode.instructions_multilinear[0];

    // Default self bytecode claim (zero point, self_bytecode_zero_eval).
    let self_claim_start = DIGEST_SIZE + inner_claim_padded;
    let self_value_offset = self_claim_start + self_bpnv * DIMENSION;
    public_input[self_value_offset] = self_bytecode.instructions_multilinear[0];

    // Inner bytecode hash.
    let inner_bh_offset = DIGEST_SIZE + inner_claim_padded + self_claim_padded;
    let inner_hash =
        utils::poseidon16_compress_pair(&inner_bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    public_input[inner_bh_offset..inner_bh_offset + DIGEST_SIZE].copy_from_slice(&inner_hash);

    // Self bytecode hash.
    let self_bh_offset = inner_bh_offset + DIGEST_SIZE;
    let self_hash =
        utils::poseidon16_compress_pair(&self_bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    public_input[self_bh_offset..self_bh_offset + DIGEST_SIZE].copy_from_slice(&self_hash);

    public_input
}

const AIR_INNER_VALUES_VAR: &str = "inner_evals";

fn air_eval_in_zk_dsl<T: TableT>(table: T) -> String
where
    T::ExtraData: Default,
{
    let (constraints, bus_flag, bus_data) =
        get_symbolic_constraints_and_bus_data_values::<F, _>(&table);
    let mut vars_counter = Counter::new();
    let mut cache: HashMap<u32, String> = HashMap::new();

    let mut res = format!(
        "def evaluate_air_constraints_table_{}({}, air_alpha_powers, bus_beta, logup_alphas_eq_poly):\n",
        table.table().index(),
        AIR_INNER_VALUES_VAR
    );

    let n_constraints = constraints.len();
    res += &format!("\n    constraints_buf = Array(DIM * {})", n_constraints);
    for (index, constraint) in constraints.iter().enumerate() {
        let dest = format!("constraints_buf + {} * DIM", index);
        eval_air_constraint(
            *constraint,
            Some(&dest),
            &mut cache,
            &mut res,
            &mut vars_counter,
        );
    }

    let flag = eval_air_constraint(bus_flag, None, &mut cache, &mut res, &mut vars_counter);
    res += &format!("\n    buff = Array(DIM * {})", bus_data.len());
    for (i, data) in bus_data.iter().enumerate() {
        let data_str = eval_air_constraint(*data, None, &mut cache, &mut res, &mut vars_counter);
        res += &format!("\n    copy_5({}, buff + DIM * {})", data_str, i);
    }
    res += "\n    bus_res_init = Array(DIM)";
    res += &format!(
        "\n    dot_product_ee(buff, logup_alphas_eq_poly, bus_res_init, {})",
        bus_data.len()
    );
    res += &format!(
        "\n    bus_res: Mut = add_extension_ret(mul_base_extension_ret(LOGUP_PRECOMPILE_DOMAINSEP, logup_alphas_eq_poly + {} * DIM), bus_res_init)",
        max_bus_width_including_domainsep().next_power_of_two() - 1
    );
    res += "\n    bus_res = mul_extension_ret(bus_res, bus_beta)";
    res += &format!("\n    sum: Mut = add_extension_ret(bus_res, {})", flag);

    res += "\n    weighted_constraints = Array(DIM)";
    res += &format!(
        "\n    dot_product_ee(air_alpha_powers + DIM, constraints_buf, weighted_constraints, {})",
        n_constraints
    );
    res += "\n    sum = add_extension_ret(sum, weighted_constraints)";

    res += "\n    return sum";
    res += "\n";
    res
}

fn eval_air_constraint(
    expr: SymbolicExpression<F>,
    dest: Option<&str>,
    cache: &mut HashMap<u32, String>,
    res: &mut String,
    ctr: &mut Counter,
) -> String {
    match expr {
        SymbolicExpression::Constant(c) => {
            let v = format!("aux_{}", ctr.get_next());
            res.push_str(&format!("\n    {} = embed_in_ef({})", v, c.as_canonical_u32()));
            v
        }
        SymbolicExpression::Variable(v) => {
            format!("{} + DIM * {}", AIR_INNER_VALUES_VAR, v.index)
        }
        SymbolicExpression::Operation(idx) => {
            if let Some(v) = cache.get(&idx) {
                if let Some(d) = dest {
                    res.push_str(&format!("\n    copy_5({}, {})", v, d));
                }
                return v.clone();
            }
            let node = get_node::<F>(idx);
            let v = match node.op {
                SymbolicOperation::Neg => {
                    let a = eval_air_constraint(node.lhs, None, cache, res, ctr);
                    let v = format!("aux_{}", ctr.get_next());
                    res.push_str(&format!("\n    {} = opposite_extension_ret({})", v, a));
                    v
                }
                _ => eval_air_binop(node.op, node.lhs, node.rhs, dest, cache, res, ctr),
            };
            if let Some(d) = dest {
                if v != d {
                    res.push_str(&format!("\n    copy_5({}, {})", v, d));
                }
            }
            cache.insert(idx, v.clone());
            v
        }
    }
}

fn eval_air_binop(
    op: SymbolicOperation,
    lhs: SymbolicExpression<F>,
    rhs: SymbolicExpression<F>,
    dest: Option<&str>,
    cache: &mut HashMap<u32, String>,
    res: &mut String,
    ctr: &mut Counter,
) -> String {
    let c0 = match lhs {
        SymbolicExpression::Constant(c) => Some(c.as_canonical_u32()),
        _ => None,
    };
    let c1 = match rhs {
        SymbolicExpression::Constant(c) => Some(c.as_canonical_u32()),
        _ => None,
    };

    match (c0, c1) {
        (None, None) => {
            let a = eval_air_constraint(lhs, None, cache, res, ctr);
            let b = eval_air_constraint(rhs, None, cache, res, ctr);
            if let Some(d) = dest {
                let f = match op {
                    SymbolicOperation::Mul => "mul_extension",
                    SymbolicOperation::Add => "add_ee",
                    SymbolicOperation::Sub => "sub_extension",
                    _ => unreachable!(),
                };
                res.push_str(&format!("\n    {}({}, {}, {})", f, a, b, d));
                d.to_string()
            } else {
                let f = match op {
                    SymbolicOperation::Mul => "mul_extension_ret",
                    SymbolicOperation::Add => "add_extension_ret",
                    SymbolicOperation::Sub => "sub_extension_ret",
                    _ => unreachable!(),
                };
                let v = format!("aux_{}", ctr.get_next());
                res.push_str(&format!("\n    {} = {}({}, {})", v, f, a, b));
                v
            }
        }
        _ if matches!(op, SymbolicOperation::Mul | SymbolicOperation::Add) => {
            let (c, ext_expr) = match (c0, c1) {
                (Some(c), _) => (c, rhs),
                (_, Some(c)) => (c, lhs),
                _ => unreachable!(),
            };
            let ext = eval_air_constraint(ext_expr, None, cache, res, ctr);
            if let Some(d) = dest {
                let f = if matches!(op, SymbolicOperation::Mul) {
                    "dot_product_be"
                } else {
                    "add_be"
                };
                emit_base_precompile(res, ctr, f, c, &ext, d);
                d.to_string()
            } else {
                let f = if matches!(op, SymbolicOperation::Mul) {
                    "mul_base_extension_ret"
                } else {
                    "add_base_extension_ret"
                };
                let v = format!("aux_{}", ctr.get_next());
                res.push_str(&format!("\n    {} = {}({}, {})", v, f, c, ext));
                v
            }
        }
        (Some(c), _) => {
            let ext = eval_air_constraint(rhs, None, cache, res, ctr);
            let v = format!("aux_{}", ctr.get_next());
            res.push_str(&format!(
                "\n    {} = sub_base_extension_ret({}, {})",
                v, c, ext
            ));
            v
        }
        (_, Some(c)) => {
            let ext = eval_air_constraint(lhs, None, cache, res, ctr);
            if let Some(d) = dest {
                emit_base_precompile(res, ctr, "add_be", c, d, &ext);
                d.to_string()
            } else {
                let v = format!("aux_{}", ctr.get_next());
                res.push_str(&format!(
                    "\n    {} = sub_extension_base_ret({}, {})",
                    v, ext, c
                ));
                v
            }
        }
    }
}

fn emit_base_precompile(
    res: &mut String,
    ctr: &mut Counter,
    func: &str,
    c: u32,
    arg2: &str,
    arg3: &str,
) {
    let tmp = format!("aux_{}", ctr.get_next());
    res.push_str(&format!(
        "\n    {} = Array(1)\n    {}[0] = {}\n    {}({}, {}, {})",
        tmp, tmp, c, func, tmp, arg2, arg3
    ));
}

// ─── Public input ──────────────────────────────────────────────────────

/// Prepare the public input buffer for the standalone (fast-path) DAS circuit.
pub fn prepare_public_input(batch: &Batch) -> Vec<F> {
    build_public_input(&batch.commitments, None)
}

/// Prepare the public input buffer including the bytecode hash
/// (needed for recursive aggregation).
pub fn prepare_public_input_with_hash(batch: &Batch, bytecode: &Bytecode) -> Vec<F> {
    build_public_input(&batch.commitments, Some(&bytecode.hash))
}

/// Build the public input for verification given commitments and bytecode.
/// Used by the verifier which doesn't have access to the full Batch.
pub fn build_verifier_public_input(commitments: &[Commitment], bytecode: &Bytecode) -> Vec<F> {
    build_public_input(commitments, Some(&bytecode.hash))
}

/// Build the non-reserved public input vector from commitments.
///
/// Layout (constant size = 2 * DIGEST_SIZE = 16):
///   [0 .. DIGEST_SIZE]:             chain_hash(commitments)
///   [DIGEST_SIZE .. 2*DIGEST_SIZE]: hash(bytecode.hash, SNARK_DOMAIN_SEP)
///
/// The chain hash is: h = poseidon(0, com_0), h = poseidon(h, com_1), ...
/// The circuit verifies this hash against commitments passed in private input.
fn build_public_input(commitments: &[Commitment], bytecode_hash: Option<&[F; DIGEST_SIZE]>) -> Vec<F> {
    // Chain-hash commitments: h_0 = hash(zero, com_0), h_i = hash(h_{i-1}, com_i)
    let batch_hash = hash_commitments(commitments);

    let pub_input_size = DIGEST_SIZE + if bytecode_hash.is_some() { DIGEST_SIZE } else { 0 };
    let mut public_input = vec![F::ZERO; pub_input_size];
    public_input[..DIGEST_SIZE].copy_from_slice(&batch_hash);

    if let Some(bh) = bytecode_hash {
        let hash = utils::poseidon16_compress_pair(bh, &lean_prover::SNARK_DOMAIN_SEP);
        public_input[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(&hash);
    }

    public_input
}

/// Build the public input for the unified circuit.
///
/// Layout: [aggregated_data_hash(DIGEST_SIZE), bytecode_claim(padded), bytecode_hash(DIGEST_SIZE)]
///
/// `batch_hashes` contains the chain-hash of commitments for each raw batch.
/// These are chain-hashed: h = poseidon(0, h_0), h = poseidon(h, h_1), ...
///
/// For leaf mode with one batch, pass `&[hash_commitments(commitments)]`.
/// For recursion mode, pass `&[]` (aggregated hash comes from children, handled separately).
pub fn build_unified_public_input(
    batch_hashes: &[[F; DIGEST_SIZE]],
    bytecode: &Bytecode,
) -> Vec<F> {
    // Chain-hash the batch hashes into aggregated_data_hash
    let mut aggregated_data_hash = [F::ZERO; DIGEST_SIZE];
    for bh in batch_hashes {
        aggregated_data_hash = utils::poseidon16_compress_pair(&aggregated_data_hash, bh);
    }
    let bytecode_point_n_vars = bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
    let claim_data_size = (bytecode_point_n_vars + 1) * DIMENSION;
    let claim_data_size_padded = claim_data_size.next_multiple_of(DIGEST_LEN);

    let pub_input_size = DIGEST_LEN + claim_data_size_padded + DIGEST_LEN;
    let mut public_input = vec![F::ZERO; pub_input_size];

    // Aggregated data hash
    public_input[..DIGEST_SIZE].copy_from_slice(&aggregated_data_hash);

    // Default bytecode claim (zero point, bytecode_zero_eval)
    // Point is all zeros (already initialized), value is bytecode[0]
    let value_offset = DIGEST_SIZE + bytecode_point_n_vars * DIMENSION;
    public_input[value_offset] = bytecode.instructions_multilinear[0];

    // Bytecode hash
    let bh_offset = DIGEST_SIZE + claim_data_size_padded;
    let hash = utils::poseidon16_compress_pair(&bytecode.hash, &lean_prover::SNARK_DOMAIN_SEP);
    public_input[bh_offset..bh_offset + DIGEST_SIZE].copy_from_slice(&hash);

    public_input
}

/// Chain-hash commitments: h = poseidon(0, com_0), h = poseidon(h, com_1), ...
pub fn hash_commitments(commitments: &[Commitment]) -> [F; DIGEST_SIZE] {
    let mut hash = [F::ZERO; DIGEST_SIZE];
    for com in commitments {
        hash = utils::poseidon16_compress_pair(&hash, &com.root);
    }
    hash
}

// ─── Private input ─────────────────────────────────────────────────────

/// Prepare the private input buffer for the standalone (fast-path) DAS circuit.
///
/// Layout: three regions, all accessible via `hint_private_input_start`:
///
/// Region 0 — commitments (for hash verification):
///   `[com_0[0..8], com_1[0..8], ..., com_{B-1}[0..8]]`
///
/// Region 1 — row-major BF (for Merkle hashing):
///   `[cw_0[0], cw_0[1], ..., cw_0[N-1], cw_1[0], ..., cw_{B-1}[N-1]]`
///
/// Region 2 — EF-embedded row-major (for vectorized RLC, first V codewords):
///   Each BF value v is stored as (v, 0, 0, 0, 0) in EF layout (DIM words).
///   `[cw_0_ef[0..N*DIM], ..., cw_{V-1}_ef[0..N*DIM]]`
///
/// Region 3 — column-major BF (for dot_product_be, remaining B-V codewords):
///   `[col_0[V..B-1], col_1[V..B-1], ..., col_{N-1}[V..B-1]]`
///   where col_j[i] = codewords[i].evals[j] for i in V..B
pub fn prepare_private_input(batch: &Batch, _degree: usize) -> Vec<F> {
    let b = batch.codewords.len();
    let n = batch.codewords[0].len();

    let mut buf = Vec::with_capacity(b * DIGEST_SIZE + b * n + n * b);

    // Region 0: commitments
    for com in &batch.commitments {
        buf.extend_from_slice(&com.root);
    }

    // Region 1: row-major BF (all B codewords, for Merkle hashing)
    for cw in &batch.codewords {
        buf.extend_from_slice(&cw.evals);
    }

    // Region 2: column-major BF (all B codewords, for RLC dot_product_be)
    // TODO: with a strided dot_product_be, we could use row-major only and
    // eliminate Region 2 — reading RLC inputs at stride N from row-major.
    // This would halve the private input and remove the dual-layout.
    for j in 0..n {
        for cw in &batch.codewords {
            buf.push(cw.evals[j]);
        }
    }

    buf
}

/// Prepare the private input for the unified circuit in leaf mode.
///
/// Layout:
///   [n_rec=0, n_raw=1, ptr_batch_0, ptr_sumcheck(unused),
///    batch_data: coms + row_major + col_major]
pub fn prepare_unified_leaf_private_input(batch: &Batch, public_memory_len: usize) -> Vec<F> {
    let batch_data = prepare_batch_data(batch);
    let header_size = 4; // [n_rec, n_raw, ptr_batch, ptr_sumcheck]
    let batch_data_ptr = public_memory_len + header_size;

    let mut buf = Vec::with_capacity(header_size + batch_data.len());
    buf.push(F::ZERO);                              // n_recursions = 0
    buf.push(F::ONE);                               // n_raw_batches = 1
    buf.push(F::from_usize(batch_data_ptr));         // ptr_batch_0
    buf.push(F::from_usize(batch_data_ptr + batch_data.len())); // ptr_sumcheck (unused)
    buf.extend_from_slice(&batch_data);

    buf
}

/// Prepare the raw batch data block for the unified circuit.
///
/// Layout: [com_0..com_{B-1}, cw_row_major(B*N), cw_ef_embedded(V*N*DIM), cw_col_major((B-V)*N)]
pub fn prepare_batch_data(batch: &Batch) -> Vec<F> {
    let b = batch.codewords.len();
    let n = batch.codewords[0].len();
    let v = 1.min(b); // RLC_VEC_BATCH

    let mut buf = Vec::with_capacity(
        b * DIGEST_SIZE + b * n + v * n * DIMENSION + (b - v) * n,
    );

    // Commitments
    for com in &batch.commitments {
        buf.extend_from_slice(&com.root);
    }

    // Row-major BF
    for cw in &batch.codewords {
        buf.extend_from_slice(&cw.evals);
    }

    // EF-embedded row-major (first V codewords)
    for cw in &batch.codewords[..v] {
        for &val in &cw.evals {
            buf.push(val);
            for _ in 1..DIMENSION {
                buf.push(F::ZERO);
            }
        }
    }

    // Column-major BF (remaining B-V codewords)
    if v < b {
        for j in 0..n {
            for cw in &batch.codewords[v..] {
                buf.push(cw.evals[j]);
            }
        }
    }

    buf
}

// ─── Proving and verification ──────────────────────────────────────────

/// Run the standalone DAS circuit inside the ZKVM and produce a STARK proof.
pub fn prove_circuit(
    bytecode: &Bytecode,
    batch: &Batch,
    degree: usize,
) -> (backend::Proof<F>, Vec<F>) {
    let public_input = prepare_public_input_with_hash(batch, bytecode);
    let private_input = prepare_private_input(batch, degree);

    let witness = ExecutionWitness {
        private_input: &private_input,
        xmss_signatures: &[],
        merkle_paths: &[],
    };

    let whir_config = default_whir_config(1);

    tracing::info!(
        batch_size = batch.codewords.len(),
        codeword_len = batch.codewords[0].len(),
        "proving DAS circuit in ZKVM"
    );

    let execution_proof = prove_execution(
        bytecode,
        &public_input,
        &witness,
        &whir_config,
        false, // no profiling
    );

    (execution_proof.proof, public_input)
}

/// Prove a batch using the unified circuit in leaf mode.
pub fn prove_unified_leaf(
    bytecode: &Bytecode,
    batch: &Batch,
    log_inv_rate: usize,
) -> (backend::Proof<F>, Vec<F>) {
    let batch_hash = hash_commitments(&batch.commitments);
    let public_input = build_unified_public_input(&[batch_hash], bytecode);

    let public_memory = build_public_memory(&public_input);
    let private_input = prepare_unified_leaf_private_input(batch, public_memory.len());

    let witness = ExecutionWitness {
        private_input: &private_input,
        xmss_signatures: &[],
        merkle_paths: &[],
    };

    let whir_config = default_whir_config(log_inv_rate);

    tracing::info!(
        batch_size = batch.codewords.len(),
        codeword_len = batch.codewords[0].len(),
        "proving unified DAS circuit (leaf mode)"
    );

    let execution_proof = prove_execution(
        bytecode,
        &public_input,
        &witness,
        &whir_config,
        false,
    );

    (execution_proof.proof, public_input)
}

/// Verify a DAS circuit proof given the full Batch (prover-side).
pub fn verify_circuit(
    bytecode: &Bytecode,
    batch: &Batch,
    proof: backend::Proof<F>,
) -> Result<(), String> {
    let public_input = prepare_public_input_with_hash(batch, bytecode);
    verify_circuit_with_public_input(bytecode, &public_input, proof)
}

/// Verify a DAS circuit proof given pre-built public input (verifier-side).
pub fn verify_circuit_with_public_input(
    bytecode: &Bytecode,
    public_input: &[F],
    proof: backend::Proof<F>,
) -> Result<(), String> {
    verify_execution(bytecode, public_input, proof)
        .map(|_| ())
        .map_err(|e| format!("proof verification failed: {:?}", e))
}

// ─── Helper: build public memory ──────────────────────────────────────

/// Build the full public memory array from non-reserved public input.
/// The public memory is padded to a power-of-two size.
pub(crate) fn build_public_memory(non_reserved_public_input: &[F]) -> Vec<F> {
    let total_size = NONRESERVED_PROGRAM_INPUT_START + non_reserved_public_input.len();
    let padded_size = total_size.next_power_of_two();
    let mut memory = vec![F::ZERO; padded_size];
    memory[NONRESERVED_PROGRAM_INPUT_START..NONRESERVED_PROGRAM_INPUT_START + non_reserved_public_input.len()]
        .copy_from_slice(non_reserved_public_input);
    memory
}

#[cfg(test)]
mod tests {
    use super::*;
    use das_core::types::{Codeword, EF, EVALS_PER_LEAF};
    use das_core::{commit_codeword, derive_challenge, rs_encode};

    fn make_batch(batch_size: usize, codeword_len: usize) -> Batch {
        // Use constant codewords (degree 0 polynomials) — always valid RS.
        // Each codeword has a distinct constant so the RLC is non-trivial.
        let codewords: Vec<Codeword> = (0..batch_size)
            .map(|i| {
                let c = F::from_usize(i + 1);
                let evals = vec![c; codeword_len];
                Codeword::new(evals)
            })
            .collect();

        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword(cw)).collect();
        let challenges: Vec<EF> = commitments.iter().map(derive_challenge).collect();

        Batch {
            indices: (0..batch_size).collect(),
            codewords,
            commitments,
            challenges,
        }
    }

    /// Create a batch of proper RS codewords from random-ish polynomials.
    fn make_rs_batch(batch_size: usize, codeword_len: usize, degree: usize) -> Batch {
        assert!(degree < codeword_len / 2, "degree must be < n/2 for RS rate 1/2");
        let codewords: Vec<Codeword> = (0..batch_size)
            .map(|i| {
                let msg: Vec<F> = (0..degree)
                    .map(|j| F::from_usize((i + 1) * (j + 1) + i * 7 + 3))
                    .collect();
                rs_encode(&msg, codeword_len)
            })
            .collect();

        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword(cw)).collect();
        let challenges: Vec<EF> = commitments.iter().map(derive_challenge).collect();

        Batch {
            indices: (0..batch_size).collect(),
            codewords,
            commitments,
            challenges,
        }
    }

    #[test]
    fn test_compile_circuit() {
        let bytecode = compile_das_circuit(2, 16, EVALS_PER_LEAF, 8);
        assert!(
            !bytecode.instructions.is_empty(),
            "compiled circuit should have instructions"
        );
    }

    #[test]
    fn test_prepare_public_input() {
        let batch = make_batch(2, 16);
        let pub_input = prepare_public_input(&batch);

        // Public input is now just the chain hash of commitments (DIGEST_SIZE elements).
        assert_eq!(pub_input.len(), DIGEST_SIZE);

        // Verify the hash matches what hash_commitments produces.
        let expected_hash = hash_commitments(&batch.commitments);
        assert_eq!(&pub_input[..DIGEST_SIZE], &expected_hash);
    }

    #[test]
    fn test_prepare_private_input() {
        let batch = make_batch(2, 16);
        let degree = 8;
        let priv_input = prepare_private_input(&batch, degree);

        let b = 2;
        let n = 16;
        let v = 1; // RLC_VEC_BATCH
        // Region 0 (B*DIGEST) + Region 1 (B*N) + Region 2 (V*N*DIM) + Region 3 ((B-V)*N)
        assert_eq!(priv_input.len(), b * DIGEST_SIZE + b * n + v * n * DIMENSION + (b - v) * n);
        // Region 0 (commitments)
        assert_eq!(&priv_input[0..DIGEST_SIZE], &batch.commitments[0].root);
        assert_eq!(&priv_input[DIGEST_SIZE..2 * DIGEST_SIZE], &batch.commitments[1].root);
        // Region 1 (row-major BF)
        let cw_base = b * DIGEST_SIZE;
        assert_eq!(priv_input[cw_base], batch.codewords[0].evals[0]);
        assert_eq!(priv_input[cw_base + n], batch.codewords[1].evals[0]);
        // Region 2 (EF-embedded row-major for first V=1 codeword)
        let ef_base = cw_base + b * n;
        assert_eq!(priv_input[ef_base], batch.codewords[0].evals[0]);
        assert_eq!(priv_input[ef_base + 1], F::ZERO); // padding
        assert_eq!(priv_input[ef_base + DIMENSION], batch.codewords[0].evals[1]); // next element
        // Region 3 (column-major for remaining B-V=1 codeword)
        let col_base = ef_base + v * n * DIMENSION;
        // col_j contains codewords[V..B][j], so col_0 = cw_1[0], col_1 = cw_1[1], ...
        assert_eq!(priv_input[col_base], batch.codewords[1].evals[0]);
        assert_eq!(priv_input[col_base + 1], batch.codewords[1].evals[1]);
    }

    #[test]
    fn test_minimal_poseidon() {
        use lean_compiler::compile_program;

        let nrpis = NONRESERVED_PROGRAM_INPUT_START;
        let evals: Vec<F> = (0..16).map(|j| F::from_usize(j + 1)).collect();
        let expected_hash = utils::poseidon16_compress(evals.clone().try_into().unwrap());

        // Test 1: read from public memory and assert value
        {
            let source = format!(
"def main():
    pub_mem = {nrpis}
    x = pub_mem[0]
    assert x == 1
    return
");
            let bytecode = compile_program(&ProgramSource::Raw(source));
            let mut public_input = vec![F::ZERO; 1024];
            public_input[0] = F::from_usize(1);
            let result = lean_vm::try_execute_bytecode(
                &bytecode,
                &public_input,
                &ExecutionWitness::empty(),
                false,
            );
            assert!(result.is_ok(), "pub_mem read failed: {:?}", result.err());
        }

        // Test 2: copy from pub_mem into Array, poseidon, assert hash
        {
            let hash0 = expected_hash[0].to_string();
            let source = format!(
"from snark_lib import *

def main():
    pub_mem = {nrpis}
    data = Array(16)
    for i in unroll(0, 16):
        data[i] = pub_mem[i]
    h = Array(8)
    poseidon16_compress(data, data + 8, h)
    assert h[0] == {hash0}
    return
");
            let flags = CompilationFlags { replacements: BTreeMap::new() };
            let bytecode = compile_program_with_flags(&ProgramSource::Raw(source), flags);
            let mut public_input = vec![F::ZERO; 1024];
            public_input[0..16].copy_from_slice(&evals);
            let result = lean_vm::try_execute_bytecode(
                &bytecode,
                &public_input,
                &ExecutionWitness::empty(),
                false,
            );
            assert!(result.is_ok(), "poseidon hash failed: {:?}", result.err());
        }
    }

    #[test]
    fn test_prove_and_verify_circuit() {
        let batch_size = 1;
        let codeword_len = 32;

        let degree = codeword_len / 2;
        let bytecode = compile_das_circuit(batch_size, codeword_len, EVALS_PER_LEAF, degree);
        let batch = make_batch(batch_size, codeword_len);

        let (proof, _pub_input) = prove_circuit(&bytecode, &batch, degree);
        let result = verify_circuit(&bytecode, &batch, proof);
        assert!(result.is_ok(), "proof should verify: {:?}", result.err());
    }

    #[test]
    fn test_tampered_codeword_rejected() {
        let batch_size = 1;
        let codeword_len = 32;

        let mut codewords: Vec<Codeword> = (0..batch_size)
            .map(|i| {
                let c = F::from_usize(i + 1);
                let evals = vec![c; codeword_len];
                Codeword::new(evals)
            })
            .collect();

        codewords[0].evals[7] = F::from_usize(999_999);

        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword(cw)).collect();
        let challenges: Vec<EF> = commitments.iter().map(derive_challenge).collect();

        let batch = Batch {
            indices: (0..batch_size).collect(),
            codewords,
            commitments,
            challenges,
        };

        let degree = codeword_len / 2;
        let bytecode = compile_das_circuit(batch_size, codeword_len, EVALS_PER_LEAF, degree);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove_circuit(&bytecode, &batch, degree)
        }));

        assert!(
            result.is_err(),
            "tampered codeword should fail FRI constant check"
        );
    }

    #[test]
    fn test_tampered_rs_codeword_rejected() {
        let batch_size = 1;
        let codeword_len = 32;
        let degree = 8;

        let mut batch = make_rs_batch(batch_size, codeword_len, degree);

        assert_ne!(batch.codewords[0].evals[0], batch.codewords[0].evals[1]);

        batch.codewords[0].evals[5] = batch.codewords[0].evals[5] + F::from_usize(1);

        batch.commitments = batch.codewords.iter().map(|cw| commit_codeword(cw)).collect();
        batch.challenges = batch.commitments.iter().map(derive_challenge).collect();

        let bytecode = compile_das_circuit(batch_size, codeword_len, EVALS_PER_LEAF, degree);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove_circuit(&bytecode, &batch, degree)
        }));

        assert!(
            result.is_err(),
            "tampered RS codeword should fail FRI constant check"
        );
    }

    #[test]
    fn test_prove_and_verify_nonconstant_rs() {
        let batch_size = 1;
        let codeword_len = 32;
        let degree = 4;

        let bytecode = compile_das_circuit(batch_size, codeword_len, EVALS_PER_LEAF, degree);
        let batch = make_rs_batch(batch_size, codeword_len, degree);

        assert_ne!(batch.codewords[0].evals[0], batch.codewords[0].evals[1]);

        let (proof, _pub_input) = prove_circuit(&bytecode, &batch, degree);
        let result = verify_circuit(&bytecode, &batch, proof);
        assert!(result.is_ok(), "non-constant RS codeword should verify: {:?}", result.err());
    }

    #[test]
    fn test_prove_and_verify_multi_codeword_rs() {
        let batch_size = 2;
        let codeword_len = 32;
        let degree = 8;

        let bytecode = compile_das_circuit(batch_size, codeword_len, EVALS_PER_LEAF, degree);
        let batch = make_rs_batch(batch_size, codeword_len, degree);

        let (proof, _pub_input) = prove_circuit(&bytecode, &batch, degree);
        let result = verify_circuit(&bytecode, &batch, proof);
        assert!(result.is_ok(), "multi-codeword RS batch should verify: {:?}", result.err());
    }

    /// Helper to make a batch using a specific EPL for commitment.
    fn make_batch_epl(batch_size: usize, codeword_len: usize, epl: usize) -> Batch {
        use das_core::commit_codeword_epl;
        let codewords: Vec<Codeword> = (0..batch_size)
            .map(|i| {
                let c = F::from_usize(i + 1);
                let evals = vec![c; codeword_len];
                Codeword::new(evals)
            })
            .collect();

        let commitments: Vec<_> = codewords.iter().map(|cw| commit_codeword_epl(cw, epl)).collect();
        let challenges: Vec<EF> = commitments.iter().map(derive_challenge).collect();

        Batch {
            indices: (0..batch_size).collect(),
            codewords,
            commitments,
            challenges,
        }
    }

    #[test]
    fn test_prove_and_verify_epl32() {
        let batch_size = 1;
        let codeword_len = 64;
        let epl = 32;

        let degree = codeword_len / 2;
        let bytecode = compile_das_circuit(batch_size, codeword_len, epl, degree);
        let batch = make_batch_epl(batch_size, codeword_len, epl);

        let (proof, _pub_input) = prove_circuit(&bytecode, &batch, degree);
        let result = verify_circuit(&bytecode, &batch, proof);
        assert!(result.is_ok(), "EPL=32 circuit should verify: {:?}", result.err());
    }

    #[test]
    fn test_prove_and_verify_epl64() {
        let batch_size = 1;
        let codeword_len = 128;
        let epl = 64;

        let degree = codeword_len / 2;
        let bytecode = compile_das_circuit(batch_size, codeword_len, epl, degree);
        let batch = make_batch_epl(batch_size, codeword_len, epl);

        let (proof, _pub_input) = prove_circuit(&bytecode, &batch, degree);
        let result = verify_circuit(&bytecode, &batch, proof);
        assert!(result.is_ok(), "EPL=64 circuit should verify: {:?}", result.err());
    }

    #[test]
    fn dump_instruction_scaling() {
        for (n, d) in [(32, 16), (64, 32), (128, 64), (256, 128), (512, 256)] {
            let bc = compile_das_circuit(1, n, EVALS_PER_LEAF, d);
            println!("n={:5} deg={:5} → {:5} instructions", n, d, bc.instructions.len());
        }
        // Dump actual instructions for smallest case
        let bc = compile_das_circuit(1, 32, EVALS_PER_LEAF, 16);
        println!("\n=== n=32 deg=16: {} instructions ===", bc.instructions.len());
        for (i, instr) in bc.instructions.iter().enumerate() {
            println!("{:4}: {}", i, instr);
        }
    }

    #[test]
    fn test_compile_unified_circuit() {
        // The compiler's deep expression trees need a large stack.
        let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
        let handle = builder
            .spawn(|| {
                let bytecode = compile_unified_das_circuit(1, 32, EVALS_PER_LEAF, 16);
                assert!(
                    !bytecode.instructions.is_empty(),
                    "unified circuit should have instructions"
                );
                println!(
                    "unified circuit compiled: {} instructions, log_size={}",
                    bytecode.instructions.len(),
                    bytecode.log_size()
                );
            })
            .expect("failed to spawn thread");
        handle.join().expect("compilation thread panicked");
    }

    #[test]
    fn test_prove_unified_leaf_mode() {
        // The compiler's deep expression trees need a large stack.
        let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
        let handle = builder
            .spawn(|| {
                let batch_size = 1;
                let codeword_len = 32;
                let epl = EVALS_PER_LEAF;
                let degree = 16;
                let log_inv_rate = 1;

                let bytecode = compile_unified_das_circuit(batch_size, codeword_len, epl, degree);
                let batch = make_batch(batch_size, codeword_len);

                let (proof, pub_input) = prove_unified_leaf(&bytecode, &batch, log_inv_rate);
                println!(
                    "unified leaf proof: pub_input len={}, proof size={}",
                    pub_input.len(),
                    proof.proof_size_fe()
                );

                // Verify the proof
                let result = verify_circuit_with_public_input(&bytecode, &pub_input, proof);
                assert!(result.is_ok(), "unified leaf proof should verify: {:?}", result.err());
            })
            .expect("failed to spawn thread");
        handle.join().expect("unified leaf proving thread panicked");
    }
}
