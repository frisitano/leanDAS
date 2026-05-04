from snark_lib import *
from recursion_param import *
from hashing import *

MAX_INNER_PROOFS = __MAX_INNER_PROOFS__
SELF_PUB_MEM_SIZE = 2**INNER_PUBLIC_MEMORY_LOG_SIZE  # self circuit's public memory size

# Inner batch circuit parameters (compile-time, from Rust)
INNER_LOG_GUEST_BL = __INNER_LOG_GUEST_BL__
INNER_PIS = __INNER_PUB_INPUT_SIZE__
INNER_PM_LOG = __INNER_PM_LOG__
INNER_BYTECODE_ZERO_EVAL_VAL = __INNER_BYTECODE_ZERO_EVAL__
INNER_BPN = INNER_LOG_GUEST_BL + log2_ceil(N_INSTRUCTION_COLUMNS)
INNER_BCS = (INNER_BPN + 1) * DIM
INNER_BCS_PAD = next_multiple_of(INNER_BCS, DIGEST_LEN)
INNER_PM_SIZE = 2**INNER_PM_LOG

# Public input layout (within non-reserved public input):
#   [0 .. DIGEST_LEN]:                                  aggregated_data_hash
#   [DIGEST_LEN .. +INNER_BCS_PAD]:                     inner bytecode claim
#   [.. + BYTECODE_CLAIM_SIZE_PADDED]:                   self bytecode claim
#   [.. + DIGEST_LEN]:                                   inner bytecode hash
#   [.. + DIGEST_LEN]:                                   self bytecode hash
INNER_CLAIM_PUB_OFFSET = DIGEST_LEN
SELF_CLAIM_PUB_OFFSET = DIGEST_LEN + INNER_BCS_PAD
INNER_HASH_PUB_OFFSET = SELF_CLAIM_PUB_OFFSET + BYTECODE_CLAIM_SIZE_PADDED
SELF_HASH_PUB_OFFSET = INNER_HASH_PUB_OFFSET + DIGEST_LEN


def main():
    debug_assert(NONRESERVED_PROGRAM_INPUT_START % DIM == 0)
    pub_mem = NONRESERVED_PROGRAM_INPUT_START

    priv_start: Imu
    hint_private_input_start(priv_start)

    # Private input header:
    #   [n_recursions, n_inner_proofs,
    #    ptr_src_0, ..., ptr_src_{N-1},
    #    ptr_inner_sumcheck, ptr_self_sumcheck]
    n_recursions = priv_start[0]
    n_inner_proofs = priv_start[1]
    assert (n_recursions + n_inner_proofs) != 0
    assert n_recursions <= MAX_INNER_PROOFS
    assert n_inner_proofs <= MAX_INNER_PROOFS
    total_sources = n_recursions + n_inner_proofs
    source_ptrs = priv_start + 2
    inner_sumcheck_proof = source_ptrs[total_sources]
    self_sumcheck_proof = source_ptrs[total_sources + 1]

    aggregated_hash: Mut = ZERO_VEC_PTR

    # Claim counts:
    #   inner_claims = n_recursions (propagated) + 2 * n_inner_proofs (default + verification)
    #   self_claims  = 2 * n_recursions (propagated + verification)
    n_inner_claims = n_recursions + n_inner_proofs * 2
    n_self_claims = n_recursions * 2
    inner_claims = Array(n_inner_claims)
    self_claims = Array(n_self_claims)

    # --- Process self-referential recursive children ---
    for i in range(0, n_recursions):
        source = source_ptrs[i]
        bytecode_value_hint = source
        inner_pub_mem = bytecode_value_hint + DIM
        proof_transcript = inner_pub_mem + SELF_PUB_MEM_SIZE

        # Copy reserved memory area into inner pub mem
        for k in unroll(0, NONRESERVED_PROGRAM_INPUT_START / DIM):
            copy_5(k * DIM, inner_pub_mem + k * DIM)

        non_reserved_inner = inner_pub_mem + NONRESERVED_PROGRAM_INPUT_START

        # Copy own bytecode hashes into inner pub mem (self-referential: child = self)
        copy_8(pub_mem + INNER_HASH_PUB_OFFSET, non_reserved_inner + INNER_HASH_PUB_OFFSET)
        copy_8(pub_mem + SELF_HASH_PUB_OFFSET, non_reserved_inner + SELF_HASH_PUB_OFFSET)

        # Chain-hash child's aggregated_data_hash into ours
        inner_data_hash = non_reserved_inner
        new_hash = Array(DIGEST_LEN)
        poseidon16_compress(aggregated_hash, inner_data_hash, new_hash)
        aggregated_hash = new_hash

        # Propagate child's inner bytecode claim
        inner_claims[i] = non_reserved_inner + INNER_CLAIM_PUB_OFFSET

        # Propagate child's self bytecode claim
        self_claims[2 * i] = non_reserved_inner + SELF_CLAIM_PUB_OFFSET

        # Verify self proof — returns verification claim about self bytecode
        self_claims[2 * i + 1] = recursion(inner_pub_mem, proof_transcript, bytecode_value_hint)

    # --- Process inner batch proof children ---
    for i in range(0, n_inner_proofs):
        source = source_ptrs[n_recursions + i]
        bytecode_value_hint = source
        inner_pub_mem = bytecode_value_hint + DIM
        proof_transcript = inner_pub_mem + INNER_PM_SIZE

        # Copy reserved memory area into inner pub mem
        for k in unroll(0, NONRESERVED_PROGRAM_INPUT_START / DIM):
            copy_5(k * DIM, inner_pub_mem + k * DIM)

        non_reserved_inner = inner_pub_mem + NONRESERVED_PROGRAM_INPUT_START

        # Copy inner bytecode hash into inner circuit's bytecode hash slot
        inner_bc_hash_offset = INNER_PIS - DIGEST_LEN
        copy_8(pub_mem + INNER_HASH_PUB_OFFSET, non_reserved_inner + inner_bc_hash_offset)

        # Chain-hash inner circuit's batch_hash into aggregated_hash
        inner_data_hash = non_reserved_inner
        new_hash = Array(DIGEST_LEN)
        poseidon16_compress(aggregated_hash, inner_data_hash, new_hash)
        aggregated_hash = new_hash

        # Default inner bytecode claim (zero point, inner_bytecode_zero_eval)
        default_inner_claim = Array(INNER_BCS_PAD)
        for k in unroll(0, INNER_BPN):
            set_to_5_zeros(default_inner_claim + k * DIM)
        default_inner_claim[INNER_BPN * DIM] = INNER_BYTECODE_ZERO_EVAL_VAL
        for k in unroll(1, DIM):
            default_inner_claim[INNER_BPN * DIM + k] = 0
        for k in unroll(INNER_BCS, INNER_BCS_PAD):
            default_inner_claim[k] = 0
        inner_claims[n_recursions + 2 * i] = default_inner_claim

        # Verify inner proof — returns verification claim about inner bytecode
        inner_claims[n_recursions + 2 * i + 1] = recursion_parameterized(
            inner_pub_mem, proof_transcript, bytecode_value_hint,
            INNER_LOG_GUEST_BL, INNER_PIS, INNER_PM_LOG)

    # --- Verify aggregated hash matches public input ---
    for d in unroll(0, DIGEST_LEN):
        assert aggregated_hash[d] == pub_mem[d]

    # --- Inner bytecode claims ---
    inner_claim_output = pub_mem + INNER_CLAIM_PUB_OFFSET
    if n_inner_claims == 0:
        # No inner claims: write default inner bytecode claim
        for k in unroll(0, INNER_BPN):
            set_to_5_zeros(inner_claim_output + k * DIM)
        inner_claim_output[INNER_BPN * DIM] = INNER_BYTECODE_ZERO_EVAL_VAL
        for k in unroll(1, DIM):
            inner_claim_output[INNER_BPN * DIM + k] = 0
    else:
        reduce_inner_claims(inner_claims, n_inner_claims, inner_claim_output, inner_sumcheck_proof)

    # --- Self bytecode claims ---
    self_claim_output = pub_mem + SELF_CLAIM_PUB_OFFSET
    if n_self_claims == 0:
        # No self claims: write default self bytecode claim
        for k in unroll(0, BYTECODE_POINT_N_VARS):
            set_to_5_zeros(self_claim_output + k * DIM)
        self_claim_output[BYTECODE_POINT_N_VARS * DIM] = BYTECODE_ZERO_EVAL
        for k in unroll(1, DIM):
            self_claim_output[BYTECODE_POINT_N_VARS * DIM + k] = 0
    else:
        reduce_self_claims(self_claims, n_self_claims, self_claim_output, self_sumcheck_proof)

    return


def reduce_inner_claims(claims, n_claims, claim_output, sumcheck_proof):
    """Reduce inner bytecode claims via sumcheck (INNER_BPN-dimensional)."""
    claims_hash: Mut = ZERO_VEC_PTR
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        for k in unroll(INNER_BCS, INNER_BCS_PAD):
            assert claim_ptr[k] == 0
        claim_hash = slice_hash(claim_ptr, INNER_BCS_PAD / DIGEST_LEN)
        new_hash = Array(DIGEST_LEN)
        poseidon16_compress(claims_hash, claim_hash, new_hash)
        claims_hash = new_hash

    reduction_fs: Mut = fs_new(sumcheck_proof)
    reduction_fs, received_claims_hash = fs_receive_chunks(reduction_fs, 1)
    copy_8(claims_hash, received_claims_hash)

    reduction_fs, alpha = fs_sample_ef(reduction_fs)
    alpha_powers = powers(alpha, n_claims)

    all_values = Array(n_claims * DIM)
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        copy_5(claim_ptr + INNER_BPN * DIM, all_values + i * DIM)

    claimed_sum = Array(DIM)
    dot_product_ee_dynamic(all_values, alpha_powers, claimed_sum, n_claims)

    reduction_fs, challenges, final_eval = sumcheck_verify(reduction_fs, INNER_BPN, claimed_sum, 2)

    # Verify: final_eval == bytecode(r) * w(r)
    eq_evals = Array(n_claims * DIM)
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        eq_val = eq_mle_extension(claim_ptr, challenges, INNER_BPN)
        copy_5(eq_val, eq_evals + i * DIM)
    w_r = Array(DIM)
    dot_product_ee_dynamic(eq_evals, alpha_powers, w_r, n_claims)

    bytecode_value_at_r = div_extension_ret(final_eval, w_r)

    copy_many_ef(challenges, claim_output, INNER_BPN)
    copy_5(bytecode_value_at_r, claim_output + INNER_BPN * DIM)
    return


def reduce_self_claims(claims, n_claims, claim_output, sumcheck_proof):
    """Reduce self bytecode claims via sumcheck (BYTECODE_POINT_N_VARS-dimensional)."""
    claims_hash: Mut = ZERO_VEC_PTR
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        for k in unroll(BYTECODE_CLAIM_SIZE, BYTECODE_CLAIM_SIZE_PADDED):
            assert claim_ptr[k] == 0
        claim_hash = slice_hash(claim_ptr, BYTECODE_CLAIM_SIZE_PADDED / DIGEST_LEN)
        new_hash = Array(DIGEST_LEN)
        poseidon16_compress(claims_hash, claim_hash, new_hash)
        claims_hash = new_hash

    reduction_fs: Mut = fs_new(sumcheck_proof)
    reduction_fs, received_claims_hash = fs_receive_chunks(reduction_fs, 1)
    copy_8(claims_hash, received_claims_hash)

    reduction_fs, alpha = fs_sample_ef(reduction_fs)
    alpha_powers = powers(alpha, n_claims)

    all_values = Array(n_claims * DIM)
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        copy_5(claim_ptr + BYTECODE_POINT_N_VARS * DIM, all_values + i * DIM)

    claimed_sum = Array(DIM)
    dot_product_ee_dynamic(all_values, alpha_powers, claimed_sum, n_claims)

    reduction_fs, challenges, final_eval = sumcheck_verify(reduction_fs, BYTECODE_POINT_N_VARS, claimed_sum, 2)

    # Verify: final_eval == bytecode(r) * w(r)
    eq_evals = Array(n_claims * DIM)
    for i in range(0, n_claims):
        claim_ptr = claims[i]
        eq_val = eq_mle_extension(claim_ptr, challenges, BYTECODE_POINT_N_VARS)
        copy_5(eq_val, eq_evals + i * DIM)
    w_r = Array(DIM)
    dot_product_ee_dynamic(eq_evals, alpha_powers, w_r, n_claims)

    bytecode_value_at_r = div_extension_ret(final_eval, w_r)

    copy_many_ef(challenges, claim_output, BYTECODE_POINT_N_VARS)
    copy_5(bytecode_value_at_r, claim_output + BYTECODE_POINT_N_VARS * DIM)
    return
