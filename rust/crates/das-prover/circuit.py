from snark_lib import *

NONRESERVED_PROGRAM_INPUT_START = __NRPIS__
BATCH_SIZE = __BATCH_SIZE__
CODEWORD_LEN = __CODEWORD_LEN__
DIM = __DIM__
DIGEST_LEN = __DIGEST_LEN__
EVALS_PER_LEAF = __EPL__
NUM_LEAVES = __NUM_LEAVES__
TREE_HEIGHT = __TREE_HEIGHT__
NUM_FOLD_ROUNDS = __NUM_FOLD_ROUNDS__
N_CHUNKS = __N_CHUNKS__
HALF = __HALF__
OMEGA_INV_ROUNDS = __OMEGA_INV_ROUNDS__


def hash_leaf(cw_base, leaf_off, dest):
    """Hash a leaf (EPL base-field elements) via sequential poseidon16_compress."""
    if N_CHUNKS == 2:
        poseidon16_compress(cw_base + leaf_off, cw_base + leaf_off + DIGEST_LEN, dest)
    else:
        # Chain: compress(chunk_0, chunk_1) -> temp -> compress(temp, chunk_2) -> ... -> dest
        temp: Mut = Array(DIGEST_LEN)
        poseidon16_compress(cw_base + leaf_off, cw_base + leaf_off + DIGEST_LEN, temp)
        for i in unroll(2, N_CHUNKS - 1):
            next_temp = Array(DIGEST_LEN)
            poseidon16_compress(temp, cw_base + leaf_off + i * DIGEST_LEN, next_temp)
            temp = next_temp
        poseidon16_compress(temp, cw_base + leaf_off + (N_CHUNKS - 1) * DIGEST_LEN, dest)
    return


def build_merkle_tree(leaf_hashes):
    prev: Mut = leaf_hashes
    for level in unroll(0, TREE_HEIGHT):
        level_size = NUM_LEAVES / (2 ** (level + 1))
        next_level = Array(level_size * DIGEST_LEN)
        for k in range(0, level_size):
            poseidon16_compress(prev + k * 2 * DIGEST_LEN, prev + (k * 2 + 1) * DIGEST_LEN, next_level + k * DIGEST_LEN)
        prev = next_level
    return prev


def main():
    pub_mem = NONRESERVED_PROGRAM_INPUT_START

    priv_start: Imu
    hint_private_input_start(priv_start)

    # Private input layout:
    #   Region 0: [0 .. B*DIGEST_LEN]         commitments (B = BATCH_SIZE)
    #   Region 1: [.. + B*N]                   row-major BF codewords (for Merkle)
    #   Region 2: [.. + N*B]                   column-major BF codewords (for RLC dot_product)
    #
    # TODO: with a strided dot_product_be, we could use row-major only and
    # eliminate Region 2 — reading RLC inputs at stride N from row-major.
    # This would halve the private input and remove the dual-layout.
    com_base = priv_start
    cw_base = com_base + BATCH_SIZE * DIGEST_LEN
    col_base = cw_base + BATCH_SIZE * CODEWORD_LEN

    # Hash commitments and verify against public input.
    batch_hash: Mut = ZERO_VEC_PTR
    for i in unroll(0, BATCH_SIZE):
        next_hash = Array(DIGEST_LEN)
        poseidon16_compress(batch_hash, com_base + i * DIGEST_LEN, next_hash)
        batch_hash = next_hash
    for d in unroll(0, DIGEST_LEN):
        assert batch_hash[d] == pub_mem[d]

    # Derive alphas from commitments (alpha_i = com_i[0..DIM]).
    alphas = Array(BATCH_SIZE * DIM)
    for i in unroll(0, BATCH_SIZE):
        for d in unroll(0, DIM):
            alphas[i * DIM + d] = com_base[i * DIGEST_LEN + d]

    # Merkle check: verify each codeword hashes to its commitment.
    for i in unroll(0, BATCH_SIZE):
        cw_i = cw_base + i * CODEWORD_LEN
        leaves_i = Array(NUM_LEAVES * DIGEST_LEN)
        for j in range(0, NUM_LEAVES):
            hash_leaf(cw_i, j * EVALS_PER_LEAF, leaves_i + j * DIGEST_LEN)
        root_i = build_merkle_tree(leaves_i)
        com_i = com_base + i * DIGEST_LEN
        for d in unroll(0, DIGEST_LEN):
            assert root_i[d] == com_i[d]

    # ── RLC via column-major dot_product_be ────────────────────────────────
    # For each position j: rlc[j] = sum_i alpha_i * cw_i[j]
    # col_base + j*B points to [cw_0[j], cw_1[j], ..., cw_{B-1}[j]] (BF, stride 1)
    # alphas is [alpha_0, ..., alpha_{B-1}] (EF, stride DIM)
    rlc = Array(CODEWORD_LEN * DIM)
    for j in range(0, CODEWORD_LEN):
        dot_product_be(col_base + j * BATCH_SIZE, alphas, rlc + j * DIM, BATCH_SIZE)

    # ── FRI folding (scalar butterfly) ─────────────────────────────────────
    #
    # Derive folding challenge beta from batch_hash (first DIM elements = EF).
    beta = Array(DIM)
    for d in unroll(0, DIM):
        beta[d] = batch_hash[d]

    # Pre-allocate half scalar.
    half_s = Array(1)
    half_s[0] = HALF

    current: Mut = rlc
    for round in unroll(0, NUM_FOLD_ROUNDS):
        half_n = CODEWORD_LEN / (2 ** (round + 1))
        omega_inv_round = OMEGA_INV_ROUNDS[round]

        # Precompute twiddle factors: inv_tw_arr[i] = half * omega_inv_round^i
        inv_tw_arr = Array(half_n)
        inv_tw: Mut = HALF
        for i in range(0, half_n):
            inv_tw_arr[i] = inv_tw
            inv_tw = inv_tw * omega_inv_round

        # Scalar butterfly for each position
        next_layer = Array(half_n * DIM)
        for i in range(0, half_n):
            left = current + i * DIM
            right = current + (half_n + i) * DIM

            # p_sum = left + right (EF + EF)
            p_sum = Array(DIM)
            add_ee(left, right, p_sum)

            # diff = left - right (constraint: right + diff = left)
            diff = Array(DIM)
            add_ee(right, diff, left)

            # tw_diff = inv_tw_arr[i] * diff (BF x EF, length=1)
            tw_diff = Array(DIM)
            dot_product_be(inv_tw_arr + i, diff, tw_diff)

            # p_even = half * p_sum (BF x EF, length=1)
            p_even = Array(DIM)
            dot_product_be(half_s, p_sum, p_even)

            # bp = beta * tw_diff (EF x EF, length=1)
            bp = Array(DIM)
            dot_product_ee(beta, tw_diff, bp)

            # result[i] = p_even + bp
            add_ee(p_even, bp, next_layer + i * DIM)

        current = next_layer

    # Check constancy: all folded values must be equal.
    num_outputs = CODEWORD_LEN / (2 ** NUM_FOLD_ROUNDS)
    for k in range(1, num_outputs):
        for d in range(0, DIM):
            assert current[d] == current[k * DIM + d]

    return
