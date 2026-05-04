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
NUM_SPOT_CHECKS = __NUM_SPOT_CHECKS__
LOG_CODEWORD_LEN = __LOG_CODEWORD_LEN__
POSITIONS_PER_ELEMENT = __POSITIONS_PER_ELEMENT__
NUM_RAND_ELEMENTS = __NUM_RAND_ELEMENTS__
NUM_FULL_RAND_ELEMENTS = __NUM_FULL_RAND_ELEMENTS__
REMAINDER_POSITIONS = __REMAINDER_POSITIONS__
NUM_HASH_ROUNDS = __NUM_HASH_ROUNDS__
N_RLC_CHUNKS = __N_RLC_CHUNKS__
LITTLE_ENDIAN = 1


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
    #   Region 2: [.. + N*B]                   column-major BF codewords (for spot-check)
    #   Region 3: [.. + N*DIM]                 RLC hint values (N extension field elements)
    com_base = priv_start
    cw_base = com_base + BATCH_SIZE * DIGEST_LEN
    col_base = cw_base + BATCH_SIZE * CODEWORD_LEN
    rlc = col_base + CODEWORD_LEN * BATCH_SIZE

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

    # ── RLC spot-check ────────────────────────────────────────────────────
    # Chain-hash the RLC hint for Fiat-Shamir binding.
    # hash = compress(compress(...compress(rlc[0..8], rlc[8..16])...), rlc[last_chunk])
    rlc_hash: Mut = Array(DIGEST_LEN)
    poseidon16_compress(rlc, rlc + DIGEST_LEN, rlc_hash)
    for ch in range(2, N_RLC_CHUNKS):
        next_rlc_hash = Array(DIGEST_LEN)
        poseidon16_compress(rlc_hash, rlc + ch * DIGEST_LEN, next_rlc_hash)
        rlc_hash = next_rlc_hash

    # Derive spot-check seed from batch_hash and rlc_hash.
    sc_seed: Mut = Array(DIGEST_LEN)
    poseidon16_compress(batch_hash, rlc_hash, sc_seed)

    # Generate random field elements via hash chain.
    # Round 0: use sc_seed directly. Round r>0: poseidon(sc_seed, [r,0,...]).
    rand_elems = Array(NUM_HASH_ROUNDS * DIGEST_LEN)
    for d in unroll(0, DIGEST_LEN):
        rand_elems[d] = sc_seed[d]
    for r in unroll(1, NUM_HASH_ROUNDS):
        counter = Array(DIGEST_LEN)
        counter[0] = r
        for ci in unroll(1, DIGEST_LEN):
            counter[ci] = 0
        round_out = Array(DIGEST_LEN)
        poseidon16_compress(sc_seed, counter, round_out)
        for d in unroll(0, DIGEST_LEN):
            rand_elems[r * DIGEST_LEN + d] = round_out[d]

    # Decompose needed random elements into bits.
    all_bits = Array(NUM_RAND_ELEMENTS * 31)
    for e in unroll(0, NUM_RAND_ELEMENTS):
        hint_decompose_bits(rand_elems + e, all_bits + e * 31, 31, LITTLE_ENDIAN)

    # Reconstruct position values from bit slices.
    # Each element yields POSITIONS_PER_ELEMENT positions (within element boundaries).
    # NUM_FULL_RAND_ELEMENTS full elements + REMAINDER_POSITIONS from the last.
    # Use Mut accumulator pattern (write-once memory: can't update in place).
    positions = Array(NUM_SPOT_CHECKS)
    for e in unroll(0, NUM_FULL_RAND_ELEMENTS):
        for c in unroll(0, POSITIONS_PER_ELEMENT):
            bit_start = e * 31 + c * LOG_CODEWORD_LEN
            acc: Mut = 0
            for b in unroll(0, LOG_CODEWORD_LEN):
                acc = acc + all_bits[bit_start + b] * (2 ** b)
            positions[e * POSITIONS_PER_ELEMENT + c] = acc
    for c in unroll(0, REMAINDER_POSITIONS):
        bit_start = NUM_FULL_RAND_ELEMENTS * 31 + c * LOG_CODEWORD_LEN
        acc: Mut = 0
        for b in unroll(0, LOG_CODEWORD_LEN):
            acc = acc + all_bits[bit_start + b] * (2 ** b)
        positions[NUM_FULL_RAND_ELEMENTS * POSITIONS_PER_ELEMENT + c] = acc

    # Verify rlc[z_t] = sum_i alpha_i * cw_i[z_t] at each spot-check position.
    for t in range(0, NUM_SPOT_CHECKS):
        z_t = positions[t]
        expected = Array(DIM)
        dot_product_be(col_base + z_t * BATCH_SIZE, alphas, expected, BATCH_SIZE)
        # Constrain expected == rlc[z_t]: add_ee(a, 0, c) verifies a + 0 = c.
        add_ee(expected, ZERO_VEC_PTR, rlc + z_t * DIM)

    # ── FRI folding (scalar butterfly) ─────────────────────────────────────
    beta = Array(DIM)
    for d in unroll(0, DIM):
        beta[d] = batch_hash[d]

    half_s = Array(1)
    half_s[0] = HALF

    current: Mut = rlc
    for round in unroll(0, NUM_FOLD_ROUNDS):
        half_n = CODEWORD_LEN / (2 ** (round + 1))
        omega_inv_round = OMEGA_INV_ROUNDS[round]

        inv_tw_arr = Array(half_n)
        inv_tw: Mut = HALF
        for i in range(0, half_n):
            inv_tw_arr[i] = inv_tw
            inv_tw = inv_tw * omega_inv_round

        next_layer = Array(half_n * DIM)
        for i in range(0, half_n):
            left = current + i * DIM
            right = current + (half_n + i) * DIM

            p_sum = Array(DIM)
            add_ee(left, right, p_sum)

            diff = Array(DIM)
            add_ee(right, diff, left)

            tw_diff = Array(DIM)
            dot_product_be(inv_tw_arr + i, diff, tw_diff)

            p_even = Array(DIM)
            dot_product_be(half_s, p_sum, p_even)

            bp = Array(DIM)
            dot_product_ee(beta, tw_diff, bp)

            add_ee(p_even, bp, next_layer + i * DIM)

        current = next_layer

    # Check constancy: all folded values must be equal.
    num_outputs = CODEWORD_LEN / (2 ** NUM_FOLD_ROUNDS)
    for k in range(1, num_outputs):
        for d in range(0, DIM):
            assert current[d] == current[k * DIM + d]

    return
