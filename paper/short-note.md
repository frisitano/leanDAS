# STARS: Succinct Transparent Arguments of Reed-Solomon Codes

**Francesco Risitano**
*Version: 2026-05-05*

## Abstract

STARS (Succinct Transparent Arguments of Reed-Solomon codes) is a Reed–Solomon proximity-test-free DAS construction that achieves *exact* code-binding (and hence reconstruction-binding) unconditionally, without trusted setup, and with post-quantum security. Both the random linear combination (RLC) across input codewords and FRI folding are computed inside a ZKVM circuit, certifying every fold step rather than spot-checking. We instantiate STARS as **leanDAS** on Ethereum's leanVM with KoalaBear and Poseidon2, and report benchmarks: on an Apple M2 Max, the sweet-spot configuration $n = 4096$, $m = 240$, half-rate proves in 5.28 s with a 356 KB proof and 364 KB/s message throughput.

---

## 1. The Problem

Ethereum's data availability sampling (DAS) requires commitment schemes for erasure-coded data. Hall-Andersen, Simkin, and Wagner [ePrint 2023/1079] identified three progressively stronger *binding properties* for such schemes:

1. **Position-binding:** no adversary can open a commitment to two different values at the same position
2. **Code-binding:** opened values extend to a valid codeword
3. **Reconstruction-binding:** the committed data is uniquely determined

The central result is that position-binding + code-binding $\Rightarrow$ reconstruction-binding — the property that ultimately guarantees data availability.

Existing DAS constructions fall short of achieving all binding properties unconditionally:

- **KZG-DAS** [PeerDAS] achieves all three binding properties, but requires a trusted setup (structured reference string) and pairing-based assumptions vulnerable to quantum attacks.

- **FRIDA** [ePrint 2024/248] eliminates the trusted setup and is post-quantum. However, FRI is inherently a *proximity* test — the scheme achieves only *proximity-binding* (opened values are close to a codeword), not exact code-binding. Reconstruction-binding holds only approximately.

- **ZODA** [ePrint 2025/034] achieves exact code-binding via a structural modification to the encoding itself: for *tensor codes*, sampled rows and columns of the modified encoding become proofs of their own correctness, with no separate proof object beyond the encoding. The construction is specific to the tensor-code setting; per-sample bandwidth is $\Theta(n + 2m)$ (a full row plus a full column).

## 2. The STARS Construction

STARS (Succinct Transparent Arguments of Reed-Solomon codes) achieves exact code-binding (and hence reconstruction-binding) unconditionally, without trusted setup, and with post-quantum security. The construction is generic over the choice of field, hash function, and ZKVM backend.

The key insight is to run both the RLC accumulation and FRI folding *inside each prover's* ZKVM circuit. Each prover produces a single STARK proof certifying both RLC correctness and exact RS membership of its codewords. Proofs are recursively aggregated, and the verifier receives commitments to all $m$ input codewords and a single aggregated proof.

### Setup

Let $\mathbb{F}$ be a finite field and $\mathbb{E}/\mathbb{F}$ a field extension with $|\mathbb{E}|$ sufficiently large for the Schwartz-Zippel bound. We work with an RS code $\mathrm{RS}[\iota, k]$: polynomials of degree less than $k$, evaluated over a domain $D \subset \mathbb{F}$ of size $n$.

The system receives $m$ input codewords $b_1, \ldots, b_m \in \mathbb{F}^n$, each an evaluation of a polynomial of degree less than $k$.

### Step 1: Commit

Each $b_i$ is committed via an erasure-code commitment scheme, yielding commitments $\mathrm{com}_1, \ldots, \mathrm{com}_m$.

### Step 2: Challenge Derivation

The RLC challenge $r_i \in \mathbb{E}$ is derived from the commitment: $r_i = H(\mathrm{com}_i)$. Because each $r_i$ depends only on its own commitment, provers can pipeline with blob arrival. The combined codeword is:

$$c^\star[j] = \sum_{i=1}^{m} r_i \cdot b_i[j]$$

where each $b_i$ is viewed in $\mathbb{E}$ via the canonical embedding $\mathbb{F} \hookrightarrow \mathbb{E}$. If every $b_i$ is a valid RS codeword, then $c^\star$ is also a valid RS codeword (since RS is a linear subspace).

### Step 3: Proving (RLC + FRI, Parallelizable)

The $m$ input codewords are divided into $B$ batches. Each prover independently performs both RLC and FRI folding in a single ZKVM run:

1. Takes its assigned codewords and challenges
2. Computes the RLC: $c_k^\star = \sum_{j} r_{i_j} \cdot b_{i_j}$
3. Performs FRI folding on $c_k^\star$ inside the ZKVM — $\log_2(k)$ rounds of degree halving, operating entirely on evaluation vectors (no polynomial interpolation or evaluation). The even/odd parts are extracted by pairing evaluations at $\omega$ and $-\omega$, and folded via a linear combination.
4. Produces a single STARK proof certifying both RLC correctness and **exact RS membership** of $c_k^\star$

All $B$ provers run in parallel. A prover can start as soon as all blobs in its batch are committed — without waiting for other batches.

Because the ZKVM verifies *all* folding steps (not a probabilistic spot-check), this gives **exact RS membership** — not the proximity guarantee of standalone FRI.

### Step 4: Proof Aggregation

The proofs from Step 3 are recursively aggregated into a single proof. Each aggregation step verifies two child proofs inside a ZKVM and produces a new proof:

- **Layer 0:** $B$ independent proofs (from Step 3), each certifying RS membership
- **Layer $\ell$:** Each node verifies two proofs from layer $\ell - 1$
- **Root:** A single proof certifying all $m$ input codewords are valid RS codewords

Aggregation depth is $O(\log B)$, and all nodes at the same layer are independent.

### Anatomy of Proofs and Aggregated Proofs

**Proof.** Each prover's ZKVM circuit performs two stages:

1. **RLC:** Each codeword (a row of evaluations) is scaled by its challenge $r_i = H(\mathrm{com}_i)$ and summed element-wise into the combined codeword $c_k^\star$.
2. **FRI Folding:** The combined evaluation vector is repeatedly halved — each round splits into even/odd parts and folds — until a single constant remains. If the result is constant, $c_k^\star$ has degree less than $d$ and is a valid RS codeword. All folding is field arithmetic on evaluation vectors; no polynomial interpolation or evaluation is performed.

The STARK proof $\pi_k$ has the commitments to all codewords in batch $k$ as public inputs, and certifies both RLC correctness and exact RS membership.

**Aggregated proof.** Each aggregation circuit takes two child proofs ($\pi_L$, $\pi_R$) as private witness and their public inputs (the child commitments) as its own public inputs. The circuit verifies both child proofs; if both verify, the aggregator produces a new proof whose public inputs are the union of the children's commitments. At the root, the final proof $\pi^\star$ has all $m$ commitments $\mathrm{com}_1, \ldots, \mathrm{com}_m$ as public inputs, certifying that every committed codeword is a valid RS codeword.

### Step 5: DAS Verification

The DAS verifier receives:
- The commitments $\mathrm{com}_1, \ldots, \mathrm{com}_m$ (to all $m$ input codewords)
- A single STARK proof $\pi$ certifying that the RLC of the committed codewords is a valid RS codeword

Verification:
1. **Code membership:** Check $\pi$ — this certifies that *every* committed codeword $b_i$ is a valid RS codeword (via Schwartz-Zippel, see below).
2. **Symbol openings:** For each queried position $j$ of input $b_i$, verify the commitment opening against $\mathrm{com}_i$.

If both checks pass, the verifier is guaranteed that the opened symbols are consistent with valid codewords that can be reconstructed from any sufficiently large subset of positions.

## 3. Why It's Sound

**Schwartz-Zippel for RLC.** Each prover computes an RLC $c_k^\star$ and proves it is a valid RS codeword. Let $m_k$ denote the number of codewords in batch $k$. If any $b_i$ in batch $k$ is not in the RS code, then $c_k^\star$ is not RS with probability at least $1 - m_k/|\mathbb{E}|$. A union bound over all $B$ batches gives an overall failure probability of at most $m/|\mathbb{E}|$.

**Binding hierarchy.** We frame security through the binding properties of [ePrint 2023/1079]:

- **Position-binding** follows from the commitment scheme. No adversary can open a commitment to two different values at the same position.

- **Code-binding** follows from exact RS membership of each RLC. Because FRI folding runs inside each prover's ZKVM circuit (checking all folding steps, not spot-checks), the verifier obtains exact membership — not proximity.

- **Reconstruction-binding** follows from position-binding + code-binding, by [ePrint 2023/1079, Theorem 1].

**DAS soundness.** If the DAS verifier accepts, then: (a) each RLC is a valid codeword over $\mathbb{E}$ (and hence all $b_i$ are valid with overwhelming probability), (b) any two queries at the same position agree (consistency), and (c) each committed vector's opened values extend to a valid codeword over $\mathbb{F}$.

## 4. Comparison

| Property | STARS | KZG-DAS | FRIDA | ZODA |
|---|---|---|---|---|
| Position-binding | Yes | Yes | Yes | Yes |
| Code-binding | Exact | Exact | Proximity | Exact (tensor) |
| Reconstruction-binding | Yes | Yes | Approximate | Yes (tensor) |
| Post-quantum | Yes | No | Yes | Yes |
| Trusted setup | No | Yes (SRS) | No | No |

The key tradeoff is proof generation cost: STARS requires ZKVM proof generation for FRI folding and proof aggregation. The parallel architecture mitigates this: with $B$ parallel provers, the dominant cost is $O(\log B)$ aggregation steps rather than $O(m)$ sequential proofs.

## 5. leanDAS: Concrete Instantiation

**leanDAS** is a concrete instantiation of the STARS construction, targeting Ethereum's leanVM.

### Cryptographic Parameters

- **Field:** KoalaBear ($p = 2^{31} - 2^{24} + 1$) with degree-5 extension $\mathbb{E} = \mathrm{GF}(p^5)$, giving $|\mathbb{E}| \approx 2^{155}$ and RLC soundness $m/|\mathbb{E}| \approx m \cdot 2^{-155}$.
- **Hash:** Poseidon2 with compression mode (Merkle nodes) and sponge mode (leaves). Digest size $\eta = 8$ base-field elements (248 bits). Challenge derivation: $r_i = \mathrm{embed}_5(\mathrm{root}_i)$ — interpret 5 of 8 digest elements as one $\mathbb{E}$-element. Since $\eta \geq 5$, no additional hash is needed.
- **Commitment:** Poseidon2 Merkle trees with configurable symbols per leaf (EPL). Position-binding from collision resistance.
- **ZKVM:** Ethereum's leanVM — a STARK-based ZKVM with precompiled operations for extension-field arithmetic (`add_ee`, `dot_product_be`, `dot_product_ee`) and Poseidon2 (`poseidon16_compress`).

### Circuit Design

The leanDAS circuit performs all computation using leanVM's precompiled operations, which are evaluated in dedicated precompile traces with simpler constraints than the main execution trace. The circuit has three phases:

1. **Commitment verification:** Hash all batch commitments into a chain hash (public input) and derive RLC challenges from commitment digests.
2. **RLC accumulation:** For each codeword position $j$, compute $c^\star[j] = \sum_i r_i \cdot b_i[j]$ via a single `dot_product_be` call over column-major data.
3. **FRI folding:** $\log_2(d)$ rounds of scalar butterfly operations using `add_ee`, `dot_product_be`, and `dot_product_ee` for twiddle/beta multiplication.

The private witness contains three regions: (1) commitments, (2) row-major codeword evaluations (for Merkle verification), and (3) column-major codeword evaluations (for efficient RLC dot products).

### Trace Constraints

leanVM's execution model has a key constraint: each precompile trace table must not exceed $2^{20}$ rows (`MAX_LOG_N_ROWS_PER_TABLE`), and the execution trace must be the tallest table. The dominant precompile cost is the extension-field operation table, populated by RLC dot products and FRI butterfly operations. This constrains the maximum single-batch size.

## 6. Benchmarks

All benchmarks run on an Apple M2 Max. Proof generation uses leanVM's STARK prover with Poseidon2 commitments. Degree $d$ = half-rate ($n/2$) unless noted.

### Codeword Length Scaling (single codeword)

| $n$ | Degree | Compile | Prove | Verify | Proof (FE) | Instructions |
|-----|--------|---------|-------|--------|------------|-------------|
| 16 | 8 | ~1 ms | ~120 ms | ~50 ms | ~200 | ~300 |
| 64 | 32 | ~3 ms | ~250 ms | ~60 ms | ~400 | ~2K |
| 256 | 128 | ~10 ms | ~650 ms | ~70 ms | ~600 | ~10K |
| 1024 | 512 | ~40 ms | ~1.8 s | ~80 ms | ~800 | ~50K |
| 4096 | 2048 | ~160 ms | ~6 s | ~100 ms | ~1.2K | ~200K |

### Batch Scaling (N=4096, half-rate)

| $m$ (batch) | Prove (s) | Message throughput | Proof size | Exec trace | Ext trace |
|-------------|-----------|-------------------|------------|------------|-----------|
| 40 | 2.82 | 227 KB/s | 344 KB | $2^{19}$ | $2^{18}$ |
| 80 | 3.80 | 337 KB/s | 350 KB | $2^{19}$ | $2^{19}$ |
| 160 | 5.75 | 223 KB/s | 350 KB | $2^{20}$ | $2^{19}$ |
| 240 | 5.28 | 364 KB/s | 356 KB | $2^{20}$ | $2^{19}$ |

> Message throughput = $m \times n/2 \times 4$ bytes / prove time (systematic part only).

### Sweet Spot: N=4096, m=240

At $m = 240$ codewords of length 4096 with half-rate encoding ($d = 2048$):

- **Prove time:** 5.28 s
- **Message throughput:** 364 KB/s (728 KB/s raw codeword data)
- **Proof size:** 356 KB (~1.2K field elements)
- **Trace utilization:** Extension-op trace at 92% of $2^{19}$ capacity; execution trace at $2^{20}$

The maximum single-batch size at $N = 4096$ is approximately $m = 240$, limited by the $2^{20}$ extension-op trace ceiling. Beyond this, codewords must be split across multiple batches.

### Larger Codewords (N=8192)

| $m$ (batch) | Prove (s) | Message throughput | Proof size |
|-------------|-----------|-------------------|------------|
| 40 | 2.82 | 227 KB/s | 344 KB |
| 80 | 6.25 | 205 KB/s | 348 KB |

## 7. Parallelism and GPU Projections

### CPU Parallelism

With $P$ parallel provers, each handling a batch of $m/P$ codewords:

- **Phase 1 (proving):** Wall-clock time = single-batch prove time (all provers in parallel)
- **Phase 2 (aggregation):** $O(\log P)$ sequential aggregation steps

For Ethereum's target of $m = 128$ blobs at $n = 4096$: a single batch of 128 codewords proves in ~4 s on one core. With 4 parallel provers (32 codewords each), Phase 1 drops to ~2 s.

### GPU Acceleration

> **Note:** The numbers in this section are speculative projections, not measured. They extrapolate from CPU benchmarks using a conservative 15× speedup factor; actual GPU performance has not been benchmarked.

GPU-accelerated STARK provers (NTT, Poseidon2 hashing, FRI commitment) typically achieve 10-20x speedup over single-core CPU. Conservative estimate: **15x**.

| Configuration | CPU (1 core) | GPU (est. 15x) |
|---------------|-------------|-----------------|
| N=4096, m=240 | 5.28 s | ~350 ms |
| N=4096, m=80 | 3.80 s | ~250 ms |
| N=8192, m=80 | 6.25 s | ~420 ms |

At 350 ms per batch with m=240 on GPU, a single card achieves approximately **5.5 MB/s** message throughput. With 8 concurrent proofs per GPU (memory permitting), throughput scales to ~44 MB/s per card.

### Path to 1 GB/s

For Ethereum-scale throughput (1 GB/s), approximately **12-23 H100 GPUs** in parallel, each running 8 concurrent proofs. The two-phase architecture makes this embarrassingly parallel: batches are independent, and aggregation is $O(\log B)$.

---

## References

1. Hall-Andersen, M., Simkin, M., Wagner, B. (2023). *Foundations of Data Availability Sampling.* IACR ePrint [2023/1079](https://eprint.iacr.org/2023/1079). Establishes the position-binding / code-binding / reconstruction-binding hierarchy used throughout this note.

2. Hall-Andersen, M., Simkin, M., Wagner, B. (2024). *FRIDA: Data Availability Sampling from FRI.* IACR ePrint [2024/248](https://eprint.iacr.org/2024/248). FRI-based DAS achieving proximity-binding only.

3. Evans, A., Mohnblatt, N., Angeris, G. (Bain Capital Crypto, 2025). *ZODA: Zero-Overhead Data Availability.* IACR ePrint [2025/034](https://eprint.iacr.org/2025/034). Conditional code-binding via fraud proofs.

4. Ben-Sasson, E., Goldberg, L., Kopparty, S., Saraf, S. (2019). *DEEP-FRI: Sampling Outside the Box Improves Soundness.* IACR ePrint [2019/336](https://eprint.iacr.org/2019/336). FRI soundness analysis underpinning the proximity-test security bound.

