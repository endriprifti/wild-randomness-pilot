"""P6 (added 2026-11-17, fixed before any P6 run): the synthetic token streams of the factorial Noah-Joseph test,
the checks every stream must pass before a model is trained on it, and the loss floors.

numpy and scipy only (no torch): p6_noah_joseph.py trains and measures on these streams, tests/test_p6.py checks
them, and `python p6_streams.py --calibrate` repeats the calibration of the checks (reference/p6_stream_checks*.csv).

Token = (rank - 1) * K + variant: K = 8 topics, R = 6,288 ranks, vocabulary R * K = 50,304 (Pythia's).
  rank     i.i.d. over tokens from the Noah level's rank law: Zipf p_r ~ r^-s (s = 1.0, or 1.25 in the full design),
           or the light-tailed control p_r ~ exp(-(r - 1)/tau) with tau = 205.87, set so that its entropy equals
           Zipf 1.0's (6.327 nats): the corners differ in the tail of the rank law, not in its entropy
  variant  the current topic with probability q = 0.75, else uniform over the K variants: the topic is read from
           the context (1.13 nats per token if known)
  topic    a switch moves it to one of the other K - 1 topics (uniform). Switches are Bernoulli per token with a
           hazard that is constant over blocks of 64 tokens: p_b = p_bar exp(b g_b) / mean_b exp(b g_b), g = fGn(H)
           over blocks (the Joseph level; H = 0.5 is i.i.d. hazards). Dividing by the stream's own mean fixes each
           stream's switch rate at p_bar = 1/128 per token: H changes only how switches cluster (with H = 0.9 the
           mean of a long-memory path over 4 million blocks still varies by about 20% between seeds, the Joseph
           effect itself, which would otherwise change the loss floor from seed to seed)
Common random numbers: a stream's random numbers depend on (seed, role) only, never on the cell. For one seed every
cell uses the same uniforms for ranks, variants and switches, the same white noise under its fGn, the same window
order and the same initial weights; the cells differ only in the rank law's inverse CDF and in the fGn's H. Seeds
are blocks in the analysis (p6_analysis.py).

Roles: "train" (n_windows = steps x sequences per step, each window used once, in a fixed random order) and
"heldout" (16,384 windows; disjoint ranges give the H7.1 / coupling pool, the eval, H_tok and probe sets).

Stream checks (check_stream; a stream that fails is not trained on, the run stops with the reason):
  rank law  discrete MLE of the Zipf exponent over ranks 1..R and over the tail 100..R (|s_hat - s| <= 0.01 both),
            or of the light control's tau (|tau_hat / tau - 1| <= 0.01), and the right family wins the likelihood
            by >= 0.05 nats per token
  variants  share equal to the topic within 0.002 of q + (1 - q)/K
  switches  rate within 1% of p_bar
  H         of the driving fGn by DFA (scales 10 .. n/10) and wavelet (octaves 3 ..): each within 0.04 of H; of
            the switch counts per block (the stream's own topic changes) by DFA over 100 .. n/10 blocks: within
            the band calibrated for that H and length (reference/p6_stream_checks.csv: mean +- max(0.03, 5 sd))
Calibration (2026-11-18, 50 latent processes per H and length, seeds 1000+): fGn H by DFA within 0.016 (train,
4.2 million blocks) and 0.023 (held-out, 0.5 million) of H, by wavelet within 0.009; switch rate within 0.25% and
0.53% of p_bar; switch-count H (biased towards 0.5 by the Poisson noise of the counts) 0.499 +- 0.008, 0.659 +-
0.009, 0.861 +- 0.011 at H = 0.5, 0.7, 0.9 (train; held-out 0.498, 0.645, 0.845, sd 0.011-0.013): the bands of the
three levels do not overlap

Streams of more than LEAN_FROM_BLOCKS hazard blocks (the raised design of the feasibility rule, 2^33 tokens: 2^27
blocks; added 2026-11-18) are built and checked lean (fgn_lean, dfa_lean): the same numbers in a sixth of the memory,
and their switch-count band must be calibrated at their own length. Calibration at 2^27 blocks (2026-11-18, 50 latent
processes per level at H = 0.5 and 0.9, the small design's levels): switch-count H 0.4993 +- 0.0045 and 0.8749 +-
0.0062, fGn H by DFA within 0.011 and by wavelet within 0.005 of H, switch rate within 0.031% of p_bar.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy import optimize

import config
from common import hurst
from common.units import rng_for, seed_for

ROOT = Path(__file__).resolve().parent

P6 = {
    "vocab": 50304, "K": 8, "R": 6288,          # R * K = vocab
    "q": 0.75,                 # OURS: P(variant = topic)
    "block": 64,               # OURS: tokens per hazard block
    "p_bar": 1 / 128,          # OURS: switches per token (topic segments of 128 tokens on average)
    "b": 1.0,                  # OURS: scale of the log hazard (sd of p_b about 1.3 p_bar)
    "p_max": 0.5,              # cap of the hazard per token (never reached at b = 1)
    "seq_len": 2048,
    # Noah levels: ordinal score 0 (light) < 1 < 2 (heaviest tail); the corners are entropy-matched
    "noah": {"light": {"kind": "exp", "entropy_of_zipf": 1.0, "score": 0},
             "zipf1.25": {"kind": "zipf", "s": 1.25, "score": 1},
             "zipf1.0": {"kind": "zipf", "s": 1.0, "score": 2}},
    # Joseph levels: H of the fGn under the switching hazard; ordinal score 0 < 1 < 2
    "joseph": {"H0.5": {"H": 0.5, "score": 0}, "H0.7": {"H": 0.7, "score": 1}, "H0.9": {"H": 0.9, "score": 2}},
    "corners": {"noah": ["light", "zipf1.0"], "joseph": ["H0.5", "H0.9"]},
    "heldout_windows": 16384,  # 2^25 tokens
    # stream checks (module docstring)
    "check": {"s_tol": 0.01, "tau_rel_tol": 0.01, "tail_from": 100, "ll_margin": 0.05, "variant_tol": 0.002,
              "rate_rel_tol": 0.01, "g_tol": 0.04, "count_s_min": 100, "count_band_min": 0.03, "count_band_sd": 5.0},
}
CHECK_REF = ROOT / "reference" / "p6_stream_checks.csv"


# ----------------------------------------------------------------------------- the rank laws
def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def zipf_law(s: float, R: int) -> np.ndarray:
    p = np.arange(1, R + 1, dtype=float) ** -float(s)
    return p / p.sum()


def exp_law(tau: float, R: int) -> np.ndarray:
    p = np.exp(-np.arange(R, dtype=float) / float(tau))
    return p / p.sum()


def light_tau(s_match: float, R: int) -> float:
    """tau of the exponential rank law whose entropy equals Zipf(s_match)'s over ranks 1..R."""
    target = entropy(zipf_law(s_match, R))
    return float(optimize.brentq(lambda t: entropy(exp_law(t, R)) - target, 2.0, 50.0 * R, xtol=1e-10))


def rank_law(noah: str, P: dict = P6) -> np.ndarray:
    lv = P["noah"][noah]
    if lv["kind"] == "zipf":
        return zipf_law(lv["s"], P["R"])
    return exp_law(light_tau(lv["entropy_of_zipf"], P["R"]), P["R"])


def law_params(noah: str, P: dict = P6) -> dict:
    lv = P["noah"][noah]
    if lv["kind"] == "zipf":
        return {"kind": "zipf", "s": float(lv["s"])}
    return {"kind": "exp", "tau": light_tau(lv["entropy_of_zipf"], P["R"])}


def floors(noah: str, P: dict = P6) -> dict:
    """Loss floors in nats per token: the rank entropy plus the variant's entropy given the topic (oracle), or plus
    ln K (a model that ignores the topic). A model at the oracle floor has learnt both the rank law and the topic."""
    q, K = P["q"], P["K"]
    p_in = q + (1 - q) / K
    p_out = (1 - q) / K
    h_var = float(-(p_in * math.log(p_in) + (K - 1) * p_out * math.log(p_out)))
    h_rank = entropy(rank_law(noah, P))
    return {"rank_entropy": h_rank, "variant_given_topic": h_var, "oracle_floor": h_rank + h_var,
            "topic_blind_floor": h_rank + math.log(K), "topic_information": math.log(K) - h_var}


# ----------------------------------------------------------------------------- long fGn
def fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Exact fGn by circulant embedding (Davies and Harte 1987), as common/hurst.fgn, with the autocovariance
    computed without cancellation: hurst.fgn's (k+1)^2H - 2k^2H + (k-1)^2H loses its precision for k of millions at
    H = 0.9 (negative circulant eigenvalues at n = 4 million). Same random draws as hurst.fgn."""
    k = np.arange(n + 1, dtype=float)
    a = 2.0 * H
    gam = np.empty(n + 1)
    gam[0] = 1.0
    kk = k[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        gam[1:] = 0.5 * kk ** a * (np.expm1(a * np.log1p(1.0 / kk)) + np.expm1(a * np.log1p(-1.0 / kk)))
    gam[1] = 0.5 * (2.0 ** a - 2.0)           # k = 1: (k-1)^2H = 0 (log1p(-1) = -inf)
    c = np.concatenate([gam, gam[-2:0:-1]])
    lam = np.fft.fft(c).real
    if lam.min() < -1e-8 * max(1.0, lam.max()):
        raise ValueError(f"circulant embedding is not nonnegative definite (min eigenvalue {lam.min():.3g})")
    lam = np.clip(lam, 0, None)
    m = len(c)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    y = np.fft.fft(np.sqrt(lam / m) * w)
    return y[:n].real


# ----------------------------------------------------------------------------- the same, in less memory
# Added 2026-11-18 for the raised design of the feasibility rule (2^33 tokens per run: 2^27 hazard blocks), before
# any P6 run. fgn and the checks above need about 200 bytes per block (26 GiB at 2^27 blocks); the functions below
# give the same numbers (to float rounding, tests/test_p6.py) in about 32 bytes per block during the fGn and 24 during
# the checks. Streams of more than LEAN_FROM_BLOCKS blocks use them; the design's streams (at most 4.2 million
# blocks) never do, so nothing changes for them.
LEAN_FROM_BLOCKS = 1 << 25
LEAN_BLOCK = 1 << 21          # elements per block of work (32 MiB complex)


def _split(n: int) -> tuple[int, int]:
    """n = n1 * n2, n1 the largest divisor of n not above sqrt(n)."""
    n1 = math.isqrt(n)
    while n % n1:
        n1 -= 1
    return n1, n // n1


def _fft_inplace(z: np.ndarray) -> tuple[int, int]:
    """Complex DFT of z (length n) in place by the four-step method (Bailey 1990): DFTs of the columns of z as an
    n2 x n1 matrix, twiddles, DFTs of the rows, in blocks of LEAN_BLOCK elements. Z[k] is left at
    z[(k % n2) * n1 + k // n2]; returns (n1, n2)."""
    n = len(z)
    n1, n2 = _split(n)
    M = z.reshape(n2, n1)
    k2 = np.arange(n2, dtype=np.int64)[:, None]
    b = max(1, LEAN_BLOCK // n2)
    for c0 in range(0, n1, b):
        c1 = min(n1, c0 + b)
        blk = np.fft.fft(M[:, c0:c1], axis=0)
        ang = ((k2 * np.arange(c0, c1, dtype=np.int64)[None, :]) % n).astype(float) * (-2.0 * math.pi / n)
        blk *= np.cos(ang) + 1j * np.sin(ang)
        M[:, c0:c1] = blk
    r = max(1, LEAN_BLOCK // n1)
    for r0 in range(0, n2, r):
        M[r0:r0 + r] = np.fft.fft(M[r0:r0 + r], axis=1)
    return n1, n2


def real_dft_lean(buf: np.ndarray, upto: int, emit) -> None:
    """DFT X of the real array buf (length m = 2n; overwritten) at k = 0 .. upto - 1 (upto <= n + 1), handed to
    emit(k0, k1, X[k0:k1]) block by block: the complex DFT of z[j] = buf[2j] + i buf[2j+1] in place, then
    X[k] = E[k] + exp(-2 pi i k / m) O[k] with E, O the DFTs of the even and odd entries."""
    n = len(buf) // 2
    z = buf.view(np.complex128)
    n1, n2 = _fft_inplace(z)
    for k0 in range(0, upto, LEAN_BLOCK):
        k = np.arange(k0, min(upto, k0 + LEAN_BLOCK), dtype=np.int64)
        a, c = k % n, (n - k) % n
        Z1 = z[(a % n2) * n1 + a // n2]
        Z2 = np.conj(z[(c % n2) * n1 + c // n2])
        ang = k.astype(float) * (-math.pi / n)
        emit(int(k[0]), int(k[-1]) + 1, 0.5 * (Z1 + Z2) - 0.5j * (np.cos(ang) + 1j * np.sin(ang)) * (Z1 - Z2))


def fgn_lean(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """fgn(n, H, rng) in about 32 bytes per point: the same autocovariance, eigenvalues and random draws (in the
    same order, so rng ends in the same state), the two real DFTs of sqrt(lam/m) a and sqrt(lam/m) b
    (Re y = Re DFT(s a) - Im DFT(s b)) in one buffer of 2n floats."""
    m = 2 * n
    a = 2.0 * H
    buf = np.empty(m)
    buf[0] = 1.0
    for s0 in range(1, n + 1, LEAN_BLOCK):
        kk = np.arange(s0, min(n + 1, s0 + LEAN_BLOCK), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            buf[s0:s0 + len(kk)] = 0.5 * kk ** a * (np.expm1(a * np.log1p(1.0 / kk))
                                                    + np.expm1(a * np.log1p(-1.0 / kk)))
    buf[1] = 0.5 * (2.0 ** a - 2.0)
    buf[n + 1:] = buf[n - 1:0:-1]
    lam = np.empty(n + 1)                     # the circulant's eigenvalues 0 .. n (the others mirror them)
    real_dft_lean(buf, n + 1, lambda k0, k1, X: lam.__setitem__(slice(k0, k1), X.real))
    if lam.min() < -1e-8 * max(1.0, lam.max()):
        raise ValueError(f"circulant embedding is not nonnegative definite (min eigenvalue {lam.min():.3g})")
    np.clip(lam, 0, None, out=lam)
    lam /= m
    np.sqrt(lam, out=lam)
    out = np.empty(n)
    for part in (0, 1):                       # w = a + i b, a drawn first (fgn: rng.normal(size=m) twice)
        for s0 in range(0, m, LEAN_BLOCK):
            rng.standard_normal(out=buf[s0:min(m, s0 + LEAN_BLOCK)])
        buf[:n + 1] *= lam
        buf[n + 1:] *= lam[n - 1:0:-1]
        if part == 0:
            real_dft_lean(buf, n, lambda k0, k1, X: out.__setitem__(slice(k0, k1), X.real))
        else:
            real_dft_lean(buf, n, lambda k0, k1, X: out.__setitem__(slice(k0, k1), out[k0:k1] - X.imag))
    return out


def dfa_lean(x: np.ndarray, scales: np.ndarray, order: int = 1) -> float:
    """hurst.hurst_dfa(x, scales, order)["H"] (DFA = MFDFA at q = 2) with the profile and the segment fits in
    blocks: the same profile (the same sequential sums), segments, least-squares fits and means, in 16 bytes per
    point instead of about 70."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    prof = np.empty(n)
    carry = 0.0
    for s0 in range(0, n, LEAN_BLOCK):
        t = x[s0:s0 + LEAN_BLOCK] - mu
        t[0] += carry
        np.cumsum(t, out=prof[s0:s0 + len(t)])
        carry = prof[s0 + len(t) - 1]
    scales = np.asarray(scales)
    logF = np.empty(len(scales))
    for j, s in enumerate(scales):
        s = int(s)
        ns = n // s
        V = np.vander(np.arange(s, dtype=float), order + 1)
        rows = max(1, LEAN_BLOCK // s)
        F2 = []
        for part in (prof[: ns * s].reshape(ns, s), prof[n - ns * s:].reshape(ns, s)):
            for r0 in range(0, ns, rows):
                seg = part[r0:r0 + rows]
                coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
                F2.append(np.mean((seg.T - V @ coef) ** 2, axis=0))
        F2 = np.concatenate(F2)
        F2 = F2[F2 > 0]
        logF[j] = np.log(np.mean(F2 ** (2.0 / 2))) / 2.0
    return float(np.polyfit(np.log(scales), logF, 1)[0])


# ----------------------------------------------------------------------------- streams
class Stream:
    """One stream: the latent switching process is built at construction (a few seconds for 2^28 tokens); windows
    of seq_len tokens are drawn on demand, each from its own generator, so any window can be read alone."""

    def __init__(self, noah: str, joseph: str, seed: int, role: str, n_windows: int, P: dict = P6,
                 lean: bool | None = None):
        self.noah, self.joseph, self.seed, self.role, self.P = noah, joseph, int(seed), role, P
        self.H = float(P["joseph"][joseph]["H"])
        self.L = int(P["seq_len"])
        self.n_windows = int(n_windows)
        self.n_tokens = self.n_windows * self.L
        self.K, self.R, self.q = int(P["K"]), int(P["R"]), float(P["q"])
        self.law = rank_law(noah, P)
        self.cdf = np.cumsum(self.law)
        self.cdf[-1] = 1.0
        blk = int(P["block"])
        self.n_blocks = -(-self.n_tokens // blk)
        rng = rng_for(seed_for(config.SEED, "p6 latent", self.seed, role))      # not the cell: common numbers
        # lean: the same process in less memory (above LEAN_FROM_BLOCKS blocks; the raised design's train streams)
        self.lean = self.n_blocks > LEAN_FROM_BLOCKS if lean is None else bool(lean)
        if self.lean:
            self._build_lean(rng, P, blk)
            return
        self.g = fgn(self.n_blocks, self.H, rng)       # H = 0.5: i.i.d. normals, from the same draws
        e = np.exp(P["b"] * self.g)
        haz = np.minimum(P["p_max"], P["p_bar"] * e / e.mean())
        self.hazard = haz
        pos, chunk = [], 1 << 24
        for s0 in range(0, self.n_tokens, chunk):
            n = min(chunk, self.n_tokens - s0)
            u = rng.random(n)
            pt = np.repeat(haz[s0 // blk: -(-(s0 + n) // blk)], blk)[(s0 % blk):(s0 % blk) + n]
            hit = np.flatnonzero(u < pt)
            pos.append(hit[hit + s0 > 0] + s0)        # token 0 starts with the initial topic
        self.switch_pos = np.concatenate(pos).astype(np.int64)
        t0 = int(rng.integers(self.K))
        steps = rng.integers(1, self.K, size=len(self.switch_pos))
        self.seg_topic = ((t0 + np.concatenate([[0], np.cumsum(steps)])) % self.K).astype(np.int8)

    def _build_lean(self, rng: np.random.Generator, P: dict, blk: int) -> None:
        """The construction above with the same draws and arithmetic, in place where it can be: fgn_lean, the hazard
        computed in its own array, the topic walk summed in place. The hazard path is not kept (its capped count
        is), and release_latent() drops g once the checks have read it."""
        self.g = fgn_lean(self.n_blocks, self.H, rng)
        haz = np.multiply(self.g, P["b"])
        np.exp(haz, out=haz)
        me = haz.mean()
        haz *= P["p_bar"]
        haz /= me
        np.minimum(haz, P["p_max"], out=haz)
        self.hazard = None
        self.hazard_capped = int((haz >= P["p_max"]).sum())
        pos, chunk = [], 1 << 24
        for s0 in range(0, self.n_tokens, chunk):
            n = min(chunk, self.n_tokens - s0)
            u = rng.random(n)
            pt = np.repeat(haz[s0 // blk: -(-(s0 + n) // blk)], blk)[(s0 % blk):(s0 % blk) + n]
            hit = np.flatnonzero(u < pt)
            pos.append(hit[hit + s0 > 0] + s0)
            del u, pt, hit
        del haz
        self.switch_pos = np.concatenate(pos).astype(np.int64, copy=False)
        del pos
        t0 = int(rng.integers(self.K))
        steps = rng.integers(1, self.K, size=len(self.switch_pos))
        np.cumsum(steps, out=steps)
        steps += t0
        steps %= self.K
        self.seg_topic = np.empty(len(steps) + 1, dtype=np.int8)
        self.seg_topic[0] = t0 % self.K
        self.seg_topic[1:] = steps

    def release_latent(self) -> None:
        """Drop the fGn path (1 GiB at 2^27 blocks) once the latent checks have read it; windows need only the
        switch positions and the topics."""
        self.g = None

    # -- latent labels
    def topics(self, start: int, n: int) -> np.ndarray:
        idx = np.searchsorted(self.switch_pos, np.arange(start, start + n), side="right")
        return self.seg_topic[idx]

    def switch_counts(self) -> np.ndarray:
        """Switches per hazard block (the stream's own Joseph series)."""
        return np.bincount(self.switch_pos // int(self.P["block"]), minlength=self.n_blocks).astype(float)

    # -- tokens
    def _draws(self, w: int):
        rng = rng_for(seed_for(config.SEED, "p6 window", self.seed, self.role, int(w)))
        return rng.random(self.L), rng.random(self.L), rng.integers(0, self.K, self.L)

    def window_parts(self, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rank 1..R, variant, topic) of window w."""
        u_rank, u_var, v_unif = self._draws(w)
        rank = np.searchsorted(self.cdf, u_rank, side="right") + 1
        rank = np.minimum(rank, self.R)
        topic = self.topics(int(w) * self.L, self.L)
        variant = np.where(u_var < self.q, topic, v_unif)
        return rank, variant, topic

    def window(self, w: int) -> np.ndarray:
        rank, variant, _ = self.window_parts(w)
        return ((rank - 1) * self.K + variant).astype(np.int64)

    def windows(self, ws) -> np.ndarray:
        return np.stack([self.window(w) for w in ws]) if len(ws) else np.empty((0, self.L), np.int64)

    def examples(self, w0: int, n_windows: int, length: int) -> np.ndarray:
        """Consecutive windows w0 .. w0 + n_windows - 1 cut into examples of `length` tokens (P1's pool layout)."""
        X = self.windows(range(w0, w0 + n_windows)).reshape(-1)
        return X[: (len(X) // length) * length].reshape(-1, length)


def heldout_layout(P: dict = P6, p1: dict | None = None) -> dict:
    """Disjoint window ranges of the held-out stream: H7.1 and coupling pool (P1's example_len), eval, H_tok
    (16 consecutive windows) and probe (P1's probe_len). p1: P1's measurement settings (config.get)."""
    p1 = p1 or config.get("p1_checkpoints")
    L = int(P["seq_len"])
    ex = int(p1["example_len"])
    nb = int(p1["h71"]["coarse_blocks"])
    n_pool = max(int(p1["coupling_blocks"]) * int(p1["coupling_B"]),
                 nb * max(int(b) for b in p1["h71"].get("batch_sizes", p1["batch_sizes_h71"])))
    w_pool = -(-n_pool * ex // L)
    w_eval = -(-int(p1["eval_seqs"]) * int(p1["eval_len"]) // L)
    w_sur = int(p1["surprisal_docs"]) * -(-int(p1["surprisal_len"]) // L)
    w_probe = -(-int(p1["probe_seqs"]) * int(p1["probe_len"]) // L)
    out, w = {}, 0
    for name, n in (("pool", w_pool), ("eval", w_eval), ("surprisal", w_sur), ("probe", w_probe)):
        out[name] = (w, n)
        w += n
    out["n_windows"] = w
    out["n_pool"] = n_pool
    return out


# ----------------------------------------------------------------------------- stream checks
def token_counts(stream: Stream, every: int = 1) -> tuple[np.ndarray, int, int]:
    """(rank counts over ranks 1..R, tokens whose variant is their topic, tokens) in one pass over the windows."""
    c = np.zeros(stream.R + 1, dtype=np.int64)
    hit = n = 0
    for w in range(0, stream.n_windows, every):
        rank, v, t = stream.window_parts(w)
        c += np.bincount(rank, minlength=stream.R + 1)
        hit += int((v == t).sum())
        n += len(v)
    return c[1:], hit, n


def zipf_mle(counts: np.ndarray, r_from: int = 1) -> tuple[float, float]:
    """MLE of s for p_r ~ r^-s on ranks r_from..R from rank counts; returns (s_hat, log-likelihood per token)."""
    R = len(counts)
    r = np.arange(r_from, R + 1, dtype=float)
    c = np.asarray(counts[r_from - 1:], float)
    n = c.sum()
    lr = np.log(r)
    mean_lr = float((c * lr).sum() / n)

    def nll(s):
        lz = np.logaddexp.reduce(-s * lr)
        return s * mean_lr + lz

    res = optimize.minimize_scalar(nll, bounds=(0.05, 5.0), method="bounded", options={"xatol": 1e-9})
    return float(res.x), float(-res.fun)


def exp_mle(counts: np.ndarray) -> tuple[float, float]:
    """MLE of tau for p_r ~ exp(-(r-1)/tau) on ranks 1..R; returns (tau_hat, log-likelihood per token)."""
    R = len(counts)
    x = np.arange(R, dtype=float)
    c = np.asarray(counts, float)
    mean_x = float((c * x).sum() / c.sum())

    def nll(lt):
        t = math.exp(lt)
        return mean_x / t + np.logaddexp.reduce(-x / t)

    res = optimize.minimize_scalar(nll, bounds=(math.log(1.0), math.log(1e7)), method="bounded",
                                   options={"xatol": 1e-10})
    return float(math.exp(res.x)), float(-res.fun)


def rank_checks(counts: np.ndarray, noah: str, P: dict = P6) -> dict:
    ck, lp = P["check"], law_params(noah, P)
    s_full, ll_z = zipf_mle(counts, 1)
    s_tail, _ = zipf_mle(counts, int(ck["tail_from"]))
    tau, ll_e = exp_mle(counts)
    out = {"law": lp["kind"], "n_tokens": int(counts.sum()), "s_hat": s_full, "s_hat_tail": s_tail,
           "tau_hat": tau, "ll_zipf": ll_z, "ll_exp": ll_e, "ll_zipf_minus_exp": ll_z - ll_e,
           "max_rank_seen": int(np.flatnonzero(counts)[-1] + 1) if counts.any() else 0}
    if lp["kind"] == "zipf":
        out["s_set"] = lp["s"]
        out["rank_ok"] = bool(abs(s_full - lp["s"]) <= ck["s_tol"] and abs(s_tail - lp["s"]) <= ck["s_tol"]
                              and ll_z - ll_e >= ck["ll_margin"])
    else:
        out["tau_set"] = lp["tau"]
        out["rank_ok"] = bool(abs(tau / lp["tau"] - 1) <= ck["tau_rel_tol"] and ll_e - ll_z >= ck["ll_margin"])
    return out


def count_band(H: float, n_blocks: int, ref: Path = CHECK_REF, P: dict = P6,
               exact: bool = False) -> tuple[float, float] | None:
    """Pass band of the switch-count H at this H and length from the calibration (nearest length, log scale; with
    exact, only a calibration at this very length, else None)."""
    if not Path(ref).exists():
        return None
    import pandas as pd
    df = pd.read_csv(ref)
    df = df[np.isclose(df["H"], H)]
    if exact:
        df = df[df["n_blocks"] == int(n_blocks)]
    if df.empty:
        return None
    lens = df["n_blocks"].unique()
    nb = lens[np.argmin(np.abs(np.log(lens) - math.log(n_blocks)))]
    d = df[df["n_blocks"] == nb]["H_count"]
    m, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    half = max(P["check"]["count_band_min"], P["check"]["count_band_sd"] * sd)
    return m - half, m + half


def latent_checks(stream: Stream, ref: Path = CHECK_REF) -> dict:
    if getattr(stream, "lean", False):
        return latent_checks_lean(stream, ref)
    ck, P = stream.P["check"], stream.P
    g = stream.g
    nb = len(g)
    H_g_dfa = hurst.hurst_dfa(g, hurst.log_scales(nb, 10))["H"]
    H_g_wav = hurst.hurst_wavelet(g)["H"]
    counts = stream.switch_counts()
    sc = hurst.log_scales(nb, int(ck["count_s_min"]))
    H_c = hurst.hurst_dfa(counts, sc)["H"]
    rate = len(stream.switch_pos) / stream.n_tokens
    band = count_band(stream.H, nb, ref, P)
    out = {"H_set": stream.H, "n_blocks": nb, "H_g_dfa": H_g_dfa, "H_g_wav": H_g_wav, "H_count": H_c,
           "count_scales": f"{int(sc[0])}..{int(sc[-1])}", "switch_rate": rate,
           "switch_rate_rel": rate / P["p_bar"], "n_switches": int(len(stream.switch_pos)),
           "hazard_capped": int((stream.hazard >= P["p_max"]).sum())}
    out["g_ok"] = bool(abs(H_g_dfa - stream.H) <= ck["g_tol"] and abs(H_g_wav - stream.H) <= ck["g_tol"])
    out["rate_ok"] = bool(abs(rate / P["p_bar"] - 1) <= ck["rate_rel_tol"])
    if band is None:
        out["count_band"] = None
        out["count_ok"] = None          # no calibration: reported, not checked (smoke, tests)
    else:
        out["count_band"] = [round(band[0], 4), round(band[1], 4)]
        out["count_ok"] = bool(band[0] <= H_c <= band[1])
    return out


def latent_checks_lean(stream: Stream, ref: Path = CHECK_REF) -> dict:
    """latent_checks for a lean stream (added 2026-11-18): the same estimators and rules (dfa_lean is
    hurst.hurst_dfa in less memory), with one difference: the switch-count band must have been calibrated at this
    very length (count_band(exact=True)); a calibration file without that length fails the check instead of
    borrowing the nearest length's band."""
    ck, P = stream.P["check"], stream.P
    if stream.g is None:
        raise ValueError("the fGn path was released before the latent checks")
    g = stream.g
    nb = len(g)
    H_g_dfa = dfa_lean(g, hurst.log_scales(nb, 10))
    H_g_wav = hurst.hurst_wavelet(g)["H"]
    counts = stream.switch_counts()
    sc = hurst.log_scales(nb, int(ck["count_s_min"]))
    H_c = dfa_lean(counts, sc)
    del counts
    rate = len(stream.switch_pos) / stream.n_tokens
    band = count_band(stream.H, nb, ref, P, exact=True)
    out = {"H_set": stream.H, "n_blocks": nb, "H_g_dfa": H_g_dfa, "H_g_wav": H_g_wav, "H_count": H_c,
           "count_scales": f"{int(sc[0])}..{int(sc[-1])}", "switch_rate": rate,
           "switch_rate_rel": rate / P["p_bar"], "n_switches": int(len(stream.switch_pos)),
           "hazard_capped": int(stream.hazard_capped), "lean": True}
    out["g_ok"] = bool(abs(H_g_dfa - stream.H) <= ck["g_tol"] and abs(H_g_wav - stream.H) <= ck["g_tol"])
    out["rate_ok"] = bool(abs(rate / P["p_bar"] - 1) <= ck["rate_rel_tol"])
    if band is None and not Path(ref).exists():
        out["count_band"] = None
        out["count_ok"] = None          # no calibration file: reported, not checked (smoke, tests)
    elif band is None:
        out["count_band"] = None
        out["count_ok"] = False         # no calibration at this length
    else:
        out["count_band"] = [round(band[0], 4), round(band[1], 4)]
        out["count_ok"] = bool(band[0] <= H_c <= band[1])
    return out


def variant_check(hit: int, n: int, P: dict = P6) -> dict:
    want = P["q"] + (1 - P["q"]) / P["K"]
    share = hit / n
    return {"variant_share": share, "variant_share_set": want,
            "variant_ok": bool(abs(share - want) <= P["check"]["variant_tol"])}


def check_stream(stream: Stream, ref: Path = CHECK_REF, every: int = 1, latent: dict | None = None) -> dict:
    """Every check of one stream (module docstring); "ok" is False if any check that ran failed. every > 1 reads
    one window in `every` for the rank and variant checks (tests only). latent: latent_checks(stream, ref) run
    earlier (a lean stream's, before its fGn path was released)."""
    t0 = time.perf_counter()
    out = {"noah": stream.noah, "joseph": stream.joseph, "seed": stream.seed, "role": stream.role,
           "n_tokens": stream.n_tokens}
    counts, hit, n = token_counts(stream, every)
    out.update(rank_checks(counts, stream.noah, stream.P))
    out.update(variant_check(hit, n, stream.P))
    out.update(latent_checks(stream, ref) if latent is None else latent)
    flags = [out["rank_ok"], out["variant_ok"], out["g_ok"], out["rate_ok"], out["count_ok"]]
    out["ok"] = bool(all(f for f in flags if f is not None))
    out["failed"] = [k for k, f in zip(("rank", "variant", "fGn H", "switch rate", "switch-count H"), flags)
                     if f is False]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# ----------------------------------------------------------------------------- calibration of the H check
def _calibration_row(job: tuple) -> dict:
    nw, jl, i = job
    s = Stream("zipf1.0", jl, 1000 + i, f"calibration{nw}", nw)
    c = latent_checks(s, ref=Path("/nonexistent"))
    return {"H": s.H, "joseph": jl, "n_windows": nw, "n_blocks": s.n_blocks, "rep": i,
            **{k: c[k] for k in ("H_g_dfa", "H_g_wav", "H_count", "switch_rate_rel")}}


def calibrate(reps: int, lengths: list[int], out: Path, workers: int = 1, P: dict = P6, log=print,
              levels: list[str] | None = None, append: bool = False):
    """The switch-count H (and the fGn H) of `reps` latent processes per H level and length (windows), at seeds
    other than the runs' (1000 + i, role "calibration<windows>"). Writes `out` (csv) and returns it. levels: the
    Joseph levels (default all); append: keep the rows of `out` for other lengths or levels."""
    import concurrent.futures as cf

    import pandas as pd
    levels = list(P["joseph"]) if levels is None else list(levels)
    jobs = [(nw, jl, i) for nw in lengths for jl in levels for i in range(reps)]
    t0 = time.perf_counter()
    if workers > 1:
        with cf.ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_calibration_row, jobs, chunksize=1))
    else:
        rows = [_calibration_row(j) for j in jobs]
    df = pd.DataFrame(rows)
    for (nw, jl), d in df.groupby(["n_windows", "joseph"]):
        log(f"{nw} windows, {jl}: H_count {d.H_count.mean():.4f} +- {d.H_count.std(ddof=1):.4f}, H_g_dfa "
            f"{d.H_g_dfa.mean():.4f} +- {d.H_g_dfa.std(ddof=1):.4f}, H_g_wav {d.H_g_wav.mean():.4f} +- "
            f"{d.H_g_wav.std(ddof=1):.4f}, rate {d.switch_rate_rel.mean():.4f} +- {d.switch_rate_rel.std(ddof=1):.4f}")
    log(f"{len(jobs)} streams in {time.perf_counter() - t0:.0f} s")
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.exists():
        old = pd.read_csv(out)
        keep = ~old.set_index(["n_windows", "joseph"]).index.isin(df.set_index(["n_windows", "joseph"]).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="calibrate the switch-count H check")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--lengths", default="131072,16384", help="stream lengths in windows (train, heldout)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--levels", default=None, help="Joseph levels to calibrate (default all), e.g. H0.5,H0.9")
    ap.add_argument("--append", action="store_true", help="keep the rows of --out for other lengths and levels")
    ap.add_argument("--out", default=str(CHECK_REF))
    args = ap.parse_args(argv)
    if args.calibrate:
        calibrate(args.reps, [int(x) for x in args.lengths.split(",")], Path(args.out), args.workers,
                  levels=args.levels.split(",") if args.levels else None, append=args.append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
