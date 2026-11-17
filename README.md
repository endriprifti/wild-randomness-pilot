# Pilot code for the Section 7 appendix (stage 1)

One Python script per test of `s7-appendix-pilot-design.tex`, shared estimators in `common/`, all settings in
`config.py`. Every script loads models, checkpoints and datasets from Hugging Face, fixes all seeds
(`config.SEED = 20261105`, Python, NumPy, PyTorch, cuDNN deterministic), and writes `results/<test>/` as CSV files
plus `run.json` (config, git commit, GPU type and memory, library versions, wall-clock, command, fallbacks).
`--smoke` writes to `results/<test>/smoke/` and finishes in under five minutes on a CPU or a 4 GB GPU.

| Script | Tests | What it measures |
|---|---|---|
| `p0_calibration.py` | Section 7 rules | CSN labels, Hurst estimators, D_q and H7.1 slope on samples of known state (CPU) |
| `emmy_check.py` | (ours) | Emmy-compiled against eager PyTorch on outlier-heavy Pythia checkpoints, after a GPT-NeoX support probe |
| `p5_escape.py` | H3.3, H6.1 (escape clause), H3.1 (injected arm) | escape times over (η, B), clipped and injected-noise arms; interval-censored likelihood fits |
| `p2_w4a4.py` | H6.2 | W4A4 and weight-only arms at matched bits, PopQA by popularity decile, isotropic-noise control |
| `p4_clipping.py` | H6.1, H7.1 control, H_grad | clipping sweep (fixed p_clip, AutoClip, ZClip, AGC), matched update length, flatness and tail measures |
| `p1_checkpoints.py` | α, α_ESD, K, H, Δh across Pythia, PolyPythias, OLMo | measurement only on public checkpoints; H7.1 at training stages; exploratory coupling |
| `p3_fractional_memory.py` | H4.1 | fractional block against S4 (LegS), S4D and Mamba at matched parameters and equal tuning budget |
| `h_link.py` | H_link (exploratory, added 2026-11-17) | memory of the magnitudes and MFDFA width of P1's token-surprisal series and P4's gradient-noise series against surrogates, from saved samples (CPU) |
| `memory_check.py` | (tooling) | peak GPU memory of the largest steps of P1, P3 and P4, estimated on the meta device (no GPU, no downloads) |

Notation follows the paper: ζ is the fitted Pareto (survival) exponent, α = min(ζ, 2) only for the stable index.

**Quantization is simulated.** P2 (and the Emmy check's comparison) quantizes and dequantizes weights and
activations in plain PyTorch and then runs the ordinary floating-point kernels. No INT4 kernel runs, so the results
concern accuracy (perplexity, PopQA, activation statistics) and say nothing about speed or memory savings.

## Jobs, cheapest first

GPU-hours are the appendix estimates (A100-class), not measurements. Marks: **4 GB** = the RTX 2050 laptop under
WSL2, **T4** = Kaggle's 16 GB T4 (not the P100, see below), **5090** = RTX 5090 32 GB (Blackwell), **L40S** = 48 GB.

| # | Job | GPU-h | Fits | Command |
|---|---|---|---|---|
| 0 | Calibration | CPU | CPU only | `python p0_calibration.py`; extra calibrations: `python p0b_calibration.py --jobs 6` (laptop, see P0b below) |
| 1 | Emmy check | ~1 (ours) | 5090 (Emmy's own benchmark GPU); needs nvcc and the Emmy env | `python emmy_check.py --frontend-only` (CPU), then `python emmy_check.py --probe-only`, then `python emmy_check.py` |
| 2 | Escape grid, H3.3 / H6.1 | 25 | 4 GB, T4, 5090 | `python p5_escape.py` and `python p5_escape.py --part h31` |
| 3 | W4A4 and weight-only, H6.2 (simulated quantization) | 30 | 5090 (Pythia-6.9B and OLMo-7B in bfloat16, 12.8 GiB of weights each, peak 16.5 GiB); L40S also fine | `python p2_w4a4.py` (one model: `--models EleutherAI/pythia-6.9b`) |
| 4 | Clipping sweep, H6.1 / H7.1 control | 30 | 5090 (transformer peak ~21 GiB, see GPU memory below); ResNet-18 also T4 | `python p4_clipping.py --model transformer` and `python p4_clipping.py --model resnet18` |
| 5 | Checkpoint measurements, with the 410M loss-spike check (pass (a) downloads about 391 GB, see below) | 45 | Pythia: 5090 (H7.1's HVP at 14M to 160M peaks ~16 GiB); PolyPythias: T4; OLMo 2 1B: 5090 | `python p1_checkpoints.py --pin-only`, then `--suite pythia`, `--suite polypythia,polypythia_data,polypythia_weight`, `--suite olmo` |
| 6 | Fractional memory, H4.1 | 65 | 5090; T4 is tight (Mamba at n_s = 64 peaks ~13 GiB) | `python p3_fractional_memory.py` (LRA image in its reduced design by default, see "P3" below; drop LRA first if time runs short: `--tasks powerlaw,exponential`) |
| 7 | **Optional, last:** OLMo 2 7B checkpoints | not in the 45 (ours: ~10) | **L40S** (fp32 weights plus per-example JVP exceed 32 GB) | `python p1_checkpoints.py --suite olmo7b`, only if credits remain |

Jobs 0 to 6 all fit the RTX 5090. Job 7 is the only one that needs the L40S, and nothing else depends on it.
Stage 1 runs on an L40S with `run_all.sh` (below).

Every script takes `--smoke`, `--seed N`, `--device cuda|cpu`, `--out DIR` and `--offline-ok` (allow the recorded
stand-ins outside smoke mode; off by default, so a real run fails loudly if a download fails). Smoke runs read the
same datasets, splits and files as the full settings, only fewer items; a dataset stand-in is used only when the
Hub cannot be reached at all. Before a cloud run, `python check_data.py` opens one item of every dataset split
that the full settings read (below, "Datasets"). Run the Emmy check
after P1 if you want it to add the three checkpoints with the largest K from P1's `checkpoints.csv` (it reads
`results/p1_checkpoints/` and the `results/p1_<suite>/` folders that `run_all.sh` writes).

## P0b: extra calibrations on the laptop (CPU only)

`p0b_calibration.py` fills the gaps that P0 left, without touching the cloud run (it is not part of
`run_all.sh`):

1. **Cascade series, 20 replicates.** The MFDFA width test (`multifractal_claim`, 99 phase, IAAFT and shuffle
   surrogates) on binomial cascade series with p = 0.5 (monofractal control: fGn, H = 0.5), 0.7 and 0.8, at
   n = 2^9, 2^13 and 2^16, 20 replicates each. P0's series is the deterministic binomial series, so its
   replicates would be identical; here each replicate is a random-order binomial cascade (each split gives p and
   1 − p to its two halves in random order). It has the same values at every dyadic level, hence the same h(q)
   (unit test).
2. **H7.1 on truncated-Lévy pools.** Signed SαS α = 1.5 values cut at |x| ≤ c, where α̂ must drift towards 2. The
   test can only see a drift that falls inside B = 8 … 1024, and where it falls depends on c. In a short check
   here (2 pools per cutoff): with P0's c = 20, α̂ is already about 1.8 at B = 8 and 2 from B = 64, so the slope is
   only b ≈ 0.03 per doubling, below the 0.05 drift threshold; c = 50 to 200 spread the drift over the range
   (b ≈ 0.06 to 0.07); with c = 500, α̂ starts to rise only near B = 512 (b ≈ 0.03). So the check runs at
   c = 20, 100 and 500. For each, 200 replicate pools give the spread of b (as P0's block check on stable pools),
   and 5 more pools get the full preregistered test (`h71_test`, bootstrap 90% interval, 200 resamples) and its
   reading.
3. **Direct stable fit and the LR test against the Gaussian.** `fit_stable` on SαS α = 1.2, 1.5, 1.8, 2.0 at
   n = 10^3, 10^4, 10^5, 200 replicates each. Reported: bias, SD and RMSE of α̂ (with Nolan's asymptotic SE
   beside it), the share of fits at the boundary α = 2, and the bootstrap LR test's rejection rate at 5% and 10%
   (its size at α = 2, its power below). The LR statistic does not depend on the Gaussian's location or scale, so
   its parametric-bootstrap null depends on n only. It is simulated once per n (1,000 Gaussian samples) and
   shared by all replicates. Refitting 200 null samples per replicate, as `lr_test_gaussian` does, would take
   about 45 h on 6 cores. Three replicates per cell also run `lr_test_gaussian` itself, to show that both give the
   same statistic and p-values within Monte Carlo error (`stable_lr_check.csv`).
4. **Gaussian pools for the H7.1 verdict** (parts `h71null` and `h71gauss`). A flat α̂(B) is also what a Gaussian
   gives, at α̂ = 2, so the verdict reads the level and the scale exponent as well as the slope (see
   [H7.1 verdict](#h71-verdict-slope-level-and-scale-exponent)). `h71null` fits 400 Gaussian pools at each setting
   of the full run (1,600 blocks: P0, P0b, P1 at 14M, P4; 200 blocks: P1's coarse sizes; B = 8 … 1024), the
   reference for the level and for the scale exponent near −1/2. `h71gauss` runs the full test on 5 Gaussian pools,
   which should read "flat at 2 (mild)" where the slope alone reads "invariant". At the end of the run every full
   test (truncated-Lévy and Gaussian) gets its verdict.

```bash
cd ~/pilot && git pull && source ~/.venv-pilot/bin/activate
python p0b_calibration.py --smoke --jobs 6           # about 1 min
nohup python p0b_calibration.py --jobs 6 > p0b.log 2>&1 &
python p0b_calibration.py --jobs 6 --parts stable    # or one part at a time: cascades, h71, h71null, h71gauss, stable
```

If a P0b run from the earlier commit (f3a59ad, three parts) is already going, let it finish: its numbers match
this version's (same seeds). Then add the two new parts in their own folder, and give its Gaussian reference to the
verdict tool for the truncated-Lévy tests:

```bash
nohup python p0b_calibration.py --jobs 6 --parts h71null,h71gauss --out results/p0b_h71_level > p0b_level.log 2>&1 &
python -m common.h71_verdict results/p0b_calibration/h71_trunc_tests.csv --null results/p0b_h71_level/h71_gauss_null.csv
```

`--jobs` sets the worker processes (default: every CPU), each with one thread. Every task has its own seed, so the
results do not depend on `--jobs`. Each part's files are written as soon as that part is complete. The run starts
with a line giving its task count and an estimate.

**Run time: about 4.5 h on 6 cores** (3.5 to 6 h, depending on the laptop's speed per core). This is an estimate
from single tasks timed on this environment's CPU, whose timing of P0 (about 13.7 h on 4 threads) matched the
13.1 h P0 took on the L40S host. It adds up to about 26 CPU-hours:

- the 15 full H7.1 tests on truncated-Lévy pools: about 11.4 h, 46 min each;
- the lr_test_gaussian check: about 3.4 h;
- the other stable fits: about 3 h;
- the 800 Gaussian reference pools (`h71null`): about 2.7 h;
- the 5 full H7.1 tests on Gaussian pools (`h71gauss`): about 2.5 h;
- the 600 H7.1 replicate pools: about 2.3 h;
- the cascades: about 0.7 h.

The two new parts alone (`--parts h71null,h71gauss`) take about 5 CPU-hours, under 1 h on 6 cores.

Outputs (`results/p0b_calibration/`):

- `cascade_series.csv` and `cascade_series_summary.csv`: the claim rate per (p, n);
- `h71_trunc.csv`, `h71_trunc_tests.csv` and `h71_trunc_summary.csv`: per cutoff, the spread of b, the share
  above 0.05, mean α̂ per B, the test readings and the verdicts;
- `h71_gauss_null.csv` and `h71_gauss_null_summary.csv`: the Gaussian reference (one row per pool; per setting the
  5% quantile of mean α̂, the central 90% of the scale exponent, the share of fits at α̂ = 2);
- `h71_gauss_tests.csv`: the full test on Gaussian pools, with verdicts; `h71_verdicts.csv`: verdict counts per pool;
- `stable_fit.csv`, `stable_fit_summary.csv`, `stable_lr_null.csv` and `stable_lr_check.csv`;
- `run.json`.

## H7.1 verdict: slope, level and scale exponent

`h71_test` reads only the slope b of α̂ on log2 B (its `reading` column): "invariant" if the 90% interval lies
inside ±0.03, "drift" if above 0.05. A flat slope is what a stable law gives, but also what a Gaussian gives once
its blocks are Gaussian: α̂ then sits at 2 at every B. `common/h71_verdict.py` therefore reads three things from
each test's summary row:

| Verdict | Slope b (90% bootstrap interval) | Level ᾱ (mean α̂ over B) | Scale exponent s (log γ̂_B on log B) |
|---|---|---|---|
| invariant (wild) | inside ±0.03 | below 2: under the 5% quantile of ᾱ on Gaussian pools of the same size, and ᾱ ≤ 1.9 | consistent with 1/ᾱ − 1: the bootstrap 90% interval of s − (1/ᾱ − 1) contains 0 |
| flat at 2 (mild) | inside ±0.03 | consistent with 2: not under that quantile | near −1/2: inside the central 90% of s on the same Gaussian pools |
| drift | above 0.05 | any | any |
| undecided | anything else, including ᾱ between 1.9 and the Gaussian quantile | | `verdict_reason` says which condition failed |

The margin 1.9 (Endri, 2026-11-14; `H71["wild_max_alpha"]` in `config.py`, `--wild-max`, 2 turns it off) is
explained under "Known limit" below. α̂ cannot exceed 2, so α = 2 is tested one-sided against Gaussian pools with the same pool length, blocks and batch
sizes, fitted exactly as a real pool (the bootstrap is not used for the level, since it fails at the boundary).
Every input is already a column of the summary CSVs (`b_lo`, `b_hi`, `alpha_mean`, `scale_exponent`,
`scale_gap_lo`, `scale_gap_hi`, `n_pool`, `n_blocks`), so the verdict is recomputed from saved files, without
refitting. Since 2026-11-17 P1 and P4 write it into their own `h71_summary.csv` (see "Batches in random order"
below), against `reference/h71_gauss_null.csv` in the repository: 400 Gaussian pools each at 1,600 and 200 blocks
and B = 8 … 1,024, the full-run settings (P0b's h71null part with the same seeds, FFT density). For files written
before, on the L40S or the laptop:

```bash
python -m common.h71_verdict results/p0_calibration/block_check_bootstrap.csv \
    results/p1_pythia/h71_summary.csv results/p4_resnet18/h71_summary.csv --jobs 6
```

It writes `<name>_verdict.csv` beside each file (the original columns, including the slope-only `reading`, plus
`alpha_null_q`, `s_null_lo`, `s_null_hi`, `null_reps`, `slope_reading`, `level_reading`, `scale_reading`,
`verdict` and `verdict_reason`) and prints the counts. The batch sizes come from the `alpha_B<B>` columns (P0b),
from `h71_per_B.csv` beside the file (P1, P4) or from `config.BATCH_SIZES` (P0; `--batch-sizes` overrides). A
setting missing from the Gaussian reference (for example a pool that came out short) is simulated on the spot
and added to the `--null` file (default `results/p0b_calibration/h71_gauss_null.csv`, read together with the
reference in the repository; 400 pools, about 1 h on 6 cores at 1,600 blocks).

**Check at full size** (this environment's CPU, 2026-11-14; 1,600 blocks, B = 8 … 1024, 100 Gaussian reference
pools, full tests with a reduced bootstrap of 30 resamples). The reference puts the 5% quantile of ᾱ at 1.993
(13% of Gaussian pools have ᾱ = 2 exactly) and s between −0.507 and −0.492. The slope alone read every flat pool
"invariant". The verdict read two SαS α = 1.5 pools (ᾱ 1.53, 1.54) as "invariant (wild)", a truncated-Lévy pool
with cutoff 100 as "drift" (b in [0.068, 0.086]), one Gaussian pool as "flat at 2 (mild)", and two pools as
"undecided" because ᾱ lay between 1.9 and 1.993: a Gaussian pool with ᾱ = 1.992 (without the margin it would read
wild, the one-sided test's 5% size) and an SαS α = 1.9 pool with ᾱ = 1.915.

**Known limit: noise near α = 2.** Point fits on finite-variance Student-t pools (no bootstrap) give a flat or
nearly flat slope with ᾱ below the Gaussian quantile: ν = 3, ᾱ 1.93 to 1.95 (b 0.015 to 0.026); ν = 4, ᾱ 1.98
(b 0.006); ν = 5 and 10 read as Gaussian (ᾱ 1.995, 1.999). SαS α = 1.95 and 1.97 give ᾱ 1.96 and 1.98. So up to
B = 1,024 a stable law with α above about 1.9 cannot be told from finite variance with ζ of 3 to 4 (slow
convergence to the Gaussian); without a margin the rule reads both "invariant (wild)". Hence the margin:
"invariant (wild)" needs ᾱ ≤ 1.9, and pools between 1.9 and the Gaussian quantile read "undecided". The
per-example tail fit of §7.1 (ζ) separates these cases where the batch-size test cannot.

## H_link: memory of the magnitudes and multifractality (exploratory, added 2026-11-17)

Added on 2026-11-17, after the first results, and fixed before any H_link value was computed (Endri). It is
exploratory: it is not in the hypothesis register, the coupling family or its Holm correction.

**Claim.** The Noah variable's magnitudes carry Joseph-type memory (volatility clustering), and the series are
multifractal.

**Series.** Only what the jobs saved, read without changing anything:

- (a) the token-surprisal series of every P1 checkpoint whose GPU pass is done (`row.json` present):
  `checkpoints/<pass>/<checkpoint>/samples/htok.npy`, 16 documents of 2,048 tokens one after another
  (n = 32,768);
- (b) the gradient-noise series of every finished P4 run (a row in `runs/<tag>.json` or `runs.csv`):
  `hgrad_series/<tag>.npy`, the unclipped minibatch gradient projected on 8 fixed random directions at every step,
  after warm-up as for H_grad (transformer 9,064 steps, ResNet-18 about 11,500). A run's statistics are means over
  its 8 directions, and every surrogate replicate replaces each direction by its own surrogate.

**Test** (`common/hlink.py`; 99 surrogates of each kind; one-sided p = (1 + #{surrogate ≥ observed}) / 100):

1. memory of the magnitudes: H of m = |x − median(x)| by DFA-1 and by the wavelet estimator (Abry–Veitch, db3,
   octaves 3 up to the coarsest with four coefficients), against shuffled surrogates of x (same values, no memory:
   the pass rule) and phase-randomized surrogates (same spectrum, Gaussian: reported as `p_phase_*` and
   `memory_beyond_linear`, memory beyond what the linear correlations of x give);
2. multifractality: MFDFA width Δh = h(0.5) − h(4) of x against IAAFT surrogates (same values and spectrum).

**Pass:** p_shuf < 0.05 by DFA and by the wavelet estimator, each over at least two decades of scale, AND
p_iaaft < 0.05.

**Scales.** DFA and MFDFA use 20 log-spaced scales from 8 to n/10. §7.2 uses 10 to n/10; the minimum was lowered
to 8 before any H_link value was computed, because P4's transformer series (9,064 steps after warm-up) span only
1.96 decades from 10 to 906 and 2.05 from 8. Every series also gets a sensitivity row at §7.2's 10 to n/10, on the
same surrogates (`scales` = "sensitivity"; the wavelet estimator has no minimum scale and is the same in both
rows). There the transformer series fail the two-decade clause and cannot pass: `two_decades` is False and
`decades_flag` reads "below 2 decades (DFA 1.96, wavelet 2.11)". `pass_ignoring_decades` (reported only) gives
the p-value part without that clause.

**Calibration** (`python h_link.py --calibrate`, run first; 100 series per kind and length, at P1's length 32,768
and at the shortest P4 length 9,064, scales 8 to n/10; parameters ours, fixed before the calibration ran: SαS
α = 1.5; fGn H = 0.75; multifractal random walk with λ² = 0.05 and integral scale n, Δh = 0.0875 in theory;
random-order binomial cascade with a = 0.75, Δh = 0.439 in theory). This environment's CPU, 4 workers at nice 19,
57 min. Shipped as `reference/h_link_calibration.csv` (one row per series and scale range) and
`reference/h_link_calibration_summary.csv`. Rates over 100 series; the medians are of the main rows:

| Series (expected) | n | Pass | Memory part | Width part | Memory beyond linear (reported) | H(\|x\|) DFA: series / shuffled | Δh: series / IAAFT | Pass at 10 to n/10 (ignoring decades) |
|---|---|---|---|---|---|---|---|---|
| Lévy, i.i.d. SαS, α = 1.5 (Noah only: fail) | 32,768 | 0.00 | 0.02 | 0.07 | 0.05 | 0.50 / 0.50 | 0.392 / 0.399 | 0.00 (0.00) |
| Lévy, i.i.d. SαS, α = 1.5 (Noah only: fail) | 9,064 | 0.00 | 0.01 | 0.05 | 0.06 | 0.50 / 0.50 | 0.392 / 0.395 | 0.00 (0.00) |
| fGn, H = 0.75 (Joseph only: fail) | 32,768 | 0.02 | 1.00 | 0.02 | 0.01 | 0.57 / 0.50 | 0.008 / 0.006 | 0.04 (0.04) |
| fGn, H = 0.75 (Joseph only: fail) | 9,064 | 0.02 | 0.98 | 0.03 | 0.03 | 0.58 / 0.50 | 0.011 / 0.009 | 0.00 (0.05) |
| MRW, λ² = 0.05 (both: pass) | 32,768 | 0.86 | 1.00 | 0.86 | 1.00 | 0.87 / 0.50 | 0.086 / 0.047 | 0.90 (0.90) |
| MRW, λ² = 0.05 (both: pass) | 9,064 | 0.81 | 1.00 | 0.81 | 1.00 | 0.84 / 0.50 | 0.087 / 0.055 | 0.00 (0.81) |
| binomial cascade, a = 0.75 (both: pass) | 32,768 | 0.10 | 1.00 | 0.10 | 1.00 | 0.83 / 0.50 | 0.443 / 0.385 | 0.17 (0.17) |
| binomial cascade, a = 0.75 (both: pass) | 9,064 | 0.01 | 1.00 | 0.01 | 0.98 | 0.82 / 0.50 | 0.446 / 0.419 | 0.00 (0.01) |

- **Size.** Lévy and fGn pass in 0 to 2% of series.
- **MRW.** It passes in 81% (9,064) and 86% (32,768) of series; its Δh, 0.087, is the theoretical 0.0875.
- **Cascade.** It does not pass (1% and 10%). Its memory part always passes, but IAAFT surrogates keep most of its
  width (0.39 to 0.42 of 0.44): that width comes mostly from its very broad marginal (values over about seven
  decades) and its spectrum, which IAAFT keeps. The width part therefore credits only multifractality beyond the
  marginal and the spectrum, the MRW kind, and a cascade-like series with a very broad marginal fails it.
- **Memory part alone.** It does not separate volatility clustering from linear long memory: fGn's magnitudes pass
  it in 98 to 100% of series (H(|x|) 0.57 against 0.50), through its correlations (the correlation of |x| at lag k
  is about ρ(k)², which decays as 1/k at H = 0.75). The phase-surrogate column separates the two (fGn 1 to 3%, MRW
  100%, cascade 98 to 100%); in the pass rule only the width part keeps fGn at 2%.
- **10 to n/10.** No series of 9,064 values can pass there (1.96 decades); without that clause the rates are those
  of the main rows.

**Commands** (in the pilot folder inside the venv, after pulling; the lanes, the queue and olmo7b keep the code they
loaded, and this job only reads their folders):

```bash
python h_link.py --dry-run          # the P1 and P4 folders and the finished series it would compute
nohup taskset -c 0-10 python h_link.py > logs/h_link.log 2>&1 &
```

CPU only: `CUDA_VISIBLE_DEVICES` is emptied before anything is imported, torch is not imported, and 2 worker
processes (`--workers`) run at nice 19 with one thread each. It finds every `results/p1_*/` with checkpoints (not the
`*_recomputed` copies, which share their source's samples) and every `results/p4_*/` with `hgrad_series/`.
Series that do not exist yet (the queued p4_transformer, checkpoints still to come) are not listed; the same
command later computes only the new series (and any whose file changed) and rewrites the CSVs. It writes
`results/h_link/`: `p1_tokens.csv` and `p4_grad.csv` (two rows per series, `scales` main and sensitivity, with
every H, surrogate mean and 95% quantile, p-value and flag), `summary.csv` (pass rates per folder, model size in P1
or model in P4, and scale range; diverged P4 runs are in `p4_grad.csv` only), `data/units/` (one JSON per series)
and `progress.json` (measured hours per series and hours left, summed over workers). On this environment's CPU a
P1 series takes about 30 s and a P4 transformer run (8 directions) about 85 s per worker; the server's own times
are in `progress.json` once the first series are saved.

**Caveats.**

- P1's series joins 16 documents. A change of surprisal level from one document to the next is a slow change in
  the magnitudes, which DFA and the wavelet estimator read as memory and shuffling removes, so the memory part
  cannot tell document-level shifts from clustering within documents (one document, 2,048 tokens, spans only 1.4
  decades from 8 to n/10).
- P4's projections include the full-batch gradient, which changes slowly over training; that drift also raises H
  of the magnitudes. The series are H_grad's, by the same definition.
- Linear memory alone passes the memory part (fGn above), so read `memory_beyond_linear` beside `memory_pass`; and a
  failed width part does not rule out cascade-type multifractality with a very broad marginal (the cascade above).
- Rates are per series; the checkpoints of one run, and the runs of one seed, are not independent.

## Full run on one L40S: `run_all.sh`

No RTX 5090 was available, so stage 1 runs on a CloudRift L40S (48 GB; the smoke run there passed all 9 jobs,
P1 in 264 s and P4 in 157 s). `run_all.sh` runs the GPU jobs one after another in the order of the table above:

p5 (escape, then h31) → p2 → p4 (transformer, then resnet18) → p1 (pythia, then polypythia with the data and weight
seeds and both spike passes, then olmo) → p3 → p1 olmo7b last.

P0 needs only a CPU, so it does not hold up the GPU: it starts in the background at the beginning and runs next to
the GPU jobs, pinned to 4 CPUs (`taskset`) with every thread pool capped at 4 (`OMP_NUM_THREADS` and the like).
While it runs, the GPU jobs are kept on the other CPUs; once it ends they get all of them. `P0_THREADS=6 bash
run_all.sh` changes the split. The script finishes when both the GPU jobs and P0 are done. The Emmy check is not
part of it (it needs nvcc and the Emmy environment; run it separately on a 5090).

```bash
cd ~/pilot && source ~/.venv-pilot/bin/activate
export HF_TOKEN=...                                    # avoids Hub rate limits over ~1,300 downloads
export PILOT_SSH=riftuser@<host> PILOT_SSH_PORT=<port>  # only for the scp line the script prints
bash run_all.sh --dry-run                              # list the jobs and the CPUs each gets, run nothing
nohup bash run_all.sh > logs/run_all.out 2>&1 &         # or inside tmux; survives a dropped SSH session
tail -f logs/progress.txt
```

- **Logs.** Each job writes `logs/<job>.log`. `logs/progress.txt` gets one line when a job starts and one when it
  ends, with the UTC time, the exit code, the elapsed hours and the running total over all runs of the script.
  Elapsed hours come from a monotonic clock. The total adds up job hours; P0 overlaps the GPU jobs, so it is not
  wall time.
- **Failures and resume.** A failed job does not stop the rest. A job that exits with 0 leaves `logs/<job>.done`,
  and a rerun of `run_all.sh` skips it, so after an interruption you simply start the script again. Failed or
  interrupted jobs run again, from their saved units where they have them (P1, P2, P3, P4 and P5's escape grid save
  each finished unit, see "A third lane" below; P0 starts again). Delete `logs/<job>.done` to rerun a finished
  job. `--only job1,job2` and `--skip job1,job2` select jobs, and only one `run_all.sh` can run at
  a time. Stopping the script (Ctrl-C, `kill`) stops the running GPU job and P0 and records both as interrupted.
- **Archives.** After each job (also after a failure) `results_archive/<job>.tar.gz` holds `results/<job>/` and
  the job's log, and `results_archive/pins.json` is refreshed. At the end the script prints the `scp` line to run
  on the laptop, `scp -P <port> '<user>@<host>:<path>/results_archive/*' .`, built from `PILOT_SSH` and
  `PILOT_SSH_PORT` (kept out of the repository).
- **Pins.** The jobs run without `--require-pins`: a name that is not pinned yet (pass (b)'s checkpoints, OLMo 2
  7B) is resolved on first use and written to `pins.json`, so no job fails for that. At the end the script says
  how many names were added; upload `results_archive/pins.json` so it can be committed.
- **Network.** A Hub call or download that fails for a transient reason (dropped connection, timeout, HTTP 408,
  429 or 5xx, looked up along the exception's causes) is retried four times, 10 s to 270 s apart, before the job
  gives up. Anything else fails at once: a wrong split (ValueError), a missing repository, revision or file, any
  other 4xx, an unpinned name under `--require-pins` (until 2026-11-16 a ValueError was retried four times, about
  7 minutes per job).
- **Output folders.** Each job writes to `results/<job>/` (`--out`), so the three P1 suites and the two P4 models do
  not overwrite each other.

### A second lane for P1 next to run_all.sh: `run_lane.sh`

P1's suites are mostly download-bound, so they can run next to another GPU job. `run_lane.sh` runs P1 jobs one
after another with the same commands and `--out` folders as `run_all.sh`, and writes the same records (START/END
lines in `logs/progress.txt`, tagged "(lane)", `logs/<job>.log`, `logs/<job>.done` on success, and atomically
`results_archive/<job>.tar.gz` and `results_archive/pins.json`), so a later `run_all.sh` skips the finished jobs.
It does not change `run_all.sh`; pull first (see "Pulling on the L40S" below), then:

```bash
nohup taskset -c 7-10 bash run_lane.sh p1_pythia p1_polypythia p1_olmo > logs/run_lane.out 2>&1 &
```

- **GPU memory.** By `python memory_check.py p1` the suites peak at 16.3 (pythia), 7.7 (polypythia) and 17.3 GiB
  (olmo); with P3's 13 GiB that is at most about 30.6 GiB of tensors, plus about 0.5 GiB of CUDA context per
  process, against 44.4 GiB. Before each job the lane waits until the GPU has the job's peak plus 4 GiB free
  (`nvidia-smi`; `LANE_HEADROOM_GIB`).
- **One P1 job at a time.** `p1_checkpoints.py` holds an exclusive lock (`logs/.p1_gpu.lock`) for its whole run,
  taken before CUDA starts. So `run_all.sh`'s `p1_olmo7b` (31.5 GiB) never overlaps a lane suite: whichever comes
  second waits, holding no GPU memory, and notes the wait in its `run.json` (its elapsed hours in progress.txt
  include the wait).
- **Shared files.** `pins.json` is written under a lock (`pins.json.lock`) from a per-process temporary file and
  renamed into place; entries only grow, and the file on disk wins where two jobs carry the same field, so no job
  undoes another's additions. progress.txt lines are appended one write each, so lines from the two scripts never
  mix. Each job has its own log, `.done` file and archive.
- **Checks.** Only P1 jobs are accepted, or `p3_fractional` in a lane of its own (see "P3" below); a job with a
  `.done` file is skipped; a job that progress.txt shows as running is refused; while `run_all.sh` runs, only jobs
  before its current job in its order are accepted (it will not run those again); a second lane of the same kind
  does not start (`logs/.run_lane.lock`, and `logs/.run_lane.p3.lock` for P3). `--dry-run` shows what would run.
  Until the lane has finished, start any new `run_all.sh` with `--skip` for the lane's jobs.
- **p1_olmo7b stays last.** `p1_checkpoints.py --suite olmo7b` first waits until no lane runs, then takes
  `logs/.p3_gpu.lock` (which P3 holds while it runs) and the P1 lock (`common/gpu_locks.py`). So when run_all.sh
  reaches it while lanes still run, it waits without GPU memory instead of taking the P1 lock in the gap between
  two suites of a lane, and it never shares the GPU with P3 (31.5 + 12.9 GiB would not fit).

### P3: a save after every run, the reduced LRA design, and replacing the first P3 (2026-11-17)

Until 2026-11-17, P3 kept every finished run in memory and wrote `runs.csv` only after the last task, so stopping
it lost everything. Now:

- **A save per run, and resume.** Each finished run is written to `runs/<task>/<phase>_<arch>_ns<n_s>_seed<seed>_lr<lr>_steps<steps>.json`
  (atomically), and `runs.csv`, `curves.csv` and `states_to_target.csv` are rewritten after every run. Run the same
  command again and it goes on from the saved runs; a run stopped in the middle starts again. A run's result
  depends only on its settings (each run seeds itself from its seed; the synthetic data are drawn from the run seed
  in config order, as before, whatever `--tasks` says), so the order of runs and restarts change nothing: a smoke
  run stopped with SIGTERM and resumed gives the same `runs.csv` and `curves.csv` as an uninterrupted old-code run. `runs/<task>/settings.json` holds the settings
  that decide a run's result (the task's data, batch, layers, H_ref, weight decay, seed); a restart with others
  stops. Main runs go seed by seed (every arm and n_s of seed 1, then seed 2, ...), so a P3 stopped early leaves
  whole seeds. Each run logs a line with its accuracy, seconds per step and the hours left by the measured step
  times; `python p3_fractional_memory.py --plan` prints the saved and remaining runs per task and phase with those
  hours, and the measured seconds per step by arm and n_s, without training.
- **LRA image designs** (`config.py`, `lra_designs`). The first plan (`--lra-design full`): 12 tuning runs (four arms
  × three learning rates at n_s = 16, seed 0) and 120 main runs (six n_s × four arms × five seeds), each of
  20 × 1,562 = 31,240 steps, 10.4 times a synthetic run's 3,000 steps at the same cost per step (L = 1,024 and
  batch 32 in every task). At the first P3's pace on the L40S next to the P1 lane (about 32 h for powerlaw and
  most of exponential) that is about 180-215 h. The reduced design (the default): three epochs (4,686 steps),
  n_s = 2, 8 and 32, the four arms, the same learning-rate grid tuned at n_s = 8 on seed 0, and main runs on seeds
  1, 2 and 3: 12 + 36 runs, an estimated 7-10 L40S hours by the same pace (the range is the cost of Mamba's scan,
  which grows with n_s, against an equal cost per run); `--plan` gives the measured figure once runs are saved.
  It can show whether, on real images at an equal short training budget and equal tuning, the fractional block's
  accuracy at small state sizes is above, at or below S4-LegS, S4D and Mamba, and whether that ordering follows the
  power-law task or the exponential one: H4.1's clause that an advantage must depend on the task type, now on real
  data, with three seeds (differences of about two seed SDs or more). It cannot show accuracies near published LRA
  numbers (two-layer models of 10-25 thousand parameters trained for three epochs; the 0.6 target of
  `states_to_target.csv` will likely not be reached, so the comparison rests on the curves), what the arms reach
  with 20 epochs (an arm that learns faster can lead at three epochs and not at twenty), n_s = 4, 16 or 64, or small
  differences; and whether images have power-law memory at all is not tested by it.
- **Locks.** P3 holds `logs/.p3_gpu.lock` while it runs on the GPU; `p1_olmo7b` takes the same lock (see the lane's
  "p1_olmo7b stays last" above). `run_lane.sh p3_fractional` runs P3 with run_all.sh's command and records, in a
  lane of its own next to the P1 lane.
The first P3 (run by `run_all.sh` since 2026-11-16) was stopped on 2026-11-17. A gdb-based tool meant to copy its
finished runs out of the running process (`p3_rescue.py`) froze it instead (gdb: "The program being debugged was
signaled while in a function called from GDB"), so its runs were lost; the tool and P3's `--adopt` option are
removed. The new P3 runs from scratch in a lane of its own next to the P1 lane:

```bash
nohup taskset -c 0-6 bash run_lane.sh p3_fractional > logs/run_lane_p3.out 2>&1 &
python p3_fractional_memory.py --out results/p3_fractional --plan      # saved runs and hours left, any time
```

When it ends, its END line widens the P1 pool to every CPU, and `logs/p3_fractional.done` makes a later
`run_all.sh` skip P3. `p1_olmo7b` starts when the P1 lane, the P3 lane and the queue below have all ended.

### A third lane for the reruns: `run_queue.py` (2026-11-17)

`run_queue.py` runs the remaining reruns, each as soon as the GPU has room for it, next to the P1 and P3 lanes,
with the same records as the lanes (START/END lines in `logs/progress.txt` tagged "(queue)", `logs/<job>.log`,
`logs/<job>.done`, `results_archive/<job>.tar.gz`, `results_archive/pins.json`). `run_all.sh` is unchanged.

| Job | Command (each with `--out`) | GPU peak (GiB) |
|---|---|---|
| `rerun_p5_floor10` | `p5_escape.py --floor-from results/p5_escape --alphas 1.5,1.8,2.0 --fine-check 10:250` → `results/p5_escape_floor10` | 1.1 |
| `rerun_p5_a12` | `p5_escape.py --modes injected --alphas 1.2 --fine-check 10:250 --second-reading --log-checks` → `results/p5_escape_a12` | 1.1 |
| `rerun_p5_refit` | `p5_escape.py --part refit --base results/p5_escape --replace results/p5_escape_floor10,results/p5_escape_a12` → `results/p5_escape_refit`, after both | CPU |
| `rerun_p5_h31_flags` | `p5_escape.py --part h31-flags --base results/p5_escape_h31` → `results/p5_escape_h31_flags` | CPU |
| `rerun_p2_popqa` | `-m common.popqa results/p2_w4a4` → `results/p2_w4a4_popqa` | CPU |
| `rerun_p2_bf16` | `p2_w4a4.py` → `results/p2_w4a4_bf16` | 16.5 |
| `p4_transformer` | `p4_clipping.py --model transformer` → `results/p4_transformer` (run_all.sh's job; its `.done` makes run_all.sh skip it) | 21.4 |

- **When a job starts.** Every process on the GPU is counted at 1.05 × its peak (`python memory_check.py`, tensors
  only) + 0.6 GiB (allocator rounding, CUDA context): the queue's running jobs, `run_all.sh`'s running job, and
  each running lane at the largest peak of the jobs it has left (the P1 lane: `p1_pythia`, `p1_polypythia`,
  `p1_olmo` without a `.done`; the P3 lane: `p3_fractional`). A GPU job starts when (1) the sum with it, plus
  1.5 GiB, is at most 44.39 GiB; (2) every lane with a job still to start keeps room for that job's peak plus the
  4 GiB `run_lane.sh` waits for, so a lane never waits for the queue; and (3) `nvidia-smi` shows the job's own
  share free. Otherwise it checks again every minute. CPU jobs start at once, one at a time; the refit waits for
  both P5 reruns. Jobs start in list order whenever they fit, so P2 (about 9 h in the first run) goes before P4
  when both fit only one at a time. A failed job runs once more (it goes on from its saved units); a job whose
  dependency failed does not run.
- **What overlaps** (`python run_queue.py --combos`, everything at its peak, lanes included):

  | Lanes running | Queue jobs that fit together | GiB counted |
  |---|---|---|
  | P1 lane (a job still to start) + P3 lane: now | both P5 reruns | 36.4 |
  | P3 lane | both P5 reruns + P2, or both P5 reruns + P4 | 35.6 / 40.7 |
  | P1 lane | both P5 reruns + P2, or P4 alone | 40.2 / 41.8 |
  | none | P2 + P4 + one P5 rerun, or both P5 reruns with P2 or P4 | 42.8 / 21.4 / 26.6 |

  P2 and P4 never run next to both lanes; P2 and P4 run together only when no lane runs. The P5 reruns and the CPU
  jobs start at once.
- **olmo7b stays last and alone.** `p1_olmo7b` waits until no `logs/.run_lane*.lock` is held, and the queue holds
  `logs/.run_lane.queue.lock` for its whole run (also while it waits), so olmo7b starts only after the queue ends.
  If olmo7b is already on the GPU when the queue starts (`logs/.p3_gpu.lock` held without a P3 lane), the queue
  starts no GPU job until it ends. Start the queue while the lanes still run.
- **Resume.** P2, P4 and P5's escape grid now save every finished unit (`common/units.py`): P2 a model's baseline or
  one scheme with its isotropic control (`units/`), P4 one training run with its measurements (`runs/`, seed by
  seed), P5 one grid cell (`cells/`, with W_S, α at W_S and g_max saved by the folder's first session). A restart
  with the same `--out` runs only the units not saved; a unit stopped in the middle runs again; a `settings.json`
  in the folder stops a restart with other settings. Each unit draws its random numbers from its own seed, so
  stopped and resumed smoke runs give the same CSVs as uninterrupted ones (unit tests). The CPU jobs take minutes
  and run again whole. P4's token IDs (300M tokens) are cached in `data/cache/` after the first start.
- **Hours.** No time estimate here rests on CPU timings. Each resumable job writes `progress.json` in its folder
  (the measured hours of every unit, their mean, and the hours left at that mean) and a log line after every unit;
  `python run_queue.py --status` shows them for every job. The first full-size unit of each job is the first
  measurement to rely on: one P2 scheme, one P4 training run (95 runs: 5 seeds × (10 arms + 9 matched)), one P5
  cell (cells differ: those that exit early are quick; `python rerun_hours.py` bounds the P5 reruns at 1.4 and
  4.1 h from the first run's cells). P2 took 9.0 h for both models in the first run.

```bash
cd ~/pilot && source ~/.venv-pilot/bin/activate
# pull as in "Pulling on the L40S" below (the running lanes and olmo7b keep the code they loaded)
python run_queue.py --combos                      # what fits next to what
python run_queue.py --dry-run                     # what would start now and why the others wait
nohup taskset -c 0-6 python run_queue.py > logs/run_queue.out 2>&1 &
sleep 120; tail -5 logs/run_queue.out; tail -4 logs/progress.txt
python run_queue.py --status                      # any time: units done, measured hours per unit, hours left
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# stop: kill -TERM <pid of run_queue.py> (its jobs stop, saved units stay); start it again to go on
```

CPUs 0-6 are shared with the P3 lane: P3 and each queue job keep about one CPU busy, and each job's thread pools
are capped at 2 (`--threads`). A job with `logs/<job>.done` is skipped; delete the file to run it again.

### GPU memory on the L40S

The first full P4 transformer job (2026-11-15) ran out of memory after training, in `measure()`: the Hessian-vector
product ran on all 512 measurement sequences at once (float32, eager attention, a 50k vocabulary). Since then:

- **Micro-batches.** Every batch measure in P4's `measure()` (the HVP behind the top eigenvalue and the Hutchinson
  trace, adaptive sharpness, the per-example projections and the losses) runs on micro-batches set by
  `measure.micro_batch` in `config.py` (16 sequences for the transformer, 128 images for ResNet-18). The loss is a
  mean over examples, so the HVP and the gradient are the sums of the micro-batch ones weighted by n_c / n, and the
  result is the same up to rounding (unit tests). `common/gradients.py` takes `chunk=` for this.
- **Allocator.** `common/runtime.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless it is already
  set; the first log line of every job records the value.
- **OLMo 2 7B.** 27.2 GiB of float32 weights leave little room: the suite sets `eval_batch: 1` and `jvp_batch: 8`
  (sequences per no-grad forward and per JVP; the other suites keep 4 and 32, and `--jvp-batch` still overrides).
- **P3's Mamba scan** kept [B, T, T, D, N] tensors per chunk for the backward pass, which grows with the state size:
  about 19, 38 and 77 GiB at n_s = 16, 32 and 64 (10, 11 and 13 GiB now). The chunk now shrinks with the state size (`ssm.scan_chunk`:
  T · n_s ≤ 256, at most 32 steps), which gives the same scan.

`python memory_check.py` estimates the peak of each large step without a GPU or downloads: it builds every model on
the meta device and adds up live tensor storage after every operation (CUDA context and allocator overhead come on
top, so a step above 85% of the card reads "tight"). It reproduces the failure (537 GiB for the unchunked HVP,
`--calibrate`) and the training step that ran (21 GiB). At the settings in `config.py`, against 44.4 GiB:

| Job | Largest step | Peak (GiB) |
|---|---|---|
| `p4_transformer` | training step, 64 × 512 tokens with AdamW (HVP at 16 sequences: 16.9) | 21.4 |
| `p4_resnet18` | HVP at 128 images | 2.5 |
| `p1_pythia` | HVP for H7.1's Hessian direction, 64 × 128 tokens (14M to 160M); 410M to 1.4B: JVP and losses | 16.3 |
| `p1_polypythia` | per-example JVP, 32 × 128 tokens (no HVP) | 7.7 |
| `p1_olmo` | per-example JVP, 32 × 128 tokens | 17.3 |
| `p3_fractional` | Mamba training step at n_s = 64, 32 × 1,024 | 12.9 |
| `p1_olmo7b` | per-example JVP, 8 × 128 tokens (27.2 of it weights) | 31.5 |
| `rerun_p2_bf16` (`p2_w4a4`) | OLMo-7B's rotation scheme being applied, bfloat16 (12.8 of it weights; PopQA generation of 16 × 272 tokens: 16.3) | 16.5 |
| `rerun_p5_*` (`p5_escape`) | full-batch gradient of all 50 restarts on 10^4 images | 1.1 |

**P2's quantization** (2026-11-17). `quant.apply_scheme` held every original layer until the whole model was
quantized, so the model was on the GPU twice for a moment: 26.2 GiB in bfloat16 (`python memory_check.py p2
--calibrate`). It now frees each original layer as it is replaced (16.5 GiB). NF4 finds each weight's nearest code
2^16 blocks at a time instead of building a [blocks, 64, 16] distance tensor for the whole matrix (4.3 GiB in float32
for a 4096 × 16384 weight); each code depends only on its own value and block, so the result is identical (unit
test). In P2's check, `memory_check.py` leaves out the weight bank's copies, which live on the CPU.

### Reruns after the first L40S run (2026-11-15)

Fixes in the `.py` files (`run_all.sh` is unchanged):

- **P2.** Both models now load in **bfloat16** (`config.py`; `--dtype` overrides): Pythia-6.9B overflowed in
  float16 and gave a NaN baseline (perplexity NaN, accuracy 0), and a non-finite baseline perplexity now stops the
  job with an error. The outlier scheme left its kept channels inside the per-token activation scale and the weight
  group scales and only restored them afterwards, so they still set the step size of every other channel; they are
  now left out of both (`quant.quant_act`, `quant.quant_weight_groups(..., keep)`). The rotation arm computes its
  rotations in float32. `trends.csv` gives the drop's slope on log10 popularity over all ten deciles, absolute and
  relative, raw and with relation types held fixed; `popqa_deciles.csv` shows what each decile holds (see
  `common/popqa.py`).
- **PopQA's decile 0** (from the dataset itself): 1,398 items with s_pop 2 to 99; ties are minor (34 items sit at the
  edge 100 and go to decile 1). It holds 20% country and 14% sport questions (5% and 2% in decile 5), whose small
  answer sets are easy to guess, which is why its baseline accuracy beats deciles 1 to 6; decile 9 holds 21%
  capital and 10% "capital of" questions.
- **P5.** Fits are interval-censored: an exit seen at check k happened in (k_lo, k]. A cell where every restart exits
  at the first check now reads as an upper bound (`floor`, `E_k_ic_hi95`) instead of E k = 250. Each restart records
  the loss and distance from W_S at its exit and at its last step, and a second reading (descended endpoint farther
  than 0.1 |W_S| from W_S, or its loss more than 0.05 nats above L(W_S); placeholders, re-thresholdable with
  `--part refit --dist-tol/--loss-tol`) is fitted next to the barrier rule. Cell reruns keep each cell's seed, so they
  repeat the full grid's trajectories. One caveat the finer checks expose: the barrier rule can fire and then clear
  between two checks, so a coarse schedule can miss a short excursion; checks.csv shows how often.
- **H3.1.** `h31.csv` flags `diverged` (train loss not finite or above ln 10), `minimum` (λ_max > 0, and λ_min ≥
  −0.01 λ_max where λ_min is measured) and `converged`; λ_max, the Hessian trace and adaptive sharpness are kept only
  for converged runs (raw values in `*_raw`), and `h31_summary.csv` averages over converged runs.

`run_queue.py` (above) runs these reruns as soon as the GPU has room for each. Every rerun writes to a new folder,
so the first results stay as they are. `python rerun_hours.py` gives the P5 bounds from this card's first-run
timings.

The floor rerun takes every cell of the first run where at least one restart exited at the first check (those
exits are known only to lie in (0, 250]); the log and `python rerun_hours.py` list the cells. It leaves out
α = 1.2 (`--alphas` restricts only the injected cells, so the SGD and clipped floor cells stay), and the α = 1.2
rerun uses the same fine checks, so every α = 1.2 cell runs once with both fixes. `--fine-check 10:250` checks at
steps 10, 20, …, 250 and then every 250 steps as before (224 checks up to k_cens = 50,000 instead of 200); a
restart's exit is never later than in the first run, so a cell never runs more steps than it did then. In the
refit the later folder wins where cells overlap. `python p5_escape.py --part h31 --out results/p5_escape_h31_v2`
reruns H3.1 with λ_min and the final gradient norm, if the flags from λ_max alone are not enough.

### Hours on the L40S

To be measured: the hours will be filled in from `logs/progress.txt` (and each job's `run.json`) after the run.

| Job | Runs on | Hours |
|---|---|---|
| `p0_calibration` | CPU, 4 threads, alongside the GPU jobs | to be measured |
| `p5_escape` | GPU | to be measured |
| `p5_escape_h31` | GPU | to be measured |
| `p2_w4a4` | GPU | to be measured |
| `p4_transformer` | GPU | to be measured |
| `p4_resnet18` | GPU | to be measured |
| `p1_pythia` | GPU | to be measured |
| `p1_polypythia` | GPU | to be measured |
| `p1_olmo` | GPU | to be measured |
| `p3_fractional` | GPU | to be measured |
| `p1_olmo7b` | GPU | to be measured |

## Setup on the Windows 11 laptop (RTX 2050, 4 GB) under WSL2

The PyPI build of torch 2.15.0 is the CUDA 13.2 build. Inside WSL2 the GPU driver is the Windows one, so update the
**Windows** NVIDIA driver to R580 or newer (R595 or newer recommended; 595.45.04 is CUDA 13.2's Linux minimum) and do
not install an NVIDIA driver inside Ubuntu.

```powershell
# Windows PowerShell as administrator
wsl --install -d Ubuntu-24.04
wsl --update
```

```bash
# Ubuntu (WSL2)
sudo apt update && sudo apt install -y python3-venv python3-pip git
nvidia-smi                                   # must list the RTX 2050 and a driver >= 580
git clone https://github.com/endriprifti/wild-randomness-pilot.git ~/pilot && cd ~/pilot
python3 -m venv ~/.venv-pilot && source ~/.venv-pilot/bin/activate
pip install --upgrade pip
pip install -r requirements.txt              # requirements-lock.txt pins every transitive package too
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash run_smoke.sh                            # unit tests + every --smoke job; summary in results/smoke_summary.csv
```

`run_smoke.sh` passes extra arguments to every job (`bash run_smoke.sh --device cpu` forces the CPU). Models and
datasets are public; set `HF_TOKEN` only to avoid Hub rate limits. Keep the code in the Linux file system (`~/pilot`),
not under `/mnt/c`, or file access becomes slow.

Kaggle: choose the T4. torch 2.15 dropped Pascal, so it does not run on Kaggle's P100. If Kaggle's driver is older
than 580, keep Kaggle's preinstalled torch and install the rest:
`grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt` (not tested here).

### Emmy environment (separate, for `emmy_check.py` only)

Emmy pins transformers 5.19.x, which conflicts with `requirements.txt`, so it gets its own venv. It needs Python
3.12 or newer (Ubuntu 24.04 ships 3.12), and `nvcc` from a CUDA toolkit new enough for the GPU (12.8 or newer for
sm_120; Ubuntu's own `nvidia-cuda-toolkit` package is older). `--frontend-only` needs neither a GPU nor nvcc.

```bash
python3 -m venv ~/.venv-emmy && source ~/.venv-emmy/bin/activate
pip install -r requirements-emmy.txt
# nvcc: CUDA toolkit 13.x from NVIDIA's repository (WSL: the wsl-ubuntu repo), e.g.
#   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
#   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit-13-2
#   export PATH=/usr/local/cuda/bin:$PATH
python emmy_check.py --frontend-only --smoke
```

## Smoke tests run here

This environment has no GPU and its network policy blocks huggingface.co and download.pytorch.org (PyPI and the
torchvision dataset mirrors work). So every smoke run here used the recorded stand-ins: a randomly initialized
GPT-NeoX of Pythia shape, a byte tokenizer, synthetic Zipfian text and random CIFAR images (Fashion-MNIST did
download). Hub loading itself is therefore not exercised here. Each stand-in is listed under `fallbacks` in that
job's `run.json`. See `SMOKE_RESULTS.md` for the table from the last run.

On the laptop (2026-11-13, with Hub access) three problems showed up and are fixed:

- **P4 vocabulary.** The model took `tok.vocab_size` (50,254 for Pythia's tokenizer), but the tokenizer emits
  50,277 IDs: its added whitespace tokens (50,254 to 50,276) occur in FineWeb-Edu text, and the GPU stopped with
  "scatter gather kernel index out of bounds". P4 now builds the model with max(config vocabulary, `len(tok)`),
  and checks before training that every token ID has a row, so a mismatch fails on the CPU with a readable message.
  The full runs are unchanged: Pythia-14M's config already has 50,304 slots.
- **The Pile copy did not load.** `monology/pile-uncopyrighted` is stored as `.jsonl.zst`, and without the
  `zstandard` package `datasets` fails with a ValueError, so P1 fell back to synthetic text (and a full run would
  have stopped). `zstandard` is now in `requirements.txt`. Its validation split was a second problem that smoke
  hid: the "Bad split" ValueError also became a synthetic-text fallback. Since 2026-11-16 a smoke run falls back
  for a dataset only when the Hub is unreachable (see "Datasets"), and no stand-in (model or dataset) covers an
  error in our own code: that rule exposed one more hidden failure, in `emmy_check.py`'s worker, whose run object
  lacked the `pins` record, so a real Emmy run would have stopped at the first model load.
- **Smoke timing.** Almost all of P3's 22 minutes went to downloading CIFAR-10 from torchvision's mirror at 130 kB/s.
  P3 and P4 now load CIFAR-10 from Hugging Face (`uoft-cs/cifar10`) first, and smoke runs stream only the 320
  images they use. P1's smoke reads fewer checkpoints (one main-grid step, H7.1 at the last stage, a four-step
  loss scan) and keeps them in the cache instead of downloading each one again. P4's smoke uses smaller
  measurement batches, because the real 50k-token vocabulary makes every Hessian-vector product costly.

## Blackwell (RTX 5090, sm_120) and library support

| Library | Used by | Blackwell | Evidence |
|---|---|---|---|
| torch 2.15.0 (PyPI, cu132) | all | yes: `torch.cuda.get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 | checked in this environment; [PyTorch 2.15 CUDA notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) (Blackwell retained; Maxwell, Pascal, Volta dropped) |
| CUDA 13.2 runtime | all | needs driver >= 580 (13.x), >= 595.45.04 for 13.2 GA on Linux | [CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html) |
| Quantization (P2) | INT4 RTN, outlier-preserving, randomized Hadamard (own FWHT), NF4 code book | yes: simulated in plain PyTorch, no custom kernels | `common/quant.py`; NF4 table checked against the QLoRA normal map in the tests |
| bitsandbytes 0.50.2 | optional, not imported | CUDA 12.8 and 13.0 to 13.2 wheels include sm120 | [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/en/installation) |
| QuaRot / fast-hadamard-transform kernels | not used | not needed | rotation is the PyTorch FWHT, so no sm_120 build is required |
| mamba-ssm, causal-conv1d | not used | not needed | P3 uses a pure-PyTorch chunked selective scan (checked against the naive scan); mamba-ssm 2.3.2.post1 is an sdist that needs a source build ([index](https://simple-repository.app.cern.ch/project/mamba-ssm)) |
| transformers 5.22.2, datasets, PyWavelets, SciPy | all | pure Python on torch, or CPU | |
| emmy-ml 0.3.25 | Emmy check | RTX 5090 is its benchmark target; also needs nvcc | [PyPI](https://pypi.org/project/emmy-ml/), [RiftStack release](https://www.ai-infra-summit.com/press-releases/riftstack-s-emmy-compiler-generates-cuda-kernels-that-beat-hand-tuned-libraries-on-nvidia-rtx-5090) |

No job has to move to the L40S because of a missing Blackwell kernel. The only L40S job is the optional last one
(P1's OLMo 2 7B suite), for memory. The L40S (Ada, sm_89) runs torch 2.15 through its sm_86 binaries; Emmy names no Ada results.

## Checkpoint suites (P1)

| Suite | Models | Purpose |
|---|---|---|
| `pythia` | `EleutherAI/pythia-{14m … 1.4b}` (seed 0) | sizes for a_N, H7.1 at the training stages |
| `polypythia` | `EleutherAI/pythia-{14m … 410m}-seed1` to `-seed9` | run-to-run variance; initialization and batch composition vary together |
| `polypythia_data` | `EleutherAI/pythia-160m-data-seed1` to `-seed3` | data order alone |
| `polypythia_weight` | `EleutherAI/pythia-160m-weight-seed1` to `-seed3` | initialization alone |
| `olmo` | `allenai/OLMo-2-0425-1B`, stage-1 checkpoints | a second model family with released checkpoints (default) |
| `olmo7b` | `allenai/OLMo-2-1124-7B` | optional, last job, L40S; never run unless named |

`seed_variance.csv` gives the spread across seeds for the joint, data-only and initialization-only 160M runs at the
same steps, with each spread's ratio to the joint one.

**Loss-spike check.** The 410M seeds 3 and 4 had loss spikes in the PolyPythias runs. The paper gives no step
numbers (Fig. 5 has ticks only at 0, 1k, 3k, 13k, 43k, 93k and 143k), and a spike that recovers between two
widely spaced reads would be missed. So the check runs in two passes, both part of `--suite polypythia`
(`--spike-only` runs just these):

- **Pass (a), loss scan.** Probe-set loss only, at every 1,000-step checkpoint (1,000 to 143,000), for seeds 3
  and 4 and one stable seed (seed 1, ours). For each flagged seed, the loss change between consecutive
  checkpoints minus the stable seed's change over the same interval removes the common downward trend. A
  jump is an increase more than 3.5 robust z (median and MAD over all of that seed's intervals) above the rest.
- **Pass (b), local reads.** α, α_ESD and K for all nine 410M seeds at every 1,000-step checkpoint from 5,000
  steps before to 5,000 steps after each jump that pass (a) finds (windows that overlap are merged). A flagged
  seed's change counts as a jump if it is larger than every other seed's change over the same interval and
  more than 3.5 robust z (median and MAD of the other seeds' changes) above them.

`spike_check.csv` holds both passes: the `pass` column (a or b), the loss-jump steps (`step_from`, `step_to`
in pass (a); `window` and `loss_jump_steps` in pass (b)) and, per row, the robust z and the `jump` verdict.
`run.json` lists the jumps under `spike_check`. `checkpoints.csv` marks each row's pass in `pass` and the
flagged seeds in `loss_spike_seed`. A spike that starts and fully recovers within 1,000 steps is still
invisible at this resolution.

**Download volume.** Pass (a) reads 3 seeds × 143 checkpoints = 429 checkpoints of 410M. At 911 MB per checkpoint
(the size of `pythia-410m`'s `model.safetensors` on the Hub; the PolyPythias checkpoints are assumed to be the
same size), that is about **391 GB** of downloads, roughly 2 hours at 50 MB/s; the compute (one forward pass on
the probe set per checkpoint) is minutes. Pass (b) adds up to 9 seeds × 12 checkpoints = 108 checkpoints, about
98 GB, per jump window. Every checkpoint is deleted from the Hugging Face cache right after it is read, so the
disk holds about one checkpoint at a time (`--keep-cache` turns this off). `--pin-only` reads the exact file sizes
from the Hub and writes the total for the main grid and for pass (a) to `run.json` (`results.download`).

### P1's GPU pass, CPU statistics in a pool, and results after every checkpoint

Each checkpoint has a GPU pass in the main process (load, probe loss, the SVD of every layer, hidden states and
activation moments, token losses, the JVP projections and the H7.1 pool, eval loss) that saves what the statistics
need to `checkpoints/<pass>/<repo>@<revision>/samples/` and writes `row.json`. The statistics run in a pool of
processes (`common/p1_tasks.py`) while the GPU goes on to the next checkpoint: the stable fit and CSN tails with
Vuong tests of the gradient noise, α_ESD per layer, act_zeta per layer, Δh_rep (on the CPU), H_tok with its
shuffled surrogates, and per H7.1 direction the fits at each B and the bootstrap in chunks of 25 replicates. A
checkpoint whose statistics are all saved gets `result.json`, and `checkpoints.csv` and `h71_summary.csv` are
rewritten from all of them (the per-layer and per-B CSVs at most every 10 minutes and at the end).

- **Resume.** Run the same command again: checkpoints with `result.json` are read back, checkpoints with only
  `row.json` run their missing statistics, and the rest are measured. A crash or a stop loses at most the GPU pass
  in progress and the statistics then running. Every statistic has its own random generator, seeded from the run
  seed, the checkpoint and the statistic's name, so the results do not depend on the number of processes, the
  order in which they finish, or a restart (a killed-and-resumed smoke run gives the same CSVs as an uninterrupted
  one). The bootstrap draws differ from runs before 2026-11-16, which used one generator for the whole run.
  `checkpoints/fingerprint.json` records the config, seed, `--skip` and stable density; a run with other settings
  refuses to resume in that folder.
- **CPUs.** One pool process per CPU the job may use (its `taskset`), at lower priority than the GPU pass, one
  thread each. `--cpus-after p3_fractional=all` (or `P1_CPUS_AFTER`, which `run_lane.sh` sets) widens this to every
  CPU once `logs/progress.txt` shows P3 ended; a file `logs/p1_cpus` (e.g. `0-10`) sets the CPUs at any time and is
  read every minute. `--workers N` fixes the number of processes.
- **Progress.** `logs/<job>.log` gets a line per GPU measurement (seconds, and seconds into the checkpoint), per
  finished statistic (seconds, tasks queued and running) and per completed checkpoint (n of N); `timings.csv` holds
  the same seconds per checkpoint and measurement.
- **Samples kept.** The JVP projections, the H7.1 pools, the token-loss series and the ESD eigenvalues stay (about
  2 MB per checkpoint, 20 MB per Pythia-14M H7.1 pool); the hidden-state points and activation samples are deleted
  once their statistic is saved (`--keep-samples` keeps them).
- **H7.1 stages.** Each stage is the one checkpoint nearest to 1, 10, 50 or 100% of training (steps 1,000, 14,000,
  71,000 or 72,000, and 143,000 for Pythia). Before 2026-11-16 a checkpoint counted as a stage when its fraction
  of training was within 0.02 of a stage, which made every Pythia step up to 4,000 a 1% stage and 16,000 a 10%
  stage: ten H7.1 tests per size and direction instead of four.
- **Stable density.** The fits use a density precomputed on an (α, β) grid (`--stable-density grid`, default;
  `common/stable.py`, built once in about 35 s into `.cache/`, about 310 MB, memory-mapped by every process) and
  one Nelder-Mead restart: about 0.09 s per fit instead of 1.5 s. On 240 simulated samples (n = 200 and 1,600;
  α 1.2 to 2, β up to ±0.6, Student t3, Gaussian) α̂ agreed with the FFT density's fit to 0.0008 in 99% of fits;
  the two larger differences (0.010 and 0.028) are fits where the FFT fit's Nelder-Mead stopped short (the grid
  fit's likelihood is higher). Two full H7.1 tests (1,600 blocks, B = 8 to 1,024, 50 bootstrap replicates; SαS
  1.8 and Student t3 pools) agreed to 1e-5 in every α̂(B), in b and in its interval, 17 times faster.
  `--stable-density fft` restores the FFT density without the restart.

Stopping and restarting the lane with new code (the lane stops its job with SIGTERM, which stops the pool, and
writes END with exit 130):

```bash
kill -TERM "$(pgrep -f 'bash run_lane.sh')"     # stops the job; its finished checkpoints stay on disk
# pull as in "Pulling on the L40S" below, then start the lane again: finished jobs (.done) are skipped and the
# interrupted one resumes from its checkpoints
nohup taskset -c 7-10 bash run_lane.sh p1_pythia p1_polypythia p1_olmo > logs/run_lane.out 2>&1 &
```

### Batches in random order, Δh_rep and the H7.1 verdict in P1 and P4 (2026-11-17)

Endri's check of P1's first 42 checkpoints: in most H7.1 rows b lay outside its own 90% bootstrap interval
(pythia-14m step 1,000, random0: b = −0.0115, interval [−0.0014, 0.0024]). The example pool is consecutive
128-token chunks of the same documents, and the main fit cut it into consecutive blocks, so a batch held
neighbouring chunks of a few documents; the bootstrap resamples the pool i.i.d., so its interval describes random
batches, another statistic. The coupling α had the same problem (P1's `grad`: blocks of 8 consecutive chunks; P4:
the first 1,600·B training sequences in order).

- **Fix.** P1's `grad` and H7.1 main fit put the pool in a fixed random order before cutting blocks
  (`common/batch.py`: `shuffled`, seeded by the run seed only, so every checkpoint and direction uses the same
  order), as SGD draws random batches. The bootstrap is unchanged: it resamples i.i.d. with replacement, so the
  pool's order does not change its distribution, and its replicates are exactly those computed before. On
  document-like pools (documents of 16 chunks sharing a level, Student t values), the shuffled main fit lay inside
  its 90% interval in 17 of 20 pools and the consecutive one in 5 of 20 (`tests/test_batch_order.py`). P4 draws
  its α pool without replacement, in random order, from the whole training set (`pool_ids`; the same examples for
  every arm and seed). `alpha_batch_order` in `checkpoints.csv` and P4's `runs.csv`, and `batch_order` in
  `h71_summary.csv`, say which order a row used. The bootstrap still treats chunks as independent; chunks of one document are not, so the interval can
  be somewhat narrow (a bootstrap by document would need the document of each example, which the samples do not
  keep).
- **Not affected.** P4's H7.1 control draws random images (`torch.randint`), so its batches were random already,
  which is why ResNet's b mostly lay inside its interval. `p4_resnet18`'s α pool was CIFAR-10's training images in
  file order, which the dataset documents as random. P5 draws its 10⁴ Fashion-MNIST images by a random permutation;
  the P0 and P0b pools are i.i.d.
- **Δh_rep.** `dh_rep` is NaN when the dimension clause fails (D_max ≥ log10 n, or fewer than two decades of r) or
  the width is negative (h(q) cannot increase in q); `dh_rep_raw` keeps `delta_h_rep`'s number and `dh_rep_note`
  says why (P1 and P4). In `p4_resnet18`'s `runs.csv`, written before, read `dh_rep` as NaN where `dh_scope` is
  "local" (the clause failed) or `dh_rep` < 0.
- **Verdict.** P1's and P4's `h71_summary.csv` carry the verdict columns of "H7.1 verdict" above. The slope-only
  `reading` is dropped there; `slope_reading` says the same thing.
- **What P1 keeps.** `grad.npy` (the coupling α pool), `h71.npy` (the H7.1 pools), `htok.npy` and `esd.npz` stay on
  disk, so α and the H7.1 fits can be recomputed on the CPU without the GPU pass. The bootstrap, α_ESD, act_zeta,
  H_tok and Δh_rep are kept as computed: the Δh_rep rule needs only the saved numbers (`dh.npy` and `act.npz` are
  deleted after use).

**A P1 job that runs on with the earlier code** (p1_pythia, started before this commit) writes the old order to the
end. Pulling does not disturb it: it keeps the code it loaded, and pool processes it starts after the pull (when its
CPUs change) run the new code but compute its tasks as before, since those carry no order (a test checks this).
`p1_recompute.py` then computes its statistics again into another folder, CPU only, at nice 19, reading the job's
folder without changing it:

```bash
python p1_recompute.py results/p1_pythia --dry-run    # finished checkpoints, tasks, and the CPU hours they took there
nohup taskset -c 0-10 python p1_recompute.py results/p1_pythia > logs/p1_recompute_pythia.log 2>&1 &
# once p1_pythia has ended, the same command again adds the checkpoints finished since (done ones are kept)
```

It writes `results/p1_pythia_recomputed/`: per checkpoint `row.json` with the order added, `stats/` (grad and the
H7.1 main fits computed again from the job's samples with the job's task seeds, everything else copied) and
`result.json`, and the job's CSVs (`checkpoints.csv`, `h71_summary.csv` with the verdict, `h71_per_B.csv`, the
per-layer CSVs, `scaling.csv`, `seed_variance.csv`, coupling and spike files when the job has them, `run.json`).
They equal what the current `p1_checkpoints.py` gives on those checkpoints (`tests/test_batch_order.py`; and a smoke
run measured with the previous commit and recomputed gave the same CSVs as one measured with this commit). Jobs started after the pull (p1_polypythia, p1_olmo, p1_olmo7b, and the queue's p4_transformer) use the
random order from the start. A job started with this code on a folder measured before upgrades it on resume: the
grad and H7.1 main-fit statistics of each checkpoint are computed again from its samples, the old ones moved to
`stats/consecutive/`, and `fingerprint.json` records `upgraded_from`.

### Datasets

Every Hub dataset is read through `common/hf.py` at a commit pinned in `pins.json`, like the models (key
`datasets/<repo>@main`, pinned on first use; `--require-pins` applies). Each config entry names the repository,
config, split and, where the builder's own split detection is not enough, the files of the split (`data_files`).
`monology/pile-uncopyrighted` needs them: the builder finds only its `train/` folder and misses `val.jsonl.zst` and
`test.jsonl.zst` at the top level, so the validation split (P1's evaluation, probe and surprisal text, in every
suite, the loss-spike passes and OLMo 2 7B) failed on the L40S with "Bad split: validation" (2026-11-16). P1 now
reads `train/*.jsonl.zst` for the gradient pool and `val.jsonl.zst` for validation.

| Dataset | Splits | Read by |
|---|---|---|
| `monology/pile-uncopyrighted` | train (`train/*.jsonl.zst`), validation (`val.jsonl.zst`) | P1 (every suite) |
| `Salesforce/wikitext` `wikitext-2-raw-v1` | test, train | P2 (perplexity, calibration), P4 (out-of-distribution text), Emmy check |
| `akariasai/PopQA` | test | P2 |
| `HuggingFaceFW/fineweb-edu` `sample-10BT` | train | P4 transformer |
| `uoft-cs/cifar10` | train, test | P3 (LRA image), P4 ResNet-18 |
| Fashion-MNIST (torchvision) | train, test | P5 |

`python check_data.py` (a few minutes, CPU) streams the first row with a non-empty text field of each of these
splits (row 1 for WikiText, whose row 0 is a blank line; the jobs skip blank rows the same way) exactly as the job
reads it, pins new dataset revisions in `pins.json` (`--no-pin` leaves the file alone), writes
`results/check_data/check_data.csv` and exits 1 if any split fails. The list comes from `config.py`, so a new
dataset entry is checked without editing the script.

The dataset pins in the committed `pins.json` come from `check_data.py` on the laptop (2026-11-16): the Pile copy
at `3be90335b6`, WikiText at `b08601e043`, PopQA at `098765c79e`, FineWeb-Edu at `87f09149ef`, CIFAR-10 at
`0b2714987f`.

### Pulling on the L40S without losing local pins

The L40S's `pins.json` gains names during the run (pinned on first use), so `git pull` refuses to overwrite it, or
would conflict with a committed update. Save it, pull, and merge it back (`merge_pins.py` keeps every entry of both
files; a name with two different commits keeps the local one, which that machine's results loaded, and is
reported):

```bash
cd ~/pilot
F=~/pins.local.$(date -u +%Y%m%dT%H%M%SZ).json
cp pins.json "$F"
git checkout -- pins.json && git pull --ff-only
python3 merge_pins.py "$F"                   # standard library only; exit code 2 if a name had two commits
```

A job that is running keeps its pins in memory and merges them into the file on disk when it saves, so pulling
between or during jobs loses nothing; `run_all.sh` is not changed by these commits, so the running script is safe.

### Pinned commits

Every Hub model (Pythia, PolyPythias, OLMo, P2's models and P4's tokenizer) is loaded at a commit hash, never at a
branch name. `pins.json` maps `repo@revision` to the commit and its date. The first run that meets a new name
resolves it on the Hub and writes it to `pins.json`; commit that file, and every later run loads exactly those
commits. `--require-pins` makes a job fail instead of resolving a name that is not pinned yet.
`python p1_checkpoints.py --pin-only` pins every P1 checkpoint of the main grid and of pass (a) without measuring
anything (a CPU is enough; set `HF_TOKEN`, since this is about 1,100 names and one Hub API call each; rerun it if
some fail, already pinned names are skipped). Pass (b)'s checkpoints depend on what pass (a) finds, so they are
pinned when pass (b) runs: run it without `--require-pins` and commit the updated `pins.json` afterwards. Each
`run.json` lists the commits it loaded under `pinned_commits`, and `checkpoints.csv` has `commit` and `commit_date`
columns.

The committed `pins.json` comes from `--pin-only` on 2026-11-13: 1,067 commits for the main grid and pass (a)
(OLMo 2 7B and pass (b) are not pinned yet). By the Hub's file sizes, the main grid is 654 checkpoints and 367.7 GB,
pass (a) is 429 checkpoints and 391.0 GB, and all 1,067 checkpoints together are 739.9 GB (the two passes share
some checkpoints). A second `--pin-only` run at a492897 added the name-switch content check (below) for the 26
pinned `pythia-14m`/`pythia-31m` checkpoints of P1: every one differs from its `-deduped` twin, and every one is
dated after the switch. The hashes and `-deduped` commits are in each entry's `deduped_check`.

This matters for Pythia. On 24 February 2026 the `pythia-14m` and `pythia-31m` repositories were re-created with
the standard-Pile models: `pythia-14m` step143000 has three commits, all from that day ("initial commit",
"Add model card", "Upload checkpoint step 143000"), `pythia-31m`'s history also starts with an "initial commit"
that day, and every pinned step commit of both is dated 24 February 2026. The 2023 deduplicated upload moved, with its history, to `pythia-14m-deduped` and `pythia-31m-deduped`. (27 February is
when main's model card announced the switch.) For these two names P1 runs two checks:

1. Content, independent of dates (`dedup_check`): the LFS sha256 of `model.safetensors` at the pinned commit
   against the same revision (step branch) of the `-deduped` repository. Both hashes come from the Hub API's file
   metadata, so nothing is downloaded. Only identical files are flagged ("CHECK: … identical to …-deduped"):
   those are the deduplicated weights. "differs" means the files are not byte-identical, which is all a hash can
   say. If either side lacks `model.safetensors`, `pytorch_model.bin` is compared instead when both carry it;
   otherwise the check reads "unknown". The result is cached in the pin entry (`deduped_check` in `pins.json`,
   with the `-deduped` commit and both hashes), so the committed file carries the evidence.
2. Date (`name_switch_check`): any commit dated before 2026-02-24 is flagged.

`checkpoints.csv` carries both columns, and a flag of either kind also goes to the `run.json` notes. `--pin-only`
runs both checks for every pinned `pythia-14m`/`pythia-31m` name (27 in the committed file, two Hub API calls each)
and writes `name_switch_check.csv`; `run.json` counts the verdicts under `name_switch`. New pin entries also record
`weights_sha256`, the sha256 of the weights files at that commit.

## Emmy

What is confirmed:

- Emmy exists: `emmy-ml` on PyPI (CloudRift / RiftStack, Apache-2.0, alpha, 0.3.25 released 2026-11-13). It
  compiles PyTorch graphs to CUDA kernels through `torch.compile(model, backend="emmy", dynamic=False)`.
- Nothing in its README, recipes or source names GPT-NeoX or Pythia. Its recipes are Llama, Qwen, Gemma, GLM,
  DeepSeek and similar models.
- CPU frontend probe (`--frontend-only`, emmy-ml 0.3.25, transformers 5.19.0, random-init GPT-NeoX of Pythia shape):
  - `torch.compile` on `GPTNeoXForCausalLM` as it is fails in Emmy's frontend with "no tracer mapping for
    aten.new_ones" (and "aten.diff" without an attention mask). Both ops come from transformers' mask construction,
    not from the GPT-NeoX layers.
  - Through a static wrapper (Emmy's own full-model wrapper adapted to GPT-NeoX: precomputed causal mask, position
    ids and rotary cos/sin), the whole forward traces into Emmy's IR. Running that IR through Emmy's torch
    reference reproduces the eager logits exactly.
  - Bug found: when `seq_len` equals `hidden_size`, Emmy's CPU evaluators (`GatherOp.forward`, `torch_ref`)
    evaluate the embedding lookup as `torch.gather`, so the output has the wrong shape and the fused-QKV view after
    it fails. The script keeps `seq_len` off every model dimension (500 in the config). A draft issue with a
    standalone reproduction is in [`docs/emmy-issue-seq-len-hidden-size.md`](docs/emmy-issue-seq-len-hidden-size.md)
    (script: `docs/emmy_seq_eq_hidden_repro.py`, CPU only, run it in the Emmy environment). Emmy's GPU lowering
    decides by output rank and is probably not affected, but that is untested.

The comparison runs twice per model and dtype, with Emmy's fast math on (its default: `EMMY_FAST_MATH` unset)
and off (`EMMY_FAST_MATH=0`). Emmy lets single precision knobs (`EMMY_FAST_EXP`, `EMMY_FAST_DIV`, …) override
fast math, so the script removes any such variable inherited from the shell and records the lane in the
`EMMY_FAST_MATH` column of `compare.csv` and `support.csv`.

Still unconfirmed: kernel lowering and numerical agreement on a GPU. Lowering the whole model on a CPU did not
finish here. So whether Emmy supports GPT-NeoX end to end is open until `python emmy_check.py --probe-only` runs on
a GPU with nvcc. The comparison reads "agrees" when top-1 agreement is at least 0.99, |ΔNLL| is at most 0.01 nats
and the outlier-channel Jaccard is at least 0.9 in every layer. These thresholds are ours; change them in `config.py`
before the run.

## Unit tests

`python -m pytest -q` (203 tests, about 6 min on a 4-core CPU; `-m "not slow"` skips the slowest). Synthetic data with
known parameters:

- stable density against the Gaussian and Cauchy closed forms and against SciPy; ML fit recovers α ∈ {1.2, 1.5, 1.8},
  β, γ and δ; the Gaussian boundary; the bootstrap LR test against the Gaussian;
- CSN: ζ on Pareto, x_min in a body-plus-tail mixture, p_KS on a power law; Vuong preferring the lognormal and the
  exponential on their own samples; Hill double bootstrap on Pareto ζ = 2 and Student-t (ζ = 3); the six labels,
  including "undecided" for ζ = 3 at n = 2·10^4 (the two-decade rule);
- Hurst: DFA and wavelet on fGn H ∈ {0.3, 0.5, 0.7}; shuffled baseline on Lévy increments (no memory) and fGn 0.8;
  phase surrogates keep the spectrum, IAAFT keeps the marginal; MFDFA width on the binomial series and on fGn;
- D_2 and Δh_rep on binomial cascades, the uniform square, and the dimension clause in 50 dimensions;
- H7.1 slope on SαS pools (invariant) and a truncated Lévy pool (drift); the 1,600-block arithmetic;
- H7.1 verdict: each branch (wild, mild, drift, and undecided for the slope, for the scale exponent and without a
  Gaussian reference); flat at 2 is not read as invariant; the 1.9 margin leaves ᾱ near 2 undecided (and 2 turns it off); the Gaussian reference is seeded per pool, tops up only
  what is missing and is matched to each row's setting; the command line recomputes verdicts from a saved summary;
- coupling test, Holm, detectable r; censored escape rates and the (η, B) power-law fit against the Gaussian rival;
- interval censoring: with checks every 250 steps the exposure estimate sits at the check (E k = 250) while the
  interval-censored MLE recovers E k = 150; a cell where every restart exits at the first check gives only an upper
  bound; on a grid with many first-check exits the interval fit recovers both slopes and the exact-time fit does
  not;
- P5 reruns: a finer check schedule sees each exit no later than the coarse one and leaves exits after the fine
  window unchanged (the cell's seed repeats its trajectories); saved runs get k_lo and g_max quantiles; floor
  cells (at least one first-check exit; α = 1.2 left to its own rerun); H3.1 flags (diverged above ln 10,
  λ_max < 0 or λ_min < −0.01 λ_max is not a minimum, sharpness kept only for converged runs);
- dataset loading: a folder laid out like `monology/pile-uncopyrighted` reproduces "Bad split: validation" and
  reads `val.jsonl.zst` with `data_files`; only transient errors are retried (a ValueError is tried once); a smoke
  run falls back for a dataset only when the Hub is unreachable; `check_data.py` lists every split and smoke reads
  the same ones; no `load_dataset` call outside `common/hf.py`; blank WikiText rows are skipped by `load_texts`
  and by `check_data.py`; `merge_pins.py` and the pull recipe keep every local and committed pin;
- P1's pool: one H7.1 checkpoint per stage (four for Pythia-14M, where the old rule gave ten); the pool's CPUs
  (the start set, widened after a job's END, and `logs/p1_cpus`); task seeds depend only on names; the tasks'
  results merge into the rows and columns written before, a failing statistic is recorded and a bug stops the
  run; a pool runs a checkpoint's tasks and a restart reads `result.json` back; other settings refuse to resume;
- the grid stable density against the FFT density (log f within 1e-4 for |x| <= 50) and its fits against the FFT
  fits (α̂ within 0.01 and γ̂ within 1%, unless the grid fit's likelihood is higher);
- `run_lane.sh`: run_all.sh's job order and its P1 and P3 commands; its records next to a running run_all.sh (a
  stand-in Python), refusal of jobs run_all.sh will still run and of other jobs, one lane of each kind at a time;
  P3 in its own lane next to a P1 lane, refused while run_all.sh still runs P3 or with P1 jobs in the same lane;
  P1's GPU lock makes a second P1 job wait; olmo7b waits for the lanes and for P3's lock (and not for its own
  lane), other suites do not; two jobs saving `pins.json` keep each other's entries;
- P3: the LRA designs' run counts and steps (31,240 per run in the first plan, 4,686 in the reduced one; no main
  run repeats a tuning run; main runs seed by seed); the synthetic data are the old code's draws whatever tasks
  run; saved runs are reused and deleted ones run again with the same results; other settings in the folder stop
  the run; P3's GPU lock;
- resume of P2, P4 and P5 (`common/units.py`): smoke runs with units deleted (P2's baseline and a scheme, P4's
  unclipped reference run and a matched run, two P5 cells) run only those again and give the same CSVs; P2 with
  every unit saved does not load the model; other settings in the folder stop the job; P4's token IDs come back
  from the cache; NF4 in chunks equals NF4 in one pass, and P2's quantization frees each original layer as it
  replaces it;
- `run_queue.py`: the admission rule (only the P5 reruns next to both lanes; P3 next to P2 or P4, not both; a lane
  with a job to start keeps its peak plus 4 GiB; nvidia-smi has the last word), every set `--combos` lists fits
  with room for the lanes, the peaks agree with `run_lane.sh` and `memory_check.py p5`; with stand-in jobs: records
  in run_all.sh's format, `.done`, archives, a dependency waits and a failed one stops its dependents, one retry, a
  missing input; a 25 GiB job waits while a P1 lane has a job to start and while olmo7b holds the GPU, then
  starts; olmo7b waits for the queue's lock; SIGTERM stops the jobs and records them, a restart goes on; a second
  queue and a job left by an earlier queue are refused;
- P2's outlier scheme: a kept channel 1000 times larger leaves every other channel's activation error, the other
  weight columns and the layer's output error unchanged (before the fix it rounded most of them to zero); the
  rotation arm in bfloat16 adds no rounding beyond bfloat16 storage;
- PopQA trends: the decile, relation-standardized and relation fixed-effect slopes recover a planted slope of the
  drop on log popularity, and the last two remove a spurious trend made by piling an easy relation into the rare
  deciles;
- per-example JVP against autograd, Hessian eigenvalues and Hutchinson trace on a quadratic;
- micro-batches: on a small GPT-NeoX, the HVP, Hutchinson trace, top eigenvalue, adaptive sharpness and
  per-example projections computed in micro-batches (1, 3, 4 and 10 sequences of 10, uneven last part included)
  match the full batch to float32 rounding (the loss is cast to float32); tuple batches on an MLP;
- `memory_check.py`'s tracker: live bytes on the meta device, with freed tensors leaving the count;
- NF4 code book, FWHT, rotation invariance, RTN error bound, matched bits, fixed-rate clipping;
- chunked selective scan against the naive scan (also at the shorter chunk that larger states get), the fractional kernel against the Mittag-Leffler impulse
  response t^(ν−1) E_{ν,ν}(−λ t^ν), and parameter matching across SSM arms;
- tail labels: alternatives that cannot be separated read "undecided"; exponential against Weibull uses the
  likelihood-ratio test;
- P1 bookkeeping: commit pins (reuse, --require-pins, merge on save), the weight bytes counted for the download
  volume, step choice for the main grid, the 1,000-step scan and OLMo, the pythia-14m/31m name-switch checks
  (date: 23 February 2026 is flagged, 24 February is not; content: identical weights to the `-deduped` revision are
  flagged, different ones are not, a missing file or branch reads "unknown", and the result is cached in
  `pins.json`), both loss-spike passes (each finds a planted spike at its onset, and only
  there), window merging, and the seed-variance table;
- P4's vocabulary: a tokenizer whose added tokens lie above `vocab_size` gets a model with a row for every ID, in
  smoke and full settings; the ID check fails readably; the full model keeps Pythia's 50,304 slots;
- P0b's helpers: random-order cascades keep the binomial values at every dyadic level, truncated-Lévy pools are
  signed and cut, task seeds do not depend on order, and the shared-null LR statistic and p-value match
  `lr_test_gaussian`;
- the GPU stable sampler against the NumPy one (same distribution for α from 1 to 2, skewed or not; reproducible
  from its seed);
- the Emmy check's streaming kurtosis and logit metrics, and the fast-math lanes' environments;
- H_link: the synthetic series (fGn's H, the MRW's heavy marginal, the cascade positive), p-values with ties, the
  main and sensitivity rows on the same surrogates (a 9,064-step series has two decades from 8 and not from 10, is
  flagged and cannot pass there; the wavelet columns agree), the MRW's magnitudes above all shuffled surrogates,
  i.i.d. Gaussian magnitudes near 1/2, a matrix's statistics as the mean over its columns, and `h_link.py` on stand-in
  P1 and P4 folders: only finished checkpoints and runs, warm-up cut as for H_grad, `*_recomputed` and empty
  folders skipped, the folders left unchanged, resume without recomputing, a changed file computed again, short
  series skipped, diverged runs left out of the summary.

## Deviations, choices of ours, and items to verify

- **H_grad in P1.** Public checkpoints have no optimizer trajectory, so P1 measures H on token-surprisal series
  (`Htok_*`, exploratory). H_grad comes from P4's training runs.
- **CIFAR-10 blocks.** 50,000 images cannot give 1,600 disjoint blocks at B > 31, so the H7.1 control in P4 draws
  augmented copies and saves the base-image IDs.
- **Two-decade rule.** A Pareto ζ = 3 sample needs n well above 10^5 to span two decades in x, so smaller samples read
  "undecided".
- **Δh_rep** is usually "local" for high-dimensional representations, because D_q exceeds log10 n (open question 2).
- **P5's injected noise is drawn on the GPU** (`stable.rvs_stable_torch`, the same Chambers-Mallows-Stuck formula as
  the NumPy sampler, computed in float64). Drawing it with NumPy took about 0.35 s per step on 4 CPU cores (six
  million variates for 50 batched restarts), which would have made the CPU, not the GPU, set the pace of the
  injected arm. A unit test checks that both samplers give the same distribution.
- **QuaRot** is simulated with RTN (no GPTQ). Dimensions that are not powers of two (OLMo's 11008) use a Haar rotation.
- **SSM arms.** S4 uses a dense LegS-initialized A instead of the NPLR form. Mamba is a pure-PyTorch scan.
  The fractional kernel spans r ∈ [1/(4L), 2]. It matches the Mittag-Leffler impulse response within 1% from t = 10
  time units, but is 10 to 25% low at t = 1 because modes above r = 2 are cut.
- **Tail labels.** When more than one alternative beats the power law, they are tested against each other: Vuong
  for the non-nested pairs, and the likelihood-ratio test (χ², 1 df) for exponential inside Weibull. A label is
  assigned only when the tests single out one model, or when every surviving model carries the same label (for
  example lognormal and Weibull with w < 1, both "slow"). Otherwise the reading is "undecided", following Clauset,
  Shalizi and Newman, and `tail_reason` says which models could not be separated. No information criterion picks
  a winner. Exponential and Weibull samples still sometimes read "truncated" instead of "mild" or "slow"; every
  such label is non-wild.
- **P0** uses 250 bootstrap replicates instead of 2,500, to stay on a CPU.
- **Placeholders for the preregistration**, marked OURS in `config.py`:
  - hill_tol 0.15, the q range 0.5 to 4, min_tail 50 and the x_min candidate grid;
  - coupling_B 8, the barrier tolerance, η values and g_max quantiles;
  - the loss-spike check's stable seed (seed 1), robust-z threshold 3.5 and ±5,000-step windows;
  - the Emmy tolerances.
- **[VERIFY]** items:
  - `monology/pile-uncopyrighted` stands in for the Pile (P1's gradient pool and evaluation text);
  - the ZClip update rule (`common/clipping.py`).

## License and citation

MIT (see `LICENSE`). To cite this code or the paper it accompanies, use `CITATION.cff` (GitHub's "Cite this
repository" button reads it).

## Acknowledgement

Compute for this pilot is provided by a compute grant from [CloudRift](https://www.cloudrift.ai).
