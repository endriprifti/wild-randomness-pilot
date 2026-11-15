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
| 3 | W4A4 and weight-only, H6.2 (simulated quantization) | 30 | 5090 (Pythia-6.9B and OLMo-7B in bfloat16, about 14 GB each); L40S also fine | `python p2_w4a4.py` (one model: `--models EleutherAI/pythia-6.9b`) |
| 4 | Clipping sweep, H6.1 / H7.1 control | 30 | 5090 (transformer peak ~21 GiB, see GPU memory below); ResNet-18 also T4 | `python p4_clipping.py --model transformer` and `python p4_clipping.py --model resnet18` |
| 5 | Checkpoint measurements, with the 410M loss-spike check (pass (a) downloads about 391 GB, see below) | 45 | Pythia: 5090 (H7.1's HVP at 14M to 160M peaks ~16 GiB); PolyPythias: T4; OLMo 2 1B: 5090 | `python p1_checkpoints.py --pin-only`, then `--suite pythia`, `--suite polypythia,polypythia_data,polypythia_weight`, `--suite olmo` |
| 6 | Fractional memory, H4.1 | 65 | 5090; T4 is tight (Mamba at n_s = 64 peaks ~13 GiB) | `python p3_fractional_memory.py` (drop LRA first if time runs short: `--tasks powerlaw,exponential`) |
| 7 | **Optional, last:** OLMo 2 7B checkpoints | not in the 45 (ours: ~10) | **L40S** (fp32 weights plus per-example JVP exceed 32 GB) | `python p1_checkpoints.py --suite olmo7b`, only if credits remain |

Jobs 0 to 6 all fit the RTX 5090. Job 7 is the only one that needs the L40S, and nothing else depends on it.
Stage 1 runs on an L40S with `run_all.sh` (below).

Every script takes `--smoke`, `--seed N`, `--device cuda|cpu`, `--out DIR` and `--offline-ok` (allow the recorded
stand-ins outside smoke mode; off by default, so a real run fails loudly if a download fails). Run the Emmy check
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
refitting and without changing the cloud run's code. On the laptop, after unpacking the archives from the L40S:

```bash
python -m common.h71_verdict results/p0_calibration/block_check_bootstrap.csv \
    results/p1_pythia/h71_summary.csv results/p4_resnet18/h71_summary.csv \
    --null results/p0b_calibration/h71_gauss_null.csv --jobs 6
```

It writes `<name>_verdict.csv` beside each file (the original columns, including the slope-only `reading`, plus
`alpha_null_q`, `s_null_lo`, `s_null_hi`, `null_reps`, `slope_reading`, `level_reading`, `scale_reading`,
`verdict` and `verdict_reason`) and prints the counts. The batch sizes come from the `alpha_B<B>` columns (P0b),
from `h71_per_B.csv` beside the file (P1, P4) or from `config.BATCH_SIZES` (P0; `--batch-sizes` overrides). A
setting missing from the Gaussian reference (for example a pool that came out short) is simulated on the spot
and added to the `--null` file (400 pools; about 1 h on 6 cores at 1,600 blocks).

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
  interrupted jobs run again from their start (a job has no checkpoints of its own). Delete `logs/<job>.done` to
  rerun a finished job. `--only job1,job2` and `--skip job1,job2` select jobs, and only one `run_all.sh` can run at
  a time. Stopping the script (Ctrl-C, `kill`) stops the running GPU job and P0 and records both as interrupted.
- **Archives.** After each job (also after a failure) `results_archive/<job>.tar.gz` holds `results/<job>/` and
  the job's log, and `results_archive/pins.json` is refreshed. At the end the script prints the `scp` line to run
  on the laptop, `scp -P <port> '<user>@<host>:<path>/results_archive/*' .`, built from `PILOT_SSH` and
  `PILOT_SSH_PORT` (kept out of the repository).
- **Pins.** The jobs run without `--require-pins`: a name that is not pinned yet (pass (b)'s checkpoints, OLMo 2
  7B) is resolved on first use and written to `pins.json`, so no job fails for that. At the end the script says
  how many names were added; upload `results_archive/pins.json` so it can be committed.
- **Network.** In a real run, a Hub call or download that fails for a transient reason (dropped connection, rate
  limit, server error) is retried four times, 10 s to 270 s apart, before the job gives up. A missing repository,
  revision or file fails at once.
- **Output folders.** Each job writes to `results/<job>/` (`--out`), so the three P1 suites and the two P4 models do
  not overwrite each other.

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

Run on the L40S once `run_all.sh` has finished, one GPU job at a time. Every rerun writes to a new folder, so the
first results stay as they are:

```bash
cd ~/pilot && source ~/.venv-pilot/bin/activate && git pull
python rerun_hours.py                                   # expected hours, from this card's first-run timings
# CPU only, minutes: refits and flags from the saved first-run files
python -m common.popqa results/p2_w4a4 --out results/p2_w4a4_popqa
python p5_escape.py --part h31-flags --base results/p5_escape_h31 --out results/p5_escape_h31_flags
# GPU, one after another
nohup sh -c '
python p5_escape.py --floor-from results/p5_escape --alphas 1.5,1.8,2.0 --fine-check 10:250 --out results/p5_escape_floor10 > logs/rerun_p5_floor10.log 2>&1
python p5_escape.py --modes injected --alphas 1.2 --fine-check 10:250 --second-reading --log-checks --out results/p5_escape_a12 > logs/rerun_p5_a12.log 2>&1
python p5_escape.py --part refit --base results/p5_escape --replace results/p5_escape_floor10,results/p5_escape_a12 --out results/p5_escape_refit > logs/rerun_p5_refit.log 2>&1
python p2_w4a4.py --out results/p2_w4a4_bf16 > logs/rerun_p2_bf16.log 2>&1
' > logs/reruns.out 2>&1 &
# afterwards
tar czf results_archive/reruns-2026-11-15.tar.gz results/p2_w4a4_bf16 results/p2_w4a4_popqa results/p5_escape_floor10 \
    results/p5_escape_a12 results/p5_escape_refit results/p5_escape_h31_flags logs/rerun_*.log
```

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
  have stopped). `zstandard` is now in `requirements.txt`.
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

`python -m pytest -q` (139 tests, about 2 min on a CPU; `-m "not slow"` skips the two slowest). Synthetic data with
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
- the Emmy check's streaming kurtosis and logit metrics, and the fast-math lanes' environments.

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
