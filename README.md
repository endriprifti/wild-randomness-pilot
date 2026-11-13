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
| `p5_escape.py` | H3.3, H6.1 (escape clause), H3.1 (injected arm) | escape times over (η, B), clipped and injected-noise arms; censored likelihood fits |
| `p2_w4a4.py` | H6.2 | W4A4 and weight-only arms at matched bits, PopQA by popularity decile, isotropic-noise control |
| `p4_clipping.py` | H6.1, H7.1 control, H_grad | clipping sweep (fixed p_clip, AutoClip, ZClip, AGC), matched update length, flatness and tail measures |
| `p1_checkpoints.py` | α, α_ESD, K, H, Δh across Pythia, PolyPythias, OLMo | measurement only on public checkpoints; H7.1 at training stages; exploratory coupling |
| `p3_fractional_memory.py` | H4.1 | fractional block against S4 (LegS), S4D and Mamba at matched parameters and equal tuning budget |

Notation follows the paper: ζ is the fitted Pareto (survival) exponent, α = min(ζ, 2) only for the stable index.

**Quantization is simulated.** P2 (and the Emmy check's comparison) quantizes and dequantizes weights and
activations in plain PyTorch and then runs the ordinary floating-point kernels. No INT4 kernel runs, so the results
concern accuracy (perplexity, PopQA, activation statistics) and say nothing about speed or memory savings.

## Jobs, cheapest first

GPU-hours are the appendix estimates (A100-class), not measurements. Marks: **4 GB** = the RTX 2050 laptop under
WSL2, **T4** = Kaggle's 16 GB T4 (not the P100, see below), **5090** = RTX 5090 32 GB (Blackwell), **L40S** = 48 GB.

| # | Job | GPU-h | Fits | Command |
|---|---|---|---|---|
| 0 | Calibration | CPU | CPU only | `python p0_calibration.py` |
| 1 | Emmy check | ~1 (ours) | 5090 (Emmy's own benchmark GPU); needs nvcc and the Emmy env | `python emmy_check.py --frontend-only` (CPU), then `python emmy_check.py --probe-only`, then `python emmy_check.py` |
| 2 | Escape grid, H3.3 / H6.1 | 25 | 4 GB, T4, 5090 | `python p5_escape.py` and `python p5_escape.py --part h31` |
| 3 | W4A4 and weight-only, H6.2 (simulated quantization) | 30 | 5090 (Pythia-6.9B and OLMo-7B in fp16, about 14 GB each); L40S also fine | `python p2_w4a4.py` (one model: `--models EleutherAI/pythia-6.9b`) |
| 4 | Clipping sweep, H6.1 / H7.1 control | 30 | T4, 5090 | `python p4_clipping.py --model transformer` and `python p4_clipping.py --model resnet18` |
| 5 | Checkpoint measurements, with the 410M loss-spike check (pass (a) downloads about 391 GB, see below) | 45 | Pythia and PolyPythias: T4 (up to 410M), 5090 (up to 1.4B); OLMo 2 1B: 5090 | `python p1_checkpoints.py --pin-only`, then `--suite pythia`, `--suite polypythia,polypythia_data,polypythia_weight`, `--suite olmo` |
| 6 | Fractional memory, H4.1 | 65 | T4, 5090 | `python p3_fractional_memory.py` (drop LRA first if time runs short: `--tasks powerlaw,exponential`) |
| 7 | **Optional, last:** OLMo 2 7B checkpoints | not in the 45 (ours: ~10) | **L40S** (fp32 weights plus per-example JVP exceed 32 GB) | `python p1_checkpoints.py --suite olmo7b`, only if credits remain |

Jobs 0 to 6 all fit the RTX 5090. Job 7 is the only one that needs the L40S, and nothing else depends on it.

Every script takes `--smoke`, `--seed N`, `--device cuda|cpu`, `--out DIR` and `--offline-ok` (allow the recorded
stand-ins outside smoke mode; off by default, so a real run fails loudly if a download fails). Run the Emmy check
after P1 if you want it to add the three checkpoints with the largest K from `results/p1_checkpoints/checkpoints.csv`.

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

This matters for Pythia: on 27 February 2026 the `pythia-14m` and `pythia-31m` names switched from the deduplicated
to the standard-Pile models (see their model cards). For these two names, `checkpoints.csv` flags
(`name_switch_check`) any commit dated before 2026-02-27. Confirm such a commit on its model card before using it.

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

`python -m pytest -q` (81 tests, about 80 s on a CPU; `-m "not slow"` skips the two slowest). Synthetic data with
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
- coupling test, Holm, detectable r; censored escape rates and the (η, B) power-law fit against the Gaussian rival;
- per-example JVP against autograd, Hessian eigenvalues and Hutchinson trace on a quadratic;
- NF4 code book, FWHT, rotation invariance, RTN error bound, matched bits, fixed-rate clipping;
- chunked selective scan against the naive scan, the fractional kernel against the Mittag-Leffler impulse
  response t^(ν−1) E_{ν,ν}(−λ t^ν), and parameter matching across SSM arms;
- tail labels: alternatives that cannot be separated read "undecided"; exponential against Weibull uses the
  likelihood-ratio test;
- P1 bookkeeping: commit pins (reuse, --require-pins, merge on save), the weight bytes counted for the download
  volume, step choice for the main grid, the 1,000-step scan and OLMo, the pythia-14m/31m name-switch flag
  (15 February 2026 is flagged), both loss-spike passes (each finds a planted spike at its onset, and only
  there), window merging, and the seed-variance table;
- the Emmy check's streaming kurtosis and logit metrics, and the fast-math lanes' environments.

## Deviations, choices of ours, and items to verify

- **H_grad in P1.** Public checkpoints have no optimizer trajectory, so P1 measures H on token-surprisal series
  (`Htok_*`, exploratory). H_grad comes from P4's training runs.
- **CIFAR-10 blocks.** 50,000 images cannot give 1,600 disjoint blocks at B > 31, so the H7.1 control in P4 draws
  augmented copies and saves the base-image IDs.
- **Two-decade rule.** A Pareto ζ = 3 sample needs n well above 10^5 to span two decades in x, so smaller samples read
  "undecided".
- **Δh_rep** is usually "local" for high-dimensional representations, because D_q exceeds log10 n (open question 2).
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
