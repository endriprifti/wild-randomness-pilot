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

## Jobs, cheapest first

GPU-hours are the appendix estimates (A100-class), not measurements. Marks: **4 GB** = the RTX 2050 laptop under
WSL2, **T4** = Kaggle's 16 GB T4 (not the P100, see below), **5090** = RTX 5090 32 GB (Blackwell), **L40S** = 48 GB.

| # | Job | GPU-h | Fits | Command |
|---|---|---|---|---|
| 0 | Calibration | CPU | CPU only | `python p0_calibration.py` |
| 1 | Emmy check | ~1 (ours) | 5090 (Emmy's own benchmark GPU); needs nvcc and the Emmy env | `python emmy_check.py --frontend-only` (CPU), then `python emmy_check.py --probe-only`, then `python emmy_check.py` |
| 2 | Escape grid, H3.3 / H6.1 | 25 | 4 GB, T4, 5090 | `python p5_escape.py` and `python p5_escape.py --part h31` |
| 3 | W4A4 and weight-only, H6.2 | 30 | 5090 (Pythia-6.9B and OLMo-7B in fp16, about 14 GB each); L40S also fine | `python p2_w4a4.py` (one model: `--models EleutherAI/pythia-6.9b`) |
| 4 | Clipping sweep, H6.1 / H7.1 control | 30 | T4, 5090 | `python p4_clipping.py --model transformer` and `python p4_clipping.py --model resnet18` |
| 5 | Checkpoint measurements | 45 | Pythia and PolyPythias: T4 (up to 410M), 5090 (up to 1.4B); **OLMo-7B: L40S** (fp32 weights plus per-example JVP exceed 32 GB) | `python p1_checkpoints.py --suite pythia`, `--suite polypythia`, `--suite olmo` |
| 6 | Fractional memory, H4.1 | 65 | T4, 5090 | `python p3_fractional_memory.py` (drop LRA first if time runs short: `--tasks powerlaw,exponential`) |

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
sudo apt update && sudo apt install -y python3-venv python3-pip git unzip
nvidia-smi                                   # must list the RTX 2050 and a driver >= 580
unzip pilot.zip -d ~ && cd ~/pilot
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

No job has to move to the L40S because of a missing Blackwell kernel. The only L40S job is P1's OLMo-7B suite, for
memory. The L40S (Ada, sm_89) runs torch 2.15 through its sm_86 binaries; Emmy names no Ada results.

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
  - Bug found: when `seq_len` equals `hidden_size`, the IR's fused-QKV reshape gets an input of the wrong size.
    The script keeps `seq_len` off every model dimension (500 in the config).

Still unconfirmed: kernel lowering and numerical agreement on a GPU. Lowering the whole model on a CPU did not
finish here. So whether Emmy supports GPT-NeoX end to end is open until `python emmy_check.py --probe-only` runs on
a GPU with nvcc. The comparison reads "agrees" when top-1 agreement is at least 0.99, |ΔNLL| is at most 0.01 nats
and the outlier-channel Jaccard is at least 0.9 in every layer. These thresholds are ours; change them in `config.py`
before the run.

## Unit tests

`python -m pytest -q` (69 tests, about 80 s on a CPU; `-m "not slow"` skips the two slowest). Synthetic data with
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
- the Emmy check's streaming kurtosis and logit metrics.

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
- **Tail labels.** When several alternatives beat the power law, the one with the smallest AIC wins (ours; §7.1
  does not rank them). Exponential and Weibull samples still sometimes read "truncated" instead of "mild" or "slow".
  Every such label is non-wild.
- **P0** uses 250 bootstrap replicates instead of 2,500, to stay on a CPU.
- **Placeholders for the preregistration**, marked OURS in `config.py`:
  - hill_tol 0.15, the q range 0.5 to 4, min_tail 50 and the x_min candidate grid;
  - coupling_B 8, the barrier tolerance, η values and g_max quantiles;
  - the Emmy tolerances.
- **[VERIFY]** items:
  - `monology/pile-uncopyrighted` stands in for the Pile (P1's gradient pool and evaluation text);
  - the PolyPythias variant: `-seed{n}` varies weights and data order together; `-weight-seed{n}` and
    `-data-seed{n}` vary one each (`config.py`);
  - the ZClip update rule (`common/clipping.py`).
