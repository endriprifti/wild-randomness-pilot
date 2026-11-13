# Smoke results (this environment, 2026-11-13)

Machine: 4 CPU cores, no GPU, Python 3.13.18, torch 2.15.0+cu132, the pinned `requirements.txt`.
Network: huggingface.co and download.pytorch.org were blocked by the session's proxy, so every Hub model,
tokenizer and dataset was replaced by a recorded stand-in (listed in each `run.json` under `fallbacks`).
Fashion-MNIST downloaded; CIFAR-10 did not, so random images stood in.

| Job | Status | Seconds | Stand-ins used |
|---|---|---|---|
| unit tests (85) | PASS | 89 | none |
| p0_calibration | PASS | 50 | none (synthetic by design) |
| emmy_check (main env) | PASS | 20 | Emmy not installed, so `aot_eager` stood in for the backend (both fast-math lanes run); random-init GPT-NeoX; Zipfian text |
| emmy_check --frontend-only (Emmy env) | PASS | 13 | random-init GPT-NeoX; Zipfian text. Real Emmy frontend: `hf_model` fails, `static_wrapper` traces |
| p5_escape | PASS | 14 | none (real Fashion-MNIST) |
| p5_escape --part h31 | PASS | 5 | none |
| p2_w4a4 | PASS | 8 | random-init GPT-NeoX, Zipfian text, synthetic PopQA |
| p4_clipping | PASS | 153 | byte tokenizer, Zipfian text, random CIFAR images |
| p1_checkpoints | PASS | 49 | random-init GPT-NeoX for every checkpoint (16 reads: 4 on the main grid, 8 in spike pass (a), 4 in pass (b); the stand-ins show no loss jump, so pass (b) ran around the largest z to exercise the code), 8 in pass (b); the stand-ins show no loss jump, so pass (b) ran around the largest z to exercise the code), Zipfian text |
| p3_fractional_memory | PASS | 33 | random CIFAR images for LRA image |

What "PASS" means here: the job ran end to end and wrote its CSV files and `run.json`. It does not mean the
numbers say anything; the smoke settings are far too small for that (P5's SGD grid, for example, reports
"too few exits", and P3 reaches no accuracy target in 20 steps). What is not exercised here: loading from the
Hub, CUDA, and Emmy's kernels.

P4 is the slowest smoke job (153 s on these 4 cores, after halving its ResNet-18 epochs). On the RTX 2050 it
runs on the GPU.

These times use the 512-token stand-in tokenizer. With Pythia-14M's real shape (50k-token vocabulary), the same
smoke settings took 112 s for P1 and 357 s for P4 on these 4 CPU cores (P4 trains and measures on the GPU on the
laptop). P1 and P4 smoke settings were cut after the laptop run on 2026-11-13 (P1: 569 s with real checkpoints).

Laptop run at f292e5c (2026-11-13; RTX 2050 under WSL2, real Hub downloads, HF_TOKEN set): no failures; every job
passed within 300 s except P4 (305 s, SLOW). P4's smoke now runs the H7.1 control at two training stages instead of
four and trains ResNet-18 for 4 epochs instead of 5 (357 s to 324 s on these CPU cores with the 50k-token
vocabulary). Durations now come from a monotonic clock (`time.perf_counter` in `run.json`, `/proc/uptime` in
`run_smoke.sh`): on that run a WSL clock jump had made one duration negative (-63 s) and inflated another.
