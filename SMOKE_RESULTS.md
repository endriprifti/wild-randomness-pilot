# Smoke results (this environment, 2026-11-13)

Machine: 4 CPU cores, no GPU, Python 3.13.18, torch 2.15.0+cu132, the pinned `requirements.txt`.
Network: huggingface.co and download.pytorch.org were blocked by the session's proxy, so every Hub model,
tokenizer and dataset was replaced by a recorded stand-in (listed in each `run.json` under `fallbacks`).
Fashion-MNIST downloaded; CIFAR-10 did not, so random images stood in.

| Job | Status | Seconds | Stand-ins used |
|---|---|---|---|
| unit tests (69) | PASS | 82 | none |
| p0_calibration | PASS | 50 | none (synthetic by design) |
| emmy_check (main env) | PASS | 13 | Emmy not installed, so `aot_eager` stood in for the backend; random-init GPT-NeoX; Zipfian text |
| emmy_check --frontend-only (Emmy env) | PASS | 12 | random-init GPT-NeoX; Zipfian text. Real Emmy frontend: `hf_model` fails, `static_wrapper` traces |
| p5_escape | PASS | 14 | none (real Fashion-MNIST) |
| p5_escape --part h31 | PASS | 4 | none |
| p2_w4a4 | PASS | 9 | random-init GPT-NeoX, Zipfian text, synthetic PopQA |
| p4_clipping | PASS | 223 | byte tokenizer, Zipfian text, random CIFAR images |
| p1_checkpoints | PASS | 53 | random-init GPT-NeoX for every checkpoint, Zipfian text |
| p3_fractional_memory | PASS | 33 | random CIFAR images for LRA image |

What "PASS" means here: the job ran end to end and wrote its CSV files and `run.json`. It does not mean the
numbers say anything; the smoke settings are far too small for that (P5's SGD grid, for example, reports
"too few exits", and P3 reaches no accuracy target in 20 steps). What is not exercised here: loading from the
Hub, CUDA, and Emmy's kernels.

P4 is the slowest smoke job (223 s on these 4 cores, after halving its ResNet-18 epochs). On the RTX 2050 it
runs on the GPU.
