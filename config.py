"""Shared configuration for every pilot job (stage 1, Appendix sec:app-pilot).

`get(test, smoke)` returns the full settings, or the full settings overridden by the smoke
settings. Values marked OURS are choices the paper does not fix; change them here, before
the preregistration, never after seeing data. GPU-hour figures are the appendix estimates
(A100-class, 6ND training FLOPs, 2N per token inference, ~30% bf16 utilization), not
measurements.
"""

from __future__ import annotations

import copy

SEED = 20261105  # base seed; every job derives its seeds from it

BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]          # sec. 7.5
TRAINING_STAGES = [0.01, 0.10, 0.50, 1.00]                    # sec. 7.5: 1, 10, 50, 100% of training

TAILS = {                                                     # sec. 7.1
    "n_boot": 2500,           # bootstrap replicates for p_KS (accuracy about 0.01)
    "level": 0.10,            # p_KS <= 0.1 rejects; alternatives preferred at the 10% level
    "min_tail": 50,           # OURS: smallest tail used in the x_min search
    "max_candidates": 200,    # OURS: x_min candidates (quantile-spaced distinct values)
    "hill_eps": 0.1,          # OURS: n1 = n^(1 - eps) in the Danielsson et al. double bootstrap
    "hill_reps": 500,         # OURS: replicates per double-bootstrap stage
    "hill_tol": 0.15,         # OURS: |zeta_DB - zeta_CSN| <= 0.15 zeta_CSN counts as agreement
    "n_jobs": 4,              # processes for the p_KS bootstrap
}

SCALE = {                                                     # sec. 7.2
    "n_surr": 99,             # shuffled / phase / IAAFT surrogates per test
    "q_lo": 0.5, "q_hi": 4.0,  # OURS (placeholder for the preregistration): q range of Delta h
    "wavelet": "db3", "wavelet_j1": 3,
    "dfa_s_min": 10,          # DFA scales from 10 to n/10 (two decades need n >= 10^4)
}

H71 = {"inv_tol": 0.03, "drift_tol": 0.05, "level": 0.90, "n_blocks": 1600, "n_boot": 200,
       # margin of the H7.1 verdict (Endri, 2026-11-14): "invariant (wild)" also needs mean alpha-hat <= 1.9; between
       # 1.9 and the Gaussian 5% quantile it reads undecided (Student t with 3-4 degrees of freedom gives a flat
       # alpha-hat of 1.93-1.98 up to B = 1,024, as SaS 1.95-1.97 does). OURS. 2.0 turns it off.
       "wild_max_alpha": 1.9}
# H7.1 verdict (common/h71_verdict.py): alpha = 2 is tested against Gaussian pools of the same length, n_blocks and
# batch sizes (5% quantile of mean alpha-hat; central 90% of the scale exponent). OURS: pools per setting.
H71_NULL_REPS = 400

CONFIG = {
    # ------------------------------------------------------------------ P0b: P0's thin spots (CPU, laptop)
    "p0b_calibration": {
        "full": {
            # MFDFA width test on cascade series: 20 replicates per (p, n) (P0 had 1); random-order cascades
            "cascades": {"p": [0.5, 0.7, 0.8], "n_values": [1000, 10000, 100000], "replicates": 20},
            # H7.1 on truncated-Levy pools (alpha 1.5). Where alpha-hat drifts to 2 depends on the cutoff: 20 (P0's
            # trunclevy_1.5) finishes the drift by B ~ 64, 100 spreads it over B = 8 ... 1024, 500 starts it near 512
            "h71": {"alpha": 1.5, "cutoffs": [20.0, 100.0, 500.0], "n_blocks": 1600,
                    "replicates": 200,        # per cutoff
                    "test_reps": 5,           # OURS: pools per cutoff that get the full test (each ~1,600 stable fits)
                    "test_n_boot": 200,       # as in sec. 7.5
                    # Gaussian pools (alpha = 2): the reference for the level of the H7.1 verdict at the settings of
                    # the full run (1,600 blocks: P0, P0b, P1 at 14M, P4; 200 blocks: P1's coarse sizes), and
                    # full tests on Gaussian pools, which should read "flat at 2 (mild)"
                    "null_n_blocks": [1600, 200], "null_reps": H71_NULL_REPS,
                    "gauss_test_reps": 5},    # OURS
            # direct stable fit and the bootstrap LR test against the Gaussian on SaS samples
            "stable": {"alphas": [1.2, 1.5, 1.8, 2.0], "n_values": [1000, 10000, 100000], "replicates": 200,
                       "lr_null_draws": 1000,  # OURS: shared null per n (the LR statistic is location-scale invariant)
                       "lr_check_reps": 3,     # OURS: replicates per cell that also run lr_test_gaussian itself
                       "lr_check_n_boot": 200},
        },
        "smoke": {
            "cascades": {"n_values": [1000], "replicates": 2},
            "h71": {"cutoffs": [20.0, 100.0], "n_blocks": 60, "batch_sizes": [8, 16, 32], "replicates": 2,
                    "test_reps": 1, "test_n_boot": 3, "null_n_blocks": [60], "null_reps": 20, "gauss_test_reps": 1},
            "stable": {"n_values": [1000], "replicates": 3, "lr_null_draws": 20, "lr_check_reps": 1,
                       "lr_check_n_boot": 5},
            "scale": {"n_surr": 9},
        },
    },
    # ------------------------------------------------------------------ P0: calibration (CPU)
    "p0_calibration": {
        "full": {
            "n_values": [1000, 10000, 100000],
            "replicates": 200,
            "tail_laws": {  # true state for the confusion matrix
                "sas_1.2": "wild", "sas_1.5": "wild", "sas_1.8": "wild",
                "pareto_1.5": "wild", "pareto_3": "pre-wild",
                "lognormal_1": "slow", "weibull_0.5": "slow", "weibull_1.5": "mild",
                "trunclevy_1.5": "truncated",
            },
            "trunc_levy_cutoff": 20.0,    # OURS: abrupt cutoff x_c in scale units (Mantegna-Stanley)
            "n_boot": 250,                # OURS: fewer replicates than sec. 7.1 to keep P0 on a CPU
            "hurst_series": {"fgn_0.3": 0.3, "fgn_0.5": 0.5, "fgn_0.7": 0.7,
                             "levy_1.2": None, "levy_1.5": None, "levy_1.8": None},
            "cascade_p": [0.5, 0.7, 0.8],  # p = 0.5 is monofractal (uniform)
            "cascade_depth": 20,
            "block_check": {"alpha": 1.5, "n_blocks": 1600, "replicates": 200, "bootstrap_reps": 3},
        },
        "smoke": {
            "n_values": [1000], "replicates": 2, "n_boot": 20,
            "tail_laws": {"sas_1.5": "wild", "pareto_1.5": "wild", "lognormal_1": "slow",
                          "weibull_1.5": "mild", "trunclevy_1.5": "truncated"},
            "hurst_series": {"fgn_0.7": 0.7, "levy_1.5": None},
            "cascade_p": [0.7], "cascade_depth": 14,
            "block_check": {"alpha": 1.5, "n_blocks": 100, "replicates": 3, "bootstrap_reps": 1},
            "tails": {"hill_reps": 30, "n_jobs": 1}, "scale": {"n_surr": 9},
        },
    },
    # ------------------------------------------------------------------ P1: checkpoints (no training)
    "p1_checkpoints": {
        "full": {
            # Every model is loaded at the commit recorded in pins.json (see README, "Pinned commits").
            # variant: base = one run per size; joint = init and batch composition vary together
            # (PolyPythias -seed{n}); data = only the data order varies; weight = only the initialization.
            # Suites marked optional run only when named (--suite olmo7b), never by default.
            "suites": {
                "pythia": {"repo": "EleutherAI/pythia-{size}", "sizes": ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b"],
                           "seeds": [0], "variant": "base"},
                "polypythia": {"repo": "EleutherAI/pythia-{size}-seed{seed}", "sizes": ["14m", "31m", "70m", "160m", "410m"],
                               "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9], "variant": "joint"},
                "polypythia_data": {"repo": "EleutherAI/pythia-160m-data-seed{seed}", "sizes": ["160m"],
                                    "seeds": [1, 2, 3], "variant": "data"},
                "polypythia_weight": {"repo": "EleutherAI/pythia-160m-weight-seed{seed}", "sizes": ["160m"],
                                      "seeds": [1, 2, 3], "variant": "weight"},
                # default OLMo: OLMo 2 1B (April 2025) with released pretraining checkpoints, fits the RTX 5090
                "olmo": {"repo": "allenai/OLMo-2-0425-1B", "sizes": ["1b"], "seeds": [0], "variant": "base",
                         "branch_regex": r"^stage1-step(\d+)-tokens",
                         "step_fracs": [0.0005, 0.002, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1.0]},
                # OPTIONAL, last job, only if credits remain: needs the L40S (48 GB). Its pretraining branches are
                # named like OLMo 2 1B's, "stage1-step{N}-tokens{T}B" (final stage-1 step 928646); a few early ones
                # lacked the "stage1-" prefix (Hugging Face discussion #6 on OLMo-2-1124-7B), so the prefix is
                # optional. Stage-2 ("stage2-ingredient...") and soup branches do not match. Until 2026-11-25 the
                # regex was r"^step(\d+)-tokens", which matched no branch on 2026-11-24 (p1_olmo7b failed with "no
                # step branches", so no unprefixed name was left then)
                "olmo7b": {"repo": "allenai/OLMo-2-1124-7B", "sizes": ["7b"], "seeds": [0], "variant": "base",
                           "branch_regex": r"^(?:stage1-)?step(\d+)-tokens", "optional": True,
                           "step_fracs": [0.002, 0.01, 0.1, 0.5, 1.0],
                           # micro-batches (python memory_check.py, float32, eager attention): 27.2 GiB of weights;
                           # eval at 4 x 2,048 tokens would peak near 39.6 GiB and the JVP at 32 x 128 near
                           # 42.1 GiB on the 44 GiB L40S; 1 and 8 peak near 30.3 and 31.5 GiB
                           "eval_batch": 1, "jvp_batch": 8},
            },
            "steps": [1, 8, 64, 512, 1000, 4000, 8000, 16000, 32000, 64000, 143000],   # Pythia-style suites
            # Loss spikes (PolyPythias paper, Fig. 5, which gives no step numbers): 410M seeds 3 and 4. Two passes:
            # (a) probe loss only, at every 1,000-step checkpoint, for the flagged seeds and one stable seed;
            # (b) alpha, alpha_ESD and K for all nine 410M seeds at every 1,000-step checkpoint from
            #     window steps before to window steps after each jump that pass (a) finds (same robust-z rule).
            "spike_check": {"suite": "polypythia", "size": "410m", "flagged_seeds": [3, 4],
                            "reference_seeds": [1],          # OURS: the stable seed (any seed other than 3 and 4)
                            "scan_every": 1000, "scan_range": [1000, 143000],
                            "window": 5000,
                            "quantities": ["alpha", "alpha_esd", "K"],
                            "robust_z": 3.5},                # OURS: the threshold in both passes
            # pythia-14m / pythia-31m: re-created on 24 February 2026 with the standard-Pile models (the step branches'
            # commits are all from that day; 27 February is when main's model card announced it); the 2023
            # deduplicated upload moved, with its history, to pythia-14m-deduped / pythia-31m-deduped. Two checks in
            # checkpoints.csv: content (dedup_check) = the pinned commit's model.safetensors sha256 against the same
            # revision of repo + deduped_suffix, flagged only if identical (content_files: the first file both
            # carry); date (name_switch_check) = commits dated before `date` are flagged
            "name_switch": {"repos": ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m"], "date": "2026-02-24",
                            "deduped_suffix": "-deduped", "content_files": ["model.safetensors", "pytorch_model.bin"]},
            "final_step": 143000,
            "dtype": "float32",
            # Pile text without the copyrighted subsets. The split names come with explicit data_files: the
            # datasets builder detects only the train/ folder of this repository and misses val.jsonl.zst and
            # test.jsonl.zst at its top level ("Bad split: validation", L40S, 2026-11-16). Both are read at the
            # commit pins.json records for datasets/monology/pile-uncopyrighted@main (python check_data.py)
            "pool_dataset": {"repo": "monology/pile-uncopyrighted", "split": "train",
                             "data_files": {"train": "train/*.jsonl.zst"}},
            "eval_dataset": {"repo": "monology/pile-uncopyrighted", "split": "validation",
                             "data_files": {"validation": "val.jsonl.zst"}},   # eval, probe and surprisal sets
            "example_len": 128,         # OURS: tokens per example for per-example gradients
            "grad_layers": ["layers.{last}.attention.query_key_value.weight", "layers.{last}.mlp.dense_4h_to_h.weight"],
            "n_random_dirs": 2, "n_hessian_dirs": 1, "hessian_batch": 64,
            "coupling_B": 8,            # OURS: fixed B for the alpha that enters the coupling matrix
            "coupling_blocks": 1600,
            "probe_seqs": 64, "probe_len": 512,   # probe set for K, outliers and Delta h_rep
            "dh_points": 10000,
            "surprisal_docs": 16, "surprisal_len": 2048,  # H_tok series (exploratory, not H_grad)
            "h71": {"full_res_sizes": ["14m"], "coarse_sizes": ["31m", "70m", "160m"], "coarse_blocks": 200},
            "eval_seqs": 256, "eval_len": 2048,
            "tails": {"n_boot": 2500},
        },
        "smoke": {
            "suites": {"pythia": {"repo": "EleutherAI/pythia-{size}", "sizes": ["14m"], "seeds": [0], "variant": "base"},
                       "polypythia": {"repo": "EleutherAI/pythia-{size}-seed{seed}", "sizes": ["14m"], "seeds": [1, 3],
                                      "variant": "joint"},
                       # smoke only: no 14M data-seed run exists, so this name falls back to a stand-in and
                       # only exercises the per-variant code (seed_variance.csv)
                       "polypythia_data": {"repo": "EleutherAI/pythia-{size}-data-seed{seed}", "sizes": ["14m"],
                                           "seeds": [1], "variant": "data"}},
            # smoke timing (RTX 2050 laptop, real Pythia-14M checkpoints): one main-grid step, H7.1 at the last stage
            # only, a 4-step loss scan and pass (b) on the jump interval itself; the cache is kept in smoke mode
            "steps": [143000], "final_step": 143000,
            # smoke_force_top: if pass (a) finds no jump in the stand-in models, pass (b) runs around the largest z
            "spike_check": {"suite": "polypythia", "size": "14m", "flagged_seeds": [3], "reference_seeds": [1],
                            "scan_every": 1000, "scan_range": [1000, 4000], "window": 0,
                            "quantities": ["alpha", "alpha_esd", "K"], "robust_z": 3.5, "smoke_force_top": True},
            "example_len": 32, "n_random_dirs": 1, "n_hessian_dirs": 0, "hessian_batch": 8,   # Hessian: unit tests and P4
            "coupling_blocks": 25, "probe_seqs": 4, "probe_len": 64, "dh_points": 256,
            "surprisal_docs": 2, "surprisal_len": 512,
            "h71": {"full_res_sizes": ["14m"], "coarse_sizes": [], "coarse_blocks": 20, "n_blocks": 30,
                    "batch_sizes": [8, 16, 32], "n_boot": 3, "stages": [1.0]},
            "eval_seqs": 2, "eval_len": 128,
            "tails": {"n_boot": 10, "hill_reps": 20, "n_jobs": 1}, "scale": {"n_surr": 9},
        },
    },
    # ------------------------------------------------------------------ P2: W4A4 (inference only)
    "p2_w4a4": {
        "full": {
            "models": ["EleutherAI/pythia-6.9b", "allenai/OLMo-7B-hf"],
            # bfloat16 for both: Pythia-6.9B overflows in float16 (NaN baseline on the L40S, 2026-11-15);
            # OLMo was trained in bfloat16. float32 also fits the L40S (--dtype float32)
            "dtype": "bfloat16",
            "arms": {
                "w4a4": [{"name": "uniform", "a_bits": 4, "group": 128},
                         {"name": "outlier", "a_bits": 4, "group": 256, "outlier_frac": 0.005},
                         {"name": "rotation", "a_bits": 4, "group": 128}],
                "weight_only": [{"name": "uniform", "a_bits": None, "group": 128},
                                {"name": "outlier", "a_bits": None, "group": 256, "outlier_frac": 0.005},
                                {"name": "rotation", "a_bits": None, "group": 128},
                                {"name": "nf4", "a_bits": None, "group": 128}],
            },
            "bits_tolerance": 0.05,
            "popqa": {"dataset": "akariasai/PopQA", "split": "test", "limit": None, "shots": 4,
                      "max_new_tokens": 16, "batch": 16, "deciles": 10},
            "doc_counts_csv": None,      # optional: subject -> pretraining document count (see README)
            "ppl": {"dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "test"], "ctx": 2048, "max_windows": None},
            "calib": {"dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "train"], "n_seqs": 32, "len": 512},
            "probe": {"n_seqs": 16, "len": 512},
            "isotropic": {"bisect_steps": 8, "ppl_rel_tol": 0.02},
        },
        "smoke": {
            "models": ["EleutherAI/pythia-14m"], "dtype": "float32",
            "popqa": {"dataset": "akariasai/PopQA", "split": "test", "limit": 40, "shots": 2,
                      "max_new_tokens": 6, "batch": 8, "deciles": 2},
            "ppl": {"dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "test"], "ctx": 128, "max_windows": 2},
            "calib": {"dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "train"], "n_seqs": 2, "len": 64},
            "probe": {"n_seqs": 2, "len": 64},
            "isotropic": {"bisect_steps": 3, "ppl_rel_tol": 0.1},
        },
    },
    # ------------------------------------------------------------------ P3: fractional memory (H4.1)
    "p3_fractional_memory": {
        "full": {
            "archs": ["frac", "s4_legs", "s4d", "mamba"],   # add "legs" for frozen HiPPO-LegS
            "n_states": [2, 4, 8, 16, 32, 64],
            "seeds": [0, 1, 2, 3, 4],
            "tasks": {
                "powerlaw": {"L": 1024, "nu_star": 0.3, "n_train": 8192, "n_test": 1024},
                "exponential": {"L": 1024, "tau_frac": 0.125, "n_train": 8192, "n_test": 1024},
                "lra_image": {"L": 1024},
            },
            "H_ref": 64, "ref_arch": "s4d", "n_layers": 2,
            "steps": 3000, "batch": 32,
            "lr_grid": [1e-3, 3e-3, 1e-2], "tune_seed": 0, "tune_n_states": 16,   # equal tuning budget
            "weight_decay": 0.01, "target_accuracy": {"powerlaw": 0.9, "exponential": 0.9, "lra_image": 0.6},
            # lra_image: epochs, state sizes, seeds and tuning per design (the other tasks use the settings above).
            # "full" is the first plan: 132 runs of 31,240 steps (20 x 1,562), about 10.4 times the steps of a
            # synthetic task, about 180-210 L40S hours by the first P3 run's pace. "reduced" (2026-11-17, the
            # default; --lra-design full for the other): 48 runs of 4,686 steps, about 6-10 L40S hours by the same
            # pace; tuned on seed 0 and measured on seeds 1-3, so no main run repeats a tuning run.
            "lra_design": "reduced",
            "lra_designs": {
                "full": {"epochs": 20, "n_states": [2, 4, 8, 16, 32, 64], "seeds": [0, 1, 2, 3, 4],
                         "tune_n_states": 16, "tune_seed": 0},
                "reduced": {"epochs": 3, "n_states": [2, 8, 32], "seeds": [1, 2, 3],
                            "tune_n_states": 8, "tune_seed": 0},
            },
        },
        "smoke": {
            "archs": ["frac", "s4_legs", "s4d", "mamba"], "n_states": [2, 8], "seeds": [0],
            "tasks": {"powerlaw": {"L": 128, "nu_star": 0.3, "n_train": 256, "n_test": 64},
                      "exponential": {"L": 128, "tau_frac": 0.125, "n_train": 256, "n_test": 64},
                      "lra_image": {"L": 1024, "limit_train": 256, "limit_test": 64}},
            "H_ref": 16, "steps": 20, "batch": 16, "lr_grid": [3e-3], "tune_n_states": 2,
            "lra_designs": {"full": {"epochs": 1, "n_states": [2, 8], "seeds": [0], "tune_n_states": 2},
                            "reduced": {"epochs": 1, "n_states": [2, 8], "seeds": [1], "tune_n_states": 2}},
        },
    },
    # ------------------------------------------------------------------ P4: clipping sweep (H6.1, H7.1 control, H_grad)
    "p4_clipping": {
        "full": {
            "models": ["transformer", "resnet18"],
            "p_clip": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0],
            "adaptive": ["autoclip", "zclip", "agc"],
            "matched_update_length": True,
            "seeds": [0, 1, 2, 3, 4],
            "transformer": {"neox": "PYTHIA_14M", "dataset": ["HuggingFaceFW/fineweb-edu", "sample-10BT", "train"],
                            "tokenizer": "EleutherAI/pythia-14m", "tokens": 300_000_000, "seq_len": 512,
                            "batch": 64, "lr": 1e-3, "weight_decay": 0.1, "warmup_frac": 0.01,
                            "ood_dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "test"], "eval_seqs": 256},
            "resnet18": {"epochs": 30, "batch": 128, "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
                         "warmup_frac": 0.02, "ood_noise_sigma": 0.1},
            "calib_steps": 200, "diverge_factor": 3.0, "autoclip_pct": 10.0, "zclip_z": 2.5, "zclip_beta": 0.97, "agc_lambda": 0.01,
            "n_hgrad_dirs": 8,
            "measure": {"hessian_batch": 512, "power_iters": 20, "trace_samples": 20, "sharp_rho": 0.05,
                        "sharp_steps": 5, "alpha_pool_blocks": 1600, "probe_points": 5000,
                        # micro-batch of every batch measure in measure() (HVP, trace, sharpness, per-example
                        # projections, losses). Peak memory, estimated with python memory_check.py: the HVP on
                        # 512 sequences of 512 tokens needs about 537 GiB (the L40S failure of 2026-11-15); 16
                        # sequences need about 17 GiB. ResNet-18: about 2.5 GiB at 128 images.
                        "micro_batch": {"transformer": 16, "resnet18": 128}},
            "h71": {"run": {"model": "resnet18", "rule": "none", "seed": 0}, "n_blocks": 1600,
                    "batch_sizes": BATCH_SIZES, "n_random_dirs": 2, "n_hessian_dirs": 1,
                    "layer": "layer4.1.conv2.weight", "n_boot": 200},
        },
        "smoke": {
            "models": ["transformer", "resnet18"], "p_clip": [0.0, 0.1], "adaptive": ["autoclip"],
            "matched_update_length": True, "seeds": [0],
            "transformer": {"neox": "TINY_NEOX", "dataset": ["HuggingFaceFW/fineweb-edu", "sample-10BT", "train"],
                            "tokenizer": "EleutherAI/pythia-14m", "tokens": 110 * 8 * 64, "seq_len": 64,   # 110 steps: enough for H_grad (>= 100 after warmup)
                            "batch": 8, "lr": 1e-3, "weight_decay": 0.1, "warmup_frac": 0.05,
                            "ood_dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "test"], "eval_seqs": 4},
            "resnet18": {"epochs": 4, "batch": 16, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
                         "warmup_frac": 0.05, "ood_noise_sigma": 0.1, "limit_train": 256, "limit_test": 64},
            "stages": [0.5, 1.0],   # H7.1 control at two training stages in smoke (305 s on the laptop with four)
            "calib_steps": 5, "n_hgrad_dirs": 2,
            # the real 50k-token vocabulary makes every HVP and per-example JVP costly: smallest batches that still run
            "measure": {"hessian_batch": 8, "power_iters": 3, "trace_samples": 2, "sharp_rho": 0.05,
                        "sharp_steps": 1, "alpha_pool_blocks": 25, "probe_points": 200,
                        "micro_batch": {"transformer": 4, "resnet18": 4}},   # smoke runs the chunked path
            "h71": {"run": {"model": "resnet18", "rule": "none", "seed": 0}, "n_blocks": 20,
                    "batch_sizes": [8, 16, 32], "n_random_dirs": 1, "n_hessian_dirs": 0,
                    "layer": "layer4.1.conv2.weight", "n_boot": 3},
            "tails": {"n_boot": 10, "hill_reps": 20, "n_jobs": 1}, "scale": {"n_surr": 9, "dfa_s_min": 4},
        },
    },
    # ------------------------------------------------------------------ P5: escape (H3.3, H6.1 clause, H3.1 injected)
    "p5_escape": {
        "full": {
            "n_train": 10000, "hidden": 128, "act": "tanh",
            "batch_sizes": BATCH_SIZES,
            "etas": [0.05, 0.1, 0.2],              # OURS; the middle rate is used for clipping and injection
            "g_max_quantiles": [0.5, 0.9, 0.99],   # OURS: g_max = these quantiles of minibatch norms at W_S (B = 64)
            "inject_alphas": [1.2, 1.5, 1.8, 2.0],
            "inject_iqr": 0.05,                    # OURS: matched interquartile range of the injected noise at B = 1
            "restarts": 50, "k_cens": 50_000, "check_every": 250,
            "descent_steps": 200, "descent_lr": 0.1, "segment_points": 11,
            "barrier_tol": 0.01,                   # OURS (preregister): loss barrier above both ends, nats
            # extra checks every fine_check[0] steps up to step fine_check[1] (reruns of floor cells: --fine-check 10:250)
            "fine_check": None,
            # second reading (OURS, placeholders; checks.csv lets refit re-threshold them): exit when the descended
            # endpoint is farther than dist_tol x |W_S| from W_S or its loss exceeds L(W_S) by loss_tol nats;
            # continue = keep stepping after a barrier exit until this reading fires too (--second-reading)
            "second_reading": {"dist_tol": 0.1, "loss_tol": 0.05, "continue": False},
            "converge_steps": 20000, "converge_lr": 0.1,
            # diverge_loss: train loss above ln 10 (a uniform guess) = diverged; min_tol: lambda_min >= -min_tol x
            # lambda_max counts as a minimum (OURS)
            "h31": {"alphas": [1.2, 1.5, 1.8, 2.0], "seeds": [0, 1, 2, 3, 4], "steps": 20000, "lr": 0.1, "iqr": 0.05,
                    "diverge_loss": 2.302585092994046, "min_tol": 0.01},
        },
        "smoke": {
            "n_train": 512, "hidden": 32, "batch_sizes": [8, 32], "etas": [0.1], "g_max_quantiles": [0.9],
            "inject_alphas": [1.5], "restarts": 4, "k_cens": 200, "check_every": 50, "descent_steps": 20,
            "converge_steps": 300,
            "h31": {"alphas": [1.5, 2.0], "seeds": [0], "steps": 100, "lr": 0.1, "iqr": 0.05,
                    "diverge_loss": 2.302585092994046, "min_tol": 0.01},
        },
    },
    # ------------------------------------------------------------------ Emmy check
    "emmy_check": {
        "full": {
            "models": [["EleutherAI/pythia-410m", "step143000"], ["EleutherAI/pythia-1b", "step143000"],
                       ["EleutherAI/pythia-1.4b", "step143000"]],
            "from_p1_top_k": 3,      # add the three checkpoints with the largest K in P1 results, if present
            # 500, not 512: emmy-ml 0.3.25 evaluates the embedding lookup as torch.gather when seq_len equals
            # hidden_size (docs/emmy-issue-seq-len-hidden-size.md); emmy_check.py also shifts seq_len off any model dimension
            "seq_len": 500, "n_seqs": 16, "dtypes": ["float16", "float32"],
            "fast_math": [1, 0],     # 1 = Emmy's default (EMMY_FAST_MATH unset), 0 = EMMY_FAST_MATH=0
            "dataset": ["Salesforce/wikitext", "wikitext-2-raw-v1", "test"],
            "outlier_threshold": 6.0,   # same absolute threshold as P1's activation statistics
            # agreement reading (OURS, placeholders until preregistration): compiled vs eager at the same dtype
            "tolerances": {"top1_min": 0.99, "abs_d_nll_max": 0.01, "outlier_jaccard_min": 0.9},
            "worker_timeout_s": 3600,
        },
        "smoke": {"models": [["EleutherAI/pythia-14m", "step143000"]], "from_p1_top_k": 0,
                  "seq_len": 60, "n_seqs": 2, "dtypes": ["float32"], "fast_math": [1, 0]},
    },
}

HARDWARE = {
    # job -> (estimated GPU-h from the appendix, smallest GPU class that fits the full job)
    "p0_calibration": ("CPU", "CPU (multi-core)"),
    "p0b_calibration": ("CPU", "CPU (multi-core); run on the laptop, outside the cloud run"),
    "emmy_check": ("~1 (OURS)", "RTX 5090 or L40S; needs nvcc"),
    "p5_escape": (25, "4 GB laptop / Kaggle T4"),
    "p2_w4a4": (30, "RTX 5090 (32 GB)"),
    "p4_clipping": (30, "Kaggle T4 (16 GB) / RTX 5090"),
    "p1_checkpoints": (45, "RTX 5090 for every default suite (OLMo 2 1B included); optional OLMo 2 7B: L40S (48 GB)"),
    "p3_fractional_memory": (65, "Kaggle T4 (16 GB) / RTX 5090"),
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("suites", "tasks", "tail_laws", "hurst_series", "arms"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(test: str, smoke: bool = False) -> dict:
    full = CONFIG[test]["full"]
    cfg = _merge(full, CONFIG[test]["smoke"]) if smoke else copy.deepcopy(full)
    cfg["tails"] = _merge(TAILS, cfg.get("tails", {}))
    cfg["scale"] = _merge(SCALE, cfg.get("scale", {}))
    cfg["h71_tolerances"] = copy.deepcopy(H71)
    cfg.setdefault("batch_sizes_h71", BATCH_SIZES)
    cfg.setdefault("stages", TRAINING_STAGES)
    cfg["seed"] = SEED
    cfg["hardware"] = HARDWARE.get(test)
    return cfg
