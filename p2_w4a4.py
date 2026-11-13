"""P2: H6.2 as inference on open models (W4A4 primary arm, weight-only control arm).

For each model: 16-bit baseline; W4A4 with uniform, outlier-preserving and rotation (QuaRot-
style) schemes; weight-only W4 with the same three plus NF4; bits per weight matched within
0.05 (scales and 16-bit outlier columns included). For every scheme an isotropic control adds
Gaussian noise (to weights, and to activations under W4A4) scaled to give the same perplexity
increase. Measured: WikiText-2 perplexity, PopQA accuracy by subject-popularity decile (and by
pretraining document count if a CSV is given), and K, Kurt0 and zeta of activations.

H6.2 reading (register): (a) uniform W4A4 loses more on rare than frequent facts; (b) outlier
preservation recovers most of it; (c) rotation recovers mean accuracy but leaves a rare-fact gap;
weight-only shows little excess. Refuted if (a) fails, if weight-only loses as much on rare
facts at equal perplexity increase, if rare-fact losses match the isotropic control, or if
rotation matches outlier preservation on rare facts.

Outputs (results/p2_w4a4/[smoke/]): schemes.csv (one row per model x arm x scheme incl.
isotropic controls), popqa_items.csv (per item correctness), deciles.csv, activations.csv, run.json

Commands:
  python p2_w4a4.py                                  # Pythia-6.9B and OLMo-7B, ~14 GB each in fp16
  python p2_w4a4.py --models EleutherAI/pythia-6.9b
  python p2_w4a4.py --smoke
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import torch

import config
from common import hf, lm, quant, runtime, spectra


def popqa_items(cfg, run, rng):
    pq = cfg["popqa"]
    ds, src = hf.load_rows(pq["dataset"], None, pq["split"], run, limit=pq["limit"])
    if ds is None:
        run.fallback("PopQA: synthetic question set with random popularity")
        items = []
        for i in range(pq["limit"] or 200):
            ans = f"w{int(rng.integers(0, 50))}"
            items.append({"question": f"What is the code of item {i}?", "answers": [ans],
                          "s_pop": float(rng.lognormal(5, 2)), "subj": f"item {i}"})
        return items, "fallback"
    items = []
    for ex in ds:
        ans = ex["possible_answers"]
        if isinstance(ans, str):
            ans = json.loads(ans)
        items.append({"question": ex["question"], "answers": ans, "s_pop": float(ex["s_pop"]), "subj": ex["subj"]})
    return items, src


@torch.no_grad()
def popqa_eval(model, tok, items, shots, cfg, device):
    pq = cfg["popqa"]
    demo = "".join(f"Q: {d['question']} A: {d['answers'][0]}\n" for d in shots)
    correct = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(items), pq["batch"]):
        chunk = items[s:s + pq["batch"]]
        prompts = [demo + f"Q: {it['question']} A:" for it in chunk]
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=pq["max_new_tokens"],
                             do_sample=False, pad_token_id=pad)
        for it, o in zip(chunk, out[:, ids.shape[1]:]):
            text = tok.decode(o.tolist(), skip_special_tokens=True).split("\n")[0].strip().lower()
            correct.append(any(a.lower() in text for a in it["answers"] if a))
    return np.array(correct, dtype=bool)


@torch.no_grad()
def perplexity(model, windows, device):
    nll = lm.token_losses(model, windows, device, batch=1)
    return float(math.exp(nll.mean()))


def decile_table(items, correct, n_dec):
    pops = np.array([it["s_pop"] for it in items])
    edges = np.quantile(pops, np.linspace(0, 1, n_dec + 1))
    d = np.clip(np.searchsorted(edges, pops, side="right") - 1, 0, n_dec - 1)
    return d, np.array([correct[d == k].mean() if (d == k).any() else np.nan for k in range(n_dec)])


class WeightBank:
    """CPU copy of the original linear layers, restored before each scheme."""

    def __init__(self, model):
        self.orig = {n: copy.deepcopy(m).cpu() for n, m in quant.linear_layers(model).items()}

    def restore(self, model, device):
        for name, lin in self.orig.items():
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, copy.deepcopy(lin).to(device))


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--models", default=None, help="comma-separated model ids (default: config)")
    ap.add_argument("--doc-counts", default=None, help="CSV with columns subj,doc_count (optional)")
    args = ap.parse_args()
    cfg = config.get("p2_w4a4", args.smoke)
    run = runtime.start_run("p2_w4a4", cfg, args)
    rng = runtime.rng(run.seed)
    dev = run.device
    dtype = getattr(torch, cfg["dtype"])
    models = args.models.split(",") if args.models else cfg["models"]
    pq = cfg["popqa"]
    rows, item_rows, dec_rows, act_rows = [], [], [], []

    for name in models:
        model, tok, src = hf.load_causal_lm(name, None, run, dtype=dtype, device=dev)
        vocab = model.get_output_embeddings().weight.shape[0]
        ppl_texts, _ = hf.load_texts(*cfg["ppl"]["dataset"], n=4000, run=run, field="text", min_chars=1, seed=run.seed)
        all_ids = hf.tokenize_stream(["\n".join(ppl_texts)], tok, cfg["ppl"]["ctx"], 10 ** 9 if not cfg["ppl"]["max_windows"] else cfg["ppl"]["max_windows"])
        windows = all_ids.clamp(max=vocab - 1)
        cal_texts, _ = hf.load_texts(*cfg["calib"]["dataset"], n=2000, run=run, min_chars=50, seed=run.seed + 1)
        calib = hf.tokenize_stream(cal_texts, tok, cfg["calib"]["len"], cfg["calib"]["n_seqs"]).clamp(max=vocab - 1)
        probe = hf.tokenize_stream(cal_texts[::-1], tok, cfg["probe"]["len"], cfg["probe"]["n_seqs"]).clamp(max=vocab - 1)
        items, isrc = popqa_items(cfg, run, rng)
        shots, items = items[:pq["shots"]], items[pq["shots"]:]
        n_dec = pq["deciles"]

        # baseline
        base_ppl = perplexity(model, windows, dev)
        base_correct = popqa_eval(model, tok, items, shots, cfg, dev)
        dec_idx, base_dec = decile_table(items, base_correct, n_dec)
        hs = lm.hidden_states(model, probe, dev)
        for li, H in enumerate(hs):
            act_rows.append({"model": name, "layer": li, **spectra.activation_stats(H)})
        del hs
        rows.append({"model": name, "arm": "baseline", "scheme": "fp16", "ppl": base_ppl,
                     "acc": float(base_correct.mean()), "bits": 16.0, "source": src, "popqa_source": isrc})
        item_rows += [{"model": name, "arm": "baseline", "scheme": "fp16", "i": i, "correct": bool(c),
                       "s_pop": it["s_pop"], "decile": int(d)} for i, (c, it, d) in enumerate(zip(base_correct, items, dec_idx))]
        absmax = quant.calibrate_absmax(model, [calib[i:i + 1].long() for i in range(len(calib))], dev)
        bank = WeightBank(model)
        d_in = max(l.in_features for l in quant.linear_layers(model).values())

        for arm, schemes in cfg["arms"].items():
            sch = [quant.Scheme(**s) for s in schemes]
            bits = quant.check_bits_matched(sch, d_in, cfg["bits_tolerance"])
            if not bits["matched"]:
                raise SystemExit(f"bits not matched within {cfg['bits_tolerance']}: {bits}")
            for s in sch:
                bank.restore(model, dev)
                quant.apply_scheme(model, s, absmax, seed=run.seed)
                ppl = perplexity(model, windows, dev)
                corr = popqa_eval(model, tok, items, shots, cfg, dev)
                _, dec = decile_table(items, corr, n_dec)
                drop = base_dec - dec
                row = {"model": name, "arm": arm, "scheme": s.name, "ppl": ppl, "dppl": ppl - base_ppl,
                       "acc": float(corr.mean()), "bits": s.bits_per_weight(d_in),
                       "drop_rare": float(drop[0]), "drop_frequent": float(drop[-1]),
                       "rare_minus_frequent": float(drop[0] - drop[-1])}
                rows.append(row)
                dec_rows += [{"model": name, "arm": arm, "scheme": s.name, "decile": k, "acc": float(dec[k]),
                              "acc_base": float(base_dec[k])} for k in range(n_dec)]
                item_rows += [{"model": name, "arm": arm, "scheme": s.name, "i": i, "correct": bool(c)}
                              for i, c in enumerate(corr)]
                run.note(f"{name} {arm}/{s.name}: ppl {ppl:.3f} (base {base_ppl:.3f}) acc {row['acc']:.3f}")

                # isotropic control at equal perplexity increase: bracket, then bisect on sigma
                target = ppl
                tol = cfg["isotropic"]["ppl_rel_tol"]

                def iso_at(sig):
                    bank.restore(model, dev)
                    noise = {"sigma_w": sig, "sigma_a": sig if s.a_bits is not None else 0.0, "seed": run.seed}
                    quant.apply_scheme(model, quant.Scheme("isotropic", a_bits=None), None, seed=run.seed, noise=noise)
                    return perplexity(model, windows, dev)

                if target <= base_ppl * (1 + tol):
                    mid, iso_ppl = 0.0, iso_at(0.0)
                    run.note(f"{arm}/{s.name}: no perplexity increase; isotropic control has sigma = 0")
                else:
                    lo, hi = 0.0, 0.05
                    iso_ppl = iso_at(hi)
                    for _ in range(12):
                        if iso_ppl >= target:
                            break
                        lo, hi = hi, hi * 2
                        iso_ppl = iso_at(hi)
                    mid = hi
                    for _ in range(cfg["isotropic"]["bisect_steps"]):
                        if abs(iso_ppl - target) <= tol * target:
                            break
                        mid = (lo + hi) / 2
                        iso_ppl = iso_at(mid)
                        if iso_ppl < target:
                            lo = mid
                        else:
                            hi = mid
                corr_i = popqa_eval(model, tok, items, shots, cfg, dev)
                _, dec_i = decile_table(items, corr_i, n_dec)
                drop_i = base_dec - dec_i
                rows.append({"model": name, "arm": arm, "scheme": f"isotropic_for_{s.name}", "ppl": iso_ppl,
                             "dppl": iso_ppl - base_ppl, "acc": float(corr_i.mean()), "sigma": mid,
                             "drop_rare": float(drop_i[0]), "drop_frequent": float(drop_i[-1]),
                             "rare_minus_frequent": float(drop_i[0] - drop_i[-1])})
        bank.restore(model, dev)
        del model, bank
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.doc_counts:
        dc = pd.read_csv(args.doc_counts)
        run.note(f"doc counts loaded for {len(dc)} subjects; join on 'subj' in popqa_items.csv for the count analysis")
    runtime.write_csv(run.out_dir / "schemes.csv", rows)
    runtime.write_csv(run.out_dir / "deciles.csv", dec_rows)
    runtime.write_csv(run.out_dir / "popqa_items.csv", item_rows)
    runtime.write_csv(run.out_dir / "activations.csv", act_rows)
    runtime.finish_run(run, extra={"n_scheme_rows": len(rows)})


if __name__ == "__main__":
    main()
