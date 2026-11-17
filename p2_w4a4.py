"""P2: H6.2 as inference on open models (W4A4 primary arm, weight-only control arm).

For each model: 16-bit baseline; W4A4 with uniform, outlier-preserving and rotation (QuaRot-
style) schemes; weight-only W4 with the same three plus NF4; bits per weight matched within
0.05 (scales and 16-bit outlier columns included). For every scheme an isotropic control adds
Gaussian noise (to weights, and to activations under W4A4) scaled to give the same perplexity
increase. Measured: WikiText-2 perplexity, PopQA accuracy by subject-popularity decile (and by
pretraining document count if a CSV is given), and K, Kurt0 and zeta of activations.
Quantization is simulated: weights and activations are quantized and dequantized in PyTorch and the
ordinary floating-point kernels run, so the results concern accuracy, not speed or memory.

H6.2 reading (register): (a) uniform W4A4 loses more on rare than frequent facts; (b) outlier
preservation recovers most of it; (c) rotation recovers mean accuracy but leaves a rare-fact gap;
weight-only shows little excess. Refuted if (a) fails, if weight-only loses as much on rare
facts at equal perplexity increase, if rare-fact losses match the isotropic control, or if
rotation matches outlier preservation on rare facts.

Outputs (results/p2_w4a4/[smoke/]): schemes.csv (one row per model x arm x scheme incl.
isotropic controls), popqa_items.csv (per item correctness, with s_pop, decile and relation type),
deciles.csv, trends.csv (drop against log popularity over all deciles, see common/popqa.py),
popqa_deciles.csv and popqa_relation_deciles.csv (what each decile holds), activations.csv, run.json

Resume: every unit (a model's baseline, or one scheme with its isotropic control) is saved as it ends, in
units/<model>__<arm>__<scheme>.json, with the calibration maxima (units/<model>__absmax.pt) and progress.json
(measured hours per unit, hours left at the measured mean). Started again with the same --out, the job runs only
the units not saved (units/settings.json refuses other settings); a unit stopped in the middle runs again.

Both models load in bfloat16 (config dtype; --dtype overrides). Pythia-6.9B overflows in float16
(the first L40S run, 2026-11-15, gave a NaN baseline), and a non-finite baseline perplexity now
stops the job with an error instead of writing invalid rows.

Commands:
  python p2_w4a4.py                                  # Pythia-6.9B and OLMo-7B, ~14 GB each in bfloat16
  python p2_w4a4.py --models EleutherAI/pythia-6.9b
  python p2_w4a4.py --dtype float32                  # ~27 GB of weights each; also fits the L40S
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
from common import hf, lm, popqa, quant, runtime, spectra
from common import units as units_mod


def popqa_items(cfg, run, rng):
    pq = cfg["popqa"]
    ds, src = hf.load_rows(popqa.spec(pq), run, limit=pq["limit"])
    if ds is None:
        run.fallback("PopQA: synthetic question set with random popularity")
        items = []
        for i in range(pq["limit"] or 200):
            ans = f"w{int(rng.integers(0, 50))}"
            items.append({"question": f"What is the code of item {i}?", "answers": [ans],
                          "s_pop": float(rng.lognormal(5, 2)), "subj": f"item {i}",
                          "prop": ["country", "sport", "occupation", "author"][i % 4]})
        return items, "fallback"
    items = []
    for ex in ds:
        ans = ex["possible_answers"]
        if isinstance(ans, str):
            ans = json.loads(ans)
        items.append({"question": ex["question"], "answers": ans, "s_pop": float(ex["s_pop"]), "subj": ex["subj"],
                      "prop": ex.get("prop")})
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
    d = popqa.decile_index(np.array([it["s_pop"] for it in items]), n_dec)
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


def scheme_unit(name, arm, s, sch, model, tok, items, shots, windows, absmax, bank, d_in, base_ppl, base_dec, cfg,
                run, dev) -> dict:
    """One scheme of one arm and its isotropic control at equal perplexity increase."""
    n_dec = cfg["popqa"]["deciles"]
    bits = quant.check_bits_matched(sch, d_in, cfg["bits_tolerance"])
    if not bits["matched"]:
        raise SystemExit(f"bits not matched within {cfg['bits_tolerance']}: {bits}")
    bank.restore(model, dev)
    quant.apply_scheme(model, s, absmax, seed=run.seed)
    ppl = perplexity(model, windows, dev)
    corr = popqa_eval(model, tok, items, shots, cfg, dev)
    _, dec = decile_table(items, corr, n_dec)
    drop = base_dec - dec
    if not math.isfinite(ppl):
        run.note(f"WARNING {name} {arm}/{s.name}: perplexity {ppl} (not finite)")
    row = {"model": name, "arm": arm, "scheme": s.name, "ppl": ppl, "dppl": ppl - base_ppl,
           "acc": float(corr.mean()), "bits": s.bits_per_weight(d_in),
           "drop_rare": float(drop[0]), "drop_frequent": float(drop[-1]),
           "rare_minus_frequent": float(drop[0] - drop[-1])}
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
    row_iso = {"model": name, "arm": arm, "scheme": f"isotropic_for_{s.name}", "ppl": iso_ppl,
               "dppl": iso_ppl - base_ppl, "acc": float(corr_i.mean()), "sigma": mid,
               "drop_rare": float(drop_i[0]), "drop_frequent": float(drop_i[-1]),
               "rare_minus_frequent": float(drop_i[0] - drop_i[-1])}
    return {"row": row, "row_iso": row_iso, "correct": bits_str(corr), "correct_iso": bits_str(corr_i)}


def bits_str(correct) -> str:
    return "".join("1" if c else "0" for c in correct)


def from_bits(b: str) -> np.ndarray:
    return np.array([c == "1" for c in b], dtype=bool)


def unit_keys(name: str, cfg: dict) -> list[str]:
    """The units of one model: the baseline, then every scheme with its isotropic control, in config order."""
    m = name.replace("/", "__")
    return [f"{m}__baseline"] + [f"{m}__{arm}__{s['name']}" for arm, schemes in cfg["arms"].items() for s in schemes]


def main():
    ap = runtime.base_parser(__doc__)
    ap.add_argument("--models", default=None, help="comma-separated model ids (default: config)")
    ap.add_argument("--doc-counts", default=None, help="CSV with columns subj,doc_count (optional)")
    ap.add_argument("--dtype", default=None, help="model dtype (default: config, bfloat16 in full runs)")
    args = ap.parse_args()
    cfg = config.get("p2_w4a4", args.smoke)
    if args.dtype:
        cfg["dtype"] = args.dtype
    run = runtime.start_run("p2_w4a4", cfg, args)
    dev = run.device
    dtype = getattr(torch, cfg["dtype"])
    models = args.models.split(",") if args.models else cfg["models"]
    pq = cfg["popqa"]
    rows, item_rows, dec_rows, act_rows = [], [], [], []
    # every unit (a model's baseline, or one scheme with its isotropic control) is saved as it ends
    # (units/<model>__<arm>__<scheme>.json); a restart with the same --out runs only the others
    units = units_mod.Units(run.out_dir, {"cfg": cfg, "seed": run.seed, "smoke": run.smoke}, "p2_w4a4")
    units.plan([k for name in models for k in unit_keys(name, cfg)])
    units.write_progress()

    def outputs():
        runtime.write_csv(run.out_dir / "schemes.csv", rows)

    for name in models:
        keys = unit_keys(name, cfg)
        # PopQA items are real data (no random numbers); the synthetic stand-in of smoke runs draws from the model's
        # own seed, so a model's items do not depend on the models before it
        items, isrc = popqa_items(cfg, run, units_mod.rng_for(units_mod.seed_for(run.seed, name, "popqa")))
        shots, items = items[:pq["shots"]], items[pq["shots"]:]
        n_dec = pq["deciles"]
        todo = [k for k in keys if not units.has(k)]
        model = tok = bank = absmax = windows = None
        if todo:
            model, tok, src = hf.load_causal_lm(name, None, run, dtype=dtype, device=dev)
            vocab = model.get_output_embeddings().weight.shape[0]
            ppl_texts, _ = hf.load_texts(cfg["ppl"]["dataset"], n=4000, run=run, field="text", min_chars=1, seed=run.seed)
            all_ids = hf.tokenize_stream(["\n".join(ppl_texts)], tok, cfg["ppl"]["ctx"], 10 ** 9 if not cfg["ppl"]["max_windows"] else cfg["ppl"]["max_windows"])
            windows = all_ids.clamp(max=vocab - 1)
            cal_texts, _ = hf.load_texts(cfg["calib"]["dataset"], n=2000, run=run, min_chars=50, seed=run.seed + 1)
            calib = hf.tokenize_stream(cal_texts, tok, cfg["calib"]["len"], cfg["calib"]["n_seqs"]).clamp(max=vocab - 1)
            probe = hf.tokenize_stream(cal_texts[::-1], tok, cfg["probe"]["len"], cfg["probe"]["n_seqs"]).clamp(max=vocab - 1)
        else:
            run.note(f"{name}: all {len(keys)} units saved; not loaded")

        # baseline
        bkey = keys[0]
        base = units.get(bkey)
        if base is None:
            units.start(bkey)
            base_ppl = perplexity(model, windows, dev)
            if not math.isfinite(base_ppl):
                raise SystemExit(f"{name}: baseline perplexity is {base_ppl} in {cfg['dtype']} (overflow?); "
                                 "no rows written. Try --dtype bfloat16 or float32.")
            base_correct = popqa_eval(model, tok, items, shots, cfg, dev)
            hs = lm.hidden_states(model, probe, dev)
            acts = [{"model": name, "layer": li, **spectra.activation_stats(H)} for li, H in enumerate(hs)]
            del hs
            run.note(f"{name} baseline ({cfg['dtype']}): ppl {base_ppl:.3f} acc {float(base_correct.mean()):.3f}")
            base = {"row": {"model": name, "arm": "baseline", "scheme": cfg["dtype"], "ppl": base_ppl,
                            "acc": float(base_correct.mean()), "bits": 16.0, "dtype": cfg["dtype"], "source": src,
                            "popqa_source": isrc},
                    "correct": bits_str(base_correct), "activations": acts}
            units.put(bkey, base)
        base_ppl = float(base["row"]["ppl"])
        base_correct = from_bits(base["correct"])
        dec_idx, base_dec = decile_table(items, base_correct, n_dec)
        act_rows += base["activations"]
        rows.append(base["row"])
        item_rows += [{"model": name, "arm": "baseline", "scheme": base["row"]["scheme"], "i": i, "correct": bool(c),
                       "s_pop": it["s_pop"], "decile": int(d), "prop": it.get("prop")}
                      for i, (c, it, d) in enumerate(zip(base_correct, items, dec_idx))]
        outputs()
        if model is not None and len(todo) > (bkey in todo):
            # calibration once per model folder: a restart quantizes with the same activation maxima
            am_path = units.dir / f"{bkey[:-len('__baseline')]}__absmax.pt"
            if am_path.exists():
                absmax = {k: v.to(dev) for k, v in torch.load(am_path, map_location="cpu").items()}
            else:
                absmax = quant.calibrate_absmax(model, [calib[i:i + 1].long() for i in range(len(calib))], dev)
                tmp = am_path.with_name(am_path.name + ".tmp")
                torch.save({k: v.cpu() for k, v in absmax.items()}, tmp)
                tmp.replace(am_path)
            bank = WeightBank(model)
            d_in = max(l.in_features for l in quant.linear_layers(model).values())

        ki = 1
        for arm, schemes in cfg["arms"].items():
            sch = [quant.Scheme(**s) for s in schemes]
            for s in sch:
                key = keys[ki]
                ki += 1
                unit = units.get(key)
                if unit is None:
                    units.start(key)
                    unit = scheme_unit(name, arm, s, sch, model, tok, items, shots, windows, absmax, bank, d_in,
                                       base_ppl, base_dec, cfg, run, dev)
                    units.put(key, unit)
                corr, corr_i = from_bits(unit["correct"]), from_bits(unit["correct_iso"])
                _, dec = decile_table(items, corr, n_dec)
                rows += [unit["row"], unit["row_iso"]]
                dec_rows += [{"model": name, "arm": arm, "scheme": s.name, "decile": k, "acc": float(dec[k]),
                              "acc_base": float(base_dec[k])} for k in range(n_dec)]
                item_rows += [{"model": name, "arm": arm, "scheme": s.name, "i": i, "correct": bool(c)}
                              for i, c in enumerate(corr)]
                item_rows += [{"model": name, "arm": arm, "scheme": f"isotropic_for_{s.name}", "i": i,
                               "correct": bool(c)} for i, c in enumerate(corr_i)]
                outputs()
        if model is not None:
            if bank is not None:
                bank.restore(model, dev)
            del model, bank
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()

    if args.doc_counts:
        dc = pd.read_csv(args.doc_counts)
        run.note(f"doc counts loaded for {len(dc)} subjects; join on 'subj' in popqa_items.csv for the count analysis")
    # the drop against log popularity over all deciles (absolute and relative), and what each decile holds
    trends, decs, rels = popqa.analyze(pd.DataFrame(item_rows), pq["deciles"])
    if len(trends):
        main_tr = trends[trends.method == "deciles"].pivot_table(index=["model", "arm", "scheme"], columns="measure",
                                                                 values=["slope", "se"])
        main_tr.columns = [f"trend_{m}_{v}" for v, m in main_tr.columns]
        rows = [{**r, **(main_tr.loc[(r["model"], r["arm"], r["scheme"])].to_dict()
                         if (r["model"], r["arm"], r["scheme"]) in main_tr.index else {})} for r in rows]
    runtime.write_csv(run.out_dir / "schemes.csv", rows)
    runtime.write_csv(run.out_dir / "deciles.csv", dec_rows)
    runtime.write_csv(run.out_dir / "popqa_items.csv", item_rows)
    runtime.write_csv(run.out_dir / "trends.csv", trends.to_dict("records"))
    runtime.write_csv(run.out_dir / "popqa_deciles.csv", decs.to_dict("records"))
    runtime.write_csv(run.out_dir / "popqa_relation_deciles.csv", rels.to_dict("records"))
    runtime.write_csv(run.out_dir / "activations.csv", act_rows)
    runtime.finish_run(run, extra={"n_scheme_rows": len(rows)})


if __name__ == "__main__":
    main()
