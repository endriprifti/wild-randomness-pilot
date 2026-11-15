"""PopQA analysis for H6.2: popularity deciles, what they contain, and the trend of the accuracy drop.

The register reads H6.2(a) as "uniform W4A4 loses more on rare than on frequent facts". Comparing
decile 0 with decile 9 alone is fragile: PopQA's relation types are not spread evenly over
popularity (the least popular decile holds many country and sport questions, whose small answer
sets are easy to guess; the most popular holds capitals), so decile 0's baseline accuracy is
above that of deciles 1 to 6. Three readings of the trend, each for the absolute drop
(base - quantized accuracy) and, where defined, the relative drop (divided by base accuracy):

  deciles               weighted least squares of the decile drop on the decile's mean
                        log10 s_pop, over all n_dec deciles (weights = items per decile);
  deciles_relation_std  the same after standardizing every decile to the overall mix of
                        relation types (relations absent from a decile are left out of its
                        weights);
  items_relation_fe     least squares over items of d_i = base_i - correct_i on log10 s_pop
                        with relation fixed effects, heteroskedasticity-robust (HC1) SE
                        (absolute drop only).

A negative slope means larger drops on rarer facts, the direction H6.2(a) predicts.

Recompute from a saved P2 folder (writes <dir>/popqa_analysis/ unless --out is given):
  python -m common.popqa results/p2_w4a4
  python -m common.popqa results/p2_w4a4 --popqa-tsv popQA.tsv   # relation types from a local copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("deciles", "deciles_relation_std", "items_relation_fe")


def decile_index(pops: np.ndarray, n_dec: int) -> np.ndarray:
    """Decile of each item by s_pop (quantile edges; ties at an edge go to the upper decile)."""
    pops = np.asarray(pops, float)
    edges = np.quantile(pops, np.linspace(0, 1, n_dec + 1))
    return np.clip(np.searchsorted(edges, pops, side="right") - 1, 0, n_dec - 1)


def composition(items: pd.DataFrame, n_dec: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per decile: size, s_pop range, ties at its lower edge, top relation types and baseline accuracy;
    and baseline accuracy per decile and relation type. items: s_pop, decile, base, prop (optional)."""
    pops = items["s_pop"].to_numpy(float)
    edges = np.quantile(pops, np.linspace(0, 1, n_dec + 1))
    rows = []
    for k in range(n_dec):
        g = items[items.decile == k]
        r = {"decile": k, "n": len(g), "s_pop_min": g.s_pop.min(), "s_pop_median": g.s_pop.median(),
             "s_pop_max": g.s_pop.max(), "lower_edge": edges[k], "n_at_lower_edge": int((pops == edges[k]).sum()),
             "acc_base": g.base.mean() if len(g) else np.nan}
        if "prop" in g and g.prop.notna().any():
            share = g.prop.value_counts(normalize=True)
            r["top_relations"] = "; ".join(f"{p} {v:.2f}" for p, v in share.head(4).items())
            for p, v in share.items():
                r[f"share_{p.replace(' ', '_')}"] = round(float(v), 4)
        rows.append(r)
    rel = pd.DataFrame()
    if "prop" in items and items.prop.notna().any():
        rel = (items.groupby(["decile", "prop"]).agg(n=("base", "size"), acc_base=("base", "mean"))
               .reset_index())
    return pd.DataFrame(rows), rel


def _wls(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan, np.nan, int(len(x))
    X = np.column_stack([np.ones_like(x), x])
    W = w / w.mean()
    XtW = X.T * W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    resid = y - X @ beta
    s2 = float((W * resid ** 2).sum() / (len(x) - 2))
    cov = s2 * np.linalg.inv(XtW @ X)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), int(len(x))


def _ols_fe_hc1(d, x, groups):
    """Slope of d on x with group fixed effects (within transformation), HC1 standard error."""
    df = pd.DataFrame({"d": d, "x": x, "g": groups}).dropna()
    if len(df) < 10 or df.x.std() == 0:
        return np.nan, np.nan, len(df)
    dm = df.d - df.groupby("g").d.transform("mean")
    xm = df.x - df.groupby("g").x.transform("mean")
    sxx = float((xm ** 2).sum())
    b = float((xm * dm).sum() / sxx)
    e = dm - b * xm
    n, k = len(df), df.g.nunique() + 1
    se = float(np.sqrt((xm ** 2 * e ** 2).sum() / sxx ** 2 * n / max(n - k, 1)))
    return b, se, n


def _decile_means(items, col, n_dec, std_weights=None):
    out = np.full(n_dec, np.nan)
    for k in range(n_dec):
        g = items[items.decile == k]
        if not len(g):
            continue
        if std_weights is None:
            out[k] = g[col].mean()
        else:
            m = g.groupby("prop")[col].mean()
            w = std_weights.reindex(m.index).fillna(0.0)
            out[k] = float((m * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    return out


def trend(items: pd.DataFrame, n_dec: int) -> list[dict]:
    """Rows for METHODS x {abs, rel}. items: s_pop, decile, base (0/1), cur (0/1), prop (optional)."""
    items = items.copy()
    items["lp"] = np.log10(items.s_pop.clip(lower=1))
    n = items.groupby("decile").size().reindex(range(n_dec)).fillna(0).to_numpy(float)
    x = items.groupby("decile").lp.mean().reindex(range(n_dec)).to_numpy(float)
    has_prop = "prop" in items and items.prop.notna().all()
    out = []
    variants = [("deciles", None)]
    if has_prop:
        variants.append(("deciles_relation_std", items.prop.value_counts(normalize=True)))
    for method, w in variants:
        b = _decile_means(items, "base", n_dec, w)
        c = _decile_means(items, "cur", n_dec, w)
        for measure, y in (("abs", b - c), ("rel", np.where(b > 0, (b - c) / np.where(b > 0, b, 1), np.nan))):
            slope, se, m = _wls(x, y, n)
            out.append({"method": method, "measure": measure, "slope": slope, "se": se, "n_units": m,
                        "drop_rare": float(y[0]), "drop_frequent": float(y[-1]),
                        "mean_drop": float(np.nanmean(y)) if np.isfinite(y).any() else np.nan})
    if has_prop:
        slope, se, m = _ols_fe_hc1((items.base - items.cur).to_numpy(float), items.lp.to_numpy(float),
                                   items.prop.to_numpy())
        out.append({"method": "items_relation_fe", "measure": "abs", "slope": slope, "se": se, "n_units": m,
                    "mean_drop": float((items.base - items.cur).mean())})
    for r in out:
        r["z"] = r["slope"] / r["se"] if r.get("se") and np.isfinite(r["se"]) and r["se"] > 0 else np.nan
        r["x"] = "log10 s_pop"
    return out


def analyze(item_rows: pd.DataFrame, n_dec: int, props: pd.Series | None = None):
    """item_rows: popqa_items.csv (baseline rows with s_pop; every scheme row with i and correct).
    Returns (trends, deciles, relation_deciles) as DataFrames."""
    trends, decs, rels = [], [], []
    for model, g in item_rows.groupby("model", sort=False):
        base = g[g.arm == "baseline"].set_index("i").sort_index()
        it = pd.DataFrame({"i": base.index, "s_pop": base.s_pop.to_numpy(float),
                           "base": base.correct.astype(float).to_numpy()})
        if "prop" in base and base.prop.notna().all():
            it["prop"] = base.prop.to_numpy()
        elif props is not None:
            it["prop"] = props.reindex(it.i).to_numpy()
        it["decile"] = decile_index(it.s_pop.to_numpy(), n_dec)
        d, r = composition(it, n_dec)
        decs.append(d.assign(model=model))
        if len(r):
            rels.append(r.assign(model=model))
        for (arm, scheme), s in g[g.arm != "baseline"].groupby(["arm", "scheme"], sort=False):
            cur = s.set_index("i").correct.astype(float).reindex(it.i).to_numpy()
            trends += [{"model": model, "arm": arm, "scheme": scheme, **row}
                       for row in trend(it.assign(cur=cur), n_dec)]
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()   # noqa: E731
    return pd.DataFrame(trends), cat(decs), cat(rels)


def load_props(shots: int, tsv: str | None = None, dataset: str = "akariasai/PopQA", split: str = "test") -> pd.Series:
    """Relation type of each evaluated item, indexed like popqa_items.csv's i (items after the shots)."""
    if tsv:
        prop = pd.read_csv(tsv, sep="\t")["prop"]
    else:
        from datasets import load_dataset
        prop = pd.Series(load_dataset(dataset, split=split)["prop"])
    return pd.Series(prop.iloc[shots:].to_numpy(), index=range(len(prop) - shots))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="a P2 results folder (popqa_items.csv, run.json)")
    ap.add_argument("--out", default=None, help="default: <results>/popqa_analysis")
    ap.add_argument("--popqa-tsv", default=None, help="local PopQA TSV for relation types (default: the Hub copy)")
    args = ap.parse_args(argv)
    res = Path(args.results)
    items = pd.read_csv(res / "popqa_items.csv")
    cfg = json.loads((res / "run.json").read_text())["config"]["popqa"] if (res / "run.json").exists() else {}
    n_dec, shots = int(cfg.get("deciles", 10)), int(cfg.get("shots", 4))
    props = None
    if "prop" not in items or items.loc[items.arm == "baseline", "prop"].isna().any():
        props = load_props(shots, args.popqa_tsv)
    trends, decs, rels = analyze(items, n_dec, props)
    out = Path(args.out) if args.out else res / "popqa_analysis"
    out.mkdir(parents=True, exist_ok=True)
    trends.to_csv(out / "trends.csv", index=False)
    decs.to_csv(out / "popqa_deciles.csv", index=False)
    rels.to_csv(out / "popqa_relation_deciles.csv", index=False)
    show = trends[(trends.measure == "abs")][["model", "arm", "scheme", "method", "slope", "se"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}/trends.csv, popqa_deciles.csv, popqa_relation_deciles.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
