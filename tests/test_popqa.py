"""PopQA deciles and the trend of the accuracy drop on log popularity (common/popqa.py)."""
import numpy as np
import pandas as pd

from common import popqa


def _items(rng, n=6000, slope=-0.05, easy_rare=False, base_trend=0.03):
    lp = rng.uniform(1, 6, n)                                           # log10 s_pop
    prop = np.where(rng.random(n) < 0.5, "country", "director")
    if easy_rare:                                                       # easy relation piles up among rare items
        prop = np.where(lp < 2, np.where(rng.random(n) < 0.8, "country", "director"), prop)
    p_base = np.where(prop == "country", 0.6, 0.2) + base_trend * lp
    base = rng.random(n) < p_base
    p_drop = np.clip(0.3 + slope * lp, 0, 1)                            # chance a correct answer is lost
    cur = base & (rng.random(n) >= p_drop)
    return pd.DataFrame({"s_pop": 10 ** lp, "prop": prop, "base": base.astype(float), "cur": cur.astype(float)})


def test_trend_recovers_the_slope_of_the_drop():
    rng = np.random.default_rng(0)
    it = _items(rng)
    it["decile"] = popqa.decile_index(it.s_pop.to_numpy(), 10)
    rows = {(r["method"], r["measure"]): r for r in popqa.trend(it, 10)}
    rel = rows[("deciles", "rel")]
    assert abs(rel["slope"] - (-0.05)) < 3 * rel["se"] + 0.01           # relative drop = p_drop
    for key in (("deciles", "abs"), ("items_relation_fe", "abs"), ("deciles_relation_std", "abs")):
        assert rows[key]["slope"] < 0 and rows[key]["z"] < -2
    assert set(m for m, _ in rows) == set(popqa.METHODS)


def test_relation_standardizing_removes_a_composition_effect():
    rng = np.random.default_rng(1)
    it = _items(rng, n=20000, slope=0.0, easy_rare=True, base_trend=0.0)  # popularity changes nothing within a relation
    it["decile"] = popqa.decile_index(it.s_pop.to_numpy(), 10)
    d, rel = popqa.composition(it, 10)
    assert d.loc[0, "share_country"] > 0.75 > d.loc[5, "share_country"]
    assert d.loc[0, "acc_base"] > d.loc[3, "acc_base"]                   # decile 0 looks easy, as in PopQA
    rows = {(r["method"], r["measure"]): r for r in popqa.trend(it, 10)}
    assert rows[("deciles", "abs")]["z"] < -2                           # the raw deciles see a spurious trend
    assert abs(rows[("items_relation_fe", "abs")]["z"]) < 3
    assert abs(rows[("deciles_relation_std", "abs")]["z"]) < 3
    assert abs(rows[("deciles_relation_std", "rel")]["slope"]) < 3 * rows[("deciles_relation_std", "rel")]["se"] + 0.005
    assert set(rel.columns) >= {"decile", "prop", "n", "acc_base"}


def test_analyze_reads_saved_item_rows_and_takes_relations_from_elsewhere():
    rng = np.random.default_rng(2)
    it = _items(rng, n=3000)
    base = pd.DataFrame({"model": "m", "arm": "baseline", "scheme": "bfloat16", "i": range(len(it)),
                         "correct": it.base.astype(bool), "s_pop": it.s_pop})
    cur = pd.DataFrame({"model": "m", "arm": "w4a4", "scheme": "uniform", "i": range(len(it)),
                        "correct": it.cur.astype(bool)})
    trends, decs, rels = popqa.analyze(pd.concat([base, cur]), 10, props=pd.Series(it.prop.to_numpy()))
    assert len(decs) == 10 and decs.n.sum() == len(it) and len(rels)
    assert set(trends.method) == set(popqa.METHODS) and (trends.scheme == "uniform").all()
    no_prop, _, _ = popqa.analyze(pd.concat([base, cur]), 10)            # first-run files: no relation types
    assert set(no_prop.method) == {"deciles"}
