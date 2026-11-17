"""Document boundaries inside P1's surprisal windows, rebuilt on the CPU for H_link's piece control (Endri,
2026-11-17; part of the same exploratory test, fixed before any H_link value on real data was computed).

P1 (p1_checkpoints.py, gpu_pass) builds its token-surprisal windows once per tokenizer and job process (data_cache):
    ev_texts = hf.load_texts(cfg["eval_dataset"], n=max(16, eval_seqs * eval_len // 400 + 16), run, seed=run.seed + 1)
    sur = hf.tokenize_stream(ev_texts[::-1], tok, surprisal_len, surprisal_docs)
and saves only the losses (samples/htok.npy). The positions of the EOS tokens that tokenize_stream puts after each
document are not saved. They are rebuilt here from the same inputs: the dataset at the commit the run recorded
(run.json pinned_commits, else pins.json; nothing is pinned or written to pins.json), the folder's run.json config
and seed, the tokenizer at the checkpoint's commit, and hf.tokenize_stream itself (stream() repeats it, records where
each document ends, and its ids are checked equal to hf.tokenize_stream's).

A rebuild is used only if it reproduces the job's saved losses. The checkpoints of a folder are grouped by the stream
they were given: the tokenizer files of every (repo, commit) are identified by their Hub blob ids, each distinct
tokenizer builds its stream, and since the job's data_cache reused the first tokenizer it loaded for a repo for every
revision of that repo, a repo whose revisions give different streams is ambiguous (no piece control). In each group
the checkpoint with the fewest parameters (at most max_params) is loaded on the CPU in float32 and lm.token_losses is
run on every window of the rebuilt stream; the group is verified if every window agrees with that checkpoint's
htok.npy (TOL: median |difference| < 0.02 nats, 99th percentile < 0.25, correlation > 0.999; a stream shifted by one
token differs by nats). Otherwise the group gets no piece control, with the reason in <out>/eos/<folder>.json and in
the H_link rows (eos_status, eos_reason).

Piece cuts: a window's losses are l_i = -log p(t_{i+1} | t_0 .. t_i), i = 0 .. L - 2. An EOS at token position e
ends a document and the next document starts at token e + 1, whose loss is l_e: a new piece starts at loss index e,
for 1 <= e <= L - 2 (e = 0: the window's own start; e = L - 1: the next window's start).

Needs torch and transformers (imported here only; h_link.py imports this module in its main process, never in its
workers) and, for checkpoints loaded from the Hub, the Hub.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

TOL = {"median_abs": 0.02, "p99_abs": 0.25, "min_corr": 0.999}
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
             "added_tokens.json", "tokenizer.model")
MAX_PARAMS = 2e9                 # OLMo 2 1B and every Pythia size fit in RAM in float32; OLMo 2 7B (29 GB) does not
RULE = "new piece at loss index e for an EOS at token position e, 1 <= e <= L - 2"


class _Quiet:
    """The few fields of runtime.Run that hf.load_texts reads: no fallback, pins required (nothing is pinned)."""
    offline_ok = False
    require_pins = True

    def __init__(self, log):
        self.pins, self._log = {}, log

    def note(self, msg):
        self._log(f"  {msg}")

    def fallback(self, what):
        self._log(f"  FALLBACK {what}")


# ----------------------------------------------------------------------------- the stream
def eos_id(tok) -> int:
    return getattr(tok, "eos_token_id", None) or 0          # as hf.tokenize_stream


def stream(texts: list[str], tok, seq_len: int, n_seqs: int) -> tuple[np.ndarray, list[list[int]]]:
    """hf.tokenize_stream's windows [n, seq_len] and, per window, the token positions of the EOS tokens it appended
    after each document."""
    ids: list[int] = []
    ends: list[int] = []
    eos = eos_id(tok)
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False))
        ids.append(eos)
        ends.append(len(ids) - 1)
        if len(ids) >= seq_len * n_seqs:
            break
    n = min(n_seqs, len(ids) // seq_len)
    if n < 1:
        raise ValueError("not enough text for one sequence")
    arr = np.asarray(ids[: n * seq_len], dtype=np.int64).reshape(n, seq_len)
    pos = [[e - w * seq_len for e in ends if w * seq_len <= e < (w + 1) * seq_len] for w in range(n)]
    return arr, pos


def cuts_from_eos(eos_pos: list[list[int]], seq_len: int) -> list[list[int]]:
    """Loss indices where a new document piece starts (RULE)."""
    return [sorted(int(e) for e in ps if 1 <= e <= seq_len - 2) for ps in eos_pos]


def stream_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest()


def compare(saved: np.ndarray, mine: np.ndarray) -> dict:
    """Agreement of the saved losses with the CPU ones, per window [n, L - 1]; ok if every window is within TOL."""
    d = np.abs(saved - mine)
    med = np.median(d, axis=1)
    p99 = np.quantile(d, 0.99, axis=1)
    corr = np.array([np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float(np.array_equal(a, b))
                     for a, b in zip(saved, mine)])
    ok = bool(np.all(med < TOL["median_abs"]) and np.all(p99 < TOL["p99_abs"]) and np.all(corr > TOL["min_corr"]))
    return {"ok": ok, "windows": int(len(d)), "median_abs_max": float(med.max()), "p99_abs_max": float(p99.max()),
            "corr_min": float(np.nanmin(corr)) if len(corr) else float("nan"), "max_abs": float(d.max())}


# ----------------------------------------------------------------------------- one folder
def folder_inputs(folder: Path) -> dict:
    """What P1 built its windows from, as the folder's run.json records it."""
    rj = json.loads((folder / "run.json").read_text())
    cfg = rj.get("config", {})
    need = ("eval_dataset", "eval_seqs", "eval_len", "surprisal_len", "surprisal_docs")
    missing = [k for k in need if k not in cfg]
    if missing:
        raise KeyError(f"run.json config lacks {missing}")
    return {"eval_dataset": cfg["eval_dataset"],
            "n_texts": max(16, int(cfg["eval_seqs"]) * int(cfg["eval_len"]) // 400 + 16),
            "seq_len": int(cfg["surprisal_len"]), "n_seqs": int(cfg["surprisal_docs"]), "seed": int(rj.get("seed", 0)),
            "smoke": bool(rj.get("smoke", False)), "dtype": cfg.get("dtype", "float32"),
            "pinned": rj.get("pinned_commits", {}) or {}, "versions": rj.get("versions", {})}


def checkpoints(folder: Path) -> list[dict]:
    out = []
    for rj in sorted((folder / "checkpoints").glob("*/*/row.json")):
        d = rj.parent
        f = d / "samples" / "htok.npy"
        if f.exists():
            row = json.loads(rj.read_text()).get("row", {})
            out.append({"key": f"{folder.name}/{d.parent.name}/{d.name}", "htok": str(f), "repo": row.get("repo"),
                        "revision": row.get("revision"), "commit": row.get("commit"), "source": row.get("source"),
                        "n_params": row.get("n_params")})
    return out


def _bytes_fp() -> str:
    from . import hf
    return f"bytes:{hf.TINY_NEOX['vocab_size']}"


def fingerprint(c: dict, cache: dict, log) -> str:
    """The tokenizer's identity: the blob ids of its files at the checkpoint's commit (Hub metadata, cached), or
    the byte stand-in of a fallback checkpoint."""
    if c["source"] == "fallback":
        return _bytes_fp()
    from . import hf
    commit = c["commit"] or (hf.PINS.entry(c["repo"], c["revision"]) or {}).get("commit")
    if not commit:
        raise KeyError(f"{c['repo']}@{c['revision']}: no commit in row.json or pins.json")
    k = f"{c['repo']}@{commit}"
    if k not in cache:
        if not hf.hub_reachable():
            raise ConnectionError("Hub unreachable")
        from huggingface_hub import HfApi
        files = hf.hub_retry(lambda: HfApi().get_paths_info(c["repo"], list(TOK_FILES), revision=commit),
                             None, f"tokenizer files of {k}")
        ids = sorted((f.path, getattr(f, "blob_id", None) or "") for f in files)
        if not ids:
            raise FileNotFoundError(f"{k}: no tokenizer files")
        cache[k] = "hub:" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    return cache[k]


def load_tokenizer(fp: str, c: dict):
    from . import hf
    if fp.startswith("bytes:"):
        return hf.ByteTokenizer(int(fp.split(":")[1]))
    from transformers import AutoTokenizer
    commit = c["commit"] or hf.PINS.entry(c["repo"], c["revision"])["commit"]
    return hf.hub_retry(lambda: AutoTokenizer.from_pretrained(c["repo"], revision=commit), None,
                        f"tokenizer of {c['repo']}@{commit[:10]}")


def load_model(c: dict):
    """The checkpoint on the CPU in float32, and whether it has to be removed from the Hub cache afterwards."""
    import torch
    from . import hf
    if c["source"] == "fallback":                           # as hf.load_causal_lm's stand-in
        seed = zlib.crc32(f"{c['repo']}@{c['revision']}".encode()) % (2**31)
        return hf.neox_from_config(dict(hf.TINY_NEOX), seed=seed).to(torch.float32).eval(), False
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM
    commit = c["commit"] or hf.PINS.entry(c["repo"], c["revision"])["commit"]
    cached = isinstance(try_to_load_from_cache(c["repo"], "config.json", revision=commit), str)
    m = hf.hub_retry(lambda: AutoModelForCausalLM.from_pretrained(c["repo"], revision=commit, dtype=torch.float32,
                                                                  attn_implementation="eager"),
                     None, f"download of {c['repo']}@{commit[:10]}")
    return m.eval(), not cached


def texts_candidates(inp: dict, log) -> dict:
    """{"hub": texts} when the Hub answers (the dataset at the run's commit), {"fallback": texts} for a smoke run
    (hf.zipf_text, as load_texts' stand-in); {"<source>_error": reason} otherwise."""
    from . import hf
    out = {}
    spec = hf.data_spec(inp["eval_dataset"])
    if hf.hub_reachable():
        try:
            k = hf.Pins.key(spec.repo, spec.revision, "dataset")
            sha = inp["pinned"].get(k) or (hf.PINS.entry(spec.repo, spec.revision, "dataset") or {}).get("commit")
            if not sha:
                raise KeyError(f"{k} is in neither run.json nor pins.json")
            texts, _ = hf.load_texts(dataclasses.replace(spec, revision=sha), n=inp["n_texts"], run=_Quiet(log))
            out["hub"] = texts
        except Exception as e:  # noqa: BLE001  reported per group
            out["hub_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    else:
        out["hub_error"] = "Hub unreachable"
    if inp["smoke"]:
        out["fallback"] = hf.zipf_text(inp["n_texts"], 400, inp["seed"] + 1)
    return out


def rebuild_folder(folder: Path, out_json: Path, max_params: float = MAX_PARAMS, threads: int = 4,
                   retry: bool = False, log=print) -> dict:
    """Rebuild and verify the EOS positions of one P1 folder; returns {checkpoint key: {"status", "reason",
    "cuts", "group"}} and saves everything to out_json (reused by later runs: verified groups are kept, failed
    ones are checked again after an error, with a larger max_params, or with retry)."""
    from . import hf
    t0 = time.perf_counter()
    prev = json.loads(out_json.read_text()) if out_json.exists() else {}
    fps: dict = dict(prev.get("fingerprints", {}))
    streams: dict = dict(prev.get("streams", {}))               # fingerprint -> {texts source: sha}
    groups: dict = dict(prev.get("groups", {}))
    res: dict = {}
    try:
        inp = folder_inputs(folder)
    except Exception as e:  # noqa: BLE001
        reason = f"run.json: {type(e).__name__}: {str(e)[:160]}"
        return {c["key"]: {"status": "unverified", "reason": reason, "cuts": None, "group": None}
                for c in checkpoints(folder)}
    if prev.get("inputs") != json.loads(json.dumps({k: v for k, v in inp.items() if k != "pinned"})):
        fps, streams, groups = {}, {}, {}                        # the folder was measured again: start afresh
    cks = checkpoints(folder)
    fp_of: dict = {}
    for c in cks:
        try:
            fp_of[c["key"]] = fingerprint(c, fps, log)
        except Exception as e:  # noqa: BLE001
            res[c["key"]] = {"status": "unverified", "reason": f"error: tokenizer files: {type(e).__name__}: "
                                                                f"{str(e)[:160]}", "cuts": None, "group": None}
    texts = None
    toks: dict = {}

    def need_texts():
        nonlocal texts
        if texts is None:
            texts = texts_candidates(inp, log)
        return texts

    rep = {}                                                    # a checkpoint per fingerprint, to load it from
    for c in cks:
        if c["key"] in fp_of:
            rep.setdefault(fp_of[c["key"]], c)
    for fp, c in rep.items():
        if fp in streams and "error" not in streams[fp]:
            continue
        try:
            tok = toks.setdefault(fp, load_tokenizer(fp, c))
            streams.setdefault(fp, {})
            for s, tx in need_texts().items():
                if not s.endswith("_error"):
                    arr, _ = stream(tx[::-1], tok, inp["seq_len"], inp["n_seqs"])
                    streams[fp][s] = stream_sha(arr)
            if not streams[fp]:                                 # no texts at all: try again next run
                streams[fp] = {"error": "no texts: " + "; ".join(f"{k}: {v}" for k, v in need_texts().items())}
        except Exception as e:  # noqa: BLE001
            streams[fp] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    # the job's stream for a checkpoint: that of the first tokenizer it loaded for the repo, so every tokenizer of
    # the repo must give the same stream
    by_repo: dict = {}
    for c in cks:
        if c["key"] in fp_of:
            by_repo.setdefault(c["repo"], set()).add(fp_of[c["key"]])
    gid_of = {}
    for c in cks:
        if c["key"] not in fp_of:
            continue
        sigs = {json.dumps(streams.get(f, {}), sort_keys=True) for f in by_repo[c["repo"]]}
        if len(sigs) > 1:
            res[c["key"]] = {"status": "unverified", "group": None, "cuts": None,
                             "reason": f"ambiguous: the tokenizers of {c['repo']}'s revisions give different streams "
                                       f"and the job used the first one it loaded"}
            continue
        sig = sigs.pop()
        if "error" in json.loads(sig):
            res[c["key"]] = {"status": "unverified", "group": None, "cuts": None,
                             "reason": f"error: stream: {json.loads(sig)['error']}"}
            continue
        gid_of[c["key"]] = hashlib.sha256(sig.encode()).hexdigest()[:12]
    members: dict = {}
    for c in cks:
        if c["key"] in gid_of:
            members.setdefault(gid_of[c["key"]], []).append(c)
    for gid, cs in members.items():
        g = groups.get(gid)
        redo = g is None or (g["status"] != "verified" and (
            retry or g.get("kind") == "error"
            or (g.get("kind") == "too_large" and max_params > float(g.get("max_params", 0)))))
        if redo:
            groups[gid] = g = verify_group(gid, cs, fp_of, toks, need_texts, inp, max_params, threads, log)
        for c in cs:
            res[c["key"]] = {"status": g["status"], "reason": g.get("reason", ""), "group": gid,
                             "cuts": g.get("cuts") if g["status"] == "verified" else None}
    doc = {"what": "EOS positions inside P1's surprisal windows, rebuilt on the CPU (common/eos_rebuild.py)",
           "rule": RULE, "tol": TOL, "max_params": max_params,
           "inputs": {k: v for k, v in inp.items() if k not in ("pinned",)},
           "fingerprints": fps, "streams": streams, "groups": groups,
           "checkpoints": {k: {kk: v[kk] for kk in ("status", "reason", "group")} for k, v in sorted(res.items())},
           "seconds": round(time.perf_counter() - t0, 1)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(out_json)
    return res


def verify_group(gid, cs, fp_of, toks, need_texts, inp, max_params, threads, log) -> dict:
    """Load the group's smallest checkpoint on the CPU and compare its losses on the rebuilt windows with the
    saved ones."""
    import torch
    from . import hf, lm
    cs = sorted(cs, key=lambda c: (float(c["n_params"] or np.inf), c["key"]))
    c = cs[0]
    base = {"checkpoint": c["key"], "repo": c["repo"], "revision": c["revision"], "commit": c["commit"],
            "n_params": c["n_params"], "n_checkpoints": len(cs), "max_params": max_params}
    if float(c["n_params"] or np.inf) > max_params:
        return {**base, "status": "unverified", "kind": "too_large",
                "reason": f"too large: the group's smallest checkpoint has {float(c['n_params'] or np.nan):.3g} "
                          f"parameters, above --eos-max-params {max_params:.3g} (float32 on the CPU)"}
    fp = fp_of[c["key"]]
    try:
        tok = toks.get(fp) or load_tokenizer(fp, c)
        toks[fp] = tok
        tx = need_texts()
        cand = [s for s in ("hub", "fallback") if s in tx]
        if not cand:
            return {**base, "status": "unverified", "kind": "error",
                    "reason": "error: no texts: " + "; ".join(f"{k}: {v}" for k, v in tx.items())}
        saved = np.asarray(np.load(c["htok"]), dtype=np.float64)
        L = inp["seq_len"]
        torch.set_num_threads(max(1, int(threads)))
        log(f"EOS check {c['key']} ({float(c['n_params'] or 0):.3g} parameters) on the CPU")
        t = time.perf_counter()
        model, purge = load_model(c)
        tried = {}
        try:
            vocab = model.get_output_embeddings().weight.shape[0]
            for s in cand:
                arr, pos = stream(tx[s][::-1], tok, L, inp["n_seqs"])
                ref = hf.tokenize_stream(tx[s][::-1], tok, L, inp["n_seqs"]).numpy()
                if not np.array_equal(arr, ref):
                    tried[s] = {"ok": False, "why": "stream() differs from hf.tokenize_stream"}
                    continue
                if saved.size != arr.shape[0] * (L - 1):
                    tried[s] = {"ok": False, "why": f"{saved.size} saved losses, {arr.shape[0]} x {L - 1} rebuilt"}
                    continue
                ids = torch.from_numpy(np.minimum(arr, vocab - 1))
                mine = np.concatenate([lm.token_losses(model, ids[w:w + 1], "cpu", batch=1).numpy()
                                       for w in range(len(ids))]).astype(np.float64)
                cmp_ = compare(saved.reshape(len(ids), L - 1), mine)
                tried[s] = cmp_
                if cmp_["ok"]:
                    eos = eos_id(tok)
                    in_win = [[int(e) for e in np.flatnonzero(arr[w] == eos)] for w in range(len(arr))]
                    return {**base, "status": "verified", "kind": "verified", "texts": s,
                            "stream_sha256": stream_sha(arr), "check": cmp_, "tried": tried,
                            "eos_token_id": int(eos), "eos_positions": pos,
                            "other_eos_ids_in_text": int(sum(len(a) for a in in_win) - sum(len(p) for p in pos)),
                            "n_eos": [len(p) for p in pos], "cuts": cuts_from_eos(pos, L),
                            "seconds": round(time.perf_counter() - t, 1)}
        finally:
            del model
            if purge:
                hf.purge_revision(c["repo"], c["commit"])
        return {**base, "status": "unverified", "kind": "mismatch", "tried": tried,
                "reason": "mismatch: the CPU losses on the rebuilt windows do not reproduce htok.npy ("
                          + "; ".join(f"{k}: " + (v.get("why") or "largest median |d| %.3g, smallest corr %.4f"
                                                  % (v["median_abs_max"], v["corr_min"])) for k, v in tried.items())
                          + ")"}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "unverified", "kind": "error", "reason": f"error: {type(e).__name__}: {str(e)[:200]}"}
