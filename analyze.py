"""Produce the paper tables from a runs.jsonl file.

Usage: python analyze.py runs.jsonl
Prints TABLE II-style breakdowns and writes tables.json. No plotting here;
figures are drawn separately so this runs anywhere.
"""
import json
import sys
from collections import defaultdict

from piscan.metrics import compute_rates, mcnemar


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def by(recs, *keys):
    g = defaultdict(list)
    for r in recs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def rates_table(recs, split_key):
    """Per-value competence and attack-success rates."""
    out = {}
    for val, rs in sorted(by(recs, split_key).items()):
        runs = [{"competent": r["competent"], "success": r["success"], "drift": r["drift"]}
                for r in rs]
        out[val[0]] = compute_rates(runs)
    return out


def print_rates(title, table):
    print(f"\n=== {title} ===")
    print(f"{'value':14} {'n':>4} {'CR':>6} {'ASR_raw':>8} {'ASR_gat':>8} {'95% CI':>16} {'drift':>6}")
    for k, r in table.items():
        ci = f"[{r.ci_low:.2f},{r.ci_high:.2f}]" if r.asr_gated is not None else "n/r"
        gat = f"{r.asr_gated:.3f}" if r.asr_gated is not None else "n/r"
        print(f"{str(k):14} {r.n:>4} {r.cr:>6.3f} {r.asr_raw:>8.3f} {gat:>8} {ci:>16} {r.drift_rate:>6.3f}")


def layer_table(recs):
    print("\n=== Failure layer of successful injections (per app) ===")
    g = defaultdict(lambda: defaultdict(int))
    for r in recs:
        if r["success"]:
            g[r["app"]][r["layer"]] += 1
    print(f"{'app':16} {'L1':>5} {'L2':>5} {'L3':>5} {'none':>5}")
    for app, d in sorted(g.items()):
        print(f"{app:16} {d.get(1,0):>5} {d.get(2,0):>5} {d.get(3,0):>5} {d.get(None,0):>5}")


def gate_reasons(recs):
    print("\n=== Gate failures by reason ===")
    c = defaultdict(int)
    for r in recs:
        if not r["gate_pass"] and not r["competent"]:
            c[r["gate_reason"]] += 1
    for reason, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")


def main(path):
    recs = load(path)
    print(f"loaded {len(recs)} runs from {path}")
    t_script = rates_table(recs, "script")
    t_app = rates_table(recs, "app")
    print_rates("By script (all apps)", t_script)
    print_rates("By app (all scripts)", t_app)
    layer_table(recs)
    gate_reasons(recs)

    # McNemar: EN vs each Uzbek script, matched on (app, task, goal, template)
    print("\n=== McNemar: EN baseline vs variant (text-goal, matched) ===")
    key = lambda r: (r["app"], r["task"], r["goal"], r["template"])
    en = {key(r): r["success"] for r in recs if r["script"] == "en"}
    for var in ("uz-latn", "uz-cyrl", "mixed"):
        pairs = []
        for r in recs:
            if r["script"] == var and key(r) in en and (r["competent"] or True):
                pairs.append((1 if en[key(r)] else 0, 1 if r["success"] else 0))
        if pairs:
            b, c, p = mcnemar(pairs)
            print(f"  EN vs {var}: pairs={len(pairs)} b={b} c={c} p={p:.4g}")

    tables = {
        "by_script": {k: vars(v) for k, v in t_script.items()},
        "by_app": {k: vars(v) for k, v in t_app.items()},
    }
    with open("tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    print("\nwrote tables.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl")
