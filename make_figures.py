"""Paper figures from the real run data. Colorblind-safe (Okabe-Ito), print-safe,
IEEE-style (serif, 300 dpi). Reads results/runs_qwen_v2.jsonl and comprehend_v2.json.
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
# Okabe-Ito
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"
SKY, VERM = "#56B4E9", "#D55E00"

R = [json.loads(l) for l in open("results/runs_qwen_v2.jsonl") if l.strip()]
C = json.load(open("results/comprehend_v2.json"))
SCRIPTS = ["en", "uz-latn", "uz-cyrl", "mixed"]
LABELS = {"en": "English", "uz-latn": "Uzbek\nLatin", "uz-cyrl": "Uzbek\nCyrillic", "mixed": "Mixed"}


def asr(subset):
    return sum(r["success"] for r in subset) / len(subset) if subset else 0.0


import os
os.makedirs("figures", exist_ok=True)

# ---- Fig 2: ASR by script (headline) ----
fig, ax = plt.subplots(figsize=(3.3, 2.4))
vals = [asr([r for r in R if r["script"] == s]) for s in SCRIPTS]
bars = ax.bar(range(4), vals, color=[VERM, ORANGE, BLUE, GREY], width=0.62, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([LABELS[s] for s in SCRIPTS])
ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 0.62)
ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
fig.savefig("figures/fig2_asr_by_script.png"); plt.close(fig)

# ---- Fig 3: comprehension vs ASR (the key result) ----
comp = defaultdict(list)
for c in C:
    comp[c["script"]].append(c["recovered"])
cvals = [sum(comp[s])/len(comp[s]) for s in SCRIPTS]
avals = [asr([r for r in R if r["script"] == s]) for s in SCRIPTS]
fig, ax = plt.subplots(figsize=(3.6, 2.5))
import numpy as np
x = np.arange(4); w = 0.38
b1 = ax.bar(x - w/2, cvals, w, label="Comprehension of the injected\ninstruction (mean recovery)",
            color=GREEN, zorder=3)
b2 = ax.bar(x + w/2, avals, w, label="Attack success rate", color=VERM, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCRIPTS])
ax.set_ylim(0, 1.18); ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5)
ax.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=1)
fig.savefig("figures/fig3_comprehension_vs_asr.png"); plt.close(fig)

# ---- Fig 4: ASR by goal class x script (grouped) ----
GC = [("text", "Text"), ("action", "Action"), ("exfil", "Exfiltration")]
fig, ax = plt.subplots(figsize=(3.8, 2.5))
x = np.arange(3); w = 0.2
colors = {"en": VERM, "uz-latn": ORANGE, "uz-cyrl": BLUE, "mixed": GREY}
for i, s in enumerate(SCRIPTS):
    vals = [asr([r for r in R if r["script"] == s and r["goal_cls"] == gc]) for gc, _ in GC]
    ax.bar(x + (i-1.5)*w, vals, w, label=LABELS[s].replace("\n", " "), color=colors[s], zorder=3)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in GC])
ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 0.75)
ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.savefig("figures/fig4_asr_by_goalclass.png"); plt.close(fig)

# ---- Fig 5: failure layer by template (stacked horizontal) ----
TEMPL = ["context_ignoring", "separators", "compound", "fake_system", "special_case", "polite"]
lay = defaultdict(lambda: defaultdict(int))
for r in R:
    if r["success"]:
        lay[r["template"]][r["layer"]] += 1
fig, ax = plt.subplots(figsize=(4.0, 2.6))
y = np.arange(len(TEMPL))[::-1]
l1 = [lay[t][1] for t in TEMPL]; l2 = [lay[t][2] for t in TEMPL]; l3 = [lay[t][3] for t in TEMPL]
ax.barh(y, l1, color=BLUE, label="L1 input", zorder=3)
ax.barh(y, l2, left=l1, color=ORANGE, label="L2 trust boundary", zorder=3)
ax.barh(y, l3, left=[a+b for a, b in zip(l1, l2)], color=VERM, label="L3 action", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(TEMPL)
ax.set_xlabel("Successful attacks"); ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
ax.set_xlim(0, 44)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig("figures/fig5_failure_layer.png"); plt.close(fig)

print("saved:", os.listdir("figures"))
