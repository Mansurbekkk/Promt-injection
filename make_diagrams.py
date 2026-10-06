"""Conceptual diagrams: Fig.1 system + threat model, Fig.6 method/scanner.
Box-and-arrow, print-safe, matching the data figures' style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
TRUST, UNTRUST = "#E8F0F7", "#FBEEDD"   # light fills
os.makedirs("figures", exist_ok=True)


def box(ax, x, y, w, h, text, fc="white", ec="#333333", fs=8.5, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


# ================= Fig 1: system + threat model =================
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")

box(ax, 0.2, 2.2, 1.6, 1.0, "User", fc=TRUST, fs=8.5)
box(ax, 2.6, 2.0, 2.4, 1.4, "LLM-integrated\napplication\n(system prompt)", fc=TRUST, fs=8)
box(ax, 6.0, 3.3, 2.0, 1.1, "Language\nmodel (M)", fc="white", fs=8)
box(ax, 5.8, 0.6, 2.4, 1.1, "Tools / APIs\n(read, send, fetch)", fc="white", fs=8)
box(ax, 9.3, 3.3, 2.6, 1.1, "Retrieved content\n(docs, web, email)", fc=UNTRUST, ec=ORANGE, fs=8)
box(ax, 9.3, 0.6, 2.6, 1.1, "Attacker-controlled\ninjected instruction", fc=UNTRUST, ec=VERM, fs=8)

arrow(ax, 1.8, 2.7, 2.6, 2.7)                      # user -> app
arrow(ax, 5.0, 2.9, 6.0, 3.6)                      # app -> M
arrow(ax, 7.0, 3.3, 7.0, 1.7)                      # M -> tools
arrow(ax, 5.8, 1.1, 5.0, 2.4)                      # tools -> app (results)
arrow(ax, 9.3, 3.8, 8.0, 3.8, color=ORANGE)       # retrieved -> M
arrow(ax, 9.3, 1.1, 8.2, 1.1, color=VERM)         # injected -> tools/context
arrow(ax, 10.6, 1.7, 10.6, 3.3, color=VERM, ls="--")  # attacker plants into retrieved

ax.text(6, 5.15, "Trusted channel (solid) vs untrusted content (shaded)", ha="center",
        fontsize=8, style="italic", color=GREY)
fig.savefig("figures/fig1_threat_model.png"); plt.close(fig)


# ================= Fig 6: method / scanner =================
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

# scanner pipeline (top)
box(ax, 0.2, 4.6, 2.3, 1.0, "Payload generator\n(6×4 variants)", fc="#EEE", fs=8)
box(ax, 2.8, 4.6, 2.3, 1.0, "Target adapter\n(RAG/email/web)", fc="#EEE", fs=8)
box(ax, 5.4, 4.6, 1.9, 1.0, "Oracle\n(goal check)", fc="#EEE", fs=8)
box(ax, 7.6, 4.6, 2.2, 1.0, "Capability gate\n(G1–G5)", fc="#EEE", fs=8)
box(ax, 10.1, 4.6, 1.7, 1.0, "Report", fc="#EEE", fs=8)
for x in (2.5, 5.1, 7.3, 9.8):
    arrow(ax, x, 5.1, x + 0.3, 5.1)

# three layers (bottom)
box(ax, 0.5, 2.6, 3.4, 1.1, "L1  Input analysis\n(normalise, classify)", fc="white", ec=BLUE, lw=1.4, fs=8)
box(ax, 4.3, 2.6, 3.4, 1.1, "L2  Trust boundary\n(context provenance)", fc="white", ec=ORANGE, lw=1.4, fs=8)
box(ax, 8.1, 2.6, 3.4, 1.1, "L3  Action monitor\n(tool-call grounding)", fc="white", ec=VERM, lw=1.4, fs=8)
arrow(ax, 3.9, 3.15, 4.3, 3.15); arrow(ax, 7.7, 3.15, 8.1, 3.15)

box(ax, 1.6, 0.4, 8.8, 1.0, "Failure localisation  λ  (first layer that observes the attack)\n"
    "+ root-cause report per successful injection", fc="#F3F7F3", ec=GREEN, lw=1.4, fs=7.5)
for x in (2.2, 6.0, 9.8):
    arrow(ax, x, 2.6, 6.0, 1.4, color=GREEN, ls="--")

arrow(ax, 6.3, 4.6, 6.0, 3.7, color="#333")       # oracle/gate feed into layers
ax.text(6, 5.78, "Black-box scanner", ha="center", fontsize=8, style="italic", color=GREY)
ax.text(6, 2.05, "Layered diagnosis", ha="center", fontsize=8, style="italic", color=GREY)
fig.savefig("figures/fig6_method.png"); plt.close(fig)

print("saved:", [f for f in os.listdir("figures") if f.startswith(("fig1", "fig6"))])
