import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.linewidth":0.6,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
BLUE,ORANGE,GREEN,GREY="#0072B2","#E69F00","#009E73","#999999"
VERM="#D55E00"

d=json.load(open("results/detect_results.json"))
SCRIPTS=["en","uz-latn","uz-cyrl","mixed"]
LAB={"en":"English","uz-latn":"Uzbek\nLatin","uz-cyrl":"Uzbek\nCyrillic","mixed":"Mixed"}
DET=[("our_l1","L1 markers (ours)",GREY),
     ("deberta","DeBERTa [SOTA]",BLUE),
     ("distilbert","DistilBERT [SOTA]",ORANGE),
     ("llm_judge","LLM-judge [SOTA]",GREEN)]
res=d["corpus"]["results"]

fig,ax=plt.subplots(figsize=(4.4,2.6))
x=np.arange(len(SCRIPTS)); w=0.2
for i,(key,label,col) in enumerate(DET):
    vals=[res[key][s]["f1"] for s in SCRIPTS]
    ax.bar(x+(i-1.5)*w, vals, w, label=label, color=col, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in SCRIPTS])
ax.set_ylabel("Detection F1"); ax.set_ylim(0,1.05)
ax.grid(axis="y",color="#dddddd",linewidth=0.5,zorder=0)
ax.legend(fontsize=6.6,frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.0))
fig.savefig("figures/fig7_detector_f1.png"); plt.close(fig)
print("saved figures/fig7_detector_f1.png")
