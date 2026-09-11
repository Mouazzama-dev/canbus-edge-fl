"""Cross-regime summary bar chart: final accuracy per strategy,
extreme vs moderate non-IID. Saves results/summary_barchart.png.
"""
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
STD = ["fedavg", "fedprox", "fedadam", "fedyogi", "fedavgm", "fedmedian", "krum"]


def finals(regime):
    out = {}
    for f in glob.glob(os.path.join(RES, regime, "*.json")):
        d = json.load(open(f))
        if d["strategy"] in STD:
            out[d["strategy"]] = d["history"][-1]["accuracy"]
    return out


def main():
    ex, di = finals("extreme"), finals("dirichlet")
    order = sorted(set(ex) | set(di), key=lambda s: -di.get(s, 0))
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, [ex.get(s, 0) for s in order], w,
                label="extreme non-IID", color="#d62728")
    b2 = ax.bar(x + w / 2, [di.get(s, 0) for s in order], w,
                label="moderate non-IID", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("FL strategy vs non-IID severity (CAN intrusion detection)")
    ax.axhline(0.2, ls="--", lw=1, color="gray")
    ax.text(len(order) - 0.5, 0.21, "random (0.20)", fontsize=8, color="gray", ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "summary_barchart.png"), dpi=150)
    print("Saved", os.path.join(RES, "summary_barchart.png"))


if __name__ == "__main__":
    main()