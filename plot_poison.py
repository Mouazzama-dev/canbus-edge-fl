"""Poisoning comparison: final accuracy with vs without 1 malicious client.
Reads results/dirichlet/ (clean) and results/poison/ (attacked).
Saves results/poison/poison_barchart.png.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
STRATS = ["fedavg", "krum", "fedmedian"]


def final(folder, strat):
    path = os.path.join(RES, folder, f"{strat}.json")
    if not os.path.exists(path):
        return 0.0
    return json.load(open(path))["history"][-1]["accuracy"]


def main():
    clean = [final("dirichlet", s) for s in STRATS]
    attacked = [final("poison", s) for s in STRATS]
    x = np.arange(len(STRATS))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, clean, w, label="no attack", color="#2ca02c")
    b2 = ax.bar(x + w / 2, attacked, w, label="1 malicious client", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATS)
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Robustness to a poisoning attack (moderate non-IID)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "poison", "poison_barchart.png"), dpi=150)
    print("Saved", os.path.join(RES, "poison", "poison_barchart.png"))


if __name__ == "__main__":
    main()