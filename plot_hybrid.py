"""The contribution figure: 4 methods x 2 stress tests.
extreme non-IID (no attack) and moderate + 1 poisoning client.
Shows the hybrid is the only method strong in BOTH columns.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# METHODS = ["fedavg", "fedprox", "fedmedian", "hybrid", "trimhybrid"]
METHODS = ["fedavg", "fedprox", "fedmedian", "filterhybrid"]


def acc(folder, strat):
    path = os.path.join(RES, folder, f"{strat}.json")
    if not os.path.exists(path):
        return 0.0
    return json.load(open(path))["history"][-1]["accuracy"]


def main():
    extreme = [acc("extreme", m) for m in METHODS]      # no attack
    poison = [acc("poison", m) for m in METHODS]         # moderate + malicious

    print(f"{'method':12} {'extreme non-IID':>16} {'moderate+poison':>16}")
    for m, e, p in zip(METHODS, extreme, poison):
        print(f"{m:12} {e:16.4f} {p:16.4f}")

    x = np.arange(len(METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, extreme, w, label="extreme non-IID (no attack)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, poison, w, label="moderate + 1 poisoning client", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Hybrid is the only strategy strong under both stresses")
    ax.axhline(0.2, ls="--", lw=1, color="gray")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "hybrid_matrix.png"), dpi=150)
    print("\nSaved", os.path.join(RES, "hybrid_matrix.png"))


if __name__ == "__main__":
    main()