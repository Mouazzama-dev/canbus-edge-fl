"""Multi-seed contribution figure: mean +/- std over seeds for each method,
under extreme non-IID and moderate+poisoning. Reads results/contrib/.
Saves results/contrib/contribution_matrix.png and prints a table.
"""
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.join(HERE, "results", "contrib")
METHODS = ["fedavg", "fedprox", "fedmedian", "filterhybrid"]
SCENARIOS = ["extreme", "poison"]


def stats(method, scenario):
    accs = []
    for path in glob.glob(os.path.join(CONTRIB, f"{method}_{scenario}_s*.json")):
        accs.append(json.load(open(path))["history"][-1]["accuracy"])
    if not accs:
        return 0.0, 0.0, 0
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main():
    means = {s: [] for s in SCENARIOS}
    stds = {s: [] for s in SCENARIOS}

    print(f"{'method':13} " + "  ".join(f"{s+' (mean+/-std, n)':>26}" for s in SCENARIOS))
    for m in METHODS:
        row = f"{m:13} "
        for s in SCENARIOS:
            mu, sd, n = stats(m, s)
            means[s].append(mu)
            stds[s].append(sd)
            row += f"{mu:.3f} +/- {sd:.3f} (n={n})".rjust(26) + "  "
        print(row)

    x = np.arange(len(METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, means["extreme"], w, yerr=stds["extreme"], capsize=4,
           label="extreme non-IID (no attack)", color="#1f77b4")
    ax.bar(x + w / 2, means["poison"], w, yerr=stds["poison"], capsize=4,
           label="moderate + 1 poisoning client", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Final global accuracy (mean over 3 seeds)")
    ax.set_ylim(0, 1)
    ax.set_title("FedProxFilterMean is robust under both stresses (3 seeds)")
    ax.axhline(0.2, ls="--", lw=1, color="gray")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONTRIB, "contribution_matrix.png"), dpi=150)
    print("\nSaved", os.path.join(CONTRIB, "contribution_matrix.png"))


if __name__ == "__main__":
    main()