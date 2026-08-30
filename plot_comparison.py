"""Plot the FL strategy comparison for each regime under results/<regime>/.

For every subfolder of results/ that holds <strategy>.json files it writes
comparison_accuracy.png and comparison_loss.png into that subfolder, and
prints a final-accuracy table per regime.
"""

import glob
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

COLOURS = {
    "fedprox": "#2ca02c", "fedavg": "#1f77b4", "fedadam": "#ff7f0e",
    "fedyogi": "#d62728", "fedavgm": "#9467bd", "fedmedian": "#8c564b",
    "krum": "#e377c2",
}


def load_regime(regime_dir):
    runs = {}
    for path in glob.glob(os.path.join(regime_dir, "*.json")):
        data = json.load(open(path))
        runs[data["strategy"]] = data["history"]
    return runs


def plot_regime(regime, runs):
    order = sorted(runs, key=lambda s: -runs[s][-1]["accuracy"])
    regime_dir = os.path.join(RESULTS_DIR, regime)

    plt.figure(figsize=(8, 5))
    for s in order:
        h = runs[s]
        plt.plot([x["round"] for x in h], [x["accuracy"] for x in h],
                 marker="o", label=s, color=COLOURS.get(s))
    plt.xlabel("Federated round"); plt.ylabel("Global accuracy")
    plt.title(f"Aggregation strategies - {regime} non-IID")
    plt.ylim(0, 1); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(regime_dir, "comparison_accuracy.png"), dpi=150)

    plt.figure(figsize=(8, 5))
    for s in order:
        h = runs[s]
        plt.plot([x["round"] for x in h], [x["loss"] for x in h],
                 marker="o", label=s, color=COLOURS.get(s))
    plt.xlabel("Federated round"); plt.ylabel("Global loss (log scale)")
    plt.yscale("log"); plt.title(f"Global loss per round - {regime} non-IID")
    plt.grid(True, alpha=0.3, which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(regime_dir, "comparison_loss.png"), dpi=150)

    print(f"\n=== {regime} ===")
    print(f"{'strategy':10} {'final_acc':>9} {'best_acc':>9} {'final_loss':>11}")
    for s in order:
        h = runs[s]
        print(f"{s:10} {h[-1]['accuracy']:9.4f} "
              f"{max(x['accuracy'] for x in h):9.4f} {h[-1]['loss']:11.2f}")


def main():
    regimes = [d for d in os.listdir(RESULTS_DIR)
               if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    if not regimes:
        raise SystemExit(f"No regime subfolders in {RESULTS_DIR}")
    for regime in sorted(regimes):
        runs = load_regime(os.path.join(RESULTS_DIR, regime))
        if runs:
            plot_regime(regime, runs)


if __name__ == "__main__":
    main()