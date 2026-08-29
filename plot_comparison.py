"""Plot the FL aggregation-strategy comparison from results/*.json.

Produces:
  results/comparison_accuracy.png  - global accuracy per round
  results/comparison_loss.png      - global loss per round (log scale)
and prints a final-accuracy summary table.
"""

import glob
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# stable colour per strategy so the two figures match
COLOURS = {
    "fedprox": "#2ca02c",
    "fedavg": "#1f77b4",
    "fedadam": "#ff7f0e",
    "fedyogi": "#d62728",
    "fedavgm": "#9467bd",
    "fedmedian": "#8c564b",
    "krum": "#e377c2",
}


def load_all():
    runs = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        data = json.load(open(path))
        runs[data["strategy"]] = data["history"]
    return runs


def main():
    runs = load_all()
    if not runs:
        raise SystemExit(f"No result files in {RESULTS_DIR}")

    # order strategies by final accuracy (best first) for a tidy legend
    order = sorted(runs, key=lambda s: -runs[s][-1]["accuracy"])

    # --- accuracy figure ---
    plt.figure(figsize=(8, 5))
    for s in order:
        h = runs[s]
        rounds = [x["round"] for x in h]
        acc = [x["accuracy"] for x in h]
        plt.plot(rounds, acc, marker="o", label=s, color=COLOURS.get(s))
    plt.xlabel("Federated round")
    plt.ylabel("Global accuracy")
    plt.title("Aggregation strategies on extreme non-IID CAN data")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_accuracy.png"), dpi=150)

    # --- loss figure (log scale: some strategies diverge) ---
    plt.figure(figsize=(8, 5))
    for s in order:
        h = runs[s]
        rounds = [x["round"] for x in h]
        loss = [x["loss"] for x in h]
        plt.plot(rounds, loss, marker="o", label=s, color=COLOURS.get(s))
    plt.xlabel("Federated round")
    plt.ylabel("Global loss (log scale)")
    plt.yscale("log")
    plt.title("Global loss per round (extreme non-IID)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_loss.png"), dpi=150)

    # --- summary table ---
    print(f"\n{'strategy':10} {'final_acc':>9} {'best_acc':>9} {'final_loss':>11}")
    for s in order:
        h = runs[s]
        final = h[-1]["accuracy"]
        best = max(x["accuracy"] for x in h)
        floss = h[-1]["loss"]
        print(f"{s:10} {final:9.4f} {best:9.4f} {floss:11.2f}")
    print(f"\nSaved figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()