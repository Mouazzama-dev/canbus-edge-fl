"""Plot the DP privacy-utility tradeoff: final accuracy vs noise multiplier.
Reads results/dp/noise_*.json, saves results/dp/dp_tradeoff.png.
"""
import glob
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DP_DIR = os.path.join(HERE, "results", "dp")


def main():
    points = []
    for path in glob.glob(os.path.join(DP_DIR, "noise_*.json")):
        d = json.load(open(path))
        points.append((d["noise_multiplier"], d["history"][-1]["accuracy"]))
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", color="#1f77b4")
    plt.axhline(0.2, ls="--", lw=1, color="gray")
    plt.text(max(xs), 0.21, "random (0.20)", fontsize=8, color="gray", ha="right")
    plt.xlabel("DP noise multiplier (more noise = more privacy)")
    plt.ylabel("Final global accuracy")
    plt.title("Privacy-utility tradeoff (moderate non-IID)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    for x, y in points:
        plt.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DP_DIR, "dp_tradeoff.png"), dpi=150)
    print("Saved", os.path.join(DP_DIR, "dp_tradeoff.png"))


if __name__ == "__main__":
    main()