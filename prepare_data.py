"""Build a balanced 5-class CAN-bus dataset from the HCRL attack files.

Classes: 0=Normal, 1=DoS, 2=Fuzzy, 3=Gear-spoof, 4=RPM-spoof
Output: a compact .npz holding scaled features X and integer labels y.
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- config ---
HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(HERE, "..", "archive")
PER_CLASS = 20000            # rows per class (balanced)
OUT_PATH = os.path.join(HERE, "canbus_5class.npz")

# each attack file -> its attack class id; R rows in any file are Normal (class 0)
ATTACK_FILES = {
    "DoS_dataset.csv": 1,
    "Fuzzy_dataset.csv": 2,
    "gear_dataset.csv": 3,
    "RPM_dataset.csv": 4,
}


def safe_hex(v):
    """Hex string -> int, 0 on anything invalid/missing."""
    try:
        return int(str(v).strip(), 16)
    except (ValueError, AttributeError):
        return 0


def parse_line(line):
    """Return (features[10], label_char) or None.

    Row layout: timestamp, CAN_ID, DLC, <DLC data bytes>, Label
    Field count varies with DLC (Fuzzy has short frames), so we take the label
    as the LAST field and the data bytes as everything between DLC and it.
    """
    parts = line.strip().split(",")
    if len(parts) < 4:
        return None
    can_id = safe_hex(parts[1])
    try:
        dlc = int(parts[2])
    except ValueError:
        dlc = 0
    label = parts[-1].strip()
    data = parts[3:-1]                      # bytes between DLC and label
    data = [safe_hex(b) for b in data]
    data = (data + [0] * 8)[:8]             # pad/truncate to exactly 8 bytes
    feats = [can_id, dlc] + data            # 10 features
    return feats, label


def collect_from_file(path, want_attack, want_normal, attack_label):
    """Scan one file, collecting up to want_attack T-rows and want_normal R-rows."""
    attack_rows, normal_rows = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if len(attack_rows) >= want_attack and len(normal_rows) >= want_normal:
                break
            parsed = parse_line(line)
            if parsed is None:
                continue
            feats, label = parsed
            if label == "T" and len(attack_rows) < want_attack:
                attack_rows.append((feats, attack_label))
            elif label == "R" and len(normal_rows) < want_normal:
                normal_rows.append((feats, 0))
    return attack_rows, normal_rows


def main():
    normal_per_file = PER_CLASS // len(ATTACK_FILES)   # spread Normal across files
    all_rows = []
    for fname, attack_label in ATTACK_FILES.items():
        path = os.path.join(ARCHIVE_DIR, fname)
        attack_rows, normal_rows = collect_from_file(
            path, PER_CLASS, normal_per_file, attack_label
        )
        all_rows.extend(attack_rows)
        all_rows.extend(normal_rows)
        print(f"{fname:20s} -> attack(class {attack_label}): {len(attack_rows):6d} | normal: {len(normal_rows):6d}")

    X = np.array([r[0] for r in all_rows], dtype=np.float32)
    y = np.array([r[1] for r in all_rows], dtype=np.int64)

    # scale features (mean 0, std 1) for stable NN training
    X = StandardScaler().fit_transform(X).astype(np.float32)

    print("\nFinal dataset:", X.shape,
          "| classes:", dict(zip(*np.unique(y, return_counts=True))))
    np.savez_compressed(OUT_PATH, X=X, y=y)
    print(f"Saved -> {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()