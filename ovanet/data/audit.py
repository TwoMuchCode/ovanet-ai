"""
Day 2: Dataset audit.

Scans raw images, computes md5/phash, detects duplicates and near-duplicates,
computes brightness/contrast stats, checks train/test overlap.

Outputs:
  - data/processed/manifest.csv  (every file with hashes, stats, group_id)
  - docs/data_audit.md           (human-readable report)

Usage:
    python -m ovanet.data.audit [RAW_ROOT]
    Default RAW_ROOT = data/raw/pcos_primary/data
"""

import hashlib
import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Near-duplicate phash threshold (hamming distance 0–64).
# Same scene with flip/brightness/small-crop → typically <10.
# Different scenes → typically >15.
# We use 10 and report the distribution so the threshold can be verified.
PHASH_THRESHOLD = 10


# ── Union-Find for grouping near-duplicates ──────────────────────────────

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_md5(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Scan ─────────────────────────────────────────────────────────────────

def scan_dataset(root: Path) -> pd.DataFrame:
    """Scan every image file under *root* and collect properties."""
    records = []
    all_files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXT)
    total = len(all_files)

    for i, p in enumerate(all_files, 1):
        if i % 500 == 0 or i == total:
            print(f"  scanning {i}/{total} …")

        rel = p.relative_to(root)
        parts = rel.parts  # e.g. ('train', 'infected', 'image.jpg')

        folder_split = parts[0] if len(parts) >= 2 else "unknown"
        label = parts[1] if len(parts) >= 3 else "unknown"

        rec = {
            "path": str(rel),
            "abs_path": str(p),
            "folder_split": folder_split,
            "label": label,
            "filename": p.name,
            "file_size": p.stat().st_size,
        }

        try:
            rec["md5"] = compute_md5(p)
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                rec["width"] = im.size[0]
                rec["height"] = im.size[1]
                rec["mode"] = im.mode
                gray = np.array(im.convert("L"), dtype=np.float32)
                rec["brightness_mean"] = round(float(gray.mean()), 2)
                rec["contrast_std"] = round(float(gray.std()), 2)
                rec["phash"] = str(imagehash.phash(im))
            rec["readable"] = True
        except Exception as e:
            rec.update(readable=False, error=str(e),
                       md5=None, phash=None, width=None, height=None,
                       brightness_mean=None, contrast_std=None)
        records.append(rec)

    return pd.DataFrame(records)


# ── Exact duplicates ────────────────────────────────────────────────────

def mark_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'md5_count' and 'is_duplicate' columns.  First occurrence kept."""
    df = df.copy()
    readable = df["readable"].fillna(False)
    df["md5_count"] = 0
    df["is_duplicate"] = False

    mask = readable
    counts = df.loc[mask, "md5"].map(df.loc[mask, "md5"].value_counts())
    df.loc[mask, "md5_count"] = counts.astype(int)
    df.loc[mask, "is_duplicate"] = df.loc[mask].duplicated(subset=["md5"], keep="first")
    return df


# ── Train / test overlap ───────────────────────────────────────────────

def check_train_test_overlap(df: pd.DataFrame) -> dict:
    r = df[df["readable"]]
    train = set(r.loc[r["folder_split"] == "train", "md5"])
    test = set(r.loc[r["folder_split"] == "test", "md5"])
    overlap = train & test
    return {
        "train_md5s": len(train),
        "test_md5s": len(test),
        "overlap": len(overlap),
        "test_only": len(test - train),
        "train_only": len(train - test),
        "test_is_copy": len(test - train) == 0 and len(overlap) == len(test),
    }


# ── Near-duplicate clustering ──────────────────────────────────────────

def cluster_near_duplicates(
    df: pd.DataFrame, threshold: int = PHASH_THRESHOLD
) -> tuple[pd.DataFrame, dict]:
    """Cluster unique images by phash hamming distance using Union-Find."""
    unique = df[(df["readable"]) & (~df["is_duplicate"])].copy()
    unique = unique.reset_index(drop=True)
    n = len(unique)

    if n == 0:
        df["group_id"] = -1
        return df, {}

    hashes = [imagehash.hex_to_hash(h) for h in unique["phash"]]
    uf = UnionFind(n)
    dist_buckets = collections.Counter()
    total_pairs = n * (n - 1) // 2
    done = 0

    print(f"  comparing {total_pairs:,} pairs …")
    for i in range(n):
        for j in range(i + 1, n):
            d = hashes[i] - hashes[j]
            dist_buckets[(d // 5) * 5] += 1
            if d <= threshold:
                uf.union(i, j)
            done += 1
        if (i + 1) % 200 == 0:
            print(f"    row {i+1}/{n} ({done:,}/{total_pairs:,} pairs)")

    # Assign sequential group IDs
    root_to_gid = {}
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_gid:
            root_to_gid[root] = len(root_to_gid)
    unique["group_id"] = [root_to_gid[uf.find(i)] for i in range(n)]

    # Map back: duplicates get same group as their md5 original
    md5_to_gid = dict(zip(unique["md5"], unique["group_id"]))
    df = df.copy()
    df["group_id"] = df["md5"].map(md5_to_gid).fillna(-1).astype(int)

    return df, dict(dist_buckets)


# ── Report ──────────────────────────────────────────────────────────────

def generate_audit_report(
    df: pd.DataFrame, overlap: dict, dist_buckets: dict, path: Path
) -> None:
    unique = df[(df["readable"]) & (~df["is_duplicate"])]
    n_groups = int(unique["group_id"].nunique())
    gsizes = unique.groupby("group_id").size()
    singletons = int((gsizes == 1).sum())
    multi = int((gsizes > 1).sum())
    largest = int(gsizes.max()) if len(gsizes) else 0

    lines = [
        "# Data Audit Report\n",
        "Generated by `ovanet/data/audit.py` (Day 2).\n",
        "## 1. File counts\n",
        f"- Total files scanned: {len(df)}",
        f"- Readable: {int(df['readable'].sum())}",
        f"- Unreadable: {int((~df['readable']).sum())}\n",
        "| Folder | Label | Count |",
        "|--------|-------|-------|",
    ]
    for s in ["train", "test"]:
        for l in ["infected", "notinfected"]:
            c = int(((df["folder_split"] == s) & (df["label"] == l)).sum())
            lines.append(f"| {s} | {l} | {c} |")

    lines += [
        "\n## 2. Exact duplicates (md5)\n",
        f"- Unique images (by md5): {len(unique)}",
        f"- Duplicate files: {int(df['readable'].sum()) - len(unique)}\n",
        "### Train / test overlap\n",
        f"- Train unique md5s: {overlap['train_md5s']}",
        f"- Test unique md5s: {overlap['test_md5s']}",
        f"- Overlap: {overlap['overlap']}",
        f"- Test-only: {overlap['test_only']}",
        f"- **Test is exact copy of train: {overlap['test_is_copy']}**\n",
        f"## 3. Near-duplicate groups (phash threshold {PHASH_THRESHOLD})\n",
        f"- Groups: {n_groups}",
        f"- Singletons: {singletons}",
        f"- Multi-image groups: {multi}",
        f"- Largest group: {largest}\n",
        "### Phash distance distribution (pairwise buckets)\n",
    ]
    for b in sorted(dist_buckets):
        lines.append(f"- {b}–{b+4}: {dist_buckets[b]:,} pairs")

    lines.append("\n## 4. Image sizes by class\n")
    for label in ["infected", "notinfected"]:
        sub = unique[unique["label"] == label]
        sizes = sub.apply(
            lambda r: f"{int(r['width'])}×{int(r['height'])}", axis=1
        ).value_counts()
        lines.append(f"### {label}\n")
        for sz, cnt in sizes.items():
            lines.append(f"- {sz}: {cnt}")
        lines.append("")

    inf_sizes = set(
        unique.loc[unique["label"] == "infected"]
        .apply(lambda r: (r["width"], r["height"]), axis=1)
    )
    not_sizes = set(
        unique.loc[unique["label"] == "notinfected"]
        .apply(lambda r: (r["width"], r["height"]), axis=1)
    )
    size_overlap = inf_sizes & not_sizes

    lines += [
        "## 5. Shortcut risk\n",
        f"- Infected sizes: {len(inf_sizes)} distinct",
        f"- Not-infected sizes: {len(not_sizes)} distinct",
        f"- **Size overlap: {len(size_overlap)}**"
        + (" — SHORTCUT: size alone separates classes!" if len(size_overlap) == 0 else ""),
        "",
    ]
    for label in ["infected", "notinfected"]:
        sub = unique[unique["label"] == label]
        lines.append(f"### Brightness / contrast: {label}\n")
        lines.append(
            f"- Mean brightness: {sub['brightness_mean'].mean():.1f}"
            f" ± {sub['brightness_mean'].std():.1f}"
        )
        lines.append(
            f"- Mean contrast (std): {sub['contrast_std'].mean():.1f}"
            f" ± {sub['contrast_std'].std():.1f}"
        )
        lines.append("")

    lines += [
        "## 6. Gate G1 — CONFIRMED\n",
        "**Full transvaginal frames**, not tight ovary crops.",
        "Ovary outline annotation is needed for masked radiomics.",
        "Until then, ovary_mask = ultrasound sector (valid foreground).\n",
        "## 7. Known limitations\n",
        "1. Label meaning unverified — 'infected' may mean cystic appearance, not confirmed PCOS.",
        "2. Image size perfectly separates classes (shortcut). Always report B2.",
        "3. Burned-in marks and overlays differ by class.",
        "4. No patient IDs — groups are perceptual-hash similarity, not patients.",
        f"5. Effective sample: ~{len(unique)} unique images, ~{n_groups} groups.",
        "6. Licence is 'Unknown'.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────────────

def main(raw_root: str = "data/raw/pcos_primary/data",
         output_dir: str = "data/processed") -> pd.DataFrame:
    root = Path(raw_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("1/5  Scanning dataset …")
    df = scan_dataset(root)
    print(f"     {len(df)} files found")

    print("2/5  Marking exact duplicates (md5) …")
    df = mark_exact_duplicates(df)
    n_unique = int((df["readable"] & ~df["is_duplicate"]).sum())
    n_dup = int(df["is_duplicate"].sum())
    print(f"     {n_unique} unique, {n_dup} duplicates")

    print("3/5  Checking train/test overlap …")
    overlap = check_train_test_overlap(df)
    print(f"     test is copy of train: {overlap['test_is_copy']}")

    print(f"4/5  Clustering near-duplicates (phash ≤ {PHASH_THRESHOLD}) …")
    df, dist_buckets = cluster_near_duplicates(df, PHASH_THRESHOLD)
    unique = df[(df["readable"]) & (~df["is_duplicate"])]
    n_groups = int(unique["group_id"].nunique())
    print(f"     {n_groups} groups from {n_unique} unique images")

    # Save manifest
    manifest_path = out / "manifest.csv"
    cols = [c for c in [
        "path", "label", "md5", "phash", "group_id",
        "width", "height", "brightness_mean", "contrast_std",
        "file_size", "readable", "is_duplicate", "folder_split",
    ] if c in df.columns]
    df[cols].to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"5/5  manifest → {manifest_path}  ({len(df)} rows)")

    # Audit report
    report_path = Path("docs/data_audit.md")
    generate_audit_report(df, overlap, dist_buckets, report_path)
    print(f"     report  → {report_path}")

    return df


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/pcos_primary/data"
    main(raw)
