"""Day 1 helper: quick facts about a raw image dataset folder (read-only)."""
import sys
import collections
from pathlib import Path
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"Total files: {len(files)}")
    print("By extension:", dict(collections.Counter(p.suffix.lower() for p in files)))

    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    per_folder = collections.Counter(str(p.parent.relative_to(root)) for p in imgs)
    print("\nImages per folder:")
    for folder, n in sorted(per_folder.items()):
        print(f"  {folder}: {n}")

    sizes, modes, bad = collections.Counter(), collections.Counter(), []
    for p in imgs:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            bad.append((str(p), repr(e)))

    print("\nMost common (width, height):", sizes.most_common(8))
    print("Distinct sizes:", len(sizes))
    print("Colour modes:", dict(modes))
    print(f"Unreadable files: {len(bad)}")
    for item in bad[:10]:
        print("  ", item)


if __name__ == "__main__":
    main(sys.argv[1])