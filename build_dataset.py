"""
build_dataset.py - make an auto-labeled YOLO dataset from Fruits-360 photos.

Why this works: every Fruits-360 photo is ONE fruit on a WHITE background,
so the fruit's box = the area of non-white pixels. No hand labeling needed.

Two kinds of training images:
  A) "belt" frames: 1-4 fruit photos pasted on a grey belt, like conveyor_sim.py
  B) "single" frames: one photo resized to fill the image, like visualize_fruits.py

Lemon / kiwi / strawberry / pear are pasted WITHOUT a label. That teaches the
model "this is background, ignore it" - the reason the extras stop being
confused with apples or oranges.

Usage:
  python build_dataset.py <Fruit-Images-Dataset path> <out dir> [exclude_list.txt]
"""
import os
import random
import sys

import cv2
import numpy as np

TARGETS = {  # class id, Fruits-360 folders
    0: ("apple", ["Apple Red 1", "Apple Golden 1", "Apple Granny Smith"]),
    1: ("banana", ["Banana"]),
    2: ("orange", ["Orange"]),
}
EXTRAS = ["Lemon", "Kiwi", "Strawberry", "Pear"]
BW, BH = 640, 320                 # belt frame size
BELT_TOP, BELT_BOT = 40, 280


def fruit_box(img):
    """Tight box around non-white pixels (x1, y1, x2, y2)."""
    mask = np.any(img < 230, axis=2)
    rows = np.where(mask.sum(1) > 2)[0]
    cols = np.where(mask.sum(0) > 2)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def list_pool(root, split, folders, exclude):
    out = []
    for f in folders:
        d = os.path.join(root, split, f)
        out += [os.path.join(d, x) for x in sorted(os.listdir(d))
                if x.endswith(".jpg") and os.path.relpath(os.path.join(d, x), root) not in exclude]
    return out


def belt_background(rng):
    frame = np.full((BH, BW, 3), rng.randint(25, 55), np.uint8)
    frame[BELT_TOP:BELT_BOT] = rng.randint(75, 110)
    off = rng.randint(0, 39)
    for x in range(off - 40, BW, 40):
        cv2.line(frame, (x, BELT_TOP), (x, BELT_BOT), (75, 75, 75), 2)
    return frame


def make_belt_frame(rng, pools, extras):
    frame = belt_background(rng)
    labels, taken = [], []
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.25:
            cls, path = None, rng.choice(extras)
        else:
            cls = rng.choice(list(pools))
            path = rng.choice(pools[cls])
        img = cv2.imread(path)
        size = rng.randint(70, 170)
        img = cv2.resize(img, (size, size))
        for _try in range(20):  # find a free spot (no overlap)
            x = rng.randint(-size // 3, BW - 2 * size // 3)   # may be cut at the edges
            y = rng.randint(BELT_TOP, BELT_BOT - size)
            if all(x + size < a or a + s < x for a, s in taken):
                break
        else:
            continue
        taken.append((x, size))
        x0, x1 = max(x, 0), min(x + size, BW)
        frame[y:y + size, x0:x1] = img[:, x0 - x:x1 - x]
        if cls is None:
            continue
        b = fruit_box(img)
        if b is None:
            continue
        bx1, by1 = max(b[0] + x, 0), b[1] + y
        bx2, by2 = min(b[2] + x, BW), b[3] + y
        if bx2 - bx1 < 0.4 * (b[2] - b[0]):   # mostly off-screen -> skip label
            continue
        labels.append((cls, bx1, by1, bx2, by2))
    return frame, labels


def make_single_frame(rng, pools, extras):
    if rng.random() < 0.2:
        cls, path = None, rng.choice(extras)
    else:
        cls = rng.choice(list(pools))
        path = rng.choice(pools[cls])
    img = cv2.resize(cv2.imread(path), (320, 320), interpolation=cv2.INTER_CUBIC)
    labels = []
    if cls is not None:
        b = fruit_box(img)
        if b is not None:
            labels.append((cls, *b))
    return img, labels


def write_split(out, split, n, rng, pools, extras):
    os.makedirs(f"{out}/images/{split}", exist_ok=True)
    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    for i in range(n):
        maker = make_belt_frame if rng.random() < 0.6 else make_single_frame
        img, labels = maker(rng, pools, extras)
        h, w = img.shape[:2]
        cv2.imwrite(f"{out}/images/{split}/{i:05d}.jpg", img)
        with open(f"{out}/labels/{split}/{i:05d}.txt", "w") as f:
            for cls, x1, y1, x2, y2 in labels:   # YOLO format: class cx cy w h (0..1)
                f.write(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def main():
    root, out = sys.argv[1], sys.argv[2]
    exclude = set()
    if len(sys.argv) > 3:
        exclude = {l.strip() for l in open(sys.argv[3]) if l.strip()}
    rng = random.Random(0)

    # train on the Training folder, validate on half of the Test folder;
    # the other half of Test is kept for the final grid (never seen at all)
    def test_half(folders, half):
        files = list_pool(root, "Test", folders, set())
        rng2 = random.Random(1)
        rng2.shuffle(files)
        return files[: len(files) // 2] if half == 0 else files[len(files) // 2:]

    train_pools = {c: list_pool(root, "Training", f, exclude) for c, (_, f) in TARGETS.items()}
    train_extras = list_pool(root, "Training", EXTRAS, exclude)
    val_pools = {c: test_half(f, 0) for c, (_, f) in TARGETS.items()}
    val_extras = test_half(EXTRAS, 0)

    write_split(out, "train", int(os.environ.get("N_TRAIN", 600)), rng, train_pools, train_extras)
    write_split(out, "val", int(os.environ.get("N_VAL", 120)), rng, val_pools, val_extras)

    with open(f"{out}/fruits.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for c, (name, _) in TARGETS.items():
            f.write(f"  {c}: {name}\n")
    print("dataset ready:", out)


if __name__ == "__main__":
    main()
