"""
visualize_fruits.py - show the shuffled fruit set as a grid, with YOLO's guess.

Each tile shows the photo (in belt order), the TRUE label from the file name,
and YOLO's prediction with its box.
  green border = the sorter would do the right thing
  red border   = the sorter would make a mistake

Usage:
  python visualize_fruits.py [folder] [output.png] [model.pt]

Model: uses the one you pass, else best.pt (your fine-tuned model) if it is
in this folder, else the original yolo26n.pt.
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

TARGETS = {"apple", "orange", "banana"}
GREEN, RED = "#2e9e5b", "#d64545"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "fruits"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "fruit_grid.png"

    # keep the shuffled belt order from order.txt
    with open(os.path.join(folder, "order.txt")) as f:
        names = [line.split("\t")[1] for line in f if line.strip()]

    weights = sys.argv[3] if len(sys.argv) > 3 else (
        "best.pt" if os.path.exists("best.pt") else "yolo26n.pt")
    print("model:", weights)
    model = YOLO(weights)
    cols = 8
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.5))
    good = 0

    for ax, name in zip(axes.flat, names):
        truth = name.rsplit("_", 1)[0]                 # "apple", "other-kiwi", ...
        img = cv2.imread(os.path.join(folder, name))
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)

        r = model.predict(img, imgsz=320, conf=0.25, verbose=False)[0]
        guess, conf = "nothing", 0.0
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            guess = model.names[int(r.boxes.cls[i])]
            conf = float(r.boxes.conf[i])
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)

        # what would the sorter do? targets must be named right;
        # "other" items must NOT be sent to a target bin
        if truth in TARGETS:
            ok = guess == truth
        else:
            ok = guess not in TARGETS
        good += ok

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GREEN if ok else RED); s.set_linewidth(4)
        ax.set_title(f"truth: {truth.replace('other-', '')}\n"
                     f"yolo: {guess}" + (f" {conf:.0%}" if conf else ""),
                     fontsize=9, color=GREEN if ok else RED)

    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")

    fig.suptitle(f"{os.path.basename(weights)} on {os.path.basename(folder.rstrip('/'))}"
                 f" - sorter correct on {good}/{len(names)} items",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"saved {out_png}: {good}/{len(names)} correct")


if __name__ == "__main__":
    main()
