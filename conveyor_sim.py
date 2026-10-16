"""
conveyor_sim.py - a fake conveyor belt to test YOLO with no hardware.

Setup:
  1. pip install ultralytics opencv-python
  2. Create a folder called "fruits" next to this file.
  3. Put a few photos in it, named by class:
       apple1.jpg, apple2.jpg, banana1.jpg, orange1.png, ...
     (the name prefix is the "truth" used to check the model)
  4. python conveyor_sim.py                 (press q to quit)
     python conveyor_sim.py fruits_unseen   (use another photo folder)
     python conveyor_sim.py fruits yolo26n.pt   (force the original model)

Model: best.pt (your fine-tuned model) if it is in this folder,
otherwise the original yolo26n.pt.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ---- settings you can play with ----
W, H = 960, 400          # window size
BELT_TOP, BELT_BOT = 80, 320
SPEED = 18             # pixels per frame (belt speed)
SPAWN_EVERY = 45         # frames between new items
ITEM_SIZE = 110          # size of each fruit in pixels
GATE_X = 650             # x position of the sorting gate
CONF = 0.5             # minimum confidence to accept a detection
CLASSES = ["apple", "orange", "banana"]


def label_from_name(path):
    name = os.path.basename(path).lower()
    return next((c for c in CLASSES if name.startswith(c)), "other")


def load_images(folder="fruits"):
    items = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = ITEM_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        items.append((img, label_from_name(path)))
    if not items:
        raise SystemExit("No images found. Put some fruit photos in a 'fruits' folder first.")
    return items


def draw_belt(offset):
    frame = np.full((H, W, 3), 40, np.uint8)
    frame[BELT_TOP:BELT_BOT] = (90, 90, 90)
    for x in range(offset % 40 - 40, W, 40):      # moving stripes = belt motion
        cv2.line(frame, (x, BELT_TOP), (x, BELT_BOT), (75, 75, 75), 2)
    return frame


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "fruits"
    weights = sys.argv[2] if len(sys.argv) > 2 else (
        "best.pt" if os.path.exists("best.pt") else "yolo26n.pt")
    print("model:", weights, "| photos:", folder)
    model = YOLO(weights)        # yolo26n.pt downloads automatically the first time
    library = load_images(folder)

    on_belt = []                                  # each: dict(x, y, img, label, counted)
    truth = {c: 0 for c in CLASSES}               # what really passed the gate
    seen = {c: 0 for c in CLASSES}                # what the model said passed
    done_ids = set()                              # tracker IDs already sorted
    t = 0

    while True:
        frame = draw_belt(t * SPEED)

        # 1) spawn a new item on the left
        if t % SPAWN_EVERY == 0:
            img, label = random.choice(library)
            y = random.randint(BELT_TOP + 5, BELT_BOT - img.shape[0] - 5)
            on_belt.append(dict(x=-img.shape[1], y=y, img=img, label=label, counted=False))

        # 2) move items and paint them on the frame
        for it in on_belt:
            it["x"] += SPEED
            x, y, img = it["x"], it["y"], it["img"]
            h, w = img.shape[:2]
            x0, x1 = max(x, 0), min(x + w, W)
            if x1 > x0:
                frame[y:y + h, x0:x1] = img[:, x0 - x:x1 - x]
            # ground truth: count when the item's center crosses the gate
            if not it["counted"] and x + w / 2 > GATE_X:
                it["counted"] = True
                if it["label"] in truth:
                    truth[it["label"]] += 1
        on_belt = [it for it in on_belt if it["x"] < W]

        # 3) run YOLO + tracking on the frame
        r = model.track(frame, persist=True, conf=CONF, verbose=False)[0]
        out = r.plot()

        # 4) sorting decision: fire once per tracked object at the gate
        if r.boxes.id is not None:
            for box, tid, cls in zip(r.boxes.xyxy.tolist(),
                                     r.boxes.id.int().tolist(),
                                     r.boxes.cls.int().tolist()):
                name = model.names[cls]
                cx = (box[0] + box[2]) / 2
                if name in seen and tid not in done_ids and cx > GATE_X:
                    done_ids.add(tid)
                    seen[name] += 1
                    print(f"{name} #{tid} reached gate -> {name} bin")

        # 5) overlay: gate line + truth vs model counts
        cv2.line(out, (GATE_X, BELT_TOP - 20), (GATE_X, BELT_BOT + 20), (0, 0, 255), 2)
        cv2.putText(out, "truth: " + "  ".join(f"{c}={truth[c]}" for c in CLASSES),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, "model: " + "  ".join(f"{c}={seen[c]}" for c in CLASSES),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("conveyor sim", out)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        t += 1

    cv2.destroyAllWindows()
    print("\nFinal  truth:", truth)
    print("Final  model:", seen)


if __name__ == "__main__":
    main()
