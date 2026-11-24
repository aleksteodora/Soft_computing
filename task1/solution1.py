import csv
import os
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Konstante
# ----------------------------------------------------------------------------
VALUE_BANANA = 1
VALUE_BUNCH = 5
VALUE_COIN = 10

# HSV - H 0-179, S 0-255, V 0-255
SEED_LO = np.array([22, 160, 228], dtype=np.uint8)    # strogi prag, seme
SEED_HI = np.array([34, 255, 255], dtype=np.uint8)
LOOSE_LO = np.array([20, 150, 170], dtype=np.uint8)   # blazi prag, pun oblik
LOOSE_HI = np.array([34, 255, 255], dtype=np.uint8)

# HUD - oblasti koje se izbacuju (x0, y0, x1, y1)
HUD_RECTS = [
    (0, 0, 200, 118),      # gore levo - ikonica lika, traka energije
    (50, 203, 200, 276),   # brojac sakupljenih banana
    (0, 688, 1280, 720),   # donja traka - ukrasi
]

# filtriranje regiona
MIN_AREA = 30              # min povrsina regiona u px
MIN_SEED_PIXELS = 8        # min br semenih piksela u regionu
MIN_SEED_FRACTION = 0.12   # min udeo semena u regionu
MIN_HUE_MEAN = 22.0        # zlatni ukrasi i kosa imaju nizi H od banana
MAX_HUE_MEAN = 31.0        # zeleno zuto lisce ima vise H
MAX_REGION_SIZE = 39       # veci regioni su spojeno lisce/trava osim novcica
COIN_MAX_SIZE = 45

# klasifikacija oblika
SINGLE_MIN_ASPECT = 1.9    # izduzeno -> pojedinacna banana
MAX_ASPECT = 5.0           # jos izduzenije -> ivice platformi, trava (sum)
MIN_SINGLE_AREA = 30
MIN_BUNCH_AREA = 120
COIN_MIN_AREA = 650
COIN_MAX_ASPECT = 1.35
MIN_SOLID_SEED_FRAC = 0.40  # grozd/novcic imaju svetlo zasiceno zuto jezgro
DIM_SEED_FRAC = 0.30        # banana sa manjim udelom semena mora biti jako zasicena
DIM_MIN_SATURATION = 232.0


# ----------------------------------------------------------------------------
# Obrada slike
# ----------------------------------------------------------------------------
def yellow_regions(img_bgr):
    # vraca listu regiona kandidata za banane
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    seed = cv2.inRange(hsv, SEED_LO, SEED_HI)
    loose = cv2.inRange(hsv, LOOSE_LO, LOOSE_HI)

    # HUD izbacujemo isto za svaku sliku
    for x0, y0, x1, y1 in HUD_RECTS:
        seed[y0:y1, x0:x1] = 0
        loose[y0:y1, x0:x1] = 0

    # zatvaranje malih rupa, konture izmedju banana u grozdu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(loose, connectivity=8)
    if n <= 1:
        return []

    seed_idx = seed > 0
    seed_count = np.bincount(labels[seed_idx], minlength=n)
    hue_sum = np.bincount(labels[seed_idx],
                          weights=hsv[:, :, 0][seed_idx].astype(np.float32),
                          minlength=n)
    sat_sum = np.bincount(labels.ravel(),
                          weights=hsv[:, :, 1].ravel().astype(np.float32),
                          minlength=n)

    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < MIN_AREA or seed_count[i] < MIN_SEED_PIXELS:
            continue

        seed_frac = seed_count[i] / area
        hue_mean = hue_sum[i] / seed_count[i]
        if seed_frac < MIN_SEED_FRACTION:
            continue
        if not (MIN_HUE_MEAN <= hue_mean <= MAX_HUE_MEAN):
            continue
        if max(w, h) > COIN_MAX_SIZE:
            continue

        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)

        regions.append({
            "area": area,
            "aspect": float(aspect),
            "seed_frac": float(seed_frac),
            "sat": float(sat_sum[i] / area),
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": w,
            "h": h,
        })
    return regions


def classify(region):
    # vraca vrednost regiona: 0 za sum, 1 banana, 5 grozd ili 10 novcic
    a, asp, sf = region["area"], region["aspect"], region["seed_frac"]
    biggest = max(region["w"], region["h"])

    if asp > MAX_ASPECT:
        return 0

    # izduzen oblik -> pojedinacna banana
    if asp >= SINGLE_MIN_ASPECT:
        if a < MIN_SINGLE_AREA or biggest > MAX_REGION_SIZE:
            return 0
        if sf < DIM_SEED_FRAC and region["sat"] < DIM_MIN_SATURATION:
            return 0
        return VALUE_BANANA

    # zbijen oblik -> novcic ili grozd, mora imati svetlo jezgro
    if sf < MIN_SOLID_SEED_FRAC:
        return 0
    if a >= COIN_MIN_AREA and asp <= COIN_MAX_ASPECT:
        return VALUE_COIN
    if biggest > MAX_REGION_SIZE:
        return 0
    return VALUE_BUNCH if a >= MIN_BUNCH_AREA else 0


def image_value(img_bgr):
    return sum(classify(r) for r in yellow_regions(img_bgr))


def draw_debug(img_bgr, out_path):
    out = img_bgr.copy()
    colors = {VALUE_BANANA: (0, 255, 0), VALUE_BUNCH: (255, 128, 0), VALUE_COIN: (0, 0, 255)}
    for r in yellow_regions(img_bgr):
        v = classify(r)
        if v == 0:
            continue
        cv2.rectangle(out, (r["x"] - 2, r["y"] - 2), (r["x"] + r["w"] + 2, r["y"] + r["h"] + 2), colors[v], 2)
        cv2.putText(out, str(v), (r["x"], max(r["y"] - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[v], 2)
    cv2.imwrite(out_path, out)


# ----------------------------------------------------------------------------
# Ucitavanje podataka i evaluacija
# ----------------------------------------------------------------------------
def read_ground_truth(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append((row["image_name"].strip(), int(float(row["bananas_value"]))))
    return rows


def get_paths(dataset_folder):
    csv_path = os.path.join(dataset_folder, "count.csv")
    img_dir = os.path.join(dataset_folder, "images")
    if not os.path.isdir(img_dir):
        img_dir = dataset_folder
    return img_dir, csv_path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dataset_folder = args[0] if args else "data"
    img_dir, csv_path = get_paths(dataset_folder)

    debug = "--debug" in sys.argv
    if debug:
        os.makedirs("debug_out", exist_ok=True)

    errors = []
    for name, true_val in read_ground_truth(csv_path):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            raise FileNotFoundError("Ne mogu da ucitam sliku: " + name)
        pred = image_value(img)
        errors.append(abs(pred - true_val))
        # print("%-14s tacno=%4d  predikcija=%4d  |greska|=%d" % (name, true_val, pred, abs(pred - true_val)))
        if debug:
            draw_debug(img, os.path.join("debug_out", name))

    mae = float(np.mean(errors))
    print(round(mae, 4))


if __name__ == "__main__":
    main()