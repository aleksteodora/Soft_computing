# Task 1: Banana Value Counting

A classical computer vision solution (OpenCV + NumPy) that estimates the total value of the bananas in game screenshots. Each object type has a fixed value: **banana = 1**, **banana bunch = 5**, **banana coin = 10**.

**Result: MAE = 0.7273** on the 11 provided images (the task's threshold for full points is MAE < 2.8).

Techniques used: HSV colour thresholding, hysteresis (two-threshold) filtering, morphological closing, connected-component analysis, and contour-based shape descriptors.

## Problem statement

The dataset consists of 11 screenshots (1280 × 720) from the game, located in `data/images`.

- Compute the total value of the objects in each image. Objects count as present if they are not mostly cropped at the image borders and are visible without zooming in. Objects seen from different angles also count.
- Object values: banana = 1, banana bunch = 5, banana coin = 10. For example, an image with one banana, one bunch and one coin has the value 1 + 5 + 10 = 16.
- `data/count.csv` holds the ground-truth value of every image and is used to compute the metric.
- The solution must perform the same operations on every image (no per-image preprocessing) and minimise the mean absolute error (MAE). Full points require MAE < 2.8.
- Only objects that are "in the world" are counted. The task statement shows examples of objects that are not valid, such as the on-screen interface.
- The script prints the final MAE.

## Approach

Counting is split into three questions: where the yellow objects are, which kind each one is, and what they are worth in total. A computer sees only pixel values, so the answers are derived from colour, shape and size. Every image goes through the same pipeline (`yellow_regions` followed by `classify` in `solution1.py`):

1. **HSV conversion.** The BGR image is converted to HSV, which separates colour (H) from brightness (V). A banana in shadow and in sunlight is still "yellow".
2. **Two colour masks** (`cv2.inRange`):
   - *seed* (strict): H 22–34, S ≥ 160, V ≥ 228. Only very bright, saturated yellow pixels, which are almost certainly banana.
   - *loose* (full shape): H 20–34, S ≥ 150, V ≥ 170. Covers the whole banana including shaded parts and edges, but also grass and foliage.
3. **HUD removal.** Fixed rectangles of the game interface (top-left character icon and energy bar, the banana counter on image 5, and a bottom strip) are zeroed in both masks. The same rectangles are used for every image.
4. **Morphological closing** (`MORPH_CLOSE`, 3 × 3 elliptical kernel). Dilation followed by erosion fills gaps of 1–2 pixels so that a bunch does not fall apart into several regions.
5. **Connected components** (`cv2.connectedComponentsWithStats`, 8-connectivity) on the loose mask. Every connected blob is one candidate and comes with its area, width and height. Label 0 is the background and is skipped.
6. **Hysteresis.** For each candidate, the seed pixels inside it are counted (`np.bincount`). A bunch or banana is mostly seed, while grass and foliage contain only a small amount.
7. **Noise filtering** (first checks that fail discard the region):
   - area < 30 px or fewer than 8 seed pixels
   - seed fraction < 0.12
   - mean hue of the seed pixels outside 22–31
   - width or height > 45 px
8. **Shape measurement.** `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`) and `cv2.minAreaRect` give the smallest rotated rectangle around the region. The *aspect ratio* is its longer side divided by its shorter side.
9. **Classification** (`classify`), in this order:
   1. aspect > 5: noise (thin line, for example a platform edge)
   2. aspect ≥ 1.9: **banana (1)**, provided it is large enough (area ≥ 30, width and height ≤ 39) and not a pale region (a region with seed fraction < 0.30 must have mean saturation ≥ 232)
   3. compact shape with seed fraction < 0.40: noise (no bright core, for example a character's hair)
   4. compact, area ≥ 650 and aspect ≤ 1.35: **coin (10)**
   5. compact, width and height ≤ 39 and area ≥ 120: **bunch (5)**
10. **Summation.** Region values are summed per image, compared with `count.csv`, and the MAE is printed.

### Why shape and not size

The camera zoom changes from image to image, so a single banana on image 1 is about as large as a bunch on image 4. The aspect ratio does not depend on zoom. Area is used only to separate coins from bunches. Measured on the provided images:

| Class | Aspect ratio | Area (px) |
|---|---|---|
| Banana | 2.00 – 4.67 | 31 – 292 |
| Bunch | 1.00 – 1.63 | 208 – 507 |
| Coin | 1.04 – 1.11 | 862 – 1058 |

## How the parameters were chosen

1. HSV values were sampled on real bananas and on false hits (foliage, grass, hair, ornaments).
2. Masks were drawn over the image to see what they capture.
3. For every candidate region the area, aspect ratio, seed fraction and saturation were printed, to see what separates real objects from noise.
4. Each threshold was placed in the gap between the measured real objects and the noise, and the MAE was re-checked after every change.

## Results

| Image | Ground truth | Prediction | Error |
|---|---|---|---|
| image_1 | 19 | 21 | +2 |
| image_2 | 25 | 26 | +1 |
| image_3 | 13 | 13 | 0 |
| image_4 | 40 | 40 | 0 |
| image_5 | 10 | 10 | 0 |
| image_6 | 35 | 35 | 0 |
| image_7 | 100 | 101 | +1 |
| image_8 | 100 | 99 | −1 |
| image_9 | 28 | 27 | −1 |
| image_10 | 20 | 20 | 0 |
| image_11 | 7 | 9 | +2 |

The sum of absolute errors is 8, so **MAE = 8 / 11 = 0.7273**. Five images are exact (3, 4, 5, 6, 10).

## Error analysis

At object level, 10 detections are wrong: 7 false positives and 3 misses. On image 9 two misses and one false positive partly cancel (−2 + 1 = −1), so the MAE is smaller than the number of mistakes. Coordinates are approximate positions in the 1280 × 720 image.

### Counted but not part of the ground truth (7)

Each is a yellow, elongated shape that colour and shape alone cannot tell apart from a banana.

| Image | Position (x, y) | What it is | Region statistics |
|---|---|---|---|
| 1 | (57, 405) | yellow-green foliage at the left border | area 65 px, aspect 2.00, seed fraction 0.28 |
| 1 | (503, 422) | yellow-orange object next to the character that resembles a banana | area 223 px, aspect 2.53, seed fraction 0.57 |
| 2 | (810, 437) | yellow part of an enemy's helmet | area 72 px, aspect 2.88, seed fraction 0.57 |
| 7 | (642, 655) | yellow tip of the red tie on Donkey Kong | area 34 px, aspect 2.00, seed fraction 0.74 |
| 9 | (845, 655) | golden ring lying on the ground | area 34 px, aspect 2.25, seed fraction 0.50 |
| 11 | (439, 372) | yellow tip of the red tie on Donkey Kong | area 58 px, aspect 2.02, seed fraction 0.83 |
| 11 | (790, 568) | yellow crest of an enemy's helmet | area 113 px, aspect 2.62, seed fraction 0.67 |

### Missed, although they should have been counted (3)

| Image | Position (x, y) | What it is | Why it was rejected |
|---|---|---|---|
| 8 | (1000, 230) | thin banana seen almost edge-on, 5 × 8 px | aspect 1.75 is below 1.9, so it is treated as a compact shape, where its seed fraction 0.37 is below 0.40 |
| 9 | (305, 352) | banana on a dark background | seed fraction 0.26 is below 0.30 and the mean saturation is just under the 232 threshold of the pale-banana rule |
| 9 | (700, 371) | banana against the pink sky | only 6 seed pixels in a 58 px region (minimum is 8) |

## Limitations

- The thresholds were tuned on these 11 images, so **0.7273 is a measurement on this set and not a guarantee for unseen images.**
- Some thresholds have narrow margins: the seed fraction for bunches (0.40, between 0.39 for noise and 0.42 for bunches) and the maximum region size (39, between 38 for objects and 40 for merged foliage).
- Colour and shape cannot separate bananas from yellow look-alikes (helmet, tie, ring). Removing these would need object recognition.
- The remaining errors were not patched with image-specific rules, because that would mean fitting the solution to individual images.

## Running the code

### Requirements

- Python 3
- `opencv-python` and `numpy`

### Setup

Windows (PowerShell):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install opencv-python numpy
```

### Run

```
python solution1.py
```

The script prints only the final result:

```
0.7273
```