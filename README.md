# template-matching-finder

Template matching **from scratch** in Python (NumPy + Pillow + SciPy) — no
OpenCV anywhere. Four methods (SSD, SAD, NCC, ZNCC), multi-template batch
search, multi-scale pyramids, discrete-rotation search, IoU-based
non-maximum suppression, and evaluation against ground truth (IoU,
precision/recall/F1). ZNCC/NCC run on the fast algorithm of Lewis (1995):
FFT cross-correlation plus integral images.

## Background

Template matching slides a template over a search image and scores every
window. It is the simplest form of object detection — no training data, just
a direct pixel comparison — and it is still the right tool when the target's
appearance is stable and known (industrial inspection, landmark finding,
tracking initialization). The four scores:

| Method | Meaning | Range | Best |
| --- | --- | --- | --- |
| SSD | sum of squared differences | `[0, ∞)` | minimum |
| SAD | sum of absolute differences | `[0, ∞)` | minimum |
| NCC | normalized cross-correlation | `[0, 1]` | maximum |
| ZNCC | zero-mean NCC (Pearson correlation of pixels) | `[-1, 1]` | maximum |

ZNCC is the default: subtracting the mean makes it invariant to a
brightness offset, and normalizing makes it invariant to a contrast change.
It is OpenCV's `TM_CCOEFF_NORMED`; NCC is `TM_CCORR_NORMED`. The full
theory — Lewis's fast NCC, integral images, pyramids, NMS — is in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Project structure

```
template-matching-finder/
├── src/template_matching/
│   ├── __init__.py       # public API
│   ├── metrics.py        # SSD / SAD / NCC / ZNCC on one patch pair
│   ├── correlation.py    # full-image score maps (FFT + integral image)
│   ├── matching.py       # single / multi / multi-scale / rotated / batch + NMS
│   ├── synthetic.py      # synthetic scenes: shapes, noise, occlusion
│   ├── evaluate.py       # IoU / precision / recall / F1 vs ground truth
│   ├── visualize.py      # annotated PNGs + score-map heatmaps
│   ├── cli.py            # `tmf` command-line interface
│   └── errors.py         # custom exceptions
├── tests/                # pytest suite (75 tests)
├── examples/
│   ├── generate_examples.py   # regenerates every example output
│   ├── RESULTS.md             # numbers from the real runs
│   ├── scenario1_clean/       # clean scene
│   ├── scenario2_noisy/       # Gaussian noise, sigma 25
│   └── scenario3_occlusion_multi/  # 30% occlusion + 2 templates
├── docs/RESEARCH.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installs the `tmf` command (optional)
```

Python ≥ 3.10, NumPy, Pillow, SciPy, Matplotlib. No OpenCV — by design.

## Usage

```bash
# Single template, ZNCC (default), threshold 0.8
tmf match scene.png template.png --output result.json

# All four methods, explicit threshold, annotated + heatmap output
tmf match scene.png template.png --method ssd --threshold 50000 \
    --annotate annotated.png --heatmap heatmap.png

# Multi-scale search (image scaled 1.0 and 0.5) and rotation search
tmf match scene.png template.png --scales 1.0 0.5
tmf match scene.png template.png --angles 0 90 180 270

# Batch: every template in a directory against one image
tmf batch scene.png templates/ --output results.json

# Build a synthetic scene with ground truth (for demos and tests)
tmf generate --width 400 --height 300 --seed 3 \
    --place cross 120 80 48 --distractor circle 300 60 40 \
    --noise 25 --out-dir my_scene

# Score a detections file against ground truth (IoU 0.5)
tmf evaluate detections.json ground_truth.json
```

Or use it as a library:

```python
from template_matching import match_all, match_multiscale, evaluate_detections
from template_matching.synthetic import Placement, generate_scene, extract_template

scene, gt = generate_scene(400, 300, placements=[Placement("cross", 120, 80, 48)], seed=3)
template = extract_template(scene, (120, 80, 48, 48))
detections = match_all(scene, template, method="zncc", threshold=0.8)
print(evaluate_detections(detections, gt))
# {'tp': 1, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, ...}
```

Without installing, run from the repo root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m template_matching.cli match scene.png template.png
```

## Examples (real runs)

Regenerate everything with `python3 examples/generate_examples.py`. Full
tables in [`examples/RESULTS.md`](examples/RESULTS.md); the headline numbers:

| Scenario | Ground truth | Detected | ZNCC | IoU | P / R / F1 |
| --- | --- | --- | --- | --- | --- |
| 1. Clean scene | cross (120, 80) | (120, 80) | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 |
| 2. Noise σ=25 | cross (120, 80) | (120, 80) | 0.9668 | 1.0000 | 1.000 / 1.000 / 1.000 |
| 3. Occlusion + multi | cross (60, 60) | (60, 60) | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 |
| 3. Occlusion + multi | ring (300, 180), 30% occluded | (300, 180) | 0.6600 | 1.0000 | — |

Each scenario ships its scene PNG, template PNG(s), annotated PNG, ZNCC
heatmap PNG, ground truth JSON, evaluation JSON, and result JSON/CSV/TXT.

## Tests

```bash
python -m pytest tests/ -q
# 75 passed
```

The suite checks hand-computed values (e.g. SSD=20, SAD=8, NCC=4/14,
ZNCC=−1 for `[[0,1],[2,3]]` vs `[[3,2],[1,0]]`), every fast score map
against an independent brute-force reference on seeded random images, exact
detection on synthetic scenes, NMS, multi-scale (2× object found at scale
0.5), rotation (90° copy found with `angle=90`), and the CLI end to end.
Edge cases: template == image matches at (0, 0) with ZNCC 1.0; a template
larger than the image raises a clean `TemplateTooLargeError` (CLI exit 1).

## Design decisions

- **No OpenCV, on purpose.** The point of the project is the algorithms:
  the FFT/integral-image ZNCC, the score maps and NMS are all written here
  and verified against brute force. OpenCV is cited only as a theory
  reference for the method formulas.
- **Grayscale float64 internally.** Colour images are converted with the
  BT.601 luma weights; matching on colour channels is out of scope.
- **Similarity vs distance is explicit.** Every function that ranks results
  asks `is_similarity(method)` instead of assuming "higher is better", so
  SSD/SAD (lower is better) flow through the same NMS/evaluation code.
- **Threshold defaults differ by method.** Similarity methods default to
  0.8; distance methods default to unthresholded ranking, because raw
  SSD/SAD values depend on template size and brightness and have no
  universal scale.
- **Constant patches score 0.0, not NaN.** ZNCC of a zero-variance window
  is undefined; returning 0.0 keeps score maps finite and total.
- **Synthetic scenes are the test oracle.** Real labelled data is
  expensive; generated scenes with exact ground truth make detection
  quality measurable (Brunelli 2009, Appendix B).

## Limitations

- **Not scale/rotation invariant by itself** — invariance comes only from
  the explicit `--scales` / `--angles` sweeps, which cost one full match
  per scale/angle.
- **No sub-pixel refinement** — detections are integer pixel positions.
- **ZNCC needs texture.** A constant template (e.g. a solid square) has
  zero variance, so ZNCC is undefined and the template can never match;
  use SSD/SAD for flat patches.
- **Occlusion hurts correlation.** A 30% occluded ring still localizes
  exactly but drops from ZNCC 1.0 to 0.66. The operating window is narrow:
  at threshold 0.5 the same scene admits a false positive — the triangle
  distractor scores 0.5733 at (375, 67) against the cross template — so the
  examples use threshold 0.6.
- **SAD is slow for large templates** (one pass per template pixel); SSD /
  NCC / ZNCC use the FFT path instead.
- **Grayscale only**, and matching is translation-only within the image
  (no perspective or deformation handling).

## License

MIT — see [LICENSE](LICENSE).
