"""Command-line interface for template-matching-finder.

Subcommands
-----------
``match``     match one template against one image.
``batch``     match every template in a directory against one image.
``generate``  build a synthetic scene (with ground truth) for demos/tests.
``evaluate``  score a detections JSON file against a ground-truth JSON file.

All match results can be written as JSON, CSV or human-readable TXT
(``--format`` or inferred from ``--output``), and the ``match``/``batch``
commands can also save an annotated PNG (``--annotate``) and a score-map
heatmap PNG (``--heatmap``).

Exit codes: 0 on success, 2 on argument errors, 1 on runtime errors
(missing file, template larger than the image, ...).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .correlation import as_gray, score_map
from .errors import TemplateMatchingError
from .evaluate import evaluate_detections
from .matching import MatchResult, match_all, match_batch, match_multiscale, match_rotated
from .metrics import METHODS
from .synthetic import Placement, generate_scene
from .visualize import annotate, save_heatmap


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image file as a float64 grayscale array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as img:
        return as_gray(np.asarray(img.convert("L"), dtype=np.float64))


def save_results(
    detections: list[MatchResult], path: str | Path, fmt: str
) -> str:
    """Write detections as JSON / CSV / TXT; return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [d.to_dict() for d in detections]
    if fmt == "json":
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    elif fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["x", "y", "w", "h", "score", "method",
                            "template_name", "scale", "angle"],
            )
            writer.writeheader()
            writer.writerows(rows)
    elif fmt == "txt":
        lines = [
            f"template-matching-finder {__version__} — {len(rows)} detection(s)"
        ]
        for d in rows:
            lines.append(
                f"{d['template_name'] or d['method']}: box=({d['x']},{d['y']},{d['w']}x{d['h']}) "
                f"score={d['score']:.4f} method={d['method']} "
                f"scale={d['scale']} angle={d['angle']}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"unknown format {fmt!r}; expected json/csv/txt")
    return str(path)


def _infer_format(output: str | None, fmt: str | None) -> str:
    if fmt:
        return fmt
    if output:
        suffix = Path(output).suffix.lower().lstrip(".")
        if suffix in ("json", "csv", "txt"):
            return suffix
    return "txt"


def _add_match_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHODS, default="zncc",
                        help="matching method (default: zncc)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="score threshold (default: 0.8 similarity, none distance)")
    parser.add_argument("--max-results", type=int, default=10,
                        help="max detections after NMS (default: 10)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--scales", type=float, nargs="*", default=None,
                        help="image scales for multi-scale search, e.g. --scales 1.0 0.5")
    parser.add_argument("--angles", type=float, nargs="*", default=None,
                        help="rotation angles in degrees, e.g. --angles 0 90 180 270")
    parser.add_argument("--output", default=None,
                        help="result file (.json/.csv/.txt)")
    parser.add_argument("--format", choices=("json", "csv", "txt"), default=None,
                        help="output format (default: inferred from --output)")
    parser.add_argument("--annotate", default=None,
                        help="save an annotated PNG to this path")
    parser.add_argument("--heatmap", default=None,
                        help="save a score-map heatmap PNG to this path")


def _run_search(image, template, args, template_name: str) -> tuple[list[MatchResult], float]:
    """Dispatch single-template search (plain / multi-scale / rotated)."""
    started = time.perf_counter()
    if args.scales:
        found = match_multiscale(
            image, template, method=args.method, scales=tuple(args.scales),
            threshold=args.threshold, max_results=args.max_results,
            iou_threshold=args.iou, template_name=template_name,
        )
    elif args.angles:
        found = match_rotated(
            image, template, method=args.method, angles=tuple(args.angles),
            threshold=args.threshold, max_results=args.max_results,
            iou_threshold=args.iou, template_name=template_name,
        )
    else:
        found = match_all(
            image, template, method=args.method, threshold=args.threshold,
            max_results=args.max_results, iou_threshold=args.iou,
            template_name=template_name,
        )
    return found, time.perf_counter() - started


def _emit(image, detections, args, elapsed: float) -> None:
    fmt = _infer_format(args.output, args.format)
    for det in detections:
        print(
            f"{det.template_name or det.method}: box=({det.x},{det.y},{det.w}x{det.h}) "
            f"score={det.score:.4f} scale={det.scale} angle={det.angle}"
        )
    print(f"{len(detections)} detection(s) in {elapsed:.3f} s")
    if args.output:
        save_results(detections, args.output, fmt)
        print(f"results written to {args.output}")
    if args.annotate:
        annotate(image, detections).save(args.annotate)
        print(f"annotated image written to {args.annotate}")
    if args.heatmap and detections:
        first = detections[0]
        template_path = getattr(args, "_heatmap_template", None)
        if template_path is not None:
            scores = score_map(image, load_gray(template_path), args.method)
            save_heatmap(scores, args.method, args.heatmap,
                         title=f"{args.method} score map")
            print(f"heatmap written to {args.heatmap}")


def cmd_match(args: argparse.Namespace) -> int:
    """`tmf match IMAGE TEMPLATE` — single-template search."""
    image = load_gray(args.image)
    template = load_gray(args.template)
    args._heatmap_template = args.template
    found, elapsed = _run_search(image, template, args, Path(args.template).stem)
    _emit(image, found, args, elapsed)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """`tmf batch IMAGE TEMPLATES_DIR` — one image, many templates."""
    image = load_gray(args.image)
    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        raise FileNotFoundError(f"templates directory not found: {templates_dir}")
    templates = {}
    for path in sorted(templates_dir.iterdir()):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".pgm"):
            templates[path.stem] = load_gray(path)
    if not templates:
        raise FileNotFoundError(f"no image templates in {templates_dir}")
    first_path = next(
        p for p in sorted(templates_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".pgm")
    )
    args._heatmap_template = str(first_path)
    started = time.perf_counter()
    results = match_batch(
        image, templates, method=args.method, threshold=args.threshold,
        max_results=args.max_results, iou_threshold=args.iou,
        scales=tuple(args.scales) if args.scales else None,
        angles=tuple(args.angles) if args.angles else None,
    )
    elapsed = time.perf_counter() - started
    flat = [d for dets in results.values() for d in dets]
    _emit(image, flat, args, elapsed)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """`tmf generate` — build a reproducible synthetic scene."""
    placements = [
        Placement(shape, int(x), int(y), int(size))
        for shape, x, y, size in (args.place or [["cross", "120", "80", "48"]])
    ]
    distractors = [
        Placement(shape, int(x), int(y), int(size))
        for shape, x, y, size in (args.distractor or [])
    ]
    scene, ground_truth = generate_scene(
        width=args.width, height=args.height, placements=placements,
        distractors=distractors, seed=args.seed, noise_sigma=args.noise,
        occlude=tuple(args.occlude) if args.occlude else None,
        occlusion_fraction=args.occlusion_fraction,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8), mode="L").save(
        out_dir / "scene.png"
    )
    (out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8"
    )
    for gt in ground_truth:
        template = scene[gt["y"] : gt["y"] + gt["h"], gt["x"] : gt["x"] + gt["w"]]
        Image.fromarray(np.clip(template, 0, 255).astype(np.uint8), mode="L").save(
            out_dir / f"template_{gt['shape']}.png"
        )
    print(f"scene {args.width}x{args.height} written to {out_dir}")
    print(f"ground truth: {ground_truth}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """`tmf evaluate DETECTIONS_JSON GROUND_TRUTH_JSON`."""
    detections = [
        MatchResult(**row)
        for row in json.loads(Path(args.detections).read_text(encoding="utf-8"))
    ]
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    report = evaluate_detections(detections, ground_truth, iou_threshold=args.iou)
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"evaluation written to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tmf",
        description="Template matching from scratch (NumPy + Pillow, no OpenCV).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_match = sub.add_parser("match", help="match one template against one image")
    p_match.add_argument("image", help="search image (PNG/JPG/...)")
    p_match.add_argument("template", help="template image (must fit inside the search image)")
    _add_match_options(p_match)
    p_match.set_defaults(func=cmd_match)

    p_batch = sub.add_parser("batch", help="match all templates in a directory")
    p_batch.add_argument("image", help="search image")
    p_batch.add_argument("templates_dir", help="directory of template images")
    _add_match_options(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_gen = sub.add_parser("generate", help="build a synthetic scene + ground truth")
    p_gen.add_argument("--width", type=int, default=400)
    p_gen.add_argument("--height", type=int, default=300)
    p_gen.add_argument("--place", nargs=4, action="append", metavar=("SHAPE", "X", "Y", "SIZE"),
                       default=None,
                       help="shape placement (repeatable; default: cross 120 80 48)")
    p_gen.add_argument("--distractor", nargs=4, action="append", default=[],
                       metavar=("SHAPE", "X", "Y", "SIZE"),
                       help="distractor shape (repeatable)")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma")
    p_gen.add_argument("--occlude", type=int, nargs=4, default=None,
                       metavar=("X", "Y", "W", "H"), help="box to partially occlude")
    p_gen.add_argument("--occlusion-fraction", type=float, default=0.3)
    p_gen.add_argument("--out-dir", default="scene_out")
    p_gen.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate", help="score detections vs ground truth")
    p_eval.add_argument("detections", help="detections JSON (list of MatchResult dicts)")
    p_eval.add_argument("ground_truth", help="ground-truth JSON")
    p_eval.add_argument("--iou", type=float, default=0.5)
    p_eval.add_argument("--output", default=None)
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (also the ``tmf`` console script)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except TemplateMatchingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
