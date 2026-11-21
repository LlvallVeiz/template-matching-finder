"""Regenerate all example outputs for template-matching-finder.

Run from the repository root::

    python3 examples/generate_examples.py

Every number in ``examples/RESULTS.md`` and every PNG/JSON/CSV/TXT file in
``examples/scenario*/`` is produced by this script from real runs of the
library and the CLI — nothing is hand-typed.  The script is deterministic
(fixed seeds), so re-running it reproduces the same files.

Scenarios
---------
1. ``scenario1_clean`` — a clean scene; the template is an exact crop, so
   ZNCC must peak at exactly 1.0 at the true location.
2. ``scenario2_noisy`` — the same layout with Gaussian noise (sigma 25);
   the template still comes from the clean render, so the score drops but
   the location must stay exact.
3. ``scenario3_occlusion_multi`` — two target shapes, one of them 30%
   occluded, matched in one batch run with a lower threshold.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"

sys.path.insert(0, str(SRC))

from template_matching.correlation import score_map  # noqa: E402
from template_matching.evaluate import evaluate_detections  # noqa: E402
from template_matching.matching import MatchResult  # noqa: E402
from template_matching.synthetic import (  # noqa: E402
    Placement,
    extract_template,
    generate_scene,
)
from template_matching.visualize import annotate, save_heatmap  # noqa: E402

METHOD = "zncc"


def save_gray(array: np.ndarray, path: Path) -> None:
    """Save a float64 grayscale array as an 8-bit PNG."""
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    """Run the real CLI (``python -m template_matching.cli``)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "template_matching.cli", *argv],
        cwd=ROOT, capture_output=True, text=True, timeout=180, env=env,
    )


def write_ground_truth(path: Path, ground_truth: list[dict]) -> None:
    path.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")


def scenario1_clean() -> dict:
    """Clean scene: exact template crop, ZNCC must be 1.0 at (120, 80)."""
    out = EXAMPLES / "scenario1_clean"
    out.mkdir(parents=True, exist_ok=True)
    placements = [Placement("cross", 120, 80, 48)]
    distractors = [Placement("circle", 300, 60, 40), Placement("triangle", 180, 220, 44)]
    scene, gt = generate_scene(400, 300, placements=placements,
                                distractors=distractors, seed=3, noise_sigma=0.0)
    template = extract_template(scene, (120, 80, 48, 48))
    save_gray(scene, out / "scene.png")
    save_gray(template, out / "template_cross.png")
    write_ground_truth(out / "ground_truth.json", gt)

    started = time.perf_counter()
    proc = run_cli(
        "match", str(out / "scene.png"), str(out / "template_cross.png"),
        "--method", METHOD, "--threshold", "0.8",
        "--output", str(out / "result.json"),
        "--annotate", str(out / "annotated.png"),
        "--heatmap", str(out / "heatmap.png"),
    )
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stderr
    for fmt in ("csv", "txt"):
        proc = run_cli(
            "match", str(out / "scene.png"), str(out / "template_cross.png"),
            "--method", METHOD, "--output", str(out / f"result.{fmt}"),
        )
        assert proc.returncode == 0, proc.stderr
    detections = [MatchResult(**row) for row in json.loads((out / "result.json").read_text())]
    evaluation = evaluate_detections(detections, gt)
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    return {
        "name": "scenario1_clean", "detections": detections,
        "evaluation": evaluation, "elapsed": elapsed,
        "ground_truth": gt, "note": "clean scene, exact template crop",
    }


def scenario2_noisy() -> dict:
    """Same layout + Gaussian noise (sigma 25); template from the clean render."""
    out = EXAMPLES / "scenario2_noisy"
    out.mkdir(parents=True, exist_ok=True)
    placements = [Placement("cross", 120, 80, 48)]
    distractors = [Placement("circle", 300, 60, 40), Placement("triangle", 180, 220, 44)]
    clean, gt = generate_scene(400, 300, placements=placements,
                                distractors=distractors, seed=3, noise_sigma=0.0)
    noisy, _ = generate_scene(400, 300, placements=placements,
                              distractors=distractors, seed=7, noise_sigma=25.0)
    template = extract_template(clean, (120, 80, 48, 48))
    save_gray(noisy, out / "scene.png")
    save_gray(template, out / "template_cross.png")
    write_ground_truth(out / "ground_truth.json", gt)

    started = time.perf_counter()
    proc = run_cli(
        "match", str(out / "scene.png"), str(out / "template_cross.png"),
        "--method", METHOD, "--threshold", "0.8",
        "--output", str(out / "result.json"),
        "--annotate", str(out / "annotated.png"),
        "--heatmap", str(out / "heatmap.png"),
    )
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stderr
    for fmt in ("csv", "txt"):
        proc = run_cli(
            "match", str(out / "scene.png"), str(out / "template_cross.png"),
            "--method", METHOD, "--output", str(out / f"result.{fmt}"),
        )
        assert proc.returncode == 0, proc.stderr
    detections = [MatchResult(**row) for row in json.loads((out / "result.json").read_text())]
    evaluation = evaluate_detections(detections, gt)
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    return {
        "name": "scenario2_noisy", "detections": detections,
        "evaluation": evaluation, "elapsed": elapsed,
        "ground_truth": gt, "note": "Gaussian noise sigma=25, template from clean render",
    }


def scenario3_occlusion_multi() -> dict:
    """Two targets, one 30% occluded; batch match of both templates."""
    out = EXAMPLES / "scenario3_occlusion_multi"
    out.mkdir(parents=True, exist_ok=True)
    placements = [Placement("cross", 60, 60, 48), Placement("ring", 300, 180, 56)]
    distractors = [Placement("triangle", 380, 60, 40)]
    clean, gt = generate_scene(480, 360, placements=placements,
                               distractors=distractors, seed=9, noise_sigma=0.0)
    occluded, _ = generate_scene(
        480, 360, placements=placements, distractors=distractors, seed=9,
        noise_sigma=0.0, occlude=(300, 180, 56, 56), occlusion_fraction=0.3,
    )
    templates_dir = out / "templates"
    templates_dir.mkdir(exist_ok=True)
    templates = {
        "cross": extract_template(clean, (60, 60, 48, 48)),
        "ring": extract_template(clean, (300, 180, 56, 56)),
    }
    for name, templ in templates.items():
        save_gray(templ, templates_dir / f"{name}.png")
    save_gray(occluded, out / "scene.png")
    write_ground_truth(out / "ground_truth.json", gt)

    started = time.perf_counter()
    proc = run_cli(
        "batch", str(out / "scene.png"), str(templates_dir),
        "--method", METHOD, "--threshold", "0.6",
        "--output", str(out / "results.json"),
    )
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stderr
    for fmt in ("csv", "txt"):
        proc = run_cli(
            "batch", str(out / "scene.png"), str(templates_dir),
            "--method", METHOD, "--threshold", "0.6",
            "--output", str(out / f"results.{fmt}"),
        )
        assert proc.returncode == 0, proc.stderr
    detections = [MatchResult(**row) for row in json.loads((out / "results.json").read_text())]
    evaluation = evaluate_detections(detections, gt)
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    annotate(occluded, detections, gt).save(out / "annotated.png")
    for name, templ in templates.items():
        save_heatmap(
            score_map(occluded, templ, METHOD), METHOD,
            str(out / f"heatmap_{name}.png"),
            title=f"ZNCC score map — {name}",
        )
    return {
        "name": "scenario3_occlusion_multi", "detections": detections,
        "evaluation": evaluation, "elapsed": elapsed,
        "ground_truth": gt, "note": "ring 30% occluded, batch of 2 templates, threshold 0.6",
    }


def write_results_md(results: list[dict]) -> None:
    """Write examples/RESULTS.md from the real run numbers."""
    lines = [
        "# Example Results",
        "",
        "All numbers below come from real runs of `examples/generate_examples.py`",
        "(library + CLI, ZNCC method). Re-run that script to reproduce them.",
        "",
    ]
    for res in results:
        ev = res["evaluation"]
        lines.append(f"## {res['name']}")
        lines.append("")
        lines.append(f"{res['note']}.")
        lines.append("")
        lines.append("| ground truth | detected | score (ZNCC) | IoU |")
        lines.append("| --- | --- | --- | --- |")
        for verdict in ev["verdicts"]:
            d = verdict["detection"]
            gt = next(
                (g for g in res["ground_truth"]
                 if g["x"] == d["x"] and g["y"] == d["y"]),
                None,
            )
            gt_text = f"({gt['x']}, {gt['y']}) {gt['shape']}" if gt else "— (false positive)"
            lines.append(
                f"| {gt_text} | ({d['x']}, {d['y']}) {d['template_name']} "
                f"| {d['score']:.4f} | {verdict['iou']:.4f} |"
            )
        for missed in ev["missed"]:
            lines.append(
                f"| ({missed['x']}, {missed['y']}) {missed['shape']} | MISSED | — | — |"
            )
        lines.append("")
        lines.append(
            f"Precision {ev['precision']:.3f} · recall {ev['recall']:.3f} · "
            f"F1 {ev['f1']:.3f} (TP {ev['tp']}, FP {ev['fp']}, FN {ev['fn']}, "
            f"IoU threshold {ev['iou_threshold']}) · "
            f"CLI wall time {res['elapsed']:.3f} s."
        )
        lines.append("")
    (EXAMPLES / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = [scenario1_clean(), scenario2_noisy(), scenario3_occlusion_multi()]
    write_results_md(results)
    for res in results:
        ev = res["evaluation"]
        print(
            f"{res['name']}: TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} "
            f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
            f"({res['elapsed']:.3f} s)"
        )
        for verdict in ev["verdicts"]:
            d = verdict["detection"]
            print(f"  {d['template_name']}: ({d['x']},{d['y']}) score={d['score']:.4f} "
                  f"IoU={verdict['iou']:.4f} {verdict['verdict']}")
    print("RESULTS.md written.")


if __name__ == "__main__":
    main()
