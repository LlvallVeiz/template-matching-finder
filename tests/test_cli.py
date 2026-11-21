"""End-to-end CLI tests: run `python -m template_matching.cli` as a subprocess.

These tests exercise the real entry point (argument parsing, file I/O, exit
codes) on small generated scenes, not just the library functions.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from template_matching.synthetic import Placement, generate_scene

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def run_cli(*argv, cwd):
    """Run the CLI module in a subprocess; return the CompletedProcess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "template_matching.cli", *argv],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env,
    )


@pytest.fixture()
def scene_files(tmp_path):
    """A clean 400x300 scene with a cross at (120, 80), plus its template."""
    scene, gt = generate_scene(
        400, 300,
        placements=[Placement("cross", 120, 80, 48)],
        distractors=[Placement("circle", 300, 60, 40)],
        seed=4,
    )
    scene_path = tmp_path / "scene.png"
    template_path = tmp_path / "template_cross.png"
    Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8)).save(scene_path)
    Image.fromarray(
        np.clip(scene[80:128, 120:168], 0, 255).astype(np.uint8)
    ).save(template_path)
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    return tmp_path, scene_path, template_path


def test_cli_match_json_output(scene_files, tmp_path):
    _, scene_path, template_path = scene_files
    out = tmp_path / "result.json"
    proc = run_cli(
        "match", str(scene_path), str(template_path),
        "--method", "zncc", "--output", str(out), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert (rows[0]["x"], rows[0]["y"]) == (120, 80)
    assert rows[0]["score"] == pytest.approx(1.0, abs=1e-9)


def test_cli_match_csv_and_txt_outputs(scene_files, tmp_path):
    _, scene_path, template_path = scene_files
    for suffix in ("csv", "txt"):
        out = tmp_path / f"result.{suffix}"
        proc = run_cli(
            "match", str(scene_path), str(template_path),
            "--output", str(out), cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        assert "120" in out.read_text(encoding="utf-8")


def test_cli_match_annotated_and_heatmap(scene_files, tmp_path):
    _, scene_path, template_path = scene_files
    annotated = tmp_path / "annotated.png"
    heat = tmp_path / "heatmap.png"
    proc = run_cli(
        "match", str(scene_path), str(template_path),
        "--annotate", str(annotated), "--heatmap", str(heat),
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert annotated.exists() and annotated.stat().st_size > 0
    assert heat.exists() and heat.stat().st_size > 0


def test_cli_match_template_too_large_exits_nonzero(scene_files, tmp_path):
    _, scene_path, _ = scene_files
    big = tmp_path / "big.png"
    Image.new("L", (500, 500), 128).save(big)
    proc = run_cli("match", str(scene_path), str(big), cwd=str(tmp_path))
    assert proc.returncode == 1
    assert "larger than image" in proc.stderr


def test_cli_match_missing_file_exits_nonzero(tmp_path):
    proc = run_cli("match", "nope.png", "also-nope.png", cwd=str(tmp_path))
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_cli_batch_two_templates(scene_files, tmp_path):
    _, scene_path, template_path = scene_files
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    Image.open(template_path).save(templates_dir / "cross.png")
    out = tmp_path / "batch.json"
    proc = run_cli(
        "batch", str(scene_path), str(templates_dir),
        "--output", str(out), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["template_name"] == "cross"


def test_cli_generate_and_evaluate(tmp_path):
    out_dir = tmp_path / "gen"
    proc = run_cli(
        "generate", "--width", "200", "--height", "160", "--seed", "9",
        "--place", "cross", "50", "40", "32",
        "--out-dir", str(out_dir), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "scene.png").exists()
    assert (out_dir / "template_cross.png").exists()
    gt = json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))
    assert gt == [{"shape": "cross", "x": 50, "y": 40, "w": 32, "h": 32}]

    # Match the generated template against the generated scene, then evaluate.
    det_path = tmp_path / "detections.json"
    proc = run_cli(
        "match", str(out_dir / "scene.png"), str(out_dir / "template_cross.png"),
        "--output", str(det_path), cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    proc = run_cli(
        "evaluate", str(det_path), str(out_dir / "ground_truth.json"),
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["tp"] == 1 and report["fp"] == 0
    assert report["precision"] == 1.0 and report["recall"] == 1.0
