# Example Results

All numbers below come from real runs of `examples/generate_examples.py`
(library + CLI, ZNCC method). Re-run that script to reproduce them.

## scenario1_clean

clean scene, exact template crop.

| ground truth | detected | score (ZNCC) | IoU |
| --- | --- | --- | --- |
| (120, 80) cross | (120, 80) template_cross | 1.0000 | 1.0000 |

Precision 1.000 · recall 1.000 · F1 1.000 (TP 1, FP 0, FN 0, IoU threshold 0.5) · CLI wall time 3.416 s.

## scenario2_noisy

Gaussian noise sigma=25, template from clean render.

| ground truth | detected | score (ZNCC) | IoU |
| --- | --- | --- | --- |
| (120, 80) cross | (120, 80) template_cross | 0.9668 | 1.0000 |

Precision 1.000 · recall 1.000 · F1 1.000 (TP 1, FP 0, FN 0, IoU threshold 0.5) · CLI wall time 2.562 s.

## scenario3_occlusion_multi

ring 30% occluded, batch of 2 templates, threshold 0.6.

| ground truth | detected | score (ZNCC) | IoU |
| --- | --- | --- | --- |
| (60, 60) cross | (60, 60) cross | 1.0000 | 1.0000 |
| (300, 180) ring | (300, 180) ring | 0.6600 | 1.0000 |

Precision 1.000 · recall 1.000 · F1 1.000 (TP 2, FP 0, FN 0, IoU threshold 0.5) · CLI wall time 1.524 s.
