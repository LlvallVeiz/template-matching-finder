# Research Notes — Template Matching from Scratch

This document records the theory this package implements, with the sources
it was taken from. Every formula below is implemented in `src/` and checked
against a brute-force reference in `tests/test_correlation.py`.

## 1. What template matching is

Template matching slides a small *template* image `T` (w × h) over a larger
search image `I` (W × H) and scores every window, producing a *score map* of
size (W − w + 1) × (H − h + 1). The best-scoring window is the match. It is
the simplest form of object detection: no training data, no learned
features — just a direct pixel comparison. Brunelli (2009) devotes a whole
book to it and notes that it remains effective whenever the target's
appearance is stable and known in advance (industrial inspection, landmark
finding, tracking initialization).

The four scores implemented here, for a window `P` and template `T` with
`N = w·h` pixels:

| Method | Formula | Range | Best |
| --- | --- | --- | --- |
| SSD | `Σ (P − T)²` | `[0, ∞)` | minimum |
| SAD | `Σ \|P − T\|` | `[0, ∞)` | minimum |
| NCC | `Σ P·T / √(Σ P² · Σ T²)` | `[0, 1]` for non-negative images | maximum |
| ZNCC | `Σ (P−P̄)(T−T̄) / √(Σ (P−P̄)² · Σ (T−T̄)²)` | `[-1, 1]` | maximum |

SSD is the squared Euclidean distance between the two patches; SAD is the
Manhattan (L1) distance. NCC is OpenCV's `TM_CCORR_NORMED` and ZNCC is
OpenCV's `TM_CCOEFF_NORMED` — the same formulas OpenCV documents for
`cv2.matchTemplate`. ZNCC is the Pearson correlation of the two patches'
pixels: subtracting the means makes it invariant to an added brightness
offset, and normalizing by the standard deviations makes it invariant to a
multiplicative contrast change. That is why ZNCC is the default method in
this package and why SSD/SAD — whose raw values depend on template size and
image brightness — are not given a default threshold.

## 2. Fast normalized cross-correlation (Lewis 1995)

Naive ZNCC costs O(W·H·w·h). Lewis (1995) showed the whole map can be built
from two ingredients that are each computed once for the full image:

1. **Cross-correlation via FFT.** `Σ P·T` over all windows is one
   convolution of the image with the flipped template, computed in the
   Fourier domain with `scipy.signal.fftconvolve` — O(W·H·log(W·H)).
2. **Window sums via integral images.** With the summed-area table of `I`
   and of `I²` (Crow 1984; popularized for vision by Viola & Jones 2001),
   `Σ P` and `Σ P²` for *any* window cost four table lookups, i.e. O(1).

The ZNCC numerator then follows from the identity

```
Σ (P−P̄)(T−T̄) = Σ P·T − T̄ · Σ P
```

and the denominator from `Σ (P−P̄)² = Σ P² − (Σ P)²/N`. SSD uses the same
two ingredients (`SSD = Σ P² − 2·Σ P·T + Σ T²`). SAD has no such shortcut —
the absolute value does not factor — so it is accumulated directly, one
shifted difference per template pixel. Lewis reports the fast NCC tracking
896 frames of 168×86 features in 1.7 hours where the direct method needed
15 hours on a 100 MHz R4000 — the same order of speedup this package gets
from the FFT formulation.

## 3. Non-maximum suppression

A true match produces a *plateau* of high scores around the true location,
so thresholding the score map alone returns dozens of overlapping boxes for
one object. The standard fix, borrowed from object detection, is greedy
non-maximum suppression: sort candidates best-first and keep a box only if
its Intersection-over-Union (IoU) with every already-kept box is at most the
threshold (0.5 by default). IoU is the PASCAL VOC detection criterion
(Everingham et al. 2010): a detection counts as a true positive when its IoU
with an unmatched ground-truth box is ≥ 0.5.

## 4. Multi-scale search (image pyramids)

Plain matching is not scale invariant: a template only matches objects of
its own size. The classical remedy is an image pyramid (Burt & Adelson
1983): the search image is repeatedly downscaled, the template is matched at
each level, and detections are mapped back to original coordinates. This
package implements the simple version — the image is resized by each scale
factor (LANCZOS resampling) and results from all levels are merged with NMS
across scales. MacLean & Tsotsos describe the same idea for normalized
grey-scale correlation specifically, with a coarse-to-fine search on the
pyramid. Note the trade-off: downscaling blurs fine detail, so very small
scale factors lose the texture ZNCC depends on.

## 5. Rotation

Matching is not rotation invariant either. This package handles the common
practical case — discrete rotations — by matching a rotated copy of the
template for each requested angle and merging the results with NMS.
Multiples of 90° use exact `np.rot90` (no resampling blur); arbitrary angles
use bilinear PIL rotation with `expand=True`. Full rotation invariance would
need rotation-invariant features or a dense angle sweep, both out of scope
here (see Limitations in the README).

## 6. Synthetic evaluation

Real labelled data is expensive; synthetic scenes with known ground truth —
what Brunelli (2009, Appendix B) calls "synthetic oracles" — let detection
quality be measured exactly with IoU, precision, recall and F1 instead of
eyeballed. This package generates scenes from geometric shapes with exact
bounding boxes, plus Gaussian noise and partial occlusion, and evaluates
detections greedily by score (best detection first, each ground-truth box
claimable once).

## References

1. Lewis, J. P. (1995). *Fast Normalized Cross-Correlation*. Vision Interface 1995, pp. 120–123. https://www.cis.rit.edu/~rlepci/Erho/Derek/Useful_References/Correlation%20Filtering/Lewis_COR_normalized.pdf
2. Brunelli, R. (2009). *Template Matching Techniques in Computer Vision: Theory and Practice*. Wiley. ISBN 978-0-470-51706-2. https://searchworks.stanford.edu/view/7934120
3. OpenCV documentation (2025). *Template Matching* tutorial (formulas for `TM_SQDIFF`, `TM_CCORR_NORMED`, `TM_CCOEFF_NORMED`). https://docs.opencv.org/5.0/tutorials/imgproc/histograms/template_matching/template_matching.html
4. Burt, P. J. & Adelson, E. H. (1983). *The Laplacian Pyramid as a Compact Image Code*. IEEE Transactions on Communications, 31(4), 532–540. https://doi.org/10.1109/tcom.1983.1095851
5. MacLean, W. J. & Tsotsos, J. K. *Fast Pattern Recognition Using Normalized Grey-Scale Correlation in a Pyramid Image Representation*. Machine Vision and Applications. http://wjamesmaclean.net/Papers/search-MVA.pdf
6. Crow, F. C. (1984). *Summed-Area Tables for Texture Mapping*. Proceedings of SIGGRAPH '84, Computer Graphics 18(3), 207–212. https://doi.org/10.1145/800031.808600
7. Viola, P. & Jones, M. (2001). *Rapid Object Detection Using a Boosted Cascade of Simple Features*. CVPR 2001, Vol. 1, pp. I-511–I-518. http://www.cs.columbia.edu/~jebara/6772/papers/violaJones_CVPR2001.pdf
8. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. & Zisserman, A. (2010). *The PASCAL Visual Object Classes (VOC) Challenge*. International Journal of Computer Vision, 88, 303–338. https://link.springer.com/article/10.1007/s11263-014-0733-5
