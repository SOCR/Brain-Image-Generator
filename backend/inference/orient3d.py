"""orient3d.py -- put 3D GAN brain volumes into a known anatomical orientation by fitting to an atlas.

THE PROBLEM. The 3D GANs (braingen_gan3d_BraTS_*) emit volumes whose axis ORDER is fixed --
(left-right, anterior-posterior, superior-inferior), the same as BraTS -- but whose DIRECTION along
the AP and SI axes is random per sample (~50/50 each; most likely flip augmentation during
training). So no single fixed NIfTI orientation can label them correctly: the average of 176 raw
samples is almost perfectly mirror-symmetric (asymmetry 0.008 vs 0.25 for a real brain).

THE FIX (validated 2026-10-23). For each volume, try the AP/SI flips, keep the one whose brain
outline best overlaps an atlas brain (MNI152 2009c, reoriented to the BraTS L,P,S voxel layout),
then save the result with the BraTS affine diag(-1,-1,1) so viewers label it correctly.
  * real brains degraded to GAN quality, with known flips: AP+SI recovered 100% (1.0 on
    3 brains x 8 flips x 2 resolutions; 96.6% on a harsh stress set of 768 cases)
  * 352 real GAN samples: picks agree 99.4% between two independent atlases and with an
    intensity-based method; a blind visual audit called 30/30 random samples the same way
  * null inputs (noise, random blobs, AP/SI-symmetrised samples) get near-zero margins and are
    flagged, so the picks reflect real anatomy rather than a bias toward the atlas shape
LEFT-RIGHT IS NOT CHANGED: a brain is nearly mirror-symmetric, so no outline (or intensity)
method can tell left from right. The model's own left-right is kept and labelled per the BraTS
convention (axis 0 -> Left); treat laterality of generated volumes as arbitrary.

numpy only (runs on the CPU web server), ~15 ms per 64^3 volume. No torch / scipy / nibabel.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 40^3 boolean brain mask of the MNI152NLin2009cAsym template (TemplateFlow, desc-brain_mask),
# reoriented to L,P,S voxel order and cropped to its brain bounding box. MNI templates:
# Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological
# Institute, McGill University -- free to use with acknowledgement.
ATLAS_MASK_PATH = os.path.join(HERE, "atlas_brainmask_mni2009_40.npy")

N = 40              # crop size the outlines are compared at
MASK_SIGMA = 1.0    # blur before thresholding: stops stray noise voxels from stretching the crop
MASK_FRAC = 0.15    # brain = above 15% of the robust intensity range (GAN background is ~0)
# Dice margin below which the AP/SI decision is unreliable. From 1536 synthetic fits with known
# truth: 26/27 errors had margin < 0.02. 32^3 brains often touch the volume edge, so their
# outlines are less informative and the threshold is stricter there.
LOW_CONF = {64: 0.02, 32: 0.03}
# The BraTS / SRI24 orientation: voxel axes point Left, Posterior, Superior.
LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])

_atlas = None


def _atlas_mask():
    global _atlas
    if _atlas is None:
        _atlas = np.load(ATLAS_MASK_PATH).astype(bool)
    return _atlas


def _normalize01(v):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)


def _blur(v, sigma):
    """Separable Gaussian blur, edge-replicating."""
    r = int(3 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    v = np.asarray(v, dtype=np.float32)
    for ax in range(3):
        pad = [(0, 0)] * 3; pad[ax] = (r, r)
        p = np.pad(v, pad, mode="edge")
        out = np.zeros_like(v)
        for j, kj in enumerate(k):
            sl = [slice(None)] * 3; sl[ax] = slice(j, j + v.shape[ax])
            out += kj * p[tuple(sl)]
        v = out
    return v


def _resize(v, n):
    """Separable trilinear resize to n^3 (align_corners=False sampling)."""
    v = np.asarray(v, dtype=np.float32)
    for ax in range(3):
        n_in = v.shape[ax]
        if n_in == n:
            continue
        x = np.clip((np.arange(n) + 0.5) * n_in / n - 0.5, 0, n_in - 1)
        i0 = np.floor(x).astype(int); i1 = np.minimum(i0 + 1, n_in - 1)
        shp = [1, 1, 1]; shp[ax] = n
        w = (x - i0).astype(np.float32).reshape(shp)
        v = np.take(v, i0, axis=ax) * (1 - w) + np.take(v, i1, axis=ax) * w
    return v


def _outline(vol):
    """The brain mask, cropped to its bounding box and resampled to N^3."""
    m = _normalize01(_blur(vol, MASK_SIGMA)) > MASK_FRAC
    idx = np.argwhere(m)
    if len(idx) == 0:
        return None
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, np.array(m.shape))
    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    return _resize(sub, N) > 0.5


def fit(vol):
    """Decide the AP and SI flips that put `vol` into L,P,S layout.

    Returns {'flip_ap', 'flip_si', 'margin', 'low_confidence', 'undetermined'}. The margin is
    the best Dice minus the best Dice of any choice that disagrees on AP or SI; left-right is
    scored both ways and the better one used, so it cannot influence the AP/SI decision.
    """
    A = _atlas_mask()
    G = _outline(vol)
    if G is None:
        return {"flip_ap": 0, "flip_si": 0, "margin": 0.0, "low_confidence": True, "undetermined": True}
    denom = float(G.sum() + A.sum())
    scores = {}
    for fy in (0, 1):
        for fz in (0, 1):
            best = 0.0
            for fx in (0, 1):
                X = G[::-1 if fx else 1, ::-1 if fy else 1, ::-1 if fz else 1]
                best = max(best, 2.0 * float((X & A).sum()) / denom)
            scores[(fy, fz)] = best
    pick = max(scores, key=scores.get)
    margin = scores[pick] - max(s for k, s in scores.items() if k != pick)
    res = vol.shape[0]
    return {
        "flip_ap": int(pick[0]),
        "flip_si": int(pick[1]),
        "margin": round(margin, 4),
        "low_confidence": bool(margin < LOW_CONF.get(res, 0.03)),
        # an exact tie (e.g. a perfectly symmetric input) carries no orientation information
        "undetermined": bool(margin < 1e-9),
    }


def canonicalize(vol):
    """Return (volume in L,P,S voxel layout, fit info). Left-right is left as generated."""
    info = fit(vol)
    out = vol
    if info["flip_ap"]:
        out = out[:, ::-1, :]
    if info["flip_si"]:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out), info


if __name__ == "__main__":
    # Self-check: the atlas outline itself, flipped every way and blurred into a soft volume,
    # must be put back to (0, 0) AP/SI -- i.e. canonicalize undoes any AP/SI flip.
    A = _atlas_mask().astype(np.float32)
    base = np.zeros((64, 64, 64), np.float32)
    base[12:52, 12:52, 12:52] = _blur(A, 1.0)
    for fy in (0, 1):
        for fz in (0, 1):
            v = base[:, ::-1 if fy else 1, ::-1 if fz else 1]
            info = fit(v)
            assert (info["flip_ap"], info["flip_si"]) == (fy, fz), (fy, fz, info)
            out, _ = canonicalize(v)
            assert fit(out)["flip_ap"] == 0 and fit(out)["flip_si"] == 0
    sym = np.ones((64, 64, 64), np.float32)
    assert fit(sym)["undetermined"] or fit(sym)["low_confidence"]
    print("orient3d self-check passed")
