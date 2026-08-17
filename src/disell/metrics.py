"""Label-map quality and comparison metrics for dislocation-cell segmentations.

All functions are label-invariant where relevant (they never compare raw label
integers between two segmentations) and operate on (Z, Y, X) or (Y, X) label
arrays with background = 0.

Definitions follow the manuscript:

- feature distance  rho(a, b) = sqrt( (1/C) * sum_c (a_c - b_c)^2 )
- inner-cell spread sigma_k^2 = mean_{x in C_k} rho(f(x), mean_k)^2
- boundary-band KAM E_bd^(k)  = median_{x in B_k} KAM(x), where B_k is the set
  of boundary elements of cell k (elements with at least one neighbour of a
  different positive label) dilated by ``r_bd`` elements.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _connectivity_structure(ndim: int, connectivity: int) -> np.ndarray:
    return ndimage.generate_binary_structure(ndim, connectivity)


def inner_cell_spread(labels: np.ndarray, field: np.ndarray) -> dict:
    """Per-cell inner spread sigma_k^2 of the feature field.

    Parameters
    ----------
    labels : (spatial) int array, background 0.
    field : (spatial + (C,)) float array. NaN elements are excluded.

    Returns
    -------
    dict mapping label -> sigma_k^2 (units: squared feature units).
    """
    if field.shape[:-1] != labels.shape:
        raise ValueError(
            f"field spatial shape {field.shape[:-1]} != labels shape {labels.shape}"
        )
    C = field.shape[-1]
    out = {}
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        vals = field[labels == lbl].reshape(-1, C)
        finite = np.isfinite(vals).all(axis=1)
        vals = vals[finite]
        if vals.shape[0] == 0:
            out[int(lbl)] = np.nan
            continue
        mean = vals.mean(axis=0)
        rho_sq = ((vals - mean) ** 2).sum(axis=1) / C
        out[int(lbl)] = float(rho_sq.mean())
    return out


def boundary_band_kam(
    labels: np.ndarray,
    kam: np.ndarray,
    r_bd: int = 1,
    connectivity: int = 1,
    include_background: bool = False,
) -> dict:
    """Per-cell median KAM over the finite-width boundary band B_k.

    Boundary elements of cell k are its elements with at least one neighbour
    (under ``connectivity``) carrying a different positive label (or, when
    ``include_background`` is True, any different value including 0). The band
    is that boundary dilated ``r_bd`` times with the same structuring element.

    The computation runs per cell on a bounding box expanded by
    ``r_bd + 1`` voxels (clamped to the volume), so the band is never clipped
    by the tight bounding box. NaN KAM values are ignored in the median.

    Returns
    -------
    dict mapping label -> median KAM over B_k (NaN when the band is empty or
    all-NaN).
    """
    if labels.shape != kam.shape:
        raise ValueError("labels and kam must share the same shape")
    struct = _connectivity_structure(labels.ndim, connectivity)

    objects = ndimage.find_objects(labels)
    out = {}
    for lbl, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        # expand the bounding box so dilation is never clipped
        pad = r_bd + 1
        sl_exp = tuple(
            slice(max(s.start - pad, 0), min(s.stop + pad, dim))
            for s, dim in zip(sl, labels.shape)
        )
        lab_crop = labels[sl_exp]
        cell = lab_crop == lbl

        # neighbours with a different label: dilate the complement of the cell
        # restricted to "other" material and intersect with the cell
        if include_background:
            other = ~cell
        else:
            other = (lab_crop != lbl) & (lab_crop > 0)
        touch = ndimage.binary_dilation(other, structure=struct)
        boundary = cell & touch

        if not boundary.any():
            out[int(lbl)] = np.nan
            continue

        band = ndimage.binary_dilation(boundary, structure=struct, iterations=r_bd) if r_bd > 0 else boundary
        vals = kam[sl_exp][band]
        vals = vals[np.isfinite(vals)]
        out[int(lbl)] = float(np.median(vals)) if vals.size else np.nan
    return out


def _contingency(a: np.ndarray, b: np.ndarray):
    """Joint label histogram of two co-registered label arrays (flattened)."""
    if a.shape != b.shape:
        raise ValueError("label arrays must share the same shape")
    a = a.ravel()
    b = b.ravel()
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(joint, (ia, ib), 1)
    return ua, ub, joint


def variation_of_information(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None,
    ignore_background: bool = True,
) -> float:
    """Variation of information (bits) between two labelings of the same grid.

    Label-invariant: identical partitions give 0 regardless of label values.
    When ``ignore_background`` is True, elements that are 0 in either labeling
    are excluded from the comparison domain.
    """
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if ignore_background:
        keep = (a > 0) & (b > 0)
        a = a[keep]
        b = b[keep]
    n = a.size
    if n == 0:
        return np.nan
    _, _, joint = _contingency(a, b)
    p = joint / n
    pa = p.sum(axis=1)
    pb = p.sum(axis=0)
    nz = p > 0
    log_p = np.zeros_like(p)
    log_p[nz] = np.log2(p[nz])
    h_ab = -(p[nz] * log_p[nz]).sum()
    h_a = -(pa[pa > 0] * np.log2(pa[pa > 0])).sum()
    h_b = -(pb[pb > 0] * np.log2(pb[pb > 0])).sum()
    # VI = H(A|B) + H(B|A) = 2*H(A,B) - H(A) - H(B)
    return float(2.0 * h_ab - h_a - h_b)


def matched_overlap(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None,
    ignore_background: bool = True,
) -> float:
    """Size-weighted symmetric best-match IoU between two labelings.

    For each region in ``a`` the best IoU against any region of ``b`` is
    found; the size-weighted mean of these is averaged with the same measure
    computed in the opposite direction. 1.0 means identical partitions;
    label-invariant.
    """
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if ignore_background:
        keep = (a > 0) & (b > 0)
        a = a[keep]
        b = b[keep]
    if a.size == 0:
        return np.nan
    _, _, joint = _contingency(a, b)
    sa = joint.sum(axis=1).astype(float)
    sb = joint.sum(axis=0).astype(float)
    union = sa[:, None] + sb[None, :] - joint
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, joint / union, 0.0)
    best_a = iou.max(axis=1)
    best_b = iou.max(axis=0)
    wa = (best_a * sa).sum() / sa.sum()
    wb = (best_b * sb).sum() / sb.sum()
    return float(0.5 * (wa + wb))


def connected_component_report(
    labels: np.ndarray, connectivity: int = 1
) -> dict:
    """Check that every label forms one connected component.

    Returns
    -------
    dict with:
      - ``n_labels``: number of positive labels;
      - ``n_disconnected``: labels split into more than one component;
      - ``fragments``: {label: sorted component sizes (descending)} for the
        disconnected labels only;
      - ``fragment_voxel_fraction``: voxels outside each label's largest
        component, as a fraction of all labelled voxels.
    """
    struct = _connectivity_structure(labels.ndim, connectivity)
    fragments = {}
    total_labelled = int((labels > 0).sum())
    stray = 0
    n_labels = 0
    objects = ndimage.find_objects(labels)
    for lbl, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        n_labels += 1
        cell = labels[sl] == lbl
        cc, n_cc = ndimage.label(cell, structure=struct)
        if n_cc > 1:
            sizes = np.sort(np.bincount(cc.ravel())[1:])[::-1]
            fragments[int(lbl)] = sizes.tolist()
            stray += int(sizes[1:].sum())
    return {
        "n_labels": n_labels,
        "n_disconnected": len(fragments),
        "fragments": fragments,
        "fragment_voxel_fraction": (stray / total_labelled) if total_labelled else 0.0,
    }


def split_disconnected_labels(
    labels: np.ndarray, connectivity: int = 1, min_size: int = 1
) -> tuple[np.ndarray, dict]:
    """Deterministically split labels that are not connected.

    Every connected component (under ``connectivity``) of every label keeps
    its own label id: the largest component retains the original id, further
    components get fresh ids appended after the current maximum, ordered by
    (original label, component size descending, first voxel index) so the
    result is reproducible. Components smaller than ``min_size`` are set to 0.

    Returns (new_labels, info) where info records the applied splits.
    """
    struct = _connectivity_structure(labels.ndim, connectivity)
    out = labels.copy()
    next_id = int(labels.max()) + 1
    info = {"splits": {}, "removed_voxels": 0, "min_size": min_size,
            "connectivity": connectivity}
    objects = ndimage.find_objects(labels)
    for lbl, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        cell = labels[sl] == lbl
        cc, n_cc = ndimage.label(cell, structure=struct)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        # order components by size descending, ties by component id
        order = sorted(range(1, n_cc + 1), key=lambda i: (-sizes[i], i))
        new_ids = []
        for rank, comp in enumerate(order):
            comp_mask = cc == comp
            if sizes[comp] < min_size:
                out[sl][comp_mask] = 0
                info["removed_voxels"] += int(sizes[comp])
                new_ids.append(0)
            elif rank == 0:
                new_ids.append(int(lbl))
            else:
                out[sl][comp_mask] = next_id
                new_ids.append(next_id)
                next_id += 1
        info["splits"][int(lbl)] = {
            "component_sizes": [int(sizes[i]) for i in order],
            "assigned_ids": new_ids,
        }
    return out, info
