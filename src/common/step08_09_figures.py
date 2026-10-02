"""Matplotlib figure builders for Step 8 (hierarchical clustering) and Step 9 (K-means) PDF/notebook style."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from common.figure_counter import bump_figure_number
from common.step1011_figures import COMPOSITE_PANEL_W_IN, COMPOSITE_WSPACE
from common.page_layout import (
    PDF_FS_FIG_AXIS,
    PDF_FS_FIG_LEGEND,
    PDF_FS_FIG_STRIP_PANEL,
    PDF_FS_FIG_SUPTITLE,
    PDF_FS_FIG_TICK,
    PDF_FS_FIG_TITLE,
)


def _ward_elbow_metrics(Z: np.ndarray) -> dict[str, np.ndarray | int | float]:
    """Sorted merge distances, Δ Ward, cumulative fraction, and crude elbow index."""
    distances = np.asarray(Z[:, 2], dtype=float)
    d_sorted = np.sort(distances)
    d_diff = np.diff(d_sorted)
    clusters_remaining = np.arange(len(d_sorted), 0, -1)
    elbow_idx = int(np.argmax(d_diff)) if d_diff.size else 0
    elbow_clusters_remaining = int(clusters_remaining[elbow_idx]) if clusters_remaining.size else 0
    d_total = float(np.sum(d_sorted)) if d_sorted.size else 0.0
    if d_total > 0:
        d_cum_frac = np.cumsum(d_sorted) / d_total
    else:
        d_cum_frac = np.zeros_like(d_sorted, dtype=float)
    return {
        "d_sorted": d_sorted,
        "d_diff": d_diff,
        "d_cum_frac": d_cum_frac,
        "clusters_remaining": clusters_remaining,
        "elbow_idx": elbow_idx,
        "elbow_clusters_remaining": elbow_clusters_remaining,
    }


def build_figure_ward_distance_structure(
    Z: np.ndarray,
    *,
    fig_num: list[int] | None,
    k_selected: int | None = None,
) -> Figure:
    """2×2 Ward HC diagnostic: elbow (left) and cumulative merge structure (right)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] < 3 or Z.shape[0] < 2:
        fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor="white")
        ax.axis("off")
        ax.set_title("Ward distance structure (invalid linkage matrix)", fontsize=PDF_FS_FIG_TITLE)
        return fig

    m = _ward_elbow_metrics(Z)
    d_sorted = m["d_sorted"]
    d_diff = m["d_diff"]
    d_cum_frac = m["d_cum_frac"]
    clusters_remaining = m["clusters_remaining"]
    elbow_idx = int(m["elbow_idx"])
    elbow_clusters_remaining = int(m["elbow_clusters_remaining"])

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.4), facecolor="white")
    n = bump_figure_number(fig_num)
    st_main = (
        f"Ward HC — elbow & cumulative structure "
        f"(elbow≈rank {elbow_idx}, clusters remaining≈{elbow_clusters_remaining})"
    )
    fig.suptitle((f"Figure {n} — {st_main}" if n else st_main), fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)

    axs[0, 0].plot(d_diff, lw=1.5, color="C1")
    axs[0, 0].set_title("Distance jumps (Δ Ward) — elbow", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 0].set_xlabel("merge rank", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].set_ylabel("jump magnitude", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].axvline(elbow_idx, color="k", ls="--", lw=1, alpha=0.7)

    axs[1, 0].plot(d_sorted, lw=1.5, color="C0")
    axs[1, 0].set_title("Ward distances (sorted)", fontsize=PDF_FS_FIG_TITLE)
    axs[1, 0].set_xlabel("merge rank", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].set_ylabel("distance", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].axvline(elbow_idx, color="k", ls="--", lw=1, alpha=0.7)

    axs[0, 1].plot(np.arange(1, d_cum_frac.size + 1), d_cum_frac, lw=1.5, color="C2")
    axs[0, 1].set_title("Cumulative merge distance (fraction of total)", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 1].set_xlabel("merge rank", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylabel("cumulative fraction", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylim(0.0, 1.02)
    axs[0, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].axvline(elbow_idx + 1, color="k", ls="--", lw=1, alpha=0.7)

    axs[1, 1].plot(clusters_remaining, d_sorted, lw=1.5, color="C2")
    axs[1, 1].set_title("Merge distance vs clusters remaining", fontsize=PDF_FS_FIG_TITLE)
    axs[1, 1].set_xlabel("clusters remaining", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 1].set_ylabel("distance", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].axvline(elbow_clusters_remaining, color="k", ls="--", lw=1, alpha=0.7)
    if k_selected is not None and int(k_selected) >= 2:
        axs[1, 1].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9, label=f"selected k={k_selected}")
        axs[1, 1].legend(fontsize=PDF_FS_FIG_TICK, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def compute_kmeans_inertia_sweep(
    X: np.ndarray,
    *,
    k_min: int = 1,
    k_max: int = 11,
    n_init: int = 50,
    random_state: int = 42,
) -> tuple[list[int], list[float]]:
    """Inertia (SSE) for k = k_min … k_max (matches notebook 09 elbow sweep)."""
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    k_hi = min(int(k_max), max(int(k_min), n - 1))
    ks: list[int] = []
    inertias: list[float] = []
    for k in range(int(k_min), k_hi + 1):
        km = KMeans(n_clusters=k, n_init=int(n_init), random_state=int(random_state))
        km.fit(X)
        ks.append(k)
        inertias.append(float(km.inertia_))
    return ks, inertias


def _kmeans_elbow_metrics(
    ks: list[int],
    inertias: list[float],
) -> dict[str, np.ndarray | int | float | list[int] | list[float]]:
    """Δ inertia elbow index and cumulative SSE reduction fraction."""
    k_arr = np.asarray(ks, dtype=int)
    inert = np.asarray(inertias, dtype=float)
    d_inert = np.diff(inert) if inert.size > 1 else np.array([], dtype=float)
    # Inertia decreases with k; elbow ≈ where marginal improvement (|Δ|) is smallest.
    elbow_k_idx = int(np.argmax(d_inert)) if d_inert.size else 0
    elbow_k = int(k_arr[elbow_k_idx + 1]) if d_inert.size and elbow_k_idx + 1 < k_arr.size else int(k_arr[0])
    span = float(inert[0] - inert[-1]) if inert.size >= 2 else 0.0
    if span > 0:
        cum_red = (inert[0] - inert) / span
    else:
        cum_red = np.zeros_like(inert, dtype=float)
    return {
        "ks": ks,
        "inertias": inertias,
        "k_arr": k_arr,
        "inert": inert,
        "d_inert": d_inert,
        "elbow_k_idx": elbow_k_idx,
        "elbow_k": elbow_k,
        "cum_red": cum_red,
    }


def build_figure_kmeans_elbow_cumulative_2x2(
    ks: list[int],
    inertias: list[float],
    *,
    fig_num: list[int] | None,
    k_selected: int | None = None,
) -> Figure:
    """2×2 K-means diagnostic: inertia elbow (left) and cumulative SSE reduction (right)."""
    if not ks or not inertias:
        fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor="white")
        ax.axis("off")
        ax.set_title("K-means elbow (no inertia sweep data)", fontsize=PDF_FS_FIG_TITLE)
        return fig

    m = _kmeans_elbow_metrics(ks, inertias)
    k_arr = m["k_arr"]
    inert = m["inert"]
    d_inert = m["d_inert"]
    cum_red = m["cum_red"]
    elbow_k_idx = int(m["elbow_k_idx"])
    elbow_k = int(m["elbow_k"])

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.4), facecolor="white")
    n = bump_figure_number(fig_num)
    st_main = f"K-means — elbow & cumulative inertia (largest Δ-SSE drop at k≈{elbow_k})"
    fig.suptitle((f"Figure {n} — {st_main}" if n else st_main), fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)

    axs[0, 0].plot(k_arr, inert, marker="o", lw=1.5, color="C0")
    axs[0, 0].set_title("Inertia (SSE) vs k — elbow", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 0].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].set_ylabel("inertia (SSE)", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].set_xticks(k_arr)
    axs[0, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 0].grid(True, alpha=0.3)
    if d_inert.size:
        axs[0, 0].axvline(elbow_k, color="k", ls="--", lw=1, alpha=0.7)
    if k_selected is not None:
        axs[0, 0].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    if d_inert.size:
        k_diff = k_arr[1:]
        axs[1, 0].plot(k_diff, d_inert, marker="o", lw=1.5, color="C1")
        axs[1, 0].axvline(elbow_k, color="k", ls="--", lw=1, alpha=0.7)
    axs[1, 0].set_title("Δ inertia vs k", fontsize=PDF_FS_FIG_TITLE)
    axs[1, 0].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].set_ylabel("Δ inertia", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[1, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(k_arr, cum_red, marker="o", lw=1.5, color="C2")
    axs[0, 1].set_title("Cumulative SSE reduction vs k", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 1].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylabel("fraction of reducible SSE", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylim(0.0, 1.02)
    axs[0, 1].set_xticks(k_arr)
    axs[0, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 1].grid(True, alpha=0.3)
    if k_selected is not None:
        axs[0, 1].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    axs[1, 1].plot(k_arr[1:], -d_inert, marker="o", lw=1.5, color="C2") if d_inert.size else axs[1, 1].axis("off")
    if d_inert.size:
        axs[1, 1].set_title("Per-step SSE reduction (positive = improvement)", fontsize=PDF_FS_FIG_TITLE)
        axs[1, 1].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
        axs[1, 1].set_ylabel("-Δ inertia", fontsize=PDF_FS_FIG_AXIS)
        axs[1, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
        axs[1, 1].grid(True, alpha=0.3)
        if k_selected is not None:
            axs[1, 1].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def compute_ward_inertia_sweep(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 11,
) -> tuple[list[int], list[float]]:
    """Within-cluster SSE for Ward cuts k = k_min … k_max (analogue of K-means inertia).

    For each k the Ward linkage ``Z`` is cut via ``scipy.cluster.hierarchy.fcluster``
    (criterion ``maxclust``); the resulting labels are used to compute the
    within-cluster sum of squared Euclidean distances to the per-cluster centroid
    in the same feature space ``X``. This mirrors the definition of
    ``sklearn.cluster.KMeans.inertia_`` so the elbow/cumulative diagnostics are
    directly comparable between Ward (Step 8) and K-means (Step 9).
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=float)
    n = int(X.shape[0])
    if Z.ndim != 2 or Z.shape[1] < 4 or Z.shape[0] != n - 1:
        return [], []
    try:
        from scipy.cluster.hierarchy import fcluster
    except Exception:
        return [], []

    k_lo = max(2, int(k_min))
    k_hi = min(int(k_max), max(k_lo, n - 1))
    ks: list[int] = []
    inertias: list[float] = []
    for k in range(k_lo, k_hi + 1):
        try:
            labels = np.asarray(fcluster(Z, k, criterion="maxclust"), dtype=int)
        except Exception:
            continue
        uniq = np.unique(labels)
        if uniq.size < 1:
            continue
        wss = 0.0
        for u in uniq:
            members = X[labels == u]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(axis=0)
            wss += float(np.sum((members - centroid) ** 2))
        ks.append(k)
        inertias.append(wss)
    return ks, inertias


def build_figure_ward_inertia_cumulative_2x2(
    ks: list[int],
    inertias: list[float],
    *,
    fig_num: list[int] | None,
    k_selected: int | None = None,
) -> Figure:
    """2×2 Ward HC diagnostic mirroring the K-means elbow: within-cluster SSE vs k."""
    if not ks or not inertias:
        fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor="white")
        ax.axis("off")
        ax.set_title("Ward HC inertia (no sweep data)", fontsize=PDF_FS_FIG_TITLE)
        return fig

    m = _kmeans_elbow_metrics(ks, inertias)
    k_arr = m["k_arr"]
    inert = m["inert"]
    d_inert = m["d_inert"]
    cum_red = m["cum_red"]
    elbow_k = int(m["elbow_k"])

    # Larger canvas + explicit subplot padding so the 16/14 pt fonts
    # (long suptitle, axes titles, x-tick labels) cannot collide with
    # the plot interior. Panel layout (2×2) is unchanged.
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.5), facecolor="white")
    n = bump_figure_number(fig_num)
    st_main = (
        f"Ward HC — within-cluster SSE elbow & cumulative reduction "
        f"(largest Δ-SSE drop at k≈{elbow_k})"
    )
    fig.suptitle((f"Figure {n} — {st_main}" if n else st_main), fontsize=PDF_FS_FIG_SUPTITLE, y=0.985)

    axs[0, 0].plot(k_arr, inert, marker="o", lw=1.5, color="C0")
    axs[0, 0].set_title("Within-cluster SSE vs k (Ward) — elbow", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 0].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].set_ylabel("within-cluster SSE", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 0].set_xticks(k_arr)
    axs[0, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 0].grid(True, alpha=0.3)
    if d_inert.size:
        axs[0, 0].axvline(elbow_k, color="k", ls="--", lw=1, alpha=0.7)
    if k_selected is not None:
        axs[0, 0].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    if d_inert.size:
        k_diff = k_arr[1:]
        axs[1, 0].plot(k_diff, d_inert, marker="o", lw=1.5, color="C1")
        axs[1, 0].axvline(elbow_k, color="k", ls="--", lw=1, alpha=0.7)
    axs[1, 0].set_title("Δ within-cluster SSE vs k (Ward)", fontsize=PDF_FS_FIG_TITLE)
    axs[1, 0].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].set_ylabel("Δ SSE", fontsize=PDF_FS_FIG_AXIS)
    axs[1, 0].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[1, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(k_arr, cum_red, marker="o", lw=1.5, color="C2")
    axs[0, 1].set_title("Cumulative SSE reduction vs k (Ward)", fontsize=PDF_FS_FIG_TITLE)
    axs[0, 1].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylabel("fraction of reducible SSE", fontsize=PDF_FS_FIG_AXIS)
    axs[0, 1].set_ylim(0.0, 1.02)
    axs[0, 1].set_xticks(k_arr)
    axs[0, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    axs[0, 1].grid(True, alpha=0.3)
    if k_selected is not None:
        axs[0, 1].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    if d_inert.size:
        axs[1, 1].plot(k_arr[1:], -d_inert, marker="o", lw=1.5, color="C2")
        axs[1, 1].set_title("Per-step SSE reduction (Ward; positive = improvement)", fontsize=PDF_FS_FIG_TITLE)
        axs[1, 1].set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
        axs[1, 1].set_ylabel("-Δ SSE", fontsize=PDF_FS_FIG_AXIS)
        axs[1, 1].tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
        axs[1, 1].grid(True, alpha=0.3)
        if k_selected is not None:
            axs[1, 1].axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)
    else:
        axs[1, 1].axis("off")

    # h_pad/w_pad keep the bottom-row titles clear of the upper-row tick
    # labels at the bumped 16/14 pt fonts; rect top leaves a wider band
    # for the long suptitle.
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.2, w_pad=1.8)
    return fig


def compute_hc_evaluation_metrics_sweep(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 10,
) -> dict[str, list]:
    """Sweep k for Ward HC labels and compute four evaluation metrics in one pass.

    All metrics are computed strictly from ``scipy.cluster.hierarchy.fcluster`` cuts of
    the Ward linkage ``Z`` — never from K-means or any other method. Returned dict has
    keys ``ks``, ``silhouettes`` (higher = better), ``wcss`` (lower = better; identical
    definition to ``KMeans.inertia_`` but on Ward labels — labelled as compactness in
    the figure to avoid conflating with K-means), ``davies_bouldin`` (lower = better),
    ``calinski_harabasz`` (higher = better).
    """
    empty: dict[str, list] = {
        "ks": [],
        "silhouettes": [],
        "wcss": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=float)
    n = int(X.shape[0])
    if Z.ndim != 2 or Z.shape[1] < 4 or Z.shape[0] != n - 1:
        return empty
    try:
        from scipy.cluster.hierarchy import fcluster
    except Exception:
        return empty
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except Exception:
        return empty

    k_lo = max(2, int(k_min))
    k_hi = min(int(k_max), max(k_lo, n - 1))
    ks: list[int] = []
    sils: list[float] = []
    wcss: list[float] = []
    dbs: list[float] = []
    chs: list[float] = []
    for kk in range(k_lo, k_hi + 1):
        try:
            labels = np.asarray(fcluster(Z, kk, criterion="maxclust"), dtype=int)
        except Exception:
            continue
        uniq = np.unique(labels)
        if uniq.size < 2:
            continue
        cluster_wss = 0.0
        for u in uniq:
            members = X[labels == u]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(axis=0)
            cluster_wss += float(np.sum((members - centroid) ** 2))
        try:
            s = float(silhouette_score(X, labels, metric="euclidean"))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
        except Exception:
            continue
        ks.append(kk)
        sils.append(s)
        wcss.append(cluster_wss)
        dbs.append(db)
        chs.append(ch)
    return {
        "ks": ks,
        "silhouettes": sils,
        "wcss": wcss,
        "davies_bouldin": dbs,
        "calinski_harabasz": chs,
    }


def _hc_eval_summary_line(metrics: dict[str, list]) -> str:
    """One-line interpretation summary for the 2×2 HC evaluation figure."""
    ks = metrics.get("ks") or []
    sils = metrics.get("silhouettes") or []
    wcss = metrics.get("wcss") or []
    dbs = metrics.get("davies_bouldin") or []
    chs = metrics.get("calinski_harabasz") or []
    if not ks:
        return "No HC evaluation metrics available."
    k_arr = np.asarray(ks, dtype=int)
    parts: list[str] = []
    if sils:
        idx = int(np.argmax(sils))
        parts.append(f"best silhouette: k={int(k_arr[idx])} ({float(sils[idx]):.3f})")
    if wcss and len(wcss) > 1:
        diffs = -np.diff(np.asarray(wcss, dtype=float))  # drop magnitudes
        elbow_idx = int(np.argmax(diffs)) + 1
        parts.append(f"compactness elbow: k≈{int(k_arr[elbow_idx])}")
    if dbs:
        idx = int(np.argmin(dbs))
        parts.append(f"best Davies–Bouldin: k={int(k_arr[idx])} ({float(dbs[idx]):.3f})")
    if chs:
        idx = int(np.argmax(chs))
        parts.append(f"best Calinski–Harabasz: k={int(k_arr[idx])} ({float(chs[idx]):.0f})")
    return " | ".join(parts)


def build_figure_hc_evaluation_metrics_2x2(
    metrics: dict[str, list],
    *,
    fig_num: list[int] | None,
    k_selected: int | None = None,
) -> Figure:
    """2×2 evaluation figure computed strictly from Ward HC labels (no K-means).

    Panels: (A) silhouette vs k, (B) within-cluster SSE compactness vs k (HC, not
    K-means inertia), (C) Davies–Bouldin vs k, (D) Calinski–Harabasz vs k. A
    one-line interpretation summary is drawn below the panels.
    """
    ks = metrics.get("ks") or []
    sils = metrics.get("silhouettes") or []
    wcss = metrics.get("wcss") or []
    dbs = metrics.get("davies_bouldin") or []
    chs = metrics.get("calinski_harabasz") or []

    if not ks:
        fig, ax = plt.subplots(figsize=(8.6, 6.0), facecolor="white")
        ax.axis("off")
        ax.set_title(
            "Hierarchical Clustering Evaluation Metrics (no sweep data)",
            fontsize=PDF_FS_FIG_TITLE,
        )
        return fig

    k_arr = np.asarray(ks, dtype=int)
    # Larger canvas + explicit subplot padding so the 16/14 pt fonts
    # (long panel titles such as "Compactness vs k …", legends, and
    # the bottom summary line) cannot collide with the plot interior.
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 9.5), facecolor="white")
    n = bump_figure_number(fig_num)
    suptitle = "Hierarchical Clustering Evaluation Metrics"
    fig.suptitle(
        (f"Figure {n} — {suptitle}" if n else suptitle),
        fontsize=PDF_FS_FIG_SUPTITLE,
        y=0.99,
    )

    def _annotate_selected(ax_):
        if k_selected is not None:
            ax_.axvline(int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9)

    ax = axs[0, 0]
    if sils:
        ax.plot(k_arr, sils, marker="o", color="C0", lw=1.5)
        ax.set_xticks(k_arr)
        idx = int(np.argmax(sils))
        ax.axvline(
            int(k_arr[idx]), color="k", ls="--", lw=1, alpha=0.7,
            label=f"max @ k={int(k_arr[idx])}",
        )
        ax.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")
    ax.set_title("Silhouette vs k (Hierarchical Clustering)", fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("silhouette (higher = better)", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.3)
    _annotate_selected(ax)

    ax = axs[0, 1]
    if wcss:
        ax.plot(k_arr, wcss, marker="o", color="C1", lw=1.5)
        ax.set_xticks(k_arr)
        if len(wcss) > 1:
            diffs = -np.diff(np.asarray(wcss, dtype=float))
            elbow_idx = int(np.argmax(diffs)) + 1
            ax.axvline(
                int(k_arr[elbow_idx]), color="k", ls="--", lw=1, alpha=0.7,
                label=f"elbow ≈ k={int(k_arr[elbow_idx])}",
            )
            ax.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")
    ax.set_title(
        "Compactness vs k (Hierarchical Clustering;\nwithin-cluster SSE, not K-means inertia)",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("within-cluster SSE (lower = tighter)", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.3)
    _annotate_selected(ax)

    ax = axs[1, 0]
    if dbs:
        ax.plot(k_arr, dbs, marker="o", color="C2", lw=1.5)
        ax.set_xticks(k_arr)
        idx = int(np.argmin(dbs))
        ax.axvline(
            int(k_arr[idx]), color="k", ls="--", lw=1, alpha=0.7,
            label=f"min @ k={int(k_arr[idx])}",
        )
        ax.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")
    ax.set_title("Davies–Bouldin vs k (Hierarchical Clustering)", fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("Davies–Bouldin (lower = better)", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.3)
    _annotate_selected(ax)

    ax = axs[1, 1]
    if chs:
        ax.plot(k_arr, chs, marker="o", color="C4", lw=1.5)
        ax.set_xticks(k_arr)
        idx = int(np.argmax(chs))
        ax.axvline(
            int(k_arr[idx]), color="k", ls="--", lw=1, alpha=0.7,
            label=f"max @ k={int(k_arr[idx])}",
        )
        ax.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")
    ax.set_title("Calinski–Harabasz vs k (Hierarchical Clustering)", fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("Calinski–Harabasz (higher = better)", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.3)
    _annotate_selected(ax)

    summary = _hc_eval_summary_line(metrics)
    if k_selected is not None:
        summary = f"selected k = {int(k_selected)}  |  " + summary
    fig.text(
        0.5, 0.03, summary,
        ha="center", va="bottom",
        fontsize=PDF_FS_FIG_AXIS,
        wrap=True,
    )

    # h_pad/w_pad keep the bottom-row titles clear of the upper-row tick
    # labels and the summary line clear of the (1,0)/(1,1) panels at the
    # bumped 16/14 pt fonts.
    fig.tight_layout(rect=[0, 0.07, 1, 0.94], h_pad=2.2, w_pad=1.8)
    return fig


def build_figure_silhouette_vs_k(
    ks: list[int],
    sils: list[float],
    *,
    fig_num: list[int] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    n = bump_figure_number(fig_num)
    ax.plot(ks, sils, marker="o", color="C0")
    ax.set_xticks(ks)
    ax.set_title(
        (f"Figure {n} — " if n else "") + "Silhouette score vs k (Ward labels, recomputed)",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("silhouette", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_figure_silhouette_and_kmeans_inertia_vs_k(
    ks_sil: list[int],
    sils: list[float],
    ks_inert: list[int],
    inertias: list[float],
    *,
    fig_num: list[int] | None,
    k_selected: int | None = None,
) -> Figure:
    """Two-row model-selection diagnostic: silhouette vs k (top) and K-means inertia vs k (bottom)."""
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.2), facecolor="white")
    n = bump_figure_number(fig_num)
    suptitle = "Silhouette (Ward) & K-means inertia vs k"
    fig.suptitle(
        (f"Figure {n} — {suptitle}" if n else suptitle),
        fontsize=PDF_FS_FIG_SUPTITLE,
        y=0.995,
    )

    ax_sil = axs[0]
    if ks_sil and sils:
        ax_sil.plot(ks_sil, sils, marker="o", color="C0")
        ax_sil.set_xticks(ks_sil)
    ax_sil.set_title(
        "Silhouette score vs k (Ward labels, recomputed)",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax_sil.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax_sil.set_ylabel("silhouette", fontsize=PDF_FS_FIG_AXIS)
    ax_sil.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax_sil.grid(True, alpha=0.3)
    if k_selected is not None and ks_sil:
        ax_sil.axvline(
            int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9, label=f"selected k={k_selected}"
        )
        ax_sil.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")

    ax_in = axs[1]
    if ks_inert and inertias:
        ax_in.plot(ks_inert, inertias, marker="o", color="C1")
        ax_in.set_xticks(ks_inert)
    ax_in.set_title(
        "K-means inertia (SSE) vs k — elbow",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax_in.set_xlabel("k", fontsize=PDF_FS_FIG_AXIS)
    ax_in.set_ylabel("inertia (SSE)", fontsize=PDF_FS_FIG_AXIS)
    ax_in.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax_in.grid(True, alpha=0.3)
    if k_selected is not None and ks_inert:
        ax_in.axvline(
            int(k_selected), color="C3", ls=":", lw=1.2, alpha=0.9, label=f"selected k={k_selected}"
        )
        ax_in.legend(fontsize=PDF_FS_FIG_LEGEND, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def build_figure_pc12_ward(
    pc1: np.ndarray,
    pc2: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    fig_num: list[int] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="white")
    n = bump_figure_number(fig_num)
    sc = ax.scatter(pc1, pc2, c=labels, s=8, cmap="tab10", alpha=0.85, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Ward label", fontsize=PDF_FS_FIG_AXIS)
    cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    ax.set_title(
        (f"Figure {n} — " if n else "") + f"PC1–PC2 by Ward labels (k={k})",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("PC1", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("PC2", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def build_figure_regime_frequency_bar(
    uniq: np.ndarray,
    frac: np.ndarray,
    *,
    k: int,
    ward: bool,
    fig_num: list[int] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.0), facecolor="white")
    n = bump_figure_number(fig_num)
    kind = "Ward hierarchical" if ward else "K-means"
    ax.bar(np.arange(1, uniq.size + 1), frac, color="C0" if ward else "C1", alpha=0.85)
    ax.set_title(
        (f"Figure {n} — " if n else "") + f"Regime frequency ({kind}, k={k})",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("fraction of days", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.set_xticks(np.arange(1, uniq.size + 1))
    ax.set_xticklabels([str(int(u)) for u in uniq])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def build_figure_pc12_ward_candidate_strip(
    pc1: np.ndarray,
    pc2: np.ndarray,
    *,
    pairs: list[tuple[int, np.ndarray]],
    fig_num: list[int] | None,
) -> Figure:
    """One row of PC1–PC2 scatters for several ``k`` (matches notebook ``candidate_ks`` strip)."""
    if not pairs:
        fig, ax = plt.subplots(figsize=(4.0, 2.5), facecolor="white")
        ax.axis("off")
        ax.set_title("PC1–PC2 (Ward) — no candidate k in range", fontsize=PDF_FS_FIG_TITLE)
        return fig

    n_k = len(pairs)
    fig_w = COMPOSITE_PANEL_W_IN * n_k + 0.35
    fig, axes = plt.subplots(1, n_k, figsize=(fig_w, 3.25), facecolor="white")
    if n_k == 1:
        axes = np.array([axes])
    n = bump_figure_number(fig_num)
    title = "PC1–PC2 by Ward labels (candidate k)"
    fig.suptitle((f"Figure {n} — {title}" if n else title), fontsize=PDF_FS_FIG_SUPTITLE, y=1.02)
    for ax, (kk, lab) in zip(np.atleast_1d(axes).ravel(), pairs):
        sc = ax.scatter(pc1, pc2, c=lab, s=5, cmap="tab10", alpha=0.85, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, shrink=0.72)
        cb.set_label("label", fontsize=PDF_FS_FIG_AXIS)
        cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
        ax.set_title(f"k={kk}", fontsize=PDF_FS_FIG_STRIP_PANEL)
        ax.set_xlabel("PC1", fontsize=PDF_FS_FIG_AXIS)
        ax.set_ylabel("PC2", fontsize=PDF_FS_FIG_AXIS)
        ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
        ax.grid(True, alpha=0.2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86 if n else 0.88, bottom=0.14, wspace=COMPOSITE_WSPACE)
    return fig


def build_figure_pc12_kmeans(
    pc1: np.ndarray,
    pc2: np.ndarray,
    yl: np.ndarray,
    *,
    k: int,
    fig_num: list[int] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="white")
    n = bump_figure_number(fig_num)
    sc = ax.scatter(pc1, pc2, c=yl, s=10, cmap="tab10", alpha=0.85, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("regime", fontsize=PDF_FS_FIG_AXIS)
    cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    ax.set_title(
        (f"Figure {n} — " if n else "") + f"PC1–PC2 coloured by regimes (K-means, k={k})",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("PC1", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("PC2", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def build_figure_step9_regime_frequency_from_ds(
    regc: np.ndarray,
    vals: np.ndarray,
    *,
    fig_num: list[int] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.0), facecolor="white")
    n = bump_figure_number(fig_num)
    ax.bar(np.arange(1, regc.size + 1), vals, color="C1", alpha=0.85)
    ax.set_title(
        (f"Figure {n} — " if n else "") + "Regime frequency (K-means)",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("fraction of days", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.set_xticks(np.arange(1, regc.size + 1))
    ax.set_xticklabels([str(int(r)) for r in regc])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def build_figure_regime_frequency_pct_days(
    regc: np.ndarray,
    pct: np.ndarray,
    n_days: np.ndarray,
    *,
    title: str = "Regime frequency",
    fig_num: list[int] | None = None,
) -> Figure:
    """Bar chart of regime frequency (% of days), each bar annotated with its raw day count.

    Pure figure builder over already-aggregated arrays (one value per regime) — meant for
    CSV-driven notebooks (e.g. ``2_3_Results_Z500_Stats.ipynb``) that no longer load the
    source NetCDF locally.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    n = bump_figure_number(fig_num)
    x = np.arange(1, regc.size + 1)
    bars = ax.bar(x, pct, color="C1", alpha=0.85)
    ax.bar_label(
        bars,
        labels=[f"{p:.1f}%\n(n={int(d)})" for p, d in zip(pct, n_days, strict=False)],
        fontsize=PDF_FS_FIG_TICK, padding=3,
    )
    ax.set_title((f"Figure {n} — " if n else "") + title, fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("% of days", fontsize=PDF_FS_FIG_AXIS)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in regc])
    ax.set_ylim(0, max(float(np.max(pct)), 0.01) * 1.25 if len(pct) else 1)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def build_figure_regime_monthly_composition(
    regc: np.ndarray,
    month_names: list[str],
    pct_of_all_matrix: np.ndarray,
    n_days_matrix: np.ndarray,
    *,
    title: str = "MJJAS composition × regime frequency",
    ylabel: str = "% of all aligned days",
    fig_num: list[int] | None = None,
) -> Figure:
    """Stacked bar per regime: MJJAS month composition scaled by overall regime frequency.

    ``pct_of_all_matrix`` and ``n_days_matrix`` are ``(n_regime, n_month)`` arrays (rows in
    ``regc`` order, columns in ``month_names`` order); each stack sums to that regime's overall
    frequency. Segments are annotated with their raw day count. Pure figure builder — same
    CSV-driven use case as :func:`build_figure_regime_frequency_pct_days`.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor="white")
    n = bump_figure_number(fig_num)
    x = np.arange(1, regc.size + 1)
    bottom = np.zeros(regc.size)
    colors = plt.cm.Blues(np.linspace(0.38, 0.92, max(len(month_names), 1)))
    for j, m in enumerate(month_names):
        vals = pct_of_all_matrix[:, j]
        ax.bar(x, vals, bottom=bottom, color=colors[j], label=m)
        for i, v in enumerate(vals):
            if v > 1.5:
                ax.text(
                    x[i], bottom[i] + v / 2, str(int(n_days_matrix[i, j])),
                    ha="center", va="center", fontsize=max(PDF_FS_FIG_TICK - 4, 6), color="black",
                )
        bottom += vals
    ax.set_title((f"Figure {n} — " if n else "") + title, fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel(ylabel, fontsize=PDF_FS_FIG_AXIS)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in regc])
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    ax.legend(title="month", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=max(PDF_FS_FIG_TICK - 4, 6))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
