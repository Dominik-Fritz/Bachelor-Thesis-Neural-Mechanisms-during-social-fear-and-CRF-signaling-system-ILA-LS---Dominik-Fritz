import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.gridspec import GridSpec
from brainglobe_atlasapi import BrainGlobeAtlas


# ============================================================
# SETTINGS
# ============================================================

ATLAS_NAME = "allen_mouse_10um"

BREGMA_AP_INDEX = 540
VOXEL_SIZE_MM = 0.01

# 36 slices (6x6 grid)
BREGMA_START_MM = 2.4
BREGMA_END_MM = -4.6
STEP_MM = 0.2

N_ROWS = 6
N_COLS = 6

# figure geometry
PANEL_W = 2.15
PANEL_H = 1.55
HEADER_H = 1.00

# atlas line style
OUTER_LINEWIDTH = 0.95
INTERNAL_LINEWIDTH = 0.18
OUTER_COLOR = "black"
INTERNAL_COLOR = "0.65"

CROP_PAD = 8

TITLE = "Whole-brain coronal progression"
SUBTITLE = "Allen CCFv3 / BrainGlobe slices, 0.2 mm spacing"


# ============================================================
# HELPERS
# ============================================================

def bregma_to_ap_index(bregma_mm):
    return int(round(BREGMA_AP_INDEX - (bregma_mm / VOXEL_SIZE_MM)))


def generate_bregma_series(start_mm, end_mm, step_mm):
    vals = []
    x = start_mm

    while x >= end_mm - 1e-9:
        vals.append(round(x, 2))
        x -= step_mm

    return vals


def crop_to_brain(slice_annot, pad=CROP_PAD):

    brain = slice_annot > 0
    ys, xs = np.where(brain)

    if len(xs) == 0 or len(ys) == 0:
        return slice(None), slice(None)

    y0 = max(0, ys.min() - pad)
    y1 = min(slice_annot.shape[0], ys.max() + pad + 1)

    x0 = max(0, xs.min() - pad)
    x1 = min(slice_annot.shape[1], xs.max() + pad + 1)

    return slice(y0, y1), slice(x0, x1)


def orient_for_display(arr):
    return np.flipud(arr)


def boundary_mask_from_labels(lbl):

    edge = np.zeros_like(lbl, dtype=bool)

    edge[:-1, :] |= (lbl[:-1, :] != lbl[1:, :])
    edge[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:])

    bg = lbl == 0
    edge &= ~bg

    return edge


def binary_edge(mask):

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    return mask ^ ndimage.binary_erosion(mask)


def draw_panel(ax, slice_annot_cropped, bregma_value):

    display_lbl = orient_for_display(slice_annot_cropped)
    display_brain = display_lbl > 0

    outer_edge = binary_edge(display_brain)
    inner_edge = boundary_mask_from_labels(display_lbl)

    ax.contour(
        inner_edge.astype(float),
        levels=[0.5],
        colors=INTERNAL_COLOR,
        linewidths=INTERNAL_LINEWIDTH
    )

    ax.contour(
        outer_edge.astype(float),
        levels=[0.5],
        colors=OUTER_COLOR,
        linewidths=OUTER_LINEWIDTH
    )

    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.04, 0.92,
        f"{bregma_value:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor="0.96",
            edgecolor="0.75",
            linewidth=0.5
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    atlas = BrainGlobeAtlas(ATLAS_NAME)
    annotation = atlas.annotation

    bregma_values = generate_bregma_series(
        BREGMA_START_MM,
        BREGMA_END_MM,
        STEP_MM
    )

    if len(bregma_values) != N_ROWS * N_COLS:
        raise ValueError(
            f"Expected {N_ROWS*N_COLS} slices but got {len(bregma_values)}"
        )

    fig_w = N_COLS * PANEL_W
    fig_h = HEADER_H + N_ROWS * PANEL_H

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        N_ROWS + 1,
        N_COLS,
        figure=fig,
        height_ratios=[0.70] + [1]*N_ROWS,
        hspace=0.04,
        wspace=0.015
    )

    # ================= HEADER =================

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")

    header_ax.text(
        0.5,
        0.82,
        TITLE,
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold"
    )

    header_ax.text(
        0.5,
        0.20,
        f"{SUBTITLE}\nBregma {BREGMA_START_MM:+.2f} to {BREGMA_END_MM:+.2f} mm",
        ha="center",
        va="center",
        fontsize=12
    )

    # ================= PANELS =================

    for i, bregma in enumerate(bregma_values):

        r = i // N_COLS
        c = i % N_COLS

        ax = fig.add_subplot(gs[r+1, c])

        ap_idx = bregma_to_ap_index(bregma)
        ap_idx = max(0, min(ap_idx, annotation.shape[0]-1))

        slice_annot = annotation[ap_idx, :, :]

        ys, xs = crop_to_brain(slice_annot, pad=CROP_PAD)
        cropped = slice_annot[ys, xs]

        draw_panel(ax, cropped, bregma)

    plt.subplots_adjust(
        left=0.018,
        right=0.982,
        top=0.985,
        bottom=0.02
    )

    outfile_base = "whole_brain_coronal_progression_6x6"

    plt.savefig(
        f"{outfile_base}.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        f"{outfile_base}.pdf",
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
