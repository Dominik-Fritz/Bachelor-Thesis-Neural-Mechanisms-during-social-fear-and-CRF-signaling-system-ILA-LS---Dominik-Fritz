import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from brainglobe_atlasapi import BrainGlobeAtlas


# ============================================================
# SETTINGS
# ============================================================

ATLAS_NAME = "allen_mouse_10um"

# Accepted CCFv3-to-Bregma approximation for Allen Mouse Atlas
BREGMA_AP_INDEX = 540
VOXEL_SIZE_MM = 0.01  # 10 µm atlas

ILA_BREGMA_VALUES = [1.94, 1.78, 1.62, 1.54]
LS_BREGMA_VALUES = [0.86, 0.82, 0.78, 0.74]

FIG_W = 9.2
FIG_H = 11.2

OUTER_LINEWIDTH = 1.4
INTERNAL_LINEWIDTH = 0.35
TARGET_LINEWIDTH = 2.2

OUTER_COLOR = "black"
INTERNAL_COLOR = "0.55"
TARGET_COLOR = "red"

CROP_PAD = 12


# ============================================================
# HELPERS
# ============================================================

def bregma_to_ap_index(bregma_mm: float) -> int:
    """
    Convert Bregma coordinate (mm) to AP slice index for allen_mouse_10um.
    """
    return int(round(BREGMA_AP_INDEX - (bregma_mm / VOXEL_SIZE_MM)))


def get_region_and_descendants(structures: dict, acronym: str):
    root_id = None
    for sid, info in structures.items():
        if info.get("acronym") == acronym:
            root_id = sid
            break

    if root_id is None:
        raise RuntimeError(f"Could not find region acronym '{acronym}'.")

    ids = []
    for sid, info in structures.items():
        path = info.get("structure_id_path", [])
        if root_id in path:
            ids.append(sid)

    return root_id, np.array(ids, dtype=np.int64)


def structure_mask(slice_annot: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.isin(slice_annot, ids)


def crop_to_brain(slice_annot: np.ndarray, pad: int = CROP_PAD):
    brain = slice_annot > 0
    ys, xs = np.where(brain)

    if len(xs) == 0 or len(ys) == 0:
        return slice(None), slice(None)

    y0 = max(0, ys.min() - pad)
    y1 = min(slice_annot.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(slice_annot.shape[1], xs.max() + pad + 1)

    return slice(y0, y1), slice(x0, x1)


def orient_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Flip vertically so dorsal is up in the plotted coronal slice.
    """
    return np.flipud(arr)


def boundary_mask_from_labels(lbl: np.ndarray) -> np.ndarray:
    """
    Combined boundary mask for all internal atlas region borders.
    """
    edge = np.zeros_like(lbl, dtype=bool)

    edge[:-1, :] |= (lbl[:-1, :] != lbl[1:, :])
    edge[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:])

    bg = lbl == 0
    edge &= ~bg

    return edge


def binary_edge(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask ^ ndimage.binary_erosion(mask)


def draw_panel(ax, slice_annot_cropped: np.ndarray, target_ids: np.ndarray, bregma_value: float):
    display_lbl = orient_for_display(slice_annot_cropped)
    display_target = orient_for_display(structure_mask(slice_annot_cropped, target_ids))
    display_brain = display_lbl > 0

    outer_edge = binary_edge(display_brain)
    inner_edge = boundary_mask_from_labels(display_lbl)
    target_edge = binary_edge(display_target)

    # internal atlas borders
    ax.contour(
        inner_edge.astype(float),
        levels=[0.5],
        colors=INTERNAL_COLOR,
        linewidths=INTERNAL_LINEWIDTH
    )

    # whole-slice outline
    ax.contour(
        outer_edge.astype(float),
        levels=[0.5],
        colors=OUTER_COLOR,
        linewidths=OUTER_LINEWIDTH
    )

    # target region outline
    if np.any(target_edge):
        ax.contour(
            target_edge.astype(float),
            levels=[0.5],
            colors=TARGET_COLOR,
            linewidths=TARGET_LINEWIDTH
        )

    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.02, 0.96,
        f"Bregma +{bregma_value:.2f} mm",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="0.96",
            edgecolor="0.75",
            linewidth=0.8
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading atlas: {ATLAS_NAME}")
    atlas = BrainGlobeAtlas(ATLAS_NAME)
    annotation = atlas.annotation
    structures = atlas.structures
    print("Atlas loaded.")

    _, ila_ids = get_region_and_descendants(structures, "ILA")
    _, ls_ids = get_region_and_descendants(structures, "LS")

    fig, axes = plt.subplots(
        4, 2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={"wspace": 0.03, "hspace": 0.08}
    )

    # title block
    fig.suptitle(
        "Brain regions analyzed for RNAscope",
        fontsize=22,
        fontweight="bold",
        y=0.975
    )

    fig.text(
        0.27, 0.91,
        "Infralimbic area (ILA)",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold"
    )

    fig.text(
        0.73, 0.91,
        "Lateral septum (LS)",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold"
    )

    fig.text(
        0.27, 0.888,
        "Bregma +1.94 to +1.54 mm",
        ha="center",
        va="bottom",
        fontsize=13
    )

    fig.text(
        0.73, 0.888,
        "Bregma +0.86 to +0.74 mm",
        ha="center",
        va="bottom",
        fontsize=13
    )

    # left column: ILA
    for i, bregma in enumerate(ILA_BREGMA_VALUES):
        ap_idx = bregma_to_ap_index(bregma)
        ap_idx = max(0, min(ap_idx, annotation.shape[0] - 1))

        slice_annot = annotation[ap_idx, :, :]
        ys, xs = crop_to_brain(slice_annot)
        cropped = slice_annot[ys, xs]

        draw_panel(axes[i, 0], cropped, ila_ids, bregma)

    # right column: LS
    for i, bregma in enumerate(LS_BREGMA_VALUES):
        ap_idx = bregma_to_ap_index(bregma)
        ap_idx = max(0, min(ap_idx, annotation.shape[0] - 1))

        slice_annot = annotation[ap_idx, :, :]
        ys, xs = crop_to_brain(slice_annot)
        cropped = slice_annot[ys, xs]

        draw_panel(axes[i, 1], cropped, ls_ids, bregma)

    plt.subplots_adjust(
        top=0.86,
        bottom=0.04,
        left=0.06,
        right=0.94
    )

    plt.savefig("brain_regions_rnascope_ap540.png", dpi=600, bbox_inches="tight")
    plt.savefig("brain_regions_rnascope_ap540.pdf", bbox_inches="tight")

    plt.show()

    print("Saved:")
    print("  brain_regions_rnascope_ap540.png")
    print("  brain_regions_rnascope_ap540.pdf")


if __name__ == "__main__":
    main()
