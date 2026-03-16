import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================

GEOJSON_DIR = "geojson"
OUTPUT_DIR = "evaluation_output_ILA"
IOU_THRESHOLD = 0.5

CLASSIFIERS = ["CRF", "cfos"]

CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
OVERLAY_DIR = os.path.join(PLOT_DIR, "overlays_tp_fp_fn")
GT_PRED_DIR = os.path.join(PLOT_DIR, "overlays_gt_vs_pred")
METRIC_DIR = os.path.join(PLOT_DIR, "metrics_per_classifier")
PUBLICATION_DIR = os.path.join(PLOT_DIR, "publication")

for path in [OUTPUT_DIR, CSV_DIR, PLOT_DIR, OVERLAY_DIR, GT_PRED_DIR, METRIC_DIR, PUBLICATION_DIR]:
    os.makedirs(path, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def compute_iou(poly1, poly2):
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0


def normalize_classifier_name(name):
    for clf in CLASSIFIERS:
        if name.lower() == clf.lower():
            return clf
    return name


def clean_geodataframe(gdf):
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    if "geometry" not in gdf.columns:
        return gdf.iloc[0:0].copy()

    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gdf

    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gdf

    repaired_geoms = []
    for geom in gdf.geometry:
        try:
            if geom is None or geom.is_empty:
                repaired_geoms.append(None)
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            repaired_geoms.append(geom)
        except Exception:
            repaired_geoms.append(None)

    gdf["geometry"] = repaired_geoms
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gdf

    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gdf

    validity_mask = []
    for geom in gdf.geometry:
        try:
            validity_mask.append(bool(geom.is_valid))
        except Exception:
            validity_mask.append(False)

    gdf = gdf[validity_mask].copy()
    if gdf.empty:
        return gdf

    gdf = gdf.explode(index_parts=False)
    gdf = gdf.reset_index(drop=True)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    validity_mask = []
    for geom in gdf.geometry:
        try:
            validity_mask.append(bool(geom.is_valid))
        except Exception:
            validity_mask.append(False)

    gdf = gdf[validity_mask].copy()
    gdf = gdf.reset_index(drop=True)

    return gdf


def save_figure(fig, outpath_without_ext, dpi=300):
    fig.savefig(outpath_without_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outpath_without_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


def to_percent(values):
    return np.array(values, dtype=float) * 100.0


def load_geojson_pairs(folder):
    files = glob.glob(os.path.join(folder, "*.geojson"))

    clf_pattern = "|".join(re.escape(c) for c in CLASSIFIERS)
    pattern = re.compile(rf"^(.*)_({clf_pattern})_(GT|Pred)\.geojson$", re.IGNORECASE)

    pairs = {}

    for filepath in files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if not match:
            continue

        base, classifier, dtype = match.groups()
        classifier = normalize_classifier_name(classifier)

        key = (base, classifier)
        if key not in pairs:
            pairs[key] = {}

        if dtype.lower() == "gt":
            pairs[key]["GT"] = filepath
        elif dtype.lower() == "pred":
            pairs[key]["Pred"] = filepath

    return pairs


def assign_image_labels(pairs):
    unique_bases = sorted({base for base, _ in pairs.keys()})
    return {base: f"Image {i + 1}" for i, base in enumerate(unique_bases)}


def _finite_total_bounds(gdf):
    if gdf is None or gdf.empty:
        return None

    try:
        bounds = gdf.total_bounds
    except Exception:
        return None

    if bounds is None or len(bounds) != 4:
        return None

    if not np.all(np.isfinite(bounds)):
        return None

    minx, miny, maxx, maxy = bounds
    if minx == maxx and miny == maxy:
        return None

    return bounds


def _set_plot_limits(ax, gdf_list, pad_ratio=0.03):
    bounds_list = []

    for gdf in gdf_list:
        bounds = _finite_total_bounds(gdf)
        if bounds is not None:
            bounds_list.append(bounds)

    if not bounds_list:
        return

    bounds_arr = np.array(bounds_list)
    minx = np.min(bounds_arr[:, 0])
    miny = np.min(bounds_arr[:, 1])
    maxx = np.max(bounds_arr[:, 2])
    maxy = np.max(bounds_arr[:, 3])

    dx = maxx - minx
    dy = maxy - miny

    pad_x = dx * pad_ratio if dx > 0 else 1.0
    pad_y = dy * pad_ratio if dy > 0 else 1.0

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)


def _plot_single_geometry(ax, geom, facecolor, edgecolor="black", alpha=0.5, linewidth=1.0):
    try:
        gpd.GeoSeries([geom]).plot(
            ax=ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            aspect="auto"
        )
    except Exception:
        pass


def mean_and_sem(series):
    vals = series.dropna().values.astype(float)
    if len(vals) == 0:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(np.mean(vals)), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


# ============================================================
# MATCHING / EVALUATION
# ============================================================

def evaluate_pair(gt_path, pred_path, iou_threshold=IOU_THRESHOLD):
    gdf_gt_raw = gpd.read_file(gt_path)
    gdf_pred_raw = gpd.read_file(pred_path)

    gdf_gt = clean_geodataframe(gdf_gt_raw)
    gdf_pred = clean_geodataframe(gdf_pred_raw)

    gt_polys = list(gdf_gt.geometry) if not gdf_gt.empty else []
    pred_polys = list(gdf_pred.geometry) if not gdf_pred.empty else []

    matched_gt = set()
    matched_pred = set()
    matches = []

    for i, gt_geom in enumerate(gt_polys):
        best_iou = 0.0
        best_j = None

        for j, pred_geom in enumerate(pred_polys):
            if j in matched_pred:
                continue

            iou = compute_iou(gt_geom, pred_geom)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou >= iou_threshold:
            matched_gt.add(i)
            matched_pred.add(best_j)
            matches.append(
                {
                    "gt_index": i,
                    "pred_index": best_j,
                    "iou": best_iou,
                }
            )

    tp = len(matches)
    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "gdf_gt": gdf_gt,
        "gdf_pred": gdf_pred,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_tp_fp_fn_overlay(gdf_gt, gdf_pred, matched_gt, matched_pred, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 6))

    if gdf_gt is not None and not gdf_gt.empty:
        for i, geom in enumerate(gdf_gt.geometry):
            facecolor = "green" if i in matched_gt else "lightgray"
            _plot_single_geometry(ax, geom, facecolor=facecolor, edgecolor="black", alpha=0.5)

    if gdf_pred is not None and not gdf_pred.empty:
        for j, geom in enumerate(gdf_pred.geometry):
            if j not in matched_pred:
                _plot_single_geometry(ax, geom, facecolor="red", edgecolor="black", alpha=0.55)

    _set_plot_limits(ax, [gdf_gt, gdf_pred])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    save_figure(fig, outpath)


def plot_gt_vs_pred(gdf_gt, gdf_pred, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        if gdf_gt is not None and not gdf_gt.empty:
            gdf_gt.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1.2,
                aspect="auto"
            )
    except Exception:
        if gdf_gt is not None and not gdf_gt.empty:
            for geom in gdf_gt.geometry:
                _plot_single_geometry(ax, geom, facecolor="none", edgecolor="black", alpha=1.0, linewidth=1.2)

    try:
        if gdf_pred is not None and not gdf_pred.empty:
            gdf_pred.plot(
                ax=ax,
                facecolor="none",
                edgecolor="tab:blue",
                linewidth=1.2,
                aspect="auto"
            )
    except Exception:
        if gdf_pred is not None and not gdf_pred.empty:
            for geom in gdf_pred.geometry:
                _plot_single_geometry(ax, geom, facecolor="none", edgecolor="tab:blue", alpha=1.0, linewidth=1.2)

    _set_plot_limits(ax, [gdf_gt, gdf_pred])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    save_figure(fig, outpath)


def plot_metric_lines(df):
    metrics = ["precision", "recall", "f1"]

    for clf in sorted(df["classifier"].unique()):
        df_clf = df[df["classifier"] == clf].copy()
        df_clf["image_num"] = df_clf["image"].str.extract(r"(\d+)").astype(int)
        df_clf = df_clf.sort_values("image_num")

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            y = to_percent(df_clf[metric].values)

            ax.plot(df_clf["image"], y, marker="o", linewidth=1.8)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Image")
            ax.set_ylabel(f"{metric.capitalize()} (%)")
            ax.set_title(f"{clf}: {metric.capitalize()} across validation images")
            ax.grid(True, alpha=0.25)

            save_figure(fig, os.path.join(METRIC_DIR, f"{clf}_{metric}"))


def plot_grouped_metrics_by_classifier(df):
    metrics = ["precision", "recall", "f1"]
    classifiers = sorted(df["classifier"].unique())
    images = sorted(df["image"].unique(), key=lambda x: int(x.split()[-1]))

    for metric in metrics:
        fig, axes = plt.subplots(1, len(classifiers), figsize=(5.5 * len(classifiers), 4.8), sharey=True)

        if len(classifiers) == 1:
            axes = [axes]

        for ax, clf in zip(axes, classifiers):
            subset = df[df["classifier"] == clf].set_index("image").reindex(images)
            y = to_percent(subset[metric].values)

            ax.bar(images, y)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Image")
            ax.set_title(clf)
            ax.grid(axis="y", alpha=0.25)

        axes[0].set_ylabel(f"{metric.capitalize()} (%)")
        fig.suptitle(f"{metric.capitalize()} per image and classifier")
        save_figure(fig, os.path.join(PUBLICATION_DIR, f"{metric}_per_classifier_panel"))


def plot_tp_fp_fn_by_classifier(df):
    classifiers = sorted(df["classifier"].unique())
    images = sorted(df["image"].unique(), key=lambda x: int(x.split()[-1]))

    for clf in classifiers:
        subset = df[df["classifier"] == clf].set_index("image").reindex(images)

        tp = subset["tp"].values
        fp = subset["fp"].values
        fn = subset["fn"].values

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(images, tp, label="TP")
        ax.bar(images, fp, bottom=tp, label="FP")
        ax.bar(images, fn, bottom=tp + fp, label="FN")

        ax.set_xlabel("Image")
        ax.set_ylabel("Count")
        ax.set_title(f"{clf}: TP / FP / FN per image")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        save_figure(fig, os.path.join(PUBLICATION_DIR, f"{clf}_tp_fp_fn_stacked"))

    fig, axes = plt.subplots(1, len(classifiers), figsize=(5.8 * len(classifiers), 4.8), sharey=True)

    if len(classifiers) == 1:
        axes = [axes]

    for ax, clf in zip(axes, classifiers):
        subset = df[df["classifier"] == clf].set_index("image").reindex(images)
        tp = subset["tp"].values
        fp = subset["fp"].values
        fn = subset["fn"].values

        ax.bar(images, tp, label="TP")
        ax.bar(images, fp, bottom=tp, label="FP")
        ax.bar(images, fn, bottom=tp + fp, label="FN")
        ax.set_title(clf)
        ax.set_xlabel("Image")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Count")
    axes[-1].legend()
    fig.suptitle("TP / FP / FN distribution by classifier")
    save_figure(fig, os.path.join(PUBLICATION_DIR, "tp_fp_fn_distribution_by_classifier"))


def plot_f1_per_classifier_separate(df):
    """
    Create one separate publication-ready F1 figure per classifier,
    with Image 1 ... Image N on the x-axis.
    """

    classifiers = sorted(df["classifier"].unique())
    images = sorted(df["image"].unique(), key=lambda x: int(x.split()[-1]))

    for clf in classifiers:
        subset = (
            df[df["classifier"] == clf]
            .set_index("image")
            .reindex(images)
            .reset_index()
        )

        y = to_percent(subset["f1"].values)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(subset["image"], y)

        ax.set_ylim(0, 100)
        ax.set_xlabel("Image")
        ax.set_ylabel("F1 score (%)")
        ax.set_title(f"{clf}: F1 score across validation images")
        ax.grid(axis="y", alpha=0.25)

        save_figure(
            fig,
            os.path.join(PUBLICATION_DIR, f"f1_per_image_{clf}")
        )

def plot_precision_recall_scatter(df):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    for clf in sorted(df["classifier"].unique()):
        subset = df[df["classifier"] == clf].copy()
        x = to_percent(subset["recall"].values)
        y = to_percent(subset["precision"].values)

        ax.scatter(x, y, s=60, label=clf)

        for (_, row), xp, yp in zip(subset.iterrows(), x, y):
            ax.annotate(row["image"], (xp, yp), fontsize=8, alpha=0.8)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision vs Recall by classifier and image")
    ax.legend()
    ax.grid(True, alpha=0.25)

    save_figure(fig, os.path.join(PUBLICATION_DIR, "precision_recall_scatter"))


def plot_iou_distribution(df_matches):
    if df_matches.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for clf in sorted(df_matches["classifier"].unique()):
        values = df_matches.loc[df_matches["classifier"] == clf, "iou"].values
        if len(values) > 0:
            ax.hist(values, bins=15, alpha=0.5, label=clf)

    ax.set_xlabel("IoU")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of IoU values for matched polygons")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, os.path.join(PUBLICATION_DIR, "iou_distribution"))


# ============================================================
# MAIN
# ============================================================

def main():
    pairs = load_geojson_pairs(GEOJSON_DIR)

    if not pairs:
        print(f"No valid GeoJSON pairs found in '{GEOJSON_DIR}'.")
        print("Expected filenames like: <prefix>_<classifier>_GT.geojson / <prefix>_<classifier>_Pred.geojson")
        return

    image_label_map = assign_image_labels(pairs)

    records = []
    match_records = []

    for (base, classifier), file_dict in sorted(pairs.items(), key=lambda x: (x[0][0], x[0][1])):
        if classifier not in CLASSIFIERS:
            continue

        if "GT" not in file_dict or "Pred" not in file_dict:
            print(f"Skipping incomplete pair: base={base}, classifier={classifier}")
            continue

        image_label = image_label_map[base]
        result = evaluate_pair(file_dict["GT"], file_dict["Pred"], iou_threshold=IOU_THRESHOLD)

        records.append(
            {
                "source_prefix": base,
                "image": image_label,
                "classifier": classifier,
                "gt_file": os.path.basename(file_dict["GT"]),
                "pred_file": os.path.basename(file_dict["Pred"]),
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )

        for match in result["matches"]:
            match_records.append(
                {
                    "source_prefix": base,
                    "image": image_label,
                    "classifier": classifier,
                    "gt_file": os.path.basename(file_dict["GT"]),
                    "pred_file": os.path.basename(file_dict["Pred"]),
                    "gt_index": match["gt_index"],
                    "pred_index": match["pred_index"],
                    "iou": match["iou"],
                }
            )

        safe_stub = f"{image_label.replace(' ', '_')}_{classifier}"

        plot_tp_fp_fn_overlay(
            result["gdf_gt"],
            result["gdf_pred"],
            result["matched_gt"],
            result["matched_pred"],
            title=f"{classifier} - {image_label} (TP / FP / FN)",
            outpath=os.path.join(OVERLAY_DIR, f"{safe_stub}_tp_fp_fn_overlay")
        )

        plot_gt_vs_pred(
            result["gdf_gt"],
            result["gdf_pred"],
            title=f"{classifier} - {image_label} (GT vs Pred)",
            outpath=os.path.join(GT_PRED_DIR, f"{safe_stub}_gt_vs_pred")
        )

    df = pd.DataFrame(records)
    df_matches = pd.DataFrame(match_records)

    if df.empty:
        print("No complete GT/Pred classifier pairs found.")
        return

    df["image_num"] = df["image"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["image_num", "classifier"]).drop(columns=["image_num"]).reset_index(drop=True)

    if not df_matches.empty:
        df_matches["image_num"] = df_matches["image"].str.extract(r"(\d+)").astype(int)
        df_matches = df_matches.sort_values(["image_num", "classifier"]).drop(columns=["image_num"]).reset_index(drop=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "results_summary_per_file.csv"), index=False)
    df.to_csv(os.path.join(CSV_DIR, "metrics_per_file.csv"), index=False)
    df_matches.to_csv(os.path.join(CSV_DIR, "matched_polygons.csv"), index=False)

    summary_rows = []
    for clf in sorted(df["classifier"].unique()):
        subset = df[df["classifier"] == clf].copy()

        precision_mean, precision_sem = mean_and_sem(subset["precision"])
        recall_mean, recall_sem = mean_and_sem(subset["recall"])
        f1_mean, f1_sem = mean_and_sem(subset["f1"])

        summary_rows.append(
            {
                "classifier": clf,
                "tp_mean": subset["tp"].mean(),
                "fp_mean": subset["fp"].mean(),
                "fn_mean": subset["fn"].mean(),
                "precision_mean": precision_mean,
                "precision_sem": precision_sem,
                "recall_mean": recall_mean,
                "recall_sem": recall_sem,
                "f1_mean": f1_mean,
                "f1_sem": f1_sem,
            }
        )

    classifier_summary = pd.DataFrame(summary_rows)
    classifier_summary.to_csv(os.path.join(CSV_DIR, "classifier_summary.csv"), index=False)

    plot_metric_lines(df)
    plot_grouped_metrics_by_classifier(df)
    plot_f1_per_classifier_separate(df)
    plot_tp_fp_fn_by_classifier(df)
    plot_precision_recall_scatter(df)
    plot_iou_distribution(df_matches)

    print(f"Evaluation complete. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
