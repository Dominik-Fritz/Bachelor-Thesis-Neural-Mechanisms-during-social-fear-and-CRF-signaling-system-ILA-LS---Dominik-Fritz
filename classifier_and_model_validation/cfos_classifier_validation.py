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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_DIR = os.path.join(SCRIPT_DIR, "geojson")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_output_cfos")
IOU_THRESHOLD = 0.5
CLASSIFIER = "cfos"

CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
OVERLAY_DIR = os.path.join(PLOT_DIR, "overlays_tp_fp_fn")
GT_PRED_DIR = os.path.join(PLOT_DIR, "overlays_gt_vs_pred")
METRIC_DIR = os.path.join(PLOT_DIR, "metrics_per_classifier")
PUBLICATION_DIR = os.path.join(PLOT_DIR, "publication")

# optional cellpose csv evaluation
CELLPOSE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_output_cellpose")
CELLPOSE_CSV_DIR = os.path.join(CELLPOSE_OUTPUT_DIR, "csv")
CELLPOSE_PLOT_DIR = os.path.join(CELLPOSE_OUTPUT_DIR, "plots")
CELLPOSE_METRIC_DIR = os.path.join(CELLPOSE_PLOT_DIR, "metrics_per_model")
CELLPOSE_PUBLICATION_DIR = os.path.join(CELLPOSE_PLOT_DIR, "publication")

# optional combined pipeline evaluation
PIPELINE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_output_pipeline")
PIPELINE_CSV_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "csv")
PIPELINE_PLOT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "plots")

for path in [
    OUTPUT_DIR, CSV_DIR, PLOT_DIR, OVERLAY_DIR, GT_PRED_DIR, METRIC_DIR, PUBLICATION_DIR,
    CELLPOSE_OUTPUT_DIR, CELLPOSE_CSV_DIR, CELLPOSE_PLOT_DIR, CELLPOSE_METRIC_DIR, CELLPOSE_PUBLICATION_DIR,
    PIPELINE_OUTPUT_DIR, PIPELINE_CSV_DIR, PIPELINE_PLOT_DIR,
]:
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


def _strip_known_suffix(filename_stem):
    return re.sub(r"_(gt|pred)$", "", filename_stem, flags=re.IGNORECASE)


def load_geojson_pairs(folder):
    files = glob.glob(os.path.join(folder, "*.geojson"))
    pairs = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        stem, _ = os.path.splitext(filename)
        if re.search(r"_gt$", stem, flags=re.IGNORECASE):
            base = _strip_known_suffix(stem)
            pairs.setdefault(base, {})["GT"] = filepath
        elif re.search(r"_pred$", stem, flags=re.IGNORECASE):
            base = _strip_known_suffix(stem)
            pairs.setdefault(base, {})["Pred"] = filepath
    return pairs


def assign_image_labels(pairs):
    unique_bases = sorted(pairs.keys())
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


def safe_sort_key(label):
    nums = re.findall(r"\d+", str(label))
    if nums:
        return tuple(int(n) for n in nums)
    return (10**9, str(label))


def compute_metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def find_optional_cellpose_csv(script_dir):
    preferred = os.path.join(script_dir, "evaluation_all_thresholds_progress.csv")
    if os.path.exists(preferred):
        return preferred

    candidates = []
    for pattern in ["*.csv", "*threshold*.csv", "*progress*.csv"]:
        candidates.extend(glob.glob(os.path.join(script_dir, pattern)))

    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)

    for path in seen:
        try:
            cols = pd.read_csv(path, nrows=2).columns.tolist()
        except Exception:
            continue
        required = {"image", "model", "TP_centroid", "FP_centroid", "FN_centroid"}
        if required.issubset(set(cols)):
            return path
    return None


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
            matches.append({"gt_index": i, "pred_index": best_j, "iou": best_iou})

    tp = len(matches)
    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp
    precision, recall, f1 = compute_metrics_from_counts(tp, fp, fn)

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
            gdf_gt.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2, aspect="auto")
    except Exception:
        if gdf_gt is not None and not gdf_gt.empty:
            for geom in gdf_gt.geometry:
                _plot_single_geometry(ax, geom, facecolor="none", edgecolor="black", alpha=1.0, linewidth=1.2)
    try:
        if gdf_pred is not None and not gdf_pred.empty:
            gdf_pred.plot(ax=ax, facecolor="none", edgecolor="tab:blue", linewidth=1.2, aspect="auto")
    except Exception:
        if gdf_pred is not None and not gdf_pred.empty:
            for geom in gdf_pred.geometry:
                _plot_single_geometry(ax, geom, facecolor="none", edgecolor="tab:blue", alpha=1.0, linewidth=1.2)
    _set_plot_limits(ax, [gdf_gt, gdf_pred])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    save_figure(fig, outpath)


def plot_metric_lines_generic(df, label_col, name_col, out_dir, entity_name):
    metrics = ["precision", "recall", "f1"]
    ordered_labels = sorted(df[label_col].unique(), key=safe_sort_key)
    subset = df.set_index(label_col).reindex(ordered_labels).reset_index()

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        y = to_percent(subset[metric].values)
        ax.plot(subset[label_col], y, marker="o", linewidth=1.8)
        ax.set_ylim(0, 100)
        ax.set_xlabel(label_col.capitalize())
        ax.set_ylabel(f"{metric.capitalize()} (%)")
        ax.set_title(f"{entity_name}: {metric.capitalize()} across validation images")
        ax.grid(True, alpha=0.25)
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, os.path.join(out_dir, f"{name_col}_{metric}"))


def plot_grouped_metrics_generic(df, label_col, name_col, out_dir, entity_name):
    metrics = ["precision", "recall", "f1"]
    ordered_labels = sorted(df[label_col].unique(), key=safe_sort_key)
    subset = df.set_index(label_col).reindex(ordered_labels)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        y = to_percent(subset[metric].values)
        ax.bar(ordered_labels, y)
        ax.set_ylim(0, 100)
        ax.set_xlabel(label_col.capitalize())
        ax.set_ylabel(f"{metric.capitalize()} (%)")
        ax.set_title(f"{entity_name}: {metric.capitalize()} per image")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, os.path.join(out_dir, f"{name_col}_{metric}_per_image"))


def plot_tp_fp_fn_generic(df, label_col, name_col, out_dir, entity_name):
    ordered_labels = sorted(df[label_col].unique(), key=safe_sort_key)
    subset = df.set_index(label_col).reindex(ordered_labels)
    tp = subset["tp"].values
    fp = subset["fp"].values
    fn = subset["fn"].values

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(ordered_labels, tp, label="TP")
    ax.bar(ordered_labels, fp, bottom=tp, label="FP")
    ax.bar(ordered_labels, fn, bottom=tp + fp, label="FN")
    ax.set_xlabel(label_col.capitalize())
    ax.set_ylabel("Count")
    ax.set_title(f"{entity_name}: TP / FP / FN per image")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, os.path.join(out_dir, f"{name_col}_tp_fp_fn_stacked"))


def plot_precision_recall_scatter_generic(df, label_col, name_col, out_dir, entity_name):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    x = to_percent(df["recall"].values)
    y = to_percent(df["precision"].values)
    ax.scatter(x, y, s=60, label=entity_name)
    for (_, row), xp, yp in zip(df.iterrows(), x, y):
        ax.annotate(str(row[label_col]), (xp, yp), fontsize=8, alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(f"Precision vs Recall ({entity_name})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_figure(fig, os.path.join(out_dir, f"{name_col}_precision_recall_scatter"))


def plot_iou_distribution(df_matches):
    if df_matches.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    values = df_matches["iou"].values
    if len(values) > 0:
        ax.hist(values, bins=15, alpha=0.6, label=CLASSIFIER)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of IoU values for matched polygons ({CLASSIFIER})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, os.path.join(PUBLICATION_DIR, "iou_distribution"))


def plot_pipeline_overview(df_summary, out_dir):
    order = ["cellpose", "cfos", "pipeline_combined"]
    subset = df_summary.set_index("stage").reindex(order).reset_index()

    x = np.arange(len(subset))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width, to_percent(subset["precision"].values), width=width, label="Precision")
    ax.bar(x, to_percent(subset["recall"].values), width=width, label="Recall")
    ax.bar(x + width, to_percent(subset["f1"].values), width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["stage"].values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Metric (%)")
    ax.set_title("Combined pipeline performance: cellpose -> cfos")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, os.path.join(out_dir, "pipeline_metric_comparison"))


def plot_pipeline_error_overview(df_summary, out_dir):
    subset = df_summary.copy()
    err = to_percent(subset["error_rate"].values)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.bar(subset["stage"], err)
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Error rates across workflow stages")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, os.path.join(out_dir, "pipeline_error_rates"))


# ============================================================
# OPTIONAL CELLPOSE CSV EVALUATION
# ============================================================
def evaluate_cellpose_csv(csv_path):
    raw = pd.read_csv(csv_path)
    required = {"image", "model", "TP_centroid", "FP_centroid", "FN_centroid"}
    if not required.issubset(set(raw.columns)):
        raise ValueError(f"CSV missing required columns: {sorted(required - set(raw.columns))}")

    # use only the custom / chosen segmentation model for the main plots
    # if not present, keep all rows but warn via print
    preferred_models = ["Custom_Model", "Custom", "Cellpose", "cellpose"]
    model_values = raw["model"].astype(str).unique().tolist()
    chosen_model = None
    for m in preferred_models:
        if m in model_values:
            chosen_model = m
            break
    if chosen_model is None:
        chosen_model = model_values[0]

    df = raw[raw["model"].astype(str) == str(chosen_model)].copy()
    df = df.reset_index(drop=True)

    df.rename(columns={"TP_centroid": "tp", "FP_centroid": "fp", "FN_centroid": "fn"}, inplace=True)
    metrics = df.apply(lambda r: compute_metrics_from_counts(r["tp"], r["fp"], r["fn"]), axis=1)
    df[["precision", "recall", "f1"]] = pd.DataFrame(metrics.tolist(), index=df.index)
    df["stage"] = "cellpose"
    df["source_csv"] = os.path.basename(csv_path)

    # save per-image metrics
    df.to_csv(os.path.join(CELLPOSE_CSV_DIR, "cellpose_metrics_per_file.csv"), index=False)

    # global counts summary
    tp_total = int(df["tp"].sum())
    fp_total = int(df["fp"].sum())
    fn_total = int(df["fn"].sum())
    precision, recall, f1 = compute_metrics_from_counts(tp_total, fp_total, fn_total)
    error_rate = 1.0 - recall

    summary = pd.DataFrame([{
        "stage": "cellpose",
        "model": chosen_model,
        "n_images": len(df),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": error_rate,
        "error_formula": "1 - recall",
        "csv_file": os.path.basename(csv_path),
    }])
    summary.to_csv(os.path.join(CELLPOSE_CSV_DIR, "cellpose_summary.csv"), index=False)

    # plots analogous to cfos metric figures
    plot_metric_lines_generic(df, "image", "cellpose", CELLPOSE_METRIC_DIR, "cellpose")
    plot_grouped_metrics_generic(df, "image", "cellpose", CELLPOSE_PUBLICATION_DIR, "cellpose")
    plot_tp_fp_fn_generic(df, "image", "cellpose", CELLPOSE_PUBLICATION_DIR, "cellpose")
    plot_precision_recall_scatter_generic(df, "image", "cellpose", CELLPOSE_PUBLICATION_DIR, "cellpose")

    return df, summary.iloc[0].to_dict()


# ============================================================
# CFOS GEOJSON EVALUATION
# ============================================================
def evaluate_cfos_geojson():
    pairs = load_geojson_pairs(GEOJSON_DIR)
    if not pairs:
        print(f"No valid GeoJSON pairs found in '{GEOJSON_DIR}'.")
        print("Expected filenames like: <prefix>_GT.geojson / <prefix>_pred.geojson")
        return None, None, None

    image_label_map = assign_image_labels(pairs)
    records = []
    match_records = []

    for base, file_dict in sorted(pairs.items(), key=lambda x: x[0]):
        if "GT" not in file_dict or "Pred" not in file_dict:
            print(f"Skipping incomplete pair: base={base}")
            continue

        image_label = image_label_map[base]
        result = evaluate_pair(file_dict["GT"], file_dict["Pred"], iou_threshold=IOU_THRESHOLD)

        records.append({
            "source_prefix": base,
            "image": image_label,
            "classifier": CLASSIFIER,
            "gt_file": os.path.basename(file_dict["GT"]),
            "pred_file": os.path.basename(file_dict["Pred"]),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        })

        for match in result["matches"]:
            match_records.append({
                "source_prefix": base,
                "image": image_label,
                "classifier": CLASSIFIER,
                "gt_file": os.path.basename(file_dict["GT"]),
                "pred_file": os.path.basename(file_dict["Pred"]),
                "gt_index": match["gt_index"],
                "pred_index": match["pred_index"],
                "iou": match["iou"],
            })

        safe_stub = f"{image_label.replace(' ', '_')}_{CLASSIFIER}"
        plot_tp_fp_fn_overlay(
            result["gdf_gt"], result["gdf_pred"], result["matched_gt"], result["matched_pred"],
            title=f"{CLASSIFIER} - {image_label} (TP / FP / FN)",
            outpath=os.path.join(OVERLAY_DIR, f"{safe_stub}_tp_fp_fn_overlay")
        )
        plot_gt_vs_pred(
            result["gdf_gt"], result["gdf_pred"],
            title=f"{CLASSIFIER} - {image_label} (GT vs Pred)",
            outpath=os.path.join(GT_PRED_DIR, f"{safe_stub}_gt_vs_pred")
        )

    df = pd.DataFrame(records)
    df_matches = pd.DataFrame(match_records)
    if df.empty:
        print("No complete GT/Pred pairs found.")
        return None, None, None

    df["image_num"] = df["image"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["image_num", "classifier"]).drop(columns=["image_num"]).reset_index(drop=True)

    if not df_matches.empty:
        df_matches["image_num"] = df_matches["image"].str.extract(r"(\d+)").astype(int)
        df_matches = df_matches.sort_values(["image_num", "classifier"]).drop(columns=["image_num"]).reset_index(drop=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "results_summary_per_file.csv"), index=False)
    df.to_csv(os.path.join(CSV_DIR, "metrics_per_file.csv"), index=False)
    df_matches.to_csv(os.path.join(CSV_DIR, "matched_polygons.csv"), index=False)

    precision_mean, precision_sem = mean_and_sem(df["precision"])
    recall_mean, recall_sem = mean_and_sem(df["recall"])
    f1_mean, f1_sem = mean_and_sem(df["f1"])

    tp_total = int(df["tp"].sum())
    fp_total = int(df["fp"].sum())
    fn_total = int(df["fn"].sum())
    precision_global, recall_global, f1_global = compute_metrics_from_counts(tp_total, fp_total, fn_total)

    classifier_summary = pd.DataFrame([{
        "stage": CLASSIFIER,
        "classifier": CLASSIFIER,
        "tp_mean": df["tp"].mean(),
        "fp_mean": df["fp"].mean(),
        "fn_mean": df["fn"].mean(),
        "precision_mean": precision_mean,
        "precision_sem": precision_sem,
        "recall_mean": recall_mean,
        "recall_sem": recall_sem,
        "f1_mean": f1_mean,
        "f1_sem": f1_sem,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "precision": precision_global,
        "recall": recall_global,
        "f1": f1_global,
        "error_rate": 1.0 - recall_global,
        "error_formula": "1 - recall",
    }])
    classifier_summary.to_csv(os.path.join(CSV_DIR, "classifier_summary.csv"), index=False)

    plot_metric_lines_generic(df, "image", CLASSIFIER, METRIC_DIR, CLASSIFIER)
    plot_grouped_metrics_generic(df, "image", CLASSIFIER, PUBLICATION_DIR, CLASSIFIER)
    plot_tp_fp_fn_generic(df, "image", CLASSIFIER, PUBLICATION_DIR, CLASSIFIER)
    plot_precision_recall_scatter_generic(df, "image", CLASSIFIER, PUBLICATION_DIR, CLASSIFIER)
    plot_iou_distribution(df_matches)

    return df, df_matches, classifier_summary.iloc[0].to_dict()


# ============================================================
# COMBINED PIPELINE EVALUATION
# ============================================================
def evaluate_combined_pipeline(cellpose_summary_row, cfos_summary_row):
    if cellpose_summary_row is None or cfos_summary_row is None:
        return None

    # Sequential workflow assumption like in physics for independent successive stages:
    # P_total = P1 * P2
    # R_total = R1 * R2
    # F1_total recomputed from combined P/R
    precision_combined = float(cellpose_summary_row["precision"]) * float(cfos_summary_row["precision"])
    recall_combined = float(cellpose_summary_row["recall"]) * float(cfos_summary_row["recall"])
    f1_combined = (2 * precision_combined * recall_combined / (precision_combined + recall_combined)) if (precision_combined + recall_combined) > 0 else 0.0
    error_rate = 1.0 - recall_combined

    summary = pd.DataFrame([
        {
            "stage": "cellpose",
            "precision": float(cellpose_summary_row["precision"]),
            "recall": float(cellpose_summary_row["recall"]),
            "f1": float(cellpose_summary_row["f1"]),
            "error_rate": float(cellpose_summary_row["error_rate"]),
            "formula": "metrics from TP_centroid / FP_centroid / FN_centroid",
        },
        {
            "stage": "cfos",
            "precision": float(cfos_summary_row["precision"]),
            "recall": float(cfos_summary_row["recall"]),
            "f1": float(cfos_summary_row["f1"]),
            "error_rate": float(cfos_summary_row["error_rate"]),
            "formula": "metrics from GeoJSON GT/Pred evaluation",
        },
        {
            "stage": "pipeline_combined",
            "precision": precision_combined,
            "recall": recall_combined,
            "f1": f1_combined,
            "error_rate": error_rate,
            "formula": "P_total = P_seg * P_cls; R_total = R_seg * R_cls; F1 recomputed from combined P/R; error = 1 - R_total",
        },
    ])

    summary.to_csv(os.path.join(PIPELINE_CSV_DIR, "pipeline_summary.csv"), index=False)
    plot_pipeline_overview(summary, PIPELINE_PLOT_DIR)
    plot_pipeline_error_overview(summary, PIPELINE_PLOT_DIR)
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    print("Starting cfos GeoJSON evaluation...")
    cfos_df, cfos_matches_df, cfos_summary_row = evaluate_cfos_geojson()

    csv_path = find_optional_cellpose_csv(SCRIPT_DIR)
    cellpose_df = None
    cellpose_summary_row = None

    if csv_path is not None:
        print(f"Detected optional cellpose CSV: {os.path.basename(csv_path)}")
        try:
            cellpose_df, cellpose_summary_row = evaluate_cellpose_csv(csv_path)
            print(f"Cellpose CSV evaluation complete. Results saved to: {CELLPOSE_OUTPUT_DIR}")
        except Exception as exc:
            print(f"Cellpose CSV found but could not be processed: {exc}")
    else:
        print("No optional cellpose CSV found in script directory.")

    if cellpose_summary_row is not None and cfos_summary_row is not None:
        evaluate_combined_pipeline(cellpose_summary_row, cfos_summary_row)
        print(f"Combined pipeline evaluation complete. Results saved to: {PIPELINE_OUTPUT_DIR}")
    else:
        print("Combined pipeline evaluation skipped (requires both cellpose CSV and cfos GeoJSON results).")

    print(f"cfos evaluation complete. Results saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
