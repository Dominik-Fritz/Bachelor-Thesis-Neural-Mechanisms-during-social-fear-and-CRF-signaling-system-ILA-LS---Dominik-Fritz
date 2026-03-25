
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def compute_iou_matrix(gt, pred):
    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)

    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = pred_labels[pred_labels != 0]

    iou_matrix = np.zeros((len(gt_labels), len(pred_labels)))

    for i, g in enumerate(gt_labels):
        gt_mask = gt == g
        for j, p in enumerate(pred_labels):
            pred_mask = pred == p
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix, gt_labels, pred_labels


def evaluate_iou(gt, pred):
    results = {}

    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(gt, pred)

    if iou_matrix.size == 0:
        for t in IOU_THRESHOLDS:
            results[f"TP_iou_{t}"] = 0
            results[f"FP_iou_{t}"] = len(pred_labels)
            results[f"FN_iou_{t}"] = len(gt_labels)
        return results

    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for threshold in IOU_THRESHOLDS:
        TP = 0
        matched_gt = set()
        matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= threshold:
                TP += 1
                matched_gt.add(r)
                matched_pred.add(c)

        FP = len(pred_labels) - len(matched_pred)
        FN = len(gt_labels) - len(matched_gt)

        results[f"TP_iou_{threshold}"] = TP
        results[f"FP_iou_{threshold}"] = FP
        results[f"FN_iou_{threshold}"] = FN

    return results


def evaluate_centroid(gt, pred):
    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)

    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = pred_labels[pred_labels != 0]

    gt_matched = set()
    TP = 0

    for p in pred_labels:
        pred_mask = pred == p
        if pred_mask.sum() == 0:
            continue

        centroid = ndimage.center_of_mass(pred_mask)
        cy, cx = int(round(centroid[0])), int(round(centroid[1]))

        if cy < 0 or cy >= gt.shape[0] or cx < 0 or cx >= gt.shape[1]:
            continue

        gt_label_at_centroid = gt[cy, cx]
        if gt_label_at_centroid != 0:
            gt_index = np.where(gt_labels == gt_label_at_centroid)[0]
            if len(gt_index) > 0 and gt_index[0] not in gt_matched:
                TP += 1
                gt_matched.add(gt_index[0])

    FP = len(pred_labels) - TP
    FN = len(gt_labels) - TP

    return {
        "TP_centroid": TP,
        "FP_centroid": FP,
        "FN_centroid": FN
    }


def process_single_image(args):
    gt_path, pred_path, model_name = args

    gt = imread(gt_path)
    pred = imread(pred_path)

    image_results = {
        "image": os.path.basename(gt_path),
        "model": model_name
    }

    image_results.update(evaluate_iou(gt, pred))
    image_results.update(evaluate_centroid(gt, pred))

    return image_results


def evaluate_folder_with_progress(gt_dir, pred_dir, model_name):
    tasks = []
    for filename in os.listdir(gt_dir):
        if not filename.endswith("_masks.png"):
            continue

        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if os.path.exists(pred_path):
            tasks.append((gt_path, pred_path, model_name))

    results = []
    n_cores = max(1, multiprocessing.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]

        for f in tqdm(as_completed(futures),
                      total=len(futures),
                      desc=f"Evaluating {model_name}",
                      unit="images"):
            try:
                res = f.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing image: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Starting evaluation with progress bar...")
    print(f"Using {multiprocessing.cpu_count()-1} CPU cores\n")

    df_custom = evaluate_folder_with_progress("GT", "custom_pred", "Custom_Model")
    df_cyto = evaluate_folder_with_progress("GT", "cyto_pred", "Cyto_Model")

    final_df = pd.concat([df_custom, df_cyto], ignore_index=True)
    final_df.to_csv("evaluation_all_thresholds_progress.csv", index=False)

    print("\nEvaluation complete.")
    print("Results saved to evaluation_all_thresholds_progress.csv")
