#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================================
# Whole-brain cFos analysis pipeline (ABBA / QuPath export)
#
# INPUT:
#   - Folder-per-group, CSV-per-animal exports from QuPath.
#   - Each CSV row represents (slice × region) with counts:
#       * n_dapi  = number of nuclei in that region (exposure)
#       * n_cfos  = number of cFos+ cells in that region (response)
#
# OUTPUT:
#   - A timestamped results folder containing QC tables, statistics, network
#     connectivity results, figures, an Excel workbook, and a text report.
#
# PHILOSOPHY:
#   - Fully automated & reproducible (no interactive stopping).
#   - Animal-level inference: brains/animals are the independent samples.
#   - QC is robust and transparent: slices are flagged (MAD), and optionally
#     excluded only in a secondary "stability" run.
# ======================================================================


import math
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import warnings
import sys
import re
import os

# tqdm provides a terminal progress bar with ETA. If it's not installed, the code
# will still run without progress bars.
try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    _TQDM_AVAILABLE = False
    def tqdm(it, **kwargs):
        return it

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.efficiency_measures import global_efficiency

import yaml

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


# -------------------------
# Helpers
# -------------------------

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (MAD).

    Robust scale estimator used for outlier detection.

    WHY MAD (instead of SD)?
    - SD/mean are sensitive to extreme slices (tears, staining failure, etc.).
    - MAD remains stable, so median ± k*MAD flags outliers reliably.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def log(msg: str, logfile: Optional[Path] = None) -> None:
    print(msg)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

def setup_warning_logging(warn_log_path: Path) -> None:
    """Redirect Python warnings to a dedicated log file (no console spam)."""
    warn_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] {category.__name__}: {message} ({filename}:{lineno})\n"
        with warn_log_path.open("a", encoding="utf-8") as f:
            f.write(msg)

    warnings.showwarning = _showwarning
    warnings.simplefilter("once")


def progress_log(i: int, total: int, t0: float, label: str, logfile: Path, every: int = 50) -> None:
    """Write progress + ETA updates to run.log (robust even when tqdm is not visible)."""
    if total <= 0:
        return
    if i == 1 or i % every == 0 or i == total:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else float("nan")
        remaining = (total - i) / rate if rate and rate > 0 else float("nan")
        pct = 100.0 * i / total
        log(
            f"[INFO] {label}: {i}/{total} ({pct:.1f}%) | elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min",
            logfile
        )


class PhaseTimer:
    """Log start/end and elapsed time for pipeline phases."""
    def __init__(self, name: str, logfile: Path):
        self.name = name
        self.logfile = logfile
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        log(f"[INFO] >>> START: {self.name}", self.logfile)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - (self.t0 or time.time())
        log(f"[INFO] <<< END:   {self.name} (elapsed {dt/60:.1f} min)", self.logfile)
        return False

def read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_acronym(a: str) -> str:
    if a is None:
        return ""
    return str(a).strip().upper()

def natural_sort_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    out = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out

def short_mouse_label(brain_id: str, fallback_idx: int) -> str:
    m = re.search(r"(\d+)", str(brain_id))
    if m:
        return f"M{int(m.group(1))}"
    return f"M{fallback_idx}"

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))

def eta_squared_from_anova(F: float, df_term: float, df_resid: float) -> float:
    if not np.isfinite(F) or not np.isfinite(df_term) or not np.isfinite(df_resid):
        return np.nan
    denom = F * df_term + df_resid
    if denom <= 0:
        return np.nan
    return float((F * df_term) / denom)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return q
    _, qv, _, _ = multipletests(pvals[mask], method="fdr_bh")
    q[mask] = qv
    return q


def _mean_sem(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    mean = float(np.nanmean(x))
    if x.size < 2:
        return mean, np.nan
    sem = float(np.nanstd(x, ddof=1) / np.sqrt(x.size))
    return mean, sem

def sanitize_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def comparison_label(groups: Iterable[str]) -> str:
    return "__".join(sanitize_name(g) for g in groups)


def all_group_pairs(groups: Iterable[str]) -> List[Tuple[str, str]]:
    gs = [str(g) for g in groups]
    return list(itertools.combinations(gs, 2))


def resolve_n_jobs(n_jobs_cfg: Optional[int], logfile: Optional[Path] = None) -> int:
    cpu_total = max(1, (os.cpu_count() or 1))
    if n_jobs_cfg is None:
        return 1
    try:
        n_jobs = int(n_jobs_cfg)
    except Exception:
        log(f"[WARN] Invalid n_jobs='{n_jobs_cfg}', falling back to 1", logfile)
        return 1
    if n_jobs in (0, -1):
        return max(1, cpu_total - 1)
    if n_jobs < -1:
        resolved = cpu_total + n_jobs + 1
        return max(1, resolved)
    return max(1, min(n_jobs, cpu_total))


def run_parallel_job_dicts(job_name: str, jobs: List[dict], worker_fn, n_jobs: int, logfile: Path) -> List[pd.DataFrame]:
    if not jobs:
        return []
    if n_jobs <= 1 or len(jobs) == 1:
        out = []
        for job in tqdm(jobs, desc=job_name, unit='job', file=sys.stdout, dynamic_ncols=True, leave=False):
            try:
                res = worker_fn(job)
                if isinstance(res, pd.DataFrame) and (not res.empty):
                    out.append(res)
            except Exception as e:
                label = job.get('job_label', '<job>')
                log(f"[WARN] {job_name} failed for {label}: {e}", logfile)
        return out

    out = []
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_job = {ex.submit(worker_fn, job): job for job in jobs}
            for fut in tqdm(as_completed(future_to_job), total=len(future_to_job), desc=job_name, unit='job', file=sys.stdout, dynamic_ncols=True, leave=False):
                job = future_to_job[fut]
                try:
                    res = fut.result()
                    if isinstance(res, pd.DataFrame) and (not res.empty):
                        out.append(res)
                except Exception as e:
                    label = job.get('job_label', '<job>')
                    log(f"[WARN] {job_name} failed for {label}: {e}", logfile)
    except Exception as e:
        log(f"[WARN] {job_name}: parallel execution unavailable ({e}); falling back to sequential.", logfile)
        return run_parallel_job_dicts(job_name, jobs, worker_fn, n_jobs=1, logfile=logfile)
    return out

def _apply_groupwise_bh(df: pd.DataFrame, p_col: str, group_cols: List[str], out_col: str = 'q') -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out
    if not group_cols:
        out[out_col] = benjamini_hochberg(out[p_col].to_numpy(dtype=float))
        return out
    grouped = out.groupby(group_cols, dropna=False, sort=False)
    for _, idx in grouped.groups.items():
        ix = list(idx)
        pvals = out.loc[ix, p_col].to_numpy(dtype=float)
        out.loc[ix, out_col] = benjamini_hochberg(pvals)
    return out

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _publication_target(outpath: Path) -> Optional[Path]:
    """Mirror a figure path into results_x/publication/..., preserving subfolders below figures/."""
    parts = list(outpath.parts)
    if "figures" not in parts:
        return None
    idx = parts.index("figures")
    if idx == 0:
        return None
    return Path(*parts[:idx]) / "publication" / Path(*parts[idx + 1:])


def save_figure_outputs(outpath: Path, publication_dpi: int = 600) -> None:
    """Save the current matplotlib figure as PNG + SVG, and mirror both to publication/."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(outpath.with_suffix('.svg'), format='svg', bbox_inches='tight')

    pub_out = _publication_target(outpath)
    if pub_out is not None:
        pub_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pub_out, dpi=publication_dpi, bbox_inches='tight')
        plt.savefig(pub_out.with_suffix('.svg'), format='svg', bbox_inches='tight')


def zscore_rows(mat: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each brain (row) across regions.

    WHY:
    - Removes global per-animal shifts (e.g., overall higher cFos across all regions).
    - Makes connectivity (correlations) focus on *relative* regional patterns.
    """
    '''Z-score per brain (row) across regions (columns), ignoring NaNs.'''
    arr = mat.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(arr.shape[0]):
        x = arr[i, :]
        mask = np.isfinite(x)
        if mask.sum() < 2:
            continue
        mu = np.nanmean(x[mask])
        sd = np.nanstd(x[mask], ddof=1)
        if sd <= 0 or not np.isfinite(sd):
            continue
        out[i, mask] = (x[mask] - mu) / sd
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)

# -------------------------
# Atlas mapping
# -------------------------

@dataclass
class AtlasMapper:
    atlas_name: str
    atlas: BrainGlobeAtlas
    acr_to_id: Dict[str, int]
    id_to_acr: Dict[int, str]
    id_to_name: Dict[int, str]
    parent: Dict[int, Optional[int]]
    children: Dict[int, List[int]]

    @staticmethod
    def build(atlas_name: str) -> "AtlasMapper":
        atlas = BrainGlobeAtlas(atlas_name)
        st = atlas.structures_list

        if isinstance(st, dict) and "structures" in st:
            nodes = st["structures"]
        elif isinstance(st, dict) and "nodes" in st:
            nodes = st["nodes"]
        elif isinstance(st, list):
            nodes = st
        else:
            nodes = list(st)

        acr_to_id: Dict[str, int] = {}
        id_to_acr: Dict[int, str] = {}
        id_to_name: Dict[int, str] = {}
        parent: Dict[int, Optional[int]] = {}
        children: Dict[int, List[int]] = {}

        for node in nodes:
            sid = int(node["id"])
            acr = normalize_acronym(node.get("acronym", ""))
            nm = str(node.get("name", ""))
            pid = node.get("parent_structure_id", None)
            parent[sid] = int(pid) if pid is not None else None

            if acr:
                acr_to_id[acr] = sid
            id_to_acr[sid] = acr
            id_to_name[sid] = nm
            children.setdefault(sid, [])

        for sid, pid in parent.items():
            if pid is not None and pid in children:
                children[pid].append(sid)

        return AtlasMapper(
            atlas_name=atlas_name,
            atlas=atlas,
            acr_to_id=acr_to_id,
            id_to_acr=id_to_acr,
            id_to_name=id_to_name,
            parent=parent,
            children=children
        )

    def map_acronym(self, acr: str) -> Optional[int]:
        return self.acr_to_id.get(normalize_acronym(acr), None)

    def descendants(self, sid: int) -> List[int]:
        out: List[int] = []
        stack = [sid]
        while stack:
            cur = stack.pop()
            for ch in self.children.get(cur, []):
                out.append(ch)
                stack.append(ch)
        return out

# -------------------------
# Data ingestion
# -------------------------


# ----------------------------------------------------------------------
# DATA INGESTION
#
# The pipeline assumes the following directory layout:
#   ROOT/
#     naive/*.csv
#     S1-/*.csv
#     S1+/*.csv
#     S6-/*.csv
#     S6+unex/*.csv
#     S6+ex/*.csv
#     analyze_cfos.py
#     config.yaml
#
# Each CSV file = one animal/brain.
# Each row = one (slice_id, region_acr) pair with DAPI and cFos counts.
# ----------------------------------------------------------------------

def discover_group_csvs(base_dir: Path, csv_glob: str, group_order: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    group_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if group_order:
        ordered = [base_dir / g for g in group_order if (base_dir / g).is_dir()]
        rest = [p for p in sorted(group_dirs) if p not in ordered]
        group_dirs = ordered + rest
    else:
        group_dirs = sorted(group_dirs)

    for group_dir in group_dirs:
        group = group_dir.name
        for csv_path in sorted(group_dir.glob(csv_glob)):
            items.append((group, csv_path))
    return items

def load_all_data(items: List[Tuple[str, Path]], colmap: Dict[str, str], logfile: Path) -> pd.DataFrame:
    dfs = []
    required = list(colmap.keys())

    for group, csv_path in tqdm(items, desc="Load CSVs", unit="file", file=sys.stdout, dynamic_ncols=True, leave=True):
        brain_id = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            log(f"[ERROR] Could not read {csv_path}: {e}", logfile)
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            log(f"[ERROR] Missing columns in {csv_path}: {missing} (skipping file)", logfile)
            continue

        df = df.rename(columns=colmap).copy()
        df["group"] = group
        df["brain_id"] = brain_id

        df["region_acr"] = df["region_acr"].apply(normalize_acronym)
        df["slice_id"] = df["slice_id"].astype(str)

        for c in ["n_dapi", "n_cfos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        bad = df["n_dapi"].isna() | df["n_cfos"].isna()
        if bad.any():
            log(f"[WARN] {csv_path}: dropping {int(bad.sum())} rows with NaN counts", logfile)
            df = df.loc[~bad].copy()

        neg = (df["n_dapi"] < 0) | (df["n_cfos"] < 0)
        if neg.any():
            log(f"[WARN] {csv_path}: {int(neg.sum())} rows with negative counts (clipping to 0)", logfile)
            df.loc[df["n_dapi"] < 0, "n_dapi"] = 0
            df.loc[df["n_cfos"] < 0, "n_cfos"] = 0

        gt = df["n_cfos"] > df["n_dapi"]
        if gt.any():
            log(f"[WARN] {csv_path}: {int(gt.sum())} rows with n_cfos > n_dapi (clipping n_cfos to n_dapi)", logfile)
            df.loc[gt, "n_cfos"] = df.loc[gt, "n_dapi"]

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid CSVs loaded.")
    data = pd.concat(dfs, ignore_index=True)
    data["p_cfos"] = np.where(data["n_dapi"] > 0, data["n_cfos"] / data["n_dapi"], np.nan)
    return data

# -------------------------
# QC
# -------------------------


# ----------------------------------------------------------------------
# QUALITY CONTROL (QC)
#
# 1) Slice-level QC: identify outlier slices within each brain.
#    Uses robust criterion: median ± k*MAD on:
#      - total_dapi per slice (tissue amount / segmentation sanity)
#      - p_cfos per slice     (global activation / staining sanity)
#
# 2) Brain-level QC: summarize slice QC per animal.
#
# WHY QC as flags first (not immediate deletion)?
# - Primary analysis should reflect the raw pipeline output.
# - Secondary analysis (excluding flagged slices) tests stability/robustness.
# ----------------------------------------------------------------------

def qc_slice_level(data: pd.DataFrame, mad_k: float) -> pd.DataFrame:
    sl = (data.groupby(["brain_id", "group", "slice_id"])
            .agg(total_dapi=("n_dapi","sum"),
                 total_cfos=("n_cfos","sum"),
                 n_regions=("region_acr","count"))
            .reset_index())
    sl["p_cfos"] = np.where(sl["total_dapi"] > 0, sl["total_cfos"]/sl["total_dapi"], np.nan)

    sl["flag_low_dapi"] = False
    sl["flag_high_dapi"] = False
    sl["flag_low_p"] = False
    sl["flag_high_p"] = False

    for bid, sub in sl.groupby("brain_id"):
        d = sub["total_dapi"].to_numpy(dtype=float)
        p = sub["p_cfos"].to_numpy(dtype=float)

        d_med = np.nanmedian(d); d_mad = mad(d)
        p_med = np.nanmedian(p); p_mad = mad(p)

        if d_mad > 0:
            sl.loc[sub.index, "flag_low_dapi"] = d < (d_med - mad_k*d_mad)
            sl.loc[sub.index, "flag_high_dapi"] = d > (d_med + mad_k*d_mad)

        if p_mad > 0:
            sl.loc[sub.index, "flag_low_p"] = p < (p_med - mad_k*p_mad)
            sl.loc[sub.index, "flag_high_p"] = p > (p_med + mad_k*p_mad)

    sl["flag_any"] = sl[["flag_low_dapi","flag_high_dapi","flag_low_p","flag_high_p"]].any(axis=1)
    return sl

def qc_brain_level(slice_qc: pd.DataFrame) -> pd.DataFrame:
    br = (slice_qc.groupby(["brain_id","group"])
            .agg(slices=("slice_id","nunique"),
                 total_dapi=("total_dapi","sum"),
                 total_cfos=("total_cfos","sum"),
                 flagged_slices=("flag_any","sum"))
            .reset_index())
    br["p_cfos"] = np.where(br["total_dapi"] > 0, br["total_cfos"]/br["total_dapi"], np.nan)
    br["flagged_slice_fraction"] = br["flagged_slices"] / br["slices"]
    return br

def aggregate_brain_region(data: pd.DataFrame) -> pd.DataFrame:
    br = (data.groupby(["brain_id","group","region_acr"])
            .agg(n_dapi=("n_dapi","sum"),
                 n_cfos=("n_cfos","sum"),
                 n_slices=("slice_id","nunique"))
            .reset_index())
    br["p_cfos"] = np.where(br["n_dapi"] > 0, br["n_cfos"]/br["n_dapi"], np.nan)
    return br

def qc_region_coverage(brain_region: pd.DataFrame, min_dapi: int, coverage_frac_warn: float) -> pd.DataFrame:
    tmp = brain_region.copy()
    tmp["present"] = tmp["n_dapi"] >= min_dapi

    per_group = (tmp.groupby(["region_acr","group"])
                   .agg(group_size=("brain_id","nunique"),
                        animals_present=("present","sum"))
                   .reset_index())
    per_group["coverage_group"] = per_group["animals_present"] / per_group["group_size"]
    per_group["min_needed_66pct"] = per_group["group_size"].apply(lambda n: int(math.ceil(coverage_frac_warn*n)))
    per_group["flag_low_coverage_66pct"] = per_group["animals_present"] < per_group["min_needed_66pct"]

    overall = (tmp.groupby("region_acr")
                 .agg(animals_total=("brain_id","nunique"),
                      animals_present=("present","sum"),
                      total_dapi=("n_dapi","sum"),
                      total_cfos=("n_cfos","sum"))
                 .reset_index())
    overall["coverage_overall"] = overall["animals_present"]/overall["animals_total"]
    overall["p_cfos_overall"] = np.where(overall["total_dapi"] > 0, overall["total_cfos"]/overall["total_dapi"], np.nan)

    return per_group.merge(overall, on="region_acr", how="left")

# -------------------------
# Stats: NB GLM with offset(log(n_dapi))
# -------------------------


# ----------------------------------------------------------------------
# DIFFERENTIAL ACTIVATION STATISTICS
#
# Model used per region (and per functional network aggregate):
#   n_cfos ~ group + offset(log(n_dapi))
#
# Interpretation:
# - This compares *rates* of cFos positivity while controlling for how many
#   nuclei were available (n_dapi).
#
# WHY Negative Binomial GLM?
# - cFos counts are integer counts and typically overdispersed (variance > mean).
# - NB is standard for biological count data when Poisson is too strict.
# ----------------------------------------------------------------------

def fit_nb_glm(sub: pd.DataFrame, group_col: str, y_col: str, exposure_col: str, control_group: str) -> pd.DataFrame:
    df = sub.copy()
    df = df[df[exposure_col] > 0].copy()
    if df[group_col].nunique() < 2:
        return pd.DataFrame()

    df[group_col] = df[group_col].astype("category")
    if control_group in df[group_col].cat.categories:
        cats = [control_group] + [c for c in df[group_col].cat.categories if c != control_group]
        df[group_col] = df[group_col].cat.reorder_categories(cats, ordered=True)

    y = df[y_col].astype(float).to_numpy()
    offset = np.log(df[exposure_col].astype(float).to_numpy())

    X = pd.get_dummies(df[group_col], drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        res = model.fit(maxiter=200, disp=False)
    except Exception:
        return pd.DataFrame()

    out = []
    for term in X.columns:
        if term == "const":
            continue
        coef = float(res.params[term])
        se = float(res.bse[term]) if term in res.bse else float("nan")
        p = float(res.pvalues[term]) if term in res.pvalues else float("nan")
        out.append({
            "term": term,
            "log_fc": coef,
            "fold_change": float(np.exp(coef)),
            "ci95_lo": float(np.exp(coef - 1.96*se)) if np.isfinite(se) else np.nan,
            "ci95_hi": float(np.exp(coef + 1.96*se)) if np.isfinite(se) else np.nan,
            "p": p,
            "n_animals": int(df["brain_id"].nunique()) if "brain_id" in df.columns else int(len(df)),
            "groups_present": int(df[group_col].nunique()),
            "median_exposure": float(np.median(df[exposure_col]))
        })
    return pd.DataFrame(out)

def run_region_stats(brain_region: pd.DataFrame, min_dapi: int, control_group: str, logfile: Path) -> pd.DataFrame:
    df = brain_region[brain_region["n_dapi"] >= min_dapi].copy()
    results = []
    groups = list(df.groupby("region_acr"))
    t0 = time.time()
    for i, (region, sub) in enumerate(
        tqdm(groups, desc="Region GLM (NB)", unit="region", file=sys.stdout, dynamic_ncols=True, leave=True),
        start=1
    ):
        progress_log(i, len(groups), t0, "Region GLM (NB)", logfile, every=50)
        r = fit_nb_glm(sub, "group", "n_cfos", "n_dapi", control_group)
        if r.empty:
            continue
        r.insert(0, "region_acr", region)
        results.append(r)

    if not results:
        log("[WARN] No region models could be fit (after min_dapi filter).", logfile)
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    out["q"] = np.nan
    for term, idx in out.groupby("term").groups.items():
        out.loc[idx, "q"] = benjamini_hochberg(out.loc[idx, "p"].to_numpy())
    return out

# -------------------------
# Networks
# -------------------------

def expand_network_regions(mapper: AtlasMapper, regions: List[str], include_desc: bool) -> List[str]:
    acrs = set()
    for r in regions:
        sid = mapper.map_acronym(r)
        if sid is None:
            acrs.add(normalize_acronym(r))
            continue
        acrs.add(mapper.id_to_acr.get(sid, normalize_acronym(r)))
        if include_desc:
            for dsid in mapper.descendants(sid):
                acr = mapper.id_to_acr.get(dsid, "")
                if acr:
                    acrs.add(normalize_acronym(acr))
    return sorted(acrs)

def compute_network_table(brain_region: pd.DataFrame, networks_cfg: dict, mapper: AtlasMapper) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    all_rows = []
    cov_rows = []
    expanded_map: Dict[str, List[str]] = {}

    present_regions = set(brain_region["region_acr"].unique())

    for net_name, net_def in networks_cfg.items():
        regs = net_def.get("regions", [])
        include_desc = bool(net_def.get("include_descendants", False))
        expanded = expand_network_regions(mapper, regs, include_desc)
        expanded_map[net_name] = expanded

        missing = [r for r in expanded if r not in present_regions]
        cov_rows.append({
            "network": net_name,
            "n_requested": len(regs),
            "n_expanded": len(expanded),
            "n_missing_in_dataset": len(missing),
            "missing_examples": ", ".join(missing[:50])
        })

        sub = brain_region[brain_region["region_acr"].isin(expanded)].copy()
        agg = (sub.groupby(["brain_id","group"])
                 .agg(n_dapi=("n_dapi","sum"),
                      n_cfos=("n_cfos","sum"))
                 .reset_index())
        agg["network"] = net_name
        agg["p_cfos"] = np.where(agg["n_dapi"] > 0, agg["n_cfos"]/agg["n_dapi"], np.nan)
        all_rows.append(agg)

    net_table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    net_cov = pd.DataFrame(cov_rows)
    return net_table, net_cov, expanded_map

def run_network_stats(net_table: pd.DataFrame, min_dapi: int, control_group: str, logfile: Path) -> pd.DataFrame:
    df = net_table[net_table["n_dapi"] >= min_dapi].copy()
    results = []
    groups = list(df.groupby("network"))
    t0 = time.time()
    for i, (net, sub) in enumerate(
        tqdm(groups, desc="Network GLM (NB)", unit="network", file=sys.stdout, dynamic_ncols=True, leave=True),
        start=1
    ):
        progress_log(i, len(groups), t0, "Network GLM (NB)", logfile, every=1)
        r = fit_nb_glm(sub, "group", "n_cfos", "n_dapi", control_group)
        if r.empty:
            continue
        r.insert(0, "network", net)
        results.append(r)

    if not results:
        log("[WARN] No network models could be fit.", logfile)
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    out["q"] = np.nan
    for term, idx in out.groupby("term").groups.items():
        out.loc[idx, "q"] = benjamini_hochberg(out.loc[idx, "p"].to_numpy())
    return out

# -------------------------
# Stats: One-way ANOVA + post-hoc Welch + BH-FDR HSD (all-pairs)
# -------------------------

def _compute_log_rate(n_cfos: pd.Series, n_dapi: pd.Series) -> pd.Series:
    """Stabilized log-rate used across the pipeline."""
    return np.log((n_cfos.astype(float) + 0.5) / (n_dapi.astype(float) + 1.0))

def run_anova_oneway(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    group_col: str,
    logfile: Path,
    min_total_n: int = 3,
) -> pd.DataFrame:
    """One-way ANOVA per feature (e.g., per region, per network).

    Returns one row per feature with F, p, dfs, group counts, means and variances.
    """
    rows = []
    groups_all = sorted(df[group_col].dropna().astype(str).unique().tolist()) if group_col in df.columns else []
    for feat, sub in df.groupby(feature_col):
        sub = sub[[feature_col, value_col, group_col]].dropna().copy()
        if sub[group_col].nunique() < 2 or len(sub) < min_total_n:
            continue

        try:
            res = smf.ols(f"{value_col} ~ C({group_col})", data=sub).fit()
            a = anova_lm(res, typ=2)
            if a.empty:
                continue
            term = f"C({group_col})"
            if term not in a.index:
                nonres = [ix for ix in a.index if str(ix).lower() != "residual"]
                if not nonres:
                    continue
                term = nonres[0]
            F = float(a.loc[term, "F"])
            p = float(a.loc[term, "PR(>F)"])
            df_term = float(a.loc[term, "df"]) if "df" in a.columns else float("nan")
            df_resid = float(a.loc["Residual", "df"]) if "Residual" in a.index else float("nan")
        except Exception as e:
            log(f"[WARN] ANOVA failed for {feature_col}='{feat}': {e}", logfile)
            continue

        counts = sub.groupby(group_col).size().to_dict()
        means = sub.groupby(group_col)[value_col].mean().to_dict()
        variances = sub.groupby(group_col)[value_col].var(ddof=1).to_dict()
        row = {
            feature_col: feat,
            "n_total": int(len(sub)),
            "n_groups": int(sub[group_col].nunique()),
            "df_term": df_term,
            "df_resid": df_resid,
            "F": F,
            "p": p,
            "eta_sq": eta_squared_from_anova(F, df_term, df_resid),
            "test": "oneway_anova",
            "group_counts": ";".join([f"{k}:{v}" for k, v in sorted(counts.items())]),
            "group_means": ";".join([f"{k}:{means.get(k, np.nan):.6g}" for k in sorted(counts)]),
            "group_variances": ";".join([f"{k}:{variances.get(k, np.nan):.6g}" for k in sorted(counts)]),
        }
        for g in groups_all:
            row[f"n_{g}"] = int(counts.get(g, 0))
            row[f"mean_{g}"] = float(means.get(g, np.nan)) if g in means else np.nan
            row[f"var_{g}"] = float(variances.get(g, np.nan)) if g in variances else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = benjamini_hochberg(out["p"].to_numpy(dtype=float))
    out["significant_q05"] = out["q"] <= 0.05
    return out


def run_posthoc_welch_fdr(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    group_col: str,
    logfile: Path,
    alpha: float = 0.05,
    min_n_per_group: int = 2,
) -> pd.DataFrame:
    """Post-hoc all-pairs Welch t-tests per feature, with BH-FDR across all tests."""
    rows = []
    for feat, sub in df.groupby(feature_col):
        sub = sub[[feature_col, value_col, group_col]].dropna().copy()
        groups = [g for g, s in sub.groupby(group_col) if len(s) >= int(min_n_per_group)]
        if len(groups) < 2:
            continue

        for g1, g2 in itertools.combinations(sorted(groups), 2):
            x = sub.loc[sub[group_col] == g1, value_col].astype(float).to_numpy()
            y = sub.loc[sub[group_col] == g2, value_col].astype(float).to_numpy()
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if len(x) < int(min_n_per_group) or len(y) < int(min_n_per_group):
                continue
            try:
                t_stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            except Exception as e:
                log(f"[WARN] Welch t-test failed for {feature_col}='{feat}', {g1} vs {g2}: {e}", logfile)
                continue
            vx = float(np.var(x, ddof=1)) if len(x) > 1 else np.nan
            vy = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
            var_ratio = np.nan
            if np.isfinite(vx) and np.isfinite(vy) and min(vx, vy) > 0:
                var_ratio = float(max(vx, vy) / min(vx, vy))
            rows.append({
                feature_col: feat,
                "group1": str(g1),
                "group2": str(g2),
                "n1": int(len(x)),
                "n2": int(len(y)),
                "mean1": float(np.nanmean(x)) if len(x) else float("nan"),
                "mean2": float(np.nanmean(y)) if len(y) else float("nan"),
                "variance1": vx,
                "variance2": vy,
                "variance_ratio": var_ratio,
                "median1": float(np.nanmedian(x)) if len(x) else float("nan"),
                "median2": float(np.nanmedian(y)) if len(y) else float("nan"),
                "meandiff": float(np.nanmean(x) - np.nanmean(y)) if (len(x) and len(y)) else float("nan"),
                "cohen_d": cohen_d(x, y),
                "test": "welch_t",
                "t": float(t_stat) if np.isfinite(t_stat) else float("nan"),
                "p": float(p) if np.isfinite(p) else float("nan"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    try:
        _, q, _, _ = multipletests(out["p"].to_numpy(dtype=float), alpha=float(alpha), method="fdr_bh")
        out["q"] = q
        out["reject_fdr"] = out["q"] <= float(alpha)
    except Exception as e:
        log(f"[WARN] FDR correction failed for post-hoc tests: {e}", logfile)
        out["q"] = np.nan
        out["reject_fdr"] = False

    out["alpha_fdr"] = float(alpha)
    out["significant_q05"] = out["q"] <= 0.05
    return out


# -------------------------
# Focused biological comparisons
# -------------------------

def _subset_groups(df: pd.DataFrame, groups_keep: List[str], group_col: str = "group") -> pd.DataFrame:
    out = df[df[group_col].isin(groups_keep)].copy()
    out[group_col] = pd.Categorical(out[group_col], categories=groups_keep, ordered=True)
    return out

def run_targeted_ttest(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    logfile: Path,
    min_n_per_group: int = 2,
) -> pd.DataFrame:
    rows = []
    subdf = _subset_groups(df, [group_a, group_b], group_col=group_col)
    groups = list(subdf.groupby(feature_col))
    t0 = time.time()
    for i, (feat, sub) in enumerate(groups, start=1):
        progress_log(i, len(groups), t0, f"Focused t-test {group_a} vs {group_b} ({feature_col})", logfile, every=25)
        sub = sub[[feature_col, value_col, group_col]].dropna().copy()
        x = sub.loc[sub[group_col] == group_a, value_col].astype(float).to_numpy()
        y = sub.loc[sub[group_col] == group_b, value_col].astype(float).to_numpy()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) < min_n_per_group or len(y) < min_n_per_group:
            continue
        try:
            t_stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        except Exception as e:
            log(f"[WARN] Focused Welch t-test failed for {feature_col}='{feat}', {group_a} vs {group_b}: {e}", logfile)
            continue
        vx = float(np.var(x, ddof=1)) if len(x) > 1 else np.nan
        vy = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
        var_ratio = np.nan
        if np.isfinite(vx) and np.isfinite(vy) and min(vx, vy) > 0:
            var_ratio = float(max(vx, vy) / min(vx, vy))
        rows.append({
            feature_col: feat,
            "comparison": f"{group_a}_vs_{group_b}",
            "group1": group_a,
            "group2": group_b,
            "n1": int(len(x)),
            "n2": int(len(y)),
            "mean1": float(np.nanmean(x)),
            "mean2": float(np.nanmean(y)),
            "median1": float(np.nanmedian(x)),
            "median2": float(np.nanmedian(y)),
            "variance1": vx,
            "variance2": vy,
            "variance_ratio": var_ratio,
            "diff_mean": float(np.nanmean(x) - np.nanmean(y)),
            "log2_fc_approx": float((np.nanmean(x) - np.nanmean(y)) / np.log(2.0)),
            "cohen_d": cohen_d(x, y),
            "test": "welch_t",
            "t": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = benjamini_hochberg(out["p"].to_numpy(dtype=float))
    out["significant_q05"] = out["q"] <= 0.05
    return out


def run_targeted_anova_and_posthoc(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    group_col: str,
    groups_keep: List[str],
    logfile: Path,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subdf = _subset_groups(df, groups_keep, group_col=group_col)
    aov = run_anova_oneway(subdf, feature_col, value_col, group_col, logfile)
    if not aov.empty:
        aov["comparison_set"] = "__".join(groups_keep)
        aov["eta_sq"] = [eta_squared_from_anova(f, dft, dfr) for f, dft, dfr in zip(aov["F"], aov["df_term"], aov["df_resid"])]
        aov["significant_q05"] = aov["q"] <= 0.05
    post = run_posthoc_welch_fdr(subdf, feature_col, value_col, group_col, logfile, alpha=alpha)
    if not post.empty:
        post["comparison_set"] = "__".join(groups_keep)
        post["cohen_d"] = np.nan
        for idx, row in post.iterrows():
            feat = row[feature_col]
            g1 = row["group1"]
            g2 = row["group2"]
            feat_sub = subdf[subdf[feature_col] == feat]
            x = feat_sub.loc[feat_sub[group_col] == g1, value_col].astype(float).to_numpy()
            y = feat_sub.loc[feat_sub[group_col] == g2, value_col].astype(float).to_numpy()
            post.loc[idx, "cohen_d"] = cohen_d(x, y)
    return aov, post

def save_hypothesis_network_summary_plot(net_df: pd.DataFrame, network_name: str, groups_keep: List[str], outpath: Path, title: str) -> None:
    if net_df.empty:
        return
    sub = net_df[(net_df["network"] == network_name) & (net_df["group"].isin(groups_keep))].copy()
    if sub.empty:
        return
    sub["group"] = pd.Categorical(sub["group"], categories=groups_keep, ordered=True)
    sub = sub.sort_values(["group", "brain_id"])
    fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * len(groups_keep) + 1.5), 4.8))
    xs = np.arange(len(groups_keep))
    vals_by_group = []
    for i, g in enumerate(groups_keep):
        vals = sub.loc[sub["group"] == g, "log_rate"].astype(float).to_numpy()
        vals_by_group.append(vals)
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=42, alpha=0.9, zorder=3)
            mean, sem = _mean_sem(vals)
            ax.errorbar(i, mean, yerr=sem, fmt='o', markersize=7, capsize=4, linewidth=1.2, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups_keep)
    ax.set_ylabel('Mean log((cFos+0.5)/(DAPI+1)) per brain')
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_figure_outputs(outpath)
    plt.close()

def save_hypothesis_region_panel(region_df: pd.DataFrame, stats_df: pd.DataFrame, network_name: str, groups_keep: List[str], outpath: Path, title: str) -> None:
    if region_df.empty:
        return
    sub = region_df[(region_df["network"] == network_name) & (region_df["group"].isin(groups_keep))].copy()
    if sub.empty:
        return
    pivot = (sub.groupby(["region_acr", "group"])['log_rate']
             .mean().unstack('group').reindex(columns=groups_keep))
    if pivot.empty:
        return
    order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]
    fig_h = max(4.0, 0.65 * len(order) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, 2.2 * len(groups_keep) + 5.5), fig_h), gridspec_kw={"width_ratios": [1.0, 1.15]})
    arr = pivot.to_numpy(dtype=float)
    vabs = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    vabs = max(vabs, 1e-3)
    im = axes[0].imshow(arr, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-vabs, vmax=vabs)
    axes[0].set_xticks(range(len(groups_keep)))
    axes[0].set_xticklabels(groups_keep)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].set_title('Group means by region')
    cb = plt.colorbar(im, ax=axes[0], fraction=0.05, pad=0.03)
    cb.set_label('Mean log-rate')

    if stats_df is not None and (not stats_df.empty):
        ss = stats_df[stats_df['network'] == network_name].copy()
        if 'diff_mean' in ss.columns:
            ss = ss.sort_values('diff_mean', key=lambda s: np.abs(s), ascending=False)
            y = np.arange(len(ss))
            axes[1].barh(y, ss['diff_mean'].to_numpy(dtype=float), alpha=0.85)
            axes[1].axvline(0, linewidth=0.8, alpha=0.5)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels(ss['region_acr'].tolist())
            axes[1].invert_yaxis()
            axes[1].set_xlabel(f'Mean difference ({groups_keep[0]} - {groups_keep[1]})')
            axes[1].set_title('Region-wise effect within network')
            qcol = 'q' if 'q' in ss.columns else None
            for yi, (_, row) in enumerate(ss.iterrows()):
                label = f"p={row['p']:.3g}"
                if qcol and np.isfinite(row[qcol]):
                    label += f", q={row[qcol]:.3g}"
                axes[1].text(float(row['diff_mean']) + (0.02 if row['diff_mean'] >= 0 else -0.02), yi, label, va='center', ha='left' if row['diff_mean'] >= 0 else 'right', fontsize=7)
        else:
            ss = ss.sort_values('p')
            y = np.arange(len(ss))
            axes[1].barh(y, -np.log10(np.clip(ss['p'].to_numpy(dtype=float), 1e-300, 1.0)), alpha=0.85)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels(ss['region_acr'].tolist())
            axes[1].invert_yaxis()
            axes[1].set_xlabel('-log10(p)')
            axes[1].set_title('Region-wise ANOVA within network')
    else:
        axes[1].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_figure_outputs(outpath)
    plt.close()

def run_focused_biological_comparisons(brain_region_all: pd.DataFrame, net_table: pd.DataFrame, expanded_map: Dict[str, List[str]], out_root: Path, cfg: dict, logfile: Path) -> None:
    alpha = float(cfg.get("stats", {}).get("alpha_fdr", 0.05))
    min_dapi = int(cfg.get("qc", {}).get("min_dapi_region", 0))

    region_df = brain_region_all[brain_region_all["n_dapi"] >= min_dapi].copy()
    region_df["log_rate"] = _compute_log_rate(region_df["n_cfos"], region_df["n_dapi"])
    net_df = net_table[net_table["n_dapi"] >= min_dapi].copy() if not net_table.empty else pd.DataFrame()
    if not net_df.empty:
        net_df["log_rate"] = _compute_log_rate(net_df["n_cfos"], net_df["n_dapi"])

    focused_dir = out_root / "stats" / "focused_biological_comparisons"
    fig_dir = out_root / "figures" / "focused_biological_comparisons"
    safe_mkdir(focused_dir)
    safe_mkdir(fig_dir)

    pairwise_sets = [("S1-", "S1+"), ("naive", "S1-"), ("naive", "S6-"), ("S1-", "S6-"), ("S6-", "S6+ex"), ("S6-", "S6+unex"), ("S6+ex", "S6+unex")]
    anova_sets = [["naive", "S1-", "S6-"], ["S6-", "S6+ex", "S6+unex"]]

    # Existing whole-brain focused outputs remain in place
    for g1, g2 in [("S1-", "S1+")]:
        if set([g1, g2]).issubset(set(region_df["group"].unique())):
            log(f"[INFO] Running focused region Welch t-tests for {g1} vs {g2}", logfile)
            save_df(run_targeted_ttest(region_df, "region_acr", "log_rate", "group", g1, g2, logfile), focused_dir / f"regions_ttest_{g1}_vs_{g2}.csv")
        if (not net_df.empty) and set([g1, g2]).issubset(set(net_df["group"].unique())):
            log(f"[INFO] Running focused network Welch t-tests for {g1} vs {g2}", logfile)
            save_df(run_targeted_ttest(net_df, "network", "log_rate", "group", g1, g2, logfile), focused_dir / f"networks_ttest_{g1}_vs_{g2}.csv")

    for trio in anova_sets:
        label = comparison_label(trio)
        if set(trio).issubset(set(region_df["group"].unique())):
            log(f"[INFO] Running focused region ANOVA for {label}", logfile)
            aov, post = run_targeted_anova_and_posthoc(region_df, "region_acr", "log_rate", "group", trio, logfile, alpha=alpha)
            save_df(aov, focused_dir / f"regions_anova_{label}.csv")
            save_df(post, focused_dir / f"regions_posthoc_{label}.csv")
        if (not net_df.empty) and set(trio).issubset(set(net_df["group"].unique())):
            log(f"[INFO] Running focused network ANOVA for {label}", logfile)
            aov, post = run_targeted_anova_and_posthoc(net_df, "network", "log_rate", "group", trio, logfile, alpha=alpha)
            save_df(aov, focused_dir / f"networks_anova_{label}.csv")
            save_df(post, focused_dir / f"networks_posthoc_{label}.csv")

    # Hypothesis-driven network-scoped analyses
    annotated_region_df = region_df.copy()
    net_rows = []
    for net_name, regs in expanded_map.items():
        regs_present = sorted(set(regs).intersection(set(annotated_region_df['region_acr'].unique())))
        if not regs_present:
            continue
        mask = annotated_region_df['region_acr'].isin(regs_present)
        tmp = annotated_region_df.loc[mask, ['brain_id', 'group', 'region_acr']].copy()
        tmp['network'] = net_name
        net_rows.append(tmp)
    if net_rows:
        net_annot = pd.concat(net_rows, ignore_index=True)
        annotated_region_df = annotated_region_df.merge(net_annot, on=['brain_id', 'group', 'region_acr'], how='left')
    else:
        annotated_region_df['network'] = np.nan

    # Pairwise comparisons: network level + region level within network with network-scoped BH
    for g1, g2 in pairwise_sets:
        if not set([g1, g2]).issubset(set(region_df['group'].unique())):
            continue
        comp = f"{g1}_vs_{g2}"
        network_level_rows = []
        for net_name, regs in expanded_map.items():
            regs_present = sorted(set(regs).intersection(set(region_df['region_acr'].unique())))
            if len(regs_present) < 1:
                continue
            reg_sub = annotated_region_df[(annotated_region_df['network'] == net_name) & (annotated_region_df['group'].isin([g1, g2]))].copy()
            if reg_sub.empty:
                continue
            reg_stats = run_targeted_ttest(reg_sub, 'region_acr', 'log_rate', 'group', g1, g2, logfile)
            if not reg_stats.empty:
                reg_stats.insert(0, 'network', net_name)
                reg_stats['n_regions_in_network'] = int(len(regs_present))
                reg_stats['fdr_scope'] = f'{net_name} regions only'
                save_df(reg_stats, focused_dir / f"hypothesis_regions_{sanitize_name(net_name)}_{comp}.csv")
                save_hypothesis_region_panel(reg_sub, reg_stats, net_name, [g1, g2], fig_dir / f"hypothesis_regions_{sanitize_name(net_name)}_{comp}.png", f"{net_name}: region-wise comparison ({g1} vs {g2})")

            if not net_df.empty:
                net_sub = net_df[(net_df['network'] == net_name) & (net_df['group'].isin([g1, g2]))].copy()
                if not net_sub.empty:
                    nres = run_targeted_ttest(net_sub, 'network', 'log_rate', 'group', g1, g2, logfile)
                    if not nres.empty:
                        row = nres.iloc[0].to_dict()
                        row['network'] = net_name
                        row['n_regions_in_network'] = int(len(regs_present))
                        row['fdr_scope'] = 'single network hypothesis'
                        row['q'] = row['p']
                        row['significant_q05'] = bool(row['q'] <= 0.05) if np.isfinite(row['q']) else False
                        network_level_rows.append(row)
                        save_hypothesis_network_summary_plot(net_sub, net_name, [g1, g2], fig_dir / f"hypothesis_network_{sanitize_name(net_name)}_{comp}.png", f"{net_name}: network-level comparison ({g1} vs {g2})")
        if network_level_rows:
            df_n = pd.DataFrame(network_level_rows).sort_values('p')
            save_df(df_n, focused_dir / f"hypothesis_network_level_{comp}.csv")

    # Multi-group comparisons with network-scoped region FDR + network-level ANOVA
    for trio in anova_sets:
        label = comparison_label(trio)
        if not set(trio).issubset(set(region_df['group'].unique())):
            continue
        network_level_rows = []
        network_posthoc_rows = []
        for net_name, regs in expanded_map.items():
            regs_present = sorted(set(regs).intersection(set(region_df['region_acr'].unique())))
            if len(regs_present) < 1:
                continue
            reg_sub = annotated_region_df[(annotated_region_df['network'] == net_name) & (annotated_region_df['group'].isin(trio))].copy()
            if reg_sub.empty:
                continue
            aov_reg, post_reg = run_targeted_anova_and_posthoc(reg_sub, 'region_acr', 'log_rate', 'group', trio, logfile, alpha=alpha)
            if not aov_reg.empty:
                aov_reg.insert(0, 'network', net_name)
                aov_reg['n_regions_in_network'] = int(len(regs_present))
                aov_reg['fdr_scope'] = f'{net_name} regions only'
                save_df(aov_reg, focused_dir / f"hypothesis_regions_anova_{sanitize_name(net_name)}_{label}.csv")
                save_hypothesis_region_panel(reg_sub, aov_reg, net_name, trio, fig_dir / f"hypothesis_regions_anova_{sanitize_name(net_name)}_{label}.png", f"{net_name}: region-wise ANOVA ({', '.join(trio)})")
            if not post_reg.empty:
                post_reg = _apply_groupwise_bh(post_reg, 'p', ['group1', 'group2'], out_col='q')
                post_reg['significant_q05'] = post_reg['q'] <= 0.05
                post_reg.insert(0, 'network', net_name)
                post_reg['n_regions_in_network'] = int(len(regs_present))
                post_reg['fdr_scope'] = f'{net_name} regions only per posthoc contrast'
                save_df(post_reg, focused_dir / f"hypothesis_regions_posthoc_{sanitize_name(net_name)}_{label}.csv")

            if not net_df.empty:
                net_sub = net_df[(net_df['network'] == net_name) & (net_df['group'].isin(trio))].copy()
                if not net_sub.empty:
                    aov_net, post_net = run_targeted_anova_and_posthoc(net_sub, 'network', 'log_rate', 'group', trio, logfile, alpha=alpha)
                    if not aov_net.empty:
                        row = aov_net.iloc[0].to_dict()
                        row['network'] = net_name
                        row['n_regions_in_network'] = int(len(regs_present))
                        row['fdr_scope'] = 'single network hypothesis'
                        row['q'] = row['p']
                        row['significant_q05'] = bool(row['q'] <= 0.05) if np.isfinite(row['q']) else False
                        network_level_rows.append(row)
                        save_hypothesis_network_summary_plot(net_sub, net_name, trio, fig_dir / f"hypothesis_network_anova_{sanitize_name(net_name)}_{label}.png", f"{net_name}: network-level ANOVA ({', '.join(trio)})")
                    if not post_net.empty:
                        post_net = _apply_groupwise_bh(post_net, 'p', ['group1', 'group2'], out_col='q')
                        post_net['significant_q05'] = post_net['q'] <= 0.05
                        post_net['network'] = net_name
                        post_net['n_regions_in_network'] = int(len(regs_present))
                        post_net['fdr_scope'] = 'single network hypothesis per posthoc contrast'
                        network_posthoc_rows.append(post_net)
        if network_level_rows:
            save_df(pd.DataFrame(network_level_rows).sort_values('p'), focused_dir / f"hypothesis_network_level_anova_{label}.csv")
        if network_posthoc_rows:
            save_df(pd.concat(network_posthoc_rows, ignore_index=True), focused_dir / f"hypothesis_network_level_posthoc_{label}.csv")

# -------------------------
# Connectivity & graphs
# -------------------------
# Connectivity & graphs
# -------------------------


# ----------------------------------------------------------------------
# CONNECTIVITY / CORRELATION NETWORKS
#
# Goal: infer "functional coupling" as co-variation of regional activity across animals.
#
# Step A: build brain × region activation matrix using a stabilized log-rate:
#   log_rate = log((n_cfos + 0.5) / (n_dapi + 1.0))
#   (pseudocounts avoid division by zero and stabilize variance)
#
# Step B: compute region-to-region Spearman correlations per group.
#   WHY Spearman?
#   - robust to non-normality and monotonic relationships.
#   - appropriate for small n.
#
# Step C: threshold edges using BH-FDR on p-values + |r| cutoff.
# ----------------------------------------------------------------------

def build_region_matrix(brain_region: pd.DataFrame, min_dapi: int) -> pd.DataFrame:
    df = brain_region[brain_region["n_dapi"] >= min_dapi].copy()
    df["log_rate"] = np.log((df["n_cfos"] + 0.5) / (df["n_dapi"] + 1.0))
    return df.pivot_table(index=["brain_id","group"], columns="region_acr", values="log_rate", aggfunc="mean")

def corr_matrix_for_group(mat: pd.DataFrame, group: str, min_brains: int) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if group not in mat.index.get_level_values("group"):
        return pd.DataFrame(), pd.DataFrame(), 0
    sub = mat.xs(group, level="group", drop_level=False)
    n_brains = sub.droplevel("group").shape[0]
    if n_brains < min_brains:
        return pd.DataFrame(), pd.DataFrame(), n_brains

    cols = sub.columns.to_list()
    n = len(cols)
    R = np.full((n,n), np.nan)
    P = np.full((n,n), np.nan)

    for i in tqdm(range(n), desc=f"Corr matrix ({group})", unit="region", file=sys.stdout, dynamic_ncols=True, leave=True):
        R[i,i] = 1.0
        P[i,i] = 0.0
        xi = sub.iloc[:, i].to_numpy()
        for j in range(i+1, n):
            xj = sub.iloc[:, j].to_numpy()
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < min_brains:
                continue
            r, p = spearmanr(xi[mask], xj[mask])
            R[i,j] = R[j,i] = r
            P[i,j] = P[j,i] = p

    return pd.DataFrame(R, index=cols, columns=cols), pd.DataFrame(P, index=cols, columns=cols), n_brains

def threshold_graph(R: pd.DataFrame, P: pd.DataFrame, alpha_fdr: float, min_abs_r: float, include_negative: bool) -> Tuple[nx.Graph, pd.DataFrame]:
    cols = R.columns
    n = len(cols)
    triu = np.triu_indices(n, k=1)
    pvals = P.to_numpy()[triu]
    rvals = R.to_numpy()[triu]

    mask = np.isfinite(pvals) & np.isfinite(rvals)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum() > 0:
        qvals[mask] = benjamini_hochberg(pvals[mask])

    edges = []
    for (i, j), r, p, q in zip(zip(triu[0], triu[1]), rvals, pvals, qvals):
        if not np.isfinite(r) or not np.isfinite(q):
            continue
        if abs(r) < min_abs_r:
            continue
        if q > alpha_fdr:
            continue
        if (not include_negative) and (r < 0):
            continue
        edges.append((cols[i], cols[j], float(r), float(p), float(q)))

    G = nx.Graph()
    G.add_nodes_from(cols)
    for u, v, r, p, q in edges:
        G.add_edge(u, v, weight=abs(r), r=r, p=p, q=q)

    return G, pd.DataFrame(edges, columns=["u","v","r","p","q"])

def graph_metrics(G: nx.Graph) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["n_nodes"] = G.number_of_nodes()
    m["n_edges"] = G.number_of_edges()
    m["density"] = nx.density(G)
    m["avg_degree"] = float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() > 0 else np.nan
    m["mean_abs_edge_r"] = float(np.mean([abs(d.get("r", np.nan)) for _, _, d in G.edges(data=True)])) if G.number_of_edges() > 0 else np.nan

    if G.number_of_edges() > 0:
        lcc = max(nx.connected_components(G), key=len)
        H = G.subgraph(lcc).copy()
    else:
        H = G.copy()

    m["lcc_nodes"] = H.number_of_nodes()
    m["lcc_edges"] = H.number_of_edges()

    try:
        m["global_efficiency_lcc"] = float(global_efficiency(H))
    except Exception:
        m["global_efficiency_lcc"] = np.nan
    try:
        m["avg_clustering_lcc"] = float(nx.average_clustering(H, weight="weight"))
    except Exception:
        m["avg_clustering_lcc"] = np.nan

    return m

def node_metrics(G: nx.Graph) -> pd.DataFrame:
    deg = dict(G.degree())
    try:
        btw = nx.betweenness_centrality(G, normalized=True, weight="weight")
    except Exception:
        btw = {n: np.nan for n in G.nodes()}
    try:
        cl = nx.clustering(G, weight="weight")
    except Exception:
        cl = {n: np.nan for n in G.nodes()}

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight") if G.number_of_edges() > 0 else {n: np.nan for n in G.nodes()}
    except Exception:
        eig = {n: np.nan for n in G.nodes()}
    strength = {n: float(sum(abs(G[n][nbr].get("r", 0.0)) for nbr in G.neighbors(n))) for n in G.nodes()}

    return pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [deg[n] for n in G.nodes()],
        "strength_abs_r": [strength[n] for n in G.nodes()],
        "betweenness": [btw[n] for n in G.nodes()],
        "clustering": [cl[n] for n in G.nodes()],
        "eigenvector": [eig[n] for n in G.nodes()]
    })

# -------------------------
# Figures
# -------------------------

def save_heatmap(df: pd.DataFrame, outpath: Path, title: str, cmap: str = "coolwarm", center_zero: bool = False, cbar_label: Optional[str] = None) -> None:
    if df.empty:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    label_step = 1
    note = ""
    if (plot_df.shape[0] == plot_df.shape[1]) and (plot_df.shape[0] > 120):
        score = plot_df.abs().mean(axis=1).sort_values(ascending=False)
        keep = score.head(120).index.tolist()
        plot_df = plot_df.loc[keep, keep]
        note = " (overview: top 120 regions by mean |r|)"
        label_step = max(1, len(keep) // 20)
    elif max(plot_df.shape) > 80:
        label_step = max(1, int(max(plot_df.shape) / 24))

    arr = plot_df.to_numpy(dtype=float)
    fig_w = max(8, min(18, 0.22 * max(plot_df.shape[1], 10)))
    fig_h = max(6, min(16, 0.22 * max(plot_df.shape[0], 10)))
    plt.figure(figsize=(fig_w, fig_h))
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin = vmax = None
    elif center_zero:
        vmax = float(np.nanpercentile(np.abs(finite), 98))
        vmax = max(vmax, 0.25)
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
    im = plt.imshow(arr, aspect='auto', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    xt = np.arange(plot_df.shape[1])
    yt = np.arange(plot_df.shape[0])
    plt.xticks(xt[::label_step], [plot_df.columns.tolist()[i] for i in xt[::label_step]], rotation=90, fontsize=7)
    plt.yticks(yt[::label_step], [plot_df.index.tolist()[i] for i in yt[::label_step]], fontsize=7)
    plt.xlabel('Region')
    plt.ylabel('Region')
    plt.title(title + note)
    plt.tight_layout()
    save_figure_outputs(outpath)
    plt.close()

def save_volcano(stats_df: pd.DataFrame, outpath: Path, title: str, use_q: bool = False, label_col: str = "region_acr", max_labels: int = 15) -> None:
    if stats_df.empty:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    pcol = "q" if use_q and "q" in stats_df.columns else "p"
    df = stats_df.copy()
    if "log_fc" not in df.columns and "log2_fc_approx" in df.columns:
        df["log_fc"] = df["log2_fc_approx"] * np.log(2.0)
    df = df[np.isfinite(df["log_fc"]) & np.isfinite(df[pcol])].copy()
    if df.empty:
        return

    df["log2_fc"] = df["log_fc"] / np.log(2.0)
    x = df["log2_fc"].to_numpy(dtype=float)
    y = -np.log10(np.clip(df[pcol].to_numpy(dtype=float), 1e-300, 1.0))
    sig = (df[pcol] <= 0.05).to_numpy()
    strong_fc = np.abs(x) >= np.log2(1.2)
    effect = np.abs(df["log2_fc"].to_numpy(dtype=float))

    plt.figure(figsize=(9.5, 7.5))
    sc = plt.scatter(x, y, c=effect, s=np.where(sig, 34, 20), alpha=0.85, cmap="viridis", edgecolors="none")
    cbar = plt.colorbar(sc, fraction=0.035, pad=0.02)
    cbar.set_label("Absolute log2 fold-change")
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    plt.axvline(np.log2(1.2), linestyle=":", linewidth=1)
    plt.axvline(-np.log2(1.2), linestyle=":", linewidth=1)
    plt.axvline(0.0, linestyle="-", linewidth=0.8, alpha=0.5)
    plt.xlabel("log2 fold-change")
    plt.ylabel(f"-log10({pcol})")
    plt.title(title)

    top = df.assign(_priority=(df[pcol].rank(method="first", ascending=True) + 0.25 / np.clip(np.abs(df["log2_fc"]), 1e-6, None)))
    top = top.sort_values([pcol, "log2_fc"], key=lambda s: np.abs(s) if s.name == "log2_fc" else s, ascending=[True, False]).head(max_labels)
    for _, row in top.iterrows():
        lbl = row[label_col] if label_col in row else row.get("network", "")
        if not lbl:
            continue
        xx = float(row["log2_fc"])
        yy = float(-np.log10(max(float(row[pcol]), 1e-300)))
        plt.text(xx, yy, str(lbl), fontsize=7, ha="left", va="bottom")

    n_sig = int(sig.sum())
    n_strong = int((sig & strong_fc).sum())
    plt.text(0.99, 0.01, f"sig ({pcol}≤0.05): {n_sig}\nstrong + sig: {n_strong}",
             transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6))
    plt.tight_layout()
    save_figure_outputs(outpath)
    plt.close()


def save_slice_density_plot(slice_qc: pd.DataFrame, outpath: Path, group_order: Optional[List[str]] = None) -> None:
    if slice_qc.empty:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    g = (slice_qc.groupby(["group", "brain_id"], as_index=False)["total_dapi"].median()
         .rename(columns={"total_dapi": "median_dapi_per_slice"}))
    if group_order:
        g["group"] = pd.Categorical(g["group"], categories=group_order, ordered=True)
    g = g.copy()
    group_rank = {grp: i for i, grp in enumerate(group_order)} if group_order else {}
    g["__group_order"] = g["group"].map(lambda x: group_rank.get(x, 999))

    def _extract_mouse_num(x):
        s = str(x)
        m = re.search(r"M(\d+)", s)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
        return 999999

    g["__brain_sort"] = g["brain_id"].astype(str)
    g["__mouse_num"] = g["__brain_sort"].map(_extract_mouse_num)
    g = g.sort_values(["__group_order", "__mouse_num", "__brain_sort"]).reset_index(drop=True)
    g["mouse_label"] = [short_mouse_label(bid, i + 1) for i, bid in enumerate(g["brain_id"].tolist())]
    g["x"] = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(max(10, len(g) * 0.38), 6.8))
    ax.scatter(g["x"], g["median_dapi_per_slice"], s=42, zorder=3)
    ax.plot(g["x"], g["median_dapi_per_slice"], linewidth=1.1, alpha=0.7, zorder=2)
    ax.set_xticks(g["x"])
    ax.set_xticklabels(g["mouse_label"], rotation=90)
    ax.set_xlabel("Mouse")
    ax.set_ylabel("Median DAPI+ nuclei per section")
    ax.set_title("QC: tissue yield per mouse")

    ymin = max(0.0, float(np.nanmin(g["median_dapi_per_slice"]) * 0.95))
    ymax = float(np.nanmax(g["median_dapi_per_slice"]) * 1.12) if len(g) else 1.0
    ax.set_ylim(ymin, ymax)
    ax.spines[["top", "right"]].set_visible(False)

    for grp_idx, (grp, sub) in enumerate(g.groupby("group", sort=False)):
        start = float(sub["x"].min()) - 0.45
        end = float(sub["x"].max()) + 0.45
        ax.axvspan(start, end, alpha=0.06 if grp_idx % 2 == 0 else 0.12, color="grey", zorder=0)
        mid = (start + end) / 2.0
        ax.text(mid, ymax, str(grp), ha="center", va="bottom", fontsize=9)
        if end < g["x"].max() + 0.45:
            ax.axvline(end, linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    save_figure_outputs(outpath)
    plt.close()


def save_activation_heatmap(brain_region: pd.DataFrame, outpath: Path, topN: int) -> None:
    if brain_region.empty:
        return
    df = brain_region.copy()
    df["log_rate"] = np.log((df["n_cfos"] + 0.5) / (df["n_dapi"] + 1.0))
    summary = (df.groupby("region_acr")
               .agg(total_dapi=("n_dapi", "sum"), variance=("log_rate", "var"), mean_rate=("log_rate", "mean"))
               .fillna(0.0))
    keep_n = min(max(int(topN), 10), 40)
    keep = summary.sort_values(["variance", "total_dapi", "mean_rate"], ascending=[False, False, False]).head(keep_n).index.tolist()
    pivot = (df[df["region_acr"].isin(keep)]
             .groupby(["group", "region_acr"])["log_rate"].mean().unstack("region_acr"))
    if pivot.empty:
        return
    pivot = pivot.loc[:, pivot.var(axis=0).sort_values(ascending=False).index]
    arr = pivot.to_numpy(dtype=float)
    col_mu = np.nanmean(arr, axis=0)
    col_sd = np.nanstd(arr, axis=0, ddof=1)
    col_sd[col_sd <= 0] = np.nan
    z = (arr - col_mu) / col_sd
    zdf = pd.DataFrame(z, index=pivot.index, columns=pivot.columns).clip(-2.5, 2.5)

    plt.figure(figsize=(max(8, 0.45 * zdf.shape[1]), max(4.5, 0.65 * zdf.shape[0] + 1.2)))
    im = plt.imshow(zdf.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("Relative activation across groups (z-score)")
    plt.xticks(range(zdf.shape[1]), zdf.columns.tolist(), rotation=90, fontsize=max(6, 11 - int(zdf.shape[1] / 10)))
    plt.yticks(range(zdf.shape[0]), zdf.index.tolist(), fontsize=10)
    plt.xlabel("Top variable regions")
    plt.ylabel("Group")
    plt.title(f"Activation heatmap: top {zdf.shape[1]} biologically variable regions")
    plt.tight_layout()
    save_figure_outputs(outpath)
    plt.close()


def save_graph_plot(edges_df: pd.DataFrame, outpath: Path, title: str, topK: int) -> None:
    if edges_df.empty:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = edges_df.copy()
    df['abs_r'] = df['r'].abs()
    sort_cols = [c for c in ['q', 'abs_r'] if c in df.columns]
    ascending = [True, False][:len(sort_cols)]
    df = df.sort_values(sort_cols, ascending=ascending) if sort_cols else df.sort_values('abs_r', ascending=False)
    df_metrics = df.head(max(topK, 25)).copy()
    df_plot = df.head(min(30, max(12, int(topK * 0.12)))).copy()
    df_plot['edge_rank'] = np.arange(1, len(df_plot) + 1)

    G = nx.Graph()
    for _, r in df_plot.iterrows():
        G.add_edge(r['u'], r['v'], weight=float(r['abs_r']), r=float(r['r']), q=float(r.get('q', np.nan)))
    degree = dict(G.degree())
    strength = {n: float(sum(abs(G[n][nbr].get('r', 0.0)) for nbr in G.neighbors(n))) for n in G.nodes()}
    df_metrics['degree_u'] = df_metrics['u'].map(degree).fillna(0)
    df_metrics['degree_v'] = df_metrics['v'].map(degree).fillna(0)
    df_metrics['strength_u'] = df_metrics['u'].map(strength).fillna(0.0)
    df_metrics['strength_v'] = df_metrics['v'].map(strength).fillna(0.0)
    metrics_path = outpath.with_suffix('').as_posix() + '_edge_metrics.csv'
    df_metrics.to_csv(metrics_path, index=False)

    fig = plt.figure(figsize=(14, 7.2))
    ax1 = fig.add_subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42, weight='weight', iterations=250)
    node_sizes = [80 + degree.get(n, 0) * 90 for n in G.nodes()]
    edge_widths = [1.0 + 5.0 * abs(G[u][v].get('r', 0.0)) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax1, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.35, width=edge_widths)
    hubs = sorted(G.nodes(), key=lambda n: (degree.get(n, 0), strength.get(n, 0.0)), reverse=True)[:min(15, len(G.nodes()))]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in hubs}, font_size=8, ax=ax1)
    ax1.set_title(f'Top {len(df_plot)} connectivity edges')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    rank_df = df_plot.head(min(20, len(df_plot))).copy()
    y = np.arange(len(rank_df))
    labels = [f"#{i+1} {u}–{v}" for i, (u, v) in enumerate(zip(rank_df['u'], rank_df['v']))]
    ax2.barh(y, rank_df['abs_r'].to_numpy(dtype=float), alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel('|Spearman r|')
    ax2.set_title('Top-ranked edges with metrics')
    for yi, (_, row) in enumerate(rank_df.iterrows()):
        txt = f"r={row['r']:.2f}"
        if 'q' in row.index and np.isfinite(row['q']):
            txt += f", q={row['q']:.3g}"
        ax2.text(float(row['abs_r']) + 0.01, yi, txt, va='center', fontsize=7)

    fig.suptitle(title)
    plt.tight_layout()
    save_figure_outputs(outpath)
    plt.close()

def save_pairwise_network_correlation_matrix(mat: pd.DataFrame, group_a: str, group_b: str, regions: List[str], min_brains: int, outpath: Path, title: str, logfile: Path) -> None:
    regs = [r for r in regions if r in mat.columns]
    if len(regs) < 3:
        log(f"[WARN] Pairwise network correlation heatmap skipped for {title}: <3 regions present.", logfile)
        return
    RA, PA, nA = corr_matrix_for_group(mat.loc[:, regs], group_a, min_brains=min_brains)
    RB, PB, nB = corr_matrix_for_group(mat.loc[:, regs], group_b, min_brains=min_brains)
    if RA.empty or RB.empty:
        log(f"[WARN] Pairwise network correlation heatmap skipped for {title}: insufficient brains ({group_a}={nA}, {group_b}={nB}).", logfile)
        return
    common = [r for r in regs if r in RA.index and r in RB.index]
    RA = RA.loc[common, common]
    RB = RB.loc[common, common]
    delta = RB - RA
    order_score = ((RA.abs().mean(axis=1) + RB.abs().mean(axis=1)) / 2.0).sort_values(ascending=False)
    order = order_score.index.tolist()
    RA = RA.loc[order, order]
    RB = RB.loc[order, order]
    delta = delta.loc[order, order]

    fig, axes = plt.subplots(1, 3, figsize=(max(11, 1.6 * len(order) + 5), max(4.8, 0.75 * len(order) + 1.5)))
    mats = [(RA, f'{group_a} (n={nA})', 'Spearman r'), (RB, f'{group_b} (n={nB})', 'Spearman r'), (delta, f'Δr ({group_b} - {group_a})', 'Δ Spearman r')]
    for ax, (cur, ttl, cbl) in zip(axes, mats):
        arr = cur.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = max(vmax, 0.25)
        im = ax.imshow(arr, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_title(ttl)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbl)
        if len(order) <= 8:
            thr = 0.45 if cbl == 'Δ Spearman r' else 0.5
            for i in range(len(order)):
                for j in range(len(order)):
                    val = arr[i, j]
                    if np.isfinite(val) and abs(val) >= thr:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_figure_outputs(outpath)
    plt.close()

def _perm_graph_metrics_worker(job: dict) -> pd.DataFrame:
    return permutation_graph_metrics(
        mat=job['mat'],
        group_a=job['group_a'],
        group_b=job['group_b'],
        n_perm=job['n_perm'],
        seed=job['seed'],
        min_brains=job['min_brains'],
        p_threshold=job['p_threshold'],
        min_abs_r=job['min_abs_r'],
        include_negative=job['include_negative'],
    )


def _perm_rewiring_worker(job: dict) -> pd.DataFrame:
    out = permutation_rewiring_pvals(
        mat=job['mat'],
        group_a=job['group_a'],
        group_b=job['group_b'],
        edges_to_test=job['edges_to_test'],
        n_perm=job['n_perm'],
        seed=job['seed'],
        min_brains=job['min_brains'],
    )
    if out.empty:
        return out
    for col in ['group_a', 'group_b']:
        out[col] = job[col]
    if 'network' in job:
        out['network'] = job['network']
    return out


def permutation_rewiring_pvals(mat: pd.DataFrame, group_a: str, group_b: str, edges_to_test: pd.DataFrame,
                               n_perm: int, seed: int, min_brains: int) -> pd.DataFrame:
    if edges_to_test.empty:
        return pd.DataFrame()
    A = mat.xs(group_a, level="group", drop_level=False).droplevel("group")
    B = mat.xs(group_b, level="group", drop_level=False).droplevel("group")
    nA, nB = A.shape[0], B.shape[0]
    if nA < min_brains or nB < min_brains:
        return pd.DataFrame()

    X = pd.concat([A, B], axis=0)
    labels = np.array([0]*nA + [1]*nB)
    rng = np.random.default_rng(seed)

    def corr_edge(group_df: pd.DataFrame, u: str, v: str) -> float:
        xu = group_df[u].to_numpy(dtype=float)
        xv = group_df[v].to_numpy(dtype=float)
        m = np.isfinite(xu) & np.isfinite(xv)
        if m.sum() < min_brains:
            return np.nan
        r, _ = spearmanr(xu[m], xv[m])
        return float(r)

    obs = []
    for _, row in edges_to_test.iterrows():
        u, v = row["u"], row["v"]
        obs.append(corr_edge(B, u, v) - corr_edge(A, u, v))
    obs = np.asarray(obs, dtype=float)

    more_extreme = np.zeros(len(obs), dtype=int)
    for _ in tqdm(range(n_perm), desc=f"Permute rewiring ({group_a}->{group_b})", unit="perm", file=sys.stdout, dynamic_ncols=True, leave=False):
        perm = rng.permutation(labels)
        A_p = X.iloc[perm == 0, :]
        B_p = X.iloc[perm == 1, :]
        deltas = []
        for _, row in edges_to_test.iterrows():
            u, v = row["u"], row["v"]
            deltas.append(corr_edge(B_p, u, v) - corr_edge(A_p, u, v))
        deltas = np.asarray(deltas, dtype=float)
        more_extreme += (np.abs(deltas) >= np.abs(obs)).astype(int)

    p = (more_extreme + 1) / (n_perm + 1)
    out = edges_to_test.copy()
    out["delta_r_obs"] = obs
    out["p_perm_two_sided"] = p
    return out

def permutation_graph_metrics(mat: pd.DataFrame, group_a: str, group_b: str, n_perm: int, seed: int,
                              min_brains: int, p_threshold: float, min_abs_r: float, include_negative: bool) -> pd.DataFrame:
    A = mat.xs(group_a, level="group", drop_level=False).droplevel("group")
    B = mat.xs(group_b, level="group", drop_level=False).droplevel("group")
    nA, nB = A.shape[0], B.shape[0]
    if nA < min_brains or nB < min_brains:
        return pd.DataFrame()

    X = pd.concat([A, B], axis=0)
    labels = np.array([0]*nA + [1]*nB)
    rng = np.random.default_rng(seed)

    regions = A.columns.tolist()
    n = len(regions)

    def build_graph(Gdf: pd.DataFrame) -> nx.Graph:
        cols = regions
        G = nx.Graph()
        G.add_nodes_from(cols)
        for i in range(n):
            xi = Gdf.iloc[:, i].to_numpy(dtype=float)
            for j in range(i+1, n):
                xj = Gdf.iloc[:, j].to_numpy(dtype=float)
                m = np.isfinite(xi) & np.isfinite(xj)
                if m.sum() < min_brains:
                    continue
                r, p = spearmanr(xi[m], xj[m])
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                if abs(r) < min_abs_r or p > p_threshold:
                    continue
                if (not include_negative) and (r < 0):
                    continue
                G.add_edge(cols[i], cols[j], weight=abs(float(r)), r=float(r), p=float(p))
        return G

    def metrics_vec(G: nx.Graph) -> np.ndarray:
        m = graph_metrics(G)
        return np.array([
            m.get("n_edges", np.nan),
            m.get("density", np.nan),
            m.get("lcc_nodes", np.nan),
            m.get("lcc_edges", np.nan),
            m.get("global_efficiency_lcc", np.nan),
            m.get("avg_clustering_lcc", np.nan),
        ], dtype=float)

    obs = metrics_vec(build_graph(B)) - metrics_vec(build_graph(A))

    more_extreme = np.zeros_like(obs, dtype=int)
    for _ in tqdm(range(n_perm), desc=f"Permute graph-metrics ({group_a}->{group_b})", unit="perm", file=sys.stdout, dynamic_ncols=True, leave=False):
        perm = rng.permutation(labels)
        A_p = X.iloc[perm == 0, :]
        B_p = X.iloc[perm == 1, :]
        d = metrics_vec(build_graph(B_p)) - metrics_vec(build_graph(A_p))
        more_extreme += (np.abs(d) >= np.abs(obs)).astype(int)

    p = (more_extreme + 1) / (n_perm + 1)

    cols = ["delta_n_edges","delta_density","delta_lcc_nodes","delta_lcc_edges","delta_global_eff_lcc","delta_avg_clust_lcc"]
    pcols = [c.replace("delta_","pperm_") for c in cols]
    row = {"group_a": group_a, "group_b": group_b, "n_perm": n_perm, "p_threshold": p_threshold, "min_abs_r": min_abs_r}
    row.update({c: float(v) for c, v in zip(cols, obs)})
    row.update({c: float(v) for c, v in zip(pcols, p)})
    return pd.DataFrame([row])

# -------------------------
# Main
# -------------------------

def main():
    base_dir = Path(".").resolve()
    cfg = read_config(base_dir / "config.yaml")

    # Create a fresh timestamped output directory so runs never overwrite each other.
    # This is essential for reproducibility and for comparing parameter variants.
    out_root = base_dir / f"results_{timestamp()}"
    for sub in ["qc","stats","networks","connectivity","figures","publication","excel","report"]:
        safe_mkdir(out_root / sub)

    logfile = out_root / "report" / "run.log"
    warnfile = out_root / "report" / "warnings.log"
    setup_warning_logging(warnfile)
    log(f"[INFO] Warning log: {warnfile}", logfile)
    log(f"[INFO] Base dir: {base_dir}", logfile)
    log(f"[INFO] Output dir: {out_root}", logfile)

    group_order = cfg.get("io", {}).get("group_order", None)
    # Discover all group CSVs. Group names come from folder names; each CSV = one brain.
    items = discover_group_csvs(base_dir, cfg["io"]["csv_glob"], group_order=group_order)
    if not items:
        raise RuntimeError("No group folders / CSVs found under base_dir.")
    log(f"[INFO] Found {len(items)} CSV files.", logfile)

    with PhaseTimer("Load CSVs", logfile):
        data = load_all_data(items, cfg["io"]["required_columns"], logfile)
        log(f"[INFO] Loaded rows: {len(data):,}", logfile)

    groups_found = sorted(data["group"].unique().tolist())
    if group_order:
        groups = [g for g in group_order if g in groups_found] + [g for g in groups_found if g not in group_order]
    else:
        groups = groups_found

    control_group = str(cfg["stats"]["control_group"])
    if control_group not in groups:
        log(f"[WARN] control_group='{control_group}' not found among groups: {groups}", logfile)

    atlas_name = cfg["atlas"]["name"]
    with PhaseTimer(f"Load atlas: {atlas_name}", logfile):
        log(f"[INFO] Loading atlas: {atlas_name}", logfile)
        mapper = AtlasMapper.build(atlas_name)

    unique_regions = sorted(set(data["region_acr"].unique()))
    unmapped = [r for r in unique_regions if mapper.map_acronym(r) is None]
    save_df(pd.DataFrame({"region_acr": unmapped}), out_root/"qc"/"unmatched_regions.csv")
    log(f"[INFO] Regions total={len(unique_regions)} unmapped={len(unmapped)}", logfile)

    # Slice-level QC flags within each brain (median ± k*MAD).
    # Brain-level QC summarizes flagged slice fractions per animal.
    log("[INFO] Running slice-level QC.", logfile)
    slice_qc = qc_slice_level(data, float(cfg["qc"]["slice_outlier_mad_k"]))
    save_df(slice_qc, out_root/"qc"/"qc_slices.csv")
    brain_qc = qc_brain_level(slice_qc)
    save_df(brain_qc, out_root/"qc"/"qc_brains.csv")
    save_slice_density_plot(slice_qc, out_root/"figures"/"qc_global_cell_density.png", group_order=groups)

    log("[INFO] Aggregating to brain × region table.", logfile)
    brain_region_all = aggregate_brain_region(data)
    save_df(brain_region_all, out_root/"qc"/"by_brain_region_all.csv")

    cov = qc_region_coverage(brain_region_all,
                             min_dapi=int(cfg["qc"]["min_dapi_region"]),
                             coverage_frac_warn=float(cfg["qc"]["coverage_fraction_warn"]))
    save_df(cov, out_root/"qc"/"qc_region_coverage.csv")

    save_activation_heatmap(brain_region_all,
                            out_root/"figures"/"activation_heatmap_top_regions.png",
                            int(cfg["figures"]["topN_regions_for_activation_heatmap"]))

    # REGION STATISTICS:
    # For each region, fit NB-GLM with exposure offset to estimate fold-changes vs control.
    # Then apply BH-FDR per contrast term across all regions.
    with PhaseTimer("Region statistics (NB-GLM)", logfile):
        region_stats_all = run_region_stats(brain_region_all,
                                        min_dapi=int(cfg["qc"]["min_dapi_region"]),
                                        control_group=control_group,
                                        logfile=logfile)
        save_df(region_stats_all, out_root/"stats"/"stats_regions_all.csv")

    # REGION: ANOVA (global group effect) + post-hoc all-pairs Welch t-tests + BH-FDR
    with PhaseTimer("Region statistics (ANOVA + post-hoc Welch + BH-FDR)", logfile):
        br_anova = brain_region_all[brain_region_all["n_dapi"] >= int(cfg["qc"]["min_dapi_region"])].copy()
        br_anova["log_rate"] = _compute_log_rate(br_anova["n_cfos"], br_anova["n_dapi"])

        region_anova = run_anova_oneway(
            br_anova,
            feature_col="region_acr",
            value_col="log_rate",
            group_col="group",
            logfile=logfile,
        )
        save_df(region_anova, out_root/"stats"/"anova_regions_lograte.csv")

        region_posthoc = run_posthoc_welch_fdr(
            br_anova,
            feature_col="region_acr",
            value_col="log_rate",
            group_col="group",
            logfile=logfile,
            alpha=float(cfg["stats"].get("alpha_fdr", 0.05)),
        )
        save_df(region_posthoc, out_root/"stats"/"posthoc_regions_welch_fdr.csv")

    if not region_stats_all.empty:
        for term in sorted(region_stats_all["term"].unique()):
            sub = region_stats_all[region_stats_all["term"] == term].copy()
            save_volcano(sub, out_root/"figures"/f"volcano_regions_{term}_p.png", f"Regions ({term}): p-values", use_q=False)
            save_volcano(sub, out_root/"figures"/f"volcano_regions_{term}_q.png", f"Regions ({term}): q-values", use_q=True)
        top50 = (region_stats_all.sort_values(["q","p"]).groupby("term").head(50))
        save_df(top50, out_root/"stats"/"top50_regions_per_term.csv")

    # Secondary run: exclude flagged slices
    region_stats_qc = pd.DataFrame()
    brain_region_qc = pd.DataFrame()
    stability_rows = []

    if bool(cfg["qc"]["drop_flagged_slices_for_secondary_run"]):
        flagged = slice_qc.loc[slice_qc["flag_any"], ["brain_id","slice_id"]].copy()
        if len(flagged) > 0:
            log(f"[INFO] Secondary run: excluding {len(flagged)} flagged brain×slice entries.", logfile)
            idx = pd.MultiIndex.from_frame(flagged)
            keep_mask = ~pd.MultiIndex.from_frame(data[["brain_id","slice_id"]]).isin(idx)
            data_qc = data.loc[keep_mask].copy()
        else:
            log("[INFO] Secondary run: no flagged slices, using same data.", logfile)
            data_qc = data.copy()

        brain_region_qc = aggregate_brain_region(data_qc)
        save_df(brain_region_qc, out_root/"qc"/"by_brain_region_qc_filtered.csv")

        region_stats_qc = run_region_stats(brain_region_qc,
                                           min_dapi=int(cfg["qc"]["min_dapi_region"]),
                                           control_group=control_group,
                                           logfile=logfile)
        save_df(region_stats_qc, out_root/"stats"/"stats_regions_qc_filtered.csv")

        alpha = float(cfg["stats"]["alpha_fdr"])
        if (not region_stats_all.empty) and (not region_stats_qc.empty):
            for term in sorted(set(region_stats_all["term"].unique()).intersection(region_stats_qc["term"].unique())):
                a = region_stats_all[region_stats_all["term"] == term]
                b = region_stats_qc[region_stats_qc["term"] == term]
                sig_a = set(a.loc[a["q"] <= alpha, "region_acr"])
                sig_b = set(b.loc[b["q"] <= alpha, "region_acr"])
                top_a = a.sort_values("p").head(50)["region_acr"].tolist()
                top_b = b.sort_values("p").head(50)["region_acr"].tolist()
                stability_rows.append({
                    "term": term,
                    "sig_all": len(sig_a),
                    "sig_qc": len(sig_b),
                    "sig_overlap": len(sig_a & sig_b),
                    "top50_overlap": len(set(top_a) & set(top_b)),
                })
            save_df(pd.DataFrame(stability_rows), out_root/"stats"/"stability_all_vs_qc_filtered.csv")

    # Networks
    with PhaseTimer("Functional network aggregation + stats", logfile):
        net_table, net_cov, expanded_map = compute_network_table(brain_region_all, cfg.get("networks", {}), mapper)
        save_df(net_table, out_root/"networks"/"network_table.csv")
        save_df(net_cov, out_root/"networks"/"network_coverage.csv")
        with (out_root/"networks"/"network_expanded_regions.json").open("w", encoding="utf-8") as f:
            f.write(pd.Series(expanded_map).to_json(indent=2))

        net_stats = run_network_stats(net_table,
                                  min_dapi=int(cfg["qc"]["min_dapi_region"]),
                                  control_group=control_group,
                                  logfile=logfile)
        save_df(net_stats, out_root/"networks"/"stats_networks.csv")

                # NETWORK: ANOVA (global group effect) + post-hoc all-pairs Welch t-tests + BH-FDR
        net_anova_df = net_table[net_table["n_dapi"] >= int(cfg["qc"]["min_dapi_region"])].copy()
        if not net_anova_df.empty:
            net_anova_df["log_rate"] = _compute_log_rate(net_anova_df["n_cfos"], net_anova_df["n_dapi"])

            net_anova = run_anova_oneway(
                net_anova_df,
                feature_col="network",
                value_col="log_rate",
                group_col="group",
                logfile=logfile,
            )
            save_df(net_anova, out_root/"networks"/"anova_networks_lograte.csv")

            net_posthoc = run_posthoc_welch_fdr(
                net_anova_df,
                feature_col="network",
                value_col="log_rate",
                group_col="group",
                logfile=logfile,
                alpha=float(cfg["stats"].get("alpha_fdr", 0.05)),
            )
            save_df(net_posthoc, out_root/"networks"/"posthoc_networks_welch_fdr.csv")

    with PhaseTimer("Focused biological subset comparisons", logfile):
        run_focused_biological_comparisons(brain_region_all, net_table, expanded_map, out_root, cfg, logfile)

    # CONNECTIVITY:
    # Build activation matrix (brain×region), optionally z-score per brain, then compute
    # group-wise Spearman correlation networks and graph metrics.
    # Connectivity matrix
    log("[INFO] Building brain × region activation matrix for connectivity.", logfile)
    mat = build_region_matrix(brain_region_all, min_dapi=int(cfg["qc"]["min_dapi_region"]))
    if bool(cfg["connectivity"].get("zscore_per_brain", False)):
        mat = zscore_rows(mat)
        mat.to_csv(out_root/"connectivity"/"region_matrix_zscored.csv")
    else:
        mat.to_csv(out_root/"connectivity"/"region_lograte_matrix.csv")

    min_brains = int(cfg["connectivity"]["min_brains_per_group"])
    edge_alpha = float(cfg["connectivity"]["edge_alpha_fdr"])
    min_abs_r = float(cfg["connectivity"]["min_abs_r"])
    include_neg = bool(cfg["connectivity"]["include_negative_edges"])
    topK_edges_plot = int(cfg["figures"]["topK_edges_for_graph_plot"])

    group_metrics = []
    node_metrics_all = []
    edges_by_group = {}
    R_by_group = {}

    for g in groups:
        R, P, n_brains = corr_matrix_for_group(mat, g, min_brains=min_brains)
        if R.empty:
            log(f"[WARN] Connectivity: skipping group '{g}' (brains={n_brains}, regions={mat.shape[1]}).", logfile)
            continue

        R.to_csv(out_root/"connectivity"/f"corr_R_{g}.csv")
        P.to_csv(out_root/"connectivity"/f"corr_P_{g}.csv")
        R_by_group[g] = R

        save_heatmap(R, out_root/"figures"/f"heatmap_corr_R_{g}.png", f"Whole-brain Spearman correlation matrix ({g})", cmap="coolwarm", center_zero=True, cbar_label="Spearman r")

        G, edges_df = threshold_graph(R, P, alpha_fdr=edge_alpha, min_abs_r=min_abs_r, include_negative=include_neg)
        save_df(edges_df, out_root/"connectivity"/f"edges_{g}.csv")
        edges_by_group[g] = edges_df

        gm = graph_metrics(G)
        gm["group"] = g
        gm["n_brains"] = n_brains
        gm["n_regions_matrix"] = int(R.shape[0])
        group_metrics.append(gm)

        nm = node_metrics(G)
        nm["group"] = g
        node_metrics_all.append(nm)

        save_graph_plot(edges_df, out_root/"figures"/f"graph_top_edges_{g}.png",
                        f"Connectivity graph (group={g}, top edges by |r|)", topK=topK_edges_plot)

        if bool(cfg["figures"]["plot_network_subgraphs"]) and expanded_map:
            for net_name, regs in expanded_map.items():
                regs_in = sorted(set(regs).intersection(R.index))
                if len(regs_in) < 5:
                    continue
                e_sub = edges_df[edges_df["u"].isin(regs_in) & edges_df["v"].isin(regs_in)].copy()
                if e_sub.empty:
                    continue
                save_graph_plot(e_sub, out_root/"figures"/f"graph_{g}_subgraph_{net_name}.png",
                                f"{net_name} subgraph (group={g})", topK=min(topK_edges_plot, len(e_sub)))

    with PhaseTimer("Pairwise network correlation matrices", logfile):
        pairwise_groups = [("S1-", "S1+"), ("naive", "S1-"), ("naive", "S6-"), ("S1-", "S6-"), ("S6-", "S6+ex"), ("S6-", "S6+unex"), ("S6+ex", "S6+unex")]
        for net_name, regs in expanded_map.items():
            for ga, gb in pairwise_groups:
                save_pairwise_network_correlation_matrix(mat, ga, gb, regs, min_brains, out_root/"figures"/f"paircorr_{net_name}_{ga}_vs_{gb}.png", f"{net_name}: pairwise network correlation comparison", logfile)

    if group_metrics:
        save_df(pd.DataFrame(group_metrics), out_root/"connectivity"/"graph_metrics_by_group.csv")
    if node_metrics_all:
        save_df(pd.concat(node_metrics_all, ignore_index=True), out_root/"connectivity"/"node_metrics_by_group.csv")

    # REWIRING:
    # ΔR matrices summarize how each pairwise correlation differs from control (group - control).
    # Rewiring ΔR vs control + permutation pvals
    if control_group in R_by_group:
        Rc = R_by_group[control_group]
        for g in groups:
            if g == control_group or g not in R_by_group:
                continue
            Rg = R_by_group[g]
            common = sorted(set(Rc.index).intersection(Rg.index))
            delta = (Rg.loc[common, common] - Rc.loc[common, common])
            delta.to_csv(out_root/"connectivity"/f"rewiring_deltaR_{g}_vs_{control_group}.csv")
            save_heatmap(delta, out_root/"figures"/f"heatmap_rewiring_deltaR_{g}_vs_{control_group}.png",
                         f"Rewiring matrix: Δ Spearman r ({g} - {control_group})", cmap="coolwarm", center_zero=True, cbar_label="Δ Spearman r")

    # PERMUTATIONS:
    # Empirical significance testing for rewiring edges and graph metrics using label shuffling.
    # Now runs for all available pairwise group comparisons, not only control-vs-treatment.
    perm_cfg = cfg.get("permutation", {})
    if bool(perm_cfg.get("enabled", False)):
        n_perm = int(perm_cfg.get("n_perm", 200))
        seed = int(perm_cfg.get("seed", 42))
        p_thr = float(perm_cfg.get("graph_metric_p_threshold", 0.01))
        edges_only = bool(perm_cfg.get("rewiring_edges_only", True))
        perm_n_jobs = resolve_n_jobs(perm_cfg.get("n_jobs", 0), logfile)
        perm_pairs = all_group_pairs(groups)

        log(f"[INFO] Permutations enabled for {len(perm_pairs)} pairwise group comparisons with n_perm={n_perm}, n_jobs={perm_n_jobs}", logfile)

        perm_graph_jobs = []
        perm_edge_jobs = []
        perm_net_edge_jobs = []

        for pair_idx, (ga, gb) in enumerate(perm_pairs):
            pair_seed = seed + 100003 * (pair_idx + 1)
            perm_graph_jobs.append({
                "job_label": f"graph metrics {ga} vs {gb}",
                "mat": mat,
                "group_a": ga,
                "group_b": gb,
                "n_perm": n_perm,
                "seed": pair_seed,
                "min_brains": min_brains,
                "p_threshold": p_thr,
                "min_abs_r": min_abs_r,
                "include_negative": include_neg,
            })

            if edges_only and (ga in edges_by_group) and (gb in edges_by_group):
                e_union = pd.concat([edges_by_group[ga], edges_by_group[gb]], ignore_index=True)
                if not e_union.empty:
                    uu = e_union[["u", "v"]].copy()
                    uu["a"] = uu.min(axis=1)
                    uu["b"] = uu.max(axis=1)
                    uu = uu.drop_duplicates(subset=["a", "b"])[["a", "b"]].rename(columns={"a": "u", "b": "v"})
                    perm_edge_jobs.append({
                        "job_label": f"rewiring edges {ga} vs {gb}",
                        "mat": mat,
                        "group_a": ga,
                        "group_b": gb,
                        "edges_to_test": uu,
                        "n_perm": n_perm,
                        "seed": pair_seed + 17,
                        "min_brains": min_brains,
                    })

                    if expanded_map:
                        for net_idx, (net_name, regs) in enumerate(expanded_map.items(), start=1):
                            regs_in = sorted(set(regs).intersection(mat.columns))
                            if len(regs_in) < 5:
                                continue
                            e_net = e_union[e_union["u"].isin(regs_in) & e_union["v"].isin(regs_in)].copy()
                            if e_net.empty:
                                continue
                            uu_net = e_net[["u", "v"]].copy()
                            uu_net["a"] = uu_net.min(axis=1)
                            uu_net["b"] = uu_net.max(axis=1)
                            uu_net = uu_net.drop_duplicates(subset=["a", "b"])[["a", "b"]].rename(columns={"a": "u", "b": "v"})
                            perm_net_edge_jobs.append({
                                "job_label": f"rewiring network {net_name}: {ga} vs {gb}",
                                "mat": mat,
                                "group_a": ga,
                                "group_b": gb,
                                "network": net_name,
                                "edges_to_test": uu_net,
                                "n_perm": n_perm,
                                "seed": pair_seed + 1000 + net_idx,
                                "min_brains": min_brains,
                            })

        perm_graph = run_parallel_job_dicts("Permutation graph metrics", perm_graph_jobs, _perm_graph_metrics_worker, perm_n_jobs, logfile)
        perm_edges = run_parallel_job_dicts("Permutation rewiring edges", perm_edge_jobs, _perm_rewiring_worker, perm_n_jobs, logfile) if edges_only else []
        perm_net_edges = run_parallel_job_dicts("Permutation rewiring edges by network", perm_net_edge_jobs, _perm_rewiring_worker, perm_n_jobs, logfile) if (edges_only and expanded_map) else []

        if perm_graph:
            perm_graph_df = pd.concat(perm_graph, ignore_index=True)
            save_df(perm_graph_df, out_root / "connectivity" / "perm_graph_metrics_all_pairs.csv")
            if control_group in groups:
                legacy = perm_graph_df[(perm_graph_df["group_a"] == control_group) | (perm_graph_df["group_b"] == control_group)].copy()
                if not legacy.empty:
                    save_df(legacy, out_root / "connectivity" / "perm_graph_metrics_vs_control.csv")

        if perm_edges:
            edges_perm = pd.concat(perm_edges, ignore_index=True)
            edges_perm["q_perm"] = np.nan
            for _, idx in edges_perm.groupby(["group_a", "group_b"]).groups.items():
                edges_perm.loc[idx, "q_perm"] = benjamini_hochberg(edges_perm.loc[idx, "p_perm_two_sided"].to_numpy())
            save_df(edges_perm, out_root / "connectivity" / "perm_rewiring_edges_all_pairs.csv")
            if control_group in groups:
                legacy = edges_perm[(edges_perm["group_a"] == control_group) | (edges_perm["group_b"] == control_group)].copy()
                if not legacy.empty:
                    save_df(legacy, out_root / "connectivity" / "perm_rewiring_edges_vs_control.csv")

        if perm_net_edges:
            net_edges = pd.concat(perm_net_edges, ignore_index=True)
            net_edges["q_perm"] = np.nan
            for _, idx2 in net_edges.groupby(["group_a", "group_b", "network"]).groups.items():
                net_edges.loc[idx2, "q_perm"] = benjamini_hochberg(net_edges.loc[idx2, "p_perm_two_sided"].to_numpy())
            save_df(net_edges, out_root / "connectivity" / "perm_rewiring_edges_all_pairs_by_network.csv")

            perm_net_summary = []
            for (ga, gb, net), subidx in net_edges.groupby(["group_a", "group_b", "network"]).groups.items():
                sub = net_edges.loc[subidx]
                perm_net_summary.append({
                    "group_a": ga,
                    "group_b": gb,
                    "network": net,
                    "n_edges_tested": int(len(sub)),
                    "n_sig_qperm_0_05": int((sub["q_perm"] <= 0.05).sum()),
                    "min_pperm": float(sub["p_perm_two_sided"].min()),
                    "min_qperm": float(sub["q_perm"].min()),
                })
            perm_net_summary_df = pd.DataFrame(perm_net_summary)
            save_df(perm_net_summary_df, out_root / "connectivity" / "perm_rewiring_network_summary_all_pairs.csv")

            if control_group in groups:
                legacy_net = net_edges[(net_edges["group_a"] == control_group) | (net_edges["group_b"] == control_group)].copy()
                if not legacy_net.empty:
                    save_df(legacy_net, out_root / "connectivity" / "perm_rewiring_edges_vs_control_by_network.csv")
                legacy_sum = perm_net_summary_df[(perm_net_summary_df["group_a"] == control_group) | (perm_net_summary_df["group_b"] == control_group)].copy()
                if not legacy_sum.empty:
                    save_df(legacy_sum, out_root / "connectivity" / "perm_rewiring_network_summary.csv")

    # Excel
    xlsx_path = out_root/"excel"/"complete_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        data.head(2000).to_excel(xl, sheet_name="Raw_preview_2000", index=False)
        slice_qc.to_excel(xl, sheet_name="QC_Slices", index=False)
        brain_qc.to_excel(xl, sheet_name="QC_Brains", index=False)
        cov.to_excel(xl, sheet_name="QC_RegionCoverage", index=False)
        pd.DataFrame({"unmatched_regions": unmapped}).to_excel(xl, sheet_name="UnmatchedRegions", index=False)
        brain_region_all.to_excel(xl, sheet_name="Brain_Region_All", index=False)
        region_stats_all.to_excel(xl, sheet_name="Stats_Regions_All", index=False)
        ar = out_root/"stats"/"anova_regions_lograte.csv"
        if ar.exists():
            pd.read_csv(ar).to_excel(xl, sheet_name="ANOVA_Regions", index=False)
        tr = out_root/"stats"/"posthoc_regions_welch_fdr.csv"
        if tr.exists():
            pd.read_csv(tr).head(200000).to_excel(xl, sheet_name="Posthoc_Regions_Welch_FDR", index=False)
        if not brain_region_qc.empty:
            brain_region_qc.to_excel(xl, sheet_name="Brain_Region_QC", index=False)
        if not region_stats_qc.empty:
            region_stats_qc.to_excel(xl, sheet_name="Stats_Regions_QC", index=False)
        if stability_rows:
            pd.DataFrame(stability_rows).to_excel(xl, sheet_name="Stability_All_vs_QC", index=False)
        net_table.to_excel(xl, sheet_name="Network_Table", index=False)
        net_stats.to_excel(xl, sheet_name="Stats_Networks", index=False)
        net_cov.to_excel(xl, sheet_name="Network_Coverage", index=False)
        an = out_root/"networks"/"anova_networks_lograte.csv"
        if an.exists():
            pd.read_csv(an).to_excel(xl, sheet_name="ANOVA_Networks", index=False)
        tn = out_root/"networks"/"posthoc_networks_welch_fdr.csv"
        if tn.exists():
            pd.read_csv(tn).to_excel(xl, sheet_name="Posthoc_Networks_Welch_FDR", index=False)
        if group_metrics:
            pd.DataFrame(group_metrics).to_excel(xl, sheet_name="GraphMetrics_ByGroup", index=False)
        pg = out_root/"connectivity"/"perm_graph_metrics_vs_control.csv"
        if pg.exists():
            pd.read_csv(pg).to_excel(xl, sheet_name="Perm_GraphMetrics", index=False)
        pe = out_root/"connectivity"/"perm_rewiring_edges_vs_control.csv"
        if pe.exists():
            pd.read_csv(pe).head(5000).to_excel(xl, sheet_name="Perm_RewiringEdges_5k", index=False)
        focused_dir = out_root/"stats"/"focused_biological_comparisons"
        if focused_dir.exists():
            for csv_path in sorted(focused_dir.glob("*.csv")):
                try:
                    pd.read_csv(csv_path).head(200000).to_excel(xl, sheet_name=csv_path.stem[:31], index=False)
                except Exception:
                    pass

    # Report
    rep = out_root/"report"/"analysis_report.txt"
    alpha = float(cfg["stats"]["alpha_fdr"])
    with rep.open("w", encoding="utf-8") as f:
        f.write("cFos Whole-brain analysis report\n")
        f.write("="*34 + "\n\n")
        f.write(f"Atlas: {cfg['atlas']['name']}\n")
        f.write(f"Control group: {control_group}\n")
        f.write(f"Groups (order): {', '.join(groups)}\n")
        f.write(f"Brains total: {brain_qc['brain_id'].nunique()}\n")
        f.write(f"Rows loaded: {len(data):,}\n\n")
        f.write("Key settings\n")
        f.write("-"*12 + "\n")
        f.write(f"min_dapi_region: {cfg['qc']['min_dapi_region']}\n")
        f.write(f"coverage warn threshold: {cfg['qc']['coverage_fraction_warn']*100:.0f}% per group\n")
        f.write(f"zscore_per_brain: {bool(cfg['connectivity'].get('zscore_per_brain', False))}\n")
        f.write(f"connectivity edge threshold: |r|>={min_abs_r}, q<={edge_alpha}\n\n")
        if not region_stats_all.empty:
            sig = region_stats_all[region_stats_all['q'] <= alpha]
            f.write(f"Region tests: {len(region_stats_all)} (significant q<={alpha}: {len(sig)})\n")
            f.write("Top regions: stats/top50_regions_per_term.csv\n\n")
        f.write("Focused biological subset comparisons: stats/focused_biological_comparisons/\n")
        f.write("Each focused comparison table includes test, group sizes, means, variances, variance ratio, p/q, and effect size.\n\n")
        f.write("Permutation outputs (if enabled):\n")
        f.write("  connectivity/perm_graph_metrics_vs_control.csv\n")
        f.write("  connectivity/perm_rewiring_edges_vs_control.csv\n")
        f.write("  connectivity/perm_rewiring_edges_vs_control_by_network.csv\n")
        f.write("  connectivity/perm_rewiring_network_summary.csv\n")

    log(f"[DONE] Excel: {xlsx_path}", logfile)
    log(f"[DONE] Report: {rep}", logfile)
    log("[DONE] All results written.", logfile)

if __name__ == "__main__":
    main()
