import base64
import io
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

TARGET_LABEL = "LABEL_R_TO_RX_DRUG_BLOOD_OR_URINE_ENT_RX_M2D_TO_P7D"
DEFAULT_PREFIX = "RadixModel-deid-20261016_1028"
TOP_K_FEATURES = 52
GRID_STEPS = 1000  # 1001 points from 0.000 to 1.000

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", "/experiments"))


def _format_cell_value(val: Any) -> str:
    """Format a single cell value cleanly for display in the pivoted table."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NaN / Missing"
    try:
        if pd.isna(val):
            return "NaN / Missing"
    except Exception:
        pass
    if isinstance(val, (np.floating, float)):
        if float(val).is_integer():
            return str(int(val))
        return f"{float(val):.4f}".rstrip("0").rstrip(".")
    if isinstance(val, (np.integer, int)):
        return str(int(val))
    if isinstance(val, (np.bool_, bool)):
        return "True" if bool(val) else "False"
    return str(val)


def _format_shap_data_value(val: Any) -> Any:
    """Format a cell value for SHAP waterfall axis labels."""
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except Exception:
        pass
    if isinstance(val, (np.floating, float)):
        if float(val).is_integer():
            return int(val)
        return round(float(val), 3)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val)


class ModelStateManager:
    """Loads and caches the XGBoost model, ranked features, and dataset splits."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.data_dir = experiments_dir / "data"
        self.metadata_path = experiments_dir / "metadata" / "ranked_features.json"
        self.ubj_path = experiments_dir / "xgb_model.ubj"
        self.json_path = experiments_dir / "xgb_model.json"

        self.booster: Optional[xgb.Booster] = None
        self.model_path_used: Optional[str] = None
        self.ranked_features: List[str] = []
        self.final_features: List[str] = []
        self.split_cache: Dict[str, Dict[str, Any]] = {}

    def _load_ranked_features(self) -> List[str]:
        if self.final_features:
            return self.final_features

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                features = [
                    item["feature"] if isinstance(item, dict) and "feature" in item
                    else (item["name"] if isinstance(item, dict) and "name" in item else str(item))
                    for item in raw
                ]
            elif isinstance(raw, dict):
                if "ranked_features" in raw and isinstance(raw["ranked_features"], list):
                    features = [str(x) for x in raw["ranked_features"]]
                elif "features" in raw and isinstance(raw["features"], list):
                    features = [str(x) for x in raw["features"]]
                else:
                    features = [str(k) for k in raw.keys()]
            else:
                features = []
            self.ranked_features = features
            self.final_features = features[:TOP_K_FEATURES]
            return self.final_features

        # Fallback to booster feature names if ranked_features.json is absent
        booster = self._load_booster()
        if booster.feature_names:
            self.ranked_features = list(booster.feature_names)
            self.final_features = self.ranked_features[:TOP_K_FEATURES]
            return self.final_features

        raise FileNotFoundError(
            f"Could not find ranked features at {self.metadata_path}"
        )

    def _load_booster(self) -> xgb.Booster:
        if self.booster is not None:
            return self.booster

        candidates = [
            self.ubj_path,
            self.json_path,
            self.experiments_dir / "models" / "xgb_model.ubj",
            self.experiments_dir / "models" / "xgb_model.json",
        ]
        for path in candidates:
            if path.exists():
                booster = xgb.Booster()
                booster.load_model(str(path))
                self.booster = booster
                self.model_path_used = str(path)
                return booster

        raise FileNotFoundError(
            f"Could not find xgb_model.ubj or xgb_model.json in {self.experiments_dir}"
        )

    def discover_splits(self) -> Dict[str, Tuple[Path, Path]]:
        """Discover available (X, y) parquet pairs for val, test, and train."""
        splits: Dict[str, Tuple[Path, Path]] = {}
        if not self.data_dir.exists():
            return splits

        for split in ["val", "test", "train"]:
            # 1. Check the exact prefix specified by the user first
            exact_x = self.data_dir / f"{DEFAULT_PREFIX}_X_{split}.parquet"
            exact_y = self.data_dir / f"{DEFAULT_PREFIX}_y_{split}.parquet"
            if exact_x.exists() and exact_y.exists():
                splits[split] = (exact_x, exact_y)
                continue

            # 2. Fallback: scan for any matching *_X_{split}.parquet & *_y_{split}.parquet
            x_matches = sorted(self.data_dir.glob(f"*X_{split}.parquet"))
            for x_path in reversed(x_matches):
                prefix = x_path.name[: -len(f"X_{split}.parquet")]
                y_path = self.data_dir / f"{prefix}y_{split}.parquet"
                if y_path.exists():
                    splits[split] = (x_path, y_path)
                    break

        return splits

    @staticmethod
    def _extract_target(y_df: pd.DataFrame) -> np.ndarray:
        if isinstance(y_df, pd.Series):
            s = y_df
        elif TARGET_LABEL in y_df.columns:
            s = y_df[TARGET_LABEL]
        elif y_df.shape[1] == 1:
            s = y_df.iloc[:, 0]
        else:
            raise ValueError(
                f"Target column '{TARGET_LABEL}' not found in y dataframe columns: {list(y_df.columns)}"
            )

        if s.dtype == bool or str(s.dtype) == "boolean":
            return s.astype(int).to_numpy(dtype=np.int32)
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int32)

    @staticmethod
    def _prepare_model_features(X_df: pd.DataFrame, final_features: List[str]) -> pd.DataFrame:
        missing = [f for f in final_features if f not in X_df.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} model features from X dataframe (first 5: {missing[:5]})"
            )
        X_sub = X_df[final_features].copy()
        for col in X_sub.columns:
            dt = X_sub[col].dtype
            if dt == object or isinstance(dt, pd.StringDtype):
                X_sub[col] = X_sub[col].astype("category")
        return X_sub

    def get_split_data(self, split: Optional[str] = None) -> Dict[str, Any]:
        available = self.discover_splits()
        if not available:
            raise FileNotFoundError(
                f"No parquet splits (*_X_val/test/train.parquet) found in {self.data_dir}"
            )

        if not split or split not in available:
            # Prefer val or test first
            for preferred in ["val", "test", "train"]:
                if preferred in available:
                    split = preferred
                    break

        assert split is not None
        if split in self.split_cache:
            return self.split_cache[split]

        x_path, y_path = available[split]
        booster = self._load_booster()
        final_features = self._load_ranked_features()

        X_df = pd.read_parquet(x_path)
        y_df = pd.read_parquet(y_path)
        y_true = self._extract_target(y_df)

        if len(X_df) != len(y_true):
            raise ValueError(
                f"Length mismatch between {x_path.name} ({len(X_df)}) and {y_path.name} ({len(y_true)})"
            )

        X_model = self._prepare_model_features(X_df, final_features)
        dmatrix = xgb.DMatrix(X_model, label=y_true, enable_categorical=True)
        raw_preds = np.asarray(booster.predict(dmatrix), dtype=np.float64)

        # Ensure probabilities are in [0, 1] in case raw margins were output
        if np.any(raw_preds < 0.0) or np.any(raw_preds > 1.0):
            y_pred = 1.0 / (1.0 + np.exp(-raw_preds))
        else:
            y_pred = raw_preds

        threshold_min = float(np.min(y_pred))
        threshold_max = float(np.max(y_pred))
        span = threshold_max - threshold_min
        if span <= 0.0:
            y_pred_norm = np.zeros_like(y_pred)
        else:
            y_pred_norm = (y_pred - threshold_min) / span

        # Compute AUROC & AUPRC
        has_both_classes = len(np.unique(y_true)) > 1
        auroc = float(roc_auc_score(y_true, y_pred)) if has_both_classes else 0.0
        auprc = float(average_precision_score(y_true, y_pred)) if has_both_classes else 0.0

        # Precompute 1,001 threshold points linearly normalised on [0, 1] from threshold_min to threshold_max
        t_norm_grid = np.linspace(0.0, 1.0, GRID_STEPS + 1)
        t_raw_grid = threshold_min + t_norm_grid * span

        # Fast exact confusion counts at every threshold via sorting
        order = np.argsort(y_pred, kind="mergesort")
        sorted_preds = y_pred[order]
        sorted_y = y_true[order]

        total_samples = len(y_true)
        total_pos = int(np.sum(y_true == 1))
        total_neg = total_samples - total_pos

        # Prefix sums of positives for fast count of samples >= threshold
        cum_pos = np.concatenate(([0], np.cumsum(sorted_y == 1)))
        idx_ge = np.searchsorted(sorted_preds, t_raw_grid, side="left")

        tp_arr = (total_pos - cum_pos[idx_ge]).astype(int)
        pred_pos_arr = (total_samples - idx_ge).astype(int)
        fp_arr = (pred_pos_arr - tp_arr).astype(int)
        fn_arr = (total_pos - tp_arr).astype(int)
        tn_arr = (total_neg - fp_arr).astype(int)

        with np.errstate(divide="ignore", invalid="ignore"):
            ppv_arr = np.where(pred_pos_arr > 0, tp_arr / pred_pos_arr, 0.0)
            sens_arr = np.where(total_pos > 0, tp_arr / total_pos, 0.0)
            fpr_arr = np.where(total_neg > 0, fp_arr / total_neg, 0.0)

        # Full ROC and PR curves for plotting
        if has_both_classes:
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred)
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred)
            # precision_recall_curve returns arrays ordered by increasing threshold (decreasing recall)
            rec_curve = rec_curve[::-1]
            prec_curve = prec_curve[::-1]
        else:
            fpr_curve, tpr_curve = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            rec_curve, prec_curve = np.array([0.0, 1.0]), np.array([1.0, 0.0])

        def _downsample(x: np.ndarray, y: np.ndarray, max_pts: int = 450) -> Tuple[List[float], List[float]]:
            if len(x) <= max_pts:
                return [round(float(v), 5) for v in x], [round(float(v), 5) for v in y]
            indices = np.unique(np.linspace(0, len(x) - 1, max_pts, dtype=int))
            return [round(float(x[i]), 5) for i in indices], [round(float(y[i]), 5) for i in indices]

        roc_x, roc_y = _downsample(fpr_curve, tpr_curve)
        pr_x, pr_y = _downsample(rec_curve, prec_curve)

        entry = {
            "split": split,
            "x_file": x_path.name,
            "y_file": y_path.name,
            "X_df": X_df,
            "X_model": X_model,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_norm": y_pred_norm,
            "n_samples": total_samples,
            "n_positives": total_pos,
            "n_negatives": total_neg,
            "prevalence": float(total_pos / total_samples) if total_samples > 0 else 0.0,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "auroc": auroc,
            "auprc": auprc,
            "grid": {
                "t_norm": [round(float(v), 4) for v in t_norm_grid],
                "t_raw": [round(float(v), 6) for v in t_raw_grid],
                "ppv": [round(float(v), 6) for v in ppv_arr],
                "sensitivity": [round(float(v), 6) for v in sens_arr],
                "fpr": [round(float(v), 6) for v in fpr_arr],
                "tp": tp_arr.tolist(),
                "fp": fp_arr.tolist(),
                "fn": fn_arr.tolist(),
                "tn": tn_arr.tolist(),
            },
            "roc_curve": {"fpr": roc_x, "tpr": roc_y},
            "pr_curve": {"recall": pr_x, "precision": pr_y},
        }
        self.split_cache[split] = entry
        return entry


state_manager = ModelStateManager(EXPERIMENTS_DIR)


async def homepage(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def api_state(request: Request) -> JSONResponse:
    requested_split = request.query_params.get("split")
    try:
        available = state_manager.discover_splits()
        split_data = state_manager.get_split_data(requested_split)
        final_features = state_manager._load_ranked_features()

        splits_info = []
        label_map = {
            "val": "Validation Set (X_val / y_val)",
            "test": "Test Set (X_test / y_test)",
            "train": "Training Set (X_train / y_train)",
        }
        for sp_key in ["val", "test", "train"]:
            if sp_key in available:
                x_p, y_p = available[sp_key]
                splits_info.append(
                    {
                        "key": sp_key,
                        "label": label_map.get(sp_key, sp_key),
                        "x_file": x_p.name,
                        "y_file": y_p.name,
                    }
                )

        return JSONResponse(
            {
                "status": "ok",
                "target_label": TARGET_LABEL,
                "model_path": state_manager.model_path_used,
                "available_splits": splits_info,
                "current_split": split_data["split"],
                "x_file": split_data["x_file"],
                "y_file": split_data["y_file"],
                "n_samples": split_data["n_samples"],
                "n_positives": split_data["n_positives"],
                "n_negatives": split_data["n_negatives"],
                "prevalence": split_data["prevalence"],
                "n_features_total": int(split_data["X_df"].shape[1]),
                "n_features_model": len(final_features),
                "final_features": final_features,
                "threshold_min": split_data["threshold_min"],
                "threshold_max": split_data["threshold_max"],
                "auroc": split_data["auroc"],
                "auprc": split_data["auprc"],
                "grid": split_data["grid"],
                "roc_curve": split_data["roc_curve"],
                "pr_curve": split_data["pr_curve"],
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "target_label": TARGET_LABEL,
                "message": str(exc),
                "experiments_dir": str(EXPERIMENTS_DIR),
            },
            status_code=500,
        )


async def api_sample(request: Request) -> JSONResponse:
    requested_split = request.query_params.get("split")
    row_index_param = request.query_params.get("row_index")
    exclude_param = request.query_params.get("exclude", "")

    try:
        split_data = state_manager.get_split_data(requested_split)
        n_samples = split_data["n_samples"]
        X_df: pd.DataFrame = split_data["X_df"]
        final_features = state_manager._load_ranked_features()
        rank_lookup = {feat: idx + 1 for idx, feat in enumerate(final_features)}

        if row_index_param is not None and row_index_param != "":
            row_idx = int(row_index_param)
            if row_idx < 0 or row_idx >= n_samples:
                return JSONResponse(
                    {"status": "error", "message": f"Row index must be between 0 and {n_samples - 1}."},
                    status_code=400,
                )
        else:
            excluded = set()
            if exclude_param:
                for part in exclude_param.split(","):
                    part = part.strip()
                    if part.isdigit():
                        excluded.add(int(part))
            if len(excluded) < n_samples:
                candidates = [i for i in range(n_samples) if i not in excluded]
                row_idx = random.choice(candidates)
            else:
                row_idx = random.randrange(n_samples)

        row_series = X_df.iloc[row_idx]
        orig_index = X_df.index[row_idx]

        # Build pivoted rows: 52 ranked model features first (in rank order), followed by any additional columns
        pivoted_rows = []
        seen_cols = set()
        for feat in final_features:
            if feat in X_df.columns:
                seen_cols.add(feat)
                pivoted_rows.append(
                    {
                        "feature": feat,
                        "value": _format_cell_value(row_series[feat]),
                        "is_model_feature": True,
                        "rank": rank_lookup[feat],
                    }
                )

        for col in X_df.columns:
            if col not in seen_cols:
                pivoted_rows.append(
                    {
                        "feature": str(col),
                        "value": _format_cell_value(row_series[col]),
                        "is_model_feature": False,
                        "rank": None,
                    }
                )

        true_label = int(split_data["y_true"][row_idx])
        prob_raw = float(split_data["y_pred"][row_idx])
        prob_norm = float(split_data["y_pred_norm"][row_idx])

        return JSONResponse(
            {
                "status": "ok",
                "split": split_data["split"],
                "row_index": row_idx,
                "sample_id": str(orig_index),
                "prob_raw": prob_raw,
                "prob_norm": prob_norm,
                "true_label": true_label,
                "true_label_text": "Resistant organism" if true_label == 1 else "Non-resistant organism",
                "pivoted_rows": pivoted_rows,
            }
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


async def api_shap(request: Request) -> JSONResponse:
    requested_split = request.query_params.get("split")
    row_index_param = request.query_params.get("row_index")

    if row_index_param is None or row_index_param == "":
        return JSONResponse(
            {"status": "error", "message": "row_index query parameter is required."},
            status_code=400,
        )

    try:
        split_data = state_manager.get_split_data(requested_split)
        row_idx = int(row_index_param)
        n_samples = split_data["n_samples"]
        if row_idx < 0 or row_idx >= n_samples:
            return JSONResponse(
                {"status": "error", "message": f"Row index {row_idx} out of range (0..{n_samples - 1})."},
                status_code=400,
            )

        booster = state_manager._load_booster()
        final_features = state_manager._load_ranked_features()
        X_model_row = split_data["X_model"].iloc[[row_idx]]

        drow = xgb.DMatrix(X_model_row, enable_categorical=True)
        contribs = booster.predict(drow, pred_contribs=True)[0]
        shap_vals = np.asarray(contribs[:-1], dtype=np.float64)
        base_val = float(contribs[-1])

        raw_row = X_model_row.iloc[0]
        display_data = np.array(
            [_format_shap_data_value(raw_row[col]) for col in final_features],
            dtype=object,
        )

        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=display_data,
            feature_names=final_features,
        )

        plt.close("all")
        fig = plt.figure(figsize=(10, 6.5), dpi=160)
        shap.plots.waterfall(explanation, max_display=15, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"Shapley Waterfall — Sample #{row_idx} ({split_data['split'].upper()} set)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
        plt.close("all")

        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        # Also return structured top contributions
        order = np.argsort(np.abs(shap_vals))[::-1]
        top_contribs = [
            {
                "feature": final_features[i],
                "value": _format_cell_value(raw_row[final_features[i]]),
                "shap_value": round(float(shap_vals[i]), 5),
            }
            for i in order[:15]
        ]

        return JSONResponse(
            {
                "status": "ok",
                "row_index": row_idx,
                "split": split_data["split"],
                "base_value": round(base_val, 5),
                "fx_value": round(float(base_val + np.sum(shap_vals)), 5),
                "image_data_url": f"data:image/png;base64,{b64_png}",
                "top_contributions": top_contribs,
            }
        )
    except Exception as exc:
        plt.close("all")
        return JSONResponse(
            {"status": "error", "message": str(exc)},
            status_code=500,
        )


routes = [
    Route("/", homepage),
    Route("/api/state", api_state),
    Route("/api/sample", api_sample),
    Route("/api/shap", api_shap, methods=["GET", "POST"]),
    Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

app = Starlette(debug=False, routes=routes)
