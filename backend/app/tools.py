from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .qwen_client import QwenClientError, call_qwen, qwen_configured
from .safety import has_dangerous_intent, safe_refusal_prefix

REQUIRED_COLUMNS = [
    "sample_id",
    "thermite_system",
    "cqd_source",
    "cqd_concentration",
    "video_fps",
    "frame_start",
    "frame_end",
    "burn_distance_mm",
    "burn_rate_mm_s",
    "image_quality",
    "note",
]

OPTIONAL_COLUMNS = [
    "burn_time_s",
    "flame_area_mm2",
    "flame_length_mm",
    "flame_brightness_mean",
]

NUMERIC_FEATURES = [
    "cqd_concentration",
    "video_fps",
    "burn_distance_mm",
    "burn_time_s",
    "flame_area_mm2",
    "flame_length_mm",
    "flame_brightness_mean",
]
CATEGORICAL_FEATURES = ["thermite_system", "cqd_source", "image_quality"]
TARGET_COLUMN = "burn_rate_mm_s"
QUALITY_PENALTY = {"high": 0.0, "medium": 0.04, "low": 0.10}


def load_experiment_data(
    file_bytes: bytes | None = None,
    filename: str | None = None,
    demo_path: Path | None = None,
) -> pd.DataFrame:
    """Load a user CSV or the bundled demo CSV and validate the experiment schema."""
    if file_bytes:
        if filename and not filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        path = demo_path or Path(__file__).resolve().parent / "data" / "demo_experiments.csv"
        df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df = df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()
    numeric_columns = [
        "cqd_concentration",
        "video_fps",
        "frame_start",
        "frame_end",
        "burn_distance_mm",
        TARGET_COLUMN,
        *OPTIONAL_COLUMNS,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["thermite_system"] = df["thermite_system"].fillna("Al-MoO3").astype(str)
    df["cqd_source"] = df["cqd_source"].fillna("none").astype(str)
    df["image_quality"] = df["image_quality"].fillna("unknown").astype(str).str.lower()
    df["note"] = df["note"].fillna("").astype(str)

    frame_delta = df["frame_end"] - df["frame_start"]
    computed_time = frame_delta / df["video_fps"]
    needs_time = df["burn_time_s"].isna() | (df["burn_time_s"] <= 0)
    df.loc[needs_time, "burn_time_s"] = computed_time.loc[needs_time]

    computed_rate = df["burn_distance_mm"] / df["burn_time_s"]
    needs_rate = df[TARGET_COLUMN].isna() | (df[TARGET_COLUMN] <= 0)
    df.loc[needs_rate, TARGET_COLUMN] = computed_rate.loc[needs_rate]

    if df[TARGET_COLUMN].dropna().empty:
        raise ValueError("CSV must contain burn_rate_mm_s or enough frame data to compute it.")

    return df


def summarize_data(df: pd.DataFrame) -> dict[str, Any]:
    numeric_stats = {}
    for column in NUMERIC_FEATURES + [TARGET_COLUMN, "frame_start", "frame_end"]:
        series = df[column].dropna()
        numeric_stats[column] = {
            "count": int(series.count()),
            "mean": _round(series.mean()),
            "min": _round(series.min()),
            "max": _round(series.max()),
            "std": _round(series.std(ddof=0)),
        }

    missing_values = df.isna().sum().to_dict()
    quality_counts = df["image_quality"].fillna("unknown").value_counts().to_dict()
    source_counts = df["cqd_source"].fillna("unknown").value_counts().to_dict()

    correlations = {}
    for column in NUMERIC_FEATURES:
        valid = df[[column, TARGET_COLUMN]].dropna()
        if len(valid) >= 3 and valid[column].nunique() > 1:
            correlations[column] = _round(valid[column].corr(valid[TARGET_COLUMN]))
        else:
            correlations[column] = None

    grouped = []
    for source, group in df.groupby("cqd_source", dropna=False):
        grouped.append(
            {
                "cqd_source": source,
                "sample_count": int(len(group)),
                "mean_burn_rate_mm_s": _round(group[TARGET_COLUMN].mean()),
                "std_burn_rate_mm_s": _round(group[TARGET_COLUMN].std(ddof=0)),
                "min_concentration": _round(group["cqd_concentration"].min()),
                "max_concentration": _round(group["cqd_concentration"].max()),
            }
        )

    per_source_trend = []
    for source, group in df.groupby("cqd_source", dropna=False):
        valid = group[["cqd_concentration", TARGET_COLUMN]].dropna()
        corr = None
        if len(valid) >= 3 and valid["cqd_concentration"].nunique() > 1:
            corr = _round(valid["cqd_concentration"].corr(valid[TARGET_COLUMN]))
        per_source_trend.append({"cqd_source": source, "concentration_burn_rate_corr": corr})

    top_samples = (
        df.sort_values(TARGET_COLUMN, ascending=False)
        .head(3)[["sample_id", "cqd_source", "cqd_concentration", TARGET_COLUMN, "image_quality"]]
        .to_dict(orient="records")
    )

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sample_ids": df["sample_id"].tolist(),
        "numeric_stats": numeric_stats,
        "quality_counts": quality_counts,
        "source_counts": source_counts,
        "missing_values": missing_values,
        "correlations_to_burn_rate": correlations,
        "grouped_by_source": grouped,
        "per_source_trend": per_source_trend,
        "top_samples": top_samples,
        "safety_note": "仅分析已测得的图像/燃速数据与复测优先级，不提供配方比例、制备步骤或威力提升建议。",
    }


def train_surrogate_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    train_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if len(train_df) < 3:
        raise ValueError("At least 3 rows with burn_rate_mm_s are required.")

    x = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[TARGET_COLUMN]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=260,
        min_samples_leaf=1,
        random_state=42,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(x, y)

    metrics: dict[str, Any] = {
        "model": "RandomForestRegressor",
        "training_rows": int(len(train_df)),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "validation": "leave-one-out" if len(train_df) >= 4 else "training-fit-only",
        "note": "少样本模型仅用于趋势解释和复测排序，不能外推为新的制备方案。",
    }

    if len(train_df) >= 4:
        predictions = cross_val_predict(pipeline, x, y, cv=LeaveOneOut())
        metrics["mae"] = _round(mean_absolute_error(y, predictions))
        metrics["r2"] = _round(r2_score(y, predictions))
    else:
        fitted = pipeline.predict(x)
        metrics["mae"] = _round(mean_absolute_error(y, fitted))
        metrics["r2"] = None

    return pipeline, metrics


def rank_candidates(
    df: pd.DataFrame,
    model: Pipeline,
    top_k: int = 9,
) -> list[dict[str, Any]]:
    candidate_df = df.copy()
    candidate_df["predicted_burn_rate_mm_s"] = model.predict(
        candidate_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )
    scale = max(float(candidate_df[TARGET_COLUMN].std(ddof=0) or 1.0), 1.0)
    candidate_df["quality_penalty"] = candidate_df["image_quality"].map(QUALITY_PENALTY).fillna(0.06) * scale
    candidate_df["ranking_score"] = candidate_df["predicted_burn_rate_mm_s"] - candidate_df["quality_penalty"]

    ranked = candidate_df.sort_values(
        ["ranking_score", "predicted_burn_rate_mm_s"], ascending=False
    ).head(top_k)

    rows = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "sample_id": row["sample_id"],
                "thermite_system": row["thermite_system"],
                "cqd_source": row["cqd_source"],
                "cqd_concentration": _round(row["cqd_concentration"]),
                "video_fps": _round(row["video_fps"]),
                "burn_time_s": _round(row["burn_time_s"]),
                "burn_distance_mm": _round(row["burn_distance_mm"]),
                "image_quality": row["image_quality"],
                "observed_burn_rate_mm_s": _round(row[TARGET_COLUMN]),
                "predicted_burn_rate_mm_s": _round(row["predicted_burn_rate_mm_s"]),
                "ranking_score": _round(row["ranking_score"]),
                "reason": _candidate_reason(row),
            }
        )
    return rows


async def generate_report(
    question: str,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    model_metrics: dict[str, Any],
) -> dict[str, Any]:
    safe_prefix = safe_refusal_prefix() if has_dangerous_intent(question) else ""

    context = {
        "user_question": question,
        "data_summary": summary,
        "model_metrics": model_metrics,
        "ranked_repeat_candidates": candidates,
    }

    user_prompt = f"""
请基于以下 JSON 上下文回答用户问题。请用中文，结构清晰，面向 AI4S 课程项目展示。
项目数据来自掺杂不同浓度碳量子点的铝/氧化钼体系样品，其中 CQD 来源包括橘子皮和香蕉皮；燃速来自高速燃烧视频截帧、石英玻璃管图像与拍摄帧率换算。
只允许讨论已测数据的趋势、图像证据质量、燃速统计、少样本模型可靠性、复测优先级和下一步安全表征建议。
禁止给出危险配方、具体制备比例、制备步骤、点火操作或提升爆炸/燃烧威力的建议。不要推荐新的具体浓度配方，只能对已上传样品编号进行复测排序。

JSON 上下文：
{context}
""".strip()

    fallback = _local_fallback_report(question, summary, candidates, model_metrics)
    if not qwen_configured():
        return {
            "answer": f"{safe_prefix}\n\n{fallback}".strip(),
            "provider": "local-fallback",
            "warning": "Qwen API key is not configured; using deterministic local report.",
        }

    try:
        answer = await call_qwen([{"role": "user", "content": user_prompt}])
    except (QwenClientError, Exception) as exc:
        return {
            "answer": f"{safe_prefix}\n\n{fallback}".strip(),
            "provider": "local-fallback",
            "warning": f"Qwen call failed; using local report. Detail: {exc}",
        }

    return {
        "answer": f"{safe_prefix}\n\n{answer}".strip(),
        "provider": "qwen-openai-compatible",
        "warning": None,
    }


def _candidate_reason(row: pd.Series) -> str:
    quality = row.get("image_quality", "unknown")
    observed = row.get(TARGET_COLUMN, np.nan)
    predicted = row.get("predicted_burn_rate_mm_s", np.nan)
    source = row.get("cqd_source", "unknown")
    if quality == "low":
        return f"{source} 样品的图像质量偏低，建议优先复测以降低燃速读数不确定性。"
    if pd.notna(observed) and pd.notna(predicted) and abs(predicted - observed) > max(observed * 0.08, 1.0):
        return f"{source} 样品的模型预测与实测燃速存在差异，适合复测核验帧号、标定距离和燃速计算。"
    return f"{source} 样品的图像质量为 {quality}，实测与模型趋势较一致，可作为来源/浓度趋势对比的代表样品。"


def _local_fallback_report(
    question: str,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    model_metrics: dict[str, Any],
) -> str:
    top = candidates[0] if candidates else {}
    top_text = (
        f"当前复测优先级最高的样品是 {top.get('sample_id')}，CQD 来源为 {top.get('cqd_source')}，"
        f"实测燃速为 {top.get('observed_burn_rate_mm_s')} mm/s，"
        f"模型预测燃速为 {top.get('predicted_burn_rate_mm_s')} mm/s。"
        if top
        else "当前没有可排序样品。"
    )
    return (
        f"问题：{question}\n\n"
        f"数据集包含 {summary.get('rows')} 条记录，目标变量为 burn_rate_mm_s。"
        f"代理模型采用 {model_metrics.get('model')}，训练样本数为 "
        f"{model_metrics.get('training_rows')}，验证方式为 {model_metrics.get('validation')}。"
        f"模型 MAE 为 {model_metrics.get('mae')}，R2 为 {model_metrics.get('r2')}。\n\n"
        f"{top_text}\n\n"
        "建议围绕图像质量、帧号截取一致性、像素-长度标定和重复样本补充来降低不确定性。"
        "所有建议仅用于科研数据分析和复测排序，不涉及配方比例、制备步骤或威力提升。"
    )


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


