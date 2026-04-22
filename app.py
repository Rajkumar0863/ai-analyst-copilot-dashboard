import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Client Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.75rem;
    }
    .muted {
        color: #9aa0a6;
        font-size: 0.92rem;
    }
    .section-gap {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# UTILS
# =========================================================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def looks_boolean_like(series: pd.Series) -> bool:
    try:
        non_null = series.dropna()
        if non_null.empty:
            return False

        if pd.api.types.is_bool_dtype(non_null):
            return True

        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            vals = set(numeric.unique().tolist())
            if vals.issubset({0, 1}):
                return True

        lowered = set(non_null.astype(str).str.strip().str.lower().unique().tolist())
        bool_words = {"true", "false", "yes", "no", "y", "n", "0", "1"}
        if lowered and lowered.issubset(bool_words):
            return True

        return False
    except Exception:
        return False


def format_number(value: float) -> str:
    value = safe_float(value, 0.0)
    return f"{value:,.2f}"


def safe_unique_ratio(series: pd.Series) -> float:
    try:
        return series.nunique(dropna=True) / max(len(series), 1)
    except Exception:
        return 1.0


# =========================================================
# COLUMN INFERENCE
# =========================================================
def infer_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    for col in df.columns:
        try:
            if df[col].dtype == "object":
                sample = df[col].dropna().astype(str).head(100)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    return col
        except Exception:
            continue

    return None


def add_time_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if not date_col or date_col not in df.columns:
        return df

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    if temp[date_col].notna().sum() == 0:
        return temp

    try:
        if "year" not in temp.columns:
            temp["year"] = temp[date_col].dt.year
        if "month" not in temp.columns:
            temp["month"] = temp[date_col].dt.month
        if "week_of_year" not in temp.columns:
            temp["week_of_year"] = temp[date_col].dt.isocalendar().week.astype("Int64")
        if "day_of_week" not in temp.columns:
            temp["day_of_week"] = temp[date_col].dt.dayofweek
        if "hour" not in temp.columns:
            temp["hour"] = temp[date_col].dt.hour
        if "is_weekend" not in temp.columns:
            temp["is_weekend"] = temp[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    except Exception:
        pass

    return temp


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []

    for col in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            if pd.api.types.is_bool_dtype(df[col]):
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if not looks_boolean_like(df[col]):
                    numeric_cols.append(col)
                continue

            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.8 and not looks_boolean_like(converted):
                    numeric_cols.append(col)
        except Exception:
            continue

    return numeric_cols


def infer_categorical_columns(df: pd.DataFrame, numeric_cols: List[str], date_col: Optional[str]) -> List[str]:
    cat_cols = []
    for col in df.columns:
        try:
            if col in numeric_cols:
                continue
            if date_col and col == date_col:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            if safe_unique_ratio(df[col]) < 0.7:
                cat_cols.append(col)
        except Exception:
            continue
    return cat_cols


def infer_primary_metric(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    if not numeric_cols:
        return None

    priority = [
        "sales_amount_gbp", "sales_amount", "revenue", "sales", "profit",
        "gmv", "amount", "ad_spend", "conversions", "clicks", "impressions",
        "roas", "ctr", "cpa", "cost", "quantity"
    ]

    for keyword in priority:
        for col in numeric_cols:
            if keyword in col.lower():
                return col

    best_col = None
    best_score = -1
    for col in numeric_cols:
        try:
            score = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if score > best_score:
                best_score = score
                best_col = col
        except Exception:
            continue
    return best_col


def get_safe_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    safe_cols = []
    for col in numeric_cols:
        try:
            if col not in df.columns:
                continue
            if pd.api.types.is_bool_dtype(df[col]):
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 5:
                continue
            if looks_boolean_like(series):
                continue
            safe_cols.append(col)
        except Exception:
            continue
    return safe_cols


# =========================================================
# DATA QUALITY
# =========================================================
def compute_data_quality_score(df: pd.DataFrame) -> int:
    missing_pct = (df.isna().sum().sum() / max(df.size, 1)) * 100
    duplicate_pct = (df.duplicated().sum() / max(len(df), 1)) * 100

    score = 100
    score -= min(40, missing_pct * 2)
    score -= min(25, duplicate_pct * 2)
    return int(max(0, min(100, round(score))))


def detect_outlier_risk(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
    risks = []

    for col in numeric_cols[:25]:
        try:
            if col not in df.columns:
                continue

            if pd.api.types.is_bool_dtype(df[col]):
                continue

            series = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(series) < 10:
                continue

            if looks_boolean_like(series):
                continue

            series = series.astype(float)

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = series[(series < lower) | (series > upper)]
            outlier_pct = (len(outliers) / len(series)) * 100

            if outlier_pct >= 5:
                risks.append({
                    "column": col,
                    "outlier_pct": round(outlier_pct, 1)
                })
        except Exception:
            continue

    return sorted(risks, key=lambda x: x["outlier_pct"], reverse=True)[:5]


def detect_category_imbalance(df: pd.DataFrame, categorical_columns: List[str]) -> List[Dict[str, Any]]:
    alerts = []
    for col in categorical_columns[:10]:
        try:
            vc = df[col].astype(str).value_counts(dropna=False, normalize=True)
            if vc.empty:
                continue
            top_share = vc.iloc[0] * 100
            if top_share >= 90:
                alerts.append({
                    "column": col,
                    "top_share": round(top_share, 1)
                })
        except Exception:
            continue
    return alerts[:5]


# =========================================================
# ANALYSIS HELPERS
# =========================================================
def detect_best_dimension(df: pd.DataFrame, metric_col: str, categorical_columns: List[str]) -> Tuple[Optional[str], Optional[str], float]:
    best_dim, best_entity, best_value = None, None, -np.inf

    for col in categorical_columns[:8]:
        try:
            grouped = df.groupby(col, dropna=False)[metric_col].sum().sort_values(ascending=False)
            if not grouped.empty:
                entity = str(grouped.index[0])
                value = float(grouped.iloc[0])
                if value > best_value:
                    best_dim, best_entity, best_value = col, entity, value
        except Exception:
            continue

    if best_dim is None:
        return None, None, 0.0
    return best_dim, best_entity, best_value


def detect_weakest_dimension(df: pd.DataFrame, metric_col: str, categorical_columns: List[str]) -> Tuple[Optional[str], Optional[str], float]:
    weak_dim, weak_entity, weak_value = None, None, np.inf

    for col in categorical_columns[:8]:
        try:
            grouped = df.groupby(col, dropna=False)[metric_col].sum().sort_values(ascending=True)
            if not grouped.empty:
                entity = str(grouped.index[0])
                value = float(grouped.iloc[0])
                if value < weak_value:
                    weak_dim, weak_entity, weak_value = col, entity, value
        except Exception:
            continue

    if weak_dim is None:
        return None, None, 0.0
    return weak_dim, weak_entity, weak_value


def month_trend_stats(df: pd.DataFrame, date_col: Optional[str], metric_col: str) -> Dict[str, Any]:
    result = {
        "monthly_series": pd.Series(dtype=float),
        "first_value": None,
        "last_value": None,
        "pct_change": None,
        "direction": "none"
    }

    if not date_col or date_col not in df.columns:
        return result

    try:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])

        if temp.empty:
            return result

        monthly = temp.set_index(date_col)[metric_col].resample("M").sum().dropna()
        result["monthly_series"] = monthly

        if len(monthly) >= 2:
            first_val = float(monthly.iloc[0])
            last_val = float(monthly.iloc[-1])
            result["first_value"] = first_val
            result["last_value"] = last_val

            if first_val != 0:
                pct = ((last_val - first_val) / abs(first_val)) * 100
                result["pct_change"] = pct
                if pct > 0:
                    result["direction"] = "up"
                elif pct < 0:
                    result["direction"] = "down"

        return result
    except Exception:
        return result


# =========================================================
# RETRIEVAL-BACKED EVIDENCE
# =========================================================
def retrieve_evidence_rows(
    df: pd.DataFrame,
    metric_col: str,
    dimension: Optional[str] = None,
    entity: Optional[str] = None,
    top_n: int = 5
) -> pd.DataFrame:
    try:
        temp = df.copy()

        if dimension and entity and dimension in temp.columns:
            temp = temp[temp[dimension].astype(str) == str(entity)]

        if metric_col in temp.columns:
            temp["_metric_sort"] = pd.to_numeric(temp[metric_col], errors="coerce").fillna(0)
            temp = temp.sort_values("_metric_sort", ascending=False).drop(columns=["_metric_sort"], errors="ignore")

        return temp.head(top_n)
    except Exception:
        return pd.DataFrame()


def build_evidence_text(insight: Dict[str, Any], df: pd.DataFrame, metric_col: str) -> str:
    metric = insight.get("metric")
    dimension = insight.get("dimension")
    entity = insight.get("entity")

    try:
        if metric and dimension and entity and metric in df.columns and dimension in df.columns:
            subset = df[df[dimension].astype(str) == str(entity)]
            if not subset.empty:
                total_val = pd.to_numeric(subset[metric], errors="coerce").fillna(0).sum()
                avg_val = pd.to_numeric(subset[metric], errors="coerce").fillna(0).mean()
                return (
                    f"Grounded from filtered dataset: {dimension}={entity}, "
                    f"rows={len(subset)}, total_{metric}={round(float(total_val),2)}, "
                    f"avg_{metric}={round(float(avg_val),2)}."
                )

        if metric and metric in df.columns:
            metric_series = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not metric_series.empty:
                return (
                    f"Grounded from filtered dataset: rows={len(df)}, "
                    f"sum_{metric}={round(float(metric_series.sum()),2)}, "
                    f"mean_{metric}={round(float(metric_series.mean()),2)}."
                )
    except Exception:
        pass

    return "Grounded from current filtered dataset."


# =========================================================
# GEMINI
# =========================================================
def clean_json_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    return text.strip()


def safe_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = clean_json_text(raw_text)
    if not cleaned:
        return None

    candidates = [
        cleaned,
        re.sub(r",\s*([}\]])", r"\1", cleaned),
        cleaned.replace("“", '"').replace("”", '"')
    ]

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def build_dataset_context(df: pd.DataFrame, metric_col: str, date_col: Optional[str], categorical_cols: List[str], numeric_cols: List[str]) -> str:
    lines = []
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append(f"Primary metric: {metric_col}")
    lines.append(f"Missing values: {int(df.isna().sum().sum())}")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")

    metric_series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if not metric_series.empty:
        lines.append(f"{metric_col} total: {round(float(metric_series.sum()),2)}")
        lines.append(f"{metric_col} mean: {round(float(metric_series.mean()),2)}")
        lines.append(f"{metric_col} median: {round(float(metric_series.median()),2)}")

    if date_col and date_col in df.columns:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])
        if not temp.empty:
            lines.append(f"Date range: {temp[date_col].min()} to {temp[date_col].max()}")
            monthly = temp.set_index(date_col)[metric_col].resample("M").sum().dropna()
            if len(monthly) >= 2:
                first_val = float(monthly.iloc[0])
                last_val = float(monthly.iloc[-1])
                if first_val != 0:
                    pct = ((last_val - first_val) / abs(first_val)) * 100
                    lines.append(
                        f"Monthly trend: first={round(first_val,2)}, last={round(last_val,2)}, pct_change={round(pct,2)}"
                    )

    for col in categorical_cols[:5]:
        try:
            grouped = df.groupby(col, dropna=False)[metric_col].sum().sort_values(ascending=False).head(5)
            if not grouped.empty:
                lines.append(f"Top categories for {col}:")
                for idx, val in grouped.items():
                    lines.append(f"- {col}={idx}: {round(float(val),2)}")
        except Exception:
            continue

    for col in numeric_cols[:6]:
        if col == metric_col:
            continue
        try:
            corr_df = df[[metric_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(corr_df) >= 10:
                corr_val = corr_df.corr(numeric_only=True).iloc[0, 1]
                if not pd.isna(corr_val):
                    lines.append(f"Correlation with {metric_col}: {col}={round(float(corr_val),2)}")
        except Exception:
            continue

    return "\n".join(lines)


def build_gemini_prompt(dataset_context: str, user_question: Optional[str] = None) -> str:
    question_block = f"\nUser question: {user_question}\n" if user_question else ""
    return f"""
You are a senior business data analyst.

Return ONLY valid JSON.

Schema:
{{
  "insights": [
    {{
      "title": "short title",
      "statement": "single grounded finding",
      "insight_type": "trend|risk|recommendation|general",
      "metric": "metric name or null",
      "dimension": "dimension name or null",
      "entity": "entity value or null",
      "direction": "up|down|high|low|mixed|none",
      "evidence": "specific evidence from the context",
      "base_confidence": 0
    }}
  ]
}}

Rules:
1. Give 3 to 5 insights.
2. Use only the dataset context.
3. Do not invent unavailable facts.
4. Keep findings grounded.
5. Avoid internal contradiction.

Dataset context:
{dataset_context}
{question_block}
""".strip()


def run_gemini_engine(dataset_context: str, user_question: Optional[str] = None) -> Dict[str, Any]:
    if not GEMINI_AVAILABLE:
        return {"status": "unavailable", "error": "google-generativeai not installed", "insights": []}

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"status": "unavailable", "error": "GEMINI_API_KEY not set", "insights": []}

    try:
        genai.configure(api_key=api_key)
        candidate_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        last_error = None

        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    build_gemini_prompt(dataset_context, user_question),
                    generation_config={"temperature": 0.2, "max_output_tokens": 1200}
                )
                raw_text = getattr(response, "text", "") or ""
                parsed = safe_parse_json(raw_text)

                if not parsed or "insights" not in parsed:
                    last_error = f"{model_name} returned invalid JSON"
                    continue

                output = []
                for item in parsed.get("insights", []):
                    output.append({
                        "engine": "Gemini LLM Engine",
                        "title": str(item.get("title", "Gemini Insight")).strip(),
                        "statement": str(item.get("statement", "")).strip(),
                        "insight_type": str(item.get("insight_type", "general")).strip().lower(),
                        "metric": item.get("metric"),
                        "dimension": item.get("dimension"),
                        "entity": item.get("entity"),
                        "direction": str(item.get("direction", "none")).strip().lower(),
                        "evidence": str(item.get("evidence", "Grounded from filtered dataset.")).strip(),
                        "base_confidence": max(0, min(100, safe_int(item.get("base_confidence", 60), 60)))
                    })

                if output:
                    return {"status": "ok", "error": None, "insights": output}

            except Exception as model_error:
                last_error = str(model_error)
                continue

        return {"status": "failed", "error": last_error or "Gemini failed", "insights": []}

    except Exception as e:
        return {"status": "failed", "error": str(e), "insights": []}


# =========================================================
# ENGINES
# =========================================================
def build_rule_based_insights(df: pd.DataFrame, metric_col: str, date_col: Optional[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
    insights = []

    best_dim, best_entity, best_value = detect_best_dimension(df, metric_col, cat_cols)
    weak_dim, weak_entity, weak_value = detect_weakest_dimension(df, metric_col, cat_cols)
    trend = month_trend_stats(df, date_col, metric_col)
    duplicate_count = int(df.duplicated().sum())

    if best_dim and best_entity:
        insights.append({
            "engine": "Rule-Based Engine",
            "title": "Strongest segment",
            "statement": f"The strongest driver of {metric_col} is {best_entity} within {best_dim}, contributing {best_value:,.2f} overall.",
            "insight_type": "trend",
            "metric": metric_col,
            "dimension": best_dim,
            "entity": best_entity,
            "direction": "high",
            "base_confidence": 88
        })

    if weak_dim and weak_entity:
        insights.append({
            "engine": "Rule-Based Engine",
            "title": "Weakest segment",
            "statement": f"Underperformance appears in {weak_entity} under {weak_dim}, making it a priority area for review.",
            "insight_type": "recommendation",
            "metric": metric_col,
            "dimension": weak_dim,
            "entity": weak_entity,
            "direction": "low",
            "base_confidence": 84
        })

    if trend["pct_change"] is not None:
        word = "improved" if trend["pct_change"] > 0 else "declined"
        insights.append({
            "engine": "Rule-Based Engine",
            "title": "Time trend",
            "statement": f"Over time, {metric_col} has {word} by {abs(trend['pct_change']):.1f}% from the first month to the latest month.",
            "insight_type": "trend",
            "metric": metric_col,
            "dimension": "time",
            "entity": "monthly",
            "direction": trend["direction"],
            "base_confidence": 90
        })

    if duplicate_count > 0:
        insights.append({
            "engine": "Rule-Based Engine",
            "title": "Duplicate risk",
            "statement": f"Duplicate record risk detected: {duplicate_count} duplicate rows should be reviewed before external reporting.",
            "insight_type": "risk",
            "metric": None,
            "dimension": "dataset",
            "entity": "duplicates",
            "direction": "high",
            "base_confidence": 95
        })

    return insights[:5]


def build_statistical_insights(df: pd.DataFrame, metric_col: str, date_col: Optional[str], safe_numeric_cols: List[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
    insights = []

    num_df = df[safe_numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(numeric_only=True)

    if metric_col in corr.columns:
        metric_corr = corr[metric_col].drop(labels=[metric_col], errors="ignore").dropna()
        if not metric_corr.empty:
            strongest_feature = metric_corr.abs().sort_values(ascending=False).index[0]
            strongest_value = corr.loc[strongest_feature, metric_col]
            insights.append({
                "engine": "Statistical Engine",
                "title": "Strongest relationship",
                "statement": f"The strongest statistical relationship is a {'positive' if strongest_value >= 0 else 'negative'} correlation of {strongest_value:.2f} between {strongest_feature} and {metric_col}.",
                "insight_type": "trend",
                "metric": metric_col,
                "dimension": strongest_feature,
                "entity": None,
                "direction": "up" if strongest_value >= 0 else "down",
                "base_confidence": 75
            })

    outlier_risks = detect_outlier_risk(df, safe_numeric_cols)
    if outlier_risks:
        top = outlier_risks[0]
        insights.append({
            "engine": "Statistical Engine",
            "title": "Outlier concentration",
            "statement": f"Outlier analysis shows that {top['column']} has the highest anomaly concentration at {top['outlier_pct']}%, which may distort averages and trends.",
            "insight_type": "risk",
            "metric": top["column"],
            "dimension": "distribution",
            "entity": None,
            "direction": "high",
            "base_confidence": 75
        })

    trend = month_trend_stats(df, date_col, metric_col)
    monthly = trend["monthly_series"]

    if len(monthly) >= 3:
        recent = monthly.tail(3).reset_index(drop=True)
        if recent.iloc[-1] < recent.iloc[0]:
            text = f"Recent monthly movement for {metric_col} shows a downward direction across the latest observed periods."
            direction = "down"
        elif recent.iloc[-1] > recent.iloc[0]:
            text = f"Recent monthly movement for {metric_col} shows an upward direction across the latest observed periods."
            direction = "up"
        else:
            text = f"Recent monthly movement for {metric_col} appears stable across the latest observed periods."
            direction = "none"

        insights.append({
            "engine": "Statistical Engine",
            "title": "Recent movement",
            "statement": text,
            "insight_type": "general",
            "metric": metric_col,
            "dimension": "time",
            "entity": "recent_months",
            "direction": direction,
            "base_confidence": 75
        })

    if cat_cols:
        try:
            col = cat_cols[0]
            grouped = df.groupby(col)[metric_col].sum().sort_values(ascending=False)
            if len(grouped) >= 2:
                spread = float(grouped.iloc[0] - grouped.iloc[-1])
                insights.append({
                    "engine": "Statistical Engine",
                    "title": "Category spread",
                    "statement": f"Category spread analysis indicates a {spread:,.2f} gap between the strongest and weakest values in {col}.",
                    "insight_type": "risk",
                    "metric": metric_col,
                    "dimension": col,
                    "entity": None,
                    "direction": "mixed",
                    "base_confidence": 75
                })
        except Exception:
            pass

    return insights[:5]


def build_narrative_insights(df: pd.DataFrame, metric_col: str, date_col: Optional[str], cat_cols: List[str], quality_score: int) -> List[Dict[str, Any]]:
    insights = []

    best_dim, best_entity, best_value = detect_best_dimension(df, metric_col, cat_cols)
    weak_dim, weak_entity, weak_value = detect_weakest_dimension(df, metric_col, cat_cols)
    trend = month_trend_stats(df, date_col, metric_col)

    if trend["pct_change"] is not None:
        if trend["pct_change"] > 0:
            insights.append({
                "engine": "Narrative Engine",
                "title": "Momentum",
                "statement": f"Narrative assessment suggests momentum is favorable because {metric_col} is trending upward over time, which may indicate healthy commercial performance.",
                "insight_type": "trend",
                "metric": metric_col,
                "dimension": "time",
                "entity": "monthly",
                "direction": "up",
                "base_confidence": 85
            })
        else:
            insights.append({
                "engine": "Narrative Engine",
                "title": "Caution",
                "statement": f"Narrative assessment suggests caution because {metric_col} is trending downward over time, which may reflect weakening demand or execution gaps.",
                "insight_type": "risk",
                "metric": metric_col,
                "dimension": "time",
                "entity": "monthly",
                "direction": "down",
                "base_confidence": 85
            })

    if best_dim and best_entity:
        insights.append({
            "engine": "Narrative Engine",
            "title": "Growth segment",
            "statement": f"From a business perspective, {best_entity} in {best_dim} appears to be a dependable growth segment that should be protected and scaled.",
            "insight_type": "recommendation",
            "metric": metric_col,
            "dimension": best_dim,
            "entity": best_entity,
            "direction": "high",
            "base_confidence": 75
        })

    if weak_dim and weak_entity:
        insights.append({
            "engine": "Narrative Engine",
            "title": "Weak segment",
            "statement": f"The weakest segment appears to be {weak_entity} in {weak_dim}, suggesting a need for targeted investigation and intervention.",
            "insight_type": "risk",
            "metric": metric_col,
            "dimension": weak_dim,
            "entity": weak_entity,
            "direction": "low",
            "base_confidence": 85
        })

    insights.append({
        "engine": "Narrative Engine",
        "title": "Confidence",
        "statement": f"Decision confidence is relatively strong because the data quality score is {quality_score}/100, supporting more reliable interpretation.",
        "insight_type": "general",
        "metric": None,
        "dimension": "dataset",
        "entity": "quality",
        "direction": "high",
        "base_confidence": 85
    })

    return insights[:5]


# =========================================================
# CONTRADICTION + BENCHMARK LOGIC
# =========================================================
def get_subject_key(insight: Dict[str, Any]) -> str:
    metric = normalize_text(insight.get("metric"))
    dimension = normalize_text(insight.get("dimension"))
    entity = normalize_text(insight.get("entity"))

    if dimension == "time":
        return f"{metric}|time"
    if metric and dimension and entity:
        return f"{metric}|{dimension}|{entity}"
    if metric and dimension:
        return f"{metric}|{dimension}"
    return f"{metric}|{normalize_text(insight.get('insight_type'))}"


def direction_bucket(direction: str) -> str:
    d = normalize_text(direction)
    if d in {"up", "high"}:
        return "positive"
    if d in {"down", "low"}:
        return "negative"
    return "neutral"


def are_contradictory(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if get_subject_key(a) != get_subject_key(b):
        return False

    da = direction_bucket(a.get("direction", "none"))
    db = direction_bucket(b.get("direction", "none"))

    if da == "neutral" or db == "neutral":
        return False

    return da != db


def detect_contradictions(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contradictions = []
    for i in range(len(insights)):
        for j in range(i + 1, len(insights)):
            a, b = insights[i], insights[j]
            if are_contradictory(a, b):
                contradictions.append({
                    "insight_a": a,
                    "insight_b": b,
                    "reason": "Same subject with opposite directional claims."
                })
    return contradictions


def score_insight(insight: Dict[str, Any], evidence_rows: pd.DataFrame, gemini_status: str) -> Dict[str, Any]:
    statement = normalize_text(insight.get("statement"))
    insight_type = normalize_text(insight.get("insight_type"))
    base_confidence = float(insight.get("base_confidence", 60))

    relevance = 75
    actionability = 35
    clarity = 80
    consistency = 4.0
    grounding = 80
    evidence_coverage = 80

    if "strongest" in statement or "correlation" in statement or "duplicate" in statement:
        relevance = 85
    elif insight_type == "general":
        relevance = 45
    elif "weakest" in statement or "underperformance" in statement:
        relevance = 55

    if insight_type == "recommendation":
        actionability = 80
    elif insight_type == "risk":
        actionability = 55

    if insight.get("engine") == "Narrative Engine":
        clarity = 95
    elif insight.get("engine") == "Gemini LLM Engine":
        clarity = 85

    if normalize_text(insight.get("direction")) == "mixed":
        consistency = 3.2
    elif normalize_text(insight.get("direction")) == "none":
        consistency = 2.5

    if evidence_rows is None or evidence_rows.empty:
        grounding = 55
        evidence_coverage = 45
    else:
        grounding = 90
        evidence_coverage = min(100, 60 + len(evidence_rows) * 8)

    llm_valid = "Yes" if gemini_status == "ok" else "Error"

    final_score = (
        0.18 * relevance +
        0.15 * actionability +
        0.12 * (consistency * 10) +
        0.12 * clarity +
        0.18 * base_confidence +
        0.15 * grounding +
        0.10 * evidence_coverage
    )

    if insight.get("engine") == "Gemini LLM Engine" and gemini_status != "ok":
        final_score -= 20

    insight["relevance"] = relevance
    insight["actionability"] = actionability
    insight["consistency"] = round(consistency, 1)
    insight["clarity"] = clarity
    insight["grounding"] = grounding
    insight["evidence_coverage"] = evidence_coverage
    insight["llm_valid"] = llm_valid
    insight["final_score"] = round(final_score, 1)

    return insight


def build_final_reliable_insights(insights: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contradiction_statements = set()
    for c in contradictions:
        contradiction_statements.add(c["insight_a"]["statement"])
        contradiction_statements.add(c["insight_b"]["statement"])

    output = []
    for insight in insights:
        penalty = 10 if insight["statement"] in contradiction_statements else 0
        adjusted = insight.get("final_score", 0) - penalty
        temp = dict(insight)
        temp["adjusted_final_score"] = round(adjusted, 1)
        output.append(temp)

    output = sorted(output, key=lambda x: x["adjusted_final_score"], reverse=True)
    return output[:5]


# =========================================================
# AGENT Q&A
# =========================================================
def classify_question(question: str) -> str:
    q = normalize_text(question)

    if any(word in q for word in ["top", "best", "highest", "strongest"]):
        return "top_segment"
    if any(word in q for word in ["worst", "weakest", "lowest", "underperform"]):
        return "weak_segment"
    if any(word in q for word in ["trend", "over time", "increase", "decrease", "growth", "decline"]):
        return "trend"
    if any(word in q for word in ["correlation", "relationship", "driver"]):
        return "correlation"
    if any(word in q for word in ["risk", "anomaly", "outlier", "issue"]):
        return "risk"
    return "general"


def answer_question_with_rules(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    date_col: Optional[str],
    cat_cols: List[str],
    safe_numeric_cols: List[str]
) -> Tuple[str, pd.DataFrame, int]:
    qtype = classify_question(question)

    if qtype == "top_segment":
        best_dim, best_entity, best_value = detect_best_dimension(df, metric_col, cat_cols)
        if best_dim and best_entity:
            evidence = retrieve_evidence_rows(df, metric_col, best_dim, best_entity, top_n=5)
            answer = f"The strongest segment is {best_entity} within {best_dim}, contributing {best_value:,.2f} to {metric_col}."
            return answer, evidence, 88

    if qtype == "weak_segment":
        weak_dim, weak_entity, weak_value = detect_weakest_dimension(df, metric_col, cat_cols)
        if weak_dim and weak_entity:
            evidence = retrieve_evidence_rows(df, metric_col, weak_dim, weak_entity, top_n=5)
            answer = f"The weakest segment is {weak_entity} within {weak_dim}, which contributes the lowest observed total to {metric_col}."
            return answer, evidence, 84

    if qtype == "trend":
        trend = month_trend_stats(df, date_col, metric_col)
        monthly = trend["monthly_series"]
        if len(monthly) >= 2:
            direction = "upward" if trend["pct_change"] and trend["pct_change"] > 0 else "downward"
            answer = f"The monthly trend for {metric_col} is {direction}, with a change of {abs(trend['pct_change']):.1f}% from the first observed month to the latest one."
            evidence = monthly.reset_index().tail(6)
            return answer, evidence, 86

    if qtype == "correlation":
        if metric_col in safe_numeric_cols:
            num_df = df[safe_numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr = num_df.corr(numeric_only=True)
            if metric_col in corr.columns:
                corr_series = corr[metric_col].drop(metric_col, errors="ignore").dropna()
                if not corr_series.empty:
                    top_feature = corr_series.abs().sort_values(ascending=False).index[0]
                    corr_val = corr.loc[top_feature, metric_col]
                    evidence = num_df[[metric_col, top_feature]].dropna().head(10)
                    answer = f"The strongest numeric relationship with {metric_col} is {top_feature}, with correlation {corr_val:.2f}."
                    return answer, evidence, 80

    if qtype == "risk":
        outlier_risks = detect_outlier_risk(df, safe_numeric_cols)
        if outlier_risks:
            top = outlier_risks[0]
            evidence = retrieve_evidence_rows(df, top["column"], None, None, top_n=5)
            answer = f"The biggest statistical risk is outlier concentration in {top['column']} at {top['outlier_pct']}%."
            return answer, evidence, 78

    evidence = df.head(5)
    answer = f"The current filtered dataset has {len(df):,} rows and the primary metric is {metric_col}, totaling {pd.to_numeric(df[metric_col], errors='coerce').fillna(0).sum():,.2f}."
    return answer, evidence, 65


# =========================================================
# DEMO DATA
# =========================================================
def make_demo_sales_dataset() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2009-12-01", "2010-12-09", freq="H")
    n = min(99263, len(dates))

    countries = ["United Kingdom", "Ireland", "Netherlands", "Germany", "France", "Sweden", "Denmark", "Spain", "Switzerland", "Australia", "Nigeria"]
    probs = np.array([0.47, 0.13, 0.10, 0.08, 0.06, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02])

    selected = np.random.choice(countries, size=n, p=probs)
    qty = np.random.poisson(2.5, n) + 1
    unit_price = np.round(np.random.gamma(2.0, 4.0, n), 2)
    sales = np.round(qty * unit_price, 2)

    sales[selected == "United Kingdom"] *= 1.8
    sales[selected == "Nigeria"] *= 0.45

    country_code_map = {
        "United Kingdom": "GBR", "Ireland": "IRL", "Netherlands": "NLD", "Germany": "DEU",
        "France": "FRA", "Sweden": "SWE", "Denmark": "DNK", "Spain": "ESP",
        "Switzerland": "CHE", "Australia": "AUS", "Nigeria": "NGA"
    }

    df = pd.DataFrame({
        "order_datetime": dates[:n],
        "country": selected,
        "country_code": [country_code_map[c] for c in selected],
        "customer_id": np.random.randint(12000, 18000, n),
        "product_id": np.random.randint(21000, 99999, n).astype(str),
        "unit_price_gbp": unit_price,
        "quantity_sold": qty,
        "sales_amount_gbp": sales
    })

    df["population_total"] = df["country"].map({
        "United Kingdom": 67e6, "Ireland": 5e6, "Netherlands": 17e6, "Germany": 83e6,
        "France": 65e6, "Sweden": 10e6, "Denmark": 5.8e6, "Spain": 47e6,
        "Switzerland": 8.7e6, "Australia": 25e6, "Nigeria": 206e6
    })
    df["gdp_current_usd"] = df["country"].map({
        "United Kingdom": 3.1e12, "Ireland": 5e11, "Netherlands": 1e12, "Germany": 4.3e12,
        "France": 3e12, "Sweden": 6e11, "Denmark": 4e11, "Spain": 1.4e12,
        "Switzerland": 8e11, "Australia": 1.7e12, "Nigeria": 4.3e11
    })
    df["gdp_growth_pct"] = np.round(np.random.normal(2.2, 1.2, n), 2)
    df["inflation_consumer_pct"] = np.round(np.random.normal(2.8, 1.0, n), 2)

    df = add_time_features(df, "order_datetime")

    dup_rows = df.sample(255, random_state=7)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=11).reset_index(drop=True)

    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Data Input")

    input_mode = st.radio(
        "Choose input mode",
        ["Upload your own file", "Use demo sales dataset"],
        index=0
    )

    st.subheader("Upload CSV or Excel")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    demo_csv = to_csv_download(make_demo_sales_dataset())
    st.download_button(
        "Download sample sales dataset",
        data=demo_csv,
        file_name="sample_sales_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# LOAD DATA
# =========================================================
df_raw = None

try:
    if input_mode == "Use demo sales dataset":
        df_raw = make_demo_sales_dataset()
    elif uploaded_file is not None:
        df_raw = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

if df_raw is None:
    st.title("Client Intelligence Platform")
    st.markdown("Upload a CSV/XLSX file or use demo mode.")
    st.stop()

df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]

date_col = infer_date_column(df)
df = add_time_features(df, date_col)
numeric_cols = infer_numeric_columns(df)
date_col = infer_date_column(df)
categorical_cols = infer_categorical_columns(df, numeric_cols, date_col)
primary_metric = infer_primary_metric(df, numeric_cols)

if primary_metric is None:
    st.error("No usable numeric primary metric found.")
    st.stop()


# =========================================================
# FILTERS
# =========================================================
filtered_df = df.copy()

with st.sidebar:
    st.header("Filters")

    if date_col and date_col in filtered_df.columns:
        try:
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")
            min_date = filtered_df[date_col].min()
            max_date = filtered_df[date_col].max()

            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input("Date range", (min_date.date(), max_date.date()))
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df[date_col].dt.date >= start_date) &
                        (filtered_df[date_col].dt.date <= end_date)
                    ]
        except Exception:
            pass

    for col in categorical_cols[:5]:
        try:
            values = sorted([str(v) for v in filtered_df[col].dropna().unique()])
            if 1 < len(values) <= 50:
                chosen = st.multiselect(f"Filter {col}", values, default=[])
                if chosen:
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(chosen)]
        except Exception:
            continue

if filtered_df.empty:
    st.error("No rows left after filtering.")
    st.stop()


# =========================================================
# RE-COMPUTE AFTER FILTERING
# =========================================================
safe_numeric_cols = get_safe_numeric_columns(filtered_df, numeric_cols)
if primary_metric not in safe_numeric_cols and primary_metric in filtered_df.columns:
    safe_numeric_cols.append(primary_metric)

quality_score = compute_data_quality_score(filtered_df)
rows = len(filtered_df)
cols = len(filtered_df.columns)
missing_values = int(filtered_df.isna().sum().sum())
duplicate_count = int(filtered_df.duplicated().sum())

metric_series = pd.to_numeric(filtered_df[primary_metric], errors="coerce").fillna(0)
total_metric = metric_series.sum()
avg_metric = metric_series.mean()

best_dim, best_entity, best_value = detect_best_dimension(filtered_df, primary_metric, categorical_cols)
weak_dim, weak_entity, weak_value = detect_weakest_dimension(filtered_df, primary_metric, categorical_cols)
trend_stats = month_trend_stats(filtered_df, date_col, primary_metric)
outlier_risks = detect_outlier_risk(filtered_df, safe_numeric_cols)
imbalance_alerts = detect_category_imbalance(filtered_df, categorical_cols)


# =========================================================
# HEADER + KPI
# =========================================================
st.title("Client Intelligence Platform")
st.markdown(
    "Upload a CSV/XLSX file or try demo mode to generate KPI dashboards, trend analysis, anomaly detection, recommendations, multi-engine insight evaluation, retrieval-backed evidence, and agent-style Q&A."
)

info_cols = st.columns(4)
with info_cols[0]:
    st.info("**Business Problem** Teams often receive raw data with little clarity on what matters most.")
with info_cols[1]:
    st.info("**What this platform does** Converts raw data into summaries, KPIs, charts, risks, and actions.")
with info_cols[2]:
    st.info("**Perplexity-style upgrade** Retrieval-backed evidence, source snippets, benchmark scoring, and agent Q&A.")
with info_cols[3]:
    st.info("**How to use it** Upload data, filter it, review insights, inspect evidence, and ask grounded questions.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{rows:,}")
k2.metric("Columns", cols)
k3.metric("Missing Values", missing_values)
k4.metric("Duplicates", duplicate_count)
k5.metric("Data Quality Score", f"{quality_score}/100")

k6, k7, k8 = st.columns(3)
k6.metric(f"Total {primary_metric}", f"{total_metric:,.2f}")
k7.metric(f"Average {primary_metric}", f"{avg_metric:,.2f}")
k8.metric("Best Segment", best_entity if best_entity else "N/A")


# =========================================================
# SUMMARY + RISKS
# =========================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("Executive Summary")

    parts = [
        f"The dataset contains {rows:,} records across {cols} columns.",
        f"The primary business metric appears to be **{primary_metric}**, with a total value of **{total_metric:,.2f}**."
    ]

    if trend_stats["pct_change"] is not None:
        if trend_stats["pct_change"] > 0:
            parts.append(f"Over time, **{primary_metric}** has **improved by {abs(trend_stats['pct_change']):.1f}%** from the first observed month to the latest one.")
        else:
            parts.append(f"Over time, **{primary_metric}** has **declined by {abs(trend_stats['pct_change']):.1f}%** from the first observed month to the latest one.")

    if best_dim and best_entity:
        parts.append(f"The strongest contribution comes from **{best_entity}** in **{best_dim}**, contributing **{best_value:,.2f}**.")
    if weak_dim and weak_entity:
        parts.append(f"The weakest contribution appears in **{weak_entity}** under **{weak_dim}**.")
    parts.append(f"From a data quality perspective, the file contains **{missing_values} missing values** and **{duplicate_count} duplicate rows**.")

    st.markdown(f"<div class='card'>{' '.join(parts)}</div>", unsafe_allow_html=True)

with right:
    st.subheader("Risk Alerts")

    if duplicate_count > 0:
        st.warning(f"Duplicate-record risk detected: {duplicate_count} duplicate rows found.")

    for item in outlier_risks[:3]:
        st.info(f"Outlier concentration in **{item['column']}** is elevated at {item['outlier_pct']}%.")

    for item in imbalance_alerts[:3]:
        st.info(f"Category imbalance found in **{item['column']}**: one value accounts for {item['top_share']}% of rows.")


# =========================================================
# RECOMMENDATIONS
# =========================================================
st.subheader("Recommendations")
recs = []

if duplicate_count > 0:
    recs.append("Remove duplicate records before external reporting, as duplicates can distort totals and conclusions.")
if best_dim and best_entity:
    recs.append(f"Protect and replicate the strongest-performing segment: **{best_entity} in {best_dim}**.")
if weak_dim and weak_entity:
    recs.append(f"Investigate the weakest area: **{weak_entity} in {weak_dim}** may need operational or retention review.")
if trend_stats["pct_change"] is not None:
    recs.append("Track the primary KPI monthly so sudden drops or unusual spikes are detected earlier.")

try:
    if primary_metric in safe_numeric_cols:
        corr_df = filtered_df[safe_numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr_series = corr_df.corr(numeric_only=True)[primary_metric].drop(primary_metric, errors="ignore").dropna()
        if not corr_series.empty:
            top_driver = corr_series.abs().sort_values(ascending=False).index[0]
            top_corr = corr_df.corr(numeric_only=True).loc[top_driver, primary_metric]
            recs.append(f"Review the relationship between **{top_driver}** and **{primary_metric}** (correlation {top_corr:.2f}) to identify controllable drivers.")
except Exception:
    pass

for i, rec in enumerate(recs[:5], start=1):
    st.markdown(f"**{i}.** {rec}")


# =========================================================
# ENGINES
# =========================================================
rule_based_insights = build_rule_based_insights(filtered_df, primary_metric, date_col, categorical_cols)
statistical_insights = build_statistical_insights(filtered_df, primary_metric, date_col, safe_numeric_cols, categorical_cols)
narrative_insights = build_narrative_insights(filtered_df, primary_metric, date_col, categorical_cols, quality_score)

dataset_context = build_dataset_context(filtered_df, primary_metric, date_col, categorical_cols, safe_numeric_cols)
gemini_result = run_gemini_engine(dataset_context)
gemini_status = gemini_result["status"]
gemini_error = gemini_result["error"]
gemini_insights = gemini_result["insights"]

if gemini_status != "ok" or not gemini_insights:
    gemini_insights = [{
        "engine": "Gemini LLM Engine",
        "title": "Gemini unavailable",
        "statement": "Gemini was unavailable in this run, so final ranking should be interpreted without successful LLM validation.",
        "insight_type": "general",
        "metric": primary_metric,
        "dimension": None,
        "entity": None,
        "direction": "none",
        "evidence": "Fallback applied because Gemini did not return valid structured output.",
        "base_confidence": 35
    }]

all_engine_insights = rule_based_insights + statistical_insights + narrative_insights + gemini_insights


# =========================================================
# EVIDENCE RETRIEVAL + SOURCE SNIPPETS
# =========================================================
evidence_map = {}

for insight in all_engine_insights:
    if not insight.get("evidence"):
        insight["evidence"] = build_evidence_text(insight, filtered_df, primary_metric)

    evidence_rows = retrieve_evidence_rows(
        filtered_df,
        metric_col=primary_metric if insight.get("metric") is None else insight.get("metric"),
        dimension=insight.get("dimension"),
        entity=insight.get("entity"),
        top_n=5
    )

    evidence_key = insight["engine"] + "||" + insight["statement"]
    evidence_map[evidence_key] = evidence_rows

    score_insight(insight, evidence_rows, gemini_status)


# =========================================================
# INSIGHT EVALUATION LAYER
# =========================================================
st.subheader("Insight Evaluation Layer")
st.caption("This section compares how different reasoning engines generate and score insights from the same filtered dataset.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Rule-Based Engine",
    "Statistical Engine",
    "Narrative Engine",
    "Gemini LLM Engine"
])

def render_engine_tab(insights: List[Dict[str, Any]]):
    for idx, insight in enumerate(insights, start=1):
        st.markdown(f"**{idx}. {insight['statement']}**")
        st.markdown(f"**Evidence:** {insight['evidence']}")
        st.markdown(f"**Base confidence:** {insight['base_confidence']}")
        key = insight["engine"] + "||" + insight["statement"]
        rows_df = evidence_map.get(key, pd.DataFrame())
        if rows_df is not None and not rows_df.empty:
            with st.expander("View source snippets"):
                st.dataframe(rows_df, use_container_width=True)
        st.markdown("---")

with tab1:
    render_engine_tab(rule_based_insights)

with tab2:
    render_engine_tab(statistical_insights)

with tab3:
    render_engine_tab(narrative_insights)

with tab4:
    if gemini_status == "ok":
        render_engine_tab(gemini_insights)
    else:
        st.warning(f"Gemini unavailable in this run: {gemini_error}")
        render_engine_tab(gemini_insights)


# =========================================================
# BENCHMARK TABLE
# =========================================================
st.subheader("Benchmark Evaluation Table")

benchmark_df = pd.DataFrame([
    {
        "Engine": x["engine"],
        "Insight Type": x["insight_type"].title(),
        "Insight": x["statement"],
        "Relevance": x["relevance"],
        "Actionability": x["actionability"],
        "Consistency": x["consistency"],
        "Clarity": x["clarity"],
        "Grounding": x["grounding"],
        "Evidence Coverage": x["evidence_coverage"],
        "Base Confidence": x["base_confidence"],
        "LLM Valid": x["llm_valid"],
        "Final Score": x["final_score"]
    }
    for x in all_engine_insights
]).sort_values(by="Final Score", ascending=False)

st.dataframe(benchmark_df, use_container_width=True, hide_index=True)


# =========================================================
# CONTRADICTIONS
# =========================================================
contradictions = detect_contradictions(all_engine_insights)

if contradictions:
    st.subheader("Contradiction Detection")
    for item in contradictions:
        st.error(
            f"Potential contradiction:\n\n"
            f"- {item['insight_a']['statement']}\n"
            f"- {item['insight_b']['statement']}\n\n"
            f"Reason: {item['reason']}"
        )


# =========================================================
# FINAL RELIABLE INSIGHTS
# =========================================================
final_reliable_insights = build_final_reliable_insights(all_engine_insights, contradictions)

st.subheader("Grounded Final Verdict")
for idx, insight in enumerate(final_reliable_insights, start=1):
    st.markdown(f"**{idx}. {insight['statement']}**")
    st.markdown(f"- **Engine:** {insight['engine']}")
    st.markdown(f"- **Adjusted Final Score:** {insight['adjusted_final_score']}")
    st.markdown(f"- **Evidence:** {insight['evidence']}")
    key = insight["engine"] + "||" + insight["statement"]
    rows_df = evidence_map.get(key, pd.DataFrame())
    if rows_df is not None and not rows_df.empty:
        with st.expander("Inspect evidence rows"):
            st.dataframe(rows_df, use_container_width=True)
    st.markdown("---")


# =========================================================
# ENGINE SCORE CHART
# =========================================================
engine_score_df = pd.DataFrame(all_engine_insights).groupby("engine", as_index=False)["final_score"].mean()
engine_score_df.columns = ["Engine", "Final Score"]
engine_score_df = engine_score_df.sort_values(by="Final Score", ascending=False)

st.subheader("Average Insight Score by Engine")
fig_engine = px.bar(engine_score_df, x="Engine", y="Final Score")
st.plotly_chart(fig_engine, use_container_width=True)

if not engine_score_df.empty:
    top_engine = engine_score_df.iloc[0]["Engine"]
    top_score = engine_score_df.iloc[0]["Final Score"]
    st.success(f"Top-performing engine in the current run: **{top_engine}** with an average score of **{top_score:.1f}**.")


# =========================================================
# AGENT-STYLE Q&A
# =========================================================
st.subheader("Ask Your Data")
question = st.text_input("Ask a grounded question about the current filtered dataset")

if question:
    rule_answer, rule_evidence, rule_conf = answer_question_with_rules(
        question, filtered_df, primary_metric, date_col, categorical_cols, safe_numeric_cols
    )

    st.markdown(f"**Agent answer:** {rule_answer}")
    st.markdown(f"**Confidence:** {rule_conf}")
    if rule_evidence is not None and not rule_evidence.empty:
        with st.expander("Supporting evidence rows"):
            st.dataframe(rule_evidence, use_container_width=True)

    ask_result = run_gemini_engine(dataset_context, question)
    if ask_result["status"] == "ok" and ask_result["insights"]:
        st.markdown("**Gemini grounded answer:**")
        st.info(ask_result["insights"][0]["statement"])
        st.caption(ask_result["insights"][0].get("evidence", "Grounded from current filtered dataset."))
    else:
        st.caption("Gemini grounded answer unavailable in this run.")


# =========================================================
# VISUALS
# =========================================================
st.subheader("Recommended Visualization")

if date_col and primary_metric:
    st.caption(f"Recommended chart: line chart because {date_col} is a time field and {primary_metric} is numeric.")
    try:
        temp = filtered_df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])

        if not temp.empty:
            trend_df = temp.groupby(date_col, as_index=False)[primary_metric].sum().sort_values(by=date_col)
            fig_trend = px.line(trend_df, x=date_col, y=primary_metric, title=f"{primary_metric} over time")
            st.plotly_chart(fig_trend, use_container_width=True)
    except Exception:
        st.info("Trend chart unavailable for this dataset.")

c1, c2 = st.columns(2)

with c1:
    try:
        if best_dim and best_dim in filtered_df.columns:
            grouped = filtered_df.groupby(best_dim, as_index=False)[primary_metric].sum().sort_values(by=primary_metric, ascending=False).head(10)
            fig_bar = px.bar(grouped, x=best_dim, y=primary_metric, title=f"Top {best_dim} by {primary_metric}")
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        st.info("Top-segment chart unavailable.")

with c2:
    try:
        fig_hist = px.histogram(filtered_df, x=primary_metric, nbins=30, title=f"Distribution of {primary_metric}")
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception:
        st.info("Distribution chart unavailable.")


# =========================================================
# CORRELATION HEATMAP
# =========================================================
if len(safe_numeric_cols) >= 2:
    st.subheader("Correlation Heatmap")
    st.markdown("**Numeric Correlation Matrix**")
    try:
        corr_matrix = filtered_df[safe_numeric_cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
        if not corr_matrix.empty and corr_matrix.shape[0] >= 2:
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        st.info("Correlation heatmap unavailable.")


# =========================================================
# PREVIEW + DOWNLOADS
# =========================================================
st.subheader("Preview Data")
st.dataframe(filtered_df.head(50), use_container_width=True)

d1, d2, d3 = st.columns(3)

with d1:
    st.download_button(
        "Download filtered dataset as CSV",
        data=to_csv_download(filtered_df),
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )

with d2:
    insights_df = pd.DataFrame([
        {
            "engine": x["engine"],
            "title": x["title"],
            "statement": x["statement"],
            "insight_type": x["insight_type"],
            "metric": x.get("metric"),
            "dimension": x.get("dimension"),
            "entity": x.get("entity"),
            "direction": x.get("direction"),
            "evidence": x.get("evidence"),
            "base_confidence": x.get("base_confidence"),
            "final_score": x.get("final_score")
        }
        for x in all_engine_insights
    ])

    st.download_button(
        "Download insights report as CSV",
        data=to_csv_download(insights_df),
        file_name="insights_report.csv",
        mime="text/csv"
    )

with d3:
    st.download_button(
        "Download evaluation table as CSV",
        data=to_csv_download(benchmark_df),
        file_name="benchmark_evaluation.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Next research-grade upgrade: row-level retrieval index, structured evidence citations, prompt benchmarking across datasets, and failure-mode dashboard.")