import io
import json
import math
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional Gemini support
try:
    import google.generativeai as genai
except Exception:
    genai = None


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Client Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 18px 20px;
            min-height: 108px;
        }
        .insight-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .hero-card {
            background: rgba(59,130,246,0.12);
            border: 1px solid rgba(59,130,246,0.2);
            border-radius: 18px;
            padding: 18px 20px;
            min-height: 132px;
        }
        .small-muted {
            color: #9ca3af;
            font-size: 0.92rem;
        }
        .section-divider {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        .good-box {
            background: rgba(34,197,94,0.15);
            border: 1px solid rgba(34,197,94,0.22);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .warn-box {
            background: rgba(245,158,11,0.15);
            border: 1px solid rgba(245,158,11,0.22);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .danger-box {
            background: rgba(239,68,68,0.15);
            border: 1px solid rgba(239,68,68,0.22);
            border-radius: 14px;
            padding: 14px 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================
def safe_str(x: Any) -> str:
    try:
        if pd.isna(x):
            return "Missing"
    except Exception:
        pass
    return str(x)


def to_label(s: str) -> str:
    return s.replace("_", " ").strip().title()


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def clean_column_name(col: str) -> str:
    return str(col).strip().replace("\n", " ")


def coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        col_l = str(col).lower()
        if any(k in col_l for k in ["date", "time", "timestamp", "datetime"]):
            converted = pd.to_datetime(out[col], errors="coerce")
            if converted.notna().mean() >= 0.6:
                out[col] = converted
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            cleaned = (
                out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() >= 0.8:
                out[col] = converted
    return out


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    df.columns = [clean_column_name(c) for c in df.columns]
    df = coerce_datetime_columns(df)
    df = coerce_numeric_columns(df)
    return df


def make_demo_sales_dataset(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2009-12-01", "2010-12-09", freq="h")

    product_types = ["cosmetics", "fashion", "electronics", "groceries", "home decor"]
    customer_demographics = ["Female", "Male", "Non-binary", "Unknown"]
    shipping_carriers = ["DHL", "FedEx", "UPS", "Royal Mail"]
    suppliers = [f"Supplier {i}" for i in range(1, 11)]
    locations = ["UK", "Ireland", "Germany", "France", "Netherlands", "Spain"]

    rows = []
    for i in range(n):
        pt = rng.choice(product_types, p=[0.28, 0.22, 0.2, 0.18, 0.12])
        price = round(float(rng.uniform(2, 95)), 4)
        availability = int(rng.integers(0, 100))
        sold = int(rng.integers(10, 1000))
        lead_times = int(rng.integers(1, 30))
        shipping_times = int(max(1, lead_times + rng.integers(-4, 5)))
        order_qty = int(rng.integers(5, 100))
        stock_levels = int(rng.integers(0, 100))
        shipping_costs = round(float(rng.uniform(2, 25)), 4)
        manufacturing_costs = round(float(price * rng.uniform(0.22, 0.65)), 4)
        costs = round(float(shipping_costs + manufacturing_costs * order_qty * rng.uniform(0.6, 1.4)), 4)
        revenue = round(float(price * sold * rng.uniform(0.35, 0.9)), 4)
        rows.append(
            {
                "Order datetime": rng.choice(dates),
                "Product type": pt,
                "SKU": f"SKU{i+1}",
                "Price": price,
                "Availability": availability,
                "Number of products sold": sold,
                "Revenue generated": revenue,
                "Customer demographics": rng.choice(customer_demographics),
                "Stock levels": stock_levels,
                "Lead times": lead_times,
                "Order quantities": order_qty,
                "Shipping times": shipping_times,
                "Shipping carriers": rng.choice(shipping_carriers),
                "Shipping costs": shipping_costs,
                "Supplier name": rng.choice(suppliers),
                "Location": rng.choice(locations),
                "Lead time": int(rng.integers(1, 25)),
                "Production volumes": int(rng.integers(50, 1500)),
                "Manufacturing lead time": int(rng.integers(1, 20)),
                "Manufacturing costs": manufacturing_costs,
                "Inspection results": rng.choice(["Pass", "Review", "Pass", "Pass"]),
                "Defect rates": round(float(rng.uniform(0.0, 0.12)), 5),
                "Transportation modes": rng.choice(["Road", "Air", "Sea"]),
                "Costs": costs,
            }
        )

    demo = pd.DataFrame(rows)
    demo = coerce_datetime_columns(demo)
    demo = coerce_numeric_columns(demo)
    return demo


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if is_datetime_series(df[c])]
    if candidates:
        return candidates[0]

    keyword_candidates = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ["date", "time", "timestamp", "datetime"])
    ]
    for c in keyword_candidates:
        converted = pd.to_datetime(df[c], errors="coerce")
        if converted.notna().mean() > 0.6:
            df[c] = converted
            return c
    return None


def detect_primary_metric(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
    if not numeric_cols:
        return None

    preferred_keywords = [
        "revenue", "sales", "amount", "profit", "income", "gmv", "total", "value"
    ]
    for key in preferred_keywords:
        for c in numeric_cols:
            if key in str(c).lower():
                return c

    best = max(numeric_cols, key=lambda c: df[c].fillna(0).sum())
    return best


def detect_best_category_column(df: pd.DataFrame, metric: Optional[str]) -> Optional[str]:
    object_cols = [
        c for c in df.columns
        if not is_numeric_series(df[c]) and not is_datetime_series(df[c])
    ]
    if not object_cols:
        return None

    preferred = [
        "product", "category", "segment", "platform", "country", "region",
        "industry", "supplier", "carrier", "location", "type"
    ]
    for key in preferred:
        for c in object_cols:
            if key in str(c).lower():
                return c

    if metric:
        candidate_scores = []
        for c in object_cols:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 20:
                grouped = df.groupby(c, dropna=False)[metric].sum(min_count=1)
                spread = float(grouped.max() - grouped.min()) if len(grouped) else 0.0
                candidate_scores.append((c, spread))
        if candidate_scores:
            return max(candidate_scores, key=lambda x: x[1])[0]

    return object_cols[0]


def detect_weakest_category_value(df: pd.DataFrame, category_col: str, metric: str) -> Optional[str]:
    grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values()
    if grouped.empty:
        return None
    return safe_str(grouped.index[0])


def detect_best_category_value(df: pd.DataFrame, category_col: str, metric: str) -> Optional[str]:
    grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False)
    if grouped.empty:
        return None
    return safe_str(grouped.index[0])


def safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    try:
        val = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def detect_duplicate_risk(df: pd.DataFrame) -> Dict[str, Any]:
    duplicates = int(df.duplicated().sum())
    return {
        "duplicate_rows": duplicates,
        "has_duplicate_risk": duplicates > 0,
        "message": f"Duplicate-record risk detected: {duplicates} duplicate rows found." if duplicates > 0 else "No duplicate-record risk detected.",
    }


def detect_outlier_risk(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        if len(s) < 8:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)
        pct = float(outlier_mask.mean() * 100)
        if pct >= 5:
            risks.append({
                "column": col,
                "outlier_pct": round(pct, 1),
                "message": f"Outlier concentration in {to_label(col)} is elevated at {round(pct, 1)}%.",
            })
    return sorted(risks, key=lambda x: x["outlier_pct"], reverse=True)


def category_imbalance_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    risks = []
    for col in df.columns:
        if is_numeric_series(df[col]) or is_datetime_series(df[col]):
            continue
        vc = df[col].astype(str).value_counts(dropna=False, normalize=True)
        if vc.empty:
            continue
        top_share = float(vc.iloc[0] * 100)
        if top_share >= 95:
            risks.append(
                {
                    "column": col,
                    "top_share": round(top_share, 1),
                    "message": f"Category imbalance found in {to_label(col)}: one value accounts for {round(top_share,1)}% of rows.",
                }
            )
    return risks


def add_time_features(df: pd.DataFrame, dt_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if dt_col and dt_col in out.columns and is_datetime_series(out[dt_col]):
        out["year"] = out[dt_col].dt.year
        out["month"] = out[dt_col].dt.month
        out["week_of_year"] = out[dt_col].dt.isocalendar().week.astype("int64")
        out["day_of_week"] = out[dt_col].dt.dayofweek
        out["order_hour"] = out[dt_col].dt.hour
        out["is_weekend"] = (out[dt_col].dt.dayofweek >= 5).astype(int)
    return out


def quality_score(df: pd.DataFrame) -> int:
    score = 100
    missing_ratio = float(df.isna().mean().mean()) if len(df) else 0.0
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0
    score -= int(min(35, missing_ratio * 100))
    score -= int(min(20, duplicate_ratio * 100))
    return max(1, min(100, score))


def choose_visualization(df: pd.DataFrame, dt_col: Optional[str], metric: Optional[str], category_col: Optional[str]) -> Dict[str, Any]:
    if metric is None:
        return {"kind": "none", "reason": "No numeric KPI column was detected."}
    if dt_col:
        return {"kind": "line", "x": dt_col, "y": metric, "reason": f"Recommended chart: line chart because {dt_col} is a time field and {metric} is numeric."}
    if category_col:
        return {"kind": "bar", "x": category_col, "y": metric, "reason": f"Recommended chart: bar chart because {category_col} is categorical and {metric} is numeric."}
    return {"kind": "hist", "x": metric, "reason": f"Recommended chart: histogram because {metric} is numeric and no strong time field was detected."}


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_snippet_rows(df: pd.DataFrame, mask: pd.Series, max_rows: int = 5) -> pd.DataFrame:
    subset = df.loc[mask].copy()
    if subset.empty:
        return subset
    return subset.head(max_rows)


def compact_summary_from_mask(df: pd.DataFrame, mask: pd.Series, metric: Optional[str], group_col: Optional[str], group_val: Optional[str]) -> str:
    subset = df.loc[mask]
    rows = len(subset)
    pieces = [f"rows={rows}"]
    if group_col is not None and group_val is not None:
        pieces.insert(0, f"{group_col}={group_val}")
    if metric and metric in subset.columns and is_numeric_series(subset[metric]):
        total_val = float(pd.to_numeric(subset[metric], errors="coerce").fillna(0).sum())
        mean_val = float(pd.to_numeric(subset[metric], errors="coerce").fillna(0).mean()) if rows else 0.0
        pieces.append(f"total_{metric}={round(total_val, 2)}")
        pieces.append(f"avg_{metric}={round(mean_val, 2)}")
    return ", ".join(pieces)


@dataclass
class Insight:
    engine: str
    insight_type: str
    insight: str
    relevance: int
    actionability: int
    consistency: float
    clarity: int
    evidence: str
    snippet_mask_name: Optional[str] = None
    base_confidence: int = 75
    adjusted_final_score: float = 0.0


# ============================================================
# Gemini
# ============================================================
def get_gemini_model() -> Optional[Any]:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def call_gemini_json(prompt: str) -> Optional[Dict[str, Any]]:
    model = get_gemini_model()
    if model is None:
        return None
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return json.loads(text)
    except Exception:
        return None


# ============================================================
# Insight engines
# ============================================================
def rule_based_engine(df: pd.DataFrame, metric: Optional[str], dt_col: Optional[str], category_col: Optional[str], snippets: Dict[str, pd.Series]) -> List[Insight]:
    insights: List[Insight] = []
    if metric and category_col:
        grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False)
        if len(grouped) > 0:
            best_val = safe_str(grouped.index[0])
            best_total = float(grouped.iloc[0])
            mask = df[category_col].astype(str) == best_val
            snippets[f"best_{category_col}"] = mask
            insights.append(
                Insight(
                    engine="Rule-Based Engine",
                    insight_type="Trend",
                    insight=f"The strongest driver of {metric} is {best_val} within {to_label(category_col)}, contributing {best_total:,.2f} overall.",
                    relevance=85,
                    actionability=35,
                    consistency=4.0,
                    clarity=80,
                    evidence=compact_summary_from_mask(df, mask, metric, category_col, best_val),
                    snippet_mask_name=f"best_{category_col}",
                    base_confidence=88,
                )
            )

        if len(grouped) > 1:
            weak_val = safe_str(grouped.index[-1])
            mask = df[category_col].astype(str) == weak_val
            snippets[f"weak_{category_col}"] = mask
            insights.append(
                Insight(
                    engine="Rule-Based Engine",
                    insight_type="Recommendation",
                    insight=f"Underperformance appears in {weak_val} under {to_label(category_col)}, making it a priority area for review.",
                    relevance=55,
                    actionability=80,
                    consistency=4.0,
                    clarity=80,
                    evidence=compact_summary_from_mask(df, mask, metric, category_col, weak_val),
                    snippet_mask_name=f"weak_{category_col}",
                    base_confidence=84,
                )
            )

    duplicate_info = detect_duplicate_risk(df)
    if duplicate_info["has_duplicate_risk"]:
        snippets["duplicates_all"] = pd.Series([True] * len(df), index=df.index)
        insights.append(
            Insight(
                engine="Rule-Based Engine",
                insight_type="Risk",
                insight=f"Duplicate-record risk detected: {duplicate_info['duplicate_rows']} duplicate rows should be reviewed before external reporting.",
                relevance=75,
                actionability=80,
                consistency=4.0,
                clarity=85,
                evidence=f"dataset_rows={len(df)}, duplicate_rows={duplicate_info['duplicate_rows']}",
                snippet_mask_name="duplicates_all",
                base_confidence=95,
            )
        )

    if metric and dt_col and is_datetime_series(df[dt_col]):
        temp = df[[dt_col, metric]].dropna().copy()
        if len(temp) >= 3:
            temp["period"] = temp[dt_col].dt.to_period("M").dt.to_timestamp()
            monthly = temp.groupby("period")[metric].sum(min_count=1)
            if len(monthly) >= 2 and monthly.iloc[0] != 0:
                pct = ((monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0])) * 100
                trend_word = "improved" if pct >= 0 else "declined"
                snippets["time_all"] = pd.Series([True] * len(df), index=df.index)
                insights.append(
                    Insight(
                        engine="Rule-Based Engine",
                        insight_type="Trend",
                        insight=f"Over time, {metric} has {trend_word} by {abs(pct):.1f}% from the first month to the latest month.",
                        relevance=75,
                        actionability=35,
                        consistency=4.0,
                        clarity=80,
                        evidence=f"monthly_points={len(monthly)}, first={round(float(monthly.iloc[0]),2)}, last={round(float(monthly.iloc[-1]),2)}",
                        snippet_mask_name="time_all",
                        base_confidence=90,
                    )
                )
    return insights


def statistical_engine(df: pd.DataFrame, metric: Optional[str], numeric_cols: List[str], snippets: Dict[str, pd.Series]) -> List[Insight]:
    insights: List[Insight] = []
    if metric:
        candidates = []
        for c in numeric_cols:
            if c == metric:
                continue
            corr_val = safe_corr(pd.to_numeric(df[c], errors="coerce"), pd.to_numeric(df[metric], errors="coerce"))
            if corr_val is not None:
                candidates.append((c, corr_val))
        if candidates:
            best_col, best_corr = max(candidates, key=lambda x: abs(x[1]))
            snippets[f"corr_{best_col}"] = pd.Series([True] * len(df), index=df.index)
            relation = "positive" if best_corr >= 0 else "negative"
            insights.append(
                Insight(
                    engine="Statistical Engine",
                    insight_type="Trend",
                    insight=f"The strongest statistical relationship is a {relation} correlation of {best_corr:.2f} between {to_label(best_col)} and {to_label(metric)}.",
                    relevance=85,
                    actionability=35,
                    consistency=4.0,
                    clarity=80,
                    evidence=f"correlation_pair={best_col}|{metric}, correlation={round(best_corr,4)}",
                    snippet_mask_name=f"corr_{best_col}",
                    base_confidence=82,
                )
            )

    outliers = detect_outlier_risk(df, numeric_cols)
    if outliers:
        top = outliers[0]
        col = top["column"]
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        snippets[f"outlier_{col}"] = mask.fillna(False)
        insights.append(
            Insight(
                engine="Statistical Engine",
                insight_type="Risk",
                insight=f"Outlier analysis shows that {to_label(col)} has the highest anomaly concentration at {top['outlier_pct']}%, which may distort averages and trends.",
                relevance=75,
                actionability=55,
                consistency=4.0,
                clarity=80,
                evidence=f"rows={len(df)}, sum_{col}={round(float(s.fillna(0).sum()),2)}, mean_{col}={round(float(s.mean()),2) if len(s.dropna()) else 0.0}",
                snippet_mask_name=f"outlier_{col}",
                base_confidence=80,
            )
        )
    return insights


def narrative_engine(df: pd.DataFrame, metric: Optional[str], category_col: Optional[str], snippets: Dict[str, pd.Series]) -> List[Insight]:
    insights: List[Insight] = []
    if metric and category_col:
        grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False)
        if len(grouped) > 0:
            best_val = safe_str(grouped.index[0])
            mask = df[category_col].astype(str) == best_val
            snippets[f"narrative_best_{category_col}"] = mask
            insights.append(
                Insight(
                    engine="Narrative Engine",
                    insight_type="Recommendation",
                    insight=f"From a business perspective, {best_val} in {to_label(category_col)} appears to be a dependable growth segment that should be protected and scaled.",
                    relevance=75,
                    actionability=80,
                    consistency=4.0,
                    clarity=95,
                    evidence=compact_summary_from_mask(df, mask, metric, category_col, best_val),
                    snippet_mask_name=f"narrative_best_{category_col}",
                    base_confidence=85,
                )
            )
        if len(grouped) > 1:
            weak_val = safe_str(grouped.index[-1])
            mask = df[category_col].astype(str) == weak_val
            snippets[f"narrative_weak_{category_col}"] = mask
            insights.append(
                Insight(
                    engine="Narrative Engine",
                    insight_type="Risk",
                    insight=f"The weakest segment appears to be {weak_val} in {to_label(category_col)}, suggesting a need for targeted investigation and intervention.",
                    relevance=55,
                    actionability=55,
                    consistency=4.0,
                    clarity=95,
                    evidence=compact_summary_from_mask(df, mask, metric, category_col, weak_val),
                    snippet_mask_name=f"narrative_weak_{category_col}",
                    base_confidence=82,
                )
            )

    score = quality_score(df)
    snippets["quality_all"] = pd.Series([True] * len(df), index=df.index)
    insights.append(
        Insight(
            engine="Narrative Engine",
            insight_type="General",
            insight=f"Decision confidence is relatively strong because the data quality score is {score}/100, supporting more reliable interpretation.",
            relevance=45,
            actionability=35,
            consistency=4.0,
            clarity=95,
            evidence=f"rows={len(df)}, quality_score={score}",
            snippet_mask_name="quality_all",
            base_confidence=78,
        )
    )
    return insights


def gemini_engine(df: pd.DataFrame, metric: Optional[str], category_col: Optional[str], snippets: Dict[str, pd.Series]) -> List[Insight]:
    prompt = textwrap.dedent(
        f"""
        You are analyzing a filtered business dataset.
        Return strict JSON with this schema:
        {{
          "available": true,
          "insights": [
            {{
              "insight_type": "General|Trend|Risk|Recommendation",
              "insight": "text",
              "relevance": 45,
              "actionability": 35,
              "clarity": 85,
              "base_confidence": 70
            }}
          ]
        }}

        Context:
        rows={len(df)}
        columns={list(df.columns)}
        metric={metric}
        category_col={category_col}
        numeric_summary={df.select_dtypes(include=[np.number]).describe().round(2).to_dict() if not df.select_dtypes(include=[np.number]).empty else {}}
        top_rows={df.head(10).to_dict(orient='records')}
        """
    )
    result = call_gemini_json(prompt)
    if not result or not result.get("available"):
        return [
            Insight(
                engine="Gemini LLM Engine",
                insight_type="General",
                insight="Gemini was unavailable in this run, so final ranking should be interpreted without supplementary LLM reasoning.",
                relevance=45,
                actionability=35,
                consistency=2.5,
                clarity=85,
                evidence="Gemini API unavailable or returned invalid JSON.",
                snippet_mask_name=None,
                base_confidence=35,
            )
        ]

    insights: List[Insight] = []
    for item in result.get("insights", [])[:4]:
        insights.append(
            Insight(
                engine="Gemini LLM Engine",
                insight_type=str(item.get("insight_type", "General")),
                insight=str(item.get("insight", "No insight returned.")),
                relevance=int(item.get("relevance", 45)),
                actionability=int(item.get("actionability", 35)),
                consistency=3.0,
                clarity=int(item.get("clarity", 85)),
                evidence="Grounded on current filtered dataset through Gemini summary layer.",
                snippet_mask_name=None,
                base_confidence=int(item.get("base_confidence", 65)),
            )
        )

    if not insights:
        insights.append(
            Insight(
                engine="Gemini LLM Engine",
                insight_type="General",
                insight="Gemini responded without usable insights, so the benchmark relies mainly on deterministic engines.",
                relevance=45,
                actionability=35,
                consistency=2.5,
                clarity=85,
                evidence="Gemini returned empty insight list.",
                snippet_mask_name=None,
                base_confidence=35,
            )
        )
    return insights


# ============================================================
# Contradiction logic
# ============================================================
def normalize_insight_text(s: str) -> str:
    return s.lower().replace("\n", " ").strip()


def contradiction_penalty(a: Insight, b: Insight) -> float:
    ta = normalize_insight_text(a.insight)
    tb = normalize_insight_text(b.insight)

    opposite_pairs = [
        ("improved", "declined"),
        ("increase", "decrease"),
        ("positive", "negative"),
        ("strongest", "weakest"),
        ("growth", "risk"),
    ]
    penalty = 0.0
    for p1, p2 in opposite_pairs:
        if (p1 in ta and p2 in tb) or (p2 in ta and p1 in tb):
            # Only penalize if they appear to talk about the same subject.
            overlap_tokens = set(ta.split()) & set(tb.split())
            if len(overlap_tokens) >= 2:
                penalty += 8.0
    return penalty


def score_insights(insights: List[Insight]) -> List[Insight]:
    for i, ins in enumerate(insights):
        base = (
            ins.relevance * 0.35
            + ins.actionability * 0.25
            + ins.clarity * 0.20
            + ins.base_confidence * 0.20
        )
        penalty = 0.0
        for j, other in enumerate(insights):
            if i == j:
                continue
            penalty += contradiction_penalty(ins, other)
        ins.adjusted_final_score = round(max(0.0, min(100.0, base - penalty)), 1)
    return insights


# ============================================================
# Ask-your-data
# ============================================================
def answer_grounded_question(df: pd.DataFrame, question: str, metric: Optional[str], category_col: Optional[str], dt_col: Optional[str]) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask a grounded question about the currently filtered dataset."

    if metric and ("total" in q or "sum" in q):
        total = float(pd.to_numeric(df[metric], errors="coerce").fillna(0).sum())
        return f"The total {metric} in the current filtered dataset is {total:,.2f}."

    if metric and ("average" in q or "mean" in q):
        avg = float(pd.to_numeric(df[metric], errors="coerce").dropna().mean()) if df[metric].notna().any() else 0.0
        return f"The average {metric} in the current filtered dataset is {avg:,.2f}."

    if metric and category_col and ("top" in q or "best" in q):
        grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False)
        if not grouped.empty:
            return f"The top {to_label(category_col)} by {metric} is {safe_str(grouped.index[0])} with {float(grouped.iloc[0]):,.2f}."

    if metric and category_col and ("lowest" in q or "weakest" in q or "worst" in q):
        grouped = df.groupby(category_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=True)
        if not grouped.empty:
            return f"The weakest {to_label(category_col)} by {metric} is {safe_str(grouped.index[0])} with {float(grouped.iloc[0]):,.2f}."

    if metric and dt_col and ("trend" in q or "over time" in q):
        temp = df[[dt_col, metric]].dropna().copy()
        temp["period"] = temp[dt_col].dt.to_period("M").dt.to_timestamp()
        monthly = temp.groupby("period")[metric].sum(min_count=1)
        if len(monthly) >= 2:
            first_v = float(monthly.iloc[0])
            last_v = float(monthly.iloc[-1])
            direction = "upward" if last_v >= first_v else "downward"
            return f"The monthly trend for {metric} is {direction}: it moves from {first_v:,.2f} in the first observed month to {last_v:,.2f} in the latest observed month."

    return "I could not map that question to a grounded canned analysis pattern yet. Try asking about total, average, top segment, weakest segment, or trend over time."


# ============================================================
# Main UI
# ============================================================
def main() -> None:
    st.title("Client Intelligence Platform")
    st.write(
        "Upload a CSV/XLSX file or try demo mode to generate KPI dashboards, trend analysis, anomaly detection, recommendations, multi-engine insight evaluation, retrieval-backed evidence, and agent-style Q&A."
    )

    hero_cols = st.columns(4)
    hero_texts = [
        ("Business Problem", "Teams often receive raw data with little clarity on what matters most."),
        ("What this platform does", "Converts raw data into summaries, KPIs, charts, risks, and actions."),
        ("Perplexity-style upgrade", "Retrieval-backed evidence, source snippets, benchmark scoring, and agent Q&A."),
        ("How to use it", "Upload data, filter it, review insights, inspect evidence, and ask grounded questions."),
    ]
    for col, (h, p) in zip(hero_cols, hero_texts):
        with col:
            st.markdown(f'<div class="hero-card"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

    st.sidebar.header("Data Input")
    mode = st.sidebar.radio(
        "Choose input mode",
        ["Upload your own file", "Use demo sales dataset"],
        index=0,
    )

    uploaded_file = None
    if mode == "Upload your own file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Limit 200MB per file • CSV, XLSX, XLS",
        )

    try:
        if mode == "Use demo sales dataset":
            df_raw = make_demo_sales_dataset()
            demo_csv = csv_bytes(df_raw)
            st.sidebar.download_button(
                "Download sample sales dataset",
                data=demo_csv,
                file_name="demo_sales_dataset.csv",
                mime="text/csv",
            )
        else:
            if uploaded_file is None:
                st.info("Upload a CSV or Excel file to begin analysis.")
                return
            df_raw = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.exception(e)
        return

    df_raw = add_time_features(df_raw, detect_datetime_column(df_raw))
    metric = detect_primary_metric(df_raw)
    dt_col = detect_datetime_column(df_raw)
    category_col = detect_best_category_column(df_raw, metric)

    # --------------------------------------------------------
    # Sidebar filters
    # --------------------------------------------------------
    st.sidebar.header("Filters")
    filtered_df = df_raw.copy()

    if dt_col and is_datetime_series(filtered_df[dt_col]):
        min_d = filtered_df[dt_col].min().date()
        max_d = filtered_df[dt_col].max().date()
        selected_range = st.sidebar.date_input("Date range", value=(min_d, max_d))
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_d, end_d = selected_range
            filtered_df = filtered_df[
                (filtered_df[dt_col].dt.date >= start_d) &
                (filtered_df[dt_col].dt.date <= end_d)
            ]

    filterable_cols = [
        c for c in filtered_df.columns
        if not is_numeric_series(filtered_df[c]) and not is_datetime_series(filtered_df[c])
    ]

    for col in filterable_cols[:6]:
        values = sorted([safe_str(v) for v in filtered_df[col].dropna().unique().tolist()])
        if 1 < len(values) <= 30:
            selected = st.sidebar.multiselect(f"Filter {to_label(col)}", values, default=[])
            if selected:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    if filtered_df.empty:
        st.warning("No rows remain after filtering. Please broaden your filters.")
        return

    numeric_cols = [c for c in filtered_df.columns if is_numeric_series(filtered_df[c]) and not pd.api.types.is_bool_dtype(filtered_df[c])]
    metric = detect_primary_metric(filtered_df)
    category_col = detect_best_category_column(filtered_df, metric)

    # --------------------------------------------------------
    # KPI row
    # --------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    best_segment = detect_best_category_value(filtered_df, category_col, metric) if metric and category_col else "N/A"
    q_score = quality_score(filtered_df)
    duplicates = int(filtered_df.duplicated().sum())
    total_metric = float(pd.to_numeric(filtered_df[metric], errors="coerce").fillna(0).sum()) if metric else 0.0
    avg_metric = float(pd.to_numeric(filtered_df[metric], errors="coerce").dropna().mean()) if metric and filtered_df[metric].notna().any() else 0.0

    kpi_cols = st.columns(6)
    kpis = [
        ("Rows", f"{len(filtered_df):,}"),
        ("Columns", str(filtered_df.shape[1])),
        ("Missing Values", str(int(filtered_df.isna().sum().sum()))),
        ("Duplicates", str(duplicates)),
        ("Data Quality Score", f"{q_score}/100"),
        ("Best Segment", safe_str(best_segment)),
    ]
    for col, (title, value) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f'<div class="metric-card"><div class="small-muted">{title}</div><h1>{value}</h1></div>', unsafe_allow_html=True)

    if metric:
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric(f"Total {metric}", f"{total_metric:,.2f}")
        with metric_cols[1]:
            st.metric(f"Average {metric}", f"{avg_metric:,.2f}")

    # --------------------------------------------------------
    # Executive summary + risk alerts
    # --------------------------------------------------------
    exec_col, risk_col = st.columns([1.8, 1])

    with exec_col:
        st.subheader("Executive Summary")
        paragraphs = [f"The dataset contains {len(filtered_df):,} records across {filtered_df.shape[1]} columns."]
        if metric:
            paragraphs.append(f"The primary business metric appears to be **{metric}**, with a total value of **{total_metric:,.2f}**.")
        if metric and category_col:
            best_val = detect_best_category_value(filtered_df, category_col, metric)
            weak_val = detect_weakest_category_value(filtered_df, category_col, metric)
            paragraphs.append(f"The strongest contribution comes from **{best_val}** in **{to_label(category_col)}**.")
            if weak_val is not None and weak_val != best_val:
                paragraphs.append(f"The weakest contribution appears in **{weak_val}** under **{to_label(category_col)}**.")
        paragraphs.append(f"From a data quality perspective, the file contains **{int(filtered_df.isna().sum().sum())} missing values** and **{duplicates} duplicate rows**.")
        st.markdown(f'<div class="insight-card">{" ".join(paragraphs)}</div>', unsafe_allow_html=True)

        st.subheader("Recommendations")
        recs = []
        if metric and category_col and best_segment:
            recs.append(f"Protect and replicate the strongest-performing segment: **{best_segment}** in **{to_label(category_col)}**.")
        if metric and category_col:
            weak_val = detect_weakest_category_value(filtered_df, category_col, metric)
            if weak_val and weak_val != best_segment:
                recs.append(f"Investigate the weakest area: **{weak_val}** in **{to_label(category_col)}** may need operational or retention review.")
        if metric and len(numeric_cols) > 1:
            candidate_corrs = []
            for c in numeric_cols:
                if c == metric:
                    continue
                corr = safe_corr(pd.to_numeric(filtered_df[c], errors="coerce"), pd.to_numeric(filtered_df[metric], errors="coerce"))
                if corr is not None:
                    candidate_corrs.append((c, corr))
            if candidate_corrs:
                col_name, corr_val = max(candidate_corrs, key=lambda x: abs(x[1]))
                recs.append(f"Review the relationship between **{to_label(col_name)}** and **{to_label(metric)}** (correlation {corr_val:.2f}) to identify controllable drivers.")
        if duplicates > 0:
            recs.append("Remove duplicate records before presenting insights externally, as duplicates can distort totals and business conclusions.")
        if not recs:
            recs.append("Upload a richer dataset or broaden filters to generate stronger action recommendations.")
        for i, rec in enumerate(recs, start=1):
            st.markdown(f"{i}. {rec}")

    with risk_col:
        st.subheader("Risk Alerts")
        duplicate_risk = detect_duplicate_risk(filtered_df)
        outlier_risks = detect_outlier_risk(filtered_df, numeric_cols)
        imbalance = category_imbalance_risks(filtered_df)
        risk_messages = []
        if duplicate_risk["has_duplicate_risk"]:
            risk_messages.append(("warn-box", duplicate_risk["message"]))
        for x in outlier_risks[:3]:
            risk_messages.append(("warn-box", x["message"]))
        for x in imbalance[:3]:
            risk_messages.append(("insight-card", x["message"]))
        if not risk_messages:
            risk_messages.append(("good-box", "No major structural risk alerts detected in the current filtered view."))
        for cls, msg in risk_messages:
            st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
            st.write("")

    # --------------------------------------------------------
    # Insight evaluation layer
    # --------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Insight Evaluation Layer")
    st.caption("This section compares how different reasoning engines generate and score insights from the same filtered dataset.")

    snippet_masks: Dict[str, pd.Series] = {}
    rb = rule_based_engine(filtered_df, metric, dt_col, category_col, snippet_masks)
    stt = statistical_engine(filtered_df, metric, numeric_cols, snippet_masks)
    narr = narrative_engine(filtered_df, metric, category_col, snippet_masks)
    gem = gemini_engine(filtered_df, metric, category_col, snippet_masks)

    all_insights = score_insights(rb + stt + narr + gem)

    engine_map = {
        "Rule-Based Engine": rb,
        "Statistical Engine": stt,
        "Narrative Engine": narr,
        "Gemini LLM Engine": gem,
    }
    tabs = st.tabs(list(engine_map.keys()))
    for tab, (engine_name, engine_insights) in zip(tabs, engine_map.items()):
        with tab:
            if not engine_insights:
                st.info("No insights generated.")
                continue
            for idx, ins in enumerate(engine_insights, start=1):
                st.markdown(f"### {idx}. {ins.insight}")
                st.write(f"**Evidence:** Grounded from filtered dataset: {ins.evidence}")
                st.write(f"**Base confidence:** {ins.base_confidence}")
                if ins.snippet_mask_name and ins.snippet_mask_name in snippet_masks:
                    with st.expander("View source snippets"):
                        st.dataframe(build_snippet_rows(filtered_df, snippet_masks[ins.snippet_mask_name], 5), use_container_width=True)

    # --------------------------------------------------------
    # Benchmark evaluation table
    # --------------------------------------------------------
    st.subheader("Benchmark Evaluation Table")
    eval_df = pd.DataFrame([
        {
            "Engine": x.engine,
            "Insight Type": x.insight_type,
            "Insight": x.insight,
            "Relevance": x.relevance,
            "Actionability": x.actionability,
            "Consistency": x.consistency,
            "Clarity": x.clarity,
            "Base Confidence": x.base_confidence,
            "Adjusted Final Score": x.adjusted_final_score,
        }
        for x in sorted(all_insights, key=lambda z: z.adjusted_final_score, reverse=True)
    ])
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # Grounded final verdict
    # --------------------------------------------------------
    st.subheader("Grounded Final Verdict")
    top_final = sorted(all_insights, key=lambda z: z.adjusted_final_score, reverse=True)[:5]
    if not top_final:
        st.info("No final insights available.")
    else:
        for i, ins in enumerate(top_final, start=1):
            st.markdown(f"### {i}. {ins.insight}")
            st.write(f"- **Engine:** {ins.engine}")
            st.write(f"- **Adjusted Final Score:** {ins.adjusted_final_score}")
            st.write(f"- **Evidence:** Grounded from filtered dataset: {ins.evidence}")
            if ins.snippet_mask_name and ins.snippet_mask_name in snippet_masks:
                with st.expander("Inspect evidence rows"):
                    st.dataframe(build_snippet_rows(filtered_df, snippet_masks[ins.snippet_mask_name], 5), use_container_width=True)

    # --------------------------------------------------------
    # Engine score chart
    # --------------------------------------------------------
    avg_engine_scores = (
        eval_df.groupby("Engine", dropna=False)["Adjusted Final Score"].mean().reset_index()
        if not eval_df.empty else pd.DataFrame(columns=["Engine", "Adjusted Final Score"])
    )
    if not avg_engine_scores.empty:
        st.subheader("Average Insight Score by Engine")
        fig_bar = px.bar(avg_engine_scores, x="Engine", y="Adjusted Final Score")
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
        best_engine_row = avg_engine_scores.sort_values("Adjusted Final Score", ascending=False).iloc[0]
        st.markdown(
            f'<div class="good-box">Top-performing engine in the current run: <b>{best_engine_row["Engine"]}</b> with an average score of <b>{best_engine_row["Adjusted Final Score"]:.1f}</b>.</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Ask your data
    # --------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Ask Your Data")
    question = st.text_input("Ask a grounded question about the current filtered dataset")
    if question:
        answer = answer_grounded_question(filtered_df, question, metric, category_col, dt_col)
        st.write(answer)

    # --------------------------------------------------------
    # Recommended visualization
    # --------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Recommended Visualization")
    viz = choose_visualization(filtered_df, dt_col, metric, category_col)
    st.caption(viz.get("reason", ""))

    if viz["kind"] == "line" and metric and dt_col:
        temp = filtered_df[[dt_col, metric]].dropna().copy()
        temp = temp.sort_values(dt_col)
        fig = px.line(temp, x=dt_col, y=metric, title=f"{metric} over time")
        st.plotly_chart(fig, use_container_width=True)
    elif viz["kind"] == "bar" and metric and category_col:
        temp = filtered_df.groupby(category_col, dropna=False)[metric].sum(min_count=1).reset_index().sort_values(metric, ascending=False).head(10)
        fig = px.bar(temp, x=category_col, y=metric, title=f"Top {to_label(category_col)} by {metric}")
        st.plotly_chart(fig, use_container_width=True)
    elif viz["kind"] == "hist" and metric:
        fig = px.histogram(filtered_df, x=metric, title=f"Distribution of {metric}")
        st.plotly_chart(fig, use_container_width=True)

    if metric and category_col:
        col_a, col_b = st.columns(2)
        with col_a:
            top_cat = filtered_df.groupby(category_col, dropna=False)[metric].sum(min_count=1).reset_index().sort_values(metric, ascending=False).head(10)
            fig1 = px.bar(top_cat, x=category_col, y=metric, title=f"Top {to_label(category_col)} by {metric}")
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            fig2 = px.histogram(filtered_df, x=metric, title=f"Distribution of {metric}")
            st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        corr_df = filtered_df[numeric_cols].corr(numeric_only=True)
        heat = px.imshow(corr_df, text_auto=True, aspect="auto", title="Numeric Correlation Matrix")
        heat.update_layout(height=720)
        st.plotly_chart(heat, use_container_width=True)

    # --------------------------------------------------------
    # Preview + downloads
    # --------------------------------------------------------
    st.subheader("Preview Data")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download filtered dataset as CSV",
            data=csv_bytes(filtered_df),
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )
    with dl2:
        insights_report = eval_df.copy()
        st.download_button(
            "Download insights report as CSV",
            data=csv_bytes(insights_report),
            file_name="insights_report.csv",
            mime="text/csv",
        )
    with dl3:
        st.download_button(
            "Download evaluation table as CSV",
            data=csv_bytes(eval_df),
            file_name="evaluation_table.csv",
            mime="text/csv",
        )

    st.caption("Next research upgrade: retrieval-backed evidence grounding, structured JSON insight contracts, benchmark datasets, and agent-level failure analysis.")


if __name__ == "__main__":
    main()
