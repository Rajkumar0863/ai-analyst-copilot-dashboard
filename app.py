import os
import io
import re
import json
import math
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

load_dotenv()


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Client Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# =========================
# STYLING
# =========================
CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}
.section-card {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
}
.small-muted {
    opacity: 0.75;
    font-size: 0.95rem;
}
.good-box {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(0, 180, 100, 0.18);
    border: 1px solid rgba(0, 180, 100, 0.4);
}
.warn-box {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(255, 170, 0, 0.12);
    border: 1px solid rgba(255, 170, 0, 0.35);
}
.info-box {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(80, 160, 255, 0.12);
    border: 1px solid rgba(80, 160, 255, 0.35);
}
h1, h2, h3 {
    letter-spacing: 0.2px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def normalize_colname(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def safe_label(col: str) -> str:
    return str(col).replace("_", " ").strip().title()


def try_parse_datetime(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() >= max(3, int(len(series) * 0.4)):
            return parsed
        return series
    except Exception:
        return series


def readable_value(x: Any) -> str:
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.2f}"
    return str(x)


def detect_column_roles(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    roles: Dict[str, Dict[str, Any]] = {}

    id_keywords = [
        "id", "uuid", "sku", "invoice", "order_id", "transaction_id",
        "customer_id", "product_id", "session_id", "row_id", "ref", "code"
    ]
    time_keywords = ["date", "time", "timestamp", "datetime", "month", "year", "week", "day"]
    geo_keywords = ["country", "region", "state", "city", "location", "market", "territory"]
    business_segment_keywords = [
        "category", "segment", "channel", "platform", "campaign", "campaign_type",
        "industry", "supplier", "vendor", "brand", "product_type", "subscription",
        "gender", "demographic", "device", "plan", "carrier", "artist", "album",
        "playlist", "genre", "product", "department"
    ]
    macro_keywords = ["population", "gdp", "inflation", "macro", "cpi", "economy"]
    numeric_metric_keywords = [
        "sales", "revenue", "profit", "cost", "spend", "ctr", "cpc", "cpa",
        "roas", "conversion", "conversions", "click", "clicks", "impression",
        "impressions", "quantity", "units", "price", "lead_time", "shipping",
        "stock", "availability", "volume", "streams", "followers", "likes",
        "rating", "duration", "hours", "minutes"
    ]

    for col in df.columns:
        col_norm = normalize_colname(col)
        series = df[col]
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

        role = {
            "normalized": col_norm,
            "is_numeric": pd.api.types.is_numeric_dtype(series),
            "is_datetime": pd.api.types.is_datetime64_any_dtype(series),
            "is_identifier": False,
            "is_time": False,
            "is_geo": False,
            "is_business_segment": False,
            "is_macro": False,
            "is_metric": False,
            "is_candidate_segment": False,
        }

        if any(k in col_norm for k in id_keywords):
            role["is_identifier"] = True

        if not role["is_numeric"] and unique_ratio > 0.85 and series.nunique(dropna=True) > 20:
            role["is_identifier"] = True

        if role["is_datetime"] or any(k in col_norm for k in time_keywords):
            role["is_time"] = True

        if any(k in col_norm for k in geo_keywords):
            role["is_geo"] = True

        if any(k in col_norm for k in business_segment_keywords):
            role["is_business_segment"] = True

        if any(k in col_norm for k in macro_keywords):
            role["is_macro"] = True

        if role["is_numeric"] and any(k in col_norm for k in numeric_metric_keywords):
            role["is_metric"] = True

        if (
            not role["is_numeric"]
            and not role["is_identifier"]
            and not role["is_time"]
            and series.nunique(dropna=True) >= 2
            and series.nunique(dropna=True) <= min(30, max(10, len(df) // 2))
        ):
            if role["is_business_segment"] or role["is_geo"]:
                role["is_candidate_segment"] = True
            else:
                role["is_candidate_segment"] = unique_ratio < 0.6

        roles[col] = role

    return roles


def get_safe_segment_columns(df: pd.DataFrame, column_roles: Dict[str, Dict[str, Any]]) -> List[str]:
    hard_block = ["id", "sku", "invoice", "code", "uuid", "transaction", "customer_id", "product_id"]
    safe_cols = []
    for col, role in column_roles.items():
        col_norm = normalize_colname(col)
        if any(h in col_norm for h in hard_block):
            continue
        if role["is_candidate_segment"]:
            safe_cols.append(col)
    return safe_cols


def get_safe_metric_columns(df: pd.DataFrame, column_roles: Dict[str, Dict[str, Any]]) -> List[str]:
    safe_metrics = []
    for col, role in column_roles.items():
        if role["is_numeric"] and not role["is_identifier"]:
            safe_metrics.append(col)
    return safe_metrics


def choose_primary_metric(df: pd.DataFrame, safe_metric_cols: List[str]) -> Optional[str]:
    preferred_metric_names = [
        "revenue", "sales", "sales_amount", "sales_amount_gbp",
        "revenue_generated", "profit", "roas", "conversions",
        "streams", "listeners", "engagement", "watch_time"
    ]

    for col in df.columns:
        col_norm = normalize_colname(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            if any(name in col_norm for name in preferred_metric_names):
                return col

    if safe_metric_cols:
        return safe_metric_cols[0]
    return None


def choose_best_segment_column(df: pd.DataFrame, candidate_cols: List[str], metric_col: str) -> Optional[str]:
    if metric_col not in df.columns:
        return None

    scored = []
    for col in candidate_cols:
        try:
            nunique = df[col].nunique(dropna=True)
            if nunique < 2:
                continue

            grouped = df.groupby(col, dropna=False)[metric_col].sum().sort_values(ascending=False)
            if grouped.empty or grouped.sum() == 0:
                continue

            top_share = grouped.iloc[0] / grouped.sum()
            cardinality_penalty = abs(nunique - 5)

            score = 100
            score -= cardinality_penalty * 3

            if top_share > 0.95:
                score -= 40
            elif top_share > 0.85:
                score -= 20

            score += min(len(grouped), 10)
            scored.append((col, score))
        except Exception:
            continue

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def summarize_strongest_and_weakest_segment(df: pd.DataFrame, segment_col: str, metric_col: str) -> Optional[Dict[str, Any]]:
    try:
        grouped = df.groupby(segment_col, dropna=False)[metric_col].sum().sort_values(ascending=False)
        if grouped.empty or len(grouped) < 2:
            return None

        return {
            "segment_col": segment_col,
            "strongest_name": str(grouped.index[0]),
            "strongest_value": float(grouped.iloc[0]),
            "weakest_name": str(grouped.index[-1]),
            "weakest_value": float(grouped.iloc[-1]),
        }
    except Exception:
        return None


def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = [normalize_colname(c) for c in df.columns]
    joined = " ".join(cols)

    if any(k in joined for k in ["ctr", "roas", "campaign", "click", "impression", "cpc", "cpa"]):
        return "marketing"
    if any(k in joined for k in ["supplier", "shipping", "stock", "lead_time", "manufacturing"]):
        return "supply_chain"
    if any(k in joined for k in ["invoice", "product", "quantity", "sales_amount", "revenue_generated"]):
        return "ecommerce"
    if any(k in joined for k in ["artist", "album", "genre", "playlist", "streams"]):
        return "media"
    if any(k in joined for k in ["subscription", "device", "watch_time", "country", "gender"]):
        return "subscription"
    return "general"


def read_uploaded_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Unsupported file format. Please upload CSV or Excel.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            parsed = try_parse_datetime(df[col])
            if pd.api.types.is_datetime64_any_dtype(parsed):
                df[col] = parsed

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(100)
        if len(sample) > 0:
            numeric_like = sample.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            success_rate = pd.to_numeric(numeric_like, errors="coerce").notna().mean()
            if success_rate > 0.8:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
                    errors="coerce"
                )
    return df


def make_demo_sales_dataset() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2009-12-01", "2010-12-09", freq="h")
    sample_size = 500

    chosen_dates = np.random.choice(dates, sample_size)
    product_types = np.random.choice(["cosmetics", "haircare", "skincare", "toys", "fashion"], sample_size, p=[0.28, 0.18, 0.20, 0.14, 0.20])
    countries = np.random.choice(["United Kingdom", "France", "Germany", "Ireland", "India", "Spain"], sample_size)
    genders = np.random.choice(["Male", "Female", "Non-binary", "Unknown"], sample_size)
    devices = np.random.choice(["Mobile", "Desktop", "Tablet"], sample_size)
    suppliers = np.random.choice(["Supplier 1", "Supplier 2", "Supplier 3", "Supplier 4"], sample_size)
    carriers = np.random.choice(["DHL", "UPS", "FedEx", "Royal Mail"], sample_size)
    stock_levels = np.random.randint(1, 100, sample_size)
    lead_times = np.random.randint(1, 30, sample_size)
    availability = np.random.randint(1, 100, sample_size)
    price = np.round(np.random.uniform(2, 100, sample_size), 4)
    num_sold = np.random.randint(20, 1000, sample_size)
    shipping_times = np.random.randint(1, 24, sample_size)
    shipping_costs = np.round(np.random.uniform(1, 25, sample_size), 4)
    order_quantities = np.random.randint(1, 120, sample_size)
    revenue_generated = np.round(price * num_sold * np.random.uniform(0.3, 0.9, sample_size), 4)
    manufacturing_costs = np.round(price * np.random.uniform(0.1, 0.5, sample_size), 4)
    production_volumes = np.random.randint(100, 5000, sample_size)
    defect_rates = np.round(np.random.uniform(0.0, 0.15, sample_size), 4)
    costs = np.round(manufacturing_costs * order_quantities + shipping_costs * 4, 4)

    df = pd.DataFrame({
        "Order datetime": chosen_dates,
        "Product type": product_types,
        "SKU": [f"SKU{i}" for i in range(sample_size)],
        "Price": price,
        "Availability": availability,
        "Number of products sold": num_sold,
        "Revenue generated": revenue_generated,
        "Customer demographics": genders,
        "Stock levels": stock_levels,
        "Lead times": lead_times,
        "Order quantities": order_quantities,
        "Shipping times": shipping_times,
        "Shipping carriers": carriers,
        "Shipping costs": shipping_costs,
        "Supplier name": suppliers,
        "Location": countries,
        "Production volumes": production_volumes,
        "Manufacturing lead time": np.random.randint(1, 40, sample_size),
        "Manufacturing costs": manufacturing_costs,
        "Inspection results": np.random.choice(["Pass", "Review", "Fail"], sample_size, p=[0.72, 0.20, 0.08]),
        "Defect rates": defect_rates,
        "Transportation modes": np.random.choice(["Road", "Air", "Sea"], sample_size),
        "Routes": np.random.choice(["A", "B", "C", "D"], sample_size),
        "Costs": costs,
    })
    return df


def infer_date_column(df: pd.DataFrame, roles: Dict[str, Dict[str, Any]]) -> Optional[str]:
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        return datetime_cols[0]
    for col, role in roles.items():
        if role["is_time"]:
            maybe = try_parse_datetime(df[col])
            if pd.api.types.is_datetime64_any_dtype(maybe):
                df[col] = maybe
                return col
    return None


def build_filters(df: pd.DataFrame, roles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    filtered_df = df.copy()

    st.sidebar.markdown("## Filters")

    date_col = infer_date_column(filtered_df, roles)
    if date_col and filtered_df[date_col].notna().sum() > 0:
        min_date = filtered_df[date_col].min().date()
        max_date = filtered_df[date_col].max().date()
        st.sidebar.markdown("### Date range")
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            label_visibility="collapsed"
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df[date_col].dt.date >= start_date) &
                (filtered_df[date_col].dt.date <= end_date)
            ]

    candidate_filters = []
    for col in filtered_df.columns:
        if filtered_df[col].dtype == "object":
            nunique = filtered_df[col].nunique(dropna=True)
            if 1 < nunique <= 25:
                candidate_filters.append(col)

    for col in candidate_filters:
        st.sidebar.markdown(f"### Filter {safe_label(col)}")
        options = sorted([x for x in filtered_df[col].dropna().unique().tolist()])
        selected = st.sidebar.multiselect(
            label=safe_label(col),
            options=options,
            default=[],
            label_visibility="collapsed"
        )
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

    return filtered_df


def compute_data_quality_score(df: pd.DataFrame) -> int:
    if len(df) == 0:
        return 0
    missing_ratio = df.isna().mean().mean()
    duplicate_ratio = df.duplicated().mean()
    score = 100 - int((missing_ratio * 60 + duplicate_ratio * 40) * 100)
    return max(0, min(100, score))


def detect_outlier_risk(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
    risks = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 8:
            continue
        if series.nunique() < 5:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_ratio = float(outlier_mask.mean())
        if outlier_ratio > 0.05:
            risks.append({
                "column": col,
                "outlier_ratio": outlier_ratio
            })
    risks.sort(key=lambda x: x["outlier_ratio"], reverse=True)
    return risks


def strongest_correlation(df: pd.DataFrame, metric_col: str, numeric_cols: List[str]) -> Optional[Tuple[str, float]]:
    if metric_col not in numeric_cols:
        return None
    corr_df = df[numeric_cols].corr(numeric_only=True)
    if metric_col not in corr_df.columns:
        return None
    corr_series = corr_df[metric_col].drop(labels=[metric_col], errors="ignore").dropna()
    if corr_series.empty:
        return None
    strongest = corr_series.abs().sort_values(ascending=False).index[0]
    return strongest, float(corr_series[strongest])


def get_evidence_rows(df: pd.DataFrame, segment_col: Optional[str], segment_value: Optional[str], metric_col: Optional[str], top_n: int = 5) -> pd.DataFrame:
    temp = df.copy()
    if segment_col and segment_value is not None and segment_col in temp.columns:
        temp = temp[temp[segment_col].astype(str) == str(segment_value)]

    if metric_col and metric_col in temp.columns:
        temp = temp.sort_values(metric_col, ascending=False)

    return temp.head(top_n)


def safe_json_load(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


# =========================
# GEMINI
# =========================
def get_gemini_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    return None


def get_gemini_model():
    api_key = get_gemini_api_key()
    if not api_key or not GEMINI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def call_gemini_json(prompt: str) -> Optional[dict]:
    model = get_gemini_model()
    if model is None:
        return None

    try:
        response = model.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else ""
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = safe_json_load(text)
        return parsed
    except Exception:
        return None


# =========================
# ENGINES
# =========================
def rule_based_engine(df: pd.DataFrame, metric_col: str, segment_summary: Optional[Dict[str, Any]], outlier_risks: List[Dict[str, Any]], date_col: Optional[str]) -> List[Dict[str, Any]]:
    insights = []

    if segment_summary:
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Trend",
            "insight": (
                f"The strongest driver of {metric_col} is {segment_summary['strongest_name']} "
                f"within {segment_summary['segment_col']}, contributing {segment_summary['strongest_value']:,.2f} overall."
            ),
            "evidence": f"Grounded from filtered dataset: {segment_summary['segment_col']}={segment_summary['strongest_name']}, total_{metric_col}={segment_summary['strongest_value']:.2f}.",
            "base_confidence": 88,
            "segment_col": segment_summary["segment_col"],
            "segment_value": segment_summary["strongest_name"]
        })

        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Recommendation",
            "insight": (
                f"Underperformance appears in {segment_summary['weakest_name']} under {segment_summary['segment_col']}, "
                f"making it a priority area for review."
            ),
            "evidence": f"Grounded from filtered dataset: weakest {segment_summary['segment_col']}={segment_summary['weakest_name']}, value={segment_summary['weakest_value']:.2f}.",
            "base_confidence": 84,
            "segment_col": segment_summary["segment_col"],
            "segment_value": segment_summary["weakest_name"]
        })

    if date_col and metric_col in df.columns:
        try:
            temp = df[[date_col, metric_col]].dropna().copy()
            if len(temp) > 3:
                temp = temp.sort_values(date_col)
                start_val = temp.iloc[0][metric_col]
                end_val = temp.iloc[-1][metric_col]
                if pd.notna(start_val) and pd.notna(end_val) and abs(start_val) > 1e-9:
                    pct_change = ((end_val - start_val) / abs(start_val)) * 100
                    direction = "improved" if pct_change >= 0 else "declined"
                    insights.append({
                        "engine": "Rule-Based Engine",
                        "insight_type": "Trend",
                        "insight": (
                            f"Over time, {metric_col} has {direction} by {abs(pct_change):.1f}% from the first observed period to the latest period."
                        ),
                        "evidence": f"Grounded from time ordering of {date_col}: start={start_val:.2f}, end={end_val:.2f}.",
                        "base_confidence": 90,
                        "segment_col": None,
                        "segment_value": None
                    })
        except Exception:
            pass

    if outlier_risks:
        top = outlier_risks[0]
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Risk",
            "insight": (
                f"Outlier concentration in {top['column']} is elevated at {top['outlier_ratio'] * 100:.1f}%, "
                f"which may distort averages and trends."
            ),
            "evidence": f"Grounded from IQR-based outlier detection on {top['column']}.",
            "base_confidence": 80,
            "segment_col": None,
            "segment_value": None
        })

    return insights


def statistical_engine(df: pd.DataFrame, metric_col: str, numeric_cols: List[str], outlier_risks: List[Dict[str, Any]], date_col: Optional[str]) -> List[Dict[str, Any]]:
    insights = []

    strongest = strongest_correlation(df, metric_col, numeric_cols)
    if strongest:
        other_col, corr_val = strongest
        direction = "positive" if corr_val >= 0 else "negative"
        insights.append({
            "engine": "Statistical Engine",
            "insight_type": "Trend",
            "insight": (
                f"The strongest statistical relationship is a {direction} correlation of {corr_val:.2f} between {other_col} and {metric_col}."
            ),
            "evidence": f"Grounded from Pearson correlation matrix on numeric fields.",
            "base_confidence": 82,
            "segment_col": None,
            "segment_value": None
        })

    if outlier_risks:
        top = outlier_risks[0]
        insights.append({
            "engine": "Statistical Engine",
            "insight_type": "Risk",
            "insight": (
                f"Outlier analysis shows that {top['column']} has the highest anomaly concentration at {top['outlier_ratio'] * 100:.1f}%."
            ),
            "evidence": f"Grounded from IQR thresholds over numeric distribution.",
            "base_confidence": 78,
            "segment_col": None,
            "segment_value": None
        })

    if date_col and metric_col in df.columns:
        try:
            temp = df[[date_col, metric_col]].dropna().copy()
            if len(temp) > 5:
                temp["period"] = temp[date_col].dt.to_period("M").astype(str)
                monthly = temp.groupby("period")[metric_col].mean()
                if len(monthly) >= 3:
                    recent = monthly.tail(3)
                    slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
                    direction = "upward" if slope > 0 else "downward"
                    insights.append({
                        "engine": "Statistical Engine",
                        "insight_type": "Trend",
                        "insight": (
                            f"Recent monthly movement for {metric_col} shows a {direction} direction across the latest observed periods."
                        ),
                        "evidence": f"Grounded from monthly average aggregation on {date_col}.",
                        "base_confidence": 76,
                        "segment_col": None,
                        "segment_value": None
                    })
        except Exception:
            pass

    return insights


def narrative_engine(df: pd.DataFrame, metric_col: str, segment_summary: Optional[Dict[str, Any]], quality_score: int) -> List[Dict[str, Any]]:
    insights = []

    if segment_summary:
        insights.append({
            "engine": "Narrative Engine",
            "insight_type": "Recommendation",
            "insight": (
                f"From a business perspective, {segment_summary['strongest_name']} in {segment_summary['segment_col']} "
                f"appears to be a dependable growth segment that should be protected and scaled."
            ),
            "evidence": f"Grounded from filtered dataset totals in {segment_summary['segment_col']}.",
            "base_confidence": 85,
            "segment_col": segment_summary["segment_col"],
            "segment_value": segment_summary["strongest_name"]
        })

        insights.append({
            "engine": "Narrative Engine",
            "insight_type": "Risk",
            "insight": (
                f"The weakest segment appears to be {segment_summary['weakest_name']} in {segment_summary['segment_col']}, "
                f"suggesting a need for targeted investigation and intervention."
            ),
            "evidence": f"Grounded from lowest contribution segment in current filtered dataset.",
            "base_confidence": 82,
            "segment_col": segment_summary["segment_col"],
            "segment_value": segment_summary["weakest_name"]
        })

    insights.append({
        "engine": "Narrative Engine",
        "insight_type": "General",
        "insight": (
            f"Decision confidence is relatively strong because the data quality score is {quality_score}/100, "
            f"supporting more reliable interpretation."
        ),
        "evidence": f"Grounded from missing-value and duplicate-rate checks.",
        "base_confidence": 75,
        "segment_col": None,
        "segment_value": None
    })

    return insights


def gemini_engine(filtered_df: pd.DataFrame, metric_col: str, category_col: Optional[str], snippet_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    api_key = get_gemini_api_key()
    if not api_key or not GEMINI_AVAILABLE:
        return [{
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini was unavailable in this run, so final ranking should be interpreted without LLM-supported reasoning.",
            "evidence": "No valid Gemini API key or Gemini SDK unavailable.",
            "base_confidence": 40,
            "segment_col": None,
            "segment_value": None
        }]

    cols_preview = filtered_df.columns.tolist()[:20]
    rows_preview = snippet_rows.head(5).to_dict(orient="records")

    prompt = f"""
You are a careful business data analyst.
Return ONLY valid JSON with this schema:
{{
  "insights": [
    {{
      "insight_type": "General|Trend|Recommendation|Risk",
      "insight": "text",
      "evidence": "text",
      "base_confidence": 0-100
    }}
  ]
}}

Context:
- Primary metric: {metric_col}
- Category column: {category_col}
- Columns preview: {cols_preview}
- Example rows: {json.dumps(rows_preview, default=str)}

Rules:
- Do not hallucinate.
- Base every claim only on the provided rows and context.
- Max 3 insights.
- Keep wording concise and business-friendly.
"""

    result = call_gemini_json(prompt)
    if not result or "insights" not in result:
        return [{
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini response could not be parsed in this run, so final ranking should be interpreted without LLM-supported reasoning.",
            "evidence": "Gemini returned invalid or empty structured output.",
            "base_confidence": 35,
            "segment_col": None,
            "segment_value": None
        }]

    insights = []
    for item in result.get("insights", [])[:3]:
        insights.append({
            "engine": "Gemini LLM Engine",
            "insight_type": item.get("insight_type", "General"),
            "insight": item.get("insight", "No insight returned."),
            "evidence": item.get("evidence", "Grounded from provided snippet rows."),
            "base_confidence": int(item.get("base_confidence", 65)),
            "segment_col": category_col,
            "segment_value": None
        })

    if not insights:
        insights.append({
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini did not return usable insights for this run.",
            "evidence": "No structured insights available.",
            "base_confidence": 35,
            "segment_col": None,
            "segment_value": None
        })

    return insights


# =========================
# SCORING
# =========================
def benchmark_score_insight(text: str, insight_type: str, evidence: str, base_confidence: int) -> Dict[str, Any]:
    relevance = 85 if insight_type in ["Trend", "Recommendation"] else 75
    actionability = 80 if insight_type == "Recommendation" else 55 if insight_type == "Risk" else 35
    clarity = 95 if len(text) < 180 else 80
    consistency = 4.0 if evidence and "Grounded" in evidence else 2.5

    final_score = (
        0.30 * relevance +
        0.25 * actionability +
        0.20 * clarity +
        0.10 * (consistency * 20) +
        0.15 * base_confidence
    )

    return {
        "Relevance": relevance,
        "Actionability": actionability,
        "Consistency": round(consistency, 1),
        "Clarity": clarity,
        "Adjusted Final Score": round(final_score, 1)
    }


def make_benchmark_table(insights: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in insights:
        scores = benchmark_score_insight(
            item["insight"],
            item["insight_type"],
            item["evidence"],
            item["base_confidence"]
        )
        rows.append({
            "Engine": item["engine"],
            "Insight Type": item["insight_type"],
            "Insight": item["insight"],
            "Relevance": scores["Relevance"],
            "Actionability": scores["Actionability"],
            "Consistency": scores["Consistency"],
            "Clarity": scores["Clarity"],
            "Base Confidence": item["base_confidence"],
            "Adjusted Final Score": scores["Adjusted Final Score"]
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Adjusted Final Score", ascending=False)


# =========================
# ASK YOUR DATA
# =========================
def answer_grounded_question(question: str, df: pd.DataFrame, metric_col: Optional[str], segment_summary: Optional[Dict[str, Any]]) -> str:
    q = question.lower().strip()
    if not q:
        return ""

    if "best segment" in q or "strongest segment" in q:
        if segment_summary:
            return (
                f"The strongest segment is **{segment_summary['strongest_name']}** under **{segment_summary['segment_col']}**, "
                f"with approximately **{segment_summary['strongest_value']:,.2f}** in {metric_col}."
            )
        return "I couldn't identify a strong segment from the current filtered dataset."

    if "weakest segment" in q or "worst segment" in q:
        if segment_summary:
            return (
                f"The weakest segment is **{segment_summary['weakest_name']}** under **{segment_summary['segment_col']}**, "
                f"with approximately **{segment_summary['weakest_value']:,.2f}** in {metric_col}."
            )
        return "I couldn't identify a weak segment from the current filtered dataset."

    if metric_col and ("total" in q or "sum" in q):
        total = df[metric_col].sum()
        return f"The total **{metric_col}** in the current filtered dataset is **{total:,.2f}**."

    if metric_col and ("average" in q or "mean" in q):
        avg = df[metric_col].mean()
        return f"The average **{metric_col}** in the current filtered dataset is **{avg:,.2f}**."

    return (
        "Try asking something like:\n"
        "- What is the best segment?\n"
        "- What is the weakest segment?\n"
        "- What is the total revenue?\n"
        "- What is the average revenue?"
    )


# =========================
# VISUALS
# =========================
def render_recommended_visualization(df: pd.DataFrame, metric_col: Optional[str], date_col: Optional[str], segment_col: Optional[str]):
    st.markdown("## Recommended Visualization")

    if metric_col is None:
        st.info("No numeric metric available for visualization.")
        return

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.markdown(f"Recommended chart: line chart because **{date_col}** is a time field and **{metric_col}** is numeric.")
        temp = df[[date_col, metric_col]].dropna().sort_values(date_col)
        fig = px.line(temp, x=date_col, y=metric_col, title=f"{metric_col} over time")
        st.plotly_chart(fig, use_container_width=True)
    elif segment_col:
        grouped = df.groupby(segment_col)[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
        st.markdown(f"Recommended chart: bar chart because **{segment_col}** is categorical and **{metric_col}** is numeric.")
        fig = px.bar(grouped, x=segment_col, y=metric_col, title=f"Top {safe_label(segment_col)} by {metric_col}")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if segment_col:
            grouped = df.groupby(segment_col)[metric_col].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(grouped, x=segment_col, y=metric_col, title=f"Top {safe_label(segment_col)} by {metric_col}")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x=metric_col, title=f"Distribution of {metric_col}")
        st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    st.markdown("## Correlation Heatmap")
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for a correlation heatmap.")
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Numeric Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# DOWNLOADS
# =========================
def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# MAIN APP
# =========================
def main():
    st.title("Client Intelligence Platform")
    st.markdown(
        "Upload a CSV/XLSX file or try demo mode to generate KPI dashboards, trend analysis, anomaly detection, recommendations, "
        "multi-engine insight evaluation, retrieval-backed evidence, and agent-style Q&A."
    )

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="info-box"><b>Business Problem</b> Teams often receive raw data with little clarity on what matters most.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-box"><b>What this platform does</b> Converts raw data into summaries, KPIs, charts, risks, and actions.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-box"><b>Perplexity-style upgrade</b> Retrieval-backed evidence, source snippets, benchmark scoring, and agent Q&A.</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="info-box"><b>How to use it</b> Upload data, filter it, review insights, inspect evidence, and ask grounded questions.</div>', unsafe_allow_html=True)

    st.sidebar.markdown("## Data Input")
    input_mode = st.sidebar.radio(
        "Choose input mode",
        options=["Upload your own file", "Use demo sales dataset"]
    )

    df = None
    uploaded_file = None

    if input_mode == "Upload your own file":
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df = read_uploaded_file(uploaded_file)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()
    else:
        df = make_demo_sales_dataset()
        demo_csv = to_csv_download(df)
        st.sidebar.download_button(
            "Download sample sales dataset",
            data=demo_csv,
            file_name="demo_sales_dataset.csv",
            mime="text/csv"
        )

    if df is None:
        st.info("Upload a file or choose demo mode to begin.")
        st.stop()

    df = clean_dataframe(df)

    roles = detect_column_roles(df)
    dataset_type = detect_dataset_type(df)
    filtered_df = build_filters(df, roles)

    if filtered_df.empty:
        st.warning("No rows remain after applying filters.")
        st.stop()

    filtered_roles = detect_column_roles(filtered_df)
    safe_segment_cols = get_safe_segment_columns(filtered_df, filtered_roles)
    safe_metric_cols = get_safe_metric_columns(filtered_df, filtered_roles)
    primary_metric = choose_primary_metric(filtered_df, safe_metric_cols)
    date_col = infer_date_column(filtered_df, filtered_roles)

    best_segment_col = choose_best_segment_column(filtered_df, safe_segment_cols, primary_metric) if primary_metric else None
    segment_summary = summarize_strongest_and_weakest_segment(filtered_df, best_segment_col, primary_metric) if best_segment_col and primary_metric else None

    quality_score = compute_data_quality_score(filtered_df)
    duplicates = int(filtered_df.duplicated().sum())
    missing_values = int(filtered_df.isna().sum().sum())

    best_segment_display = "N/A"
    if segment_summary:
        best_segment_display = f"{segment_summary['strongest_name']} ({segment_summary['segment_col']})"

    # KPIs
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Rows", f"{len(filtered_df):,}")
    with k2:
        st.metric("Columns", f"{filtered_df.shape[1]:,}")
    with k3:
        st.metric("Missing Values", f"{missing_values:,}")
    with k4:
        st.metric("Duplicates", f"{duplicates:,}")
    with k5:
        st.metric("Data Quality Score", f"{quality_score}/100")

    if primary_metric:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"Total {primary_metric}", readable_value(filtered_df[primary_metric].sum()))
        with c2:
            st.metric(f"Average {primary_metric}", readable_value(filtered_df[primary_metric].mean()))
        with c3:
            st.metric("Best Segment", best_segment_display)

    # Executive Summary + Risk Alerts
    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.markdown("## Executive Summary")
        executive_lines = []
        executive_lines.append(f"The dataset contains {len(filtered_df):,} records across {filtered_df.shape[1]} columns.")
        executive_lines.append(f"The detected dataset type is **{dataset_type}**.")

        if primary_metric:
            total_metric = filtered_df[primary_metric].sum()
            executive_lines.append(
                f"The primary business metric appears to be **{primary_metric}**, with a total value of **{total_metric:,.2f}**."
            )

        if segment_summary and primary_metric:
            executive_lines.append(
                f"The strongest contribution comes from **{segment_summary['strongest_name']}** in **{segment_summary['segment_col']}**, "
                f"contributing **{segment_summary['strongest_value']:,.2f}** to {primary_metric}."
            )
            executive_lines.append(
                f"The weakest contribution appears in **{segment_summary['weakest_name']}** under **{segment_summary['segment_col']}**, "
                f"which may need further review."
            )

        executive_lines.append(
            f"From a data quality perspective, the current filtered dataset contains **{missing_values} missing values** and **{duplicates} duplicate rows**."
        )

        st.markdown(" ".join(executive_lines))

        st.markdown("## Recommendations")
        recs = []
        if segment_summary:
            recs.append(
                f"Protect and replicate the strongest-performing segment: **{segment_summary['strongest_name']}** in **{segment_summary['segment_col']}**."
            )
            recs.append(
                f"Investigate the weakest area: **{segment_summary['weakest_name']}** in **{segment_summary['segment_col']}** may need operational or retention review."
            )

        outlier_risks = detect_outlier_risk(filtered_df, safe_metric_cols)
        corr_pair = strongest_correlation(filtered_df, primary_metric, safe_metric_cols) if primary_metric else None
        if corr_pair:
            recs.append(
                f"Review the relationship between **{corr_pair[0]}** and **{primary_metric}** (correlation {corr_pair[1]:.2f}) to identify controllable drivers."
            )
        if duplicates > 0:
            recs.append("Remove duplicate records before presenting insights externally, as duplicates can distort totals and business conclusions.")

        if not recs:
            recs.append("No strong recommendation could be generated from the current filtered dataset.")

        for idx, rec in enumerate(recs, 1):
            st.markdown(f"{idx}. {rec}")

    with right:
        st.markdown("## Risk Alerts")
        if duplicates > 0:
            st.markdown(f'<div class="warn-box">Duplicate-record risk detected: {duplicates} duplicate rows found.</div>', unsafe_allow_html=True)

        outlier_risks = detect_outlier_risk(filtered_df, safe_metric_cols)
        if outlier_risks:
            for risk in outlier_risks[:3]:
                st.markdown(
                    f'<div class="info-box">Outlier concentration in <b>{risk["column"]}</b> is elevated at {risk["outlier_ratio"] * 100:.1f}%.</div>',
                    unsafe_allow_html=True
                )
        elif duplicates == 0:
            st.markdown('<div class="good-box">No major duplicate or outlier risk was detected in the current filtered view.</div>', unsafe_allow_html=True)

    # Engines
    st.markdown("---")
    st.markdown("## Insight Evaluation Layer")
    st.markdown("This section compares how different reasoning engines generate and score insights from the same filtered dataset.")

    snippet_rows = get_evidence_rows(
        filtered_df,
        segment_summary["segment_col"] if segment_summary else None,
        segment_summary["strongest_name"] if segment_summary else None,
        primary_metric,
        top_n=5
    )

    all_insights = []
    if primary_metric:
        all_insights.extend(rule_based_engine(filtered_df, primary_metric, segment_summary, outlier_risks, date_col))
        all_insights.extend(statistical_engine(filtered_df, primary_metric, safe_metric_cols, outlier_risks, date_col))
        all_insights.extend(narrative_engine(filtered_df, primary_metric, segment_summary, quality_score))
        all_insights.extend(gemini_engine(filtered_df, primary_metric, best_segment_col, snippet_rows))

    benchmark_df = make_benchmark_table(all_insights)

    if not benchmark_df.empty:
        top_insights = benchmark_df.sort_values("Adjusted Final Score", ascending=False).head(5)

        for i, (_, row) in enumerate(top_insights.iterrows(), 1):
            source_item = next((x for x in all_insights if x["engine"] == row["Engine"] and x["insight"] == row["Insight"]), None)
            st.markdown(f"### {i}. {row['Insight']}")
            st.markdown(f"- **Engine:** {row['Engine']}")
            st.markdown(f"- **Adjusted Final Score:** {row['Adjusted Final Score']}")
            if source_item:
                st.markdown(f"- **Evidence:** {source_item['evidence']}")
                ev_rows = get_evidence_rows(
                    filtered_df,
                    source_item.get("segment_col"),
                    source_item.get("segment_value"),
                    primary_metric,
                    top_n=5
                )
                with st.expander("Inspect evidence rows"):
                    st.dataframe(ev_rows, use_container_width=True)

        st.markdown("## Benchmark Evaluation Table")
        st.dataframe(benchmark_df, use_container_width=True)

        st.markdown("## Grounded Final Verdict")
        final_top = benchmark_df.sort_values("Adjusted Final Score", ascending=False).head(5)
        for i, (_, row) in enumerate(final_top.iterrows(), 1):
            st.markdown(f"**{i}. {row['Insight']}**")
            st.markdown(f"- Engine: {row['Engine']}")
            st.markdown(f"- Adjusted Final Score: {row['Adjusted Final Score']}")

        avg_scores = benchmark_df.groupby("Engine")["Adjusted Final Score"].mean().sort_values(ascending=False)
        st.markdown("## Average Insight Score by Engine")
        fig_avg = px.bar(
            avg_scores.reset_index(),
            x="Engine",
            y="Adjusted Final Score",
            title="Average Insight Score by Engine"
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        if not avg_scores.empty:
            best_engine = avg_scores.index[0]
            best_score = avg_scores.iloc[0]
            st.markdown(
                f'<div class="good-box">Top-performing engine in the current run: <b>{best_engine}</b> with an average score of <b>{best_score:.1f}</b>.</div>',
                unsafe_allow_html=True
            )

    # Ask your data
    st.markdown("---")
    st.markdown("## Ask Your Data")
    question = st.text_input("Ask a grounded question about the current filtered dataset")
    if question:
        answer = answer_grounded_question(question, filtered_df, primary_metric, segment_summary)
        st.markdown(answer)

    # Recommended charts
    st.markdown("---")
    render_recommended_visualization(filtered_df, primary_metric, date_col, best_segment_col)

    # Heatmap
    st.markdown("---")
    render_correlation_heatmap(filtered_df, safe_metric_cols)

    # Preview + Downloads
    st.markdown("---")
    st.markdown("## Preview Data")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download filtered dataset as CSV",
            data=to_csv_download(filtered_df),
            file_name="filtered_dataset.csv",
            mime="text/csv"
        )

    with d2:
        if not benchmark_df.empty:
            st.download_button(
                "Download evaluation table as CSV",
                data=to_csv_download(benchmark_df),
                file_name="benchmark_evaluation_table.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.caption("Next research upgrade: retrieval-backed evidence grounding, structured JSON insight contracts, benchmark datasets, and safer LLM failure handling.")


if __name__ == "__main__":
    main()