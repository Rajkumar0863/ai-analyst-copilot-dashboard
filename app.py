import os
import io
import json
import math
import textwrap
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

try:
    import google.generativeai as genai
except Exception:
    genai = None

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Client Intelligence Platform",
    layout="wide",
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .main-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px;
        background-color: rgba(255,255,255,0.02);
        margin-bottom: 16px;
    }
    .metric-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px;
        background-color: rgba(255,255,255,0.01);
        min-height: 120px;
    }
    .section-gap {
        margin-top: 18px;
        margin-bottom: 6px;
    }
    .tiny-muted {
        color: #9aa4b2;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# HELPERS
# -----------------------------
def safe_read_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        return pd.DataFrame()


def make_demo_sales_dataset() -> pd.DataFrame:
    # FIXED: use lowercase "h", not "H"
    dates = pd.date_range("2009-12-01", "2010-12-09", freq="h")

    np.random.seed(42)
    n = 1200

    chosen_dates = np.random.choice(dates, n)

    countries = [
        "United Kingdom", "Ireland", "Germany", "France",
        "Spain", "Netherlands", "Sweden", "Switzerland", "Australia"
    ]
    categories = ["cosmetics", "fashion", "electronics", "food"]
    genders = ["Male", "Female", "Non-binary", "Unknown"]
    device_types = ["Mobile", "Desktop", "Tablet"]
    subscription_types = ["Basic", "Premium", "Student", "Family"]
    suppliers = [f"Supplier {i}" for i in range(1, 8)]
    carriers = ["DHL", "FedEx", "UPS", "Royal Mail"]
    locations = ["London", "Dublin", "Berlin", "Paris", "Madrid", "Amsterdam"]

    df = pd.DataFrame({
        "order_datetime": pd.to_datetime(chosen_dates),
        "Product type": np.random.choice(categories, n, p=[0.35, 0.25, 0.25, 0.15]),
        "SKU": [f"SKU{i}" for i in range(n)],
        "Price": np.round(np.random.uniform(2, 100, n), 4),
        "Availability": np.random.randint(1, 100, n),
        "Number of products sold": np.random.randint(10, 1000, n),
        "Revenue generated": np.round(np.random.uniform(1000, 10000, n), 4),
        "Customer demographics": np.random.choice(genders, n),
        "Stock levels": np.random.randint(1, 100, n),
        "Lead times": np.random.randint(1, 30, n),
        "Order quantities": np.random.randint(1, 100, n),
        "Shipping times": np.random.randint(1, 25, n),
        "Shipping carriers": np.random.choice(carriers, n),
        "Shipping costs": np.round(np.random.uniform(5, 200, n), 4),
        "Supplier name": np.random.choice(suppliers, n),
        "Location": np.random.choice(locations, n),
        "Lead time": np.random.randint(1, 20, n),
        "Production volumes": np.random.randint(100, 5000, n),
        "Manufacturing lead time": np.random.randint(5, 40, n),
        "Manufacturing costs": np.round(np.random.uniform(100, 5000, n), 4),
        "Inspection results": np.random.choice(["Pass", "Fail"], n, p=[0.9, 0.1]),
        "Defect rates": np.round(np.random.uniform(0, 0.2, n), 4),
        "Costs": np.round(np.random.uniform(50, 6000, n), 4),
        "Country": np.random.choice(countries, n, p=[0.35, 0.12, 0.1, 0.1, 0.08, 0.08, 0.06, 0.06, 0.05]),
    })

    return df


def detect_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            try:
                parsed = pd.to_datetime(out[col], errors="raise")
                # only accept if enough variation and not all NaT
                if parsed.notna().mean() > 0.85:
                    out[col] = parsed
            except Exception:
                pass
    return out


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    cats = []
    for col in df.columns:
        if col in get_datetime_columns(df):
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            cats.append(col)
    return cats


def choose_primary_metric(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return None

    preferred_names = [
        "Revenue generated", "revenue", "sales", "sales_amount_gbp",
        "amount", "profit", "income", "cost"
    ]
    for p in preferred_names:
        for col in numeric_cols:
            if p.lower() == col.lower():
                return col
        for col in numeric_cols:
            if p.lower() in col.lower():
                return col

    return numeric_cols[0]


def choose_primary_category(df: pd.DataFrame) -> Optional[str]:
    categorical_cols = get_categorical_columns(df)
    if not categorical_cols:
        return None

    preferred_names = [
        "Product type", "category", "Country", "region",
        "Supplier name", "segment", "channel"
    ]
    for p in preferred_names:
        for col in categorical_cols:
            if p.lower() == col.lower():
                return col
        for col in categorical_cols:
            if p.lower() in col.lower():
                return col

    return categorical_cols[0]


def choose_time_col(df: pd.DataFrame) -> Optional[str]:
    dt_cols = get_datetime_columns(df)
    if not dt_cols:
        return None

    preferred = ["date", "datetime", "timestamp", "order_datetime", "time"]
    for p in preferred:
        for col in dt_cols:
            if p.lower() == col.lower():
                return col
        for col in dt_cols:
            if p.lower() in col.lower():
                return col
    return dt_cols[0]


def infer_best_segment(df: pd.DataFrame, metric: str, category_col: Optional[str]) -> str:
    if category_col is None or metric is None or category_col not in df.columns or metric not in df.columns:
        return "N/A"

    grp = df.groupby(category_col, dropna=False)[metric].sum().sort_values(ascending=False)
    if len(grp) == 0:
        return "N/A"
    return str(grp.index[0])


def format_number(x) -> str:
    if pd.isna(x):
        return "N/A"
    try:
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"{x:,.2f}"
        if abs(x) >= 1_000:
            return f"{x:,.2f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# GEMINI
# -----------------------------
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or genai is None:
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
        text = response.text.strip()

        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        return json.loads(text)
    except Exception:
        return None


# -----------------------------
# FILTERING
# -----------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    st.sidebar.markdown("## Filters")

    # Date filter
    time_col = choose_time_col(filtered_df)
    if time_col and time_col in filtered_df.columns and len(filtered_df) > 0:
        temp = filtered_df[time_col].dropna()
        if len(temp) > 0:
            min_date = temp.min().date()
            max_date = temp.max().date()

            date_range = st.sidebar.text_input(
                "Date range",
                f"{min_date.strftime('%Y/%m/%d')} – {max_date.strftime('%Y/%m/%d')}"
            )

            try:
                parts = date_range.split("–")
                start = pd.to_datetime(parts[0].strip()).date()
                end = pd.to_datetime(parts[1].strip()).date()
                filtered_df = filtered_df[
                    (filtered_df[time_col].dt.date >= start) &
                    (filtered_df[time_col].dt.date <= end)
                ]
            except Exception:
                pass

    # categorical filters
    categorical_cols = get_categorical_columns(filtered_df)

    for col in categorical_cols[:8]:
        vals = filtered_df[col].dropna().astype(str).unique().tolist()
        if len(vals) == 0:
            continue

        selected = st.sidebar.multiselect(
            f"Filter {col}",
            options=sorted(vals),
            default=[]
        )
        if selected:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    return filtered_df


# -----------------------------
# RISK / INSIGHT HELPERS
# -----------------------------
def detect_outlier_risk(df: pd.DataFrame, numeric_cols: List[str]) -> List[dict]:
    risks = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)
        share = mask.mean()

        if share >= 0.07:
            risks.append({
                "column": col,
                "share": share,
                "message": f"Outlier concentration in {col} is elevated at {share * 100:.1f}%."
            })

    return sorted(risks, key=lambda x: x["share"], reverse=True)


def strongest_and_weakest_segment(df: pd.DataFrame, metric: str, category_col: str) -> Tuple[Optional[tuple], Optional[tuple]]:
    if metric not in df.columns or category_col not in df.columns:
        return None, None

    grp = df.groupby(category_col, dropna=False)[metric].sum().sort_values(ascending=False)
    if len(grp) == 0:
        return None, None

    strongest = (grp.index[0], grp.iloc[0])
    weakest = (grp.index[-1], grp.iloc[-1])
    return strongest, weakest


def trend_summary(df: pd.DataFrame, time_col: Optional[str], metric: Optional[str]) -> Optional[dict]:
    if not time_col or not metric:
        return None
    if time_col not in df.columns or metric not in df.columns:
        return None

    temp = df[[time_col, metric]].dropna().copy()
    if len(temp) < 2:
        return None

    temp["period"] = temp[time_col].dt.to_period("M").astype(str)
    trend = temp.groupby("period")[metric].sum().reset_index()

    if len(trend) < 2:
        return None

    first_val = trend[metric].iloc[0]
    last_val = trend[metric].iloc[-1]

    if first_val == 0:
        pct_change = np.nan
    else:
        pct_change = ((last_val - first_val) / first_val) * 100

    direction = "improved" if last_val >= first_val else "declined"

    return {
        "trend_df": trend,
        "first_val": first_val,
        "last_val": last_val,
        "pct_change": pct_change,
        "direction": direction,
    }


def strongest_correlation(df: pd.DataFrame, metric: str) -> Optional[dict]:
    numeric_cols = get_numeric_columns(df)
    if metric not in numeric_cols or len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr(numeric_only=True)
    if metric not in corr.columns:
        return None

    s = corr[metric].drop(index=metric).dropna()
    if len(s) == 0:
        return None

    strongest = s.abs().sort_values(ascending=False).index[0]
    value = s[strongest]
    return {
        "column": strongest,
        "value": value
    }


def get_top_rows_for_filter(df: pd.DataFrame, filter_col: str, filter_val, metric: str, top_n: int = 5) -> pd.DataFrame:
    temp = df[df[filter_col].astype(str) == str(filter_val)].copy()
    if metric in temp.columns:
        temp = temp.sort_values(metric, ascending=False)
    return temp.head(top_n)


# -----------------------------
# ENGINES
# -----------------------------
def rule_based_engine(df: pd.DataFrame, metric: str, category_col: str, time_col: Optional[str], outlier_risks: List[dict]) -> List[dict]:
    insights = []

    strongest, weakest = strongest_and_weakest_segment(df, metric, category_col)
    trend = trend_summary(df, time_col, metric)

    if strongest:
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Trend",
            "insight": f"The strongest driver of {metric} is {strongest[0]} within {category_col}, contributing {format_number(strongest[1])} overall.",
            "relevance": 85,
            "actionability": 35,
            "consistency": 4.0,
            "clarity": 80,
            "base_confidence": 88,
            "evidence_col": category_col,
            "evidence_val": strongest[0],
        })

    if weakest:
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Recommendation",
            "insight": f"Underperformance appears in {weakest[0]} under {category_col}, making it a priority area for review.",
            "relevance": 55,
            "actionability": 80,
            "consistency": 4.0,
            "clarity": 80,
            "base_confidence": 84,
            "evidence_col": category_col,
            "evidence_val": weakest[0],
        })

    if trend is not None and not pd.isna(trend["pct_change"]):
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Trend",
            "insight": f"Over time, {metric} has {trend['direction']} by {abs(trend['pct_change']):.1f}% from the first month to the latest month.",
            "relevance": 75,
            "actionability": 55,
            "consistency": 4.0,
            "clarity": 85,
            "base_confidence": 90,
            "evidence_col": None,
            "evidence_val": None,
        })

    if outlier_risks:
        r = outlier_risks[0]
        insights.append({
            "engine": "Rule-Based Engine",
            "insight_type": "Risk",
            "insight": r["message"],
            "relevance": 55,
            "actionability": 65,
            "consistency": 4.0,
            "clarity": 80,
            "base_confidence": 80,
            "evidence_col": None,
            "evidence_val": None,
        })

    return insights


def statistical_engine(df: pd.DataFrame, metric: str, category_col: str, outlier_risks: List[dict]) -> List[dict]:
    insights = []

    corr = strongest_correlation(df, metric)
    if corr:
        direction = "positive" if corr["value"] >= 0 else "negative"
        insights.append({
            "engine": "Statistical Engine",
            "insight_type": "Trend",
            "insight": f"The strongest statistical relationship is a {direction} correlation of {corr['value']:.2f} between {corr['column']} and {metric}.",
            "relevance": 85,
            "actionability": 35,
            "consistency": 4.0,
            "clarity": 80,
            "base_confidence": 82,
            "evidence_col": None,
            "evidence_val": None,
        })

    if outlier_risks:
        r = outlier_risks[0]
        insights.append({
            "engine": "Statistical Engine",
            "insight_type": "Risk",
            "insight": r["message"],
            "relevance": 75,
            "actionability": 55,
            "consistency": 4.0,
            "clarity": 80,
            "base_confidence": 78,
            "evidence_col": None,
            "evidence_val": None,
        })

    return insights


def narrative_engine(df: pd.DataFrame, metric: str, category_col: str) -> List[dict]:
    insights = []
    strongest, weakest = strongest_and_weakest_segment(df, metric, category_col)

    if strongest:
        insights.append({
            "engine": "Narrative Engine",
            "insight_type": "Recommendation",
            "insight": f"From a business perspective, {strongest[0]} in {category_col} appears to be a dependable growth segment that should be protected and scaled.",
            "relevance": 75,
            "actionability": 80,
            "consistency": 4.0,
            "clarity": 95,
            "base_confidence": 75,
            "evidence_col": category_col,
            "evidence_val": strongest[0],
        })

    if weakest:
        insights.append({
            "engine": "Narrative Engine",
            "insight_type": "Risk",
            "insight": f"The weakest segment appears to be {weakest[0]} in {category_col}, suggesting a need for targeted investigation and intervention.",
            "relevance": 55,
            "actionability": 55,
            "consistency": 4.0,
            "clarity": 95,
            "base_confidence": 75,
            "evidence_col": category_col,
            "evidence_val": weakest[0],
        })

    insights.append({
        "engine": "Narrative Engine",
        "insight_type": "General",
        "insight": "Decision confidence is relatively strong because the filtered dataset is internally consistent and provides enough rows for directional business interpretation.",
        "relevance": 45,
        "actionability": 35,
        "consistency": 4.0,
        "clarity": 95,
        "base_confidence": 70,
        "evidence_col": None,
        "evidence_val": None,
    })

    return insights


def gemini_engine(df: pd.DataFrame, metric: str, category_col: str) -> List[dict]:
    model = get_gemini_model()

    if model is None:
        return [{
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini was unavailable in this run, so final ranking should be interpreted without supplemental LLM reasoning.",
            "relevance": 45,
            "actionability": 35,
            "consistency": 2.5,
            "clarity": 85,
            "base_confidence": 35,
            "evidence_col": None,
            "evidence_val": None,
        }]

    sample = df.head(10).to_dict(orient="records")
    prompt = f"""
You are a careful data analyst.
Dataset rows: {len(df)}
Metric: {metric}
Category: {category_col}

Sample rows:
{json.dumps(sample, default=str)}

Return valid JSON only:
{{
  "insights": [
    {{
      "insight": "text",
      "insight_type": "General",
      "relevance": 0,
      "actionability": 0,
      "consistency": 0,
      "clarity": 0,
      "base_confidence": 0
    }}
  ]
}}
"""

    result = call_gemini_json(prompt)

    if not result or "insights" not in result:
        return [{
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini returned no structured output, so fallback handling was applied.",
            "relevance": 45,
            "actionability": 35,
            "consistency": 2.5,
            "clarity": 80,
            "base_confidence": 30,
            "evidence_col": None,
            "evidence_val": None,
        }]

    outputs = []
    for item in result.get("insights", [])[:3]:
        outputs.append({
            "engine": "Gemini LLM Engine",
            "insight_type": item.get("insight_type", "General"),
            "insight": item.get("insight", "No insight returned."),
            "relevance": item.get("relevance", 50),
            "actionability": item.get("actionability", 50),
            "consistency": item.get("consistency", 3.0),
            "clarity": item.get("clarity", 80),
            "base_confidence": item.get("base_confidence", 50),
            "evidence_col": None,
            "evidence_val": None,
        })

    if not outputs:
        outputs.append({
            "engine": "Gemini LLM Engine",
            "insight_type": "General",
            "insight": "Gemini produced an empty response set.",
            "relevance": 45,
            "actionability": 35,
            "consistency": 2.5,
            "clarity": 80,
            "base_confidence": 30,
            "evidence_col": None,
            "evidence_val": None,
        })

    return outputs


# -----------------------------
# EVIDENCE / BENCHMARK / FINAL SCORING
# -----------------------------
def attach_evidence(df: pd.DataFrame, insights: List[dict], metric: str) -> List[dict]:
    enriched = []

    for ins in insights:
        item = ins.copy()

        if item.get("evidence_col") and item.get("evidence_val") is not None:
            e_df = get_top_rows_for_filter(df, item["evidence_col"], item["evidence_val"], metric, top_n=5)
        else:
            e_df = df.head(5)

        item["evidence_rows"] = e_df
        item["evidence_text"] = build_evidence_text(df, item, metric)
        item["adjusted_final_score"] = compute_final_score(item)
        enriched.append(item)

    return enriched


def build_evidence_text(df: pd.DataFrame, insight: dict, metric: str) -> str:
    e_col = insight.get("evidence_col")
    e_val = insight.get("evidence_val")

    if e_col and e_val is not None and e_col in df.columns:
        sub = df[df[e_col].astype(str) == str(e_val)]
        total = sub[metric].sum() if metric in sub.columns else np.nan
        avg = sub[metric].mean() if metric in sub.columns else np.nan
        return f"Grounded from filtered dataset: {e_col}={e_val}, rows={len(sub)}, total_{metric}={format_number(total)}, avg_{metric}={format_number(avg)}."

    total = df[metric].sum() if metric in df.columns else np.nan
    avg = df[metric].mean() if metric in df.columns else np.nan
    return f"Grounded from filtered dataset: rows={len(df)}, sum_{metric}={format_number(total)}, mean_{metric}={format_number(avg)}."


def compute_final_score(insight: dict) -> float:
    relevance = float(insight.get("relevance", 50))
    actionability = float(insight.get("actionability", 50))
    clarity = float(insight.get("clarity", 70))
    consistency = float(insight.get("consistency", 3.0)) * 20
    confidence = float(insight.get("base_confidence", 50))

    score = (
        0.25 * relevance +
        0.20 * actionability +
        0.20 * clarity +
        0.15 * consistency +
        0.20 * confidence
    )
    return round(score, 1)


def build_benchmark_table(all_insights: List[dict]) -> pd.DataFrame:
    rows = []
    for ins in all_insights:
        rows.append({
            "Engine": ins["engine"],
            "Insight Type": ins["insight_type"],
            "Insight": ins["insight"],
            "Relevance": ins["relevance"],
            "Actionability": ins["actionability"],
            "Consistency": ins["consistency"],
            "Clarity": ins["clarity"],
            "Base Confidence": ins["base_confidence"],
            "Adjusted Final Score": ins["adjusted_final_score"],
        })
    return pd.DataFrame(rows).sort_values("Adjusted Final Score", ascending=False)


def get_final_reliable_insights(all_insights: List[dict], top_n: int = 5) -> List[dict]:
    ordered = sorted(all_insights, key=lambda x: x["adjusted_final_score"], reverse=True)
    final_list = []
    seen_text = set()

    for ins in ordered:
        normalized = ins["insight"].strip().lower()
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        final_list.append(ins)
        if len(final_list) >= top_n:
            break

    return final_list


# -----------------------------
# ASK YOUR DATA
# -----------------------------
def answer_grounded_question(df: pd.DataFrame, question: str, metric: Optional[str], category_col: Optional[str]) -> str:
    q = question.strip().lower()
    if not q:
        return ""

    if metric and "total" in q and metric.lower() in q:
        return f"Total {metric}: {format_number(df[metric].sum())}"

    if metric and "average" in q and metric.lower() in q:
        return f"Average {metric}: {format_number(df[metric].mean())}"

    if category_col and metric and ("best" in q or "top" in q):
        grp = df.groupby(category_col)[metric].sum().sort_values(ascending=False)
        if len(grp) > 0:
            return f"Top {category_col} by {metric}: {grp.index[0]} with {format_number(grp.iloc[0])}"

    if category_col and metric and ("worst" in q or "lowest" in q):
        grp = df.groupby(category_col)[metric].sum().sort_values(ascending=True)
        if len(grp) > 0:
            return f"Weakest {category_col} by {metric}: {grp.index[0]} with {format_number(grp.iloc[0])}"

    return "I could not map that question cleanly yet. Try asking about total, average, top segment, or weakest segment."


# -----------------------------
# VISUALS
# -----------------------------
def render_recommended_visuals(df: pd.DataFrame, metric: Optional[str], category_col: Optional[str], time_col: Optional[str]):
    st.markdown("## Recommended Visualization")

    c1, c2 = st.columns(2)

    with c1:
        if category_col and metric and category_col in df.columns and metric in df.columns:
            grp = df.groupby(category_col)[metric].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(grp, x=category_col, y=metric, title=f"Top {category_col} by {metric}")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if metric and metric in df.columns:
            fig = px.histogram(df, x=metric, title=f"Distribution of {metric}")
            st.plotly_chart(fig, use_container_width=True)

    if len(get_numeric_columns(df)) >= 2:
        st.markdown("## Correlation Heatmap")
        corr = df[get_numeric_columns(df)].corr(numeric_only=True)
        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        heat.update_layout(height=650)
        st.plotly_chart(heat, use_container_width=True)

    if time_col and metric and time_col in df.columns and metric in df.columns:
        st.markdown("## Time Trend")
        temp = df[[time_col, metric]].dropna().copy()
        if len(temp) > 1:
            monthly = temp.resample("M", on=time_col)[metric].sum().reset_index()
            fig = px.line(monthly, x=time_col, y=metric, title=f"{metric} over time")
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("Client Intelligence Platform")
    st.write(
        "Upload a CSV/XLSX file or use demo mode to generate KPI dashboards, trend analysis, "
        "anomaly detection, recommendations, multi-engine insight evaluation, retrieval-backed evidence, and agent-style Q&A."
    )

    # Sidebar input
    st.sidebar.markdown("## Data Input")
    input_mode = st.sidebar.radio(
        "Choose input mode",
        ["Upload your own file", "Use demo sales dataset"]
    )

    df = pd.DataFrame()

    if input_mode == "Use demo sales dataset":
        df = make_demo_sales_dataset()
        demo_csv = to_csv_download(df)
        st.sidebar.download_button(
            "Download sample sales dataset",
            data=demo_csv,
            file_name="sample_sales_dataset.csv",
            mime="text/csv",
        )
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            df = safe_read_file(uploaded)

    if df.empty:
        st.info("Upload a CSV/XLSX file or switch to demo mode.")
        st.stop()

    df = detect_datetime_columns(df)

    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning("No rows left after filtering.")
        st.stop()

    metric = choose_primary_metric(filtered_df)
    category_col = choose_primary_category(filtered_df)
    time_col = choose_time_col(filtered_df)
    numeric_cols = get_numeric_columns(filtered_df)

    # Header cards
    best_segment = infer_best_segment(filtered_df, metric, category_col)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Rows", f"{len(filtered_df):,}")
    with c2:
        st.metric("Columns", f"{filtered_df.shape[1]:,}")
    with c3:
        st.metric("Missing Values", int(filtered_df.isna().sum().sum()))
    with c4:
        st.metric("Duplicates", int(filtered_df.duplicated().sum()))
    with c5:
        quality_score = max(0, 100 - min(25, int(filtered_df.duplicated().sum())) - min(25, int(filtered_df.isna().sum().sum())))
        st.metric("Data Quality Score", f"{quality_score}/100")

    c6, c7, c8 = st.columns(3)
    with c6:
        st.metric(f"Total {metric}" if metric else "Total", format_number(filtered_df[metric].sum()) if metric else "N/A")
    with c7:
        st.metric(f"Average {metric}" if metric else "Average", format_number(filtered_df[metric].mean()) if metric else "N/A")
    with c8:
        st.metric("Best Segment", best_segment)

    st.divider()

    # Summary + risks
    strongest, weakest = strongest_and_weakest_segment(filtered_df, metric, category_col) if metric and category_col else (None, None)
    trend = trend_summary(filtered_df, time_col, metric) if metric else None
    outlier_risks = detect_outlier_risk(filtered_df, numeric_cols)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("## Executive Summary")
        summary_parts = [
            f"The dataset contains {len(filtered_df):,} records across {filtered_df.shape[1]} columns."
        ]
        if metric:
            summary_parts.append(f"The primary business metric appears to be **{metric}**, with a total value of **{format_number(filtered_df[metric].sum())}**.")
        if strongest and category_col:
            summary_parts.append(f"The strongest contribution comes from **{strongest[0]}** in **{category_col}**, contributing **{format_number(strongest[1])}**.")
        if weakest and category_col:
            summary_parts.append(f"The weakest contribution appears in **{weakest[0]}** under **{category_col}**.")
        if trend is not None and not pd.isna(trend['pct_change']):
            summary_parts.append(f"Over time, **{metric}** has **{trend['direction']} by {abs(trend['pct_change']):.1f}%** from the first observed month to the latest month.")

        st.markdown('<div class="main-card">' + " ".join(summary_parts) + "</div>", unsafe_allow_html=True)

        st.markdown("## Recommendations")
        recs = []
        if strongest and category_col:
            recs.append(f"1. Protect and replicate the strongest-performing segment: **{strongest[0]}** in **{category_col}**.")
        if weakest and category_col:
            recs.append(f"2. Investigate the weakest area: **{weakest[0]}** in **{category_col}** may need operational or retention review.")
        corr = strongest_correlation(filtered_df, metric) if metric else None
        if corr:
            recs.append(f"3. Review the relationship between **{corr['column']}** and **{metric}** (correlation {corr['value']:.2f}) to identify controllable drivers.")

        for r in recs:
            st.markdown(r)

    with right:
        st.markdown("## Risk Alerts")
        risk_msgs = []

        dup_count = int(filtered_df.duplicated().sum())
        if dup_count > 0:
            risk_msgs.append(f"Duplicate-record risk detected: {dup_count} duplicate rows found.")

        if outlier_risks:
            for r in outlier_risks[:3]:
                risk_msgs.append(r["message"])

        if not risk_msgs:
            risk_msgs.append("No major structural risk alerts detected in the current filtered dataset.")

        for msg in risk_msgs:
            st.markdown(f'<div class="main-card">{msg}</div>', unsafe_allow_html=True)

    st.divider()

    # Engines
    st.markdown("## Insight Evaluation Layer")
    st.caption("This section compares how different reasoning engines generate and score insights from the same filtered dataset.")

    all_insights = []
    if metric and category_col:
        all_insights.extend(rule_based_engine(filtered_df, metric, category_col, time_col, outlier_risks))
        all_insights.extend(statistical_engine(filtered_df, metric, category_col, outlier_risks))
        all_insights.extend(narrative_engine(filtered_df, metric, category_col))
        all_insights.extend(gemini_engine(filtered_df, metric, category_col))

    all_insights = attach_evidence(filtered_df, all_insights, metric) if metric else []

    if all_insights:
        tabs = st.tabs(["Rule-Based Engine", "Statistical Engine", "Narrative Engine", "Gemini LLM Engine"])

        engine_names = ["Rule-Based Engine", "Statistical Engine", "Narrative Engine", "Gemini LLM Engine"]

        for tab, engine_name in zip(tabs, engine_names):
            with tab:
                engine_items = [x for x in all_insights if x["engine"] == engine_name]
                if not engine_items:
                    st.info("No insights produced.")
                    continue

                for idx, ins in enumerate(engine_items, 1):
                    st.markdown(f"### {idx}. {ins['insight']}")
                    st.markdown(f"- **Engine:** {ins['engine']}")
                    st.markdown(f"- **Adjusted Final Score:** {ins['adjusted_final_score']}")
                    st.markdown(f"- **Evidence:** {ins['evidence_text']}")
                    st.markdown(f"- **Base confidence:** {ins['base_confidence']}")

                    with st.expander("Inspect evidence rows"):
                        st.dataframe(ins["evidence_rows"], use_container_width=True)

    st.divider()

    # Benchmark table
    if all_insights:
        st.markdown("## Benchmark Evaluation Table")
        benchmark_df = build_benchmark_table(all_insights)
        st.dataframe(benchmark_df, use_container_width=True, height=320)

        # Final verdict
        st.markdown("## Grounded Final Verdict")
        final_insights = get_final_reliable_insights(all_insights, top_n=5)

        for i, ins in enumerate(final_insights, 1):
            st.markdown(f"### {i}. {ins['insight']}")
            st.markdown(f"- **Engine:** {ins['engine']}")
            st.markdown(f"- **Adjusted Final Score:** {ins['adjusted_final_score']}")
            st.markdown(f"- **Evidence:** {ins['evidence_text']}")

            with st.expander("Inspect evidence rows"):
                st.dataframe(ins["evidence_rows"], use_container_width=True)

        avg_scores = (
            benchmark_df.groupby("Engine")["Adjusted Final Score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.markdown("## Average Insight Score by Engine")
        fig_avg = px.bar(avg_scores, x="Engine", y="Adjusted Final Score")
        st.plotly_chart(fig_avg, use_container_width=True)

        top_engine = avg_scores.iloc[0]["Engine"]
        top_engine_score = avg_scores.iloc[0]["Adjusted Final Score"]
        st.success(f"Top-performing engine in the current run: **{top_engine}** with an average score of **{top_engine_score:.1f}**.")

    st.divider()

    # Ask your data
    st.markdown("## Ask Your Data")
    question = st.text_input("Ask a grounded question about the current filtered dataset")
    if question:
        answer = answer_grounded_question(filtered_df, question, metric, category_col)
        st.info(answer)

    st.divider()

    render_recommended_visuals(filtered_df, metric, category_col, time_col)

    st.divider()

    # Preview
    st.markdown("## Preview Data")
    st.dataframe(filtered_df.head(50), use_container_width=True, height=420)

    # Downloads
    cdl1, cdl2, cdl3 = st.columns(3)

    with cdl1:
        st.download_button(
            "Download filtered dataset as CSV",
            data=to_csv_download(filtered_df),
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )

    with cdl2:
        if all_insights:
            insights_df = pd.DataFrame([
                {
                    "engine": x["engine"],
                    "insight_type": x["insight_type"],
                    "insight": x["insight"],
                    "adjusted_final_score": x["adjusted_final_score"],
                    "evidence": x["evidence_text"],
                }
                for x in all_insights
            ])
            st.download_button(
                "Download insights report as CSV",
                data=to_csv_download(insights_df),
                file_name="insights_report.csv",
                mime="text/csv",
            )

    with cdl3:
        if all_insights:
            benchmark_df = build_benchmark_table(all_insights)
            st.download_button(
                "Download evaluation table as CSV",
                data=to_csv_download(benchmark_df),
                file_name="evaluation_table.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()