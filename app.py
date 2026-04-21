import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Client Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .subtitle {
            color: #9aa4b2;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .small-note {
            color: #96a0ad;
            font-size: 0.9rem;
        }
        .risk-box {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.6rem;
        }
        .risk-high {
            background: rgba(255, 99, 71, 0.14);
            border: 1px solid rgba(255, 99, 71, 0.35);
        }
        .risk-medium {
            background: rgba(255, 165, 0, 0.13);
            border: 1px solid rgba(255, 165, 0, 0.35);
        }
        .risk-low {
            background: rgba(0, 191, 255, 0.12);
            border: 1px solid rgba(0, 191, 255, 0.28);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Utility helpers
# =========================
def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return numeric, categorical, datetime, identifier-like columns."""
    working = df.copy()

    for col in working.columns:
        if working[col].dtype == "object":
            parsed = pd.to_datetime(working[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                working[col] = parsed

    numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
    datetime_cols = [c for c in working.columns if pd.api.types.is_datetime64_any_dtype(working[c])]

    identifier_cols = []
    categorical_cols = []
    row_count = max(len(working), 1)

    for col in working.columns:
        if col in numeric_cols or col in datetime_cols:
            continue
        nunique_ratio = working[col].nunique(dropna=True) / row_count
        if nunique_ratio > 0.9 or col.lower().endswith(("id", "_id")):
            identifier_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols, identifier_cols


def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                out[col] = parsed
    return out


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")


def create_sample_sales_data(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    regions = rng.choice(["North", "South", "East", "West"], size=rows, p=[0.27, 0.24, 0.22, 0.27])
    categories = rng.choice(["Software", "Hardware", "Services"], size=rows, p=[0.45, 0.30, 0.25])
    segments = rng.choice(["SMB", "Mid-Market", "Enterprise"], size=rows, p=[0.5, 0.3, 0.2])
    orders = rng.integers(10, 120, size=rows)
    avg_ticket = rng.normal(220, 45, size=rows).clip(40, None)
    revenue = orders * avg_ticket
    cost = revenue * rng.uniform(0.45, 0.78, size=rows)
    profit = revenue - cost
    satisfaction = rng.normal(4.1, 0.45, size=rows).clip(2.0, 5.0)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Region": regions,
            "Category": categories,
            "Customer Segment": segments,
            "Orders": orders,
            "Average Ticket": avg_ticket.round(2),
            "Revenue": revenue.round(2),
            "Cost": cost.round(2),
            "Profit": profit.round(2),
            "Satisfaction Score": satisfaction.round(2),
        }
    )

    sample_idx = rng.choice(df.index, size=12, replace=False)
    df.loc[sample_idx[:5], "Revenue"] = np.nan
    df.loc[sample_idx[5:8], "Region"] = np.nan
    df = pd.concat([df, df.sample(3, random_state=7)], ignore_index=True)
    return df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def compute_data_quality_score(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    duplicate_ratio = df.duplicated().mean() if len(df) else 0

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_penalty = 0.0
    if numeric_cols:
        penalties = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_rate = ((series < lower) | (series > upper)).mean()
            penalties.append(outlier_rate)
        outlier_penalty = float(np.mean(penalties)) if penalties else 0.0

    score = 100
    score -= missing_ratio * 45
    score -= duplicate_ratio * 30
    score -= outlier_penalty * 25
    return max(0, min(100, int(round(score))))


def get_primary_metric(numeric_cols: List[str]) -> Optional[str]:
    preferred = ["revenue", "sales", "profit", "amount", "value", "orders", "cost", "quantity"]
    lowered = {c.lower(): c for c in numeric_cols}
    for key in preferred:
        for col_lower, original in lowered.items():
            if key in col_lower:
                return original
    return numeric_cols[0] if numeric_cols else None


def get_primary_datetime(datetime_cols: List[str]) -> Optional[str]:
    preferred = ["date", "time", "month", "year"]
    lowered = {c.lower(): c for c in datetime_cols}
    for key in preferred:
        for col_lower, original in lowered.items():
            if key in col_lower:
                return original
    return datetime_cols[0] if datetime_cols else None


def top_category_by_metric(df: pd.DataFrame, cat_cols: List[str], metric: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    if not cat_cols or not metric or metric not in df.columns:
        return None, None, None

    best_col = None
    best_value = -np.inf
    best_category = None

    for col in cat_cols:
        grouped = df.groupby(col, dropna=False)[metric].sum(numeric_only=True)
        if grouped.empty:
            continue
        local_best_category = grouped.idxmax()
        local_best_value = grouped.max()
        if local_best_value > best_value:
            best_value = float(local_best_value)
            best_category = str(local_best_category)
            best_col = col

    return best_col, best_category, best_value if best_col is not None else None


def worst_category_by_metric(df: pd.DataFrame, cat_cols: List[str], metric: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    if not cat_cols or not metric or metric not in df.columns:
        return None, None, None

    worst_col = None
    worst_value = np.inf
    worst_category = None

    for col in cat_cols:
        grouped = df.groupby(col, dropna=False)[metric].sum(numeric_only=True)
        if grouped.empty:
            continue
        local_worst_category = grouped.idxmin()
        local_worst_value = grouped.min()
        if local_worst_value < worst_value:
            worst_value = float(local_worst_value)
            worst_category = str(local_worst_category)
            worst_col = col

    return worst_col, worst_category, worst_value if worst_col is not None else None


def calculate_metric_trend(df: pd.DataFrame, metric: str, date_col: str) -> Dict[str, Optional[float]]:
    temp = df[[date_col, metric]].dropna().copy()
    if temp.empty:
        return {"pct_change": None, "direction": None, "monthly": None}

    temp = temp.sort_values(date_col)
    temp["period"] = temp[date_col].dt.to_period("M").astype(str)
    monthly = temp.groupby("period")[metric].sum(numeric_only=True)
    if len(monthly) < 2:
        return {"pct_change": None, "direction": None, "monthly": monthly}

    first_val = monthly.iloc[0]
    last_val = monthly.iloc[-1]
    if first_val == 0:
        return {"pct_change": None, "direction": None, "monthly": monthly}

    pct_change = ((last_val - first_val) / abs(first_val)) * 100
    direction = "up" if pct_change >= 0 else "down"
    return {"pct_change": float(pct_change), "direction": direction, "monthly": monthly}


def get_strongest_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[Tuple[str, str, float]]:
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    if corr.empty:
        return None
    pairs = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(key=lambda s: s.abs(), ascending=False)
    if pairs.empty:
        return None
    a, b = pairs.index[0]
    return str(a), str(b), float(pairs.iloc[0])


def get_outlier_signal(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[Tuple[str, float]]:
    best_col = None
    best_rate = 0.0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rate = ((series < lower) | (series > upper)).mean()
        if outlier_rate > best_rate:
            best_rate = float(outlier_rate)
            best_col = col
    if best_col is None:
        return None
    return best_col, best_rate


def classify_insight(insight: str) -> str:
    text = insight.lower()
    if any(word in text for word in ["missing", "duplicate", "risk", "outlier", "decline", "drop", "weak", "imbalance"]):
        return "risk"
    if any(word in text for word in ["recommend", "focus", "investigate", "prioritize", "review", "protect"]):
        return "recommendation"
    if any(word in text for word in ["correlation", "trend", "increase", "decrease", "grew", "improved", "strongest"]):
        return "trend"
    return "general"


def agreement_label(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def insight_keywords(text: str) -> set:
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "by", "with", "is", "are", "was", "were",
        "this", "that", "from", "on", "as", "it", "be", "at", "into", "across", "than", "may", "can", "should"
    }
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = {tok for tok in cleaned.split() if len(tok) > 2 and tok not in stopwords}
    return tokens


def score_insight(insight: str, primary_metric: Optional[str], peer_insights: List[str], cat_cols: List[str], dt_cols: List[str]) -> Dict[str, float]:
    text = insight.lower()

    relevance = 45.0
    if primary_metric and primary_metric.lower() in text:
        relevance += 30
    if any(col.lower() in text for col in cat_cols[:3]):
        relevance += 10
    if any(col.lower() in text for col in dt_cols[:2]):
        relevance += 10
    relevance = min(relevance, 100)

    actionability = 35.0
    if any(word in text for word in ["recommend", "investigate", "review", "prioritize", "focus", "protect", "improve"]):
        actionability += 45
    if any(word in text for word in ["risk", "decline", "drop", "weak", "underperformance"]):
        actionability += 10
    actionability = min(actionability, 100)

    clarity = 40.0
    word_count = len(insight.split())
    if 12 <= word_count <= 32:
        clarity += 30
    if "." in insight or "," in insight:
        clarity += 10
    if any(word in text for word in ["because", "suggests", "indicates", "appears"]):
        clarity += 15
    clarity = min(clarity, 100)

    current_keywords = insight_keywords(insight)
    consistency_values = []
    for peer in peer_insights:
        peer_keywords = insight_keywords(peer)
        union = current_keywords | peer_keywords
        if not union:
            continue
        overlap = len(current_keywords & peer_keywords) / len(union)
        consistency_values.append(overlap)
    consistency = (float(np.mean(consistency_values)) * 100) if consistency_values else 40.0

    total = round(relevance * 0.4 + actionability * 0.3 + consistency * 0.2 + clarity * 0.1, 1)
    return {
        "relevance": round(relevance, 1),
        "actionability": round(actionability, 1),
        "consistency": round(consistency, 1),
        "clarity": round(clarity, 1),
        "total": total,
    }


def generate_rule_based_insights(df: pd.DataFrame, primary_metric: Optional[str], cat_cols: List[str], dt_cols: List[str]) -> List[str]:
    insights = []

    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    if missing_cells > 0:
        insights.append(f"Data quality risk detected: the dataset contains {missing_cells:,} missing values that may affect reporting accuracy.")
    if duplicates > 0:
        insights.append(f"Duplicate record risk detected: {duplicates:,} duplicate rows should be reviewed before external reporting.")

    if primary_metric and cat_cols:
        best_col, best_cat, best_val = top_category_by_metric(df, cat_cols, primary_metric)
        worst_col, worst_cat, worst_val = worst_category_by_metric(df, cat_cols, primary_metric)
        if best_col and best_cat is not None:
            insights.append(f"The strongest driver of {primary_metric} is {best_cat} within {best_col}, contributing {best_val:,.2f} overall.")
        if worst_col and worst_cat is not None:
            insights.append(f"Underperformance appears in {worst_cat} under {worst_col}, making it a priority area for review.")

    if primary_metric and dt_cols:
        trend = calculate_metric_trend(df, primary_metric, dt_cols[0])
        if trend["pct_change"] is not None:
            direction_word = "improved" if trend["pct_change"] >= 0 else "declined"
            insights.append(f"Over time, {primary_metric} has {direction_word} by {abs(trend['pct_change']):.1f}% from the first month to the latest month.")

    return insights[:4]


def generate_statistical_insights(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], dt_cols: List[str], primary_metric: Optional[str]) -> List[str]:
    insights = []

    corr = get_strongest_correlation(df, numeric_cols)
    if corr:
        a, b, val = corr
        relation = "positive" if val >= 0 else "negative"
        insights.append(f"The strongest statistical relationship is a {relation} correlation of {val:.2f} between {a} and {b}.")

    outlier_signal = get_outlier_signal(df, numeric_cols)
    if outlier_signal:
        col, rate = outlier_signal
        insights.append(f"Outlier analysis shows that {col} has the highest anomaly concentration at {rate:.1%}, which may distort averages and trends.")

    if primary_metric and dt_cols:
        trend = calculate_metric_trend(df, primary_metric, dt_cols[0])
        monthly = trend.get("monthly")
        if monthly is not None and len(monthly) >= 3:
            recent = monthly.tail(3).values
            if len(recent) == 3:
                slope = recent[-1] - recent[0]
                direction = "upward" if slope >= 0 else "downward"
                insights.append(f"Recent monthly movement for {primary_metric} shows a {direction} direction across the latest observed periods.")

    if cat_cols and primary_metric:
        grouped = df.groupby(cat_cols[0], dropna=False)[primary_metric].sum(numeric_only=True)
        if not grouped.empty:
            spread = grouped.max() - grouped.min()
            insights.append(f"Category spread analysis indicates a {spread:,.2f} gap between the strongest and weakest values in {cat_cols[0]}." )

    return insights[:4]


def generate_narrative_insights(df: pd.DataFrame, primary_metric: Optional[str], cat_cols: List[str], dt_cols: List[str], quality_score: int) -> List[str]:
    insights = []

    if primary_metric and dt_cols:
        trend = calculate_metric_trend(df, primary_metric, dt_cols[0])
        if trend["pct_change"] is not None:
            if trend["pct_change"] >= 0:
                insights.append(f"Narrative assessment suggests momentum is favorable because {primary_metric} is trending upward over time, which may indicate healthy commercial performance.")
            else:
                insights.append(f"Narrative assessment suggests caution because {primary_metric} is trending downward over time, which may reflect weakening demand, execution gaps, or retention issues.")

    if cat_cols and primary_metric:
        best_col, best_cat, _ = top_category_by_metric(df, cat_cols, primary_metric)
        worst_col, worst_cat, _ = worst_category_by_metric(df, cat_cols, primary_metric)
        if best_col and best_cat:
            insights.append(f"From a business perspective, {best_cat} in {best_col} appears to be a dependable growth segment that should be protected and scaled.")
        if worst_col and worst_cat:
            insights.append(f"The weakest segment appears to be {worst_cat} in {worst_col}, suggesting a need for targeted investigation and intervention.")

    if quality_score < 75:
        insights.append(f"Decision confidence should be moderated because the current data quality score is {quality_score}/100, indicating reporting and interpretation risk.")
    else:
        insights.append(f"Decision confidence is relatively strong because the data quality score is {quality_score}/100, supporting more reliable interpretation.")

    return insights[:4]


def build_evaluation_table(engine_outputs: Dict[str, List[str]], primary_metric: Optional[str], cat_cols: List[str], dt_cols: List[str]) -> pd.DataFrame:
    rows = []
    all_entries = []
    for engine, insights in engine_outputs.items():
        for insight in insights:
            all_entries.append((engine, insight))

    for engine, insight in all_entries:
        peer_texts = [text for e, text in all_entries if text != insight]
        scores = score_insight(insight, primary_metric, peer_texts, cat_cols, dt_cols)
        rows.append(
            {
                "Engine": engine,
                "Insight Type": classify_insight(insight).title(),
                "Insight": insight,
                "Relevance": scores["relevance"],
                "Actionability": scores["actionability"],
                "Consistency": scores["consistency"],
                "Clarity": scores["clarity"],
                "Total Score": scores["total"],
                "Agreement": agreement_label(scores["consistency"]),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Total Score", "Engine"], ascending=[False, True]).reset_index(drop=True)
    return result


def generate_executive_summary(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], dt_cols: List[str]) -> str:
    if df.empty:
        return "No data is available to generate an executive summary."

    metric = get_primary_metric(numeric_cols)
    date_col = get_primary_datetime(dt_cols)

    summary_parts = []
    summary_parts.append(
        f"The dataset contains {len(df):,} records across {df.shape[1]} columns, providing a broad view of the uploaded business data."
    )

    if metric:
        metric_total = pd.to_numeric(df[metric], errors="coerce").sum()
        summary_parts.append(f"The primary business metric appears to be **{metric}**, with a total value of **{metric_total:,.2f}**.")

    if date_col and metric:
        trend = calculate_metric_trend(df, metric, date_col)
        if trend["pct_change"] is not None:
            trend_word = "improved" if trend["pct_change"] >= 0 else "declined"
            summary_parts.append(
                f"Over time, **{metric}** has **{trend_word} by {abs(trend['pct_change']):.1f}%** from the first observed month to the latest one."
            )

    if cat_cols and metric:
        best_col, best_cat, best_val = top_category_by_metric(df, cat_cols, metric)
        worst_col, worst_cat, _ = worst_category_by_metric(df, cat_cols, metric)
        if best_col and best_cat is not None:
            summary_parts.append(
                f"The strongest contribution comes from **{best_cat}** in **{best_col}**, contributing **{best_val:,.2f}** to {metric}."
            )
        if worst_col and worst_cat is not None:
            summary_parts.append(
                f"The weakest contribution appears in **{worst_cat}** under **{worst_col}**, which should be reviewed for underperformance or data issues."
            )

    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    if missing_cells > 0 or duplicates > 0:
        summary_parts.append(
            f"From a data quality perspective, the file contains **{missing_cells:,} missing values** and **{duplicates:,} duplicate rows**, which may affect downstream interpretation."
        )

    return " ".join(summary_parts)


def compute_risk_alerts(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> List[Tuple[str, str]]:
    alerts: List[Tuple[str, str]] = []

    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    if missing_ratio > 0.1:
        alerts.append(("high", f"High missing-data risk detected: {missing_ratio:.1%} of all cells are missing."))
    elif missing_ratio > 0.03:
        alerts.append(("medium", f"Moderate missing-data risk detected: {missing_ratio:.1%} of all cells are missing."))

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        level = "high" if duplicate_count / max(len(df), 1) > 0.05 else "medium"
        alerts.append((level, f"Duplicate-record risk detected: {duplicate_count:,} duplicate rows found."))

    for col in numeric_cols[:4]:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rate = ((series < lower) | (series > upper)).mean()
        if outlier_rate > 0.08:
            alerts.append(("medium", f"Outlier concentration in **{col}** is elevated at {outlier_rate:.1%}."))

    for col in cat_cols[:4]:
        value_share = df[col].astype(str).value_counts(normalize=True, dropna=False)
        if not value_share.empty and value_share.iloc[0] > 0.75:
            alerts.append(("low", f"Category imbalance found in **{col}**: one value accounts for {value_share.iloc[0]:.1%} of rows."))

    if not alerts:
        alerts.append(("low", "No major structural data risks were detected in the uploaded dataset."))

    return alerts[:6]


def generate_recommendations(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], dt_cols: List[str]) -> List[str]:
    recs: List[str] = []
    metric = get_primary_metric(numeric_cols)

    if df.isna().sum().sum() > 0:
        recs.append("Prioritize data cleanup before decision-making, especially in fields with missing values that influence revenue, sales, or performance reporting.")

    if int(df.duplicated().sum()) > 0:
        recs.append("Remove duplicate records before presenting insights externally, as duplicates can overstate totals and distort business conclusions.")

    if cat_cols and metric:
        best_col, best_cat, _ = top_category_by_metric(df, cat_cols, metric)
        worst_col, worst_cat, _ = worst_category_by_metric(df, cat_cols, metric)
        if best_col and best_cat:
            recs.append(f"Protect and replicate the strongest-performing segment: **{best_cat}** in **{best_col}** appears to be a leading value driver.")
        if worst_col and worst_cat:
            recs.append(f"Investigate the weakest area: **{worst_cat}** in **{worst_col}** may need operational, pricing, or retention review.")

    if dt_cols and metric:
        recs.append("Track the primary KPI over time using monthly trend reviews so sudden drops or unusual spikes are identified earlier.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        if not corr.empty:
            corr_vals = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(key=lambda s: s.abs(), ascending=False)
            if not corr_vals.empty:
                a, b = corr_vals.index[0]
                val = corr_vals.iloc[0]
                recs.append(f"Review the relationship between **{a}** and **{b}** (correlation {val:.2f}) to identify controllable drivers of performance.")

    if not recs:
        recs.append("The dataset appears structurally stable. Focus next on defining business-specific KPIs and adding decision rules for action prioritization.")

    return recs[:5]


def choose_chart(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], dt_cols: List[str]):
    metric = get_primary_metric(numeric_cols)
    date_col = get_primary_datetime(dt_cols)

    if date_col and metric:
        temp = df[[date_col, metric]].dropna().copy()
        temp = temp.sort_values(date_col)
        if not temp.empty:
            fig = px.line(temp, x=date_col, y=metric, title=f"{metric} over time")
            return fig, f"Recommended chart: line chart because **{date_col}** is a time field and **{metric}** is numeric."

    if cat_cols and metric:
        col = cat_cols[0]
        grouped = df.groupby(col, dropna=False)[metric].sum(numeric_only=True).reset_index()
        grouped = grouped.sort_values(metric, ascending=False).head(10)
        fig = px.bar(grouped, x=col, y=metric, title=f"Top {col} by {metric}")
        return fig, f"Recommended chart: bar chart because **{col}** is categorical and **{metric}** is numeric."

    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[:2]
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        return fig, f"Recommended chart: scatter plot because both **{x_col}** and **{y_col}** are numeric."

    if numeric_cols:
        metric = numeric_cols[0]
        fig = px.histogram(df, x=metric, nbins=30, title=f"Distribution of {metric}")
        return fig, f"Recommended chart: histogram because **{metric}** is numeric and no strong grouping field was detected."

    return None, "No recommended chart could be generated because no suitable numeric field was found."


def make_download_report(summary: str, risks: List[Tuple[str, str]], recs: List[str], quality_score: int, evaluation_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Section": "Executive Summary", "Detail": summary},
        {"Section": "Data Quality Score", "Detail": quality_score},
    ]
    for level, msg in risks:
        rows.append({"Section": f"Risk Alert ({level.title()})", "Detail": msg})
    for idx, rec in enumerate(recs, start=1):
        rows.append({"Section": f"Recommendation {idx}", "Detail": rec})
    if not evaluation_df.empty:
        top = evaluation_df.iloc[0]
        rows.append({"Section": "Top Insight Engine", "Detail": f"{top['Engine']} with score {top['Total Score']}"})
    return pd.DataFrame(rows)


def render_risk_box(level: str, message: str) -> None:
    css_class = {
        "high": "risk-high",
        "medium": "risk-medium",
        "low": "risk-low",
    }.get(level, "risk-low")
    st.markdown(f'<div class="risk-box {css_class}">{message}</div>', unsafe_allow_html=True)


# =========================
# Header
# =========================
st.markdown('<div class="main-title">Client Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a CSV/XLSX file or try demo mode to generate KPI dashboards, trend analysis, anomaly detection, recommendations, and multi-engine insight evaluation.</div>',
    unsafe_allow_html=True,
)

intro1, intro2, intro3, intro4 = st.columns(4)
intro1.info("**Business Problem**
Teams often receive raw data with little clarity on what matters most.")
intro2.info("**What this platform does**
Converts raw data into summaries, KPIs, charts, risks, and actions.")
intro3.info("**Research Upgrade**
Compares rule-based, statistical, and narrative insight engines side by side.")
intro4.info("**How to use it**
Upload a file or choose demo mode, then filter and review insights.")

st.markdown("---")


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Data Input")
input_mode = st.sidebar.radio(
    "Choose input mode",
    ["Upload your own file", "Use demo sales dataset"],
)

sample_df = create_sample_sales_data()

if input_mode == "Upload your own file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.sidebar.download_button(
        "Download sample sales dataset",
        data=dataframe_to_csv_bytes(sample_df),
        file_name="sample_sales_data.csv",
        mime="text/csv",
    )
    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to begin analysis, or switch to demo mode from the sidebar.")
        st.stop()
    raw_df = load_uploaded_file(uploaded_file)
else:
    raw_df = sample_df.copy()
    st.sidebar.success("Demo mode enabled with a built-in sales dataset.")


df = coerce_datetimes(raw_df)
numeric_cols, categorical_cols, datetime_cols, identifier_cols = infer_column_types(df)
primary_metric = get_primary_metric(numeric_cols)
primary_date = get_primary_datetime(datetime_cols)


# =========================
# Filters
# =========================
st.sidebar.header("Filters")
filtered_df = df.copy()

if primary_date and primary_date in filtered_df.columns:
    min_date = filtered_df[primary_date].dropna().min()
    max_date = filtered_df[primary_date].dropna().max()
    if pd.notna(min_date) and pd.notna(max_date):
        selected_dates = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = filtered_df[
                filtered_df[primary_date].between(pd.to_datetime(start_date), pd.to_datetime(end_date))
            ]

for col in categorical_cols[:3]:
    options = filtered_df[col].dropna().astype(str).unique().tolist()
    if 1 < len(options) <= 30:
        selected = st.sidebar.multiselect(f"Filter {col}", options=sorted(options))
        if selected:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

if filtered_df.empty:
    st.warning("No data remains after applying filters. Please relax one or more filters.")
    st.stop()


# =========================
# KPI metrics
# =========================
quality_score = compute_data_quality_score(filtered_df)
missing_count = int(filtered_df.isna().sum().sum())
duplicate_count = int(filtered_df.duplicated().sum())

metric1, metric2, metric3, metric4, metric5 = st.columns(5)
metric1.metric("Rows", f"{len(filtered_df):,}")
metric2.metric("Columns", filtered_df.shape[1])
metric3.metric("Missing Values", f"{missing_count:,}")
metric4.metric("Duplicates", f"{duplicate_count:,}")
metric5.metric("Data Quality Score", f"{quality_score}/100")

if primary_metric and primary_metric in filtered_df.columns:
    extra1, extra2, extra3 = st.columns(3)
    total_metric = pd.to_numeric(filtered_df[primary_metric], errors="coerce").sum()
    mean_metric = pd.to_numeric(filtered_df[primary_metric], errors="coerce").mean()
    extra1.metric(f"Total {primary_metric}", f"{total_metric:,.2f}")
    extra2.metric(f"Average {primary_metric}", f"{mean_metric:,.2f}")

    best_col, best_cat, best_val = top_category_by_metric(filtered_df, categorical_cols, primary_metric)
    if best_col and best_cat is not None:
        extra3.metric("Best Segment", f"{best_cat}", help=f"Top value from {best_col}: {best_val:,.2f}")

st.markdown("---")


# =========================
# Executive summary / risks / recommendations
# =========================
summary = generate_executive_summary(filtered_df, numeric_cols, categorical_cols, datetime_cols)
risks = compute_risk_alerts(filtered_df, numeric_cols, categorical_cols)
recommendations = generate_recommendations(filtered_df, numeric_cols, categorical_cols, datetime_cols)

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Executive Summary")
    st.markdown(f'<div class="section-card">{summary}</div>', unsafe_allow_html=True)

with right:
    st.subheader("Risk Alerts")
    for level, message in risks:
        render_risk_box(level, message)

st.subheader("Recommendations")
for i, rec in enumerate(recommendations, start=1):
    st.markdown(f"**{i}.** {rec}")

st.markdown("---")


# =========================
# Evaluation layer
# =========================
st.subheader("Insight Evaluation Layer")
st.caption("This research-style section compares how different reasoning engines generate and score insights from the same dataset.")

engine_outputs = {
    "Rule-Based Engine": generate_rule_based_insights(filtered_df, primary_metric, categorical_cols, datetime_cols),
    "Statistical Engine": generate_statistical_insights(filtered_df, numeric_cols, categorical_cols, datetime_cols, primary_metric),
    "Narrative Engine": generate_narrative_insights(filtered_df, primary_metric, categorical_cols, datetime_cols, quality_score),
}

evaluation_df = build_evaluation_table(engine_outputs, primary_metric, categorical_cols, datetime_cols)

engine_tabs = st.tabs(list(engine_outputs.keys()))
for tab, (engine_name, insights) in zip(engine_tabs, engine_outputs.items()):
    with tab:
        if insights:
            for idx, insight in enumerate(insights, start=1):
                st.markdown(f"**{idx}.** {insight}")
        else:
            st.info("This engine did not generate enough evidence-based insights for the current dataset.")

st.markdown("### Comparison Table")
st.dataframe(evaluation_df, width="stretch", hide_index=True)

if not evaluation_df.empty:
    avg_scores = evaluation_df.groupby("Engine")["Total Score"].mean().reset_index()
    avg_fig = px.bar(avg_scores, x="Engine", y="Total Score", title="Average Insight Score by Engine")
    st.plotly_chart(avg_fig, width="stretch")

    top_engine = avg_scores.sort_values("Total Score", ascending=False).iloc[0]
    st.success(f"Top-performing engine in the current run: **{top_engine['Engine']}** with an average score of **{top_engine['Total Score']:.1f}**.")

st.markdown("---")


# =========================
# Recommended chart + additional visuals
# =========================
st.subheader("Recommended Visualization")
recommended_fig, chart_reason = choose_chart(filtered_df, numeric_cols, categorical_cols, datetime_cols)
st.caption(chart_reason)
if recommended_fig is not None:
    st.plotly_chart(recommended_fig, width="stretch")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    if categorical_cols and primary_metric:
        cat_col = categorical_cols[0]
        grouped = (
            filtered_df.groupby(cat_col, dropna=False)[primary_metric]
            .sum(numeric_only=True)
            .reset_index()
            .sort_values(primary_metric, ascending=False)
            .head(10)
        )
        fig = px.bar(grouped, x=cat_col, y=primary_metric, title=f"Top {cat_col} by {primary_metric}")
        st.plotly_chart(fig, width="stretch")

with viz_col2:
    if len(numeric_cols) >= 1:
        hist_metric = primary_metric or numeric_cols[0]
        fig = px.histogram(filtered_df, x=hist_metric, nbins=30, title=f"Distribution of {hist_metric}")
        st.plotly_chart(fig, width="stretch")

if len(numeric_cols) >= 2:
    st.subheader("Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr(numeric_only=True)
    corr_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric Correlation Matrix")
    st.plotly_chart(corr_fig, width="stretch")

st.markdown("---")


# =========================
# Data preview and export
# =========================
st.subheader("Preview Data")
st.dataframe(filtered_df.head(100), width="stretch")

report_df = make_download_report(summary, risks, recommendations, quality_score, evaluation_df)
report_csv = dataframe_to_csv_bytes(report_df)
filtered_csv = dataframe_to_csv_bytes(filtered_df)
evaluation_csv = dataframe_to_csv_bytes(evaluation_df) if not evaluation_df.empty else b""

export1, export2, export3 = st.columns(3)
export1.download_button(
    "Download filtered dataset as CSV",
    data=filtered_csv,
    file_name="filtered_client_intelligence_data.csv",
    mime="text/csv",
)
export2.download_button(
    "Download insights report as CSV",
    data=report_csv,
    file_name="client_intelligence_insights_report.csv",
    mime="text/csv",
)
export3.download_button(
    "Download evaluation table as CSV",
    data=evaluation_csv,
    file_name="insight_evaluation_table.csv",
    mime="text/csv",
    disabled=evaluation_df.empty,
)

st.markdown("---")
st.caption(
    "Next research upgrade: replace the Narrative Engine with a real LLM-backed engine and compare prompt outputs against rule-based and statistical baselines."
)
