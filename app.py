import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Client Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# =========================
# Constants
# =========================
ID_KEYWORDS = [
    "id", "order_id", "product_id", "customer_id", "invoice", "transaction_id",
    "user_id", "record_id", "item_id", "code", "sku", "empid", "employeeid",
    "reference", "ref", "account_id", "customer_code"
]

BUSINESS_METRIC_KEYWORDS = [
    "sales", "revenue", "profit", "amount", "price", "cost", "quantity",
    "income", "expense", "salary", "rate", "score", "total", "value",
    "count", "margin", "growth", "hours", "workingyears", "years"
]

CATEGORY_PRIORITY_KEYWORDS = [
    "country", "region", "department", "category", "segment", "product",
    "jobrole", "educationfield", "gender", "attrition", "businesstravel",
    "maritalstatus", "overtime", "agegroup", "customer", "channel"
]

BUSINESS_CONTEXTS = [
    "General Exploration",
    "Sales Performance",
    "Customer Retention",
    "HR Attrition",
    "Operational Efficiency",
    "Financial Performance"
]

TRIVIAL_TIME_COLUMNS = {
    "year", "month", "week", "week_of_year", "day", "day_of_week",
    "quarter", "is_weekend", "order_hour", "hour"
}

LOW_VALUE_METRIC_COLUMNS = {
    "year", "month", "week_of_year", "day_of_week", "order_hour", "is_weekend"
}

# =========================
# Helpers
# =========================
def safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        converted = pd.to_datetime(series, errors="coerce")
        valid_ratio = converted.notna().mean()
        return converted if valid_ratio > 0.7 else series
    except Exception:
        return series


def normalize_column_name(col: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(col).strip().lower())


def is_integer_like(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False
    return np.all(np.isclose(clean, clean.astype(int)))


def detect_column_types(df: pd.DataFrame):
    temp_df = df.copy()
    temp_df.columns = [str(c) for c in temp_df.columns]

    datetime_cols = []
    datetime_confidence = {}

    for col in temp_df.columns:
        if temp_df[col].dtype == "object":
            converted = pd.to_datetime(temp_df[col], errors="coerce")
            valid_ratio = converted.notna().mean()
            datetime_confidence[col] = round(valid_ratio * 100, 2)
            if valid_ratio > 0.7:
                temp_df[col] = converted
                datetime_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(temp_df[col]):
            datetime_cols.append(col)
            datetime_confidence[col] = 100.0

    numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
    all_cols = temp_df.columns.tolist()

    identifier_cols = []
    for col in all_cols:
        col_lower = col.lower().strip()

        if any(keyword in col_lower for keyword in ID_KEYWORDS):
            identifier_cols.append(col)
            continue

        if col in numeric_cols:
            nunique_ratio = temp_df[col].nunique(dropna=True) / max(len(temp_df), 1)
            metric_like = any(k in col_lower for k in BUSINESS_METRIC_KEYWORDS)
            if nunique_ratio > 0.95 and is_integer_like(temp_df[col]) and not metric_like:
                identifier_cols.append(col)

    numeric_measure_cols = [c for c in numeric_cols if c not in identifier_cols]
    categorical_cols = [
        c for c in all_cols
        if c not in numeric_cols and c not in datetime_cols
    ]

    return temp_df, numeric_measure_cols, categorical_cols, datetime_cols, identifier_cols, datetime_confidence


def format_number(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float, np.number)):
        return f"{value:,.2f}"
    return str(value)


def compute_metric(series: pd.Series, agg: str):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    if agg == "sum":    return clean.sum()
    if agg == "mean":   return clean.mean()
    if agg == "median": return clean.median()
    if agg == "min":    return clean.min()
    if agg == "max":    return clean.max()
    if agg == "count":  return clean.count()
    if agg == "std":    return clean.std()
    return np.nan


def get_outlier_mask(series: pd.Series):
    clean = pd.to_numeric(series, errors="coerce")
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (clean < lower) | (clean > upper)


def infer_confidence(label, value=None, extra=None):
    label_lower = label.lower()
    if "trend" in label_lower or "decreased" in label_lower or "increased" in label_lower:
        return "High"
    if "outlier" in label_lower or "anomal" in label_lower:
        return "Medium"
    if "correlation" in label_lower:
        if value is not None and abs(value) >= 0.7:
            return "Medium"
        return "Low"
    if "top" in label_lower or "segment" in label_lower or "country" in label_lower:
        return "High"
    return "Medium"


def classify_dataset(categorical_cols, numeric_cols, all_cols):
    joined = " ".join([c.lower() for c in all_cols])
    if any(word in joined for word in ["attrition", "jobrole", "department", "overtime", "employee"]):
        return "hr"
    if any(word in joined for word in ["sales", "revenue", "product", "customer", "order", "quantity"]):
        return "sales"
    if any(word in joined for word in ["profit", "expense", "cost", "income", "margin", "finance"]):
        return "finance"
    return "general"


def pick_best_metric_column(numeric_cols, context="General Exploration"):
    if not numeric_cols:
        return None
    context_priority = {
        "Sales Performance": ["sales", "revenue", "amount", "quantity", "price", "profit"],
        "Customer Retention": ["customer", "revenue", "sales", "count", "frequency", "orders"],
        "HR Attrition": ["attrition", "income", "salary", "years", "hours", "rate"],
        "Operational Efficiency": ["time", "hours", "cost", "quantity", "rate", "productivity"],
        "Financial Performance": ["profit", "revenue", "expense", "cost", "margin", "income"],
        "General Exploration": BUSINESS_METRIC_KEYWORDS
    }
    priorities = context_priority.get(context, BUSINESS_METRIC_KEYWORDS)
    for keyword in priorities:
        for col in numeric_cols:
            if keyword in col.lower() and col.lower() not in LOW_VALUE_METRIC_COLUMNS:
                return col
    filtered = [col for col in numeric_cols if col.lower() not in LOW_VALUE_METRIC_COLUMNS]
    if filtered:
        return filtered[0]
    return numeric_cols[0]


def pick_best_category_column(categorical_cols, context="General Exploration"):
    if not categorical_cols:
        return None
    context_priority = {
        "Sales Performance": ["region", "country", "segment", "category", "product", "customer"],
        "Customer Retention": ["customer", "segment", "region", "channel", "category"],
        "HR Attrition": ["department", "jobrole", "gender", "maritalstatus", "overtime", "educationfield"],
        "Operational Efficiency": ["department", "region", "category", "team", "process"],
        "Financial Performance": ["region", "department", "category", "segment"],
        "General Exploration": CATEGORY_PRIORITY_KEYWORDS
    }
    priorities = context_priority.get(context, CATEGORY_PRIORITY_KEYWORDS)
    for keyword in priorities:
        for col in categorical_cols:
            if keyword in col.lower():
                return col
    return categorical_cols[0]


def pick_best_datetime_column(datetime_cols):
    return datetime_cols[0] if datetime_cols else None


def strongest_correlations(df, numeric_cols):
    valid_numeric_cols = []
    for col in numeric_cols:
        clean = pd.to_numeric(df[col], errors="coerce")
        if clean.nunique(dropna=True) > 1:
            valid_numeric_cols.append(col)

    if len(valid_numeric_cols) < 2:
        return [], valid_numeric_cols

    corr = df[valid_numeric_cols].corr(numeric_only=True)
    pairs = []

    def is_trivial_pair(col1, col2):
        c1 = col1.lower()
        c2 = col2.lower()
        trivial_time = c1 in TRIVIAL_TIME_COLUMNS and c2 in TRIVIAL_TIME_COLUMNS
        exact_family = (
            ("month" in c1 and "week" in c2) or ("month" in c2 and "week" in c1) or
            ("year" in c1 and "month" in c2) or ("year" in c2 and "month" in c1) or
            ("day" in c1 and "hour" in c2) or ("day" in c2 and "hour" in c1)
        )
        return trivial_time or exact_family

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if pd.notna(val) and not is_trivial_pair(col1, col2):
                pairs.append((col1, col2, val))

    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5], valid_numeric_cols


def recommend_chart_type(df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols):
    x_is_numeric = x_col in numeric_cols
    y_is_numeric = y_col in numeric_cols if y_col else False
    x_is_categorical = x_col in categorical_cols or x_col in identifier_cols
    x_is_datetime = x_col in datetime_cols
    unique_x = df[x_col].nunique(dropna=True)

    if x_is_datetime and y_is_numeric:
        return "line", "A line chart is best for showing how the selected metric changes over time."
    if x_is_numeric and y_is_numeric:
        if x_col in identifier_cols:
            return "bar_top_n", "This looks like an identifier field, so a Top N bar chart is more meaningful."
        return "scatter", "A scatter plot is useful for checking relationships, clusters, and unusual values."
    if x_is_categorical and y_is_numeric:
        if unique_x <= 12:
            return "bar", "A bar chart is appropriate for comparing a numeric metric across categories."
        return "bar_top_n", "This field has many categories, so a Top N horizontal bar chart improves readability."
    if x_is_numeric and not y_col:
        return "histogram", "A histogram helps explain the distribution of a numeric variable."
    if x_is_categorical and not y_col:
        if unique_x <= 6:
            return "pie", "A pie chart can work when there are very few categories."
        return "count_bar", "A count bar chart is better for comparing category frequencies."
    return "bar", "A bar chart is the safest default for this selection."


def aggregate_data(grouped_df, x_col, y_col, agg_method):
    if agg_method == "sum":    return grouped_df.groupby(x_col, as_index=False)[y_col].sum()
    if agg_method == "mean":   return grouped_df.groupby(x_col, as_index=False)[y_col].mean()
    if agg_method == "median": return grouped_df.groupby(x_col, as_index=False)[y_col].median()
    if agg_method == "count":  return grouped_df.groupby(x_col, as_index=False)[y_col].count()
    return grouped_df.groupby(x_col, as_index=False)[y_col].sum()


def build_chart(df, x_col, y_col, chart_type, top_n=10, agg_method="sum"):
    chart_df = df.copy()

    if chart_type == "line":
        chart_df = chart_df[[x_col, y_col]].dropna().sort_values(by=x_col)
        return px.line(chart_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
    if chart_type == "scatter":
        chart_df = chart_df[[x_col, y_col]].dropna()
        return px.scatter(chart_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", opacity=0.7)
    if chart_type == "bar":
        grouped = aggregate_data(chart_df[[x_col, y_col]].dropna(), x_col, y_col, agg_method)
        grouped = grouped.sort_values(by=y_col, ascending=False)
        return px.bar(grouped, x=x_col, y=y_col, title=f"{agg_method.title()} {y_col} by {x_col}")
    if chart_type == "bar_top_n":
        grouped = aggregate_data(chart_df[[x_col, y_col]].dropna(), x_col, y_col, agg_method)
        grouped = grouped.sort_values(by=y_col, ascending=False).head(top_n).sort_values(by=y_col, ascending=True)
        return px.bar(grouped, x=y_col, y=x_col, orientation="h",
                      title=f"Top {top_n} {x_col} by {agg_method.title()} {y_col}")
    if chart_type == "histogram":
        chart_df = chart_df[[x_col]].dropna()
        return px.histogram(chart_df, x=x_col, nbins=30, title=f"Distribution of {x_col}")
    if chart_type == "pie":
        counts = chart_df[x_col].value_counts(dropna=False).reset_index()
        counts.columns = [x_col, "count"]
        return px.pie(counts, names=x_col, values="count", title=f"{x_col} distribution")
    if chart_type == "count_bar":
        counts = chart_df[x_col].value_counts(dropna=False).head(15).reset_index()
        counts.columns = [x_col, "count"]
        return px.bar(counts, x=x_col, y="count", title=f"Top categories in {x_col}")
    return px.bar(title="No chart available")


def moving_average_projection(df, date_col, value_col, window=7, projection_points=8):
    ts = df[[date_col, value_col]].dropna().copy()
    ts = ts.sort_values(date_col)
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
    ts = ts.dropna()

    if ts.empty or len(ts) < 5:
        return None

    grouped = ts.groupby(date_col, as_index=False)[value_col].sum()
    grouped["moving_avg"] = grouped[value_col].rolling(window=min(window, len(grouped)), min_periods=1).mean()

    last_date = grouped[date_col].max()
    inferred_freq = pd.infer_freq(grouped[date_col].sort_values())

    if inferred_freq is None:
        date_diffs = grouped[date_col].sort_values().diff().dropna()
        avg_diff = date_diffs.mode().iloc[0] if not date_diffs.empty else pd.Timedelta(days=1)
        future_dates = [last_date + avg_diff * (i + 1) for i in range(projection_points)]
    else:
        future_dates = pd.date_range(start=last_date, periods=projection_points + 1, freq=inferred_freq)[1:]

    last_ma = grouped["moving_avg"].iloc[-1]
    forecast_df = pd.DataFrame({
        date_col: future_dates,
        value_col: [last_ma] * len(future_dates),
        "series_type": ["Projection"] * len(future_dates)
    })

    actual_df = grouped.copy()
    actual_df["series_type"] = "Actual"

    return pd.concat([actual_df[[date_col, value_col, "series_type"]], forecast_df], ignore_index=True)


# =========================
# Anomaly Detection Engine
# =========================
def detect_anomalies(series: pd.Series, method: str = "IQR") -> dict:
    """Run anomaly detection and return full results dict with explainability metadata."""
    mean, std = series.mean(), series.std()
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_lower = Q1 - 1.5 * IQR
    iqr_upper = Q3 + 1.5 * IQR
    z_scores = (series - mean) / std if std > 0 else pd.Series(0.0, index=series.index)

    iqr_mask   = (series < iqr_lower) | (series > iqr_upper)
    zscore_mask = z_scores.abs() > 3

    if method == "IQR":
        mask = iqr_mask
    elif method == "Z-Score (>3σ)":
        mask = zscore_mask
    else:
        mask = iqr_mask | zscore_mask

    return {
        "mask": mask,
        "z_scores": z_scores,
        "iqr_lower": iqr_lower,
        "iqr_upper": iqr_upper,
        "mean": mean,
        "std": std,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "iqr_count": int(iqr_mask.sum()),
        "zscore_count": int(zscore_mask.sum()),
        "total_anomalies": int(mask.sum()),
        "anomaly_pct": round(mask.mean() * 100, 2),
    }


def build_anomaly_df(df: pd.DataFrame, col: str, results: dict) -> pd.DataFrame:
    """Attach per-row anomaly metadata to the dataframe."""
    adf = df.copy()
    adf["_is_anomaly"]    = results["mask"]
    adf["_z_score"]       = results["z_scores"].round(3)
    adf["_deviation_pct"] = ((adf[col] - results["mean"]) / results["mean"] * 100).round(1)
    adf["_trigger"]       = adf[col].apply(
        lambda v: "HIGH — above IQR upper fence" if v > results["iqr_upper"]
                  else ("LOW — below IQR lower fence" if v < results["iqr_lower"] else "Normal")
    )
    adf["_severity"] = adf["_z_score"].abs().apply(
        lambda z: "🔴 Extreme" if z > 6 else ("🟠 High" if z > 4 else ("🟡 Moderate" if z > 2 else "🟢 Mild"))
    )
    return adf


# =========================
# FDE Layer
# =========================
def generate_top_priorities(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    priorities = []
    best_metric   = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date     = pick_best_datetime_column(datetime_cols)

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            outlier_mask  = get_outlier_mask(df[best_metric])
            outlier_count = int(outlier_mask.sum())
            outlier_pct   = (outlier_count / len(metric_series) * 100) if len(metric_series) > 0 else 0
            if outlier_pct >= 10:
                priorities.append({
                    "priority": "Medium",
                    "title": "Data anomaly review",
                    "detail": f"'{best_metric}' has {outlier_count:,} potential anomalies ({outlier_pct:.1f}%), so summary metrics may be distorted."
                })

    if best_date and best_metric:
        trend = (df[[best_date, best_metric]].dropna()
                 .groupby(best_date)[best_metric].sum().sort_index())
        if len(trend) >= 2:
            first_val = trend.iloc[0]
            last_val  = trend.iloc[-1]
            if first_val != 0:
                change_pct = ((last_val - first_val) / first_val) * 100
                if change_pct <= -20:
                    priorities.insert(0, {
                        "priority": "High",
                        "title": "Metric decline",
                        "detail": f"'{best_metric}' declined by {abs(change_pct):.2f}% over the observed period."
                    })
                elif change_pct >= 20:
                    priorities.insert(0, {
                        "priority": "High",
                        "title": "Metric shift",
                        "detail": f"'{best_metric}' increased by {change_pct:.2f}% over the observed period, which may indicate a major structural change."
                    })

    if best_category and best_metric:
        grouped = (df[[best_category, best_metric]].dropna()
                   .groupby(best_category)[best_metric].sum().sort_values(ascending=False))
        if len(grouped) >= 2:
            total     = grouped.sum()
            top_share = (grouped.iloc[0] / total * 100) if total != 0 else 0
            if top_share >= 40:
                priorities.append({
                    "priority": "Medium",
                    "title": "Concentration risk",
                    "detail": f"The top '{best_category}' group contributes {top_share:.1f}% of total '{best_metric}', suggesting dependency risk."
                })

    if not priorities:
        priorities.append({
            "priority": "Low",
            "title": "No major risk signal",
            "detail": "No urgent risk pattern was automatically detected. Continue with deeper segmented analysis."
        })

    return priorities[:3]


def generate_business_findings(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    findings = []
    best_metric   = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date     = pick_best_datetime_column(datetime_cols)

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            text = (f"The primary business metric '{best_metric}' totals {metric_series.sum():,.2f} "
                    f"with an average of {metric_series.mean():,.2f}.")
            findings.append({"text": text, "confidence": infer_confidence(text)})

            outlier_mask  = get_outlier_mask(df[best_metric])
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(metric_series) * 100) if len(metric_series) > 0 else 0
                text = (f"'{best_metric}' contains {outlier_count:,} potential anomalies "
                        f"({outlier_pct:.1f}% of non-null records), so averages and trends should be interpreted carefully.")
                findings.append({"text": text, "confidence": infer_confidence(text)})

    if best_category and best_metric:
        grouped = (df[[best_category, best_metric]].dropna()
                   .groupby(best_category)[best_metric].sum().sort_values(ascending=False))
        if not grouped.empty:
            text = (f"Performance is uneven across '{best_category}'. "
                    f"'{grouped.index[0]}' leads with {grouped.iloc[0]:,.2f} in '{best_metric}', "
                    f"while '{grouped.index[-1]}' contributes only {grouped.iloc[-1]:,.2f}.")
            findings.append({"text": text, "confidence": infer_confidence(text)})

    if best_date and best_metric:
        trend = (df[[best_date, best_metric]].dropna()
                 .groupby(best_date)[best_metric].sum().sort_index())
        if len(trend) >= 2:
            first_val = trend.iloc[0]
            last_val  = trend.iloc[-1]
            direction = "increased" if last_val > first_val else "decreased"
            if first_val != 0:
                change_pct = ((last_val - first_val) / first_val) * 100
                text = (f"Over time, '{best_metric}' has {direction} from {first_val:,.2f} "
                        f"to {last_val:,.2f} ({change_pct:.2f}%). This is a material shift that deserves attention.")
            else:
                text = (f"Over time, '{best_metric}' has {direction} from {first_val:,.2f} to {last_val:,.2f}.")
            findings.append({"text": text, "confidence": infer_confidence(text)})

    corr_pairs, _ = strongest_correlations(df, numeric_cols)
    if corr_pairs:
        col1, col2, corr_val = corr_pairs[0]
        direction = "positive" if corr_val > 0 else "negative"
        strength  = "strong" if abs(corr_val) >= 0.7 else "moderate" if abs(corr_val) >= 0.4 else "weak"
        text = (f"The strongest non-trivial measurable relationship is a {strength} {direction} correlation "
                f"between '{col1}' and '{col2}' ({corr_val:.2f}). This may indicate a business driver, "
                f"but it should not be treated as proof of causation.")
        findings.append({"text": text, "confidence": infer_confidence(text, value=corr_val)})

    return findings


def generate_recommendations(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    recommendations = []
    best_metric   = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date     = pick_best_datetime_column(datetime_cols)

    if best_metric:
        outlier_count = int(get_outlier_mask(df[best_metric]).sum())
        if outlier_count > 0:
            recommendations.append(
                f"Review the {outlier_count:,} unusual values in '{best_metric}' first, because they may be inflating or masking the true business pattern."
            )

    if best_date and best_metric:
        trend = (df[[best_date, best_metric]].dropna()
                 .groupby(best_date)[best_metric].sum().sort_index())
        if len(trend) >= 2 and trend.iloc[0] != 0:
            change_pct = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100
            if change_pct < 0:
                recommendations.append(
                    f"Investigate the period where '{best_metric}' declined by {abs(change_pct):.2f}% and compare that period against pricing, demand, and category mix changes."
                )

    if best_category and best_metric:
        grouped = (df[[best_category, best_metric]].dropna()
                   .groupby(best_category)[best_metric].sum().sort_values(ascending=False))
        if len(grouped) >= 2:
            recommendations.append(
                f"Compare high-performing '{best_category}' group '{grouped.index[0]}' against low-performing group '{grouped.index[-1]}' to identify operational, market, or customer-mix differences."
            )

    context_recs = {
        "Sales Performance": "Review whether revenue concentration is too dependent on a small number of regions, categories, or products.",
        "Customer Retention": "Segment customers by value and repeat behavior to determine where retention interventions will have the highest payoff.",
        "HR Attrition": "Test whether attrition risk is concentrated in specific departments, tenure bands, or overtime-heavy groups.",
        "Operational Efficiency": "Use high-variance process or time metrics to isolate bottlenecks and inconsistent execution.",
        "Financial Performance": "Separate revenue growth from cost pressure so that apparent performance gains do not hide profitability leakage.",
        "General Exploration": "Use the strongest trend, anomaly, and segment imbalance signals as the starting point for deeper root-cause analysis."
    }
    recommendations.append(context_recs.get(business_context, context_recs["General Exploration"]))

    seen  = set()
    final = []
    for rec in recommendations:
        if rec not in seen:
            final.append(rec)
            seen.add(rec)
    return final[:6]


def generate_client_story(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    best_metric   = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date     = pick_best_datetime_column(datetime_cols)

    what_happened       = "Not enough signal detected to summarize what happened."
    why_it_matters      = "This dataset needs more business context or stronger field coverage."
    what_to_investigate = "Review data quality, field definitions, and analysis scope."
    suggested_actions   = "Validate the business objective and refine the analysis with a target question."

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            what_happened = (f"The dataset suggests that '{best_metric}' is the most decision-relevant measure, "
                             f"with meaningful variation across the records.")

    if best_category and best_metric:
        grouped = (df[[best_category, best_metric]].dropna()
                   .groupby(best_category)[best_metric].sum().sort_values(ascending=False))
        if not grouped.empty:
            why_it_matters = (f"Performance is not evenly distributed. The strongest contribution comes from "
                              f"'{grouped.index[0]}', which suggests concentration risk and uneven business "
                              f"performance across '{best_category}'.")

    if best_date and best_metric:
        trend = (df[[best_date, best_metric]].dropna()
                 .groupby(best_date)[best_metric].sum().sort_index())
        if len(trend) >= 2:
            direction = "improved" if trend.iloc[-1] > trend.iloc[0] else "weakened"
            what_to_investigate = (f"Investigate when '{best_metric}' {direction} over time and compare that period "
                                   f"with category-level shifts, anomalies, or operational changes.")

    suggested_actions = ("Prioritize the main metric, review high- and low-performing segments, validate outliers, "
                         "and use the resulting pattern differences to guide the next business decision.")

    return {
        "What happened": what_happened,
        "Why it matters": why_it_matters,
        "What to investigate next": what_to_investigate,
        "Suggested actions": suggested_actions
    }


def generate_executive_summary(df, numeric_cols, categorical_cols, datetime_cols, all_cols, business_context):
    summary     = []
    best_metric   = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date     = pick_best_datetime_column(datetime_cols)

    if best_date and best_metric:
        temp = df[[best_date, best_metric]].dropna().copy()
        temp[best_metric] = pd.to_numeric(temp[best_metric], errors="coerce")
        temp = temp.dropna()
        if not temp.empty:
            trend = temp.groupby(best_date)[best_metric].sum().sort_index()
            if len(trend) >= 2:
                direction = "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing"
                summary.append(f"📈 Overall trend shows a {direction} pattern in '{best_metric}'.")

    if best_category:
        if best_metric:
            grouped = (df[[best_category, best_metric]].dropna()
                       .groupby(best_category)[best_metric].sum().sort_values(ascending=False))
            if not grouped.empty:
                summary.append(f"🏆 Leading '{best_category}' segment is '{grouped.index[0]}' based on total '{best_metric}'.")
        else:
            vc = df[best_category].astype(str).value_counts(dropna=False)
            if not vc.empty:
                summary.append(f"🏆 Most frequent value in '{best_category}' is '{vc.index[0]}' with {vc.iloc[0]:,} records.")

    if best_metric:
        outlier_count = int(get_outlier_mask(df[best_metric]).sum())
        if outlier_count > 0:
            summary.append(f"⚠️ Detected {outlier_count} anomalies in '{best_metric}', which may influence averages and trends.")
        else:
            summary.append(f"✅ No major anomalies detected in '{best_metric}' based on the IQR method.")

    corr_pairs, _ = strongest_correlations(df, numeric_cols)
    if corr_pairs:
        c1, c2, val = corr_pairs[0]
        summary.append(f"🔗 Strongest non-trivial measurable relationship is between '{c1}' and '{c2}' ({val:.2f}).")

    summary.append(f"💡 Business context selected: {business_context}. Recommendations below are tailored accordingly.")
    return summary


# =========================
# Downloads
# =========================
def df_to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def text_download(findings, recommendations, exec_summary, client_story, top_priorities):
    buffer = io.StringIO()
    buffer.write("Client Intelligence Platform - Executive Summary\n")
    buffer.write("=" * 70 + "\n\n")
    for i, item in enumerate(exec_summary, start=1):
        buffer.write(f"{i}. {item}\n")
    buffer.write("\nTop Priorities\n" + "=" * 70 + "\n\n")
    for i, item in enumerate(top_priorities, start=1):
        buffer.write(f"{i}. [{item['priority']}] {item['title']}: {item['detail']}\n")
    buffer.write("\nClient Story\n" + "=" * 70 + "\n\n")
    for key, value in client_story.items():
        buffer.write(f"{key}: {value}\n\n")
    buffer.write("Key Findings\n" + "=" * 70 + "\n\n")
    for i, item in enumerate(findings, start=1):
        buffer.write(f"{i}. {item['text']} (Confidence: {item['confidence']})\n")
    buffer.write("\nRecommended Focus Areas\n" + "=" * 70 + "\n\n")
    for i, item in enumerate(recommendations, start=1):
        buffer.write(f"{i}. {item}\n")
    return buffer.getvalue().encode("utf-8")


def anomaly_csv_download(adf: pd.DataFrame, col: str) -> bytes:
    export = (adf[adf["_is_anomaly"]]
              .drop(columns=["_is_anomaly"])
              .rename(columns={
                  "_z_score": "z_score",
                  "_deviation_pct": "deviation_pct",
                  "_trigger": "anomaly_trigger",
                  "_severity": "severity_tier"
              }))
    return export.to_csv(index=False).encode("utf-8")


# =========================
# Anomaly Tab Renderer
# =========================
def render_anomaly_tab(filtered_df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    st.markdown("Understand **which** records are flagged, **why** each one was caught, "
                "and **what** the anomalies mean for your analysis.")

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        target_col = st.selectbox(
            "Metric to analyse", numeric_cols,
            index=numeric_cols.index("sales_amount_gbp") if "sales_amount_gbp" in numeric_cols else 0,
            key="anom_metric"
        )
    with ctrl2:
        method = st.selectbox(
            "Detection method",
            ["IQR", "Z-Score (>3σ)", "Both (union)"],
            key="anom_method"
        )
    with ctrl3:
        top_n_table = st.number_input("Rows in table", min_value=10, max_value=500, value=50, step=10,
                                      key="anom_top_n")

    # ── Run detection ─────────────────────────────────────────────────────────
    series  = pd.to_numeric(filtered_df[target_col], errors="coerce").fillna(filtered_df[target_col].median())
    results = detect_anomalies(series, method)
    adf     = build_anomaly_df(filtered_df, target_col, results)

    # ── Method explainer ──────────────────────────────────────────────────────
    with st.expander("ℹ️  How does this detection method work?", expanded=False):
        if method == "IQR":
            st.markdown(f"""
**Interquartile Range (IQR) — how the fences are calculated**

| Statistic | Value |
|-----------|-------|
| 25th percentile (Q1) | **{results['Q1']:.2f}** |
| 75th percentile (Q3) | **{results['Q3']:.2f}** |
| IQR (Q3 − Q1) | **{results['IQR']:.2f}** |
| Lower fence (Q1 − 1.5 × IQR) | **{results['iqr_lower']:.2f}** |
| Upper fence (Q3 + 1.5 × IQR) | **{results['iqr_upper']:.2f}** |

Any record outside these fences is flagged. This method is **robust to extreme values** because
the fences are anchored on the middle 50% of data, not the mean — so a handful of giant outliers
will not distort the thresholds.

> **Note:** IQR can be conservative on skewed distributions. Not every flagged record is a
> business problem — use the severity tiers below to prioritise which ones to investigate first.
""")
        elif method == "Z-Score (>3σ)":
            st.markdown(f"""
**Z-Score — threshold at ±3 standard deviations**

| Statistic | Value |
|-----------|-------|
| Mean | **{results['mean']:.2f}** |
| Standard deviation | **{results['std']:.2f}** |
| Threshold | **±3σ** |

A record is flagged when `|z| > 3`. Under a normal distribution, only ~0.3% of values should
exceed this level. This method is sensitive to the mean and std, both of which are themselves
influenced by extreme outliers.

> **Note:** If the distribution is heavily skewed (as with sales data), Z-score will flag fewer
> records than IQR but may miss moderate anomalies that sit just outside the bulk of the data.
""")
        else:
            st.markdown("""
**Both (Union) — IQR ∪ Z-Score**

A record is flagged if caught by **either** method. This maximises recall (fewer missed anomalies)
at the cost of a higher false-positive rate. Use this when you want a comprehensive sweep before
data cleaning, and then triage by severity tier.
""")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("🔍 Total Anomalies",   f"{results['total_anomalies']:,}")
    kc2.metric("📊 Anomaly Rate",       f"{results['anomaly_pct']}%")
    kc3.metric("📈 IQR Upper Fence",   f"{results['iqr_upper']:.2f}")
    kc4.metric("📉 IQR Lower Fence",   f"{results['iqr_lower']:.2f}")
    avg_normal  = adf.loc[~adf["_is_anomaly"], target_col].mean()
    avg_anomaly = adf.loc[adf["_is_anomaly"],  target_col].mean()
    lift = round((avg_anomaly / avg_normal - 1) * 100, 1) if avg_normal and avg_normal != 0 else 0
    kc5.metric("⬆️ Avg vs Normal",     f"+{lift}%" if lift >= 0 else f"{lift}%")

    st.divider()

    # ── Row 1: Distribution chart + Severity donut ────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("**Distribution: Normal vs Anomaly**")
        normal_vals    = filtered_df.loc[~results["mask"], target_col]
        anomalous_vals = filtered_df.loc[results["mask"],  target_col]

        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(x=normal_vals,    name="Normal",  nbinsx=80,
                                        marker_color="#4FC3F7", opacity=0.75))
        dist_fig.add_trace(go.Histogram(x=anomalous_vals, name="Anomaly", nbinsx=80,
                                        marker_color="#FF6B6B", opacity=0.85))
        dist_fig.add_vline(x=results["iqr_upper"], line_dash="dash", line_color="orange",
                           annotation_text=f"Upper fence ({results['iqr_upper']:.1f})",
                           annotation_position="top right")
        if results["iqr_lower"] > float(filtered_df[target_col].min()):
            dist_fig.add_vline(x=results["iqr_lower"], line_dash="dash", line_color="orange",
                               annotation_text=f"Lower fence ({results['iqr_lower']:.1f})")
        dist_fig.update_layout(barmode="overlay", height=340,
                               xaxis_title=target_col, yaxis_title="Count",
                               legend=dict(orientation="h"))
        st.plotly_chart(dist_fig, use_container_width=True)

    with right_col:
        st.markdown("**Anomaly Severity Breakdown**")
        severity_counts = (adf[adf["_is_anomaly"]]["_severity"]
                           .value_counts().reset_index())
        severity_counts.columns = ["Severity", "Count"]
        donut_fig = px.pie(
            severity_counts, names="Severity", values="Count", hole=0.55,
            color_discrete_sequence=["#FF4444", "#FF8C00", "#FFD700", "#90EE90"]
        )
        donut_fig.update_layout(height=340, showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(donut_fig, use_container_width=True)

    st.divider()

    # ── Row 2: Z-score scatter ────────────────────────────────────────────────
    st.markdown("**Z-Score Plot — distance of each record from the mean**")
    sample_size = min(3000, len(adf))
    sample      = adf.sample(sample_size, random_state=42).copy()
    sample["_label"] = sample["_is_anomaly"].map({True: "Anomaly", False: "Normal"})

    z_fig = px.scatter(
        sample, x=sample.index, y="_z_score", color="_label",
        color_discrete_map={"Anomaly": "#FF6B6B", "Normal": "#4FC3F7"},
        opacity=0.55, height=340,
        labels={"_z_score": "Z-Score", "index": "Record Index"},
        title=f"Z-Score per record (sample of {sample_size:,})"
    )
    z_fig.add_hline(y=3,  line_dash="dash", line_color="orange", annotation_text="+3σ")
    z_fig.add_hline(y=-3, line_dash="dash", line_color="orange", annotation_text="-3σ")
    z_fig.add_hline(y=0,  line_dash="dot",  line_color="#888888")
    st.plotly_chart(z_fig, use_container_width=True)

    st.divider()

    # ── Row 3: Anomaly rate by segment ────────────────────────────────────────
    seg_options = [c for c in categorical_cols if c not in ("country_code", "product_id", "customer_id")]
    if seg_options:
        st.markdown("**Anomaly Rate by Segment**")
        seg_col = st.selectbox("Break down anomalies by", seg_options,
                               index=seg_options.index("country") if "country" in seg_options else 0,
                               key="anom_seg")
        grp = (adf.groupby(seg_col)
               .agg(total=(target_col, "count"), anomalies=("_is_anomaly", "sum"))
               .reset_index())
        grp["anomaly_rate"] = (grp["anomalies"] / grp["total"] * 100).round(1)
        grp = grp[grp["total"] >= 30].sort_values("anomaly_rate", ascending=True).tail(15)

        seg_fig = px.bar(
            grp, x="anomaly_rate", y=seg_col, orientation="h",
            text="anomaly_rate",
            color="anomaly_rate",
            color_continuous_scale=["#4FC3F7", "#FF6B6B"],
            labels={"anomaly_rate": "Anomaly Rate (%)", seg_col: seg_col},
            title=f"Anomaly Rate (%) by {seg_col}"
        )
        seg_fig.update_traces(texttemplate="%{text}%", textposition="outside")
        seg_fig.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(seg_fig, use_container_width=True)

    st.divider()

    # ── Analyst narrative ─────────────────────────────────────────────────────
    st.markdown("### 📝 Analyst Interpretation")
    top_seg_country = (adf[adf["_is_anomaly"]]["country"].value_counts().idxmax()
                       if "country" in adf.columns else "the top segment")
    top_seg_pct = round(
        adf[adf["_is_anomaly"]]["country"].value_counts().iloc[0] / results["total_anomalies"] * 100, 1
    ) if "country" in adf.columns else 0
    extreme_count = int((adf["_severity"] == "🔴 Extreme").sum())

    st.info(f"""
**What the anomalies tell us:**

- The **{results['total_anomalies']:,} flagged records** ({results['anomaly_pct']}% of the dataset) all exceed 
  the IQR upper fence of **{results['iqr_upper']:.2f}**, meaning they represent unusually large individual 
  transactions relative to the typical order.

- The average anomalous transaction is **{avg_anomaly:.2f}**, which is **{lift}% higher** than the 
  {avg_normal:.2f} average for normal records. This means anomalies are materially distorting the 
  overall mean and trend lines.

- **{top_seg_country}** accounts for **{top_seg_pct}%** of all anomalies — investigate whether this 
  reflects genuine high-value activity or data quality issues in that market.

- **{extreme_count:,} Extreme anomalies** (Z-score > 6σ) are the highest-priority records for 
  review and should be validated before any modelling or forecasting.

**Recommended next step:** Filter to Extreme + High severity records first, validate them as 
legitimate bulk orders or data errors, then re-run the trend analysis on the cleaned dataset.
""")

    st.divider()

    # ── Drill-down table ──────────────────────────────────────────────────────
    st.markdown(f"### 📋 Top {int(top_n_table)} Anomalous Records")
    st.caption("Sorted by Z-Score (most extreme first). Every row shows the exact trigger reason and severity.")

    display_cols = [target_col, "_z_score", "_deviation_pct", "_trigger", "_severity"]
    optional     = ["country", "unit_price_gbp", "quantity_sold", "order_datetime"]
    display_cols += [c for c in optional if c in adf.columns]

    drilldown = (adf[adf["_is_anomaly"]][display_cols]
                 .rename(columns={
                     target_col:      "Value",
                     "_z_score":      "Z-Score",
                     "_deviation_pct":"Deviation (%)",
                     "_trigger":      "Trigger Reason",
                     "_severity":     "Severity",
                     "unit_price_gbp":"Unit Price",
                     "quantity_sold": "Qty Sold",
                     "order_datetime":"Order Date"
                 })
                 .sort_values("Z-Score", ascending=False)
                 .head(int(top_n_table))
                 .reset_index(drop=True))
    st.dataframe(drilldown, use_container_width=True, height=380)

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️  Download Full Anomaly Report (CSV)",
        data=anomaly_csv_download(adf, target_col),
        file_name="anomaly_report.csv",
        mime="text/csv"
    )


# =========================
# UI Header
# =========================
st.title("Client Intelligence Platform")
st.caption("Upload a CSV or Excel file and get business-focused diagnostics, findings, and recommendations.")
st.write("Turn unfamiliar client data into actionable analysis.")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file. Error: {e}")
        st.stop()

    if raw_df.empty:
        st.warning("The uploaded file is empty.")
        st.stop()

    raw_df.columns = [normalize_column_name(col) for col in raw_df.columns]

    df, numeric_cols, categorical_cols, datetime_cols, identifier_cols, datetime_confidence = detect_column_types(raw_df)

    st.success("File uploaded successfully.")
    st.info("The platform has automatically profiled the dataset and enabled context-aware analysis.")

    # =========================
    # Sidebar
    # =========================
    st.sidebar.header("Analysis Controls")
    business_context = st.sidebar.selectbox("Business Context", BUSINESS_CONTEXTS, index=0)

    filtered_df = df.copy()

    if datetime_cols:
        with st.sidebar.expander("Time Filters", expanded=True):
            for col in datetime_cols:
                series = filtered_df[col].dropna()
                if not series.empty:
                    min_date = series.min().date()
                    max_date = series.max().date()
                    selected_range = st.date_input(
                        f"Filter {col}", value=(min_date, max_date), key=f"time_{col}"
                    )
                    if isinstance(selected_range, tuple) and len(selected_range) == 2:
                        start_date, end_date = selected_range
                        filtered_df = filtered_df[filtered_df[col].dt.date.between(start_date, end_date)]

    with st.sidebar.expander("Categorical Filters", expanded=True):
        for col in categorical_cols[:12]:
            options = sorted(filtered_df[col].dropna().astype(str).unique().tolist())
            if 0 < len(options) <= 100:
                selected = st.multiselect(f"Filter {col}", options=options, key=f"cat_{col}")
                if selected:
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    with st.sidebar.expander("Numeric Filters", expanded=False):
        for col in numeric_cols[:10]:
            series = pd.to_numeric(filtered_df[col], errors="coerce").dropna()
            if not series.empty:
                min_val = float(series.min())
                max_val = float(series.max())
                if min_val != max_val:
                    selected_range = st.slider(
                        f"Filter {col}", min_value=min_val, max_value=max_val,
                        value=(min_val, max_val), key=f"num_{col}"
                    )
                    filtered_df = filtered_df[
                        pd.to_numeric(filtered_df[col], errors="coerce").between(selected_range[0], selected_range[1])
                    ]

    selected_outlier_col = None
    with st.sidebar.expander("Display Options", expanded=False):
        remove_outliers    = st.checkbox("Exclude outliers from charts", value=False)
        show_data_preview  = st.checkbox("Show filtered data preview", value=True)
        top_n              = st.slider("Top N for high-cardinality charts", min_value=5, max_value=25, value=10)
        if numeric_cols:
            selected_outlier_col = st.selectbox("Outlier column", ["<None>"] + numeric_cols)

    if remove_outliers and selected_outlier_col and selected_outlier_col != "<None>":
        mask        = get_outlier_mask(filtered_df[selected_outlier_col])
        filtered_df = filtered_df[~mask]

    # =========================
    # KPI Cards
    # =========================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",            f"{len(filtered_df):,}")
    c2.metric("Columns",         f"{filtered_df.shape[1]:,}")
    c3.metric("Missing Values",  f"{int(filtered_df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows",  f"{int(filtered_df.duplicated().sum()):,}")

    # =========================
    # Executive Summary
    # =========================
    st.subheader("Executive Summary")
    exec_summary = generate_executive_summary(
        filtered_df, numeric_cols, categorical_cols, datetime_cols,
        filtered_df.columns.tolist(), business_context
    )
    for item in exec_summary:
        st.success(item)

    # =========================
    # Top Priorities
    # =========================
    st.subheader("Top Priorities")
    top_priorities = generate_top_priorities(
        filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context
    )
    for item in top_priorities:
        if item["priority"] == "High":
            st.error(f"**{item['priority']} Priority — {item['title']}**\n\n{item['detail']}")
        elif item["priority"] == "Medium":
            st.warning(f"**{item['priority']} Priority — {item['title']}**\n\n{item['detail']}")
        else:
            st.info(f"**{item['priority']} Priority — {item['title']}**\n\n{item['detail']}")

    # =========================
    # Client Story
    # =========================
    st.subheader("Client Story")
    client_story = generate_client_story(
        filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context
    )
    cs1, cs2 = st.columns(2)
    with cs1:
        st.info(f"**What happened**\n\n{client_story['What happened']}")
        st.info(f"**Why it matters**\n\n{client_story['Why it matters']}")
    with cs2:
        st.info(f"**What to investigate next**\n\n{client_story['What to investigate next']}")
        st.info(f"**Suggested actions**\n\n{client_story['Suggested actions']}")

    # =========================
    # Key Metric Explorer
    # =========================
    st.subheader("Key Metric Explorer")
    if numeric_cols:
        m1, m2 = st.columns([2, 1])
        default_metric       = pick_best_metric_column(numeric_cols, business_context)
        default_metric_index = numeric_cols.index(default_metric) if default_metric in numeric_cols else 0
        selected_metric_col  = m1.selectbox("Choose numeric column", numeric_cols, index=default_metric_index)
        selected_agg         = m2.selectbox("Choose aggregation",
                                            ["sum", "mean", "median", "min", "max", "count", "std"])
        metric_value = compute_metric(filtered_df[selected_metric_col], selected_agg)
        st.metric(f"{selected_agg.title()} of {selected_metric_col}", format_number(metric_value))
        st.caption(f"Exact value: {metric_value}")
    else:
        st.warning("No numeric measure columns detected for metric analysis.")

    # =========================
    # Tabs — now with Anomaly Drill-Down
    # =========================
    tabs = st.tabs([
        "Overview",
        "Visual Analysis",
        "Trend Outlook",
        "Key Findings",
        "🔬 Anomaly Drill-Down",   # ← new tab
        "Recommendations",
        "Downloads"
    ])

    # ── Tab 0: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset Overview")
        if show_data_preview:
            st.dataframe(filtered_df.head(20), use_container_width=True)

        with st.expander("Data Quality & Schema", expanded=False):
            quality_df = pd.DataFrame({
                "column":       filtered_df.columns,
                "dtype":        [str(filtered_df[col].dtype) for col in filtered_df.columns],
                "missing_values": [int(filtered_df[col].isna().sum()) for col in filtered_df.columns],
                "missing_pct":  [round(filtered_df[col].isna().mean() * 100, 2) for col in filtered_df.columns],
                "unique_values": [int(filtered_df[col].nunique(dropna=True)) for col in filtered_df.columns],
                "datetime_parse_confidence_pct": [
                    datetime_confidence.get(col, np.nan) for col in filtered_df.columns
                ]
            })
            st.dataframe(quality_df, use_container_width=True)
            st.write("**Detected Column Groups**")
            st.write(f"- Numeric measure columns: {numeric_cols if numeric_cols else 'None'}")
            st.write(f"- Categorical columns: {categorical_cols if categorical_cols else 'None'}")
            st.write(f"- Datetime columns: {datetime_cols if datetime_cols else 'None'}")
            st.write(f"- Identifier-like columns: {identifier_cols if identifier_cols else 'None'}")

        left, right = st.columns(2)
        with left:
            if numeric_cols:
                selected_hist_col = st.selectbox("Numeric distribution", numeric_cols, key="overview_hist")
                st.plotly_chart(
                    px.histogram(filtered_df, x=selected_hist_col, nbins=30,
                                 title=f"Distribution of {selected_hist_col}"),
                    use_container_width=True
                )
        with right:
            target_cat_cols = categorical_cols if categorical_cols else identifier_cols
            if target_cat_cols:
                selected_cat_col = st.selectbox("Category frequency", target_cat_cols, key="overview_cat")
                counts = filtered_df[selected_cat_col].astype(str).value_counts().head(top_n).reset_index()
                counts.columns = [selected_cat_col, "count"]
                st.plotly_chart(
                    px.bar(counts.sort_values("count", ascending=True),
                           x="count", y=selected_cat_col, orientation="h",
                           title=f"Top {top_n} values in {selected_cat_col}"),
                    use_container_width=True
                )

    # ── Tab 1: Visual Analysis ────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Visual Analysis")
        all_x_options = list(dict.fromkeys(categorical_cols + datetime_cols + identifier_cols + numeric_cols))

        if all_x_options:
            default_x = pick_best_category_column(categorical_cols, business_context)
            if default_x is None and datetime_cols:
                default_x = datetime_cols[0]
            if default_x is None:
                default_x = all_x_options[0]
            default_x_index = all_x_options.index(default_x) if default_x in all_x_options else 0
            x_col = st.selectbox("Select X-axis / grouping column", all_x_options, index=default_x_index)
        else:
            st.warning("No usable columns found for charting.")
            st.stop()

        y_options     = ["<None>"] + numeric_cols
        default_y     = pick_best_metric_column(numeric_cols, business_context)
        default_y_index = y_options.index(default_y) if default_y in y_options else 0
        y_choice      = st.selectbox("Select Y-axis / metric column", y_options, index=default_y_index)
        y_col         = None if y_choice == "<None>" else y_choice

        agg_method = "sum"
        if y_col is not None:
            agg_method = st.selectbox("Aggregation method", ["sum", "mean", "median", "count"], index=0)

        chart_type, chart_reason = recommend_chart_type(
            filtered_df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols
        )
        st.write(f"**Recommended chart type:** {chart_type.replace('_', ' ').title()}")
        st.caption(chart_reason)

        try:
            fig = build_chart(filtered_df, x_col, y_col, chart_type, top_n=top_n, agg_method=agg_method)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build chart: {e}")

    # ── Tab 2: Trend Outlook ──────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Trend Outlook")
        if datetime_cols and numeric_cols:
            date_col  = st.selectbox("Select datetime column", datetime_cols, key="trend_date")
            default_trend_metric = pick_best_metric_column(numeric_cols, business_context)
            default_trend_index  = numeric_cols.index(default_trend_metric) if default_trend_metric in numeric_cols else 0
            value_col = st.selectbox("Select numeric column for trend/projection", numeric_cols,
                                     index=default_trend_index, key="trend_value")
            ma_window = st.slider("Moving average window", min_value=3, max_value=30, value=7)

            projection_df = moving_average_projection(filtered_df, date_col, value_col, window=ma_window)
            if projection_df is not None:
                trend_fig = px.line(projection_df, x=date_col, y=value_col, color="series_type",
                                    title=f"{value_col} Trend and Projection")
                st.plotly_chart(trend_fig, use_container_width=True)
                st.caption("This projection is based on a moving average. It is useful for directional "
                           "planning and early pattern recognition, not for production-grade forecasting.")
            else:
                st.warning("Not enough time-series data to create a projection.")
        else:
            st.info("A trend view requires at least one datetime column and one numeric measure column.")

    # ── Tab 3: Key Findings ───────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Key Findings")
        findings = generate_business_findings(
            filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context
        )
        if findings:
            for idx, item in enumerate(findings, start=1):
                st.info(f"{idx}. {item['text']}\n\n**Confidence:** {item['confidence']}")
        else:
            st.warning("Not enough data to generate findings.")

        st.markdown("### Correlation Heatmap")
        corr_pairs, valid_corr_cols = strongest_correlations(filtered_df, numeric_cols)
        if len(valid_corr_cols) >= 2:
            corr = filtered_df[valid_corr_cols].corr(numeric_only=True)
            heatmap_fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                    title="Interactive Correlation Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption("These relationships may help identify possible drivers. Correlation alone does not confirm causation.")
        else:
            st.info("Need at least two non-constant numeric columns for correlation analysis.")

        if numeric_cols:
            st.markdown("### Outlier Review")
            outlier_col   = st.selectbox("Choose numeric column for outlier analysis", numeric_cols, key="outlier_col")
            outlier_mask  = get_outlier_mask(filtered_df[outlier_col])
            outlier_count = int(outlier_mask.sum())
            non_null_count = int(pd.to_numeric(filtered_df[outlier_col], errors="coerce").notna().sum())
            outlier_pct   = (outlier_count / non_null_count * 100) if non_null_count > 0 else 0

            oc1, oc2 = st.columns(2)
            oc1.metric("Potential Outliers", f"{outlier_count:,}")
            oc2.metric("Outlier %",          f"{outlier_pct:.2f}%")
            st.plotly_chart(
                px.box(filtered_df, y=outlier_col, title=f"Box Plot of {outlier_col}"),
                use_container_width=True
            )

    # ── Tab 4: Anomaly Drill-Down (NEW) ───────────────────────────────────────
    with tabs[4]:
        st.subheader("🔬 Anomaly Drill-Down & Explainability")
        if numeric_cols:
            render_anomaly_tab(filtered_df, numeric_cols, categorical_cols)
        else:
            st.info("No numeric columns available for anomaly analysis.")

    # ── Tab 5: Recommendations ────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Recommended Focus Areas")
        recommendations = generate_recommendations(
            filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context
        )
        if recommendations:
            for idx, rec in enumerate(recommendations, start=1):
                st.success(f"{idx}. {rec}")
        else:
            st.warning("Not enough signal to generate recommendations.")

    # ── Tab 6: Downloads ──────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Downloads")

        findings        = generate_business_findings(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)
        recommendations = generate_recommendations(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)
        top_priorities  = generate_top_priorities(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)

        st.download_button(
            label="Download Filtered Data (CSV)",
            data=df_to_csv_download(filtered_df),
            file_name="filtered_data.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Executive Summary, Priorities, Client Story, Findings & Recommendations (TXT)",
            data=text_download(findings, recommendations, exec_summary, client_story, top_priorities),
            file_name="client_intelligence_report.txt",
            mime="text/plain"
        )

        # Anomaly export (pre-computed for the best metric)
        best_metric = pick_best_metric_column(numeric_cols, business_context)
        if best_metric:
            series  = pd.to_numeric(filtered_df[best_metric], errors="coerce").fillna(filtered_df[best_metric].median())
            results = detect_anomalies(series, "IQR")
            adf     = build_anomaly_df(filtered_df, best_metric, results)
            st.download_button(
                label=f"Download Anomaly Report — {best_metric} (CSV)",
                data=anomaly_csv_download(adf, best_metric),
                file_name="anomaly_report.csv",
                mime="text/csv"
            )

        st.write("These exports can be used for portfolio demos, case-study storytelling, and stakeholder reporting.")

else:
    st.info("Upload a CSV or Excel file to explore KPIs, findings, trends, recommendations, and business context-aware analysis.")