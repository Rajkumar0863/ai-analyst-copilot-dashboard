import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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

STRUCTURAL_SCALE_COLUMNS = {
    "population", "population_total", "gdp", "gdp_current_usd", "total_assets",
    "market_cap", "headcount", "employees"
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

    if agg == "sum":
        return clean.sum()
    if agg == "mean":
        return clean.mean()
    if agg == "median":
        return clean.median()
    if agg == "min":
        return clean.min()
    if agg == "max":
        return clean.max()
    if agg == "count":
        return clean.count()
    if agg == "std":
        return clean.std()
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
    return filtered[0] if filtered else numeric_cols[0]


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


def is_structural_scale_column(col_name: str) -> bool:
    c = col_name.lower()
    return any(token in c for token in STRUCTURAL_SCALE_COLUMNS)


def infer_confidence(
    finding_type: str,
    rows: int,
    missing_ratio: float = 0.0,
    outlier_pct: float | None = None,
    corr_value: float | None = None
):
    if rows < 30:
        return "Low"

    if finding_type == "descriptive":
        if missing_ratio < 0.05 and rows >= 100:
            return "High"
        return "Medium"

    if finding_type == "trend":
        if missing_ratio < 0.05 and rows >= 50:
            return "High"
        return "Medium"

    if finding_type == "segmentation":
        if missing_ratio < 0.05 and rows >= 100:
            return "High"
        return "Medium"

    if finding_type == "outlier":
        if outlier_pct is not None:
            if outlier_pct <= 5:
                return "High"
            if outlier_pct <= 20:
                return "Medium"
            return "Medium"
        return "Medium"

    if finding_type == "correlation":
        if corr_value is None:
            return "Low"
        if abs(corr_value) >= 0.8 and rows >= 100:
            return "Medium"
        if abs(corr_value) >= 0.5:
            return "Medium"
        return "Low"

    return "Medium"


def strongest_correlations(df, numeric_cols, focus_metric=None):
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
            ("month" in c1 and "week" in c2) or
            ("month" in c2 and "week" in c1) or
            ("year" in c1 and "month" in c2) or
            ("year" in c2 and "month" in c1) or
            ("day" in c1 and "hour" in c2) or
            ("day" in c2 and "hour" in c1)
        )
        return trivial_time or exact_family

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if pd.isna(val) or is_trivial_pair(col1, col2):
                continue

            score = abs(val)

            # prioritize relationships involving the main metric
            if focus_metric and (col1 == focus_metric or col2 == focus_metric):
                score += 1.0

            # de-prioritize structural scale pairs unless one side is the focus metric
            if (
                is_structural_scale_column(col1)
                and is_structural_scale_column(col2)
                and not (focus_metric and (col1 == focus_metric or col2 == focus_metric))
            ):
                score -= 0.35

            pairs.append((col1, col2, val, score))

    ranked = sorted(pairs, key=lambda x: x[3], reverse=True)[:5]
    return [(col1, col2, val) for col1, col2, val, _ in ranked], valid_numeric_cols


def recommend_chart_type(df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols):
    x_is_numeric = x_col in numeric_cols
    y_is_numeric = y_col in numeric_cols if y_col else False
    x_is_categorical = x_col in categorical_cols or x_col in identifier_cols
    x_is_datetime = x_col in datetime_cols
    unique_x = df[x_col].nunique(dropna=True)

    if x_is_datetime and y_is_numeric:
        return "line", "A line chart is best for showing how the selected metric changes over time.", "area"

    if x_is_numeric and y_is_numeric:
        if x_col in identifier_cols:
            return "bar_top_n", "This looks like an identifier field, so a Top N bar chart is more meaningful than a scatter plot.", "count_bar"
        return "scatter", "A scatter plot is useful for checking relationships, clusters, and unusual values.", "line"

    if x_is_categorical and y_is_numeric:
        if unique_x <= 12:
            return "bar", "A bar chart is appropriate for comparing a numeric metric across categories.", "bar_top_n"
        return "bar_top_n", "This field has many categories, so a Top N horizontal bar chart improves readability.", "bar"

    if x_is_numeric and not y_col:
        return "histogram", "A histogram helps explain the distribution of a numeric variable.", "box"

    if x_is_categorical and not y_col:
        if unique_x <= 6:
            return "pie", "A pie chart can work when there are very few categories.", "count_bar"
        return "count_bar", "A count bar chart is better for comparing category frequencies.", "pie"

    return "bar", "A bar chart is the safest default for this selection.", "count_bar"


def recommend_aggregation(df, x_col, y_col):
    if y_col is None:
        return "count", "No metric selected, so count/frequency is the most meaningful aggregation."

    clean = pd.to_numeric(df[y_col], errors="coerce").dropna()
    if clean.empty:
        return "sum", "The selected metric is sparse, so sum is the safest aggregation."

    outlier_mask = get_outlier_mask(clean)
    outlier_pct = outlier_mask.mean() * 100 if len(clean) > 0 else 0
    unique_groups = df[x_col].nunique(dropna=True)

    if outlier_pct > 10:
        return "median", f"'{y_col}' shows notable outliers ({outlier_pct:.1f}%), so median is more robust than mean."

    if unique_groups > 20:
        return "sum", "There are many groups, so sum is often easier to interpret for Top N comparisons."

    return "mean", "Mean is appropriate here because the metric does not show severe distortion from outliers."


def aggregate_data(grouped_df, x_col, y_col, agg_method):
    if agg_method == "sum":
        return grouped_df.groupby(x_col, as_index=False)[y_col].sum()
    if agg_method == "mean":
        return grouped_df.groupby(x_col, as_index=False)[y_col].mean()
    if agg_method == "median":
        return grouped_df.groupby(x_col, as_index=False)[y_col].median()
    if agg_method == "count":
        return grouped_df.groupby(x_col, as_index=False)[y_col].count()
    return grouped_df.groupby(x_col, as_index=False)[y_col].sum()


def build_chart(df, x_col, y_col, chart_type, top_n=10, agg_method="sum"):
    chart_df = df.copy()

    if chart_type == "line":
        chart_df = chart_df[[x_col, y_col]].dropna().sort_values(by=x_col)
        return px.line(chart_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")

    if chart_type == "area":
        chart_df = chart_df[[x_col, y_col]].dropna().sort_values(by=x_col)
        return px.area(chart_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")

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
        return px.bar(
            grouped,
            x=y_col,
            y=x_col,
            orientation="h",
            title=f"Top {top_n} {x_col} by {agg_method.title()} {y_col}"
        )

    if chart_type == "histogram":
        chart_df = chart_df[[x_col]].dropna()
        return px.histogram(chart_df, x=x_col, nbins=30, title=f"Distribution of {x_col}")

    if chart_type == "box":
        chart_df = chart_df[[x_col]].dropna()
        return px.box(chart_df, y=x_col, title=f"Box Plot of {x_col}")

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


def get_missing_ratio(df, cols):
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return 0.0
    return float(df[valid].isna().mean().mean())


# =========================
# FDE Layer
# =========================
def generate_top_priorities(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    priorities = []

    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            outlier_mask = get_outlier_mask(df[best_metric])
            outlier_count = int(outlier_mask.sum())
            outlier_pct = (outlier_count / len(metric_series) * 100) if len(metric_series) > 0 else 0

            if outlier_pct >= 10:
                detail = f"'{best_metric}' has {outlier_count:,} potential anomalies ({outlier_pct:.1f}%), so summary metrics may be distorted."
                if is_structural_scale_column(best_metric):
                    detail += " Some extremes may reflect natural scale differences rather than bad records."
                priorities.append({
                    "priority": "Medium",
                    "title": "Data anomaly review",
                    "detail": detail,
                    "why_ranked": "Ranked highly because distorted values can weaken trust in all downstream insights."
                })

    if best_date and best_metric:
        trend = (
            df[[best_date, best_metric]]
            .dropna()
            .groupby(best_date)[best_metric]
            .sum()
            .sort_index()
        )
        if len(trend) >= 2:
            first_val = trend.iloc[0]
            last_val = trend.iloc[-1]
            if first_val != 0:
                change_pct = ((last_val - first_val) / first_val) * 100
                if change_pct <= -20:
                    priorities.insert(0, {
                        "priority": "High",
                        "title": "Metric decline",
                        "detail": f"'{best_metric}' declined by {abs(change_pct):.2f}% over the observed period.",
                        "why_ranked": "Ranked first because it combines large business impact with a clear directional deterioration."
                    })
                elif change_pct >= 20:
                    priorities.insert(0, {
                        "priority": "High",
                        "title": "Metric shift",
                        "detail": f"'{best_metric}' increased by {change_pct:.2f}% over the observed period, which may indicate a major structural change.",
                        "why_ranked": "Ranked first because the magnitude suggests a meaningful business shift worth immediate validation."
                    })

    if best_category and best_metric:
        grouped = (
            df[[best_category, best_metric]]
            .dropna()
            .groupby(best_category)[best_metric]
            .sum()
            .sort_values(ascending=False)
        )
        if len(grouped) >= 2:
            total = grouped.sum()
            top_share = (grouped.iloc[0] / total * 100) if total != 0 else 0
            if top_share >= 40:
                priorities.append({
                    "priority": "Medium",
                    "title": "Concentration risk",
                    "detail": f"The top '{best_category}' group contributes {top_share:.1f}% of total '{best_metric}', suggesting dependency risk.",
                    "why_ranked": "Ranked highly because over-reliance on one segment can amplify downside risk."
                })

    if not priorities:
        priorities.append({
            "priority": "Low",
            "title": "No major risk signal",
            "detail": "No urgent risk pattern was automatically detected. Continue with deeper segmented analysis.",
            "why_ranked": "No single issue crossed the urgency threshold."
        })

    return priorities[:3]


def generate_business_findings(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    findings = []

    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            text = (
                f"The primary business metric '{best_metric}' totals {metric_series.sum():,.2f} "
                f"with an average of {metric_series.mean():,.2f}."
            )
            findings.append({
                "text": text,
                "confidence": infer_confidence(
                    "descriptive",
                    rows=len(metric_series),
                    missing_ratio=get_missing_ratio(df, [best_metric])
                )
            })

            outlier_mask = get_outlier_mask(df[best_metric])
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(metric_series) * 100) if len(metric_series) > 0 else 0
                text = (
                    f"'{best_metric}' contains {outlier_count:,} potential anomalies "
                    f"({outlier_pct:.1f}% of non-null records), so averages and trends should be interpreted carefully."
                )
                if is_structural_scale_column(best_metric):
                    text += " Some extreme values may reflect natural scale differences rather than data quality issues."

                findings.append({
                    "text": text,
                    "confidence": infer_confidence(
                        "outlier",
                        rows=len(metric_series),
                        missing_ratio=get_missing_ratio(df, [best_metric]),
                        outlier_pct=outlier_pct
                    )
                })

    if best_category and best_metric:
        grouped = (
            df[[best_category, best_metric]]
            .dropna()
            .groupby(best_category)[best_metric]
            .sum()
            .sort_values(ascending=False)
        )
        if not grouped.empty:
            top_group = grouped.index[0]
            top_value = grouped.iloc[0]
            bottom_group = grouped.index[-1]
            bottom_value = grouped.iloc[-1]
            text = (
                f"Performance is uneven across '{best_category}'. "
                f"'{top_group}' leads with {top_value:,.2f} in '{best_metric}', "
                f"while '{bottom_group}' contributes only {bottom_value:,.2f}."
            )
            findings.append({
                "text": text,
                "confidence": infer_confidence(
                    "segmentation",
                    rows=len(df[[best_category, best_metric]].dropna()),
                    missing_ratio=get_missing_ratio(df, [best_category, best_metric])
                )
            })

    if best_date and best_metric:
        trend = (
            df[[best_date, best_metric]]
            .dropna()
            .groupby(best_date)[best_metric]
            .sum()
            .sort_index()
        )
        if len(trend) >= 2:
            first_val = trend.iloc[0]
            last_val = trend.iloc[-1]
            direction = "increased" if last_val > first_val else "decreased"
            if first_val != 0:
                change_pct = ((last_val - first_val) / first_val) * 100
                text = (
                    f"Over time, '{best_metric}' has {direction} from {first_val:,.2f} "
                    f"to {last_val:,.2f} ({change_pct:.2f}%). This is a material shift that deserves attention."
                )
            else:
                text = f"Over time, '{best_metric}' has {direction} from {first_val:,.2f} to {last_val:,.2f}."

            findings.append({
                "text": text,
                "confidence": infer_confidence(
                    "trend",
                    rows=len(trend),
                    missing_ratio=get_missing_ratio(df, [best_date, best_metric])
                )
            })

    corr_pairs, _ = strongest_correlations(df, numeric_cols, focus_metric=best_metric)
    if corr_pairs:
        col1, col2, corr_val = corr_pairs[0]
        direction = "positive" if corr_val > 0 else "negative"
        strength = "strong" if abs(corr_val) >= 0.7 else "moderate" if abs(corr_val) >= 0.4 else "weak"

        if best_metric and (col1 == best_metric or col2 == best_metric):
            relevance_text = "This is directly tied to the main business metric and is a useful candidate driver to investigate."
        else:
            relevance_text = "This relationship is notable, but it is not directly tied to the main metric, so it should be treated as secondary context."

        text = (
            f"The strongest non-trivial measurable relationship is a {strength} {direction} correlation "
            f"between '{col1}' and '{col2}' ({corr_val:.2f}). {relevance_text} "
            f"It should not be treated as proof of causation."
        )
        findings.append({
            "text": text,
            "confidence": infer_confidence(
                "correlation",
                rows=len(df[[col1, col2]].dropna()),
                missing_ratio=get_missing_ratio(df, [col1, col2]),
                corr_value=corr_val
            )
        })

    return findings


def generate_recommendations(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    recommendations = []

    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    if best_metric:
        outlier_mask = get_outlier_mask(df[best_metric])
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            rec = (
                f"Review the {outlier_count:,} unusual values in '{best_metric}' first, because they may be inflating or masking the true business pattern."
            )
            if is_structural_scale_column(best_metric):
                rec += " Separate natural scale-driven extremes from genuinely suspicious records."
            recommendations.append(rec)

    if best_date and best_metric:
        trend = (
            df[[best_date, best_metric]]
            .dropna()
            .groupby(best_date)[best_metric]
            .sum()
            .sort_index()
        )
        if len(trend) >= 2:
            first_val = trend.iloc[0]
            last_val = trend.iloc[-1]
            if first_val != 0:
                change_pct = ((last_val - first_val) / first_val) * 100
                if change_pct < 0:
                    recommendations.append(
                        f"Investigate the period where '{best_metric}' declined by {abs(change_pct):.2f}% and compare that period against pricing, demand, and category mix changes."
                    )

    if best_category and best_metric:
        grouped = (
            df[[best_category, best_metric]]
            .dropna()
            .groupby(best_category)[best_metric]
            .sum()
            .sort_values(ascending=False)
        )
        if len(grouped) >= 2:
            top_group = grouped.index[0]
            bottom_group = grouped.index[-1]
            recommendations.append(
                f"Compare high-performing '{best_category}' group '{top_group}' against low-performing group '{bottom_group}' to identify operational, market, or customer-mix differences."
            )

    if business_context == "Sales Performance":
        recommendations.append("Review whether revenue concentration is too dependent on a small number of regions, categories, or products.")
    elif business_context == "Customer Retention":
        recommendations.append("Segment customers by value and repeat behavior to determine where retention interventions will have the highest payoff.")
    elif business_context == "HR Attrition":
        recommendations.append("Test whether attrition risk is concentrated in specific departments, tenure bands, or overtime-heavy groups.")
    elif business_context == "Operational Efficiency":
        recommendations.append("Use high-variance process or time metrics to isolate bottlenecks and inconsistent execution.")
    elif business_context == "Financial Performance":
        recommendations.append("Separate revenue growth from cost pressure so that apparent performance gains do not hide profitability leakage.")
    else:
        recommendations.append("Use the strongest trend, anomaly, and segment imbalance signals as the starting point for deeper root-cause analysis.")

    seen = set()
    final_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            final_recommendations.append(rec)
            seen.add(rec)

    return final_recommendations[:6]


def generate_client_story(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    story = {
        "What happened": "Not enough signal detected to summarize what happened.",
        "Why it matters": "This dataset needs more business context or stronger field coverage.",
        "What to investigate next": "Review data quality, field definitions, and analysis scope.",
        "Suggested actions": "Validate the business objective and refine the analysis with a target question."
    }

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if not metric_series.empty:
            story["What happened"] = (
                f"'{best_metric}' emerges as the most decision-relevant measure, with meaningful variation across the dataset."
            )

    if best_category and best_metric:
        grouped = (
            df[[best_category, best_metric]]
            .dropna()
            .groupby(best_category)[best_metric]
            .sum()
            .sort_values(ascending=False)
        )
        if not grouped.empty:
            top_group = grouped.index[0]
            story["Why it matters"] = (
                f"Performance is uneven across '{best_category}', with '{top_group}' contributing the most, which suggests concentration risk."
            )

    if best_date and best_metric:
        trend = (
            df[[best_date, best_metric]]
            .dropna()
            .groupby(best_date)[best_metric]
            .sum()
            .sort_index()
        )
        if len(trend) >= 2:
            direction = "weakened" if trend.iloc[-1] < trend.iloc[0] else "improved"
            story["What to investigate next"] = (
                f"Investigate when '{best_metric}' {direction} and compare that period with segment shifts, anomalies, or operational changes."
            )

    story["Suggested actions"] = (
        "Prioritize the main metric, compare strong and weak segments, validate anomalies, and use the resulting differences to guide the next business decision."
    )

    return story


def generate_executive_summary(df, numeric_cols, categorical_cols, datetime_cols, all_cols, business_context):
    summary = []

    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    if best_date and best_metric:
        temp = df[[best_date, best_metric]].dropna().copy()
        temp[best_metric] = pd.to_numeric(temp[best_metric], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            trend = temp.groupby(best_date)[best_metric].sum().sort_index()
            if len(trend) >= 2:
                change = trend.iloc[-1] - trend.iloc[0]
                direction = "increasing" if change > 0 else "decreasing"
                summary.append(f"📈 Overall trend shows a {direction} pattern in '{best_metric}'.")

    if best_category:
        vc = df[best_category].astype(str).value_counts(dropna=False)
        if not vc.empty:
            if best_metric:
                grouped = (
                    df[[best_category, best_metric]]
                    .dropna()
                    .groupby(best_category)[best_metric]
                    .sum()
                    .sort_values(ascending=False)
                )
                if not grouped.empty:
                    summary.append(f"🏆 Leading '{best_category}' segment is '{grouped.index[0]}' based on total '{best_metric}'.")
            else:
                summary.append(f"🏆 Most frequent value in '{best_category}' is '{vc.index[0]}' with {vc.iloc[0]:,} records.")

    if best_metric:
        outliers = get_outlier_mask(df[best_metric])
        outlier_count = int(outliers.sum())
        if outlier_count > 0:
            summary.append(f"⚠️ Detected {outlier_count} anomalies in '{best_metric}', which may influence averages and trends.")
        else:
            summary.append(f"✅ No major anomalies detected in '{best_metric}' based on the IQR method.")

    corr_pairs, _ = strongest_correlations(df, numeric_cols, focus_metric=best_metric)
    if corr_pairs:
        c1, c2, val = corr_pairs[0]
        summary.append(f"🔗 Strongest non-trivial measurable relationship is between '{c1}' and '{c2}' ({val:.2f}).")

    summary.append(f"💡 Business context selected: {business_context}. Recommendations below are tailored accordingly.")
    return summary


def generate_bottom_line(df, numeric_cols, categorical_cols, datetime_cols, business_context):
    best_metric = pick_best_metric_column(numeric_cols, business_context)
    best_category = pick_best_category_column(categorical_cols, business_context)
    best_date = pick_best_datetime_column(datetime_cols)

    messages = []

    if best_date and best_metric:
        trend = (
            df[[best_date, best_metric]]
            .dropna()
            .groupby(best_date)[best_metric]
            .sum()
            .sort_index()
        )
        if len(trend) >= 2 and trend.iloc[0] != 0:
            change_pct = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100
            if change_pct < 0:
                messages.append(f"{best_metric} is declining materially ({abs(change_pct):.1f}%).")
            elif change_pct > 0:
                messages.append(f"{best_metric} is rising materially ({change_pct:.1f}%).")

    if best_category and best_metric:
        grouped = (
            df[[best_category, best_metric]]
            .dropna()
            .groupby(best_category)[best_metric]
            .sum()
            .sort_values(ascending=False)
        )
        if len(grouped) > 0:
            top_share = (grouped.iloc[0] / grouped.sum() * 100) if grouped.sum() != 0 else 0
            if top_share >= 40:
                messages.append(f"Performance is concentrated in '{grouped.index[0]}' ({top_share:.1f}% share).")

    if best_metric:
        metric_series = pd.to_numeric(df[best_metric], errors="coerce").dropna()
        if len(metric_series) > 0:
            outlier_pct = get_outlier_mask(df[best_metric]).sum() / len(metric_series) * 100
            if outlier_pct >= 10:
                messages.append(f"Anomaly volume is high enough ({outlier_pct:.1f}%) to justify data validation before major decisions.")

    if not messages:
        return "No single urgent pattern dominates the dataset; use segmented analysis to identify the highest-value next question."

    return " ".join(messages)


def df_to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def text_download(findings, recommendations, exec_summary, client_story, top_priorities, bottom_line):
    buffer = io.StringIO()
    buffer.write("Client Intelligence Platform - Executive Summary\n")
    buffer.write("=" * 70 + "\n\n")
    for i, item in enumerate(exec_summary, start=1):
        buffer.write(f"{i}. {item}\n")

    buffer.write("\nBottom Line\n")
    buffer.write("=" * 70 + "\n\n")
    buffer.write(f"{bottom_line}\n")

    buffer.write("\nTop Priorities\n")
    buffer.write("=" * 70 + "\n\n")
    for i, item in enumerate(top_priorities, start=1):
        buffer.write(f"{i}. [{item['priority']}] {item['title']}: {item['detail']} Why ranked here: {item['why_ranked']}\n")

    buffer.write("\nClient Story\n")
    buffer.write("=" * 70 + "\n\n")
    for key, value in client_story.items():
        buffer.write(f"{key}: {value}\n\n")

    buffer.write("Key Findings\n")
    buffer.write("=" * 70 + "\n\n")
    for i, item in enumerate(findings, start=1):
        buffer.write(f"{i}. {item['text']} (Confidence: {item['confidence']})\n")

    buffer.write("\nRecommended Focus Areas\n")
    buffer.write("=" * 70 + "\n\n")
    for i, item in enumerate(recommendations, start=1):
        buffer.write(f"{i}. {item}\n")

    return buffer.getvalue().encode("utf-8")


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
                        f"Filter {col}",
                        value=(min_date, max_date),
                        key=f"time_{col}"
                    )
                    if isinstance(selected_range, tuple) and len(selected_range) == 2:
                        start_date, end_date = selected_range
                        filtered_df = filtered_df[
                            filtered_df[col].dt.date.between(start_date, end_date)
                        ]

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
                        f"Filter {col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"num_{col}"
                    )
                    filtered_df = filtered_df[
                        pd.to_numeric(filtered_df[col], errors="coerce").between(selected_range[0], selected_range[1])
                    ]

    with st.sidebar.expander("Display Options", expanded=False):
        remove_outliers = st.checkbox("Exclude outliers from charts", value=False)
        show_data_preview = st.checkbox("Show filtered data preview", value=True)
        top_n = st.slider("Top N for high-cardinality charts", min_value=5, max_value=25, value=10)

    best_metric_for_defaults = pick_best_metric_column(numeric_cols, business_context)

    if remove_outliers and best_metric_for_defaults:
        mask = get_outlier_mask(filtered_df[best_metric_for_defaults])
        filtered_df = filtered_df[~mask]

    # =========================
    # KPI Cards
    # =========================
    row_count = len(filtered_df)
    col_count = filtered_df.shape[1]
    missing_count = int(filtered_df.isna().sum().sum())
    duplicate_count = int(filtered_df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{row_count:,}")
    c2.metric("Columns", f"{col_count:,}")
    c3.metric("Missing Values", f"{missing_count:,}")
    c4.metric("Duplicate Rows", f"{duplicate_count:,}")

    # =========================
    # Executive Summary
    # =========================
    st.subheader("Executive Summary")
    exec_summary = generate_executive_summary(
        filtered_df,
        numeric_cols,
        categorical_cols,
        datetime_cols,
        filtered_df.columns.tolist(),
        business_context
    )
    for item in exec_summary:
        st.success(item)

    # =========================
    # Bottom Line
    # =========================
    bottom_line = generate_bottom_line(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)
    st.subheader("Bottom Line")
    st.info(bottom_line)

    # =========================
    # Top Priorities
    # =========================
    st.subheader("Top Priorities")
    top_priorities = generate_top_priorities(
        filtered_df,
        numeric_cols,
        categorical_cols,
        datetime_cols,
        business_context
    )

    for item in top_priorities:
        body = f"**{item['priority']} Priority — {item['title']}**\n\n{item['detail']}\n\n**Why ranked here:** {item['why_ranked']}"
        if item["priority"] == "High":
            st.error(body)
        elif item["priority"] == "Medium":
            st.warning(body)
        else:
            st.info(body)

    # =========================
    # Client Story
    # =========================
    st.subheader("Client Story")
    client_story = generate_client_story(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)

    cs_col1, cs_col2 = st.columns(2)
    with cs_col1:
        st.markdown("#### What happened")
        st.info(client_story["What happened"])
        st.markdown("#### Why it matters")
        st.info(client_story["Why it matters"])
    with cs_col2:
        st.markdown("#### What to investigate next")
        st.info(client_story["What to investigate next"])
        st.markdown("#### Suggested actions")
        st.info(client_story["Suggested actions"])

    # =========================
    # Key Metric Explorer
    # =========================
    st.subheader("Key Metric Explorer")

    if numeric_cols:
        m1, m2 = st.columns([2, 1])
        default_metric_index = numeric_cols.index(best_metric_for_defaults) if best_metric_for_defaults in numeric_cols else 0
        selected_metric_col = m1.selectbox("Choose numeric column", numeric_cols, index=default_metric_index)
        selected_agg = m2.selectbox("Choose aggregation", ["sum", "mean", "median", "min", "max", "count", "std"])

        metric_value = compute_metric(filtered_df[selected_metric_col], selected_agg)
        st.metric(f"{selected_agg.title()} of {selected_metric_col}", format_number(metric_value))
        st.caption(f"Exact value: {metric_value}")
    else:
        st.warning("No numeric measure columns detected for metric analysis.")

    # =========================
    # Tabs
    # =========================
    tabs = st.tabs([
        "Overview",
        "Visual Analysis",
        "Trend Outlook",
        "Key Findings",
        "Recommendations",
        "Downloads"
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")

        if show_data_preview:
            st.dataframe(filtered_df.head(20), use_container_width=True)

        with st.expander("Data Quality & Schema", expanded=False):
            quality_df = pd.DataFrame({
                "column": filtered_df.columns,
                "dtype": [str(filtered_df[col].dtype) for col in filtered_df.columns],
                "missing_values": [int(filtered_df[col].isna().sum()) for col in filtered_df.columns],
                "missing_pct": [round(filtered_df[col].isna().mean() * 100, 2) for col in filtered_df.columns],
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
                default_hist_col = best_metric_for_defaults if best_metric_for_defaults in numeric_cols else numeric_cols[0]
                selected_hist_col = st.selectbox(
                    "Numeric distribution",
                    numeric_cols,
                    index=numeric_cols.index(default_hist_col) if default_hist_col in numeric_cols else 0,
                    key="overview_hist"
                )
                hist_fig = px.histogram(
                    filtered_df,
                    x=selected_hist_col,
                    nbins=30,
                    title=f"Distribution of {selected_hist_col}"
                )
                st.plotly_chart(hist_fig, use_container_width=True)

        with right:
            target_cat_cols = categorical_cols if categorical_cols else identifier_cols
            if target_cat_cols:
                default_cat = pick_best_category_column(categorical_cols, business_context)
                if default_cat not in target_cat_cols:
                    default_cat = target_cat_cols[0]

                selected_cat_col = st.selectbox(
                    "Category frequency",
                    target_cat_cols,
                    index=target_cat_cols.index(default_cat) if default_cat in target_cat_cols else 0,
                    key="overview_cat"
                )
                counts = filtered_df[selected_cat_col].astype(str).value_counts().head(top_n).reset_index()
                counts.columns = [selected_cat_col, "count"]
                count_fig = px.bar(
                    counts.sort_values("count", ascending=True),
                    x="count",
                    y=selected_cat_col,
                    orientation="h",
                    title=f"Top {top_n} values in {selected_cat_col}"
                )
                st.plotly_chart(count_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Visual Analysis")

        all_x_options = categorical_cols + datetime_cols + identifier_cols + numeric_cols
        all_x_options = list(dict.fromkeys(all_x_options))

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

        y_options = ["<None>"] + numeric_cols
        default_y = best_metric_for_defaults
        default_y_index = y_options.index(default_y) if default_y in y_options else 0
        y_choice = st.selectbox("Select Y-axis / metric column", y_options, index=default_y_index)
        y_col = None if y_choice == "<None>" else y_choice

        recommended_agg, agg_reason = recommend_aggregation(filtered_df, x_col, y_col)
        available_aggs = ["sum", "mean", "median", "count"]
        agg_default_index = available_aggs.index(recommended_agg) if recommended_agg in available_aggs else 0

        agg_method = "sum"
        if y_col is not None:
            agg_method = st.selectbox("Aggregation method", available_aggs, index=agg_default_index)

        chart_type, chart_reason, alternative_chart = recommend_chart_type(
            filtered_df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols
        )

        st.write(f"**Recommended chart type:** {chart_type.replace('_', ' ').title()}")
        st.caption(chart_reason)
        st.write(f"**Suggested alternative:** {alternative_chart.replace('_', ' ').title()}")
        if y_col is not None:
            st.write(f"**Recommended aggregation:** {recommended_agg.title()}")
            st.caption(agg_reason)

        try:
            fig = build_chart(
                filtered_df,
                x_col,
                y_col,
                chart_type,
                top_n=top_n,
                agg_method=agg_method
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build chart: {e}")

    with tabs[2]:
        st.subheader("Trend Outlook")

        if datetime_cols and numeric_cols:
            date_col = st.selectbox("Select datetime column", datetime_cols, key="trend_date")
            default_trend_index = numeric_cols.index(best_metric_for_defaults) if best_metric_for_defaults in numeric_cols else 0

            value_col = st.selectbox(
                "Select numeric column for trend/projection",
                numeric_cols,
                index=default_trend_index,
                key="trend_value"
            )

            ma_window = st.slider("Moving average window", min_value=3, max_value=30, value=7)

            projection_df = moving_average_projection(filtered_df, date_col, value_col, window=ma_window)

            if projection_df is not None:
                trend_fig = px.line(
                    projection_df,
                    x=date_col,
                    y=value_col,
                    color="series_type",
                    title=f"{value_col} Trend and Projection"
                )
                st.plotly_chart(trend_fig, use_container_width=True)
                st.caption(
                    "This projection is based on a moving average. It is useful for directional planning and early pattern recognition, not for production-grade forecasting."
                )
            else:
                st.warning("Not enough time-series data to create a projection.")
        else:
            st.info("A trend view requires at least one datetime column and one numeric measure column.")

    with tabs[3]:
        st.subheader("Key Findings")
        findings = generate_business_findings(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)

        if findings:
            for idx, item in enumerate(findings, start=1):
                st.info(f"{idx}. {item['text']}\n\n**Evidence strength:** {item['confidence']}")
        else:
            st.warning("Not enough data to generate findings.")

        st.markdown("### Correlation Heatmap")
        corr_pairs, valid_corr_cols = strongest_correlations(filtered_df, numeric_cols, focus_metric=best_metric_for_defaults)
        if len(valid_corr_cols) >= 2:
            corr = filtered_df[valid_corr_cols].corr(numeric_only=True)
            heatmap_fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Interactive Correlation Heatmap"
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption("These relationships may help identify possible drivers. Correlation alone does not confirm causation.")
        else:
            st.info("Need at least two non-constant numeric columns for correlation analysis.")

        if numeric_cols:
            st.markdown("### Outlier Review")
            default_outlier_col = best_metric_for_defaults if best_metric_for_defaults in numeric_cols else numeric_cols[0]
            outlier_col = st.selectbox(
                "Choose numeric column for outlier analysis",
                numeric_cols,
                index=numeric_cols.index(default_outlier_col) if default_outlier_col in numeric_cols else 0,
                key="outlier_col"
            )
            outlier_mask = get_outlier_mask(filtered_df[outlier_col])
            outlier_count = int(outlier_mask.sum())
            non_null_count = int(pd.to_numeric(filtered_df[outlier_col], errors="coerce").notna().sum())
            outlier_pct = (outlier_count / non_null_count * 100) if non_null_count > 0 else 0

            oc1, oc2 = st.columns(2)
            oc1.metric("Potential Outliers", f"{outlier_count:,}")
            oc2.metric("Outlier %", f"{outlier_pct:.2f}%")

            if is_structural_scale_column(outlier_col):
                st.caption(
                    "This field may naturally contain large scale differences across records. High outlier rates here can reflect genuine structural variation, not necessarily bad data."
                )
            else:
                st.caption(
                    "High outlier rates may indicate unusual transactions, data quality issues, or a highly skewed business process."
                )

            box_fig = px.box(filtered_df, y=outlier_col, title=f"Box Plot of {outlier_col}")
            st.plotly_chart(box_fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Recommended Focus Areas")
        recommendations = generate_recommendations(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)

        if recommendations:
            for idx, rec in enumerate(recommendations, start=1):
                st.success(f"{idx}. {rec}")
        else:
            st.warning("Not enough signal to generate recommendations.")

    with tabs[5]:
        st.subheader("Downloads")

        findings = generate_business_findings(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)
        recommendations = generate_recommendations(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)
        top_priorities = generate_top_priorities(filtered_df, numeric_cols, categorical_cols, datetime_cols, business_context)

        st.download_button(
            label="Download Filtered Data (CSV)",
            data=df_to_csv_download(filtered_df),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Executive Summary, Priorities, Client Story, Findings & Recommendations (TXT)",
            data=text_download(findings, recommendations, exec_summary, client_story, top_priorities, bottom_line),
            file_name="client_intelligence_report.txt",
            mime="text/plain"
        )

        st.write("These exports can be used for portfolio demos, case-study storytelling, and stakeholder reporting.")

else:
    st.info("Upload a CSV or Excel file to explore KPIs, findings, trends, recommendations, and business context-aware analysis.")