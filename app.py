import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="AI Analyst Copilot Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# Helpers
# =========================
ID_KEYWORDS = [
    "id", "order_id", "product_id", "customer_id", "invoice", "transaction_id",
    "user_id", "record_id", "item_id", "code", "sku", "empid", "employeeid"
]


def safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        converted = pd.to_datetime(series, errors="coerce")
        valid_ratio = converted.notna().mean()
        return converted if valid_ratio > 0.7 else series
    except Exception:
        return series


def normalize_column_name(col: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(col).strip().lower())


def detect_column_types(df: pd.DataFrame):
    temp_df = df.copy()
    temp_df.columns = [str(c) for c in temp_df.columns]

    datetime_cols = []
    for col in temp_df.columns:
        if temp_df[col].dtype == "object":
            converted = safe_to_datetime(temp_df[col])
            if pd.api.types.is_datetime64_any_dtype(converted):
                temp_df[col] = converted
                datetime_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(temp_df[col]):
            datetime_cols.append(col)

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
            if nunique_ratio > 0.85:
                identifier_cols.append(col)

    numeric_measure_cols = [c for c in numeric_cols if c not in identifier_cols]
    categorical_cols = [
        c for c in all_cols
        if c not in numeric_cols and c not in datetime_cols
    ]

    return temp_df, numeric_measure_cols, categorical_cols, datetime_cols, identifier_cols


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


def recommend_chart_type(df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols):
    x_is_numeric = x_col in numeric_cols
    y_is_numeric = y_col in numeric_cols if y_col else False
    x_is_categorical = x_col in categorical_cols or x_col in identifier_cols
    x_is_datetime = x_col in datetime_cols
    unique_x = df[x_col].nunique(dropna=True)

    if x_is_datetime and y_is_numeric:
        return "line", "A line chart is best for showing change over time."

    if x_is_numeric and y_is_numeric:
        if x_col in identifier_cols:
            return "bar_top_n", "This looks like an identifier field, so aggregated bar charts are more meaningful than a scatter plot."
        return "scatter", "A scatter plot is best for exploring relationships, clusters, or outliers between two numeric variables."

    if x_is_categorical and y_is_numeric:
        if unique_x <= 12:
            return "bar", "A bar chart is suitable for comparing a numeric metric across categories."
        return "bar_top_n", "This field has many categories, so a Top N horizontal bar chart improves readability."

    if x_is_numeric and not y_col:
        return "histogram", "A histogram is best for understanding the distribution of a numeric variable."

    if x_is_categorical and not y_col:
        if unique_x <= 6:
            return "pie", "A pie chart can work when there are very few categories."
        return "count_bar", "A count bar chart is better for comparing category frequencies."

    return "bar", "A bar chart is the safest default for this selection."


def build_chart(df, x_col, y_col, chart_type, top_n=10):
    chart_df = df.copy()

    if chart_type == "line":
        chart_df = chart_df[[x_col, y_col]].dropna().sort_values(by=x_col)
        return px.line(chart_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")

    if chart_type == "scatter":
        chart_df = chart_df[[x_col, y_col]].dropna()
        return px.scatter(chart_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", opacity=0.7)

    if chart_type == "bar":
        grouped = (
            chart_df[[x_col, y_col]]
            .dropna()
            .groupby(x_col, as_index=False)[y_col]
            .sum()
            .sort_values(by=y_col, ascending=False)
        )
        return px.bar(grouped, x=x_col, y=y_col, title=f"{y_col} by {x_col}")

    if chart_type == "bar_top_n":
        grouped = (
            chart_df[[x_col, y_col]]
            .dropna()
            .groupby(x_col, as_index=False)[y_col]
            .sum()
            .sort_values(by=y_col, ascending=False)
            .head(top_n)
            .sort_values(by=y_col, ascending=True)
        )
        return px.bar(
            grouped,
            x=y_col,
            y=x_col,
            orientation="h",
            title=f"Top {top_n} {x_col} by {y_col}"
        )

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
        future_dates = pd.date_range(
            start=last_date,
            periods=projection_points + 1,
            freq=inferred_freq
        )[1:]

    last_ma = grouped["moving_avg"].iloc[-1]
    forecast_df = pd.DataFrame({
        date_col: future_dates,
        value_col: [last_ma] * len(future_dates),
        "series_type": ["Projection"] * len(future_dates)
    })

    actual_df = grouped.copy()
    actual_df["series_type"] = "Actual"

    return pd.concat(
        [actual_df[[date_col, value_col, "series_type"]], forecast_df],
        ignore_index=True
    )


def strongest_correlations(df, numeric_cols):
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if pd.notna(val):
                pairs.append((col1, col2, val))

    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]


def generate_insights(df, numeric_cols, categorical_cols, datetime_cols):
    insights = []

    if numeric_cols:
        for col in numeric_cols[:3]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                insights.append(
                    f"'{col}' has a mean of {s.mean():.2f}, median of {s.median():.2f}, and standard deviation of {s.std():.2f}."
                )

                outlier_mask = get_outlier_mask(s)
                outlier_count = int(outlier_mask.sum())
                if len(s) > 0 and outlier_count > 0:
                    outlier_pct = outlier_count / len(s) * 100
                    insights.append(
                        f"'{col}' contains approximately {outlier_count} potential outliers ({outlier_pct:.1f}% of non-null values)."
                    )

    if categorical_cols:
        for col in categorical_cols[:2]:
            vc = df[col].astype(str).value_counts(dropna=False)
            if not vc.empty:
                top_cat = vc.index[0]
                top_count = vc.iloc[0]
                insights.append(
                    f"The most frequent value in '{col}' is '{top_cat}', appearing {top_count:,} times."
                )

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        metric_col = numeric_cols[0]
        temp = df[[date_col, metric_col]].dropna().copy()
        temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            trend = temp.groupby(date_col, as_index=False)[metric_col].sum().sort_values(date_col)
            if len(trend) >= 2:
                first_val = trend[metric_col].iloc[0]
                last_val = trend[metric_col].iloc[-1]
                direction = "increased" if last_val > first_val else "decreased"
                insights.append(
                    f"Over the observed time range, '{metric_col}' generally {direction} from {first_val:.2f} to {last_val:.2f}."
                )

                peak_row = trend.loc[trend[metric_col].idxmax()]
                insights.append(
                    f"The peak value of '{metric_col}' occurred on {peak_row[date_col]} with a total of {peak_row[metric_col]:.2f}."
                )

    top_corrs = strongest_correlations(df, numeric_cols)
    if top_corrs:
        col1, col2, corr_val = top_corrs[0]
        strength = "strong" if abs(corr_val) >= 0.7 else "moderate" if abs(corr_val) >= 0.4 else "weak"
        direction = "positive" if corr_val > 0 else "negative"
        insights.append(
            f"The strongest detected numeric relationship is a {strength} {direction} correlation between '{col1}' and '{col2}' ({corr_val:.2f})."
        )

    return insights


def generate_executive_summary(df, numeric_cols, categorical_cols, datetime_cols):
    summary = []

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        value_col = numeric_cols[0]

        temp = df[[date_col, value_col]].dropna().copy()
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            trend = temp.groupby(date_col)[value_col].sum().sort_index()
            if len(trend) >= 2:
                change = trend.iloc[-1] - trend.iloc[0]
                direction = "increasing" if change > 0 else "decreasing"
                summary.append(f"📈 Overall trend shows a {direction} pattern in '{value_col}'.")

    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]

        grouped = (
            df[[cat, num]]
            .dropna()
            .groupby(cat)[num]
            .sum()
            .sort_values(ascending=False)
        )
        if not grouped.empty:
            summary.append(f"🏆 Top '{cat}' is '{grouped.index[0]}' contributing the highest '{num}'.")

    if numeric_cols:
        col = numeric_cols[0]
        outliers = get_outlier_mask(df[col])
        if outliers.sum() > 0:
            summary.append(f"⚠️ Detected {int(outliers.sum())} anomalies in '{col}', which may skew analysis.")
        else:
            summary.append(f"✅ No major anomalies detected in '{col}' based on the IQR method.")

    corr_pairs = strongest_correlations(df, numeric_cols)
    if corr_pairs:
        c1, c2, val = corr_pairs[0]
        summary.append(f"🔗 Strong relationship found between '{c1}' and '{c2}' ({val:.2f}).")

    summary.append("💡 Focus on the top drivers, review anomalies, and validate patterns before making business decisions.")

    return summary


def df_to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def text_download(insights, exec_summary):
    buffer = io.StringIO()
    buffer.write("AI Analyst Copilot Dashboard - Executive Summary\n")
    buffer.write("=" * 60 + "\n\n")
    for i, item in enumerate(exec_summary, start=1):
        buffer.write(f"{i}. {item}\n")

    buffer.write("\nAI Analyst Copilot Dashboard - Generated Insights\n")
    buffer.write("=" * 60 + "\n\n")
    for i, insight in enumerate(insights, start=1):
        buffer.write(f"{i}. {insight}\n")

    return buffer.getvalue().encode("utf-8")


# =========================
# Title
# =========================
st.title("AI Analyst Copilot Dashboard")
st.caption("Upload a CSV or Excel file and get business-focused analysis.")
st.write("Turn raw data into actionable business insights instantly.")

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

    df, numeric_cols, categorical_cols, datetime_cols, identifier_cols = detect_column_types(raw_df)

    st.success("File uploaded successfully.")
    st.info("The app has automatically detected column types and enabled dynamic analysis for your dataset.")

    # =========================
    # Sidebar Filters
    # =========================
    st.sidebar.header("Filters")
    filtered_df = df.copy()

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

    with st.sidebar.expander("Analysis Options", expanded=False):
        remove_outliers = st.checkbox("Exclude outliers from charts", value=False)
        show_data_preview = st.checkbox("Show filtered data preview", value=True)
        top_n = st.slider("Top N for high-cardinality charts", min_value=5, max_value=25, value=10)

    if remove_outliers and numeric_cols:
        for col in numeric_cols:
            mask = get_outlier_mask(filtered_df[col])
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

    st.caption(
        f"Rows: {row_count:,} | Columns: {col_count:,} | Missing Values: {missing_count:,} | Duplicate Rows: {duplicate_count:,}"
    )

    # =========================
    # Executive Summary
    # =========================
    st.subheader("Executive Summary")
    exec_summary = generate_executive_summary(filtered_df, numeric_cols, categorical_cols, datetime_cols)

    for item in exec_summary:
        st.success(item)

    # =========================
    # Data Quality Summary
    # =========================
    with st.expander("Data Quality Summary", expanded=False):
        quality_df = pd.DataFrame({
            "column": filtered_df.columns,
            "dtype": [str(filtered_df[col].dtype) for col in filtered_df.columns],
            "missing_values": [int(filtered_df[col].isna().sum()) for col in filtered_df.columns],
            "missing_pct": [round(filtered_df[col].isna().mean() * 100, 2) for col in filtered_df.columns],
            "unique_values": [int(filtered_df[col].nunique(dropna=True)) for col in filtered_df.columns]
        })
        st.dataframe(quality_df, use_container_width=True)

        st.write("**Detected Column Groups**")
        st.write(f"- Numeric measure columns: {numeric_cols if numeric_cols else 'None'}")
        st.write(f"- Categorical columns: {categorical_cols if categorical_cols else 'None'}")
        st.write(f"- Datetime columns: {datetime_cols if datetime_cols else 'None'}")
        st.write(f"- Identifier-like columns: {identifier_cols if identifier_cols else 'None'}")

    # =========================
    # Custom Metric
    # =========================
    st.subheader("Custom Metric")

    if numeric_cols:
        m1, m2 = st.columns([2, 1])
        selected_metric_col = m1.selectbox("Choose numeric column", numeric_cols)
        selected_agg = m2.selectbox("Choose aggregation", ["sum", "mean", "median", "min", "max", "count", "std"])

        metric_value = compute_metric(filtered_df[selected_metric_col], selected_agg)
        metric_label = f"{selected_agg.title()} of {selected_metric_col}"
        st.metric(metric_label, format_number(metric_value))
        st.caption(f"Exact value: {metric_value}")
    else:
        st.warning("No numeric measure columns detected for custom metrics.")

    # =========================
    # Tabs
    # =========================
    tabs = st.tabs([
        "Overview",
        "Visual Explorer",
        "Forecast & Trends",
        "Insights",
        "Downloads"
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")

        if show_data_preview:
            st.dataframe(filtered_df.head(20), use_container_width=True)

        left, right = st.columns(2)

        with left:
            if numeric_cols:
                selected_hist_col = st.selectbox("Numeric distribution", numeric_cols, key="overview_hist")
                hist_fig = px.histogram(filtered_df, x=selected_hist_col, nbins=30, title=f"Distribution of {selected_hist_col}")
                st.plotly_chart(hist_fig, use_container_width=True)

        with right:
            target_cat_cols = categorical_cols if categorical_cols else identifier_cols
            if target_cat_cols:
                selected_cat_col = st.selectbox("Category frequency", target_cat_cols, key="overview_cat")
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
        st.subheader("Smart Chart Builder")

        all_x_options = categorical_cols + datetime_cols + identifier_cols + numeric_cols
        all_x_options = list(dict.fromkeys(all_x_options))

        if all_x_options:
            x_col = st.selectbox("Select X-axis / grouping column", all_x_options)
        else:
            st.warning("No usable columns found for charting.")
            st.stop()

        y_options = ["<None>"] + numeric_cols
        y_choice = st.selectbox("Select Y-axis / metric column", y_options)
        y_col = None if y_choice == "<None>" else y_choice

        chart_type, chart_reason = recommend_chart_type(
            filtered_df, x_col, y_col, numeric_cols, categorical_cols, datetime_cols, identifier_cols
        )

        st.write(f"**Recommended chart type:** {chart_type.replace('_', ' ').title()}")
        st.caption(chart_reason)

        try:
            fig = build_chart(filtered_df, x_col, y_col, chart_type, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build chart: {e}")

    with tabs[2]:
        st.subheader("Forecast & Trends")

        if datetime_cols and numeric_cols:
            date_col = st.selectbox("Select datetime column", datetime_cols, key="trend_date")
            value_col = st.selectbox("Select numeric column for trend/projection", numeric_cols, key="trend_value")
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
                    "This trend view includes a simple moving-average projection. "
                    "It is useful for directional planning, not for high-stakes forecasting."
                )
            else:
                st.warning("Not enough time-series data to create a projection.")
        else:
            st.info("A trend/projection view requires at least one datetime column and one numeric measure column.")

    with tabs[3]:
        st.subheader("Business Insights")

        generated_insights = generate_insights(filtered_df, numeric_cols, categorical_cols, datetime_cols)

        if generated_insights:
            for idx, insight in enumerate(generated_insights, start=1):
                st.info(f"{idx}. {insight}")
        else:
            st.warning("Not enough data to generate insights.")

        if numeric_cols:
            st.markdown("### Correlation Heatmap")
            if len(numeric_cols) >= 2:
                corr = filtered_df[numeric_cols].corr(numeric_only=True)
                heatmap_fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Interactive Correlation Heatmap"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.caption("This heatmap shows how strongly numeric variables are related to each other.")
            else:
                st.info("Need at least two numeric columns for correlation analysis.")

        if numeric_cols:
            st.markdown("### Outlier Check")
            outlier_col = st.selectbox("Choose numeric column for outlier analysis", numeric_cols, key="outlier_col")
            outlier_mask = get_outlier_mask(filtered_df[outlier_col])
            outlier_count = int(outlier_mask.sum())
            non_null_count = int(pd.to_numeric(filtered_df[outlier_col], errors="coerce").notna().sum())
            outlier_pct = (outlier_count / non_null_count * 100) if non_null_count > 0 else 0

            oc1, oc2 = st.columns(2)
            oc1.metric("Potential Outliers", f"{outlier_count:,}")
            oc2.metric("Outlier %", f"{outlier_pct:.2f}%")

            box_fig = px.box(filtered_df, y=outlier_col, title=f"Box Plot of {outlier_col}")
            st.plotly_chart(box_fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Downloads")

        generated_insights = generate_insights(filtered_df, numeric_cols, categorical_cols, datetime_cols)

        st.download_button(
            label="Download Filtered Data (CSV)",
            data=df_to_csv_download(filtered_df),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Executive Summary & Insights (TXT)",
            data=text_download(generated_insights, exec_summary),
            file_name="executive_summary_and_insights.txt",
            mime="text/plain"
        )

        st.write("You can use these exports for reporting, portfolio demos, or sharing analysis outputs.")

else:
    st.info("Upload a CSV or Excel file to explore KPIs, trends, chart recommendations, automated insights, and executive summaries.")