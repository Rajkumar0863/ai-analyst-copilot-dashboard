import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI Analyst Copilot Dashboard", layout="wide")


# ---------------------------
# Formatting helpers
# ---------------------------
def format_compact_number(value):
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def format_full_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    if decimals == 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


# ---------------------------
# File loading and cleaning
# ---------------------------
def load_file(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def clean_data(df_local: pd.DataFrame) -> pd.DataFrame:
    df_local = df_local.loc[:, ~df_local.columns.str.contains("^Unnamed", na=False)].copy()

    # Try converting object columns to datetime if many values parse cleanly
    for col in df_local.columns:
        if df_local[col].dtype == "object":
            converted = pd.to_datetime(df_local[col], errors="coerce")
            success_ratio = converted.notna().mean()
            if success_ratio > 0.7:
                df_local[col] = converted

    # Try converting object columns to numeric if appropriate
    for col in df_local.columns:
        if df_local[col].dtype == "object":
            converted = pd.to_numeric(df_local[col], errors="coerce")
            success_ratio = converted.notna().mean()
            if success_ratio > 0.7:
                df_local[col] = converted

    return df_local


# ---------------------------
# Type detection
# ---------------------------
def detect_column_types(df_local: pd.DataFrame):
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df_local.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categorical_cols = [col for col in df_local.columns if col not in numeric_cols + datetime_cols]

    return numeric_cols, categorical_cols, datetime_cols


# ---------------------------
# Filters
# ---------------------------
def apply_filters(df_local: pd.DataFrame, categorical_cols, numeric_cols, datetime_cols):
    filtered_df = df_local.copy()

    st.sidebar.header("Filters")

    # Categorical filters
    for col in categorical_cols[:5]:
        unique_vals = filtered_df[col].dropna().astype(str).unique().tolist()
        if 1 < len(unique_vals) <= 100:
            selected_vals = st.sidebar.multiselect(f"Filter {col}", sorted(unique_vals))
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_vals)]

    # Numeric filters
    for col in numeric_cols[:3]:
        non_null = filtered_df[col].dropna()
        if not non_null.empty:
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            if min_val != max_val and np.isfinite(min_val) and np.isfinite(max_val):
                selected_range = st.sidebar.slider(
                    f"Filter {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) &
                    (filtered_df[col] <= selected_range[1])
                ]

    # Datetime filters
    for col in datetime_cols[:2]:
        non_null = filtered_df[col].dropna()
        if not non_null.empty:
            min_date = non_null.min().date()
            max_date = non_null.max().date()
            if min_date != max_date:
                selected_dates = st.sidebar.date_input(
                    f"Filter {col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    filtered_df = filtered_df[
                        (filtered_df[col].dt.date >= start_date) &
                        (filtered_df[col].dt.date <= end_date)
                    ]

    return filtered_df


# ---------------------------
# Aggregation
# ---------------------------
def compute_aggregation(df_local: pd.DataFrame, metric_col: str, agg_type: str):
    if metric_col is None or metric_col not in df_local.columns:
        return None

    series = df_local[metric_col].dropna()
    if series.empty:
        return None

    if agg_type == "sum":
        return series.sum()
    if agg_type == "mean":
        return series.mean()
    if agg_type == "median":
        return series.median()
    if agg_type == "count":
        return series.count()
    if agg_type == "min":
        return series.min()
    if agg_type == "max":
        return series.max()

    return None


# ---------------------------
# KPI generation
# ---------------------------
def create_kpis(df_local: pd.DataFrame, numeric_cols):
    st.subheader("KPI Overview")

    c1, c2, c3 = st.columns(3)

    total_rows = len(df_local)
    total_cols = df_local.shape[1]
    missing_values = int(df_local.isnull().sum().sum())

    c1.metric("Rows", format_compact_number(total_rows))
    c2.metric("Columns", format_compact_number(total_cols))
    c3.metric("Missing Values", format_compact_number(missing_values))

    st.caption(
        f"Rows: {format_full_number(total_rows, 0)} | "
        f"Columns: {format_full_number(total_cols, 0)} | "
        f"Missing Values: {format_full_number(missing_values, 0)}"
    )

    if numeric_cols:
        st.markdown("### Custom Metric")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        selected_metric = metric_col1.selectbox("Choose numeric column", numeric_cols)
        selected_agg = metric_col2.selectbox("Choose aggregation", ["sum", "mean", "median", "count", "min", "max"])

        metric_value = compute_aggregation(df_local, selected_metric, selected_agg)
        metric_col3.metric(f"{selected_agg.title()} of {selected_metric}", format_compact_number(metric_value))

        st.caption(f"Exact value: {format_full_number(metric_value, 2) if metric_value is not None else 'N/A'}")


# ---------------------------
# Smart chart recommendation
# ---------------------------
def recommend_chart(x_col, y_col, numeric_cols, categorical_cols, datetime_cols):
    x_type = (
        "datetime" if x_col in datetime_cols else
        "numeric" if x_col in numeric_cols else
        "categorical"
    )

    y_type = None
    if y_col is not None:
        y_type = (
            "datetime" if y_col in datetime_cols else
            "numeric" if y_col in numeric_cols else
            "categorical"
        )

    if y_col is None:
        if x_type == "numeric":
            return "histogram"
        return "count_bar"

    if x_type == "datetime" and y_type == "numeric":
        return "line"
    if x_type == "categorical" and y_type == "numeric":
        return "bar"
    if x_type == "numeric" and y_type == "numeric":
        return "scatter"
    if x_type == "categorical" and y_type == "categorical":
        return "heatmap_table"

    return "bar"


# ---------------------------
# Chart builders
# ---------------------------
def build_chart(df_local, x_col, y_col, chart_type):
    if chart_type == "histogram":
        fig = px.histogram(df_local, x=x_col, nbins=30, title=f"Distribution of {x_col}")
        caption = f"This histogram shows the distribution of values in '{x_col}'."
        return fig, caption

    if chart_type == "count_bar":
        counts = (
            df_local[x_col]
            .astype(str)
            .value_counts()
            .head(20)
            .reset_index()
        )
        counts.columns = [x_col, "Count"]
        fig = px.bar(counts, x=x_col, y="Count", title=f"Count of {x_col}")
        caption = f"This bar chart shows the frequency of the top values in '{x_col}'."
        return fig, caption

    if chart_type == "line":
        temp = df_local[[x_col, y_col]].dropna().sort_values(x_col)
        if temp.empty:
            return None, "No valid data available for line chart."
        grouped = temp.groupby(x_col, as_index=False)[y_col].sum()
        fig = px.line(grouped, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        caption = f"This line chart shows how '{y_col}' changes over time based on '{x_col}'."
        return fig, caption

    if chart_type == "bar":
        temp = df_local[[x_col, y_col]].dropna()
        if temp.empty:
            return None, "No valid data available for bar chart."
        grouped = temp.groupby(x_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False).head(20)
        grouped[x_col] = grouped[x_col].astype(str)
        fig = px.bar(
            grouped.sort_values(y_col, ascending=True),
            x=y_col,
            y=x_col,
            orientation="h",
            title=f"{y_col} by {x_col}"
        )
        caption = f"This horizontal bar chart compares aggregated '{y_col}' values across '{x_col}'."
        return fig, caption

    if chart_type == "scatter":
        temp = df_local[[x_col, y_col]].dropna()
        if temp.empty:
            return None, "No valid data available for scatter plot."
        fig = px.scatter(temp, x=x_col, y=y_col, opacity=0.6, title=f"{x_col} vs {y_col}")
        caption = f"This scatter plot compares '{x_col}' and '{y_col}' to reveal relationships, clusters, or outliers."
        return fig, caption

    if chart_type == "heatmap_table":
        temp = df_local[[x_col, y_col]].dropna().copy()
        if temp.empty:
            return None, "No valid data available for categorical heatmap."
        pivot = pd.crosstab(temp[x_col].astype(str), temp[y_col].astype(str))
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Blues"
            )
        )
        fig.update_layout(title=f"{x_col} vs {y_col}")
        caption = f"This heatmap shows how categories in '{x_col}' and '{y_col}' intersect."
        return fig, caption

    return None, "Could not determine a suitable chart type."


def create_correlation_heatmap(df_local, numeric_cols):
    if len(numeric_cols) < 2:
        return None, "At least two numeric columns are required for correlation analysis."

    corr = df_local[numeric_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Interactive Correlation Heatmap",
        height=700,
    )

    caption = "This heatmap shows how strongly numeric variables are related to each other."
    return fig, caption


# ---------------------------
# Insights
# ---------------------------
def generate_business_insights(df_local, numeric_cols, categorical_cols, datetime_cols):
    insights = []

    insights.append(f"The dataset contains {len(df_local):,} rows and {df_local.shape[1]} columns.")
    insights.append(f"There are {int(df_local.isnull().sum().sum()):,} missing values in the filtered dataset.")

    if numeric_cols:
        highest_missing_numeric = df_local[numeric_cols].isnull().sum().sort_values(ascending=False)
        if not highest_missing_numeric.empty:
            col = highest_missing_numeric.index[0]
            val = highest_missing_numeric.iloc[0]
            insights.append(f"The numeric column with the most missing values is '{col}' with {val:,} missing entries.")

        variance_series = df_local[numeric_cols].std(numeric_only=True).sort_values(ascending=False)
        if not variance_series.empty:
            insights.append(
                f"The most variable numeric column is '{variance_series.index[0]}', suggesting it may contain the widest spread of values."
            )

    if categorical_cols:
        top_cat = categorical_cols[0]
        top_vals = df_local[top_cat].astype(str).value_counts().head(3)
        if not top_vals.empty:
            text = ", ".join([f"{idx} ({val})" for idx, val in top_vals.items()])
            insights.append(f"The most common values in '{top_cat}' are {text}.")

    if len(numeric_cols) >= 2:
        corr = df_local[numeric_cols].corr().abs().unstack().reset_index()
        corr.columns = ["A", "B", "Corr"]
        corr = corr[corr["A"] != corr["B"]]
        corr["pair"] = corr.apply(lambda r: tuple(sorted([r["A"], r["B"]])), axis=1)
        corr = corr.drop_duplicates("pair")
        corr = corr[corr["Corr"] < 0.99]
        if not corr.empty:
            top_corr = corr.sort_values("Corr", ascending=False).iloc[0]
            insights.append(
                f"The strongest non-trivial numeric relationship is between '{top_corr['A']}' and '{top_corr['B']}' with correlation {top_corr['Corr']:.2f}."
            )

    if datetime_cols:
        insights.append(
            "Datetime columns were detected, so time-based trend analysis is available in the visualization section."
        )

    return insights


def generate_executive_summary(df_local, insights, numeric_cols, datetime_cols):
    sentences = []
    sentences.append(
        "This dashboard automatically profiles the uploaded dataset and converts it into summary statistics, interactive visuals, and business-oriented observations."
    )

    if numeric_cols:
        sentences.append(
            f"The dataset includes {len(numeric_cols)} numeric columns, which allows quantitative analysis such as aggregation, comparison, and relationship detection."
        )

    if datetime_cols:
        sentences.append(
            "Because datetime fields are present, trend analysis and simple forecasting can be applied to understand how values evolve over time."
        )

    sentences.append(
        "The main recommendation is to focus on the highest-variance fields, the strongest relationships, and any time-based patterns or anomalies that could affect interpretation."
    )

    if insights:
        sentences.append(
            "The automatically generated insights should be used as a starting point for deeper domain-specific investigation rather than a final business conclusion."
        )

    return " ".join(sentences)


# ---------------------------
# Main app UI
# ---------------------------
st.title("AI Analyst Copilot Dashboard")
st.subheader("Upload a CSV or Excel file and get business-focused analysis")
st.markdown("### Turn raw data into actionable business insights instantly.")
st.caption("Upload a CSV or Excel file to explore KPIs, trends, product performance, and business insights.")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        st.caption("Tip: Upload structured business, sales, operations, or analytics data for best results.")

        with st.spinner("Analyzing data..."):
            df = load_file(uploaded_file)
            df = clean_data(df)
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
            df, exclude_outliers, forecast_periods = apply_filters(df, categorical_cols, numeric_cols, datetime_cols)

        if df.empty:
            st.warning("No data available after applying filters.")
            st.stop()

        numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

        st.success("File uploaded successfully")

        if datetime_cols and numeric_cols:
            st.info("Detected a dataset suitable for dynamic trend and forecasting analysis.")
        else:
            st.info("Detected a dataset suitable for profiling, aggregation, and chart-based exploration.")

        create_kpis(df, numeric_cols)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Visual Explorer", "Relationship Analysis", "Insights", "Downloads"]
        )

        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [str(dtype) for dtype in df.dtypes],
                "Missing Values": df.isnull().sum().values,
                "Unique Values": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True)

            st.subheader("Summary Statistics")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
            else:
                st.warning("No numeric columns found.")

        with tab2:
            st.subheader("Smart Chart Builder")

            all_cols = df.columns.tolist()
            x_col = st.selectbox("Select X-axis / category column", all_cols)

            y_options = ["None"] + all_cols
            y_col_raw = st.selectbox("Select Y-axis / metric column", y_options)
            y_col = None if y_col_raw == "None" else y_col_raw

            recommended_chart = recommend_chart(x_col, y_col, numeric_cols, categorical_cols, datetime_cols)
            st.caption(f"Recommended chart type: **{recommended_chart.replace('_', ' ').title()}**")

            fig, caption = build_chart(df, x_col, y_col, recommended_chart)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)

            if datetime_cols and numeric_cols:
                st.subheader("Time Trend and Forecast")
                dt_col = st.selectbox("Select datetime column", datetime_cols, key="forecast_dt")
                val_col = st.selectbox("Select numeric column for trend", numeric_cols, key="forecast_val")

                temp = df[[dt_col, val_col]].dropna().copy()
                temp = temp.sort_values(dt_col)
                temp["date"] = temp[dt_col].dt.date
                grouped = temp.groupby("date", as_index=False)[val_col].sum()
                grouped["date"] = pd.to_datetime(grouped["date"])

                if exclude_outliers and len(grouped) >= 5:
                    q1 = grouped[val_col].quantile(0.25)
                    q3 = grouped[val_col].quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 1.5 * iqr
                    grouped = grouped[grouped[val_col] <= upper].copy()

                trend_fig = px.line(grouped, x="date", y=val_col, title=f"{val_col} Trend Over Time")

                if len(grouped) >= 2:
                    x = np.arange(len(grouped))
                    y = grouped[val_col].values
                    slope, intercept = np.polyfit(x, y, 1)
                    future_x = np.arange(len(grouped) + forecast_periods)
                    future_y = intercept + slope * future_x
                    future_dates = pd.date_range(start=grouped["date"].iloc[0], periods=len(grouped) + forecast_periods, freq="D")

                    trend_fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_y,
                            mode="lines",
                            name="Forecast Trend",
                            line=dict(dash="dash")
                        )
                    )

                st.plotly_chart(trend_fig, use_container_width=True)
                st.caption("This trend view includes a simple forecast extension to support directional analysis.")

        with tab3:
            st.subheader("Relationship Analysis")

            if len(numeric_cols) >= 2:
                heatmap_fig, heatmap_caption = create_correlation_heatmap(df, numeric_cols)
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                st.caption(heatmap_caption)
            else:
                st.warning("At least two numeric columns are needed for correlation analysis.")

        with tab4:
            st.subheader("Key Business Insights")
            insights = generate_business_insights(df, numeric_cols, categorical_cols, datetime_cols)
            for insight in insights:
                st.write(f"- {insight}")

            st.subheader("Executive Summary")
            summary = generate_executive_summary(df, insights, numeric_cols, datetime_cols)
            st.text_area("Summary", summary, height=240)

        with tab5:
            st.subheader("Download Outputs")
            cleaned_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")