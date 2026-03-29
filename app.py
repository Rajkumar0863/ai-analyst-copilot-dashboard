import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI Analyst Copilot Dashboard", layout="wide")


# ---------------------------
# Helpers
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
# File loading
# ---------------------------
def load_file(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file, low_memory=False)
    return pd.read_excel(file)


# ---------------------------
# Cleaning and type inference
# ---------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    # Try datetime conversion for object columns
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_datetime(df[col], errors="coerce")
            success_ratio = converted.notna().mean()
            if success_ratio >= 0.8:
                df[col] = converted

    # Try numeric conversion for remaining object columns
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            success_ratio = converted.notna().mean()
            if success_ratio >= 0.8:
                df[col] = converted

    return df


def detect_column_profile(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

    low_cardinality_numeric = []
    continuous_numeric = []

    for col in numeric_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 20:
            low_cardinality_numeric.append(col)
        else:
            continuous_numeric.append(col)

    return {
        "numeric": numeric_cols,
        "continuous_numeric": continuous_numeric,
        "low_cardinality_numeric": low_cardinality_numeric,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
    }


# ---------------------------
# Filters
# ---------------------------
def apply_filters(df: pd.DataFrame, profile: dict) -> dict:
    filtered_df = df.copy()

    st.sidebar.header("Filters")

    # categorical filters
    for col in profile["categorical"][:5]:
        unique_vals = filtered_df[col].dropna().astype(str).unique().tolist()
        if 1 < len(unique_vals) <= 100:
            selected = st.sidebar.multiselect(f"Filter {col}", sorted(unique_vals))
            if selected:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    # numeric filters
    numeric_candidates = profile["continuous_numeric"][:3] + profile["low_cardinality_numeric"][:2]
    for col in numeric_candidates[:5]:
        non_null = filtered_df[col].dropna()
        if not non_null.empty:
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
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

    # datetime filters
    for col in profile["datetime"][:2]:
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
                if isinstance(selected_dates, tuple) or isinstance(selected_dates, list):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        filtered_df = filtered_df[
                            (filtered_df[col].dt.date >= start_date) &
                            (filtered_df[col].dt.date <= end_date)
                        ]

    # options
    with st.sidebar.expander("Analysis Options", expanded=False):
        exclude_outliers = st.checkbox("Exclude outliers in trend charts", value=False)
        forecast_periods = st.slider("Forecast periods", min_value=3, max_value=30, value=7)

    return {
        "df": filtered_df,
        "exclude_outliers": exclude_outliers,
        "forecast_periods": forecast_periods,
    }


# ---------------------------
# KPI section
# ---------------------------
def compute_aggregation(df: pd.DataFrame, metric_col: str, agg_type: str):
    if metric_col not in df.columns:
        return None

    series = df[metric_col].dropna()
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


def create_kpis(df: pd.DataFrame, profile: dict):
    c1, c2, c3 = st.columns(3)

    total_rows = len(df)
    total_cols = df.shape[1]
    missing_vals = int(df.isnull().sum().sum())

    c1.metric("Rows", format_compact_number(total_rows))
    c2.metric("Columns", format_compact_number(total_cols))
    c3.metric("Missing Values", format_compact_number(missing_vals))

    st.caption(
        f"Rows: {format_full_number(total_rows, 0)} | "
        f"Columns: {format_full_number(total_cols, 0)} | "
        f"Missing Values: {format_full_number(missing_vals, 0)}"
    )

    if profile["numeric"]:
        st.markdown("### Custom Metric")
        m1, m2, m3 = st.columns(3)

        metric_col = m1.selectbox("Choose numeric column", profile["numeric"])
        agg_type = m2.selectbox("Choose aggregation", ["sum", "mean", "median", "count", "min", "max"])
        metric_value = compute_aggregation(df, metric_col, agg_type)

        m3.metric(f"{agg_type.title()} of {metric_col}", format_compact_number(metric_value))
        st.caption(f"Exact value: {format_full_number(metric_value, 2) if metric_value is not None else 'N/A'}")


# ---------------------------
# Smart charting
# ---------------------------
def classify_single_column(col: str, profile: dict) -> str:
    if col in profile["datetime"]:
        return "datetime"
    if col in profile["numeric"]:
        return "numeric"
    return "categorical"


def recommend_chart(x_col: str, y_col: str | None, profile: dict) -> str:
    x_type = classify_single_column(x_col, profile)
    y_type = None if y_col is None else classify_single_column(y_col, profile)

    if y_col is None:
        if x_type == "numeric":
            if x_col in profile["low_cardinality_numeric"]:
                return "count_bar"
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


def build_chart(df: pd.DataFrame, x_col: str, y_col: str | None, chart_type: str):
    if chart_type == "histogram":
        fig = px.histogram(df, x=x_col, nbins=30, title=f"Distribution of {x_col}")
        caption = f"This histogram shows the distribution of values in '{x_col}'."
        return fig, caption

    if chart_type == "count_bar":
        counts = df[x_col].astype(str).value_counts().head(20).reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.bar(counts, x=x_col, y="Count", title=f"Count of {x_col}")
        fig.update_xaxes(type="category")
        caption = f"This bar chart shows the frequency of the most common values in '{x_col}'."
        return fig, caption

    if chart_type == "line":
        temp = df[[x_col, y_col]].dropna().sort_values(x_col)
        if temp.empty:
            return None, "No valid data available for line chart."
        grouped = temp.groupby(x_col, as_index=False)[y_col].sum()
        fig = px.line(grouped, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        caption = f"This line chart shows how '{y_col}' changes over time using '{x_col}'."
        return fig, caption

    if chart_type == "bar":
        temp = df[[x_col, y_col]].dropna()
        if temp.empty:
            return None, "No valid data available for bar chart."
        grouped = temp.groupby(x_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False).head(20)
        grouped[x_col] = grouped[x_col].astype(str)
        fig = px.bar(
            grouped.sort_values(y_col, ascending=True),
            x=y_col,
            y=x_col,
            orientation="h",
            title=f"{y_col} by {x_col}",
        )
        caption = f"This horizontal bar chart compares aggregated '{y_col}' values across '{x_col}'."
        return fig, caption

    if chart_type == "scatter":
        temp = df[[x_col, y_col]].dropna()
        if temp.empty:
            return None, "No valid data available for scatter plot."
        fig = px.scatter(temp, x=x_col, y=y_col, opacity=0.6, title=f"{x_col} vs {y_col}")
        caption = f"This scatter plot compares '{x_col}' and '{y_col}' to reveal relationships or outliers."
        return fig, caption

    if chart_type == "heatmap_table":
        temp = df[[x_col, y_col]].dropna().copy()
        if temp.empty:
            return None, "No valid data available for categorical heatmap."
        pivot = pd.crosstab(temp[x_col].astype(str), temp[y_col].astype(str))
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Blues",
            )
        )
        fig.update_layout(title=f"{x_col} vs {y_col}")
        caption = f"This heatmap shows how '{x_col}' and '{y_col}' categories intersect."
        return fig, caption

    return None, "Could not determine a suitable chart type."


def build_forecast_chart(df: pd.DataFrame, datetime_col: str, numeric_col: str, exclude_outliers: bool, forecast_periods: int):
    temp = df[[datetime_col, numeric_col]].dropna().copy()
    if temp.empty:
        return None, "No valid rows available for forecasting."

    temp = temp.sort_values(datetime_col)
    temp["date"] = temp[datetime_col].dt.date
    grouped = temp.groupby("date", as_index=False)[numeric_col].sum()
    grouped["date"] = pd.to_datetime(grouped["date"])

    if exclude_outliers and len(grouped) >= 5:
        q1 = grouped[numeric_col].quantile(0.25)
        q3 = grouped[numeric_col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        grouped = grouped[grouped[numeric_col] <= upper].copy()

    fig = px.line(grouped, x="date", y=numeric_col, title=f"{numeric_col} Trend Over Time")

    if len(grouped) >= 2:
        x = np.arange(len(grouped))
        y = grouped[numeric_col].values
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(grouped) + forecast_periods)
        future_y = intercept + slope * future_x
        future_dates = pd.date_range(start=grouped["date"].iloc[0], periods=len(grouped) + forecast_periods, freq="D")

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_y,
                mode="lines",
                name="Forecast Trend",
                line=dict(dash="dash"),
            )
        )

    caption = "This trend view includes a simple forecast extension based on the detected datetime and numeric columns."
    return fig, caption


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    if len(numeric_cols) < 2:
        return None, "At least two numeric columns are required for correlation analysis."

    corr = df[numeric_cols].corr()

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
    fig.update_layout(title="Interactive Correlation Heatmap", height=700)

    caption = "This heatmap shows how strongly numeric variables are related to each other."
    return fig, caption


# ---------------------------
# Insights
# ---------------------------
def generate_insights(df: pd.DataFrame, profile: dict) -> list[str]:
    insights = []

    insights.append(f"The dataset contains {len(df):,} rows and {df.shape[1]} columns.")
    insights.append(f"There are {int(df.isnull().sum().sum()):,} missing values in the filtered dataset.")

    if profile["numeric"]:
        most_missing_num = df[profile["numeric"]].isnull().sum().sort_values(ascending=False)
        if not most_missing_num.empty:
            insights.append(
                f"The numeric column with the most missing values is '{most_missing_num.index[0]}' with {int(most_missing_num.iloc[0]):,} missing entries."
            )

        variance_series = df[profile["numeric"]].std(numeric_only=True).sort_values(ascending=False)
        if not variance_series.empty:
            insights.append(
                f"The most variable numeric column is '{variance_series.index[0]}', suggesting it has the widest spread of values."
            )

    if profile["categorical"]:
        cat_col = profile["categorical"][0]
        top_vals = df[cat_col].astype(str).value_counts().head(3)
        if not top_vals.empty:
            text = ", ".join([f"{idx} ({val})" for idx, val in top_vals.items()])
            insights.append(f"The most common values in '{cat_col}' are {text}.")

    if len(profile["numeric"]) >= 2:
        corr = df[profile["numeric"]].corr().abs().unstack().reset_index()
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

    if profile["datetime"]:
        insights.append("Datetime columns were detected, so time-based trend and forecast analysis is available.")

    return insights


def generate_summary(df: pd.DataFrame, profile: dict, insights: list[str]) -> str:
    summary = []
    summary.append(
        "This dashboard automatically profiles the uploaded dataset and converts it into interactive metrics, visualizations, and exploratory insights."
    )

    if profile["numeric"]:
        summary.append(
            f"The dataset contains {len(profile['numeric'])} numeric columns, enabling aggregation, distribution analysis, and relationship detection."
        )

    if profile["datetime"]:
        summary.append(
            "Datetime fields are present, which means the dashboard can support time-based trend analysis and simple forecasting."
        )

    if profile["categorical"]:
        summary.append(
            "Categorical fields are available for segmentation and comparison, making it possible to explore frequency patterns and grouped performance."
        )

    summary.append(
        "The most useful next step is to focus on the highest-variance variables, strongest correlations, and any trend anomalies that may deserve closer investigation."
    )

    return " ".join(summary)


# ---------------------------
# Main App
# ---------------------------
st.title("AI Analyst Copilot Dashboard")
st.subheader("Upload a CSV or Excel file and get business-focused analysis")
st.markdown("### Turn raw data into actionable business insights instantly.")
st.caption("Upload a CSV or Excel file to explore KPIs, trends, chart recommendations, and automated insights.")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        st.caption("Tip: Upload structured business, sales, operations, HR, finance, survey, or analytics data for best results.")

        with st.spinner("Analyzing data..."):
            df = load_file(uploaded_file)
            df = clean_data(df)
            profile = detect_column_profile(df)
            filter_result = apply_filters(df, profile)
            df = filter_result["df"]
            exclude_outliers = filter_result["exclude_outliers"]
            forecast_periods = filter_result["forecast_periods"]
            profile = detect_column_profile(df)

        if df.empty:
            st.warning("No data available after applying filters.")
            st.stop()

        st.success("File uploaded successfully")
        st.info("The app has automatically detected column types and enabled dynamic analysis for your dataset.")

        create_kpis(df, profile)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Visual Explorer", "Forecast & Trends", "Insights", "Downloads"]
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
            if profile["numeric"]:
                st.dataframe(df[profile["numeric"]].describe().T, use_container_width=True)
            else:
                st.warning("No numeric columns found in this dataset.")

        with tab2:
            st.subheader("Smart Chart Builder")

            all_cols = df.columns.tolist()
            x_col = st.selectbox("Select X-axis / grouping column", all_cols)
            y_options = ["None"] + all_cols
            y_choice = st.selectbox("Select Y-axis / metric column", y_options)
            y_col = None if y_choice == "None" else y_choice

            recommended_chart = recommend_chart(x_col, y_col, profile)
            st.caption(f"Recommended chart type: **{recommended_chart.replace('_', ' ').title()}**")

            fig, caption = build_chart(df, x_col, y_col, recommended_chart)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)

            if len(profile["numeric"]) >= 2:
                st.subheader("Correlation Heatmap")
                heatmap_fig, heatmap_caption = create_correlation_heatmap(df, profile["numeric"])
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                st.caption(heatmap_caption)

        with tab3:
            st.subheader("Forecast & Trends")
            if profile["datetime"] and profile["numeric"]:
                dt_col = st.selectbox("Select datetime column", profile["datetime"])
                num_col = st.selectbox("Select numeric column for forecasting", profile["numeric"])

                trend_fig, trend_caption = build_forecast_chart(
                    df,
                    dt_col,
                    num_col,
                    exclude_outliers,
                    forecast_periods,
                )
                if trend_fig is not None:
                    st.plotly_chart(trend_fig, use_container_width=True)
                st.caption(trend_caption)
            else:
                st.info("No valid datetime + numeric combination was detected for forecasting.")

        with tab4:
            st.subheader("Key Dataset Insights")
            insights = generate_insights(df, profile)
            for insight in insights:
                st.write(f"- {insight}")

            st.subheader("Executive Summary")
            summary = generate_summary(df, profile, insights)
            st.text_area("Summary", summary, height=220)

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