import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Analyst Copilot Dashboard", layout="wide")

st.title("AI Analyst Copilot Dashboard")
st.subheader("Upload a CSV or Excel file and get business-focused analysis")
st.markdown("### Turn raw data into actionable business insights instantly.")

st.markdown("### Turn raw data into actionable business insights instantly.")
st.caption("Upload a CSV or Excel file to explore KPIs, trends, product performance, and business insights.")


uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
st.caption("Tip: Upload sales or transaction data for best results.")


# ---------------------------
# Data loading and cleaning
# ---------------------------
def load_file(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df_local = pd.read_csv(file)
    else:
        df_local = pd.read_excel(file)
    return df_local


def clean_data(df_local: pd.DataFrame) -> pd.DataFrame:
    df_local = df_local.loc[:, ~df_local.columns.str.contains("^Unnamed", na=False)].copy()

    # Parse datetime if present
    if "order_datetime" in df_local.columns:
        df_local["order_datetime"] = pd.to_datetime(df_local["order_datetime"], errors="coerce")

    # Ensure time-like numeric columns are integers when possible
    time_like_cols = ["year", "month", "week_of_year", "day_of_week", "order_hour", "is_weekend"]
    for col in time_like_cols:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
            if df_local[col].notna().any():
                df_local[col] = df_local[col].round().astype("Int64")

    # Numeric conversion for key business columns
    numeric_candidates = [
        "sales_amount_gbp",
        "quantity_sold",
        "population_total",
        "gdp_current_usd",
        "gdp_growth_pct",
    ]
    for col in numeric_candidates:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    return df_local


# ---------------------------
# Filters
# ---------------------------
def apply_filters(df_local: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df_local.copy()

    st.sidebar.header("Filters")

    if "country" in filtered_df.columns:
        country_options = sorted(filtered_df["country"].dropna().astype(str).unique().tolist())
        selected_countries = st.sidebar.multiselect("Select Country", country_options)
        if selected_countries:
            filtered_df = filtered_df[filtered_df["country"].astype(str).isin(selected_countries)]

    if "year" in filtered_df.columns:
        year_values = filtered_df["year"].dropna().astype(int).unique().tolist()
        if year_values:
            year_values = sorted(year_values)
            selected_year_range = st.sidebar.slider(
                "Select Year Range",
                min_value=min(year_values),
                max_value=max(year_values),
                value=(min(year_values), max(year_values))
            )
            filtered_df = filtered_df[
                (filtered_df["year"] >= selected_year_range[0]) &
                (filtered_df["year"] <= selected_year_range[1])
            ]

    if "product_id" in filtered_df.columns:
        product_options = sorted(filtered_df["product_id"].dropna().astype(str).unique().tolist())
        limited_products = product_options[:200]
        selected_products = st.sidebar.multiselect("Select Product ID", limited_products)
        if selected_products:
            filtered_df = filtered_df[filtered_df["product_id"].astype(str).isin(selected_products)]

    return filtered_df


# ---------------------------
# KPIs
# ---------------------------
def create_kpis(df_local: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    total_records = len(df_local)

    total_sales = df_local["sales_amount_gbp"].sum() if "sales_amount_gbp" in df_local.columns else None
    total_qty = df_local["quantity_sold"].sum() if "quantity_sold" in df_local.columns else None
    avg_order_value = df_local["sales_amount_gbp"].mean() if "sales_amount_gbp" in df_local.columns else None

    peak_hour = None
    if "order_hour" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        hourly_sales = df_local.groupby("order_hour", dropna=True)["sales_amount_gbp"].sum()
        if not hourly_sales.empty:
            peak_hour = int(hourly_sales.idxmax())

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Total Sales (GBP)", f"{total_sales:,.2f}" if total_sales is not None else "N/A")
    col3.metric("Total Quantity", f"{int(total_qty):,}" if total_qty is not None else "N/A")
    col4.metric("Avg Order Value", f"{avg_order_value:,.2f}" if avg_order_value is not None else "N/A")
    col5.metric("Peak Hour", f"{peak_hour}:00" if peak_hour is not None else "N/A")


# ---------------------------
# Insight generation
# ---------------------------
def generate_business_insights(df_local: pd.DataFrame) -> list[str]:
    insights = []

    insights.append(f"Dataset contains {len(df_local):,} rows and {df_local.shape[1]} columns.")
    insights.append(f"Missing values: {int(df_local.isnull().sum().sum()):,}")

    if "sales_amount_gbp" in df_local.columns:
        total_sales = df_local["sales_amount_gbp"].sum()
        avg_sales = df_local["sales_amount_gbp"].mean()
        insights.append(f"Total sales are {total_sales:,.2f} GBP, with an average order value of {avg_sales:,.2f} GBP.")

    if "order_hour" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        hourly_sales = df_local.groupby("order_hour", dropna=True)["sales_amount_gbp"].sum().sort_values(ascending=False)
        if not hourly_sales.empty:
            best_hour = int(hourly_sales.index[0])
            best_hour_sales = hourly_sales.iloc[0]
            insights.append(f"Peak sales occur at {best_hour}:00, generating {best_hour_sales:,.2f} GBP.")

    if "country" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        country_sales = df_local.groupby("country", dropna=True)["sales_amount_gbp"].sum().sort_values(ascending=False)
        if not country_sales.empty:
            top_country = str(country_sales.index[0])
            top_country_sales = country_sales.iloc[0]
            insights.append(f"Top country by sales is {top_country} with {top_country_sales:,.2f} GBP.")

    if "is_weekend" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        weekend_sales = df_local.groupby("is_weekend", dropna=True)["sales_amount_gbp"].mean()
        if 0 in weekend_sales.index and 1 in weekend_sales.index:
            weekday_avg = weekend_sales.loc[0]
            weekend_avg = weekend_sales.loc[1]
            if weekend_avg > weekday_avg:
                insights.append("Weekend orders have a higher average sales value than weekday orders.")
            elif weekday_avg > weekend_avg:
                insights.append("Weekday orders have a higher average sales value than weekend orders.")
            else:
                insights.append("Weekend and weekday orders have very similar average sales values.")

    num_df = df_local.select_dtypes(include=[np.number]).copy()
    excluded_corr = {"year", "month", "week_of_year", "day_of_week", "is_weekend"}

    usable_corr_cols = [c for c in num_df.columns if c not in excluded_corr]
    if len(usable_corr_cols) >= 2:
        corr = num_df[usable_corr_cols].corr().abs().unstack().reset_index()
        corr.columns = ["A", "B", "Corr"]
        corr = corr[corr["A"] != corr["B"]]
        corr["pair"] = corr.apply(lambda r: tuple(sorted([r["A"], r["B"]])), axis=1)
        corr = corr.drop_duplicates("pair")
        corr = corr[corr["Corr"] < 0.99]

        if not corr.empty:
            top_corr = corr.sort_values("Corr", ascending=False).iloc[0]
            insights.append(
                f"The strongest non-trivial numeric relationship is between {top_corr['A']} and {top_corr['B']} ({top_corr['Corr']:.2f})."
            )

    return insights


def generate_executive_summary(df_local: pd.DataFrame, insights: list[str]) -> str:
    parts = []
    parts.append("This dashboard summarizes the uploaded dataset into operational and commercial signals that can support faster decision-making.")

    if "sales_amount_gbp" in df_local.columns:
        total_sales = df_local["sales_amount_gbp"].sum()
        parts.append(f"Sales performance is a major theme in this dataset, with total observed sales reaching {total_sales:,.2f} GBP.")

    if "country" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        country_sales = df_local.groupby("country", dropna=True)["sales_amount_gbp"].sum().sort_values(ascending=False)
        if not country_sales.empty:
            top_country = str(country_sales.index[0])
            parts.append(f"Geographically, {top_country} currently leads performance in the filtered view, which may indicate a strong market or customer concentration.")

    if "order_hour" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        hourly_sales = df_local.groupby("order_hour", dropna=True)["sales_amount_gbp"].sum().sort_values(ascending=False)
        if not hourly_sales.empty:
            best_hour = int(hourly_sales.index[0])
            parts.append(f"Operationally, demand appears concentrated around {best_hour}:00, which could inform staffing, promotions, and inventory planning.")

    if "is_weekend" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        weekend_sales = df_local.groupby("is_weekend", dropna=True)["sales_amount_gbp"].mean()
        if 0 in weekend_sales.index and 1 in weekend_sales.index:
            if weekend_sales.loc[0] > weekend_sales.loc[1]:
                parts.append("Customer behavior differs by day type, with weekday orders showing stronger average value than weekend orders.")
            else:
                parts.append("Customer behavior differs by day type, with weekend orders showing stronger average value than weekday orders.")

    parts.append("Recommended next steps are to focus on top-performing markets, validate the causes behind peak sales windows, and investigate whether product mix or timing is driving higher-value orders.")

    return " ".join(parts)


def get_top_highlight(df_local: pd.DataFrame) -> str:
    if "order_hour" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        hourly_sales = df_local.groupby("order_hour", dropna=True)["sales_amount_gbp"].sum()
        if not hourly_sales.empty:
            top_hour = int(hourly_sales.idxmax())
            top_value = hourly_sales.max()
            return f"Peak sales occur at {top_hour}:00, generating {top_value:,.2f} GBP."
    return "Use the filters and visuals below to uncover the strongest business signals in your data."


# ---------------------------
# Chart helpers
# ---------------------------
def get_plot_friendly_numeric_columns(df_local: pd.DataFrame) -> list[str]:
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def get_continuous_numeric_columns(df_local: pd.DataFrame) -> list[str]:
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"year", "month", "week_of_year", "day_of_week", "order_hour", "is_weekend"}
    return [c for c in numeric_cols if c not in excluded]


def create_distribution_chart(df_local: pd.DataFrame, selected_col: str):
    if selected_col in {"year", "month", "week_of_year", "day_of_week", "order_hour", "is_weekend"}:
        counts = (
            df_local[selected_col]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        counts.columns = [selected_col, "Count"]
        fig = px.bar(
            counts,
            x=selected_col,
            y="Count",
            title=f"Distribution of {selected_col}",
            labels={selected_col: selected_col, "Count": "Count"}
        )
        return fig, f"This bar chart shows how frequently each {selected_col} value appears in the filtered dataset."
    else:
        fig = px.histogram(
            df_local,
            x=selected_col,
            nbins=30,
            title=f"Distribution of {selected_col}"
        )
        return fig, f"This histogram shows how values in '{selected_col}' are distributed across the filtered dataset."


def create_trend_chart(df_local: pd.DataFrame):
    if "order_datetime" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        temp = df_local.dropna(subset=["order_datetime"]).copy()
        if not temp.empty:
            temp["order_date"] = temp["order_datetime"].dt.date
            trend = temp.groupby("order_date", as_index=False)["sales_amount_gbp"].sum()
            fig = px.line(
                trend,
                x="order_date",
                y="sales_amount_gbp",
                title="Sales Trend Over Time",
                labels={"order_date": "Date", "sales_amount_gbp": "Sales (GBP)"}
            )
            return fig, "This line chart shows how sales change over time and is more useful for trend analysis than a year-versus-month scatter plot."

    if {"year", "month", "sales_amount_gbp"}.issubset(df_local.columns):
        temp = df_local.dropna(subset=["year", "month"]).copy()
        if not temp.empty:
            temp["year"] = temp["year"].astype(int)
            temp["month"] = temp["month"].astype(int)
            temp["year_month"] = temp["year"].astype(str) + "-" + temp["month"].astype(str).str.zfill(2)
            trend = temp.groupby("year_month", as_index=False)["sales_amount_gbp"].sum()
            fig = px.line(
                trend,
                x="year_month",
                y="sales_amount_gbp",
                title="Sales Trend by Year-Month",
                labels={"year_month": "Year-Month", "sales_amount_gbp": "Sales (GBP)"}
            )
            return fig, "This line chart aggregates sales across year and month to reveal time-based trends."

    return None, "No suitable date or time fields were available to generate a trend chart."


def create_top_products_chart(df_local: pd.DataFrame):
    if "product_id" in df_local.columns and "sales_amount_gbp" in df_local.columns:
        top_products = (
            df_local.groupby("product_id", as_index=False)["sales_amount_gbp"]
            .sum()
            .sort_values("sales_amount_gbp", ascending=False)
            .head(10)
        )
        if not top_products.empty:
            fig = px.bar(
                top_products,
                x="product_id",
                y="sales_amount_gbp",
                title="Top 10 Products by Sales",
                labels={"product_id": "Product ID", "sales_amount_gbp": "Sales (GBP)"}
            )
            return fig, "This bar chart highlights the products contributing the most sales in the filtered view."
    return None, "Top product analysis is unavailable because required product or sales fields are missing."


def create_scatter_chart(df_local: pd.DataFrame, x_axis: str, y_axis: str):
    fig = px.scatter(
        df_local,
        x=x_axis,
        y=y_axis,
        opacity=0.6,
        title=f"{x_axis} vs {y_axis}"
    )
    caption = f"This scatter plot compares '{x_axis}' against '{y_axis}' to help identify visible relationships, clusters, or outliers."
    return fig, caption


def create_interactive_heatmap(df_local: pd.DataFrame):
    num_df = df_local.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        return None, "At least two numeric columns are required for correlation analysis."

    corr = num_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation")
        )
    )
    fig.update_layout(
        title="Interactive Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=700
    )
    caption = "This interactive heatmap highlights how strongly numeric variables move together. Hover over cells to inspect exact relationships."
    return fig, caption


# ---------------------------
# Main app
# ---------------------------
if uploaded_file is not None:
    try:
        with st.spinner("Analyzing data..."):
            df = load_file(uploaded_file)
            df = clean_data(df)
            df = apply_filters(df)

        if df.empty:
            st.warning("No data available after applying filters.")
            st.stop()

        st.success("File uploaded successfully")
        st.info(get_top_highlight(df))

        create_kpis(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Trends & Products",
            "Relationship Analysis",
            "Insights",
            "Downloads"
        ])

        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [str(dtype) for dtype in df.dtypes],
                "Missing Values": df.isnull().sum().values,
                "Unique Values": df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True)

            st.subheader("Summary Statistics")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe().T, use_container_width=True)
            else:
                st.warning("No numeric columns found.")

            st.subheader("Distribution Analysis")
            plot_cols = get_plot_friendly_numeric_columns(df)
            if plot_cols:
                selected_col = st.selectbox("Choose a column", plot_cols)
                fig, caption = create_distribution_chart(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(caption)
            else:
                st.warning("No numeric columns available for distribution analysis.")

        with tab2:
            st.subheader("Sales Trend")
            trend_fig, trend_caption = create_trend_chart(df)
            if trend_fig is not None:
                st.plotly_chart(trend_fig, use_container_width=True)
            st.caption(trend_caption)

            st.subheader("Top Products")
            product_fig, product_caption = create_top_products_chart(df)
            if product_fig is not None:
                st.plotly_chart(product_fig, use_container_width=True)
            st.caption(product_caption)

        with tab3:
            st.subheader("Scatter Plot")
            continuous_cols = get_continuous_numeric_columns(df)
            if len(continuous_cols) >= 2:
                x_axis = st.selectbox("Select X-axis", continuous_cols, key="x_axis")
                y_axis = st.selectbox("Select Y-axis", [c for c in continuous_cols if c != x_axis], key="y_axis")
                scatter_fig, scatter_caption = create_scatter_chart(df, x_axis, y_axis)
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.caption(scatter_caption)
            else:
                st.warning("At least two continuous numeric columns are needed for scatter plot analysis.")

            st.subheader("Correlation Heatmap")
            heatmap_fig, heatmap_caption = create_interactive_heatmap(df)
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption(heatmap_caption)

        with tab4:
            st.subheader("Key Business Insights")
            insights = generate_business_insights(df)
            for insight in insights:
                st.write(f"- {insight}")

            st.subheader("Executive Summary")
            summary = generate_executive_summary(df, insights)
            st.text_area("Summary", summary, height=220)

        with tab5:
            st.subheader("Download Outputs")
            cleaned_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")
