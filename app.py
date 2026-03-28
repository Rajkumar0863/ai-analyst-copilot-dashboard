import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="AI Analyst Copilot Dashboard", layout="wide")

st.title("AI Analyst Copilot Dashboard")
st.subheader("Upload a CSV or Excel file and get business-focused analysis")
st.markdown("### Turn raw data into actionable business insights instantly.")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
st.markdown("### Turn raw data into actionable business insights instantly.")


def load_file(file):
    if file.name.endswith(".csv"):
        df_local = pd.read_csv(file)
    else:
        df_local = pd.read_excel(file)
    return df_local


def clean_data(df_local):
    df_local = df_local.loc[:, ~df_local.columns.str.contains("^Unnamed")]
    return df_local


def apply_filters(df_local):
    filtered_df = df_local.copy()
    st.sidebar.header("Filters")

    if "country" in filtered_df.columns:
        options = sorted(filtered_df["country"].dropna().astype(str).unique())
        selected = st.sidebar.multiselect("Select Country", options)
        if selected:
            filtered_df = filtered_df[filtered_df["country"].astype(str).isin(selected)]

    if "year" in filtered_df.columns:
        years = sorted(filtered_df["year"].dropna().unique())
        if years:
            yr_min, yr_max = int(min(years)), int(max(years))
            yr_range = st.sidebar.slider("Select Year Range", yr_min, yr_max, (yr_min, yr_max))
            filtered_df = filtered_df[
                (filtered_df["year"] >= yr_range[0]) &
                (filtered_df["year"] <= yr_range[1])
            ]

    return filtered_df


def create_kpis(df):
    col1, col2, col3, col4, col5 = st.columns(5)

    total_rows = len(df)
    total_sales = df["sales_amount_gbp"].sum() if "sales_amount_gbp" in df.columns else None
    total_qty = df["quantity_sold"].sum() if "quantity_sold" in df.columns else None
    avg_order = df["sales_amount_gbp"].mean() if "sales_amount_gbp" in df.columns else None

    if "order_hour" in df.columns and not df["order_hour"].dropna().empty:
        peak_hour = int(df["order_hour"].mode().iloc[0])
    else:
        peak_hour = None

    col1.metric("Total Records", f"{total_rows:,}")
    col2.metric("Total Sales (GBP)", f"{total_sales:,.2f}" if total_sales is not None else "N/A")
    col3.metric("Total Quantity", f"{int(total_qty):,}" if total_qty is not None else "N/A")
    col4.metric("Avg Order Value", f"{avg_order:,.2f}" if avg_order is not None else "N/A")
    col5.metric("Peak Hour", str(peak_hour) if peak_hour is not None else "N/A")


def get_top_highlight(df):
    if "order_hour" in df.columns and "sales_amount_gbp" in df.columns:
        grp = df.groupby("order_hour")["sales_amount_gbp"].sum()
        if not grp.empty:
            top = grp.idxmax()
            val = grp.max()
            return f"Peak sales occur at {top}:00, generating {val:,.2f} GBP."
    return "Use filters to uncover key business insights."


def generate_business_insights(df):
    insights = []

    insights.append(f"Dataset contains {len(df)} rows and {df.shape[1]} columns.")
    insights.append(f"Missing values: {int(df.isnull().sum().sum())}")

    if "sales_amount_gbp" in df.columns:
        insights.append(f"Total sales: {df['sales_amount_gbp'].sum():,.2f} GBP")

    if "order_hour" in df.columns and "sales_amount_gbp" in df.columns:
        hourly_sales = df.groupby("order_hour")["sales_amount_gbp"].sum()
        if not hourly_sales.empty:
            peak = hourly_sales.idxmax()
            peak_value = hourly_sales.max()
            insights.append(f"Peak sales hour: {peak}:00 with total sales of {peak_value:,.2f} GBP")

    if "country" in df.columns and "sales_amount_gbp" in df.columns:
        country_sales = df.groupby("country")["sales_amount_gbp"].sum().sort_values(ascending=False)
        if not country_sales.empty:
            top_country = country_sales.index[0]
            top_country_sales = country_sales.iloc[0]
            insights.append(f"Top country by sales: {top_country} with {top_country_sales:,.2f} GBP")

    if "is_weekend" in df.columns and "sales_amount_gbp" in df.columns:
        weekend_sales = df.groupby("is_weekend")["sales_amount_gbp"].mean()
        if len(weekend_sales) >= 2:
            weekday_avg = weekend_sales.get(0, None)
            weekend_avg = weekend_sales.get(1, None)

            if weekday_avg is not None and weekend_avg is not None:
                if weekend_avg > weekday_avg:
                    insights.append("Weekend orders have a higher average sales value than weekday orders.")
                elif weekend_avg < weekday_avg:
                    insights.append("Weekday orders have a higher average sales value than weekend orders.")
                else:
                    insights.append("Weekend and weekday orders show similar average sales values.")

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) > 1:
        corr = num_df.corr().abs().unstack().reset_index()
        corr.columns = ["A", "B", "Corr"]
        corr = corr[corr["A"] != corr["B"]]

        corr["pair"] = corr.apply(lambda x: tuple(sorted([x["A"], x["B"]])), axis=1)
        corr = corr.drop_duplicates("pair")
        corr = corr[corr["Corr"] < 0.99]

        if not corr.empty:
            top = corr.sort_values("Corr", ascending=False).iloc[0]
            insights.append(f"Strong relationship: {top['A']} ↔ {top['B']} ({top['Corr']:.2f})")

    return insights


def generate_summary(insights):
    text = "EXECUTIVE SUMMARY\n\n"
    text += "Key findings:\n"
    for i, ins in enumerate(insights[:6], 1):
        text += f"{i}. {ins}\n"

    text += "\nRecommendations:\n"
    text += "- Focus on peak sales hours\n"
    text += "- Optimize high-performing regions\n"
    text += "- Compare weekday and weekend order behavior\n"
    text += "- Use correlations to guide decisions\n"

    return text


if uploaded_file:
    try:
        df = load_file(uploaded_file)
        df = clean_data(df)
        df = apply_filters(df)

        if df.empty:
            st.warning("No data available after applying filters.")
            st.stop()

        st.success("File uploaded successfully")
        st.info(get_top_highlight(df))

        st.header("1. KPI Overview")
        create_kpis(df)

        st.header("2. Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.header("3. Histogram")
        num_cols = df.select_dtypes(include="number").columns

        if len(num_cols) > 0:
            col = st.selectbox("Column", num_cols)
            fig, ax = plt.subplots()
            df[col].hist(ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.header("4. Scatter Plot")
        if len(num_cols) > 1:
            x = st.selectbox("X", num_cols, key="x")
            y = st.selectbox("Y", [c for c in num_cols if c != x], key="y")
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y], alpha=0.5)
            ax.set_title(f"{x} vs {y}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            st.pyplot(fig)
            st.caption("This shows how weekly patterns affect sales volume.")

        st.header("5. Heatmap")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        st.header("6. Key Business Insights")
        insights = generate_business_insights(df)
        for i in insights:
            st.write("-", i)

        st.header("7. Executive Summary")
        summary = generate_summary(insights)
        st.text_area("Summary", summary, height=260)

        st.header("8. Download")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_data.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")
