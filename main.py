import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai

st.set_page_config(page_title="Keyword Ranking Analysis", layout="wide")
st.title("Keyword Ranking Analysis")

# Sidebar inputs
st.sidebar.header("GPT-4o-mini Credentials")
api_key = st.sidebar.text_input("Enter your GPT-4o-mini API key", type="password")

st.sidebar.header("Designated Domain")
designated_domain = st.sidebar.text_input("Enter your designated domain (e.g. example.com)")

st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

def call_gpt_api(api_key, prompt):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful SEO content strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

if uploaded_files and designated_domain:
    # Read and merge uploaded CSV files
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(data_frames, ignore_index=True)

    # Extract designated domain and competitor columns
    keyword_column = "Keyword"
    designated_rank_column = "Rank"
    designated_url_column = "URL"
    competitor_rank_columns = [col for col in df.columns if "Competitor Rank" in col]
    competitor_url_columns = [col for col in df.columns if "Competitor URL" in col]

    if not all([keyword_column, designated_rank_column, designated_url_column]):
        st.error("Missing required columns (Keyword, Rank, or URL). Please check your data.")
        st.stop()

    # Rename columns for clarity
    df = df.rename(columns={
        designated_rank_column: "Designated Rank",
        designated_url_column: "Designated URL"
    })

    # Ensure ranks are numeric
    df["Designated Rank"] = pd.to_numeric(df["Designated Rank"], errors="coerce")
    for col in competitor_rank_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add recommendation column
    def generate_recommendation(row):
        des_rank = row["Designated Rank"]
        comp_ranks = [row[col] for col in competitor_rank_columns if not pd.isna(row[col])]

        if pd.isna(des_rank):
            return "Create New Page"
        elif not comp_ranks or des_rank < min(comp_ranks):
            return "Defend"
        else:
            return "Optimize Page"

    df["Recommendation"] = df.apply(generate_recommendation, axis=1)

    # Filter and display
    st.subheader("Data Filters")
    keyword_filter = st.text_input("Filter by keyword:")
    recommendation_filter = st.selectbox("Filter by recommendation:", ["All", "Defend", "Optimize Page", "Create New Page"])

    filtered_df = df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df[keyword_column].str.contains(keyword_filter, case=False, na=False)]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df["Recommendation"] == recommendation_filter]

    # Highlight rank columns
    rank_columns = ["Designated Rank"] + competitor_rank_columns
    styled_df = filtered_df.style.background_gradient(subset=rank_columns, cmap="RdYlGn_r")

    st.subheader("Keyword Rankings Table")
    st.dataframe(styled_df, use_container_width=True)

    # Generate insights
    st.subheader("Summary Insights")
    if not api_key:
        st.warning("Provide your GPT-4o-mini API key to generate insights.")
    elif st.button("Generate Insights"):
        prompt = f"""
        Analyze the keyword rankings data for {designated_domain}. Highlight key gaps, opportunities, and strengths.
        """
        insights = call_gpt_api(api_key, prompt)
        if insights:
            st.write(insights)

    # Export to Excel
    st.subheader("Export Data")
    if st.button("Download Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Rankings")
        buffer.seek(0)
        st.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="keyword_rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Upload your data and enter the designated domain to get started.")
