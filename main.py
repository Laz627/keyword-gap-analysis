import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai

st.set_page_config(page_title="Keyword Ranking Analysis", layout="wide")
st.title("Keyword Ranking Analysis")

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
            max_tokens=16000
        )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

if uploaded_files and designated_domain:
    # Concatenate all uploaded CSVs
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(data_frames, ignore_index=True)

    # Check for required columns
    required_columns = ["Keyword", "Rank", "BrightEdge Volume", "URL"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the uploaded files. Please ensure all files have this column.")
            st.stop()

    # Rename known columns to a standard format
    # We'll map:
    # Keyword -> Keyword
    # Rank -> Designated_Rank
    # BrightEdge Volume -> BrightEdge_Volume
    # URL -> Designated_URL
    rename_map = {
        "Keyword": "Keyword",
        "Rank": "Designated_Rank",
        "BrightEdge Volume": "BrightEdge_Volume",
        "URL": "Designated_URL"
    }
    df = df.rename(columns=rename_map)

    # All columns beyond these four are considered competitor columns
    # They should come in pairs: Competitor_Rank, Competitor_URL
    # Check how many extra columns we have
    extra_cols = [c for c in df.columns if c not in rename_map.values()]

    # Competitor rank columns will be those with numeric data (if available),
    # and their corresponding URL columns come next. 
    # If your CSVs are structured consistently (rank then URL), you might rely 
    # on even/odd indexing of these extra columns. Otherwise, you'll need a more robust approach.

    # For simplicity, we assume the original structure after the four main columns 
    # is consistently pairs of (CompetitorX_Rank, CompetitorX_URL).
    # We can identify columns by their order:
    competitor_rank_cols = extra_cols[0::2]  # even indexed extra cols
    competitor_url_cols = extra_cols[1::2]   # odd indexed extra cols

    # Convert rank columns to numeric
    df["Designated_Rank"] = pd.to_numeric(df["Designated_Rank"], errors="coerce")
    for c in competitor_rank_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Recommendation logic
    recommendations = []
    for idx, row in df.iterrows():
        des_rank = row["Designated_Rank"]
        comp_ranks = [row[c] for c in competitor_rank_cols if not pd.isna(row[c])]

        if pd.isna(des_rank):
            # Domain does not rank
            rec = "Create New Page"
        else:
            if len(comp_ranks) == 0:
                # No competitor data, domain is de facto best
                rec = "Defend"
            else:
                # Check positioning
                if des_rank < min(comp_ranks):
                    rec = "Defend"
                elif des_rank > max(comp_ranks):
                    rec = "Optimize Page"
                else:
                    rec = "Optimize Page"
        recommendations.append(rec)

    df.insert(4, "Recommendation", recommendations)

    # Display the raw DataFrame first to ensure all columns are present
    st.subheader("Raw Merged DataFrame")
    st.write(df)

    # Filters
    st.subheader("Data Filters")
    keyword_filter = st.text_input("Filter by keyword (contains):", "")
    recommendation_filter = st.selectbox("Filter by recommendation:", options=["All", "Defend", "Optimize Page", "Create New Page"])

    filtered_df = df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].str.contains(keyword_filter, case=False, na=False)]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df["Recommendation"] == recommendation_filter]

    # Styling rank columns
    rank_cols = ["Designated_Rank"] + competitor_rank_cols
    import matplotlib  # ensure matplotlib is installed
    styled_df = filtered_df.style.background_gradient(
        subset=rank_cols,
        cmap="RdYlGn_r"
    )

    st.subheader("Keyword Rankings Table (Filtered & Styled)")
    st.dataframe(styled_df, use_container_width=True)

    # GPT Insights
    st.subheader("Summary Insights")
    if not api_key:
        st.warning("Please provide your GPT-4o-mini API key in the sidebar to generate insights.")
    else:
        if st.button("Generate GPT Insights"):
            summary_prompt = f"""
            You are analyzing SEO keyword rankings data for the designated domain: {designated_domain}.
            The data includes:
            - Keywords
            - Designated domain rank
            - BrightEdge Volume (search volume)
            - Designated domain URL
            - Competitor ranks and URLs
            - A Recommendation column (Defend, Optimize Page, or Create New Page)

            Based on this data, provide a brief summary of key opportunities, gaps, and strengths for {designated_domain}.
            """
            gpt_response = call_gpt_api(api_key, summary_prompt)
            if gpt_response:
                st.write(gpt_response)

    # Export the styled table to Excel
    st.subheader("Export Data")
    if st.button("Download Styled Excel"):
        # Convert styled df to Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            styled_df.to_excel(writer, index=False, sheet_name='Rankings')
        buffer.seek(0)

        st.download_button(
            label="Download Excel file",
            data=buffer,
            file_name="styled_rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not designated_domain:
        st.info("Please enter your designated domain in the sidebar.")
    else:
        st.info("Please upload one or more CSV files to begin.")
