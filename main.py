import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai

# Set up page
st.set_page_config(page_title="Keyword Ranking Analysis", layout="wide")
st.title("Keyword Ranking Analysis")

# Sidebar Inputs
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
    data_frames = []
    for file in uploaded_files:
        df_temp = pd.read_csv(file)
        data_frames.append(df_temp)
    df = pd.concat(data_frames, ignore_index=True)

    # Expected columns:
    # 1: Keyword
    # 2: Rank (Designated_Rank)
    # 3: BrightEdge_Volume (Search Volume)
    # 4: URL (Designated_URL)
    # Columns 5 onward: Competitor Rank/URL pairs
    expected_cols = ["Keyword", "Designated_Rank", "BrightEdge_Volume", "Designated_URL"]

    if len(df.columns) < len(expected_cols):
        st.error("The uploaded CSV(s) does not have the expected structure.")
    else:
        df.columns = expected_cols + list(df.columns[len(expected_cols):])
        
        # Identify competitor columns
        comp_cols = df.columns[4:]
        if len(comp_cols) % 2 != 0:
            st.warning("The number of competitor columns is odd, meaning ranks/URLs may not be properly paired.")

        competitor_rank_cols = [col for i, col in enumerate(comp_cols) if i % 2 == 0]
        competitor_url_cols = [col for i, col in enumerate(comp_cols) if i % 2 == 1]

        # Ensure rank columns are numeric
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

        # Insert Recommendations column after Designated_URL
        df.insert(4, "Recommendation", recommendations)

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
        styled_df = filtered_df.style.background_gradient(
            subset=rank_cols,
            cmap="RdYlGn_r"
        )

        st.subheader("Keyword Rankings Table")
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
                - A Recommendation column with: Defend, Optimize Page, or Create New Page

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
