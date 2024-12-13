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
    # Read and merge CSVs
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(data_frames, ignore_index=True)
    
    # Check for at least the main columns existing in the combined DataFrame
    main_cols_original = ["Keyword", "Rank", "BrightEdge Volume", "URL"]
    for col in main_cols_original:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the uploaded files. Please ensure all files have this column.")
            st.stop()

    # Identify the first occurrence of each main column and use it as the designated domain column
    # We rename these main columns to standardized names
    rename_map = {
        "Keyword": "Keyword",
        "Rank": "Designated_Rank",
        "BrightEdge Volume": "BrightEdge_Volume",
        "URL": "Designated_URL"
    }
    
    # Rename just once; if duplicates exist, they remain as competitor columns
    # We'll do this by selecting columns in order and renaming just the first occurrence
    # to ensure we don't overwrite competitor columns with the same name.
    df = df.copy()  # Ensure a mutable copy
    # Create a mapping only for the first occurrence of each column
    # We'll achieve this by reordering columns: main first occurrences first
    # Then follow with other columns
    # This ensures that the first occurrences of the main columns appear at the start
    # and get renamed. Others remain as is.

    # Extract columns in a fixed order:
    cols_ordered = []
    for col in main_cols_original:
        # Find index of the first occurrence of this column
        first_idx = df.columns.get_loc(col)
        cols_ordered.append((first_idx, col))
    # Sort by their original order of appearance
    cols_ordered = sorted(cols_ordered, key=lambda x: x[0])
    # Extract just the column names in that order
    main_cols_in_order = [x[1] for x in cols_ordered]

    # Now, create a list of columns where main columns appear first (in discovered order),
    # followed by all other columns
    remaining_cols = [c for c in df.columns if c not in main_cols_in_order]
    final_cols = main_cols_in_order + remaining_cols

    df = df[final_cols]

    # Rename the first occurrences of the main columns
    df = df.rename(columns=rename_map)

    # Now we have:
    # Keyword | Designated_Rank | BrightEdge_Volume | Designated_URL | ...competitor columns...
    
    # Convert Designated_Rank to numeric
    df["Designated_Rank"] = pd.to_numeric(df["Designated_Rank"], errors="coerce")

    # Identify competitor rank columns as numeric columns beyond designated rank
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    competitor_rank_cols = [c for c in numeric_cols if c not in ["Designated_Rank", "BrightEdge_Volume"]]
    # We exclude BrightEdge_Volume from competitor ranks since it's not a rank column

    # Recommendation logic
    recommendations = []
    for _, row in df.iterrows():
        des_rank = row["Designated_Rank"]
        comp_ranks = [row[c] for c in competitor_rank_cols if pd.notna(row[c])]

        if pd.isna(des_rank):
            rec = "Create New Page"
        else:
            if len(comp_ranks) == 0:
                rec = "Defend"
            else:
                if des_rank < min(comp_ranks):
                    rec = "Defend"
                elif des_rank > max(comp_ranks):
                    rec = "Optimize Page"
                else:
                    rec = "Optimize Page"
        recommendations.append(rec)

    # Insert Recommendation column after Designated_URL
    if "Recommendation" not in df.columns:
        insert_pos = df.columns.get_loc("Designated_URL") + 1
        df.insert(insert_pos, "Recommendation", recommendations)
    else:
        df["Recommendation"] = recommendations

    # Filters
    st.subheader("Data Filters")
    keyword_filter = st.text_input("Filter by keyword (contains):", "")
    recommendation_filter = st.selectbox("Filter by recommendation:", options=["All", "Defend", "Optimize Page", "Create New Page"])

    filtered_df = df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].str.contains(keyword_filter, case=False, na=False)]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df["Recommendation"] == recommendation_filter]

    # Apply background gradient only to rank columns, not BrightEdge_Volume
    # rank_cols includes designated rank and competitor ranks.
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
            - Competitor data (ranks, URLs if available)
            - A Recommendation column (Defend, Optimize Page, or Create New Page)

            Based on this data, provide a brief summary of key opportunities, gaps, and strengths for {designated_domain}.
            """
            gpt_response = call_gpt_api(api_key, summary_prompt)
            if gpt_response:
                st.write(gpt_response)

    # Export the styled table to Excel
    st.subheader("Export Data")
    if st.button("Download Styled Excel"):
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
