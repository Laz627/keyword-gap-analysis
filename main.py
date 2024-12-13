import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai
import matplotlib

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
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from the uploaded files: {missing_cols}")
        st.stop()

    # Rename known columns to standardized names for the designated domain
    rename_map = {
        "Keyword": "Keyword",
        "Rank": "Designated_Rank",
        "BrightEdge Volume": "BrightEdge_Volume",
        "URL": "Designated_URL"
    }
    df = df.rename(columns=rename_map)

    # Identify any columns beyond the first four as competitor columns
    standard_cols = list(rename_map.values())
    extra_cols = [c for c in df.columns if c not in standard_cols]

    # Convert ranks to numeric
    df["Designated_Rank"] = pd.to_numeric(df["Designated_Rank"], errors="coerce")

    # Any extra columns are treated as competitor data.
    # We won't enforce pairs. If competitor columns exist, they are shown as is.
    # If you do have pairs (Rank, URL), they will appear in the final table.
    # If competitor rank columns exist, try to convert them to numeric
    # We'll guess competitor rank columns by checking if they are numeric.
    # If not numeric, we skip conversion.
    for c in extra_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except:
            pass

    # Recommendation logic
    # Assume competitor rank columns are any numeric columns except the designated rank
    # This is a heuristic. Adjust if needed.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    competitor_rank_cols = [c for c in numeric_cols if c not in ["Designated_Rank"]]

    recommendations = []
    for idx, row in df.iterrows():
        des_rank = row["Designated_Rank"]
        comp_ranks = [row[c] for c in competitor_rank_cols if pd.notna(row[c])]

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
    if "Recommendation" not in df.columns:
        df.insert(df.columns.get_loc("Designated_URL") + 1, "Recommendation", recommendations)
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

    # Styling rank columns
    # Highlight designated and competitor rank columns
    rank_cols = ["Designated_Rank"] + competitor_rank_cols if competitor_rank_cols else ["Designated_Rank"]
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
