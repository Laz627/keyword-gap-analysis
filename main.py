import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai

# Streamlit app configuration
st.set_page_config(page_title="Keyword Ranking Analysis", layout="wide")
st.title("Keyword Ranking Analysis")

# Sidebar inputs
st.sidebar.header("Target Domain")
target_domain = st.sidebar.text_input("Enter the target domain (e.g., 'pella.com')")

st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

def clean_rank_data(df, rank_columns):
    """Clean and prepare rank data for styling"""
    df = df.copy()
    for col in rank_columns:
        if col in df.columns:
            # Convert to numeric, replace non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace NaN with a high number (e.g., 9999)
            df[col] = df[col].fillna(9999)
            # Ensure values are within reasonable range
            df[col] = df[col].clip(1, 9999)
    return df

# Function to generate recommendations
def generate_recommendation(row, target_rank_column, competitor_rank_columns):
    target_rank = row[target_rank_column]
    competitor_ranks = [row[col] for col in competitor_rank_columns if not pd.isna(row[col])]

    if pd.isna(target_rank):  # If target domain does not rank
        return "Create New Page"
    elif not competitor_ranks or target_rank < min(competitor_ranks):  # If target domain ranks better
        return "Defend"
    else:  # If target domain ranks worse
        return "Optimize Page"

if uploaded_files and target_domain:
    # Load and merge CSV files
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    merged_df = pd.concat(data_frames, ignore_index=True)

    # Standardize column names
    merged_df.columns = [col.strip() for col in merged_df.columns]

    # Identify target domain's rank and URL based on URL column values
    merged_df["Target Domain"] = merged_df["URL"].apply(lambda x: target_domain in str(x).lower())
    target_rows = merged_df[merged_df["Target Domain"]]

    if target_rows.empty:
        st.error(f"Target domain '{target_domain}' not found in the uploaded data URLs.")
        st.stop()

    target_rank_column = "Rank"
    target_url_column = "URL"

    # Competitor rank columns
    competitor_rank_columns = ["Rank"] if len(data_frames) > 1 else []

    # Generate recommendations
    merged_df["Recommendation"] = merged_df.apply(
        generate_recommendation, axis=1, args=(target_rank_column, competitor_rank_columns)
    )

    # Display filters
    st.subheader("Filters")
    keyword_filter = st.text_input("Filter by keyword:")
    recommendation_filter = st.selectbox("Filter by recommendation:", ["All", "Defend", "Optimize Page", "Create New Page"])

    filtered_df = merged_df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].str.contains(keyword_filter, case=False, na=False)]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df["Recommendation"] == recommendation_filter]

    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        st.warning("No data to display after applying filters.")
        st.stop()

    # Rank columns to style
    rank_columns_to_style = [target_rank_column] + competitor_rank_columns
    
    # Clean the rank data
    filtered_df = clean_rank_data(filtered_df, rank_columns_to_style)

    # Display the DataFrame with styling
    st.subheader("Keyword Rankings Table")

    try:
        # Create a copy for styling
        display_df = filtered_df.copy()
        
        # Identify numeric columns for styling
        numeric_columns = [col for col in rank_columns_to_style 
                         if col in display_df.columns 
                         and pd.api.types.is_numeric_dtype(display_df[col])]
        
        if numeric_columns:
            # Create styler object
            styler = display_df.style
            
            # Apply background gradient to numeric columns
            styler = styler.background_gradient(
                subset=numeric_columns,
                cmap='RdYlGn_r',
                vmin=1,
                vmax=100
            )
            
            # Display the styled DataFrame
            st.dataframe(styler, use_container_width=True)
        else:
            # If no numeric columns, display without styling
            st.dataframe(display_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error applying styling: {str(e)}")
        # Fallback to displaying unstyled DataFrame
        st.dataframe(display_df, use_container_width=True)

    # Summary insights using OpenAI
    st.subheader("Summary Insights")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        if st.button("Generate Insights"):
            openai.api_key = api_key
            summary_prompt = f"""
            You are analyzing SEO keyword ranking data for the target domain '{target_domain}'. 
            Provide a summary of strengths, gaps, and opportunities based on the data. Highlight major optimization areas.
            """
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # Changed from gpt-4o-mini to gpt-4
                    messages=[
                        {"role": "system", "content": "You are a helpful SEO strategist."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                insights = response.choices[0].message.content
                st.write(insights)
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    else:
        st.warning("Provide your OpenAI API key to generate insights.")

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
    st.info("Upload your data and specify the target domain to begin.")
