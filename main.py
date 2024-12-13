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
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(9999)
            df[col] = df[col].clip(1, 9999)
    return df

def process_competitor_data(data_frames, target_domain):
    """Process and combine competitor data side by side"""
    # Initialize an empty DataFrame to store the combined results
    combined_df = None
    
    for idx, df in enumerate(data_frames):
        df = df.copy()
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Identify if this is target domain's data or competitor data
        is_target = idx == 0  # Assuming first file is target domain
        prefix = "Target_" if is_target else f"Competitor_{idx}_"
        
        # Rename columns to prevent conflicts
        df = df.rename(columns={
            'Keyword': 'Keyword',
            'URL': f'{prefix}URL',
            'Rank': f'{prefix}Rank',
            'Search Volume': 'Search Volume'  # Keep one copy of search volume
        })
        
        if combined_df is None:
            combined_df = df
        else:
            # Merge on keyword
            combined_df = pd.merge(combined_df, df[['Keyword', f'{prefix}URL', f'{prefix}Rank']], 
                                 on='Keyword', 
                                 how='outer')
    
    return combined_df

if uploaded_files and target_domain:
    # Process the uploaded files
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    
    # Combine competitor data side by side
    merged_df = process_competitor_data(data_frames, target_domain)
    
    # Display filters
    st.subheader("Filters")
    keyword_filter = st.text_input("Filter by keyword:")
    
    filtered_df = merged_df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].str.contains(keyword_filter, case=False, na=False)]

    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        st.warning("No data to display after applying filters.")
        st.stop()

    # Identify rank columns for styling
    rank_columns = [col for col in filtered_df.columns if 'Rank' in col]
    
    # Clean the rank data
    filtered_df = clean_rank_data(filtered_df, rank_columns)

    # Display the DataFrame with styling
    st.subheader("Keyword Rankings Table")

    try:
        # Create a copy for styling
        display_df = filtered_df.copy()
        
        # Apply styling
        styler = display_df.style.background_gradient(
            subset=rank_columns,
            cmap='RdYlGn_r',
            vmin=1,
            vmax=100
        )
        
        # Display the styled DataFrame
        st.dataframe(styler, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error applying styling: {str(e)}")
        # Fallback to displaying unstyled DataFrame
        st.dataframe(display_df, use_container_width=True)

    # Add summary statistics
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Metric': ['Average Rank', 'Keywords in Top 10', 'Total Keywords'],
        'Target': [
            filtered_df['Target_Rank'].mean(),
            len(filtered_df[filtered_df['Target_Rank'] <= 10]),
            len(filtered_df)
        ]
    })
    
    # Add competitor stats
    for i in range(1, len(data_frames)):
        comp_rank_col = f'Competitor_{i}_Rank'
        if comp_rank_col in filtered_df.columns:
            summary_stats[f'Competitor {i}'] = [
                filtered_df[comp_rank_col].mean(),
                len(filtered_df[filtered_df[comp_rank_col] <= 10]),
                len(filtered_df)
            ]
    
    st.dataframe(summary_stats)

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
                    model="gpt-4",
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
