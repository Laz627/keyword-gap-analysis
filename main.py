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

def process_competitor_data(data_frames, target_domain):
    """Process and combine competitor data with specific column ordering"""
    # Initialize empty lists to store all keywords and data
    all_keywords = set()
    all_data = []
    
    # Collect all unique keywords
    for df in data_frames:
        all_keywords.update(df['Keyword'].unique())
    
    # Create a base DataFrame with all keywords
    combined_df = pd.DataFrame(list(all_keywords), columns=['Keyword'])
    
    # Process each dataframe
    for df in data_frames:
        df = df.copy()
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Create a mapping of keywords to ranks and URLs
        rank_map = dict(zip(df['Keyword'], df['Rank']))
        url_map = dict(zip(df['Keyword'], df['URL']))
        
        # Add columns to combined_df
        combined_df[f'temp_rank'] = combined_df['Keyword'].map(rank_map)
        combined_df[f'temp_url'] = combined_df['Keyword'].map(url_map)
    
    # Now process URLs to identify target domain and competitors
    target_ranks = []
    target_urls = []
    competitor_ranks = []
    competitor_urls = []
    
    # Process each row
    for idx, row in combined_df.iterrows():
        keyword_data = []
        for df in data_frames:
            mask = df['Keyword'] == row['Keyword']
            if mask.any():
                url = df.loc[mask, 'URL'].iloc[0]
                rank = df.loc[mask, 'Rank'].iloc[0]
                keyword_data.append((url, rank))
            else:
                keyword_data.append((None, None))
        
        # Find target domain data
        target_found = False
        target_rank = 100  # Default to 100 if not found
        target_url = None
        
        for url, rank in keyword_data:
            if url and target_domain.lower() in str(url).lower():
                target_found = True
                target_rank = rank
                target_url = url
                break
        
        target_ranks.append(target_rank)
        target_urls.append(target_url)
        
        # Collect competitor data (excluding target domain)
        comp_ranks = []
        comp_urls = []
        for url, rank in keyword_data:
            if url and (not target_domain.lower() in str(url).lower()):
                comp_ranks.append(rank)
                comp_urls.append(url)
        
        # Pad competitor data if needed
        while len(comp_ranks) < len(data_frames) - 1:
            comp_ranks.append(100)
            comp_urls.append(None)
            
        competitor_ranks.append(comp_ranks)
        competitor_urls.append(comp_urls)
    
    # Create final DataFrame with desired column order
    final_df = pd.DataFrame()
    final_df['Keyword'] = combined_df['Keyword']
    
    # Convert target ranks to pandas Series before processing
    target_ranks_series = pd.Series(target_ranks)
    final_df['Target Rank'] = pd.to_numeric(target_ranks_series, errors='coerce').fillna(100).astype(int)
    
    # Add competitor rank columns and convert to whole numbers
    for i in range(len(data_frames) - 1):
        comp_ranks = [ranks[i] if ranks else 100 for ranks in competitor_ranks]
        comp_ranks_series = pd.Series(comp_ranks)
        final_df[f'Competitor {i+1} Rank'] = pd.to_numeric(comp_ranks_series, errors='coerce').fillna(100).astype(int)
    
    # Add URL columns at the end
    final_df['Target URL'] = target_urls
    for i in range(len(data_frames) - 1):
        final_df[f'Competitor {i+1} URL'] = [urls[i] if urls else None for urls in competitor_urls]
    
    return final_df

def calculate_average_rank(series):
    """Calculate average rank excluding 100 values (which represent N/A)"""
    valid_ranks = series[series != 100]
    if len(valid_ranks) == 0:
        return "N/A"
    return round(valid_ranks.mean(), 2)

def generate_recommendations(row):
    """Generate recommendations based on ranking positions"""
    target_rank = row['Target Rank']
    competitor_ranks = [row[col] for col in row.index if 'Competitor' in col and 'Rank' in col]
    competitor_ranks = [r for r in competitor_ranks if r != 100]  # Exclude non-ranking positions
    
    if target_rank == 100:
        return "Create New"
    elif not competitor_ranks or target_rank < min(competitor_ranks):
        return "Defend"
    elif target_rank <= 20:
        return "Optimize"
    else:
        return "Create New"

if uploaded_files and target_domain:
    # Process the uploaded files
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    
    # Combine competitor data side by side
    merged_df = process_competitor_data(data_frames, target_domain)
    
    # Add recommendations
    merged_df['Recommendation'] = merged_df.apply(generate_recommendations, axis=1)
    
    # Display filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        keyword_filter = st.text_input("Filter by keyword:")
    with col2:
        recommendation_filter = st.selectbox(
            "Filter by recommendation:",
            ["All", "Defend", "Optimize", "Create New"]
        )
    
    filtered_df = merged_df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].str.contains(keyword_filter, case=False, na=False)]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df["Recommendation"] == recommendation_filter]

    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        st.warning("No data to display after applying filters.")
        st.stop()

    # Identify rank columns for styling
    rank_columns = [col for col in filtered_df.columns if 'Rank' in col]
    
    # Display the DataFrame with styling
    st.subheader("Keyword Rankings Table")

    try:
        # Create a copy for styling
        display_df = filtered_df.copy()
        
        # Create a mask for styling (exclude rows where rank is 100)
        style_masks = {}
        for col in rank_columns:
            style_masks[col] = display_df[col] != 100
        
        # Apply styling to rank columns with masks
        styler = display_df.style
        
        # Apply background gradient only to non-100 values
        for col in rank_columns:
            mask = style_masks[col]
            styler = styler.background_gradient(
                cmap='RdYlGn_r',
                vmin=1,
                vmax=99,
                subset=pd.IndexSlice[mask, col]
            )
        
        # Format rank columns as integers
        rank_format = {col: '{:.0f}' for col in rank_columns}
        styler = styler.format(rank_format)
        
        # Display the styled DataFrame
        st.dataframe(styler, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error applying styling: {str(e)}")
        # Fallback to displaying unstyled DataFrame
        st.dataframe(display_df, use_container_width=True)

    # Summary statistics with recommendations
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Metric': [
            'Average Rank (excluding N/A)', 
            'Keywords in Top 10', 
            'Total Keywords', 
            'Not Ranking (N/A)',
            'Defend Keywords',
            'Optimize Keywords',
            'Create New Keywords'
        ],
        'Target': [
            calculate_average_rank(filtered_df['Target Rank']),
            len(filtered_df[filtered_df['Target Rank'] <= 10]),
            len(filtered_df),
            len(filtered_df[filtered_df['Target Rank'] == 100]),
            len(filtered_df[filtered_df['Recommendation'] == 'Defend']),
            len(filtered_df[filtered_df['Recommendation'] == 'Optimize']),
            len(filtered_df[filtered_df['Recommendation'] == 'Create New'])
        ]
    })
    
    st.dataframe(summary_stats)

    # Enhanced OpenAI insights
    st.subheader("Strategic Insights")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        if st.button("Generate Strategic Insights"):
            openai.api_key = api_key
            
            # Prepare detailed data analysis
            defend_keywords = filtered_df[filtered_df['Recommendation'] == 'Defend']
            optimize_keywords = filtered_df[filtered_df['Recommendation'] == 'Optimize']
            create_keywords = filtered_df[filtered_df['Recommendation'] == 'Create New']
            
            top_defend = defend_keywords.nsmallest(5, 'Target Rank')[['Keyword', 'Target Rank']].to_dict('records')
            top_optimize = optimize_keywords.nsmallest(5, 'Target Rank')[['Keyword', 'Target Rank']].to_dict('records')
            
            summary_prompt = f"""
            As an SEO strategist, analyze this keyword ranking data for {target_domain}:

            Current Position:
            - Ranking for {len(filtered_df[filtered_df['Target Rank'] != 100])} out of {len(filtered_df)} total keywords
            - {len(filtered_df[filtered_df['Target Rank'] <= 10])} keywords in top 10 positions
            - Average ranking position: {calculate_average_rank(filtered_df['Target Rank'])}

            Opportunity Breakdown:
            1. DEFEND ({len(defend_keywords)} keywords):
            - Top performing keywords to protect: {top_defend}
            
            2. OPTIMIZE ({len(optimize_keywords)} keywords):
            - Priority optimization targets: {top_optimize}
            
            3. CREATE NEW ({len(create_keywords)} keywords):
            - {len(create_keywords)} unranked or low-ranking opportunities

            Provide a strategic analysis with:
            1. IMMEDIATE ACTIONS (Next 30 days):
            - Specific, high-priority optimization targets
            - Critical content defense strategies
            - Quick-win opportunities
            
            2. COMPETITIVE ADVANTAGES:
            - Areas where {target_domain} outperforms competitors
            - Unique ranking opportunities
            
            3. CONTENT GAPS:
            - Priority topics for new content
            - Specific keyword clusters to target
            
            4. STRATEGIC RECOMMENDATIONS:
            - Concrete steps to improve rankings
            - Resource allocation priorities
            - Risk mitigation strategies
            
            Focus on actionable insights that can drive immediate impact while building long-term SEO strength.
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert SEO strategist providing actionable insights and specific recommendations."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                insights = response.choices[0].message.content
                st.markdown(insights)
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    else:
        st.warning("Provide your OpenAI API key to generate strategic insights.")

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
