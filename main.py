import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openai
import openpyxl
from openpyxl.styles import PatternFill, Font, Color, Border, Side, Alignment
from openpyxl.formatting.rule import ColorScaleRule

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
    search_volumes = {}  # Dictionary to store search volumes
    
    # Collect all unique keywords and their search volumes
    for df in data_frames:
        df.columns = [col.strip() for col in df.columns]
        all_keywords.update(df['Keyword'].unique())
        # Update search volumes, preferring non-zero values
        for keyword, volume in zip(df['Keyword'], df['Search Volume']):
            if keyword not in search_volumes or (volume > 0 and search_volumes[keyword] == 0):
                search_volumes[keyword] = volume
    
    # Create a base DataFrame with all keywords
    combined_df = pd.DataFrame(list(all_keywords), columns=['Keyword'])
    combined_df['Search Volume'] = combined_df['Keyword'].map(search_volumes)
    
    # Now process URLs to identify target domain and competitors
    target_ranks = []
    target_urls = []
    competitor_ranks = []
    competitor_urls = []
    
    # Process each row
    for keyword in combined_df['Keyword']:
        keyword_data = []
        for df in data_frames:
            mask = df['Keyword'] == keyword
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
    final_df['Search Volume'] = combined_df['Search Volume']
    
    # Add target rank (convert numpy array to pandas Series first)
    final_df['Target Rank'] = pd.Series(target_ranks).replace({None: 100}).astype(int)
    
    # Add competitor rank columns
    for i in range(len(data_frames) - 1):
        comp_ranks = [ranks[i] if ranks else 100 for ranks in competitor_ranks]
        final_df[f'Competitor {i+1} Rank'] = pd.Series(comp_ranks).replace({None: 100}).astype(int)
    
    # Generate recommendations after all rank data is available
    final_df['Recommendation'] = final_df.apply(generate_recommendations, axis=1)
    
    # Add URL columns
    final_df['Target URL'] = target_urls
    for i in range(len(data_frames) - 1):
        final_df[f'Competitor {i+1} URL'] = [urls[i] if urls else None for urls in competitor_urls]
    
    # Reorder columns to put recommendation second
    rank_cols = [col for col in final_df.columns if 'Rank' in col]
    url_cols = [col for col in final_df.columns if 'URL' in col]
    final_df = final_df[['Keyword', 'Recommendation', 'Search Volume'] + rank_cols + url_cols]
    
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
    elif competitor_ranks:  # If there are competing rankings
        best_competitor_rank = min(competitor_ranks)
        if target_rank > best_competitor_rank:  # If any competitor ranks better
            if target_rank <= 20:
                return "Optimize"
            else:
                return "Create New"
        else:
            return "Defend"
    else:  # If no competitors are ranking
        return "Defend"

if uploaded_files and target_domain:
    # Process the uploaded files
    data_frames = [pd.read_csv(file) for file in uploaded_files]
    
    # Combine competitor data side by side
    merged_df = process_competitor_data(data_frames, target_domain)
    
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

# Display the DataFrame with styling
    st.subheader("Keyword Rankings Table")

    try:
        display_df = filtered_df.copy()  # Create copy outside the try block
        
        # Create styler object
        styler = display_df.style
        
        # Define custom CSS properties
        styler.set_properties(**{
            'background-color': '#f5f5f5',
            'color': '#333333',
            'border': '1px solid #e0e0e0',
            'padding': '8px',
            'text-align': 'left'
        })
        
        # Apply background gradient only to rank columns (excluding 100s)
        rank_columns = [col for col in display_df.columns if 'Rank' in col]
        for col in rank_columns:
            mask = display_df[col] != 100
            styler.background_gradient(
                cmap='RdYlGn_r',
                vmin=1,
                vmax=30,
                subset=pd.IndexSlice[mask, col],
                text_color_threshold=0.7
            )
        
        # Format columns
        format_dict = {
            'Search Volume': '{:,.0f}',  # Add thousands separator
            **{col: '{:.0f}' for col in rank_columns}  # Format rank columns as integers
        }
        styler.format(format_dict)
        
        # Add header styling
        styler.set_table_styles([
            {'selector': 'thead th', 
             'props': [('background-color', '#2c3e50'), 
                      ('color', 'white'),
                      ('font-weight', 'bold'),
                      ('padding', '12px')]},
            {'selector': 'tbody tr:nth-of-type(even)',
             'props': [('background-color', '#f8f9fa')]},
            {'selector': 'td', 
             'props': [('padding', '8px'),
                      ('border', '1px solid #dee2e6')]}
        ])
        
        # Display the styled DataFrame
        st.dataframe(styler, use_container_width=True, height=600)
    
    except Exception as e:
        st.error(f"Error applying styling: {str(e)}")
        # Display unstyled DataFrame
        st.dataframe(filtered_df, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Metric': [
            'Average Rank (excluding N/A)', 
            'Keywords in Top 10',
            'Keywords in Top 3',
            'Not Ranking (N/A)',
            'Defend Keywords',
            'Optimize Keywords',
            'Create New Keywords'
        ],
        'Target': [
            calculate_average_rank(filtered_df['Target Rank']),
            len(filtered_df[filtered_df['Target Rank'] <= 10]),
            len(filtered_df[filtered_df['Target Rank'] <= 3]),
            len(filtered_df[filtered_df['Target Rank'] == 100]),
            len(filtered_df[filtered_df['Recommendation'] == 'Defend']),
            len(filtered_df[filtered_df['Recommendation'] == 'Optimize']),
            len(filtered_df[filtered_df['Recommendation'] == 'Create New'])
        ]
    })
    
    # Add competitor metrics
    for i in range(1, len(data_frames)):
        comp_col = f'Competitor {i} Rank'
        if comp_col in filtered_df.columns:
            summary_stats[f'Competitor {i}'] = [
                calculate_average_rank(filtered_df[comp_col]),
                len(filtered_df[filtered_df[comp_col] <= 10]),
                len(filtered_df[filtered_df[comp_col] <= 3]),
                len(filtered_df[filtered_df[comp_col] == 100]),
                '-',
                '-',
                '-'
            ]
    
    st.dataframe(summary_stats, use_container_width=True)

# Enhanced OpenAI insights
    st.subheader("Strategic Insights")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        if st.button("Generate Strategic Insights"):
            openai.api_key = api_key
            
            # Get top 20 keywords by search volume for target domain and competitors
            def get_top_keywords(df, rank_column, max_rank=100):
                return df[df[rank_column] != 100].sort_values(
                    by='Search Volume', ascending=False
                ).head(20)[['Keyword', 'Search Volume', rank_column]].to_dict('records')
            
            # Target domain top keywords
            target_top_keywords = get_top_keywords(filtered_df, 'Target Rank')
            
            # Competitor top keywords
            competitor_keywords = {}
            for i in range(1, len(data_frames)):
                comp_col = f'Competitor {i} Rank'
                if comp_col in filtered_df.columns:
                    competitor_keywords[f'Competitor {i}'] = get_top_keywords(filtered_df, comp_col)
            
            # Format keyword data for the prompt
            def format_keyword_list(keywords):
                return "\n".join([
                    f"- {kw['Keyword']} "
                    f"(Search Volume: {kw['Search Volume']:,}, "
                    f"Rank: {list(kw.values())[2]})"
                    for kw in keywords
                ])
            
            summary_prompt = f"""
            Analyze the top-performing keywords (by search volume) for {target_domain} and competitors:

            1. {target_domain} TOP 20 RANKINGS:
            {format_keyword_list(target_top_keywords)}

            2. COMPETITOR RANKINGS:
            """
            
            # Add competitor sections to prompt
            for comp_name, comp_keywords in competitor_keywords.items():
                summary_prompt += f"\n\n{comp_name} TOP 20 RANKINGS:\n"
                summary_prompt += format_keyword_list(comp_keywords)
            
            summary_prompt += """
            
            Please provide:
            1. KEY INSIGHTS FOR EACH DOMAIN:
            - Identify each domain's strongest keyword themes
            - Note any unique high-value keywords
            - Highlight competitive advantages in specific areas
            
            2. COMPETITIVE GAP ANALYSIS:
            - Identify valuable keywords where competitors rank but target domain doesn't
            - Highlight opportunities where target domain can improve existing rankings
            - Note any patterns in competitor keyword strategies
            
            Format the output with clear headers and bullet points.
            Focus on search volume and ranking positions in your analysis.
            Prioritize actionable insights based on search volume and ranking patterns.
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an SEO analyst providing concise, data-driven keyword analysis with a focus on competitive advantages and opportunities."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.6,
                    max_tokens=2000
                )
                insights = response.choices[0].message.content
                
                # Format and display insights with Streamlit
                st.markdown("## Keyword Performance Analysis")
                st.markdown(insights)
                
                # Add download button for insights
                st.download_button(
                    label="Download Insights",
                    data=insights,
                    file_name="keyword_analysis.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    else:
        st.warning("Provide your OpenAI API key to generate strategic insights.")

# Export to Excel with styling
    st.subheader("Export Data")
    if st.button("Download Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Write the DataFrame
            filtered_df.to_excel(writer, index=False, sheet_name="Rankings")
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets["Rankings"]
            
            # Define styles
            from openpyxl.styles import PatternFill, Font, Color, Border, Side, Alignment
            from openpyxl.formatting.rule import ColorScaleRule
            
            # Header style
            header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            
            # Cell border
            thin_border = Border(
                left=Side(style='thin', color='DEE2E6'),
                right=Side(style='thin', color='DEE2E6'),
                top=Side(style='thin', color='DEE2E6'),
                bottom=Side(style='thin', color='DEE2E6')
            )
            
            # Apply header styling
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='left')
            
            # Get column letters for rank columns
            rank_columns = [col for col in filtered_df.columns if 'Rank' in col]
            rank_column_letters = []
            for col in rank_columns:
                col_idx = filtered_df.columns.get_loc(col) + 1  # +1 because Excel is 1-based
                rank_column_letters.append(openpyxl.utils.get_column_letter(col_idx))
            
            # Apply conditional formatting to rank columns
            for col_letter in rank_column_letters:
                # Color scale for ranks (excluding 100)
                worksheet.conditional_formatting.add(
                    f'{col_letter}2:{col_letter}{len(filtered_df) + 1}',
                    ColorScaleRule(
                        start_type='num',
                        start_value=1,
                        start_color='63BE7B',  # Green
                        mid_type='num',
                        mid_value=15,
                        mid_color='FFEB84',  # Yellow
                        end_type='num',
                        end_value=30,
                        end_color='F8696B'  # Red
                    )
                )
            
            # Format number columns
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='left')
                    
                    # Format Search Volume with thousands separator
                    if openpyxl.utils.get_column_letter(cell.column) == openpyxl.utils.get_column_letter(filtered_df.columns.get_loc('Search Volume') + 1):
                        cell.number_format = '#,##0'
                    
                    # Format rank columns as integers
                    if openpyxl.utils.get_column_letter(cell.column) in rank_column_letters:
                        cell.number_format = '0'
            
            # Set column widths
            for idx, col in enumerate(filtered_df.columns):
                column_letter = openpyxl.utils.get_column_letter(idx + 1)
                if 'URL' in col:
                    worksheet.column_dimensions[column_letter].width = 50
                elif 'Keyword' in col:
                    worksheet.column_dimensions[column_letter].width = 30
                else:
                    worksheet.column_dimensions[column_letter].width = 15
            
            # Alternate row colors
            for row in range(2, len(filtered_df) + 2):  # Start from row 2 (after header)
                if row % 2 == 0:  # Even rows
                    for cell in worksheet[row]:
                        cell.fill = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
            
            # Freeze header row
            worksheet.freeze_panes = 'A2'
        
        buffer.seek(0)
        st.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="keyword_rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Upload your data and specify the target domain to begin.")
