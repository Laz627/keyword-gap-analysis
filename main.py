import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openpyxl
from openpyxl.styles import PatternFill, Font, Color, Border, Side, Alignment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.formatting.rule import CellIsRule

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
    
    # Add target rank
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
    return int(round(valid_ranks.mean(), 0))  # Round to whole number
    
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

def get_top_keywords_by_category(df, domain_type='target'):
    """Get top 20 keywords by search volume for each category"""
    if domain_type == 'target':
        defend_kw = df[df['Recommendation'] == 'Defend'].sort_values(
            by='Search Volume', ascending=False).head(20)[['Keyword', 'Search Volume', 'Target Rank']]
        optimize_kw = df[df['Recommendation'] == 'Optimize'].sort_values(
            by='Search Volume', ascending=False).head(20)[['Keyword', 'Search Volume', 'Target Rank']]
        create_kw = df[df['Recommendation'] == 'Create New'].sort_values(
            by='Search Volume', ascending=False).head(20)[['Keyword', 'Search Volume', 'Target Rank']]
        return {
            'Defend Keywords': defend_kw,
            'Optimize Keywords': optimize_kw,
            'Create New Keywords': create_kw
        }
    else:
        # For competitors, just get top 20 overall
        return df[df[f'{domain_type} Rank'] != 100].sort_values(
            by='Search Volume', ascending=False).head(20)[['Keyword', 'Search Volume', f'{domain_type} Rank']]

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
        display_df = filtered_df.copy()
        
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
    
# Style the summary statistics
    summary_styler = summary_stats.style.set_properties(**{
        'text-align': 'left',
        'padding': '12px'
    }).format({
        'Target': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x,
        **{f'Competitor {i}': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x 
           for i in range(1, len(data_frames))}
    }).set_table_styles([
        {'selector': 'thead th', 
         'props': [('background-color', '#2c3e50'), 
                  ('color', 'white'),
                  ('font-weight', 'bold'),
                  ('padding', '12px'),
                  ('text-align', 'left')]},
        {'selector': 'tbody tr:nth-of-type(even)',
         'props': [('background-color', '#f8f9fa')]}
    ])
    
    st.dataframe(summary_styler, use_container_width=True)

    # Display top keywords
    st.subheader("Top Keywords by Search Volume")
    
    # Get top keywords for target domain
    target_top_kw = get_top_keywords_by_category(filtered_df, 'target')
    
    # Display target domain top keywords by category
    for category, df in target_top_kw.items():
        st.write(f"\n{category}:")
        st.dataframe(df, use_container_width=True)
    
    # Get and display competitor top keywords
    for i in range(1, len(data_frames)):
        comp_col = f'Competitor {i}'
        if f'{comp_col} Rank' in filtered_df.columns:
            st.write(f"\n{comp_col} Top Keywords:")
            comp_top_kw = get_top_keywords_by_category(filtered_df, comp_col)
            st.dataframe(comp_top_kw, use_container_width=True)

    # Export to Excel
    st.subheader("Export Data")
    if st.button("Download Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Write main rankings sheet
            filtered_df.to_excel(writer, index=False, sheet_name="Rankings")
            
            # Write summary statistics
            summary_stats.to_excel(writer, index=False, sheet_name="Summary")
            
            # Get top keywords
            target_top_kw = get_top_keywords_by_category(filtered_df, 'target')
            competitor_top_kw = {}
            for i in range(1, len(data_frames)):
                comp_col = f'Competitor {i}'
                if f'{comp_col} Rank' in filtered_df.columns:
                    competitor_top_kw[comp_col] = get_top_keywords_by_category(filtered_df, comp_col)
            
            # Prepare top keywords data for single write
            top_kw_rows = []
            current_row = 0
            
            # Add target domain keywords
            for category, df in target_top_kw.items():
                # Add category header
                top_kw_rows.append(pd.DataFrame({'Category': [category]}))
                # Add data
                top_kw_rows.append(df)
                # Add blank row
                top_kw_rows.append(pd.DataFrame({'Category': ['']}))
            
            # Add competitor keywords
            for comp_name, comp_df in competitor_top_kw.items():
                # Add competitor header
                top_kw_rows.append(pd.DataFrame({'Category': [f"{comp_name} Top Keywords"]}))
                # Add data
                top_kw_rows.append(comp_df)
                # Add blank row
                top_kw_rows.append(pd.DataFrame({'Category': ['']}))
            
            # Combine all top keywords data
            top_kw_df = pd.concat(top_kw_rows, ignore_index=True)
            
            # Write top keywords
            top_kw_df.to_excel(writer, sheet_name="Top Keywords", index=False)
            
            # Get workbook and worksheets
            workbook = writer.book
            rankings_ws = workbook["Rankings"]
            summary_ws = workbook["Summary"]
            top_kw_ws = workbook["Top Keywords"]
            
            # Define styles
            header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            thin_border = Border(
                left=Side(style='thin', color='DEE2E6'),
                right=Side(style='thin', color='DEE2E6'),
                top=Side(style='thin', color='DEE2E6'),
                bottom=Side(style='thin', color='DEE2E6')
            )
            
            # Function to apply basic styling to a worksheet
            def apply_basic_styling(ws):
                # Style headers
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='left')
                
                # Style all cells
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.border = thin_border
                        cell.alignment = Alignment(horizontal='left')
                
                # Freeze top row
                ws.freeze_panes = 'A2'
            
            # Apply basic styling to all sheets
            for ws in [rankings_ws, summary_ws, top_kw_ws]:
                apply_basic_styling(ws)
            
# Rankings sheet specific styling
            rank_columns = [col for col in filtered_df.columns if 'Rank' in col]
            for col in rank_columns:
                col_idx = filtered_df.columns.get_loc(col) + 1
                col_letter = openpyxl.utils.get_column_letter(col_idx)
                
                # First add light gray for rank 100 (this should be applied first)
                rankings_ws.conditional_formatting.add(
                    f'{col_letter}2:{col_letter}{len(filtered_df) + 1}',
                    CellIsRule(
                        operator='equal',
                        formula=['100'],
                        stopIfTrue=True,
                        fill=PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
                    )
                )
                
                # Then add color scale only for ranks 1-99
                rankings_ws.conditional_formatting.add(
                    f'{col_letter}2:{col_letter}{len(filtered_df) + 1}',
                    ColorScaleRule(
                        start_type='percentile',
                        start_value=0,
                        start_color='63BE7B',  # Green
                        mid_type='percentile',
                        mid_value=50,
                        mid_color='FFEB84',  # Yellow
                        end_type='percentile',
                        end_value=100,
                        end_color='F8696B',  # Red
                    )
                )

                # Add number formatting
                for cell in rankings_ws[col_letter][1:]:
                    cell.number_format = '0'
                    if cell.value == 100:
                        cell.fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
            
            # Format numbers in all sheets
            for ws in [rankings_ws, summary_ws, top_kw_ws]:
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        column_header = ws.cell(1, cell.column).value
                        if 'Search Volume' in str(column_header):
                            cell.number_format = '#,##0'
                        elif 'Rank' in str(column_header):
                            cell.number_format = '0'
            
            # Auto-fit columns and wrap URLs
            for ws in [rankings_ws, summary_ws, top_kw_ws]:
                for column in ws.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    is_url_column = 'URL' in str(column[0].value)
                    
                    if is_url_column:
                        # Set fixed width for URL columns and wrap text
                        ws.column_dimensions[column[0].column_letter].width = 50
                        for cell in column:
                            cell.alignment = Alignment(wrap_text=True)
                    else:
                        # Calculate max length for other columns
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = (max_length + 2)
                        ws.column_dimensions[column[0].column_letter].width = min(adjusted_width, 40)
            
            # Add alternating row colors
            for ws in [rankings_ws, summary_ws, top_kw_ws]:
                for row in range(2, ws.max_row + 1):
                    if row % 2 == 0:
                        for cell in ws[row]:
                            cell.fill = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
        
        buffer.seek(0)
        st.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="keyword_rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Upload your data and specify the target domain to begin.")
