import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

st.title("📊 Data Filter & Visualization Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Try to identify date column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'open_date' in col.lower()]
    
    st.sidebar.header("🔧 Configuration")
    
    # Select date column
    date_column = st.sidebar.selectbox("Select Date Column", date_cols if date_cols else df.columns)
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.sort_values(date_column)
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True)
    
    st.sidebar.subheader("📅 Date Range Filter")
    min_date = df[date_column].min().date()
    max_date = df[date_column].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        mask = (df[date_column].dt.date >= date_range[0]) & (df[date_column].dt.date <= date_range[1])
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Column filters
    st.sidebar.subheader("🔍 Column Filters")
    
    # Get non-date, non-numeric columns for filtering
    categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
    
    selected_filters = {}
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        unique_vals = filtered_df[col].dropna().unique()
        if len(unique_vals) > 0 and len(unique_vals) < 100:
            selected_vals = st.sidebar.multiselect(
                f"Filter by {col}",
                options=sorted(unique_vals.astype(str)),
                default=None
            )
            if selected_vals:
                selected_filters[col] = selected_vals
    
    # Apply filters
    for col, vals in selected_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(vals)]
    
    st.write(f"**Filtered Records:** {len(filtered_df):,} rows")
    
    # Pivot Table Section
    st.header("📊 Pivot Table & Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get numeric columns for values
        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        value_column = st.selectbox("Select Value Column (Y-axis)", numeric_cols)
        
        # Aggregation function
        agg_func = st.selectbox("Aggregation Function", 
                                ['sum', 'mean', 'median', 'count', 'min', 'max'])
    
    with col2:
        # Group by columns (for multiple lines)
        all_cols = [col for col in filtered_df.columns if col != date_column]
        group_by_col = st.selectbox("Group By (for multiple lines)", 
                                    ['None'] + categorical_cols)
    
    # Create pivot/aggregation
    if group_by_col != 'None':
        # Pivot table with grouping
        pivot_df = filtered_df.groupby([date_column, group_by_col])[value_column].agg(agg_func).reset_index()
        pivot_wide = pivot_df.pivot(index=date_column, columns=group_by_col, values=value_column).reset_index()
        
        # Display pivot table
        with st.expander("📋 View Pivot Table"):
            st.dataframe(pivot_wide, use_container_width=True)
        
        # Plot
        fig = go.Figure()
        
        for col in pivot_wide.columns[1:]:  # Skip date column
            fig.add_trace(go.Scatter(
                x=pivot_wide[date_column],
                y=pivot_wide[col],
                mode='lines+markers',
                name=str(col),
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"{value_column} by {group_by_col} over Time ({agg_func})",
            xaxis_title=date_column,
            yaxis_title=f"{value_column} ({agg_func})",
            hovermode='x unified',
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
    else:
        # Simple aggregation by date
        agg_df = filtered_df.groupby(date_column)[value_column].agg(agg_func).reset_index()
        
        # Display aggregated table
        with st.expander("📋 View Aggregated Table"):
            st.dataframe(agg_df, use_container_width=True)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg_df[date_column],
            y=agg_df[value_column],
            mode='lines+markers',
            name=value_column,
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{value_column} over Time ({agg_func})",
            xaxis_title=date_column,
            yaxis_title=f"{value_column} ({agg_func})",
            hovermode='x unified',
            height=600
        )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.header("📈 Summary Statistics")
    
    if group_by_col != 'None':
        stats_df = pivot_df.groupby(group_by_col)[value_column].agg(['count', 'sum', 'mean', 'median', 'std', 'min', 'max']).round(2)
    else:
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Sum', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                agg_df[value_column].count(),
                agg_df[value_column].sum(),
                agg_df[value_column].mean(),
                agg_df[value_column].median(),
                agg_df[value_column].std(),
                agg_df[value_column].min(),
                agg_df[value_column].max()
            ]
        }).round(2)
    
    st.dataframe(stats_df, use_container_width=True)
    
    # Download options
    st.header("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if group_by_col != 'None':
            csv = pivot_wide.to_csv(index=False)
        else:
            csv = agg_df.to_csv(index=False)
        
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        filtered_csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Raw Data as CSV",
            data=filtered_csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Sample data format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV should contain:
    - A date column (e.g., 'OPEN_DATE', 'date', 'Date')
    - Numeric columns for values to plot
    - Optional categorical columns for grouping/filtering
    
    Example:
    ```
    OPEN_DATE,OPENS_COUNT,Market,Campaign
    2025-09-21,606,US,A
    2025-09-22,416,US,A
    ```
    """)
