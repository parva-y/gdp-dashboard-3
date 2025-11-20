# streamlit_hdfc_lift_app.py
# Streamlit app to analyse lift from branding spends using search volumes + market indicators + competitors
# Default file paths (will be transformed to URLs by platform):
# - Searches Excel: /mnt/data/Searches Last 90 Days.xlsx
# - Spends CSV: /mnt/data/Spends.csv

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.tools.tools import add_constant
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Optional advanced methods (ridge, elastic net, PCA)
try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title='HDFC Sky Lift Analyzer', layout='wide')

st.title('HDFC Sky — Branding Lift & Attribution Analyzer')
st.markdown("""
This app:
- Ingests daily search volumes, campaign daily spends, and market indicators.
- Auto-detects campaign period (spend > 0; assumes continuous campaign).
- Runs multiple lift analyses: pre-vs-campaign tests, ITS regression, DiD-like controls, OLS with spends + controls, correlations.
- Weekly toggle to aggregate to weekly frequency.
- Includes an "Interpretation" tab summarising results.

**Defaults:** the app will attempt to load files from the provided paths if you don't upload new ones.
""")

# ----------------------------
# Helper functions
# ----------------------------

def parse_dates(df, date_col='Date'):
    # try dd/mm/yyyy then fallback
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', dayfirst=True)
    except Exception:
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def aggregate_weekly(df, date_col='Date'):
    df = df.copy()
    df['WeekStart'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
    agg = df.groupby('WeekStart').sum(numeric_only=True).reset_index().rename(columns={'WeekStart': 'Date'})
    return agg


def detect_campaign(spend_series):
    # campaign where spend > 0 (continuous period)
    mask = spend_series > 0
    if mask.sum() == 0:
        return None, None
    first = mask.idxmax()
    last = len(mask) - 1 - mask[::-1].idxmax()
    return first, last


def summarize_period(df, date_col, col, start_idx, end_idx):
    pre = df.loc[:start_idx-1, col]
    camp = df.loc[start_idx:end_idx, col]
    post = df.loc[end_idx+1:, col]
    return pre, camp, post


def bootstrap_diff(pre, camp, n_boot=2000):
    # bootstrap mean diff camp - pre
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_boot):
        s_pre = rng.choice(pre, size=len(pre), replace=True)
        s_camp = rng.choice(camp, size=len(camp), replace=True)
        diffs.append(np.nanmean(s_camp) - np.nanmean(s_pre))
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.mean(diffs)


def ols_with_controls(y, X):
    # Convert to pandas if needed
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Ensure all inputs are numeric and handle any remaining issues
    y_clean = pd.to_numeric(y, errors='coerce')
    X_clean = X.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN
    valid_mask = y_clean.notna() & X_clean.notna().all(axis=1)
    y_clean = y_clean[valid_mask]
    X_clean = X_clean[valid_mask]
    
    Xc = add_constant(X_clean)
    model = sm.OLS(y_clean.astype(float), Xc.astype(float)).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 7}
    )
    return model


# helper to coerce numeric-like strings to numbers
def safe_to_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r'[,\s%₹]', '', regex=True).str.strip(),
        errors='coerce'
    )

# ----------------------------
# Load data
# ----------------------------

st.sidebar.header('Data inputs')
searches_file = st.sidebar.file_uploader('Upload searches file (Excel with Date + columns)', type=['xlsx','xls','csv'])
spends_file = st.sidebar.file_uploader('Upload spends file (CSV with Date, Spend)', type=['csv','xlsx','xls'])

default_search_path = '/mnt/data/Searches Last 90 Days.xlsx'
default_spend_path = '/mnt/data/Spends.csv'

use_defaults = False
if searches_file is None and spends_file is None:
    st.sidebar.info('No uploads detected — app will attempt to use default files from the environment.')
    use_defaults = True

if searches_file is not None:
    if searches_file.name.lower().endswith('.csv'):
        searches = pd.read_csv(searches_file)
    else:
        searches = pd.read_excel(searches_file)
elif use_defaults:
    try:
        searches = pd.read_excel(default_search_path)
        st.sidebar.success(f'Loaded searches from {default_search_path}')
    except Exception as e:
        st.sidebar.error('Could not load default searches file. Please upload one.')
        st.stop()

if spends_file is not None:
    if spends_file.name.lower().endswith('.csv'):
        spends = pd.read_csv(spends_file)
    else:
        spends = pd.read_excel(spends_file)
elif use_defaults:
    try:
        spends = pd.read_csv(default_spend_path)
        st.sidebar.success(f'Loaded spends from {default_spend_path}')
    except Exception as e:
        st.sidebar.error('Could not load default spends file. Please upload one.')
        st.stop()

# parse dates and robust cleaning (handles non-numeric strings, commas, symbols)

# Ensure Date columns are parsed first (try dayfirst)
searches = parse_dates(searches, date_col='Date')
spends = parse_dates(spends, date_col='Date')

# Identify non-date columns in searches and coerce to numeric where appropriate
non_date_cols = [c for c in searches.columns if c != 'Date']
for col in non_date_cols:
    before_nonnull = searches[col].notna().sum()
    searches[col] = safe_to_numeric(searches[col])
    after_nonnull = searches[col].notna().sum()
    if after_nonnull < before_nonnull:
        st.warning(f"Column {col}: {before_nonnull-after_nonnull} non-numeric entries coerced to NaN.")

# Ensure spends has 'Spend' column and coerce to numeric
if 'Spend' not in spends.columns:
    possible = [c for c in spends.columns if 'spend' in c.lower() or 'amount' in c.lower()]
    if len(possible) > 0:
        spends.rename(columns={possible[0]:'Spend'}, inplace=True)
    else:
        st.error("Spends file doesn't contain a 'Spend' column. Rename the column to 'Spend' and re-upload.")
        st.stop()

spends['Spend'] = safe_to_numeric(spends['Spend'])
num_spend_na = spends['Spend'].isna().sum()
if num_spend_na > 0:
    st.warning(f'{num_spend_na} Spend values could not be parsed to numeric and were set to NaN. Filling with 0.')
    spends['Spend'] = spends['Spend'].fillna(0)

# Merge cleaned datasets (both should have datetime Date column now)
DF = pd.merge(searches, spends[['Date', 'Spend']], on='Date', how='left')
DF['Spend'] = DF['Spend'].fillna(0)

# Ensure ALL non-Date columns are numeric
for c in DF.columns:
    if c == 'Date':
        continue
    DF[c] = pd.to_numeric(DF[c], errors='coerce')

# Fill any remaining NaNs with 0
numeric_cols = [c for c in DF.columns if c != 'Date']
DF[numeric_cols] = DF[numeric_cols].fillna(0)

# Final dtype check
problem_cols = [c for c in DF.columns if c != 'Date' and not np.issubdtype(DF[c].dtype, np.number)]
if len(problem_cols) > 0:
    st.error('The following columns are still non-numeric and will break numeric ops: ' + ', '.join(problem_cols))
    st.dataframe(DF[problem_cols].head())
    st.stop()

st.success('Data cleaning complete — all numeric columns verified.')

# ----------------------------
# Settings
# ----------------------------

st.sidebar.header('Settings')
weekly = st.sidebar.checkbox('Aggregate to weekly', value=False)
if weekly:
    DF = aggregate_weekly(DF, date_col='Date')

# Determine campaign period (prefer explicit 27 Sep 2025 start if present, else autodetect where Spend>0)
try:
    user_campaign_start = pd.to_datetime('2025-09-27', dayfirst=True)
except Exception:
    user_campaign_start = pd.to_datetime('2025-09-27')

idxs_after = DF.index[DF['Date'] >= user_campaign_start].tolist()
if len(idxs_after) > 0:
    start_idx = idxs_after[0]
    spend_after = DF.loc[start_idx:, 'Spend']
    if (spend_after > 0).any():
        end_idx = int(spend_after[spend_after > 0].index.max())
    else:
        end_idx = len(DF) - 1
    st.sidebar.success(f'Using user campaign start: {DF.loc[start_idx, "Date"].date()} -> {DF.loc[end_idx, "Date"].date()}')
else:
    start_idx, end_idx = detect_campaign(DF['Spend'])
    if start_idx is None:
        st.warning('No campaign spend detected (all spends are 0). App will still show correlations and diagnostics.')
    else:
        camp_start_date_auto = DF.loc[start_idx, 'Date']
        camp_end_date_auto = DF.loc[end_idx, 'Date']
        st.sidebar.success(f"Auto-detected campaign: {camp_start_date_auto.date()} → {camp_end_date_auto.date()}")

# For date-based logic in PCA etc.
if start_idx is not None:
    camp_start_date = DF.loc[start_idx, 'Date']
    camp_end_date = DF.loc[end_idx, 'Date']
else:
    camp_start_date = None
    camp_end_date = None

# columns
st.sidebar.markdown('**Columns detected**')
cols = [c for c in DF.columns if c not in ['Date','Spend']]
st.sidebar.text(', '.join(cols[:10]) + (', ...' if len(cols)>10 else ''))

# default outcome is first column (assumed HDFC Sky)
outcome = st.sidebar.selectbox('Primary outcome (search column)', options=cols, index=0)

# indicators and competitors — by default all others
controls = st.sidebar.multiselect(
    'Controls (competitors + market indicators)',
    options=[c for c in cols if c!=outcome],
    default=[c for c in cols if c!=outcome]
)

alpha = st.sidebar.number_input(
    'Significance alpha',
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.01
)

# ----------------------------
# Base ITS design & model (used by several tabs)
# ----------------------------

X_base = pd.DataFrame()
X_base['time_idx'] = np.arange(len(DF))
if start_idx is not None:
    X_base['campaign'] = 0
    X_base.loc[start_idx:end_idx, 'campaign'] = 1
else:
    X_base['campaign'] = 0

# add day-of-week
try:
    X_base['dow'] = DF['Date'].dt.dayofweek
    X_base = pd.get_dummies(X_base, columns=['dow'], drop_first=True)
except Exception:
    pass

# add controls
for c in controls:
    X_base[c] = DF[c].values

y = DF[outcome]

model = ols_with_controls(y, X_base)

# fitted for plotting
fitted_full = pd.Series(index=DF.index, dtype=float)
fitted_full.loc[model.fittedvalues.index] = model.fittedvalues.values
DF['fitted'] = fitted_full

# ----------------------------
# Tabs
# ----------------------------

tabs = st.tabs([
    'Overview & Plot',
    'Pre vs Campaign',
    'ITS / Regression',
    'DiD & Controls',
    'Correlations',
    'Lift Summary',
    'Ridge / Elastic Net',
    'Lagged ITS',
    'PCA Controls',
    'Weekly DiD',
    'Interpretation'
])

# Overview
with tabs[0]:
    st.header('Time series overview')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF['Date'], y=DF[outcome], name=outcome))
    fig.add_trace(go.Bar(x=DF['Date'], y=DF['Spend'], name='Spend', yaxis='y2', opacity=0.4))
    fig.update_layout(
        yaxis=dict(title='Search Volume'),
        yaxis2=dict(title='Spend (INR)', overlaying='y', side='right'),
        legend=dict(orientation='h')
    )
    if start_idx is not None:
        fig.add_vrect(
            x0=DF.loc[start_idx,'Date'],
            x1=DF.loc[end_idx,'Date'],
            fillcolor='green',
            opacity=0.1,
            layer='below',
            line_width=0
        )
    st.plotly_chart(fig, use_container_width=True)

# Pre vs Campaign
with tabs[1]:
    st.header('Pre vs Campaign — simple tests')
    if start_idx is None:
        st.info('No campaign — skipping pre vs campaign tests')
    else:
        pre, camp, post = summarize_period(DF, 'Date', outcome, start_idx, end_idx)
        st.subheader('Descriptive stats')
        st.write(pd.DataFrame({
            'period': ['pre','campaign'],
            'n': [len(pre), len(camp)],
            'mean': [np.mean(pre), np.mean(camp)],
            'std': [np.std(pre, ddof=1), np.std(camp, ddof=1)]
        }))

        # t-test
        tstat, pval, dfree = ttest_ind(camp, pre, usevar='unequal')
        st.write(f'T-test (campaign vs pre): t = {tstat:.3f}, p = {pval:.4f}')

        # bootstrap
        lower, upper, mean_diff = bootstrap_diff(np.array(pre), np.array(camp))
        st.write(f'Bootstrap mean diff (campaign - pre): {mean_diff:.2f}; 95% CI = [{lower:.2f}, {upper:.2f}]')

        st.plotly_chart(
            px.box(
                pd.DataFrame({'pre':pre, 'campaign':camp}).melt(var_name='period', value_name='value'),
                x='period', y='value', points='all'
            ),
            use_container_width=True
        )

# ITS / Regression
with tabs[2]:
    st.header('Interrupted Time Series / OLS with controls')
    st.subheader('Model summary (HAC SE)')
    st.text(model.summary())

    # coefficient for campaign
    if 'campaign' in model.params.index:
        coef = model.params['campaign']
        se = model.bse['campaign']
        p = model.pvalues['campaign']
        st.write(f'Campaign coefficient = {coef:.2f} (se {se:.2f}), p = {p:.4f}')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=DF['Date'], y=DF[outcome], name='Actual'))
    fig2.add_trace(go.Scatter(x=DF['Date'], y=DF['fitted'], name='Fitted'))
    if start_idx is not None:
        fig2.add_vrect(
            x0=DF.loc[start_idx,'Date'],
            x1=DF.loc[end_idx,'Date'],
            fillcolor='green',
            opacity=0.08,
            layer='below',
            line_width=0
        )
    st.plotly_chart(fig2, use_container_width=True)

# DiD & Controls
with tabs[3]:
    st.header('Difference-in-Differences style check & Controls')
    # Create a synthetic control by averaging controls
    if len(controls) > 0:
        DF['controls_mean'] = DF[controls].mean(axis=1)
        if start_idx is not None:
            pre_mean_outcome = DF.loc[:start_idx-1, outcome].mean()
            camp_mean_outcome = DF.loc[start_idx:end_idx, outcome].mean()
            pre_mean_ctrl = DF.loc[:start_idx-1, 'controls_mean'].mean()
            camp_mean_ctrl = DF.loc[start_idx:end_idx, 'controls_mean'].mean()
            st.write('Raw pre vs campaign means (outcome vs controls_mean)')
            st.write(pd.DataFrame({
                'metric': ['pre_outcome','camp_outcome','pre_ctrl','camp_ctrl'],
                'value':[pre_mean_outcome,camp_mean_outcome,pre_mean_ctrl,camp_mean_ctrl]
            }))

        st.subheader('Regression using competitor controls only')
        Xc = add_constant(DF[['controls_mean']])
        mod_ctrl = sm.OLS(DF[outcome].astype(float), Xc.astype(float)).fit()
        st.text(mod_ctrl.summary())
    else:
        st.info('No controls selected')

# Correlations
with tabs[4]:
    st.header('Correlations')
    st.subheader('Correlation matrix (outcome + controls)')
    corr_cols = [outcome] + controls
    if len(corr_cols) > 1:
        corr_df = DF[corr_cols].corr()
        st.dataframe(corr_df)
        st.plotly_chart(px.imshow(corr_df, text_auto=True, aspect='auto'), use_container_width=True)
    else:
        st.info('Select at least one control to see correlations.')

# Lift Summary
with tabs[5]:
    st.header('Lift summary — absolute, relative, DiD & ITS')

    if start_idx is None:
        st.info('No campaign detected — cannot compute lift statistics.')
    else:
        # Simple pre vs campaign split
        pre, camp, post = summarize_period(DF, 'Date', outcome, start_idx, end_idx)
        pre_mean = pre.mean()
        camp_mean = camp.mean()

        # Absolute and relative lift (simple pre vs campaign)
        abs_lift_simple = camp_mean - pre_mean
        rel_lift_simple = (camp_mean / pre_mean - 1) * 100 if pre_mean != 0 else np.nan

        # Ensure synthetic control exists if controls are provided
        did_abs = np.nan
        did_rel = np.nan
        if len(controls) > 0:
            if 'controls_mean' not in DF.columns:
                DF['controls_mean'] = DF[controls].mean(axis=1)

            pre_ctrl = DF.loc[:start_idx-1, 'controls_mean'].mean()
            camp_ctrl = DF.loc[start_idx:end_idx, 'controls_mean'].mean()

            did_abs = (camp_mean - pre_mean) - (camp_ctrl - pre_ctrl)
            did_rel = (did_abs / pre_mean * 100) if pre_mean != 0 else np.nan

        # ITS-based lift (campaign coefficient from OLS with controls)
        its_abs = np.nan
        its_rel = np.nan
        try:
            if 'campaign' in model.params.index:
                its_abs = model.params['campaign']
                its_rel = (its_abs / pre_mean * 100) if pre_mean != 0 else np.nan
        except Exception:
            pass

        # Build dataframes for plotting
        abs_rows = [
            {'Metric': 'Absolute (Δ mean)', 'Lift': abs_lift_simple},
            {'Metric': 'DiD (Δ vs controls)', 'Lift': did_abs},
            {'Metric': 'ITS (campaign coef)', 'Lift': its_abs}
        ]
        rel_rows = [
            {'Metric': 'Relative (%)', 'Lift': rel_lift_simple},
            {'Metric': 'DiD (% vs pre)', 'Lift': did_rel},
            {'Metric': 'ITS (% vs pre)', 'Lift': its_rel}
        ]

        abs_df = pd.DataFrame(abs_rows).dropna()
        rel_df = pd.DataFrame(rel_rows).dropna()

        st.subheader('Absolute lift (incremental searches per period)')
        if not abs_df.empty:
            fig_abs = px.bar(abs_df, x='Metric', y='Lift', text='Lift')
            fig_abs.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_abs.update_layout(
                yaxis_title='Incremental searches',
                xaxis_title='',
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_abs, use_container_width=True)
        else:
            st.info('Not enough information to compute absolute lift metrics.')

        st.subheader('Relative lift (% vs pre-campaign mean)')
        if not rel_df.empty:
            fig_rel = px.bar(rel_df, x='Metric', y='Lift', text='Lift')
            fig_rel.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_rel.update_layout(
                yaxis_title='Lift (%)',
                xaxis_title='',
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.info('Not enough information to compute relative lift metrics.')

        st.markdown("""
        **Definitions:**
        - **Absolute (Δ mean)**: Difference in average searches per day/period (campaign − pre).
        - **Relative (%)**: Percentage change in average searches: (campaign / pre − 1) × 100.
        - **DiD (Δ vs controls)**: (Δ outcome) − (Δ synthetic control), using mean of selected controls.
        - **ITS (campaign coef)**: Incremental level shift estimated by the ITS regression with controls.
        """)

# Ridge / Elastic Net
with tabs[6]:
    st.header('Ridge / Elastic Net (regularised regression)')
    if not SKLEARN_AVAILABLE:
        st.error('scikit-learn is not installed in this environment. Install it to use Ridge/Elastic Net.')
    else:
        st.markdown("Regularised regression to stabilise coefficients with many correlated controls.")

        feature_cols = list(X_base.columns)  # time_idx, campaign, dummies, controls
        X_reg = X_base[feature_cols].copy()
        y_reg = y.loc[X_reg.index]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reg)

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.radio('Model type', ['Ridge', 'Elastic Net'], horizontal=True)
        with col2:
            alpha_reg = st.slider('Alpha (penalty λ)', 0.0, 10.0, 1.0, 0.1)

        if model_type == 'Ridge':
            reg = Ridge(alpha=alpha_reg)
        else:
            l1_ratio = st.slider('Elastic Net l1_ratio', 0.0, 1.0, 0.5, 0.05)
            reg = ElasticNet(alpha=alpha_reg, l1_ratio=l1_ratio)

        reg.fit(X_scaled, y_reg)
        coef_series = pd.Series(reg.coef_, index=feature_cols)

        st.subheader('Regularised coefficients')
        st.dataframe(coef_series.to_frame('coef'))

        if 'campaign' in coef_series.index:
            st.markdown(f"**Campaign coefficient ({model_type}) ≈ {coef_series['campaign']:.2f} (on standardised predictors)**")

        top_coef = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
        max_plot_n = min(20, len(top_coef))
        n_plot = st.slider('Number of coefficients to plot', 3, max_plot_n, min(10, max_plot_n))

        top_df = top_coef.head(n_plot).reset_index()
        top_df.columns = ['feature', 'coef']

        fig_r = px.bar(top_df, x='feature', y='coef', text='coef')
        fig_r.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_r.update_layout(
            xaxis_title='Feature',
            yaxis_title='Coefficient',
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        st.plotly_chart(fig_r, use_container_width=True)

# Lagged ITS
with tabs[7]:
    st.header('Lagged ITS (spend lags)')
    st.markdown("Explore delayed impact of **Spend** on HDFC Sky searches using lagged spend terms.")

    max_lag = st.slider('Max lag (periods)', 1, 14, 7)

    # Build lagged design
    X_lag = pd.DataFrame(index=DF.index)
    X_lag['time_idx'] = np.arange(len(DF))

    # simple campaign indicator
    if start_idx is not None:
        camp_dummy = pd.Series(0, index=DF.index)
        camp_dummy.loc[start_idx:end_idx] = 1
        X_lag['campaign'] = camp_dummy
    else:
        X_lag['campaign'] = 0

    # current spend + lags
    X_lag['Spend'] = DF['Spend']
    for lag in range(1, max_lag+1):
        X_lag[f'Spend_lag{lag}'] = DF['Spend'].shift(lag)

    # controls as before
    for c in controls:
        X_lag[c] = DF[c]

    y_lag = DF[outcome]

    valid_mask = X_lag.notna().all(axis=1) & y_lag.notna()
    X_lag_clean = X_lag[valid_mask]
    y_lag_clean = y_lag[valid_mask]

    if len(X_lag_clean) < 10:
        st.info('Not enough data after applying lags to run lagged ITS.')
    else:
        model_lag = ols_with_controls(y_lag_clean, X_lag_clean)
        st.subheader('Lagged ITS model summary')
        st.text(model_lag.summary())

        # Extract spend lag coefficients
        lag_names = [f'Spend_lag{lag}' for lag in range(1, max_lag+1)]
        lag_coefs = []
        for lag_name in lag_names:
            if lag_name in model_lag.params.index:
                lag_coefs.append({
                    'lag': int(lag_name.replace('Spend_lag','')),
                    'coef': model_lag.params[lag_name]
                })
        if len(lag_coefs) > 0:
            lag_df = pd.DataFrame(lag_coefs).sort_values('lag')
            st.subheader('Spend lag coefficients')
            st.dataframe(lag_df)

            fig_lag = px.bar(lag_df, x='lag', y='coef', text='coef')
            fig_lag.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_lag.update_layout(
                xaxis_title='Lag (periods)',
                yaxis_title='Coefficient',
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_lag, use_container_width=True)
        else:
            st.info('No lag coefficients were estimated (possibly dropped in regression).')

# PCA Controls
with tabs[8]:
    st.header('PCA Controls (competitor compression)')
    if not SKLEARN_AVAILABLE:
        st.error('scikit-learn is not installed in this environment. Install it to use PCA.')
    elif len(controls) < 2:
        st.info('Select at least two control columns to run PCA.')
    else:
        st.markdown("Use PCA to compress many correlated competitors/indicators into a few orthogonal market factors.")

        C = DF[controls].copy()
        valid_mask = C.notna().all(axis=1)
        C_clean = C[valid_mask]

        if C_clean.shape[0] < 10:
            st.info('Not enough valid rows to run PCA on controls.')
        else:
            scaler_c = StandardScaler()
            C_scaled = scaler_c.fit_transform(C_clean)

            max_components = min(5, C_scaled.shape[1])
            n_components = st.slider(
                'Number of PCA components',
                1,
                max_components,
                min(3, max_components)
            )

            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(C_scaled)
            pc_cols = [f'PC{i+1}' for i in range(n_components)]

            # build regression design
            X_pca = pd.DataFrame(index=C_clean.index)
            X_pca['time_idx'] = np.arange(len(C_clean))

            if camp_start_date is not None and camp_end_date is not None:
                dates_sub = DF.loc[C_clean.index, 'Date']
                X_pca['campaign'] = ((dates_sub >= camp_start_date) & (dates_sub <= camp_end_date)).astype(int)
            else:
                X_pca['campaign'] = 0

            for i, col_name in enumerate(pc_cols):
                X_pca[col_name] = pcs[:, i]

            y_pca = DF.loc[C_clean.index, outcome]

            model_pca = ols_with_controls(y_pca, X_pca)

            st.subheader('ITS with PCA factors')
            st.text(model_pca.summary())

            if 'campaign' in model_pca.params.index:
                coef_pca = model_pca.params['campaign']
                se_pca = model_pca.bse['campaign']
                p_pca = model_pca.pvalues['campaign']
                st.markdown(f"**Campaign coefficient (PCA ITS)** = {coef_pca:.2f} (se {se_pca:.2f}), p = {p_pca:.4f}")

            st.subheader('Explained variance by PCA components')
            ev_df = pd.DataFrame({
                'PC': pc_cols,
                'explained_variance_ratio': pca.explained_variance_ratio_
            })
            st.dataframe(ev_df)

            fig_ev = px.bar(ev_df, x='PC', y='explained_variance_ratio', text='explained_variance_ratio')
            fig_ev.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_ev.update_layout(
                yaxis_title='Explained variance ratio',
                xaxis_title='Component',
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_ev, use_container_width=True)

            st.subheader('PCA loadings (which competitors drive each PC)')
            loadings = pd.DataFrame(
                pca.components_.T,
                index=controls,
                columns=pc_cols
            )
            st.dataframe(loadings)

# Weekly DiD
with tabs[9]:
    st.header('Weekly DiD view (Sky vs synthetic control)')
    st.markdown("""
    Aggregates to weekly and compares **HDFC Sky** vs a **synthetic control** (mean of selected controls),
    highlighting the campaign weeks.
    """)

    if len(controls) == 0:
        st.info('Select at least one control in the sidebar to build a synthetic control.')
    else:
        # If already weekly, use DF; else aggregate to weekly
        if weekly:
            DF_w = DF.copy()
        else:
            DF_w = aggregate_weekly(DF, date_col='Date')

        start_w, end_w = detect_campaign(DF_w['Spend'])
        if start_w is None:
            st.info('No non-zero weekly spend detected to define a campaign period.')
        else:
            DF_w['controls_mean'] = DF_w[controls].mean(axis=1)

            pre_out_w = DF_w.loc[:start_w-1, outcome].mean()
            camp_out_w = DF_w.loc[start_w:end_w, outcome].mean()
            pre_ctrl_w = DF_w.loc[:start_w-1, 'controls_mean'].mean()
            camp_ctrl_w = DF_w.loc[start_w:end_w, 'controls_mean'].mean()

            did_abs_w = (camp_out_w - pre_out_w) - (camp_ctrl_w - pre_ctrl_w)

            st.subheader('Weekly DiD summary')
            st.write(pd.DataFrame({
                'metric': ['pre_outcome','camp_outcome','pre_ctrl','camp_ctrl','DiD_abs'],
                'value': [pre_out_w, camp_out_w, pre_ctrl_w, camp_ctrl_w, did_abs_w]
            }))

            # Time series of outcome vs control
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=DF_w['Date'], y=DF_w[outcome], name='HDFC Sky'))
            fig_w.add_trace(go.Scatter(x=DF_w['Date'], y=DF_w['controls_mean'], name='Synthetic control'))
            fig_w.add_vrect(
                x0=DF_w.loc[start_w,'Date'],
                x1=DF_w.loc[end_w,'Date'],
                fillcolor='green',
                opacity=0.1,
                layer='below',
                line_width=0
            )
            fig_w.update_layout(
                yaxis_title='Weekly searches (or aggregated metric)',
                legend=dict(orientation='h')
            )
            st.subheader('Weekly outcome vs synthetic control')
            st.plotly_chart(fig_w, use_container_width=True)

            # Weekly difference bar chart
            DF_w['diff'] = DF_w[outcome] - DF_w['controls_mean']
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Bar(x=DF_w['Date'], y=DF_w['diff'], name='HDFC Sky − control'))
            fig_diff.add_vrect(
                x0=DF_w.loc[start_w,'Date'],
                x1=DF_w.loc[end_w,'Date'],
                fillcolor='green',
                opacity=0.1,
                layer='below',
                line_width=0
            )
            fig_diff.update_layout(
                yaxis_title='Difference (Sky − control)',
                legend=dict(orientation='h')
            )
            st.subheader('Weekly difference (Sky − synthetic control)')
            st.plotly_chart(fig_diff, use_container_width=True)

# Interpretation
with tabs[10]:
    st.header('Interpretation & Notes')
    st.markdown(r"""
## 1. TL;DR – What is this model saying?

- **Simple pre vs campaign**:  
  - Mean daily searches went from **pre** to **campaign** at a noticeable lift.
  - But the **t-test** and **bootstrap CI** tell you whether that lift is statistically robust or could be noise.

- **ITS with controls** (time trend, day-of-week, competitors, market factors):
  - The **campaign coefficient** captures incremental lift **after** adjusting for trends and controls.
  - If this coefficient is small and/or not significant, the raw uplift is likely explained by **market/competitor moves and time trend**.

- **DiD / synthetic control intuition**:
  - Compare HDFC Sky vs a **synthetic control** (average of competitors/market indicators).
  - If both grow similarly pre→campaign, then **Sky is riding the same tide** as the rest of the market.

Overall: Use **Pre vs Campaign** for raw lift, and **ITS / DiD / advanced views** to decide how much of that lift is truly **incremental** to the HDFC Sky branding spends.

---

## 2. How to read the main tabs

### (a) Pre vs Campaign – Simple lift view

- Compares **average searches pre vs during campaign**.
- Shows:
  - Descriptive stats,
  - T-test,
  - Bootstrap CI.

Use this tab as:

- A **quick gut-check** on whether searches moved up at all.
- A **storytelling anchor** (“we see ~X% higher searches in the campaign period”).

### (b) ITS / Regression – Causal-ish with controls

- Includes:
  - `time_idx` (trend),
  - `campaign` dummy,
  - Day-of-week,
  - Selected competitors & market indicators.
- Uses **HAC robust SE** to handle autocorrelation/heteroskedasticity.

Read:

- **Campaign coefficient** → incremental level shift after accounting for all included controls.
- Check:
  - Sign (positive/negative),
  - Magnitude,
  - p-value vs your chosen alpha.

This is the main **“did we create lift or just ride the market?”** view.

### (c) DiD & Controls – Synthetic control intuition

- Constructs a **simple synthetic control** = mean of selected controls.
- Compares pre vs campaign means for:
  - HDFC Sky,
  - `controls_mean`.

If Sky’s growth ≈ control’s growth:

- The campaign is likely **moving with the market** rather than driving extra lift.

---

## 3. Advanced views

### (a) Lift Summary

Combines:

- **Absolute lift (Δ mean)**,
- **Relative lift (%)**,
- **DiD-based lift**,
- **ITS-based lift (campaign coefficient)**

into two bar charts (absolute and %). This is your **single screenshot** for leadership.

### (b) Ridge / Elastic Net

- Addresses **multicollinearity** and many overlapping controls.
- Shrinks noisy coefficients (Ridge) or shrinks + sparsifies (Elastic Net).
- Look at:
  - **Campaign coefficient under regularisation**,
  - Which controls retain strong influence after shrinkage.

If campaign stays small even after regularisation → **weak attribution case**.

### (c) Lagged ITS (spend lags)

- Adds **lagged spend variables** (`Spend_lag1`, `Spend_lag2`, …).
- Lets you see whether **spend today drives searches with a delay**.
- Positive, significant coefficients at lag k → **evidence of a k-period delayed effect**.

### (d) PCA Controls

- Compresses the whole competitor/indicator block into a few **PCA factors (PC1, PC2, …)**.
- Reduces dimensionality and multicollinearity.
- Run ITS on:
  - `time_idx`,
  - `campaign`,
  - PCs instead of raw controls.

Check:

- How much variance each PC explains,
- PCA loadings (which competitors drive each factor),
- Campaign coefficient in the **PCA ITS** model.

### (e) Weekly DiD

- Aggregates to **weekly**.
- Plots:
  - Weekly **HDFC Sky**,
  - Weekly **synthetic control**,
  - A **difference bar chart (Sky − control)**, with campaign weeks highlighted.

If weekly differences jump up and stay up during campaign weeks → **clearer visual evidence of lift**.  
If they stay flat around zero → **Sky is just tracking the market**.

---

## 4. How to use this app in practice

1. **Start simple** – Pre vs Campaign to establish raw lift.
2. **Move to ITS** – Ask: “How much survives after controlling for trend & market?”
3. **Use DiD & Weekly DiD** – Compare Sky vs synthetic control over time.
4. **Stress-test with advanced tabs**:
   - Ridge / Elastic Net to stabilise,
   - Lagged ITS for timing of effects,
   - PCA to simplify complex competitive landscapes.

Your final narrative should go from:

> “Searches went up during the campaign”

to:

> “Searches went up, but after adjusting for time, competitors, and market conditions, the **incremental lift attributable purely to the HDFC Sky branding campaign is / is not statistically strong, and here’s the evidence across multiple methods**.”
    """)

# Export results
st.sidebar.header('Export')
if st.sidebar.button('Download model summary as CSV'):
    buf = BytesIO()
    try:
        summary_df = pd.DataFrame({
            'param': model.params.index,
            'coef': model.params.values,
            'pval': model.pvalues.values
        })
        summary_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button('Download CSV', data=buf, file_name='model_summary.csv', mime='text/csv')
    except Exception as e:
        st.error('No model to export or error: '+str(e))

st.sidebar.markdown('---')
st.sidebar.markdown('App created: uses default files from environment if not uploaded.')

# End of app
