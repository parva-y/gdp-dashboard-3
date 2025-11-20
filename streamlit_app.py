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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title='HDFC Sky Lift Analyzer', layout='wide')

st.title('HDFC Sky — Branding Lift & Attribution Analyzer')
st.markdown("""
This app:
- Ingests daily search volumes, campaign daily spends, and market indicators.
- Auto-detects campaign period (spend > 0; assumes continuous campaign).
- Runs multiple lift analyses: pre-vs-campaign tests, ITS regression, DiD-like controls, OLS with spends + controls, cross-correlation and lag analysis.
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
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def aggregate_weekly(df, date_col='Date'):
    df = df.copy()
    df['WeekStart'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
    agg = df.groupby('WeekStart').sum().reset_index().rename(columns={'WeekStart': 'Date'})
    return agg


def detect_campaign(spend_series):
    # campaign where spend > 0
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
    Xc = add_constant(X)
    model = sm.OLS(y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags':7})
    return model

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

# helper to coerce numeric-like strings to numbers
def safe_to_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[,\s%₹]', '', regex=True).str.strip(), errors='coerce')

# Ensure Date columns are parsed first (try dayfirst)
searches = parse_dates(searches, date_col='Date')
spends = parse_dates(spends, date_col='Date')

# Identify non-date columns in searches and coerce to numeric where appropriate
non_date_cols = [c for c in searches.columns if c != 'Date']
for col in non_date_cols:
    # coerce and replace in-place, but keep original if needed for debugging
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

# Ensure remaining non-date columns are numeric; coerce if needed and stop if any remain non-numeric
for c in DF.columns:
    if c == 'Date':
        continue
    if DF[c].dtype == 'O':
        try:
            DF[c] = safe_to_numeric(DF[c])
            st.info(f'Coerced {c} to numeric.')
        except Exception:
            st.warning(f'Could not coerce column {c}; it remains object dtype.')

problem_cols = [c for c in DF.columns if c != 'Date' and not np.issubdtype(DF[c].dtype, np.number)]
if len(problem_cols) > 0:
    st.error('The following columns are still non-numeric and will break numeric ops: ' + ', '.join(problem_cols))
    st.stop()

st.success('Data cleaning complete — numeric types OK.')

# weekly toggle
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
        camp_start_date = DF.loc[start_idx, 'Date']
        camp_end_date = DF.loc[end_idx, 'Date']
        st.sidebar.success(f"Auto-detected campaign: {camp_start_date.date()} → {camp_end_date.date()}")


# columns
st.sidebar.markdown('**Columns detected**')
cols = [c for c in DF.columns if c not in ['Date','Spend']]
st.sidebar.text(', '.join(cols[:10]) + (', ...' if len(cols)>10 else ''))

# default outcome is HDFC Sky
outcome = st.sidebar.selectbox('Primary outcome (search column)', options=cols, index=0)

# indicators and competitors — by default all others
controls = st.sidebar.multiselect('Controls (competitors + market indicators)', options=[c for c in cols if c!=outcome], default=[c for c in cols if c!=outcome])

# lags
st.sidebar.markdown('Lag settings')
max_lag = st.sidebar.number_input('Max lag (days) to consider', min_value=0, max_value=30, value=7)

alpha = st.sidebar.number_input('Significance alpha', min_value=0.001, max_value=0.5, value=0.05, step=0.01)

# ----------------------------
# Tabs
# ----------------------------

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
        camp_start_date = DF.loc[start_idx, 'Date']
        camp_end_date = DF.loc[end_idx, 'Date']
        st.sidebar.success(f'Auto-detected campaign: {camp_start_date.date()} -> {camp_end_date.date()}')

tabs = st.tabs(['Overview & Plot','Pre vs Campaign','ITS / Regression','DiD & Controls','Correlations & Lags','Interpretation'])

# Overview
with tabs[0]:
    st.header('Time series overview')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF['Date'], y=DF[outcome], name=outcome))
    fig.add_trace(go.Bar(x=DF['Date'], y=DF['Spend'], name='Spend', yaxis='y2', opacity=0.4))
    fig.update_layout(yaxis=dict(title='Search Volume'), yaxis2=dict(title='Spend (INR)', overlaying='y', side='right'), legend=dict(orientation='h'))
    if start_idx is not None:
        fig.add_vrect(x0=DF.loc[start_idx,'Date'], x1=DF.loc[end_idx,'Date'], fillcolor='green', opacity=0.1, layer='below', line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# Pre vs Campaign
with tabs[1]:
    st.header('Pre vs Campaign — simple tests')
    if start_idx is None:
        st.info('No campaign — skipping pre vs campaign tests')
    else:
        pre, camp, post = summarize_period(DF, 'Date', outcome, start_idx, end_idx)
        st.subheader('Descriptive stats')
        st.write(pd.DataFrame({'period': ['pre','campaign'], 'n': [len(pre), len(camp)], 'mean': [np.mean(pre), np.mean(camp)], 'std': [np.std(pre, ddof=1), np.std(camp, ddof=1)]}))

        # t-test
        tstat, pval, dfree = ttest_ind(camp, pre, usevar='unequal')
        st.write(f'T-test (campaign vs pre): t = {tstat:.3f}, p = {pval:.4f}')

        # bootstrap
        lower, upper, mean_diff = bootstrap_diff(np.array(pre), np.array(camp))
        st.write(f'Bootstrap mean diff (campaign - pre): {mean_diff:.2f}; 95% CI = [{lower:.2f}, {upper:.2f}]')

        st.plotly_chart(px.box(pd.DataFrame({'pre':pre, 'campaign':camp}).melt(var_name='period', value_name='value'), x='period', y='value', points='all'), use_container_width=True)

# ITS / Regression
with tabs[2]:
    st.header('Interrupted Time Series / OLS with controls')
    # Prepare design
    X = pd.DataFrame()
    X['time_idx'] = np.arange(len(DF))
    if start_idx is not None:
        X['campaign'] = 0
        X.loc[start_idx:end_idx,'campaign'] = 1
    else:
        X['campaign'] = 0
    # add day-of-week
    try:
        X['dow'] = DF['Date'].dt.dayofweek
        X = pd.get_dummies(X, columns=['dow'], drop_first=True)
    except Exception:
        pass

    # add controls
    for c in controls:
        X[c] = DF[c]

    y = DF[outcome]

    model = ols_with_controls(y, X)
    st.subheader('Model summary (HAC SE)')
    st.text(model.summary())

    # coefficient for campaign
    if 'campaign' in model.params.index:
        coef = model.params['campaign']
        se = model.bse['campaign']
        p = model.pvalues['campaign']
        st.write(f'Campaign coefficient = {coef:.2f} (se {se:.2f}), p = {p:.4f}')

    # plot fitted vs actual
    DF['fitted'] = model.fittedvalues
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=DF['Date'], y=DF[outcome], name='Actual'))
    fig2.add_trace(go.Scatter(x=DF['Date'], y=DF['fitted'], name='Fitted'))
    if start_idx is not None:
        fig2.add_vrect(x0=DF.loc[start_idx,'Date'], x1=DF.loc[end_idx,'Date'], fillcolor='green', opacity=0.08, layer='below', line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

# DiD & Controls
with tabs[3]:
    st.header('Difference-in-Differences style check & Controls')
    # Create a synthetic control by averaging controls
    DF['controls_mean'] = DF[controls].mean(axis=1)
    if start_idx is not None:
        pre_mean_outcome = DF.loc[:start_idx-1, outcome].mean()
        camp_mean_outcome = DF.loc[start_idx:end_idx, outcome].mean()
        pre_mean_ctrl = DF.loc[:start_idx-1, 'controls_mean'].mean()
        camp_mean_ctrl = DF.loc[start_idx:end_idx, 'controls_mean'].mean()
        st.write('Raw pre vs campaign means (outcome vs controls_mean)')
        st.write(pd.DataFrame({'metric': ['pre_outcome','camp_outcome','pre_ctrl','camp_ctrl'], 'value':[pre_mean_outcome,camp_mean_outcome,pre_mean_ctrl,camp_mean_ctrl]}))

    st.subheader('Regression using competitor controls only')
    Xc = add_constant(DF[['controls_mean']])
    mod_ctrl = sm.OLS(DF[outcome], Xc).fit()
    st.text(mod_ctrl.summary())

# Correlations & Lags
with tabs[4]:
    st.header('Correlations, cross-correlation & lag analysis')
    st.subheader('Correlation matrix (outcome + controls)')
    corr_df = DF[[outcome]+controls].corr()
    st.dataframe(corr_df)
    st.plotly_chart(px.imshow(corr_df, text_auto=True, aspect='auto'), use_container_width=True)

    st.subheader('Cross-correlation (lag) between Spend and Outcome)')
    maxlag = int(max_lag)
    s = DF['Spend'].values - np.mean(DF['Spend'].values)
    y = DF[outcome].values - np.mean(DF[outcome].values)
    ccs = [np.corrcoef(s[:-lag] if lag>0 else s, y[lag:] if lag>0 else y)[0,1] for lag in range(0, maxlag+1)]
    lag_df = pd.DataFrame({'lag': list(range(0,maxlag+1)), 'corr': ccs})
    st.line_chart(lag_df.rename(columns={'lag':'index'}).set_index('index'))

    # Granger causality on top competitors
    try:
        st.subheader('Granger causality tests (top controls)')
        top_controls = list(DF[controls].corrwith(DF[outcome]).abs().sort_values(ascending=False).index[:5])
        gc_results = {}
        from statsmodels.tsa.stattools import grangercausalitytests
        for ctl in top_controls:
            test_df = DF[[outcome, ctl]].dropna()
            try:
                res = grangercausalitytests(test_df[[outcome, ctl]], maxlag=min(7, maxlag), verbose=False)
                pvals = [res[l][0]['ssr_ftest'][1] for l in res]
                gc_results[ctl] = pvals
            except Exception as e:
                gc_results[ctl] = str(e)
        st.write(gc_results)
    except Exception as e:
        st.write('Granger tests failed: ', e)

# Interpretation
with tabs[5]:
    st.header('Interpretation & Notes')
    st.markdown('''
    - The **Pre vs Campaign** tests give a quick check: mean changes, t-test and bootstrap CI.
    - The **ITS / Regression** tab attempts to control for time trend, day-of-week and the competitor/indicator columns you selected.
    - The **DiD** style check creates a simple synthetic control (mean of competitor columns) — not a full synthetic control method but useful as a quick check.
    - The **Correlations & Lags** tab helps detect whether rises in HDFC Sky follow spend with a lag, or whether competitors/market indicators move together (which might indicate market-wide effects).

    **How to read results:**
    - Focus on the campaign coefficient in the ITS model: its sign, magnitude and p-value. If positive and significant (p < alpha) that suggests lift after accounting for included controls.
    - If competitors and market indicators move similarly and their coefficients absorb the campaign effect, that suggests your campaign may be confounded by market-wide trends.
    - Use bootstrapped CIs for robust, distribution-free estimates of incremental mean change.

    **Limitations:**
    - This app uses OLS + HAC SE for quick inference. For a full Bayesian structural-time-series causal impact analysis consider running a dedicated Bayesian package offline.
    - The synthetic control here is a simple average; for stronger attribution use proper synthetic control libraries.

    ''')

# Export results
st.sidebar.header('Export')
if st.sidebar.button('Download model summary as CSV'):
    buf = BytesIO()
    try:
        summary_df = pd.DataFrame({'param': model.params.index, 'coef': model.params.values, 'pval': model.pvalues.values})
        summary_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button('Download CSV', data=buf, file_name='model_summary.csv', mime='text/csv')
    except Exception as e:
        st.error('No model to export or error: '+str(e))

st.sidebar.markdown('---')
st.sidebar.markdown('App created: uses default files from environment if not uploaded.')

# End of app
