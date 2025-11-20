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
    model = sm.OLS(y_clean.astype(float), Xc.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':7})
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

# CRITICAL FIX: Ensure ALL non-Date columns are numeric
for c in DF.columns:
    if c == 'Date':
        continue
    DF[c] = pd.to_numeric(DF[c], errors='coerce')

# Fill any remaining NaNs with 0 (or you could drop these rows)
numeric_cols = [c for c in DF.columns if c != 'Date']
DF[numeric_cols] = DF[numeric_cols].fillna(0)

# Final dtype check
problem_cols = [c for c in DF.columns if c != 'Date' and not np.issubdtype(DF[c].dtype, np.number)]
if len(problem_cols) > 0:
    st.error('The following columns are still non-numeric and will break numeric ops: ' + ', '.join(problem_cols))
    st.dataframe(DF[problem_cols].head())
    st.stop()

st.success('Data cleaning complete — all numeric columns verified.')

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

alpha = st.sidebar.number_input(
    'Significance alpha',
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.01
)

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
        X[c] = DF[c].values

    y = DF[outcome]  # Keep as Series, not .values

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
    fitted_full = pd.Series(index=DF.index, dtype=float)
    fitted_full.loc[model.fittedvalues.index] = model.fittedvalues.values
    DF['fitted'] = fitted_full
    
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

        st.subheader('Absolute lift (incremental searches per day)')
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
        - **Absolute (Δ mean)**: Difference in average searches per day (campaign − pre).
        - **Relative (%)**: Percentage change in average searches: (campaign / pre − 1) × 100.
        - **DiD (Δ vs controls)**: (Δ outcome) − (Δ synthetic control), using mean of selected controls.
        - **ITS (campaign coef)**: Incremental level shift estimated by the ITS regression with controls.
        """)

# Interpretation
with tabs[6]:
    st.header('Interpretation & Notes')
    st.markdown(r"""
## 1. TL;DR – What is this model saying?

- **Simple pre vs campaign**:  
  - Mean daily searches went from **≈ 722 (pre)** to **≈ 817 (campaign)**.  
  - That’s an **absolute change of ≈ +95 searches/day** and a **relative change of ≈ +13%**.
  - However, the **t-test p ≈ 0.27** and the **bootstrap 95% CI [−72, +252]** both include 0 →  
    **we cannot statistically rule out “no lift”** on a pure pre/post basis.

- **ITS with controls** (time trend, day-of-week, competitors, market factors):
  - The **campaign coefficient is −24** (se ≈ 64, p ≈ 0.71) → **no evidence of incremental lift**  
    once you control for time, competitors, and market indicators.
  - The model fits the series fairly well overall (**R² ≈ 0.78**), but the **incremental “on top of everything else” effect of the campaign is not statistically different from zero**.

- **DiD / synthetic control intuition**:
  - HDFC Sky grows from ~722 to ~817 (**~+13.1%**).
  - The synthetic control (mean of competitors/market indicators) grows from ~49.6k to ~55.9k (**~+12.8%**).
  - **Sky is broadly moving in line with the market/competition, not clearly outperforming it**.

Overall: **Search volumes are higher during the campaign, but once you account for trends and a very strong market/competitor uplift, the model does *not* see a clean, statistically significant incremental lift attributable purely to the HDFC Sky branding spends.**

---

## 2. How to read each tab in this app

### (a) Pre vs Campaign – Simple lift view

This is the most intuitive, “raw” view:

- Compares **average daily searches pre vs during the campaign**.
- Reports:
  - **Descriptive stats** (n, mean, std),
  - **T-test** for mean difference,
  - **Bootstrap confidence interval** for the mean difference.

For your current runs:

- **Absolute lift (campaign − pre)** ≈ **+94.6 searches/day**.
- **Relative lift** ≈ **+13%**.
- But:
  - **p ≈ 0.27** (not significant at typical α = 0.05),
  - **Bootstrap 95% CI includes 0**, so the “true” lift could plausibly be small, zero, or even slightly negative.

Use this tab as:

- A **quick gut-check**: did searches move up during the campaign window at all?
- A **storytelling entry point** for non-technical stakeholders: “raw pre vs campaign change is +13%, but we still need to adjust for trend and market factors.”

---

### (b) ITS / Regression – Causal-ish, with controls

Here we do an **Interrupted Time Series (ITS)** style OLS regression with:

- **time_idx**: linear time trend,
- **campaign dummy**: 0 pre, 1 during campaign,
- **day-of-week dummies**,
- **competitors & market indicators** (Zerodha, Sensex, etc.) as controls,
- **HAC robust standard errors** to handle autocorrelation/heteroskedasticity.

Key numbers from your run:

- **R² ≈ 0.776** (model explains a good chunk of variation).
- **Campaign coef ≈ −23.97**, se ≈ 64.34, **p ≈ 0.71**:
  - After conditioning on trend, day-of-week and competitors/market variables,
  - There is **no statistically significant level shift** attributed to the campaign.
- Some controls show meaningful relationships:
  - **Zerodha** has a positive and significant coefficient → when Zerodha searches go up, HDFC Sky searches tend to go up.
  - **Sensex** is also positive and significant → index sentiment/market conditions matter.

How to read this for attribution:

- The pre/post lift we see in raw averages is **largely explained by general trends and market/competitor movements**.
- **Incremental lift on top of those forces is not detectable** in this specification.
- This is a **more conservative, “causal-ish” view** than the simple pre vs campaign tab.

---

### (c) DiD & Controls – Synthetic control intuition

This tab:

1. Builds a **simple synthetic control**:  
   \- `controls_mean = mean(selected competitors & market indicators)`.

2. Looks at **pre vs campaign mean** for:
   - HDFC Sky outcome,
   - `controls_mean`.

From your current outputs:

- **HDFC Sky**:
  - Pre: ≈ 721.7  
  - Campaign: ≈ 816.5  
  - Growth: **≈ +13.1%**

- **Synthetic control (controls_mean)**:
  - Pre: ≈ 49,585  
  - Campaign: ≈ 55,945  
  - Growth: **≈ +12.8%**

Interpretation:

- The **market/competitor environment itself is lifting strongly** (~13%).
- HDFC Sky’s relative growth is **very similar** to that background movement.
- In Difference-in-Differences terms, **there is little clear “excess lift” vs synthetic control**.

The “Regression using competitor controls only” below that simply quantifies:

- How much HDFC Sky searches **co-move** with the competitors’ synthetic index.
- With **R² ≈ 0.68** and a highly significant `controls_mean` coefficient, **competitors plus market factors explain a large share of HDFC Sky variation**.

---

### (d) Correlations – Are we just seeing a market story?

This tab gives you:

- A **correlation matrix** between:
  - HDFC Sky (outcome),
  - Selected competitors,
  - Market indicators.

Use it to:

- Identify **which competitors/indicators move closely with HDFC Sky**.
- Spot **clusters** (e.g. brokerage terms moving together, macro indices moving together).
- Decide which controls are **essential** vs **redundant**.

If many competitors are highly correlated with each other, that supports the need for **regularisation (ridge/elastic net)** or **dimensionality reduction (PCA)** – see below.

---

### (e) Lift Summary – tying the story together

The Lift Summary tab compresses multiple perspectives into **two bar charts**:

1. **Absolute lift (incremental searches/day)**:
   - **Absolute (Δ mean)** – raw pre vs campaign difference (~+95/day).
   - **DiD (Δ vs controls)** – how much of that remains after netting out movement in the synthetic control.
   - **ITS (campaign coef)** – incremental shift estimated by the ITS regression.

2. **Relative lift (% vs pre)**:
   - **Relative (%)** – raw percentage change (~+13%).
   - **DiD (% vs pre)** – relative lift vs synthetic control.
   - **ITS (% vs pre)** – ITS campaign coefficient as % of pre-campaign mean.

Expected pattern given your numbers:

- Simple pre/post bars will show a **positive absolute and % lift**.
- DiD/ITS bars may cluster closer to **zero (and possibly slightly negative)**, reflecting the fact that:
  - The **market/competitor environment also lifted strongly**, and
  - Once that is soaked up, **Sky’s incremental lift is not statistically compelling**.

This is your **“single screenshot explanation chart”** for leadership:
> “We see uplift in raw numbers, but relative to market & after controlling for trends, the incremental lift from this specific branding burst is not statistically solid.”

---

## 3. Advanced / future extensions (not all implemented yet)

The following concepts are **natural next steps** and can be wired into new tabs:

### (a) Synthetic control regression

- Instead of a simple mean of competitors, we can build a **weighted synthetic control**:
  - Choose weights on competitor/indicator series to **best match pre-campaign HDFC Sky**.
  - Then compare **post-campaign divergence** between HDFC Sky and this optimized synthetic series.
- Interpretation:
  - If Sky diverges **upward** from the synthetic control post-launch → evidence of lift.
  - If Sky tracks the synthetic control closely → campaign likely just rode the overall market.

Inside the app, this would look like:

- A **“Synthetic Control” tab** with:
  - Pre-period fit chart (Sky vs synthetic),
  - Post-period divergence plot,
  - Simple summary metrics (average incremental lift, % lift vs synthetic).

---

### (b) Ridge / Elastic Net regression tabs

Problem today: many controls are **highly correlated** (Zerodha, Trading, Stocks, etc.), which can cause:

- High **multicollinearity**,
- Unstable individual coefficients,
- Inflated standard errors.

Solution:

- Add **Ridge** or **Elastic Net** regression tabs:
  - Same outcome (HDFC Sky),
  - Same predictors (time, dow, campaign, competitors, markets),
  - But with **L2 (ridge)** or **L1+L2 (elastic net)** penalties.

Benefits:

- **Shrinks** noisy coefficients,
- **Stabilises** the influence estimates for each control,
- Helps you see whether **campaign still adds incremental explanatory power** once the model is regularised.

Interpretation tab would then describe:

> “When we penalise and shrink noisy competitor features, the campaign coefficient remains small/insignificant (or becomes stronger), which strengthens/weakens the attribution case accordingly.”

---

### (c) Lagged ITS – allowing delayed effects

Right now, the campaign dummy is an **instant level shift**. In reality, branding effects may be:

- **Delayed** (e.g. awareness → consideration → searches),
- **Spread** over several days/weeks.

Lagged ITS extension:

- Add **lagged versions** of the campaign dummy and/or spend:
  - `campaign_lag1`, `campaign_lag7` or moving-average terms,
  - Or lagged ad spends by 1–7 days.
- Re-estimate ITS with these lagged variables.

Interpretation:

- Significant positive coefficients on **lagged campaign/spend** terms would support:
  - “Branding effect manifests with a ~X-day lag,”
  - Or “Effect accumulates over a week rather than a sharp break on launch day.”

---

### (d) PCA competitor compression

Given many overlapping competitor & keyword series, we can:

- Run **Principal Component Analysis (PCA)** on the competitor/indicator block.
- Replace dozens of collinear controls with a handful of **orthogonal “market factors”** (PC1, PC2, PC3…).

Benefits:

- Reduces **dimensionality**,
- Tackles **multicollinearity** explicitly,
- Makes the ITS & DiD models more stable.

In the UI, a **“PCA Controls” tab** could show:

- Variance explained by each component,
- Loadings (which competitors contribute to each factor),
- ITS results using only **PC1–PCk** instead of raw competitors.

---

### (e) Weekly DiD chart – cleaner visual for management

To smooth noise and provide a more presentation-friendly view:

1. Aggregate to **weekly data** (checkbox already exists).
2. Compute:
   - Weekly **HDFC Sky search totals or averages**,
   - Weekly **synthetic control** (competitor/market average or PCA factor),
   - **Pre vs post weekly differences**.
3. Plot:
   - A **time series** of HDFC Sky vs synthetic control by week,
   - A **bar chart** of **weekly difference (Sky − synthetic)**, highlighting the campaign weeks.

Interpretation:

- If weekly differences **consistently jump up** during campaign weeks vs pre-trend → visual DiD-style evidence of lift.
- If they **oscillate around zero** or **decline**, the campaign is more likely just tracking or slightly underperforming the broader market wave.

---

## 4. How to use this app in practice

1. **Start with Pre vs Campaign**  
   - Use this to **anchor the business story** (“we saw ~13% higher searches”).

2. **Cross-check with ITS & Lift Summary**  
   - Ask: *“How much of this survives after accounting for time, competition and the market?”*
   - Use the **ITS and Lift Summary** to calibrate expectations.

3. **Look at DiD / synthetic control**  
   - See whether **Sky moved differently from the crowd**.
   - If not, **position the campaign as “riding the tide” rather than purely creating it.**

4. **Use advanced views (ridge, PCA, lagged ITS, weekly DiD)**  
   - For deeper modelling and to stress-test attribution conclusions,
   - Especially when presenting to analytics / data science stakeholders.

In short, the Interpretation layer should help you move the conversation from:

> “Searches went up during the campaign, so it worked”

to the more nuanced and rigorous:

> “Searches went up, but **after adjusting for trend, competition, and market conditions, we do / do not see robust incremental lift clearly attributable to the campaign**.”
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
