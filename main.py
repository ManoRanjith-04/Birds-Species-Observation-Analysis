# --- 0. Imports ---
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from datetime import datetime

# --- 1. File paths ---
forest_path = Path("C://Users//Mano Ranjith//Downloads//Birds Species//Bird_Monitoring_Data_FOREST.XLSX - ANTI.csv")
grass_path = Path("C://Users//Mano Ranjith//Downloads//Birds Species//Bird_Monitoring_Data_GRASSLAND.XLSX - ANTI.csv")

# Output directories
out_dir = Path("C://Users//Mano Ranjith//Downloads//Birds Species//bird_analysis_outputs")
fig_dir = out_dir / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Helper functions ---

def safe_read_csv(p):
    """Try reading a CSV with different encodings and separators."""
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, encoding='latin1')
        except Exception:
            # Try semicolon separator
            return pd.read_csv(p, sep=';')


def parse_date_col(df, col_candidates=['Date', 'date', 'Observation_Date']):
    """Try to parse a usable date column from several possible column names."""
    for c in col_candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
                return df, c
            except Exception:
                df[c] = pd.to_datetime(df[c], dayfirst=False, errors='coerce')
                return df, c
                
    return df, None

# --- Column Standardization ---
def standardize_columns(df):
    """Standardize column names: strip, lowercase, replace spaces with underscores."""
    rename_map = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    df = df.rename(columns=rename_map)
    return df


# --- Cleaning Pipeline for a Single Sheet/File ---
def clean_df(df, source_label=None):
    df = df.copy()
    df = standardize_columns(df)

    # Record source (forest/grassland)
    if source_label:
        df['location_group'] = source_label

    # Parse date column
    df, date_col = parse_date_col(df, ['date', 'Date', 'observation_date'])
    if date_col:
        df['obs_date'] = df[date_col]
    else:
        df['obs_date'] = pd.NaT

    # Parse Year, Month, Weekday
    df['year'] = df['obs_date'].dt.year
    df['month'] = df['obs_date'].dt.month
    df['day_of_week'] = df['obs_date'].dt.day_name()

    # Standardize categorical columns
    for col in [
        'location_type', 'site_name', 'plot_name', 'observer',
        'id_method', 'distance', 'sex', 'common_name',
        'scientific_name'
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan})

    # Numeric conversions
    for num_col in ['temperature', 'humidity', 'initial_three_min_cnt']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    # Flyover boolean normalization
    if 'flyover_observed' in df.columns:
        df['flyover_observed'] = df['flyover_observed'].map(
            lambda v: True if str(v).strip().upper() in ['TRUE', 'T', '1', 'YES', 'Y']
            else False if str(v).strip().upper() in ['FALSE', 'F', '0', 'NO', 'N']
            else np.nan
        )

    # Disturbance normalization
    if 'disturbance' in df.columns:
        df['disturbance'] = df['disturbance'].replace({'': np.nan})

    # Clean scientific_name
    if 'scientific_name' in df.columns:
        df['scientific_name'] = df['scientific_name'].where(
            pd.isna(df['scientific_name']),
            df['scientific_name'].str.title()
        )

    return df


# --- 3. Load data ---
print('Reading forest file:', forest_path)
forest_df = safe_read_csv(forest_path)
print('Reading grassland file:', grass_path)
grass_df = safe_read_csv(grass_path)

print('\nRaw shapes:')
print(' - forest:', forest_df.shape)
print(' - grassland:', grass_df.shape)

# --- 4. Clean each dataframe ---
forest_clean = clean_df(forest_df, source_label='Forest')
grass_clean = clean_df(grass_df, source_label='Grassland')

print('\nAfter cleaning shapes:')
print(' - forest:', forest_clean.shape)
print(' - grassland:', grass_clean.shape)

# --- 5. Harmonize schemas and merge ---
# Find union of columns
all_cols = sorted(set(forest_clean.columns).union(set(grass_clean.columns)))

# Reindex to same columns (missing columns will be NaN)
forest_clean = forest_clean.reindex(columns=all_cols)
grass_clean = grass_clean.reindex(columns=all_cols)

# Add a source file column for traceability
forest_clean['source_file'] = os.path.basename(forest_path)
grass_clean['source_file'] = os.path.basename(grass_path)

master = pd.concat([forest_clean, grass_clean], axis=0, ignore_index=True)
print('\nMaster shape after concat:', master.shape)

# Create a cleaned master subset with commonly used columns first
preferred_cols = [c for c in ['location_group','location_type','site_name','plot_name','obs_date','year','month','day_of_week','start_time','end_time','observer','visit','interval_length','id_method','distance','flyover_observed','sex','common_name','scientific_name','temperature','humidity','sky','wind','disturbance','initial_three_min_cnt','source_file'] if c in master.columns]
other_cols = [c for c in master.columns if c not in preferred_cols]
master = master[preferred_cols + other_cols]

# Save master csv for downstream use (Power BI, further analysis)
master_csv_path = out_dir / 'master_bird_observations_cleaned.csv'
master.to_csv(master_csv_path, index=False)
print(f"Saved master CSV to: {master_csv_path}")


# --- 6. Quick quality checks ---
print('\nQuality checks:')

print('Missing obs_date:', master['obs_date'].isna().sum() if 'obs_date' in master.columns else 'no obs_date')

if 'scientific_name' in master.columns:
    print('Unique species (scientific_name):', master['scientific_name'].nunique())

# --- 7. Exploratory Analysis ---

# 7.1 Observations by habitat
hab_counts = master['location_group'].value_counts(dropna=False)
print('\nObservations by habitat:\n', hab_counts)

# 7.2 Top species overall
if 'scientific_name' in master.columns:
    top_species = master['scientific_name'].value_counts().head(20)
    print('\nTop species (by record count):')
    print(top_species)

# 7.3 Yearly trend (if year exists)
if 'year' in master.columns and master['year'].notna().any():
    yearly = master.groupby('year').size().reset_index(name='observations')
    print('\nYearly counts:')
    print(yearly)

# 7.4 Seasonal / monthly pattern
if 'month' in master.columns and master['month'].notna().any():
    monthly = master.groupby('month').size().reindex(range(1, 13), fill_value=0).reset_index(name='observations')
    print('\nMonthly counts (1-12):')
    print(monthly)

import plotly.express as px

# Ensure the habitat counts variable is created
if 'location_group' in master.columns:
    hab_counts = master['location_group'].value_counts()
elif 'habitat' in master.columns:
    hab_counts = master['habitat'].value_counts()
else:
    raise KeyError("No habitat/location_group column found in master dataframe.")

# --- 8.1 Observations by habitat bar ---
hab_df = hab_counts.reset_index()
hab_df.columns = ['habitat', 'count']

fig1 = px.bar(
    hab_df,
    x='habitat',
    y='count',
    title='Observations by Habitat'
)
fig1.show()


# --- 8.2 Top species bar ---
if 'scientific_name' in master.columns:
    top_df = master['scientific_name'].value_counts().head(30).reset_index()
    top_df.columns = ['scientific_name', 'count']

    fig2 = px.bar(
        top_df,
        x='scientific_name',
        y='count',
        title='Top 30 Species (by records)'
    )
    fig2.update_layout(xaxis_tickangle=45)
    fig2.show()


# --- 8.3 Monthly time series ---
if 'month' in master.columns:
    monthly = master.groupby('month').size().reset_index(name='observations')

    fig3 = px.line(
        monthly.sort_values('month'),
        x='month',
        y='observations',
        title='Observations by Month'
    )
    fig3.show()


# --- 8.4 Species diversity per habitat ---
if 'scientific_name' in master.columns:
    diversity = (
        master.groupby('location_group')['scientific_name']
        .nunique()
        .reset_index(name='unique_species')
    )

    fig4 = px.bar(
        diversity,
        x='location_group',
        y='unique_species',
        title='Unique Species per Habitat'
    )
    fig4.show()


# --- 8.5 Scatter: Temperature vs Initial_Three_Min_Cnt ---
if 'temperature' in master.columns and 'initial_three_min_cnt' in master.columns:
    scatter_df = master.dropna(subset=['temperature', 'initial_three_min_cnt'])

    if not scatter_df.empty:
        fig5 = px.scatter(
            scatter_df,
            x='temperature',
            y='initial_three_min_cnt',
            hover_data=['scientific_name', 'location_group'],
            title='Temperature vs Initial 3-min Count'
        )
        fig5.show()


# --- 8.6 Map visualization (lat/lon) ---
if 'latitude' in master.columns and 'longitude' in master.columns:
    map_df = master.dropna(subset=['latitude', 'longitude'])

    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df,
            lat='latitude',
            lon='longitude',
            hover_name='scientific_name',
            zoom=6,
            height=600
        )
        fig_map.update_layout(mapbox_style='open-street-map')
        fig_map.show()


