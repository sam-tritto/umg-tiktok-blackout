import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env if it exists
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print("Warning: .env file not found. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in your environment.")

def download_dataset():
    # Make sure credentials are set
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("Warning: KAGGLE_USERNAME and KAGGLE_KEY environment variables are not set.")
        print("We will fall back to generating a highly realistic mock weekly streaming dataset.")
        print("To use the real dataset, please copy .env.template to .env and fill in your credentials.")
        return False
        
    print("Authenticating with Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error during Kaggle authentication: {e}")
        print("We will fall back to generating a highly realistic mock weekly streaming dataset.")
        return False
        
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    dataset_slug = "federicocester97/spotify-global-chart-2024"
    print(f"Downloading dataset '{dataset_slug}' to './data'...")
    try:
        api.dataset_download_files(dataset_slug, path=str(data_dir), unzip=True)
        print("Download and extraction complete.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("We will fall back to generating a highly realistic mock weekly streaming dataset.")
        return False

def generate_mock_data():
    print("\nGenerating realistic mock Spotify global streaming panel data for 2024...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 26 weeks from Jan 1, 2024 to June 30, 2024
    dates = pd.date_range(start='2024-01-07', end='2024-06-30', freq='W-SUN')
    
    treated_artists = ["Billie Eilish", "Noah Kahan"]
    donor_pool = [
        "Dua Lipa", 
        "Travis Scott", 
        "Ed Sheeran", 
        "Miley Cyrus", 
        "SZA", 
        "Harry Styles", 
        "Jack Harlow", 
        "Tate McRae"
    ]
    all_artists = treated_artists + donor_pool
    
    np.random.seed(42)
    records = []
    
    # Base stream counts in millions
    base_streams = {
        "Billie Eilish": 38.0,
        "Noah Kahan": 25.0,
        "Dua Lipa": 42.0,
        "Travis Scott": 35.0,
        "Ed Sheeran": 48.0,
        "Miley Cyrus": 28.0,
        "SZA": 45.0,
        "Harry Styles": 30.0,
        "Jack Harlow": 25.0,
        "Tate McRae": 22.0
    }
    
    # Generate weekly records
    for w_idx, dt in enumerate(dates):
        # Time components
        # General market trend: slight summer growth
        market_trend = 1.0 + (w_idx / len(dates)) * 0.1
        
        # SCM pre-treatment end
        is_blackout = 1 if (dt >= pd.Timestamp('2024-02-01') and dt <= pd.Timestamp('2024-05-05')) else 0
        
        # Donors have random walks + noise
        donor_values = {}
        for artist in donor_pool:
            # Random walk step
            noise = np.random.normal(0, 0.02)
            # Add a slight dip for some donor artists to make weights interesting
            individual_trend = 1.0 + np.sin(w_idx / 3.0) * 0.05
            streams_val = base_streams[artist] * market_trend * individual_trend * (1.0 + noise)
            donor_values[artist] = streams_val * 1e6
            
        # The Treated artists
        # 1. Billie Eilish (flat/stable)
        synthetic_billie = (
            0.40 * donor_values["Dua Lipa"] + 
            0.35 * donor_values["SZA"] + 
            0.25 * donor_values["Miley Cyrus"]
        )
        billie_noise = np.random.normal(0, 0.01)
        billie_streams = synthetic_billie * (1.0 + billie_noise)
        
        # 2. Noah Kahan (fast growing pre-blackout, then drops)
        synthetic_noah = (
            0.50 * donor_values["Ed Sheeran"] + 
            0.30 * donor_values["Jack Harlow"] + 
            0.20 * donor_values["Tate McRae"]
        )
        noah_noise = np.random.normal(0, 0.01)
        noah_streams = synthetic_noah * (1.0 + noah_noise)
        # Pre-treatment growth spurt for Noah
        if not is_blackout and dt < pd.Timestamp('2024-02-01'):
            noah_streams *= (1.0 + w_idx * 0.05)
            
        # Causal drop during blackout (Feb 1 to May 2)
        if is_blackout:
            # Billie Eilish has no/flat drop (simulating mega-artist immunity)
            billie_streams *= 0.99
            # Noah Kahan has a major 22% causal drop
            noah_streams *= 0.78
            
        # Add records for all artists
        for artist in all_artists:
            is_umg = 1 if artist in treated_artists else 0
            if artist == "Billie Eilish":
                val = billie_streams
            elif artist == "Noah Kahan":
                val = noah_streams
            else:
                val = donor_values[artist]
            
            records.append({
                'week_date': dt,
                'artist': artist,
                'streams': int(val),
                'is_umg': is_umg,
                'is_blackout': is_blackout
            })
            
    panel_df = pd.DataFrame(records)
    output_path = data_dir / "umg_tiktok_blackout_weekly_streams.csv"
    panel_df.to_csv(output_path, index=False)
    print(f"Successfully generated mock dataset at: {output_path}")

def prepare_data():
    data_dir = Path("data")
    # Search for both csv and xlsx files
    data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx"))
    
    # Exclude our output file from processing if it's there
    data_files = [f for f in data_files if f.name != "umg_tiktok_blackout_weekly_streams.csv"]
    
    if not data_files:
        print("Error: No raw CSV or XLSX files found in data/ directory. We cannot prepare real data.")
        sys.exit(1)
        
    file_path = data_files[0]
    print(f"Loading raw data from: {file_path}")
    
    # Load raw data
    if file_path.suffix == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows. Columns: {list(df.columns)}")
    
    # Identify key columns based on case-insensitive names
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'artist' in col_lower:
            col_mapping[col] = 'artist_names'
        elif 'stream' in col_lower:
            col_mapping[col] = 'streams'
        elif 'track' in col_lower:
            col_mapping[col] = 'track_name'
        elif 'date' in col_lower:
            col_mapping[col] = 'week_date'
        elif 'label' in col_lower:
            col_mapping[col] = 'record_label'
            
    df = df.rename(columns=col_mapping)
    
    # Extract week_date from Source.Name if it's not present
    if 'week_date' not in df.columns and 'Source.Name' in df.columns:
        print("Extracting week_date from 'Source.Name'...")
        df['week_date'] = df['Source.Name'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        
    # Make sure we have the required columns
    required = ['artist_names', 'streams', 'week_date']
    for req in required:
        if req not in df.columns:
            print(f"Error: Required column mapping for '{req}' not found. Available columns: {list(df.columns)}")
            sys.exit(1)
            
    print("Standardized columns mapping:")
    for k, v in col_mapping.items():
        print(f"  {k} -> {v}")
        
    # Convert week_date to datetime and sort
    df['week_date'] = pd.to_datetime(df['week_date'])
    df = df.sort_values('week_date')
    
    # Parse streams to numeric
    df['streams'] = pd.to_numeric(df['streams'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Define our treated artists and donor pool
    treated_artists = ["Billie Eilish", "Noah Kahan"]
    donor_pool = [
        "Dua Lipa", 
        "Travis Scott", 
        "Ed Sheeran", 
        "Miley Cyrus", 
        "SZA", 
        "Harry Styles", 
        "Jack Harlow", 
        "Tate McRae"
    ]
    all_target_artists = treated_artists + donor_pool
    
    print("\nFiltering and aggregating streams for target artists...")
    panel_records = []
    unique_dates = sorted(df['week_date'].unique())
    print(f"Found {len(unique_dates)} unique weeks in 2024: from {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    
    for dt in unique_dates:
        week_df = df[df['week_date'] == dt]
        
        for artist in all_target_artists:
            # Match artist name case-insensitively or as substring
            artist_mask = week_df['artist_names'].astype(str).str.contains(artist, case=False, na=False)
            artist_week_df = week_df[artist_mask]
            
            # Sum streams
            total_streams = artist_week_df['streams'].sum()
            
            is_umg = 1 if artist in treated_artists else 0
            is_blackout = 1 if (dt >= pd.Timestamp('2024-02-01') and dt <= pd.Timestamp('2024-05-05')) else 0
            
            panel_records.append({
                'week_date': dt,
                'artist': artist,
                'streams': total_streams,
                'is_umg': is_umg,
                'is_blackout': is_blackout
            })
            
    panel_df = pd.DataFrame(panel_records)
    
    print("\nOverview of weekly data coverage per artist:")
    summary_data = []
    for artist in all_target_artists:
        artist_df = panel_df[panel_df['artist'] == artist]
        weeks_with_streams = (artist_df['streams'] > 0).sum()
        mean_streams = artist_df['streams'].mean()
        summary_data.append({
            'Artist': artist,
            'Weeks Present': weeks_with_streams,
            'Avg Weekly Streams': f"{mean_streams:,.0f}"
        })
    print(pd.DataFrame(summary_data).to_string(index=False))
    
    output_path = data_dir / "umg_tiktok_blackout_weekly_streams.csv"
    panel_df.to_csv(output_path, index=False)
    print(f"\nSaved processed panel dataset to: {output_path}")

if __name__ == "__main__":
    success = download_dataset()
    if success:
        prepare_data()
    else:
        generate_mock_data()
