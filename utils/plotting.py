import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set standard plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

# Custom premium palette
COLOR_ACTUAL = "#2D3047"       # Dark Slate Blue
COLOR_SYNTHETIC = "#FF6B6B"     # Coral Red
COLOR_DID = "#4D96FF"           # Bright Blue
COLOR_CI = "#6BCB77"            # Grass Green
COLOR_SHADE = "#F1F3F8"         # Light Gray for Shading
COLOR_TEXT = "#4A4A4A"

def plot_raw_trends(df, treated_unit, donor_pool, title="Raw Weekly Spotify Streams (2024)"):
    """Plot raw stream counts for the treated artist and donor pool."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot to wide format for easy plotting
    wide_df = df.pivot(index='week_date', columns='artist', values='streams')
    
    # Plot donors in light gray
    for col in donor_pool:
        if col in wide_df.columns:
            ax.plot(wide_df.index, wide_df[col] / 1e6, color='#BDC3C7', alpha=0.6, linewidth=1.5)
            
    # Add dummy line for legend
    ax.plot([], [], color='#BDC3C7', label='Donor Pool (Sony/Warner Artists)', alpha=0.6, linewidth=1.5)
    
    # Plot treated artist
    if treated_unit in wide_df.columns:
        ax.plot(wide_df.index, wide_df[treated_unit] / 1e6, color=COLOR_ACTUAL, 
                linewidth=3, marker='o', markersize=5, label=f"{treated_unit} (UMG - Treated)")
        
    ax.set_ylabel("Weekly Streams (Millions)")
    ax.set_xlabel("Date")
    ax.set_title(title, pad=15, fontweight='bold')
    
    # Shade treatment period
    ax.axvspan(pd.Timestamp('2024-02-01'), pd.Timestamp('2024-05-02'), 
               color=COLOR_SHADE, alpha=0.6, label='TikTok Blackout Period')
    
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    plt.tight_layout()
    return fig

def plot_counterfactual_comparison(dates, actual, scm_cf, did_cf, ci_cf=None, 
                                   blackout_start='2024-02-01', blackout_end='2024-05-02',
                                   title="Counterfactual Stream Comparisons for Billie Eilish"):
    """Plot actual streams vs. SCM, DiD, and CausalImpact counterfactuals."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    # Convert dates to pandas datetime
    dates = pd.to_datetime(dates)
    b_start = pd.Timestamp(blackout_start)
    b_end = pd.Timestamp(blackout_end)
    
    # Plot Actual
    ax.plot(dates, actual / 1e6, color=COLOR_ACTUAL, linewidth=3, marker='o', markersize=5, label='Actual Streams')
    
    # Plot SCM Counterfactual
    ax.plot(dates, scm_cf / 1e6, color=COLOR_SYNTHETIC, linewidth=2.5, linestyle='--', label='Synthetic Counterfactual (SCM)')
    
    # Plot DiD Counterfactual
    ax.plot(dates, did_cf / 1e6, color=COLOR_DID, linewidth=2, linestyle=':', label='Difference-in-Differences Counterfactual')
    
    # Plot CausalImpact Counterfactual if available
    if ci_cf is not None:
        ax.plot(dates, ci_cf / 1e6, color=COLOR_CI, linewidth=2, linestyle='-.', label='CausalImpact (BSTS) Counterfactual')
        
    ax.set_ylabel("Weekly Streams (Millions)")
    ax.set_xlabel("Date")
    ax.set_title(title, pad=15, fontweight='bold')
    
    # Shade blackout period
    ax.axvspan(b_start, b_end, color=COLOR_SHADE, alpha=0.7, label='TikTok Blackout Period')
    
    # Add vertical reference lines
    ax.axvline(b_start, color='#7F8C8D', linestyle='-', linewidth=0.8)
    ax.axvline(b_end, color='#7F8C8D', linestyle='-', linewidth=0.8)
    
    # Annotate blackout start
    ax.text(b_start - pd.Timedelta(days=4), (actual.max() / 1e6) * 0.9, 'Blackout Starts\nFeb 1, 2024', 
            color=COLOR_TEXT, ha='right', fontsize=9, fontweight='semibold')
    ax.text(b_end + pd.Timedelta(days=4), (actual.max() / 1e6) * 0.9, 'Blackout Ends\nMay 2, 2024', 
            color=COLOR_TEXT, ha='left', fontsize=9, fontweight='semibold')
            
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    plt.tight_layout()
    return fig

def plot_gaps(dates, gap_scm, gap_did, gap_ci=None, 
              blackout_start='2024-02-01', blackout_end='2024-05-02',
              title="Pointwise Causal Effect Gaps (Actual - Counterfactual)"):
    """Plot the differences/gaps between actual and counterfactuals."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    dates = pd.to_datetime(dates)
    b_start = pd.Timestamp(blackout_start)
    b_end = pd.Timestamp(blackout_end)
    
    # Plot SCM gap
    ax.plot(dates, gap_scm / 1e6, color=COLOR_SYNTHETIC, linewidth=2.5, label='SCM Gap (Actual - Synthetic)')
    
    # Plot DiD gap
    ax.plot(dates, gap_did / 1e6, color=COLOR_DID, linewidth=2, linestyle=':', label='DiD Gap (Actual - DiD)')
    
    # Plot CausalImpact gap
    if gap_ci is not None:
        ax.plot(dates, gap_ci / 1e6, color=COLOR_CI, linewidth=2, linestyle='-.', label='CausalImpact Gap')
        
    ax.axhline(0, color='#7F8C8D', linestyle='-', linewidth=1.2)
    ax.axvspan(b_start, b_end, color=COLOR_SHADE, alpha=0.7, label='TikTok Blackout Period')
    
    ax.set_ylabel("Stream Gap (Millions)")
    ax.set_xlabel("Date")
    ax.set_title(title, pad=15, fontweight='bold')
    
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    plt.tight_layout()
    return fig

def plot_placebo_test(dates, treated_gap, placebo_gaps, treated_name, pre_mspe_cutoff=None,
                      blackout_start='2024-02-01', blackout_end='2024-05-02',
                      title="SCM In-Space Placebo Permutation Test"):
    """
    Plot SCM placebo gaps for the donor pool to show statistical significance.
    Optionally filters out placebos with bad pre-treatment fits.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(dates)
    b_start = pd.Timestamp(blackout_start)
    b_end = pd.Timestamp(blackout_end)
    
    placebo_count = 0
    filtered_count = 0
    
    # Plot placebo gaps
    for unit, p_data in placebo_gaps.items():
        if unit == treated_name:
            continue
            
        # Filter by pre-treatment MSPE if cutoff provided
        if pre_mspe_cutoff is not None and p_data['pre_mspe'] > pre_mspe_cutoff:
            filtered_count += 1
            continue
            
        ax.plot(dates, p_data['gap'] / 1e6, color='#BDC3C7', alpha=0.5, linewidth=1.2)
        placebo_count += 1
        
    # Plot treated gap on top
    ax.plot(dates, treated_gap / 1e6, color=COLOR_SYNTHETIC, linewidth=3, label=f"{treated_name} (Actual Effect)")
    
    # Horizontal line at 0
    ax.axhline(0, color='#2C3E50', linestyle='-', linewidth=1.2)
    
    # Shade treatment period
    ax.axvspan(b_start, b_end, color=COLOR_SHADE, alpha=0.7, label='TikTok Blackout Period')
    
    ax.set_ylabel("Stream Gap (Millions)")
    ax.set_xlabel("Date")
    
    subtitle = f"Showing {placebo_count} placebo runs"
    if pre_mspe_cutoff is not None:
        subtitle += f" (pruned {filtered_count} with bad pre-treatment fit, MSPE > {pre_mspe_cutoff:,.2E})"
        
    ax.set_title(f"{title}\n{subtitle}", pad=15, fontweight='bold', fontsize=13)
    
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    plt.tight_layout()
    return fig
