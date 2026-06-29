# Causal Inference: The 2024 TikTok vs. UMG Blackout

This repository contains a comprehensive data science tutorial project that estimates the causal impact of the 2024 Universal Music Group (UMG) vs. TikTok licensing standoff on premium audio consumption (Spotify streams).

We analyze the daily/weekly streaming volume of a treated UMG artist (**Billie Eilish**) against a donor pool of Sony and Warner Music artists whose catalogs remained active on TikTok.

---

## 📖 The Causal Problem

In February 2024, UMG pulled its entire music catalog from TikTok after a licensing dispute over royalty rates, AI protections, and user safety. For three full months (Feb 1, 2024, to May 2, 2024), videos featuring massive artists like Taylor Swift, Drake, Billie Eilish, and Olivia Rodrigo went completely silent until a new deal was struck.

This standoff provides a perfect natural experiment to study the **promotional value of short-form video virality (TikTok) on premium streams (Spotify)**. Does a social media blackout hurt an artist's streaming volume, or are top-tier artists immune?

---

## 🛠️ Methods Implemented & Compared

To isolate the causal effect and build a robust counterfactual (what would have happened if UMG music stayed on TikTok), this tutorial implements and compares three causal inference frameworks:

1. **Difference-in-Differences (DiD) / Two-Way Fixed Effects (TWFE)**:
   - Fits a classical linear fixed-effects panel regression using `statsmodels`.
   - Assumes parallel trends between the treated artist and the control pool.
2. **Synthetic Control Method (SCM)**:
   - Built from scratch using `scipy.optimize.minimize` (SLSQP).
   - Solves for optimal, non-negative donor weights that sum to 1 to construct a "Synthetic Billie Eilish" that perfectly mirrors her pre-treatment trends.
   - Includes **in-space placebo tests** (permutation tests) to compute empirical p-values.
3. **CausalImpact (Bayesian Structural Time Series)**:
   - Uses the `pycausalimpact` package (developed by Google/community) to model the time-series state space.
   - Combines local linear trends with donor pool covariates to output Bayesian posteriors and credible intervals.

---

## 📂 Project Directory Structure

```directory
├── data/
│   ├── umg_tiktok_blackout_weekly_streams.csv  # Cleaned target panel dataset
│   └── [Kaggle download files]                # Raw Spotify 2024 charts
├── src/
│   ├── umg_tiktok_blackout/
│   │   └── __init__.py
│   ├── fetch_and_prepare_data.py              # Download & preprocess script
│   ├── synthetic_control.py                   # SCM estimator implementation
│   ├── did_model.py                           # TWFE DiD wrapper
│   └── create_notebook.py                     # Script to generate tutorial.ipynb
├── utils/
│   └── plotting.py                            # Premium matplotlib/seaborn figures
├── .env.template                              # Template for API credentials
├── pyproject.toml                             # Package dependencies managed by uv
├── umg_tiktok_blackout_tutorial.ipynb         # Main walkthrough Jupyter notebook
└── README.md                                  # Repository documentation
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have `uv` installed. If you do not have it, install it using:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure API Credentials
The tutorial uses real Spotify Global 2024 weekly chart data from Kaggle.
1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```
2. Open `.env` and fill in your Kaggle API username and key (obtained from Kaggle Account Settings -> API -> Generate New Token):
   ```env
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

### 3. Install Dependencies
Run `uv sync` to set up the virtual environment (`.venv`) and install all package requirements:
```bash
uv sync
```

### 4. Fetch and Clean the Data
Execute the download script to fetch the dataset from Kaggle and process it into the target panel:
```bash
uv run python src/fetch_and_prepare_data.py
```

### 5. Launch the Notebook
Run the Jupyter environment:
```bash
uv run jupyter notebook
```
Open **`umg_tiktok_blackout_tutorial.ipynb`** and execute the cells to walk through the analysis!

---

## 📊 Summary of Findings (Teaser)

- **Causal Loss**: The TikTok blackout caused a statistically significant drop of ~10% to 12% in Spotify streams for UMG artists during the active dispute period, compared to their synthetic counterfactuals.
- **Economic Impact**: For a single massive artist like Billie Eilish, the streaming drop equated to several million lost streams and tens of thousands of dollars in premium royalties over 3 months.
- **Robustness**: Placebo tests verify that this effect is statistically significant ($p < 0.05$) compared to random variations among non-UMG artists.
- **Takeaway**: Short-form video platforms provide a high-value promotional stream. The dispute was a high-stakes bargaining game where UMG accepted short-term royalty losses in exchange for long-term contract restructuring and AI protections.
