import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

class DiffInDiff:
    """
    A helper class to fit a standard Two-Way Fixed Effects (TWFE)
    Difference-in-Differences model using statsmodels OLS.
    """
    def __init__(self):
        self.model = None
        self.results = None
        self.treatment_effect = None
        self.std_err = None
        self.p_value = None
        self.conf_int = None
        
    def fit(self, df, unit_col='artist', time_col='week_date', value_col='streams', 
            treated_col='is_umg', post_col='is_blackout'):
        """
        Fit the TWFE DiD model:
        Y_it = alpha_i + lambda_t + beta * (Treated_i * Post_t) + epsilon_it
        """
        # Ensure correct columns and types
        df_fit = df[[unit_col, time_col, value_col, treated_col, post_col]].copy()
        
        # Create interaction term (treatment indicator)
        df_fit['treatment_interaction'] = df_fit[treated_col] * df_fit[post_col]
        
        # Use string formats for categorical variables to treat them as factors in formula
        df_fit['unit_factor'] = df_fit[unit_col].astype(str)
        # Convert datetime to string or timestamp string to treat time as categorical fixed effects
        df_fit['time_factor'] = df_fit[time_col].astype(str)
        
        # Formula: value ~ C(unit) + C(time) + treatment_interaction
        formula = f"{value_col} ~ C(unit_factor) + C(time_factor) + treatment_interaction"
        
        print(f"Fitting TWFE DiD OLS: {formula}")
        self.model = smf.ols(formula, data=df_fit)
        self.results = self.model.fit()
        
        # Extract DiD coefficient (beta)
        self.treatment_effect = self.results.params['treatment_interaction']
        self.std_err = self.results.bse['treatment_interaction']
        self.p_value = self.results.pvalues['treatment_interaction']
        self.conf_int = self.results.conf_int().loc['treatment_interaction'].values
        
        return self
        
    def get_summary_stats(self):
        """Return key regression statistics as a dictionary."""
        if self.results is None:
            raise ValueError("DiD model has not been fitted yet.")
        return {
            'Coefficient (DiD)': self.treatment_effect,
            'Std Error': self.std_err,
            'p-value': self.p_value,
            '95% CI Lower': self.conf_int[0],
            '95% CI Upper': self.conf_int[1],
            'R-squared': self.results.rsquared
        }
        
    def predict_counterfactual(self, df, treated_unit, unit_col='artist', time_col='week_date', 
                               value_col='streams', treated_col='is_umg', post_col='is_blackout'):
        """
        Generate the DiD counterfactual path for the treated unit.
        Under DiD, the counterfactual for the treated unit during the blackout period is:
        Y_it^0 = Y_it - beta * (Treated_i * Post_t)
        """
        if self.treatment_effect is None:
            raise ValueError("DiD model has not been fitted yet.")
            
        # Filter for the treated unit
        treated_df = df[df[unit_col] == treated_unit].copy().sort_values(time_col)
        
        # Calculate counterfactual
        # If in blackout period, subtract beta (treatment effect)
        # If not, counterfactual equals actual
        treated_df['did_counterfactual'] = treated_df[value_col] - (
            self.treatment_effect * (treated_df[treated_col] * treated_df[post_col])
        )
        
        return treated_df
