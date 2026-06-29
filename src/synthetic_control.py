import numpy as np
import pandas as pd
from scipy.optimize import minimize

class SyntheticControl:
    """
    A custom implementation of the Synthetic Control Method (SCM)
    using scipy.optimize.minimize to estimate optimal donor weights.
    """
    def __init__(self):
        self.weights = None
        self.donor_names = None
        self.treated_name = None
        self.loss = None
        
    def fit(self, Y_treated_pre, Y_donors_pre, donor_names=None, treated_name=None):
        """
        Fit SCM weights using pre-treatment data.
        
        Parameters:
        -----------
        Y_treated_pre : np.ndarray or pd.Series
            Pre-treatment outcome for the treated unit, shape (T_pre,)
        Y_donors_pre : np.ndarray or pd.DataFrame
            Pre-treatment outcomes for the donor pool, shape (T_pre, J)
        donor_names : list of str, optional
            Names of the donor units.
        treated_name : str, optional
            Name of the treated unit.
        """
        # Convert inputs to numpy arrays
        y1 = np.array(Y_treated_pre, dtype=float)
        y0 = np.array(Y_donors_pre, dtype=float)
        
        T_pre, J = y0.shape
        assert len(y1) == T_pre, f"Length mismatch: treated pre-treatment has {len(y1)} periods, but donors have {T_pre} periods."
        
        self.donor_names = list(donor_names) if donor_names is not None else [f"Donor_{i}" for i in range(J)]
        self.treated_name = treated_name if treated_name is not None else "Treated"
        
        # Scale for numerical stability (normalize by treated unit's mean)
        scale_factor = np.mean(y1) if np.mean(y1) != 0 else 1.0
        y1_scaled = y1 / scale_factor
        y0_scaled = y0 / scale_factor
        
        # Objective function: Mean Squared Prediction Error (MSPE) on scaled data
        def objective(w):
            prediction = y0_scaled @ w
            return np.mean((y1_scaled - prediction) ** 2)
            
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Bounds: Non-negativity constraint (w_j >= 0) and <= 1
        bounds = [(0.0, 1.0) for _ in range(J)]
        
        # Initial guess: equal weights
        w0 = np.ones(J) / J
        
        # Run optimization
        res = minimize(
            fun=objective,
            x0=w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        
        if not res.success:
            print(f"Warning: SCM optimization did not converge: {res.message}")
            
        self.weights = res.x
        # Store loss in original scale
        self.loss = res.fun * (scale_factor ** 2)
        
        return self
        
    def get_weights(self):
        """Return the fitted weights as a pandas Series."""
        if self.weights is None:
            raise ValueError("SCM model has not been fitted yet.")
        return pd.Series(self.weights, index=self.donor_names, name="Weight")
        
    def predict(self, Y_donors):
        """
        Generate the synthetic counterfactual path using fitted weights.
        
        Parameters:
        -----------
        Y_donors : np.ndarray or pd.DataFrame
            Outcomes for the donor pool across all periods, shape (T, J)
        """
        if self.weights is None:
            raise ValueError("SCM model has not been fitted yet.")
        y0 = np.array(Y_donors, dtype=float)
        return y0 @ self.weights

    @staticmethod
    def run_placebos(df, time_col, unit_col, value_col, treated_unit, pre_treatment_end):
        """
        Run in-space placebos (permutation tests) by sequentially treating each
        donor unit as the treated unit and the rest as the donor pool.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Long-format panel dataset.
        time_col : str
            Name of the time column.
        unit_col : str
            Name of the unit/artist column.
        value_col : str
            Name of the outcome/streams column.
        treated_unit : str
            Name of the actual treated unit.
        pre_treatment_end : pd.Timestamp or str
            Last date of the pre-treatment period.
            
        Returns:
        --------
        dict
            A dictionary mapping each unit to its placebo gap (Actual - Synthetic)
            and pre-treatment loss.
        """
        units = df[unit_col].unique()
        placebo_results = {}
        
        # Pivot df to wide format for easy manipulation
        # Rows: Time, Columns: Units, Values: Outcome
        wide_df = df.pivot(index=time_col, columns=unit_col, values=value_col).sort_index()
        
        pre_mask = wide_df.index <= pd.Timestamp(pre_treatment_end)
        
        for u in units:
            # Treated unit for this run is u
            y1_pre = wide_df.loc[pre_mask, u]
            
            # Donor pool consists of all other units
            donors = [col for col in units if col != u]
            y0_pre = wide_df.loc[pre_mask, donors]
            
            # Skip if either has no variance or too many NaNs
            if y1_pre.isna().sum() > 0 or y0_pre.isna().sum().sum() > 0:
                continue
                
            # Fit SCM
            scm = SyntheticControl()
            try:
                scm.fit(y1_pre, y0_pre, donor_names=donors, treated_name=u)
                
                # Predict for all periods
                y0_all = wide_df[donors]
                synthetic = scm.predict(y0_all)
                
                actual = wide_df[u].values
                gap = actual - synthetic
                
                placebo_results[u] = {
                    'actual': actual,
                    'synthetic': synthetic,
                    'gap': gap,
                    'pre_mspe': scm.loss,
                    'weights': scm.get_weights()
                }
            except Exception as e:
                print(f"Error running placebo for {u}: {e}")
                
        return placebo_results
