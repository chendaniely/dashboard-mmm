"""
Model Utilities for Marketing Mix Modeling

Helper functions for loading models, making predictions, and calculating contributions.
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

class MMMModel:
    """Marketing Mix Model wrapper for predictions and analysis"""
    
    def __init__(self, model_dir='ml'):
        """Load model artifacts"""
        self.model_dir = Path(model_dir)
        
        # Load model
        self.model = joblib.load(self.model_dir / 'mmm_model.joblib')
        
        # Load scaler
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')
        
        # Load metadata
        with open(self.model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_cols']
        self.feature_names = self.metadata['feature_names']
        self.spend_cols = self.metadata['spend_cols']
    
    def predict(self, X):
        """Make sales predictions"""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_features(self, **kwargs):
        """
        Make prediction from individual feature values
        
        Example:
            model.predict_with_features(
                tv_spend=50,
                radio_spend=20,
                ...,
                promotion=0,
                competitor_activity=0,
                holiday_week=0
            )
        """
        # Create feature vector
        features = {col: kwargs.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([features])
        return self.predict(X)[0]
    
    def calculate_channel_contributions(self, X):
        """Calculate individual channel contributions to predicted sales"""
        if isinstance(X, pd.DataFrame):
            X_values = X[self.feature_cols].values
        else:
            X_values = X
        
        X_scaled = self.scaler.transform(X_values)
        
        # Calculate contribution for each feature
        contributions = {}
        for i, col in enumerate(self.feature_cols):
            # Contribution = coefficient * scaled_value
            contrib = self.model.coef_[i] * X_scaled[:, i]
            contributions[col] = contrib
        
        # Base contribution (intercept)
        contributions['base'] = np.full(len(X_scaled), self.model.intercept_)
        
        return pd.DataFrame(contributions)
    
    def get_feature_importance(self):
        """Get feature importance based on coefficients"""
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return importance
    
    def get_metrics(self):
        """Get model performance metrics"""
        return self.metadata['metrics']
    
    def optimize_budget(self, total_budget, min_spend_pct=0.05, max_spend_pct=0.40):
        """
        Suggest optimal budget allocation across channels
        
        Args:
            total_budget: Total marketing budget to allocate
            min_spend_pct: Minimum percentage for any channel (default 5%)
            max_spend_pct: Maximum percentage for any channel (default 40%)
        
        Returns:
            DataFrame with recommended spend by channel
        """
        # Get channel coefficients (positive ones only)
        channel_importance = []
        for i, col in enumerate(self.spend_cols):
            if self.model.coef_[i] > 0:
                channel_importance.append({
                    'channel': col,
                    'coefficient': self.model.coef_[i],
                    'feature_idx': i
                })
        
        importance_df = pd.DataFrame(channel_importance)
        
        # Normalize coefficients to get allocation weights
        total_coef = importance_df['coefficient'].sum()
        importance_df['weight'] = importance_df['coefficient'] / total_coef
        
        # Apply constraints
        importance_df['weight'] = importance_df['weight'].clip(min_spend_pct, max_spend_pct)
        
        # Renormalize after clipping
        importance_df['weight'] = importance_df['weight'] / importance_df['weight'].sum()
        
        # Calculate recommended spend
        importance_df['recommended_spend'] = importance_df['weight'] * total_budget
        
        # Clean up column names
        importance_df['channel'] = importance_df['channel'].str.replace('_spend', '').str.replace('_', ' ').str.title()
        
        result = importance_df[['channel', 'weight', 'recommended_spend']].copy()
        result.columns = ['Channel', 'Budget %', 'Recommended Spend ($K)']
        result['Budget %'] = result['Budget %'] * 100
        
        return result.sort_values('Recommended Spend ($K)', ascending=False)


def apply_adstock(spend, decay_rate):
    """Apply adstock transformation to capture carryover effects"""
    adstocked = np.zeros(len(spend))
    adstocked[0] = spend[0]
    for i in range(1, len(spend)):
        adstocked[i] = spend[i] + decay_rate * adstocked[i-1]
    return adstocked


def apply_saturation(spend, saturation_point):
    """Apply saturation curve (diminishing returns)"""
    return saturation_point * (1 - np.exp(-spend / saturation_point))


def calculate_roi(contribution, spend):
    """Calculate ROI for a channel"""
    return contribution / spend if spend > 0 else 0


def calculate_roas(revenue, spend):
    """Calculate Return on Ad Spend"""
    return revenue / spend if spend > 0 else 0
