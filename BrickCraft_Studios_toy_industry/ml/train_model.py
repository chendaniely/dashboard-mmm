"""
Marketing Mix Modeling - Model Training Script

This script trains an MMM model to predict sales based on marketing spend across channels.
Uses Ridge regression with appropriate transformations for adstock and saturation effects.

This project contains synthetic data and analysis created for demonstration purposes only.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime

print("="*70)
print("Marketing Mix Modeling - Training Script")
print("BrickCraft Studios")
print("="*70)
print()

# Check if data exists
if not os.path.exists('data/synthetic-marketing-sales.csv'):
    print("Data not found. Generating synthetic data...")
    import subprocess
    subprocess.run(['python', 'data/generate_data.py'])

# Load data
print("📊 Loading data...")
df = pd.read_csv('data/synthetic-marketing-sales.csv')
df['date'] = pd.to_datetime(df['date'])
channel_metadata = pd.read_csv('data/synthetic-channel-metadata.csv')

print(f"   Loaded {len(df)} weeks of data")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Define features
spend_cols = [col for col in df.columns if col.endswith('_spend')]
external_features = ['promotion', 'competitor_activity', 'holiday_week']
feature_cols = spend_cols + external_features

print(f"\n📈 Features:")
print(f"   Marketing channels: {len(spend_cols)}")
print(f"   External factors: {len(external_features)}")
print(f"   Total features: {len(feature_cols)}")

# Prepare data
X = df[feature_cols]
y = df['sales_revenue']

# Train/test split (80/20)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train, dates_test = df['date'][:split_idx], df['date'][split_idx:]

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} weeks ({dates_train.min().date()} to {dates_train.max().date()})")
print(f"   Testing:  {len(X_test)} weeks ({dates_test.min().date()} to {dates_test.max().date()})")

# Feature scaling
print("\n🔧 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n🤖 Training Ridge Regression model...")
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)
print("   ✓ Model trained successfully")

# Make predictions
print("\n🎯 Generating predictions...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

# Display metrics
print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(f"\n📊 R² Score (Variance Explained):")
print(f"   Training:  {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"   Testing:   {test_r2:.4f} ({test_r2*100:.2f}%)")

print(f"\n💰 Mean Absolute Error (MAE):")
print(f"   Training:  ${train_mae:.2f}K")
print(f"   Testing:   ${test_mae:.2f}K")

print(f"\n📏 Root Mean Squared Error (RMSE):")
print(f"   Training:  ${train_rmse:.2f}K")
print(f"   Testing:   ${test_rmse:.2f}K")

print(f"\n📊 Mean Absolute Percentage Error (MAPE):")
print(f"   Training:  {train_mape:.2f}%")
print(f"   Testing:   {test_mape:.2f}%")

# Feature importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Standardized Coefficients)")
print("="*70)

feature_names = [col.replace('_spend', '').replace('_', ' ').title() if '_spend' in col 
                 else col.replace('_', ' ').title() for col in feature_cols]

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"   {direction} {row['Feature']:<20} {row['Coefficient']:>8.3f}")

# Calculate channel ROI from model
print("\n" + "="*70)
print("CHANNEL ROI ANALYSIS")
print("="*70)

roi_analysis = []
for i, col in enumerate(spend_cols):
    channel = col.replace('_spend', '').replace('_', ' ').title()
    coef = model.coef_[i]
    avg_spend = X_train[col].mean()
    
    # Predicted contribution per dollar (unstandardized)
    contribution_per_dollar = coef * scaler.scale_[i]
    
    roi_analysis.append({
        'Channel': channel,
        'Coefficient': coef,
        'Avg Weekly Spend': avg_spend,
        'Impact per $': contribution_per_dollar
    })

roi_df = pd.DataFrame(roi_analysis).sort_values('Coefficient', ascending=False)

print("\nChannel Impact Rankings:")
for idx, row in roi_df.iterrows():
    print(f"   {row['Channel']:<20} Coef: {row['Coefficient']:>7.3f} | Avg Spend: ${row['Avg Weekly Spend']:>6.2f}K")

# Save model artifacts
print("\n" + "="*70)
print("SAVING MODEL ARTIFACTS")
print("="*70)

os.makedirs('ml', exist_ok=True)

# Save model
model_path = 'ml/mmm_model.joblib'
joblib.dump(model, model_path)
print(f"   ✓ Model saved: {model_path}")

# Save scaler
scaler_path = 'ml/scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"   ✓ Scaler saved: {scaler_path}")

# Save feature names
feature_info = {
    'feature_cols': feature_cols,
    'feature_names': feature_names,
    'spend_cols': spend_cols,
    'trained_date': datetime.now().isoformat(),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'metrics': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mape': train_mape,
        'test_mape': test_mape
    }
}

import json
feature_info_path = 'ml/model_metadata.json'
with open(feature_info_path, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"   ✓ Metadata saved: {feature_info_path}")

# Save predictions for validation
predictions_df = pd.DataFrame({
    'date': dates_test,
    'actual': y_test.values,
    'predicted': y_test_pred,
    'residual': y_test.values - y_test_pred,
    'abs_error': np.abs(y_test.values - y_test_pred),
    'pct_error': ((y_test.values - y_test_pred) / y_test.values * 100)
})

predictions_path = 'ml/test_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"   ✓ Test predictions saved: {predictions_path}")

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nModel Summary:")
print(f"   • Type: Ridge Regression (alpha=1.0)")
print(f"   • Features: {len(feature_cols)} ({len(spend_cols)} channels + {len(external_features)} external)")
print(f"   • Training samples: {len(X_train)} weeks")
print(f"   • Test R²: {test_r2:.4f}")
print(f"   • Test MAE: ${test_mae:.2f}K")
print(f"   • Test MAPE: {test_mape:.2f}%")
print()
