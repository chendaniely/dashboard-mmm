"""
Synthetic Marketing Mix Modeling Data Generator for BrickCraft Studios

This script generates artificial marketing and sales data for demonstration purposes only.
All data is synthetic and created through AI for illustrative purposes.

Generates:
- Weekly time series data over 3 years (156 weeks)
- Marketing spend across 7 channels
- Sales revenue with realistic MMM characteristics
- Seasonality, trends, and external factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2021, 1, 4)  # Start on a Monday
N_WEEKS = 156  # 3 years of weekly data
OUTPUT_DIR = "data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating synthetic Marketing Mix Modeling data for BrickCraft Studios...")
print(f"Time period: {N_WEEKS} weeks starting {START_DATE.strftime('%Y-%m-%d')}")

# Generate date range
dates = [START_DATE + timedelta(weeks=i) for i in range(N_WEEKS)]
week_of_year = [d.isocalendar()[1] for d in dates]
month = [d.month for d in dates]
quarter = [(m - 1) // 3 + 1 for m in month]

# Base sales (in thousands of dollars)
base_sales = 500

# Seasonality factors (toy industry has strong Q4 seasonality)
def get_seasonality(week_num, month_num):
    """Strong Q4 seasonality for toy industry (holiday season)"""
    # Monthly seasonality
    monthly_factors = {
        1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95,  # Q1: Post-holiday low
        5: 0.95, 6: 1.00, 7: 1.00, 8: 0.95,  # Q2-Q3: Moderate
        9: 1.05, 10: 1.15, 11: 1.35, 12: 1.50  # Q4: Holiday peak
    }
    return monthly_factors.get(month_num, 1.0)

seasonality = [get_seasonality(w, m) for w, m in zip(week_of_year, month)]

# Trend (gradual growth over time)
trend = np.linspace(1.0, 1.15, N_WEEKS)

# Marketing channel configuration
# Each channel has: mean spend, std, effectiveness (ROI), adstock decay, saturation point
channels = {
    'tv_spend': {
        'mean': 50, 'std': 15, 'effectiveness': 2.5, 'adstock': 0.5, 'saturation': 100
    },
    'radio_spend': {
        'mean': 20, 'std': 8, 'effectiveness': 1.8, 'adstock': 0.3, 'saturation': 40
    },
    'print_spend': {
        'mean': 15, 'std': 6, 'effectiveness': 1.2, 'adstock': 0.6, 'saturation': 30
    },
    'paid_search_spend': {
        'mean': 40, 'std': 12, 'effectiveness': 3.0, 'adstock': 0.1, 'saturation': 80
    },
    'display_ads_spend': {
        'mean': 35, 'std': 10, 'effectiveness': 2.0, 'adstock': 0.2, 'saturation': 70
    },
    'social_media_spend': {
        'mean': 30, 'std': 10, 'effectiveness': 2.8, 'adstock': 0.15, 'saturation': 60
    },
    'email_spend': {
        'mean': 10, 'std': 4, 'effectiveness': 3.5, 'adstock': 0.05, 'saturation': 25
    }
}

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

# Generate marketing spend data
marketing_data = {}
for channel, config in channels.items():
    # Generate base spend with some randomness
    base_spend = np.random.gamma(
        shape=(config['mean'] / config['std']) ** 2,
        scale=config['std'] ** 2 / config['mean'],
        size=N_WEEKS
    )
    
    # Add seasonality to spending (increase during holiday season)
    seasonal_multiplier = [1.0 + 0.3 * (s - 1.0) for s in seasonality]
    spend = base_spend * seasonal_multiplier
    
    marketing_data[channel] = spend

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'week': list(range(1, N_WEEKS + 1)),
    'year': [d.year for d in dates],
    'quarter': quarter,
    'month': month,
    'week_of_year': week_of_year,
    **marketing_data
})

# Calculate sales contribution from each channel
sales_contributions = {}
for channel, config in channels.items():
    # Apply adstock
    adstocked = apply_adstock(df[channel].values, config['adstock'])
    
    # Apply saturation
    saturated = apply_saturation(adstocked, config['saturation'])
    
    # Calculate contribution
    contribution = saturated * config['effectiveness']
    sales_contributions[f'{channel}_contribution'] = contribution
    df[f'{channel}_contribution'] = contribution

# Calculate total sales
total_contribution = sum(sales_contributions.values())
base_sales_vector = base_sales * np.array(seasonality) * trend

# Add external factors
promotions = np.random.binomial(1, 0.15, N_WEEKS)  # 15% of weeks have promotions
promotion_lift = promotions * np.random.uniform(50, 150, N_WEEKS)

# Add competitor activity (negative impact)
competitor_activity = np.random.binomial(1, 0.10, N_WEEKS)
competitor_impact = competitor_activity * np.random.uniform(-80, -30, N_WEEKS)

# Calculate final sales
df['base_sales'] = base_sales_vector
df['promotion'] = promotions
df['promotion_lift'] = promotion_lift
df['competitor_activity'] = competitor_activity
df['competitor_impact'] = competitor_impact

# Total sales with noise
noise = np.random.normal(0, 30, N_WEEKS)
df['sales_revenue'] = (
    base_sales_vector + 
    total_contribution + 
    promotion_lift + 
    competitor_impact + 
    noise
)

# Ensure no negative sales
df['sales_revenue'] = df['sales_revenue'].clip(lower=50)

# Add derived metrics
df['total_marketing_spend'] = df[[col for col in df.columns if col.endswith('_spend')]].sum(axis=1)
df['total_marketing_contribution'] = df[[col for col in df.columns if col.endswith('_contribution')]].sum(axis=1)

# Calculate week-over-week changes
df['sales_wow_change'] = df['sales_revenue'].pct_change()
df['spend_wow_change'] = df['total_marketing_spend'].pct_change()

# Add holiday indicators
def is_holiday_week(date):
    """Check if week contains major holidays"""
    month, week = date.month, date.isocalendar()[1]
    # Approximate holiday weeks
    if month == 12 and week >= 48:  # Christmas
        return 1
    elif month == 11 and week >= 45 and week <= 47:  # Thanksgiving
        return 1
    elif month == 7 and week >= 27 and week <= 28:  # July 4th
        return 1
    elif month == 1 and week <= 2:  # New Year
        return 1
    return 0

df['holiday_week'] = [is_holiday_week(d) for d in dates]

# Round numeric columns for readability
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['week', 'year', 'quarter', 'month', 'week_of_year', 'promotion', 'competitor_activity', 'holiday_week']:
        df[col] = df[col].round(2)

# Save main dataset
output_file = os.path.join(OUTPUT_DIR, "synthetic-marketing-sales.csv")
df.to_csv(output_file, index=False)
print(f"\nGenerated {len(df)} weeks of data")
print(f"Saved to: {output_file}")

# Generate summary statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"\nSales Revenue Statistics (in thousands):")
print(f"  Mean: ${df['sales_revenue'].mean():.2f}k")
print(f"  Min:  ${df['sales_revenue'].min():.2f}k")
print(f"  Max:  ${df['sales_revenue'].max():.2f}k")
print(f"  Std:  ${df['sales_revenue'].std():.2f}k")

print(f"\nTotal Marketing Spend Statistics (in thousands):")
print(f"  Mean: ${df['total_marketing_spend'].mean():.2f}k")
print(f"  Min:  ${df['total_marketing_spend'].min():.2f}k")
print(f"  Max:  ${df['total_marketing_spend'].max():.2f}k")

print(f"\nMarketing Channel Average Weekly Spend (in thousands):")
for channel in channels.keys():
    print(f"  {channel.replace('_', ' ').title()}: ${df[channel].mean():.2f}k")

print(f"\nExternal Factors:")
print(f"  Promotion weeks: {df['promotion'].sum()} ({df['promotion'].sum()/len(df)*100:.1f}%)")
print(f"  Competitor activity weeks: {df['competitor_activity'].sum()} ({df['competitor_activity'].sum()/len(df)*100:.1f}%)")
print(f"  Holiday weeks: {df['holiday_week'].sum()} ({df['holiday_week'].sum()/len(df)*100:.1f}%)")

print("\n" + "="*60)
print("✓ Synthetic data generation complete!")
print("="*60)

# Create a separate file with channel metadata
channel_metadata = pd.DataFrame([
    {
        'channel': channel.replace('_spend', ''),
        'channel_type': 'traditional' if channel in ['tv_spend', 'radio_spend', 'print_spend'] else 'digital',
        'effectiveness': config['effectiveness'],
        'adstock_rate': config['adstock'],
        'saturation_point': config['saturation'],
        'avg_weekly_spend': df[channel].mean()
    }
    for channel, config in channels.items()
])

metadata_file = os.path.join(OUTPUT_DIR, "synthetic-channel-metadata.csv")
channel_metadata.to_csv(metadata_file, index=False)
print(f"\nChannel metadata saved to: {metadata_file}")
