import pandas as pd
import numpy as np
import os

def engineer_features():
    print("⚙️ Booting up Feature Engineering Module...")
    
    # Load the unified dataset
    df = pd.read_csv('data/unified_dataset.csv')
    
    # Sort values to ensure chronological order for time-series math
    df = df.sort_values(by=['Skill', 'Year']).reset_index(drop=True)
    
    # 1. Year-over-Year Growth Rate (% change from previous year)
    df['YoY_Growth'] = df.groupby('Skill')['Job_Demand'].pct_change().fillna(0)
    
    # 2. Moving Average Trend (3-year rolling average)
    df['Moving_Average'] = df.groupby('Skill')['Job_Demand'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # 3. Demand Volatility (Standard deviation over a 3-year window)
    df['Demand_Volatility'] = df.groupby('Skill')['Job_Demand'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    
    # 4. Decline Acceleration Rate (Derivative of YoY Growth)
    # Negative values mean the decline is speeding up
    df['Decline_Acceleration'] = df.groupby('Skill')['YoY_Growth'].diff().fillna(0)
    
    # 5. Trend Slope (Linear regression slope over the last 3 years)
    def calc_slope(x):
        if len(x) > 1:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0
        
    df['Trend_Slope'] = df.groupby('Skill')['Job_Demand'].transform(
        lambda x: x.rolling(window=3, min_periods=2).apply(calc_slope, raw=False).fillna(0)
    )
    
    # 6. Adoption Stability Index (Inverse of Adoption Volatility)
    # Closer to 1.0 means highly stable adoption. Closer to 0 means erratic adoption.
    adoption_volatility = df.groupby('Skill')['Adoption_Rate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    df['Adoption_Stability'] = 1 / (1 + adoption_volatility)
    
    # Clean up any potential infinite values from division by zero
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Save the feature-rich dataset
    df.to_csv('data/engineered_features.csv', index=False)
    
    print("✅ Feature Engineering Pipeline completed successfully!")
    print("Dataset saved to: data/engineered_features.csv")
    print("\nPreview of New Features:")
    print(df[['Skill', 'Year', 'YoY_Growth', 'Trend_Slope', 'Demand_Volatility', 'Decline_Acceleration']].head())

if __name__ == "__main__":
    engineer_features()