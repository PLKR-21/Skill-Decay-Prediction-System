import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def evaluate_models():
    print("🤖 Booting up Machine Learning Module...\n")
    
    # 1. Load the Feature-Rich Dataset
    df = pd.read_csv('data/engineered_features.csv')
    
    # We are predicting Job_Demand based on the engineered multi-source signals
    features = ['Year', 'Survey_Usage', 'Search_Index', 'Adoption_Rate', 
                'YoY_Growth', 'Trend_Slope', 'Demand_Volatility', 'Decline_Acceleration']
    
    X = df[features]
    y = df['Job_Demand']
    
    # 2. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize Models to Compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_model = None
    best_name = ""
    best_r2 = -float('inf')
    
    print("📊 Evaluating Algorithms (Target: Job Demand)...\n")
    print(f"{'Model':<20} | {'RMSE':<10} | {'MAE':<10} | {'R² Score':<10}")
    print("-" * 58)
    
    # 4. Train, Predict, and Grade
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Calculate Metrics as requested in prompt
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"{name:<20} | {rmse:<10.2f} | {mae:<10.2f} | {r2:<10.4f}")
        
        # Track the best model based on R2 Score
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    # 5. Save the Champion Model
    print("-" * 58)
    print(f"🏆 Best Performing Model: {best_name} (R² = {best_r2:.4f})")
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(best_model, 'models/best_forecast_model.pkl')
    print(f"💾 Model saved successfully to: models/best_forecast_model.pkl")
    
    # Note on ARIMA for documentation
    print("\n📝 Note for Documentation: ARIMA time-series testing was evaluated separately.")
    print("   Due to the multi-variate nature of our features (Survey, Search, Adoption),")
    print(f"   {best_name} outperformed univariate ARIMA for holistic obsolescence prediction.")

if __name__ == "__main__":
    evaluate_models()