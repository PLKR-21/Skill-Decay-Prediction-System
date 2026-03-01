import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_skill_model():
    # 1. Load the normalized data
    if not os.path.exists('data/normalized_skills.csv'):
        print("Error: normalized_skills.csv not found!")
        return
        
    df = pd.read_csv('data/normalized_skills.csv')
    
    # 2. Prepare Features (X) and Target (y)
    # We use 2023 and 2024 stats to learn the 'pattern' leading to 2025
    # X = Input Data (Past), y = Target Data (Future)
    X = df[['so_2023', 'gh_2023', 'so_2024', 'gh_2024']]
    y = df['so_2025'] 
    
    # 3. Initialize and Train the Random Forest
    print("🧠 Training the Random Forest model on all 21 skills...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 4. Save the trained 'Brain' so the UI can use it
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/decay_model.pkl')
    
    # 5. Output Accuracy Score (R^2)
    # 1.0 is a perfect score
    score = model.score(X, y)
    
    print("\n✅ AI Model Trained and Saved!")
    print(f"Model Accuracy Score (R^2): {score:.4f}")
    print("The 'Brain' is now stored in: models/decay_model.pkl")

if __name__ == "__main__":
    train_skill_model()
    