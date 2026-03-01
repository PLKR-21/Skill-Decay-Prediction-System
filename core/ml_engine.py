import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split

# 1. The 100+ Skill Dictionary Organized by Domain
tech_domains = {
    "Frontend Development": ["React", "Angular", "Vue.js", "Svelte", "Next.js", "Nuxt.js", "jQuery", "Bootstrap", "Tailwind CSS", "HTML5", "CSS3", "WebAssembly", "SASS", "LESS", "Ember.js"],
    "Backend & Languages": ["Python", "Node.js", "Django", "Flask", "FastAPI", "Spring Boot", "Ruby on Rails", "Laravel", "ASP.NET Core", "Express.js", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby"],
    "Mobile Development": ["React Native", "Flutter", "Swift", "Kotlin", "Ionic", "Xamarin", "Objective-C", "Dart", "Android SDK"],
    "Databases & Data Warehousing": ["MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "Cassandra", "Oracle DB", "Microsoft SQL Server", "DynamoDB", "Firebase", "Neo4j", "Elasticsearch", "MariaDB", "Couchbase", "Snowflake"],
    "Cloud & DevOps": ["AWS", "Microsoft Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Nginx", "Apache", "Prometheus", "Grafana"],
    "AI & Data Science": ["TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", "XGBoost", "Hugging Face", "OpenCV", "Apache Spark", "Hadoop", "Kafka", "Tableau", "Power BI", "MATLAB"],
    "Cybersecurity": ["Wireshark", "Metasploit", "Kali Linux", "Nmap", "Burp Suite", "Snort", "Splunk", "OpenVPN", "OAuth", "Cryptography"],
    "Game Development": ["Unity", "Unreal Engine", "Godot", "CryEngine", "Blender"],
    "Blockchain & Web3": ["Solidity", "Web3.js", "Ethereum", "Smart Contracts", "Hyperledger"]
}

def generate_market_data():
    """Generates realistic market data and forces float values to fix the 0 forecast bug."""
    print("⚙️ Generating enterprise market data for 100+ skills...")
    data = []
    
    for domain, skills in tech_domains.items():
        for skill in skills:
            # Create realistic market behaviors based on the domain
            if domain in ["AI & Data Science", "Cloud & DevOps", "Blockchain & Web3"]:
                base_demand = np.random.uniform(70.0, 100.0)
                trend_slope = np.random.uniform(2.0, 15.0)  # Growing heavily
            elif skill in ["jQuery", "Objective-C", "PHP", "Ruby", "Apache", "Ember.js"]:
                base_demand = np.random.uniform(20.0, 50.0)
                trend_slope = np.random.uniform(-10.0, -2.0) # Declining heavily
            else:
                base_demand = np.random.uniform(40.0, 85.0)
                trend_slope = np.random.uniform(-3.0, 5.0)   # Stable / Slight growth
                
            volatility = np.random.uniform(1.0, 10.0)
            
            # The Target Variable (What XGBoost will learn to predict)
            # Forced to float to ensure decimals don't round to 0
            future_forecast = float(base_demand + (trend_slope * 3.0)) 
            future_forecast = max(0.0, future_forecast) # Can't go below 0
            
            data.append({
                "Skill_Name": skill,
                "Domain": domain,
                "Job_Demand": float(base_demand),
                "Trend_Slope": float(trend_slope),
                "Volatility": float(volatility),
                "3_Year_Forecast": float(future_forecast)
            })
            
    df = pd.DataFrame(data)
    
    # Calculate Risk Score (0 to 100%) - High risk if forecast drops heavily compared to current demand
    df['Risk_Score'] = np.where(
        df['3_Year_Forecast'] < df['Job_Demand'],
        ((df['Job_Demand'] - df['3_Year_Forecast']) / df['Job_Demand']) * 100.0 + df['Volatility'],
        df['Volatility'] * 0.5
    )
    
    # Cap Risk Score at 100 and force float
    df['Risk_Score'] = df['Risk_Score'].clip(upper=100.0).astype(float)
    
    return df

def train_xgboost(df):
    """Trains the XGBoost Regressor on the new 110+ skill dataset."""
    print("🧠 Training XGBoost AI Model...")
    
    # Features (X) and Target (y)
    X = df[['Job_Demand', 'Trend_Slope', 'Volatility']]
    y = df['3_Year_Forecast']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_forecast_model.pkl')
    print("✅ Model trained and saved to models/best_forecast_model.pkl")

if __name__ == "__main__":
    # 1. Generate Data
    skill_df = generate_market_data()
    
    # 2. Train Model
    train_xgboost(skill_df)
    
    # 3. Save to local CSV (We will use this to update TiDB next)
    os.makedirs('data', exist_ok=True)
    skill_df.to_csv('data/engineered_features.csv', index=False)
    print(f"✅ Successfully processed {len(skill_df)} skills and saved to data/engineered_features.csv")
    print("🚀 Next Step: Run your cloud_migration.py to push this new data to TiDB Cloud!")