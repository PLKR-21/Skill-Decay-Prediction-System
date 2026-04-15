import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# Define realistic metadata domains for our 5 models
tech_domains = {
    'python': 'Languages', 'javascript': 'Languages', 'java': 'Languages', 'c++': 'Languages', 'c#': 'Languages',
    'rust': 'Languages', 'go': 'Languages', 'typescript': 'Languages', 'swift': 'Languages', 'kotlin': 'Languages',
    'ruby': 'Languages', 'php': 'Languages', 'scala': 'Languages', 'r': 'Languages', 'dart': 'Languages', 'julia': 'Languages',
    'objective-c': 'Languages', 'perl': 'Languages', 'lua': 'Languages', 'bash': 'Languages',
    'reactjs': 'Frontend', 'angular': 'Frontend', 'vue.js': 'Frontend', 'svelte': 'Frontend', 'next.js': 'Frontend',
    'html': 'Frontend', 'css': 'Frontend', 'tailwind-css': 'Frontend', 'bootstrap': 'Frontend', 'jquery': 'Frontend',
    'node.js': 'Backend', 'django': 'Backend', 'flask': 'Backend', 'spring': 'Backend', 'laravel': 'Backend',
    'fastapi': 'Backend', 'express': 'Backend', 'graphql': 'Backend', 'asp.net': 'Backend',
    'react-native': 'Mobile', 'flutter': 'Mobile', 'android': 'Mobile', 'ios': 'Mobile', 'xamarin': 'Mobile', 'ionic': 'Mobile',
    'mysql': 'Databases', 'postgresql': 'Databases', 'mongodb': 'Databases', 'redis': 'Databases', 'sqlite': 'Databases',
    'elasticsearch': 'Databases', 'oracle': 'Databases', 'cassandra': 'Databases', 'dynamodb': 'Databases', 'firebase': 'Databases',
    'aws': 'Cloud & DevOps', 'azure': 'Cloud & DevOps', 'google-cloud-platform': 'Cloud & DevOps', 'docker': 'Cloud & DevOps',
    'kubernetes': 'Cloud & DevOps', 'terraform': 'Cloud & DevOps', 'linux': 'Cloud & DevOps', 'git': 'Cloud & DevOps',
    'jenkins': 'Cloud & DevOps', 'github-actions': 'Cloud & DevOps', 'ansible': 'Cloud & DevOps', 'nginx': 'Cloud & DevOps',
    'pandas': 'Data & AI', 'numpy': 'Data & AI', 'tensorflow': 'Data & AI', 'pytorch': 'Data & AI', 'scikit-learn': 'Data & AI',
    'hadoop': 'Data & AI', 'apache-spark': 'Data & AI', 'apache-kafka': 'Data & AI', 'snowflake': 'Data & AI', 'airflow': 'Data & AI',
    'cypress': 'Testing & Tools', 'selenium': 'Testing & Tools', 'jest': 'Testing & Tools', 'pytest': 'Testing & Tools', 
    'mocha': 'Testing & Tools', 'figma': 'Testing & Tools', 'jira': 'Testing & Tools',
    # Cloud additions
    'vercel': 'Cloud & DevOps', 'netlify': 'Cloud & DevOps', 'pulumi': 'Cloud & DevOps', 'cloudflare': 'Cloud & DevOps',
    # Database additions
    'supabase': 'Databases', 'prisma': 'Databases', 'neo4j': 'Databases', 'clickhouse': 'Databases',
    # Cybersecurity
    'kali-linux': 'Cybersecurity', 'burp-suite': 'Cybersecurity', 'wireshark': 'Cybersecurity',
    'splunk': 'Cybersecurity', 'ethical-hacking': 'Cybersecurity',
    # Generative AI & LLMs
    'langchain': 'Generative AI & LLMs', 'openai-api': 'Generative AI & LLMs', 'hugging-face': 'Generative AI & LLMs',
    'llama': 'Generative AI & LLMs', 'stable-diffusion': 'Generative AI & LLMs', 'prompt-engineering': 'Generative AI & LLMs',
    # Web3 & Blockchain
    'solidity': 'Web3 & Blockchain', 'web3.js': 'Web3 & Blockchain', 'hardhat': 'Web3 & Blockchain',
    # Game Dev
    'unity': 'Game Dev', 'unreal-engine': 'Game Dev', 'godot': 'Game Dev'
}

def generate_production_data():
    print("Booting XGBoost Production Pipeline...")
    
    if not os.path.exists('data/unified_dataset.csv'):
        print("Error: Need historical data first! Run master_ingestion.py")
        return

    df = pd.read_csv('data/unified_dataset.csv')
    skills = df['Skill'].unique()
    
    production_data = []

    for skill in skills:
        print(f"Running XGBoost forecast for: {skill.upper()}")
        
        # Get historical 5-year data
        skill_df = df[df['Skill'] == skill].sort_values('Year')
        ts_values = skill_df['Job_Demand'].values
        
        if len(ts_values) == 0:
            continue
            
        current_demand = ts_values[-1] # Usually 2024 value
        
        # We use Year index for XGBoost (e.g., 1, 2, 3, 4, 5)
        # Using numpy ranges based on the available data size to be safe
        n_points = len(ts_values)
        X_train = np.arange(1, n_points + 1).reshape(-1, 1)
        y_train = ts_values
        
        # Forecast 3 years into the future
        X_test = np.arange(n_points + 1, n_points + 4).reshape(-1, 1)
        
        try:
            # booster='gblinear' operates like a regularized linear model, allowing extrapolation for future time steps!
            model = xgb.XGBRegressor(booster='gblinear', objective='reg:squarederror')
            model.fit(X_train, y_train)
            
            future_forecasts = model.predict(X_test)
            future_forecast = future_forecasts[-1] # Forecast for Year 8 (3 years out)
            
        except Exception as e:
            print(f"Fallback for {skill}. XGBoost failed. Error: {e}")
            future_forecast = current_demand
            
        # UI Metrics Calculations
        # 1. Trend Slope (Average growth per year over the next 3 years)
        trend_slope = (future_forecast - current_demand) / 3.0
        
        # 2. Risk Score (0 to 100%)
        # If demand is falling, risk is the % drop. If stable/growing, risk is low.
        if future_forecast < current_demand and current_demand > 0:
            drop_percentage = ((current_demand - future_forecast) / current_demand) * 100.0
            risk_score = min(100.0, drop_percentage * 1.5) # Add a 1.5x multiplier to make decays more obvious
        else:
            risk_score = 5.0 # Baseline low risk for stable/growing tech
            
        # Provide domain metadata for UI filtering
        domain = tech_domains.get(skill.lower(), "General Tech")
        
        production_data.append({
            "Skill_Name": skill,
            "Domain": domain,
            "Job_Demand": float(current_demand),
            "Trend_Slope": float(trend_slope), # Linear slope extracted from gblinear
            "3_Year_Forecast": float(max(0, future_forecast)), # Prevent negative forecast
            "Risk_Score": float(risk_score)
        })

    prod_df = pd.DataFrame(production_data)
    
    # Save the golden production dataset
    prod_df.to_csv('data/production_forecast.csv', index=False)
    print("\nSuccessfully generated XGBoost forecasts!")
    print("Saved to: data/production_forecast.csv")
    print(prod_df.head())

if __name__ == "__main__":
    generate_production_data()
