import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
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
    'mocha': 'Testing & Tools', 'figma': 'Testing & Tools', 'jira': 'Testing & Tools'
}

def generate_production_data():
    print("Booting ARIMA Production Pipeline...")
    
    if not os.path.exists('data/unified_dataset.csv'):
        print("Error: Need historical data first!")
        return

    df = pd.read_csv('data/unified_dataset.csv')
    skills = df['Skill'].unique()
    
    production_data = []

    for skill in skills:
        print(f"Running ARIMA forecast for: {skill.upper()}")
        
        # Get historical 5-year data
        skill_df = df[df['Skill'] == skill].sort_values('Year')
        ts_values = skill_df['Job_Demand'].values
        current_demand = ts_values[-1] # 2025 Value
        
        # Fit ARIMA on all 5 years
        try:
            model = ARIMA(ts_values, order=(1, 1, 0))
            fit_model = model.fit()
            
            # Forecast 3 years into the future (2026, 2027, 2028)
            forecasts = fit_model.forecast(steps=3)
            future_forecast = forecasts[-1] # 2028 Value
            
        except Exception as e:
            print(f"Fallback for {skill}. ARIMA failed.")
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
            "Trend_Slope": float(trend_slope),
            "3_Year_Forecast": float(max(0, future_forecast)), # Prevent negative forecast
            "Risk_Score": float(risk_score)
        })

    prod_df = pd.DataFrame(production_data)
    
    # Save the golden production dataset
    prod_df.to_csv('data/production_forecast.csv', index=False)
    print("\nSuccessfully generated ARIMA forecasts!")
    print("Saved to: data/production_forecast.csv")
    print(prod_df.head())

if __name__ == "__main__":
    generate_production_data()
