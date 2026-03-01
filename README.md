# 🛡️ Skill Decay Prediction System

An enterprise-grade, predictive analytics dashboard designed to forecast technology obsolescence and provide intelligent career guidance. Built with a cloud-native architecture, this system leverages machine learning and real-time data to calculate a dynamic "Skill Decay Risk Score" for modern software tools.

## 🚀 System Architecture

* **Frontend:** Streamlit (Python)
* **Machine Learning:** XGBoost Regressor 
* **Data Layer:** TiDB Serverless (Cloud MySQL)
* **Live Integration:** Google Trends API 

## 🧠 Core Features

1. **Multi-Variate Time Series Forecasting:** Utilizes XGBoost to predict job demand trajectories up to 3 years into the future.
2. **Dynamic Risk Scoring:** Calculates a 0-100% obsolescence risk score based on trend slope penalties and market volatility.
3. **Cloud-Native Data Pipeline:** Fully decoupled architecture querying a remote TiDB SQL database.
4. **Live Market Pulse:** Integrates real-time 30-day global search interest, featuring an automated algorithmic failsafe if API rate limits are exceeded.

## 🛠️ Local Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your TiDB Cloud database connection in `.streamlit/secrets.toml`.
4. Run the application: `streamlit run ui/dashboard.py`