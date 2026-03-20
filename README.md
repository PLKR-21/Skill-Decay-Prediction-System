# 🛡️ Skill Decay Prediction System

An enterprise-grade predictive analytics platform that tracks the real-world market trajectory of **109 industry technologies** across 10 architectural domains. Powered by live Stack Overflow data, ARIMA time-series forecasting, and a cloud-native TiDB Serverless database.

> **Live Demo:** Streamlit Dashboard | **Data:** Stack Overflow Tag API (2020–2024) | **Forecast:** ARIMA 3-Year Model | **Currency:** INR

---

## 🖥️ Dashboard Pages

### 🎯 Individual Skill Terminal
Search any of the 109 tracked technologies and instantly get:
- **Current Market Demand** — normalized 0-100 index based on real Stack Overflow activity
- **Predicted 3-Year Demand (2027–2029)** — ARIMA model forecast with trend direction
- **Obsolescence Risk Score** — 0-100% decay risk with visual risk classification
- **Estimated 3-Year Salary Impact** — projected salary in INR based on demand trajectory
- **Predicted Market Trajectory Chart** — interactive Plotly line chart (2026 → 2029)
- **Strategic Career Insights** — actionable guidance based on risk level
- **Live Global Search Pulse** — Google Trends integration (via SerpApi)

### 🌍 Global Market View
Macro-level industry intelligence across all 109 skills:
- **KPI Summary Cards** — total skills tracked, growing count, high-risk count, top skill
- **Top 10 Most In-Demand Skills** — ranked by real 2024 Stack Overflow question volume
- **Top 10 Highest Risk Skills** — ranked by obsolescence risk score
- **Domain Demand Bar Chart** — average market activity per architectural domain, color-coded by risk
- **Industry Scale Tracking Matrix** — interactive bubble chart (all 84 skills); above diagonal = growing, below = declining

---

## 🚀 System Architecture

| Layer | Technology |
|---|---|
| **UI** | Streamlit (Python) |
| **ML Forecasting** | ARIMA (statsmodels) |
| **Cloud Database** | TiDB Serverless (MySQL-compatible) |
| **Live Trends** | Google Trends via SerpApi |
| **Data Sources** | Stack Overflow Tag API, PyPI Stats, npm Registry |

---

## 🧠 Data Pipeline

```
Stack Overflow API (5 years)
PyPI Stats + npm Registry       → master_ingestion.py
        ↓
  unified_dataset.csv           → run_arima_pipeline.py (ARIMA)
        ↓
  production_forecast.csv       → cloud_migration.py
        ↓
  TiDB Cloud DB                 → ui/dashboard.py
```

**Run time:** ~10 minutes for all 109 skills (rate-limit safe, 0.3s delay per API call)

---

## 📊 ML Models Covered (Notebook)

| Model | R² Score | Best For |
|---|---|---|
| **ARIMA** | per-skill | Time-series 3-year forecasting |
| **XGBoost** | 0.960 | Tabular demand prediction |
| **Random Forest** | 0.957 | Robust baseline comparison |

Full analysis available in [`notebooks/skill_decay_analysis.ipynb`](notebooks/skill_decay_analysis.ipynb)

---

## 📁 Project Structure

```
Skill_Decay_Project/
├── core/
│   ├── master_ingestion.py    # Real data fetcher (SO + PyPI + npm)
│   ├── data_builder.py        # Synthetic fallback generator
│   ├── run_arima_pipeline.py  # ARIMA ML forecasting engine
│   └── cloud_migration.py    # TiDB Cloud uploader
├── notebooks/
│   └── skill_decay_analysis.ipynb  # Full ML analysis + comparison charts
├── ui/
│   └── dashboard.py           # Streamlit dashboard (2 pages)
├── models/                    # Trained .pkl model files
├── .gitignore                 # Excludes secrets, data/, decay_env/
└── requirements.txt
```

---

## ⚙️ Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/PLKR-21/Skill-Decay-Prediction-System.git
cd Skill-Decay-Prediction-System

# 2. Create and activate virtual environment
python -m venv decay_env
decay_env\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
# Create .streamlit/secrets.toml with:
# [connections.cloud_db]
# url = "mysql+pymysql://..."
# [api_keys]
# serpapi = "your_key_here"

# 5. Run the data pipeline (first time only, ~8 minutes)
python core/master_ingestion.py
python core/run_arima_pipeline.py
python core/cloud_migration.py

# 6. Launch the dashboard
streamlit run ui/dashboard.py
```

---

## 🌐 109 Tracked Technologies Across 10 Domains

| Domain | Skills |
|---|---|
| **Languages** | Python, JavaScript, Java, C++, C#, Rust, Go, TypeScript, Swift, Kotlin, Ruby, PHP, Scala, R, Dart, Julia, Perl, Lua, Bash |
| **Frontend** | ReactJS, Angular, Vue.js, Svelte, Next.js, HTML, CSS, Tailwind CSS, Bootstrap, jQuery |
| **Backend** | Node.js, Django, Flask, Spring, Laravel, FastAPI, Express, GraphQL, ASP.NET |
| **Mobile** | React Native, Flutter, Android, iOS, Xamarin, Ionic |
| **Databases** | MySQL, PostgreSQL, MongoDB, Redis, SQLite, Elasticsearch, Oracle, Cassandra, DynamoDB, Firebase |
| **Cloud & DevOps** | AWS, Azure, GCP, Docker, Kubernetes, Terraform, Linux, Git, Jenkins, GitHub Actions, Ansible, Nginx |
| **Data & AI** | Pandas, NumPy, TensorFlow, PyTorch, scikit-learn, Hadoop, Apache Spark, Kafka, Snowflake, Airflow |
| **Testing & Tools** | Cypress, Selenium, Jest, Pytest, Mocha, Figma, Jira |
