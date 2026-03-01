import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pytrends.request import TrendReq

# --- CONFIGURATION ---
st.set_page_config(page_title="Skill Decay Intelligence", layout="wide", page_icon="📈")
st.title("🛡️ Skill Decay Prediction System")
st.markdown("Predictive analytics for technology obsolescence and career intelligence.")

# --- LOAD DATA (CLOUD) & MODEL (LOCAL) ---
@st.cache_data(ttl=600)
def load_data():
    try:
        # Connects to TiDB using your .streamlit/secrets.toml
        conn = st.connection("cloud_db", type="sql")
        df = conn.query("SELECT * FROM skill_features")
        return df
    except Exception as e:
        st.error(f"Cloud Connection Failed: {e}")
        # Local fallback for offline development
        return pd.read_csv('data/engineered_features.csv')

@st.cache_resource
def load_model():
    return joblib.load('models/best_forecast_model.pkl')

df = load_data()
model = load_model()

# --- SIDEBAR: USER INPUT ---
st.sidebar.header("🔍 Skill Analysis")
skill_list = sorted(df['Skill'].unique())
selected_skill = st.sidebar.selectbox("Select Technology to Analyze", skill_list)

# --- GET SKILL DATA ---
skill_data = df[df['Skill'] == selected_skill].sort_values('Year')
latest_data = skill_data.iloc[-1]

# --- FORECASTING MODULE (3-YEAR) ---
future_years = [2026, 2027, 2028]
predictions = []

for year in future_years:
    future_features = pd.DataFrame([{
        'Year': year,
        'Survey_Usage': max(0, latest_data['Survey_Usage'] * (1 + latest_data['YoY_Growth'])),
        'Search_Index': max(0, latest_data['Search_Index'] * (1 + latest_data['YoY_Growth'])),
        'Adoption_Rate': latest_data['Adoption_Rate'],
        'YoY_Growth': latest_data['YoY_Growth'],
        'Trend_Slope': latest_data['Trend_Slope'],
        'Demand_Volatility': latest_data['Demand_Volatility'],
        'Decline_Acceleration': latest_data['Decline_Acceleration']
    }])
    pred = max(0, model.predict(future_features)[0])
    predictions.append(pred)

# --- CUSTOM RISK SCORE FORMULA (0 - 100) ---
if predictions[-1] < latest_data['Job_Demand']:
    decline_rate = (latest_data['Job_Demand'] - predictions[-1]) / (latest_data['Job_Demand'] + 1)
else:
    decline_rate = 0.0  

slope_penalty = min(1.0, abs(min(0, latest_data['Trend_Slope'])) / 5000.0)
volatility_penalty = min(1.0, latest_data['Demand_Volatility'] / 5000.0)
instability_penalty = 1.0 - latest_data['Adoption_Stability']

risk_score = (decline_rate * 40) + (slope_penalty * 30) + (volatility_penalty * 15) + (instability_penalty * 15)
risk_score = min(100.0, max(0.0, risk_score))

if risk_score <= 30:
    risk_category, color = "STABLE", "green"
elif risk_score <= 60:
    risk_category, color = "MODERATE RISK", "orange"
else:
    risk_category, color = "HIGH RISK", "red"

# --- UI VISUALIZATION ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Job Demand (2025)", f"{int(latest_data['Job_Demand']):,}")
col2.metric("3-Year Forecast (2028)", f"{int(predictions[-1]):,}", 
            delta=f"{int(predictions[-1] - latest_data['Job_Demand']):,}", delta_color="normal")
col3.metric("Skill Decay Risk Score", f"{risk_score:.1f}%", delta=risk_category, delta_color="inverse" if risk_score > 30 else "normal")

st.divider()

# --- HISTORICAL & AI FORECAST GRAPH ---
st.subheader(f"📈 10-Year Trajectory & XGBoost Forecast: {selected_skill}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=skill_data['Year'], y=skill_data['Job_Demand'], 
                         mode='lines+markers', name='Historical Demand', line=dict(color='cyan', width=3)))
fig.add_trace(go.Scatter(x=[2025] + future_years, y=[latest_data['Job_Demand']] + predictions, 
                         mode='lines+markers', name='AI Forecast', line=dict(color='orange', width=3, dash='dash')))
fig.update_layout(xaxis_title="Year", yaxis_title="Job Demand Frequency", template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- LIVE REAL-WORLD DATA ENGINE (GOOGLE TRENDS) ---
st.divider()
st.subheader("🌐 Live Market Pulse (Last 30 Days)")
st.write(f"Fetching real-time global search interest for **{selected_skill}**...")

@st.cache_data(ttl=3600, show_spinner=False) 
def get_live_trends(skill_name, current_risk):
    try:
        pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.5)
        # Avoid generic terms
        search_term = skill_name + ' programming' if skill_name in ['Go', 'Rust', 'Swift', 'ActionScript'] else skill_name
        pytrends.build_payload([search_term], timeframe='today 1-m')
        data = pytrends.interest_over_time()
        
        if not data.empty:
            return data.drop(columns=['isPartial']), True
        raise ValueError("Empty data from Google")
        
    except Exception as e:
        # --- FAILSAFE: Generate realistic 30-day pulse if API is blocked ---
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
        base_interest = 60 if current_risk == "STABLE" else 30
        simulated_data = np.random.normal(0, 5, size=30).cumsum() + base_interest
        simulated_data = np.clip(simulated_data, 10, 100) 
        
        fallback_df = pd.DataFrame(simulated_data, index=dates, columns=['Search Volume'])
        return fallback_df, False

live_data, is_real = get_live_trends(selected_skill, risk_category)

if not is_real:
    st.warning("⚠️ Google Trends API rate-limited. Displaying mathematically simulated 30-day pulse based on risk profile.")

if live_data is not None:
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(x=live_data.index, y=live_data.iloc[:, 0], 
                             mode='lines', name='Search Volume', line=dict(color='#00ffcc', width=2, shape='spline')))
    fig_live.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
    st.plotly_chart(fig_live, use_container_width=True)

# --- RECOMMENDATION ENGINE ---
st.divider()
st.subheader("💡 Intelligent Career Guidance")

tech_clusters = {
    'jQuery': ['React', 'Vue.js', 'Next.js'], 'PHP': ['Node.js', 'Go', 'Python'],
    'Ruby': ['Python', 'Go', 'Rust'], 'Objective-C': ['Swift', 'Flutter', 'Kotlin'],
    'Perl': ['Python', 'Go'], 'AngularJS': ['Angular', 'React', 'Vue.js'],
    'COBOL': ['Java', 'Python', 'Cloud Architecture'], 'VBA': ['Python (Pandas)', 'JavaScript']
}

if risk_category in ["MODERATE RISK", "HIGH RISK"]:
    st.error(f"**Warning:** {selected_skill} is exhibiting mathematical signals of market obsolescence.")
    alternatives = tech_clusters.get(selected_skill, ['Cloud Architecture (AWS/Azure)', 'AI/ML Engineering', 'Go / Rust'])
    st.info(f"**Strategic Upskilling Recommended:** Consider pivoting your learning path toward highly demanded adjacent skills such as: **{', '.join(alternatives)}**.")
else:
    st.success(f"**Favorable Outlook:** {selected_skill} shows robust market stability.")
    st.info("Recommendation: Deepen your expertise in advanced design patterns, performance optimization, or integrate AI APIs within this ecosystem.")

st.caption("System architecture integrates XGBoost modeling, Cloud MySQL (TiDB), and Live API extrapolation.")