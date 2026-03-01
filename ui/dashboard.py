import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
import numpy as np
from serpapi import GoogleSearch

# 1. Page Configuration (Makes it wide and sleek)
st.set_page_config(page_title="Skill Decay Predictor", page_icon="🛡️", layout="wide")

# Custom CSS for cleaner UI
st.markdown("""
    <style>
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# 2. Database Connection
@st.cache_resource
def init_connection():
    db_url = st.secrets["connections"]["cloud_db"]["url"]
    return create_engine(db_url)

engine = init_connection()

# 3. Fetch Data from TiDB
@st.cache_data(ttl=600) # Caches data for 10 minutes to make the app lightning fast
def load_data():
    query = "SELECT * FROM skill_features"
    return pd.read_sql(query, engine)

try:
    df = load_data()
except Exception as e:
    # THIS is the critical change. It will now print the exact Python error!
    st.error(f"🚨 System Error Details: {e}")
    st.stop()

# 4. Header Section
st.title("🛡️ Tech Obsolescence & Skill Decay Predictor")
st.markdown("An AI-driven analytics terminal tracking the 3-year trajectory of 100+ industry technologies.")
st.divider()

# 5. Domain-Wise Selection UI
st.subheader("1. Select Technology Profile")
col1, col2 = st.columns(2)

with col1:
    # Get unique domains from the database
    domains = sorted(df['Domain'].unique().tolist())
    selected_domain = st.selectbox("📂 Filter by Domain", domains)

with col2:
    # Filter skills based on the chosen domain
    domain_skills = sorted(df[df['Domain'] == selected_domain]['Skill_Name'].tolist())
    selected_skill = st.selectbox("🎯 Select Specific Skill", domain_skills)

# 6. Extract the selected skill's data
skill_data = df[df['Skill_Name'] == selected_skill].iloc[0]

st.divider()

# 7. Core Metrics Display
st.subheader(f"📊 Market Analysis: {selected_skill}")
m1, m2, m3 = st.columns(3)

# 🚨 THE FIX: Enforce strict 0 to 100 limits on all data points
current_demand = min(100.0, max(0.0, float(skill_data['Job_Demand'])))
forecast = min(100.0, max(0.0, float(skill_data['3_Year_Forecast'])))
risk = min(100.0, max(0.0, float(skill_data['Risk_Score'])))
slope = float(skill_data['Trend_Slope'])

m1.metric("Current Market Demand", f"{current_demand:.2f} / 100")

# Color code the slope direction
slope_arrow = "📈" if slope > 0 else "📉"
m2.metric("Predicted 3-Year Demand", f"{forecast:.2f} / 100", f"{slope_arrow} {slope:.2f} annual shift")

# Risk Score styling
if risk >= 60:
    risk_status = "🔴 High Risk"
elif risk >= 30:
    risk_status = "🟡 Moderate Risk"
else:
    risk_status = "🟢 Safe / Growing"

m3.metric("Obsolescence Risk Score", f"{risk:.2f}%", risk_status, delta_color="off")

# 8. Visual Graph & Actionable Insights
col_chart, col_insights = st.columns([2, 1])

with col_chart:
    st.markdown("**Predicted Market Trajectory**")
    
    # Calculate capped intermediate years for the chart so lines don't break the ceiling
    year1 = min(100.0, max(0.0, current_demand + (slope*1)))
    year2 = min(100.0, max(0.0, current_demand + (slope*2)))
    
    # Clean Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=["Current Year", "Year 1", "Year 2", "Year 3"],
        y=[current_demand, year1, year2, forecast],
        mode='lines+markers',
        name='Demand Trend',
        line=dict(color='#00FFAA' if slope > 0 else '#FF4444', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Market Demand Index", range=[0, 105]), # Locked Y-axis perfectly
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_insights:
    st.markdown("**💡 Strategic Career Insights**")
    st.info(f"**Domain Context:** {selected_skill} operates within the {selected_domain} sector.")
    
    if risk >= 60:
        st.warning("⚠️ **Strategic Pivot Recommended.** This technology is showing significant market decay. Begin transitioning your expertise to modern alternatives within this domain.")
    elif risk >= 30:
        st.info("🔄 **Skill Maintenance Required.** Demand is plateauing or slightly declining. Pair this skill with a high-growth technology to remain competitive.")
    else:
        st.success("✅ **High-Value Asset.** This technology is experiencing stable or rapid growth. Deepen your expertise here as it provides strong career leverage.")

# 9. Live Google Trends API
st.subheader(f"🌍 Live Global Market Pulse: {selected_skill}")

@st.cache_data(ttl=86400) # Caches for 24 hours to save API credits
def get_live_google_trend(skill):
    try:
        api_key = st.secrets["api_keys"]["serpapi"]
        params = {
          "engine": "google_trends",
          "q": skill,
          "data_type": "TIMESERIES",
          "api_key": api_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        interest_over_time = results.get("interest_over_time", {})
        timeline_data = interest_over_time.get("timeline_data", [])
        
        if not timeline_data:
            return None
            
        dates = [item["date"] for item in timeline_data]
        values = [item["values"][0]["extracted_value"] for item in timeline_data]
        
        return pd.DataFrame({"Date": dates, "Interest": values})
        
    except Exception as e:
        return None

with st.spinner(f"Fetching live global search data for {selected_skill}..."):
    live_trend_df = get_live_google_trend(selected_skill)

if live_trend_df is not None:
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=live_trend_df["Date"],
        y=live_trend_df["Interest"],
        mode='lines',
        name='Google Search Interest',
        line=dict(color='#0078FF', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 120, 255, 0.1)'
    ))
    fig_live.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Relative Search Volume (0-100)"),
        xaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    st.plotly_chart(fig_live, use_container_width=True)
else:
    st.warning("⚠️ Live data temporarily unavailable. Showing AI forecast above.")