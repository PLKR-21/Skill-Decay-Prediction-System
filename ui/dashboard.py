import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
import plotly.express as px
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
@st.cache_resource(ttl=3600)
def init_connection():
    db_url = st.secrets["connections"]["cloud_db"]["url"]
    # pool_pre_ping=True automatically tests connections & handles dropped transactions
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)

engine = init_connection()

# 3. Fetch Data from TiDB
@st.cache_data(ttl=600) # Caches data for 10 minutes to make the app lightning fast
def load_data():
    query = "SELECT * FROM skill_features"
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

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

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["🎯 Individual Skill Terminal", "🌍 Global Market View"])
st.sidebar.divider()
st.sidebar.markdown(
    """
    ### ⚙️ System Telemetry
    🟢 **Status:** Live
    
    * **Engine:** ARIMA Forecasting
    * **Scope:** 109 Industry Technologies
    * **Sectors:** 10 Architectural Domains
    * **Database:** TiDB Serverless
    """
)

if page == "🌍 Global Market View":
    st.subheader("🌍 Global Market Intelligence Dashboard")
    st.markdown("Real-world IT market analysis powered by **Stack Overflow Tag Data (2020–2024)** across 84 technologies and 7 domains.")
    st.divider()

    # ── Prep normalized columns ───────────────────────────────────────────
    max_val = max(1.0, df['Job_Demand'].max())
    df['Demand_Index'] = (df['Job_Demand'] / max_val * 100).round(1)
    df['Forecast_Index'] = (df['3_Year_Forecast'] / max_val * 100).clip(lower=0).round(1)
    df['Visual_Size'] = df['Risk_Score'].apply(lambda x: max(x, 8))

    # ── Section 1: KPI Summary ────────────────────────────────────────────
    total_skills = len(df)
    high_risk = len(df[df['Risk_Score'] >= 60])
    growing = len(df[df['Trend_Slope'] > 0])
    top_skill = df.loc[df['Job_Demand'].idxmax(), 'Skill_Name']

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔬 Skills Tracked", f"{total_skills}")
    k2.metric("📈 Growing Technologies", f"{growing} / {total_skills}")
    k3.metric("🔴 High Obsolescence Risk", f"{high_risk} / {total_skills}")
    k4.metric("👑 Most In-Demand", top_skill.upper())

    st.divider()

    # ── Section 2: Top 10 vs Bottom 10 ───────────────────────────────────
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown("#### 🚀 Top 10 Most In-Demand Skills (2024)")
        top10 = df.nlargest(10, 'Job_Demand')[['Skill_Name', 'Domain', 'Job_Demand']].copy()
        top10['Stack Overflow Questions (2024)'] = top10['Job_Demand'].apply(lambda x: f"{int(x):,}")
        top10 = top10[['Skill_Name', 'Domain', 'Stack Overflow Questions (2024)']].rename(columns={'Skill_Name': 'Skill'})
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)

    with col_bot:
        st.markdown("#### ⚠️ Top 10 Highest Obsolescence Risk Skills")
        risk10 = df.nlargest(10, 'Risk_Score')[['Skill_Name', 'Domain', 'Risk_Score']].copy()
        risk10['Risk Score'] = risk10['Risk_Score'].apply(lambda x: f"{x:.1f}%")
        risk10 = risk10[['Skill_Name', 'Domain', 'Risk Score']].rename(columns={'Skill_Name': 'Skill'})
        st.dataframe(risk10.reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── Section 3: Domain-Level Demand Bar Chart ──────────────────────────
    st.markdown("#### 📊 Average Market Demand by Domain")
    domain_df = df.groupby('Domain').agg(
        Avg_Demand=('Job_Demand', 'mean'),
        Avg_Risk=('Risk_Score', 'mean'),
        Skill_Count=('Skill_Name', 'count')
    ).reset_index().sort_values('Avg_Demand', ascending=True)

    fig_bar = px.bar(
        domain_df,
        x='Avg_Demand', y='Domain',
        orientation='h',
        color='Avg_Risk',
        color_continuous_scale='RdYlGn_r',
        text=domain_df['Avg_Demand'].apply(lambda x: f"{int(x):,}"),
        labels={'Avg_Demand': 'Avg SO Questions (2024)', 'Avg_Risk': 'Risk %'},
        template='plotly_dark',
        title='Average Stack Overflow Activity per Domain (colored by risk)'
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=400, coloraxis_showscale=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Section 4: Bubble Chart (Demand vs Forecast) ──────────────────────
    st.markdown("#### 🔵 Industry Scale Tracking Matrix — All 84 Skills")
    st.caption("X = Current demand | Y = Predicted 2029 demand | Bubble size = obsolescence risk")

    fig_bubble = px.scatter(
        df,
        x='Demand_Index', y='Forecast_Index',
        size='Visual_Size', color='Domain',
        hover_name='Skill_Name',
        hover_data={
            'Demand_Index': ':.1f',
            'Forecast_Index': ':.1f',
            'Risk_Score': ':.1f',
            'Visual_Size': False
        },
        size_max=40, template='plotly_dark'
    )
    fig_bubble.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                         line=dict(color='grey', width=1, dash='dot'))
    fig_bubble.add_annotation(x=85, y=90, text="↑ Growing above line",
                               showarrow=False, font=dict(color='lightgreen', size=11))
    fig_bubble.add_annotation(x=85, y=78, text="↓ Declining below line",
                               showarrow=False, font=dict(color='salmon', size=11))
    fig_bubble.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_bubble.update_layout(
        xaxis_title="Current Demand Index (2024)",
        yaxis_title="Predicted Demand Index (2029)",
        height=620,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.stop()


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
m1, m2, m3, m4 = st.columns(4)

# 🚨 THE FIX: Normalize data points relative to the highest demand in the market
max_demand = max(1.0, float(df['Job_Demand'].max()))
current_demand = (max(0.0, float(skill_data['Job_Demand'])) / max_demand) * 100.0
forecast = (max(0.0, float(skill_data['3_Year_Forecast'])) / max_demand) * 100.0
slope = (float(skill_data['Trend_Slope']) / max_demand) * 100.0
risk = min(100.0, max(0.0, float(skill_data['Risk_Score'])))

# --- Salary Impact Calculation (INR) ---
# Base salary range in INR: ₹6,00,000 – ₹14,00,000/year
INR_RATE = 83.5
base_salary = (80000 + (current_demand * 600)) * INR_RATE
penalty_percentage = (risk / 100.0) * 0.20  # Up to 20% salary hit for high risk
salary_impact = -1 * (base_salary * penalty_percentage)

if risk < 30 and slope > 0:
    reward_percentage = (slope / 10.0) * 0.10  # Up to 10% gain for high growth
    salary_impact = base_salary * reward_percentage

if salary_impact > 10000:
    salary_impact_str = f"+₹{int(salary_impact):,}"
elif salary_impact < -10000:
    salary_impact_str = f"-₹{abs(int(salary_impact)):,}"
else:
    salary_impact_str = "Stable"
# ---------------------------------

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

m3.metric("Obsolescence Risk", f"{risk:.2f}%", risk_status, delta_color="off")
m4.metric("Est. 3-Yr Salary Impact", f"₹{int(base_salary + salary_impact):,}", salary_impact_str, delta_color="normal")

# 8. Visual Graph & Actionable Insights
col_chart, col_insights = st.columns([2, 1])

with col_chart:
    st.markdown("**Predicted Market Trajectory**")
    
    # Calculate capped intermediate years for the chart so lines don't break the ceiling
    year1 = max(0.0, current_demand + (slope*1))
    year2 = max(0.0, current_demand + (slope*2))
    
    # Clean Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=["2026 (Current)", "2027", "2028", "2029"],
        y=[current_demand, year1, year2, forecast],
        mode='lines+markers',
        name='Demand Trend',
        line=dict(color='#00FFAA' if slope > 0 else '#FF4444', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Market Demand Index", rangemode='tozero'), # Dynamic auto-scaling Y-axis
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_insights:
    st.markdown("**💡 Strategic Career Insights**")
    st.info(f"**Domain Context:** {selected_skill} operates within the {selected_domain} sector. The estimated base salary for an expert in this technology is currently **₹{int(base_salary):,}/year**.")
    
    if risk >= 60:
        st.warning(f"⚠️ **Strategic Pivot Recommended.** This technology is showing significant market decay. You face an estimated **{salary_impact_str}** salary penalty over the next 3 years if you do not adapt. Begin transitioning your expertise to modern alternatives within this domain.")
    elif risk >= 30:
        st.info(f"🔄 **Skill Maintenance Required.** Demand is plateauing or slightly declining, reflecting a **{salary_impact_str}** salary impact pipeline. Pair this skill with a high-growth technology to remain competitive.")
    else:
        st.success(f"✅ **High-Value Asset.** This technology is experiencing stable or rapid growth, potentially adding **{salary_impact_str}** to your market value. Deepen your expertise here as it provides strong career leverage.")

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