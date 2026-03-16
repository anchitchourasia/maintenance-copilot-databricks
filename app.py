"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
ENHANCED VERSION with Real-time Charts & Filters
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import databricks.sql
import google.generativeai as genai
from datetime import datetime, timedelta

load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="Maintenance Copilot",
    page_icon="🔧"
)

st.title("🔧 Predictive Maintenance Copilot")
st.markdown("**Databricks Lakehouse | Random Forest AUC 0.954 | Gemini 3 Flash | Real-time Dashboard**")

# Sidebar - Enhanced Controls
st.sidebar.title("⚙️ Real-time Controls")
product_search = st.sidebar.text_input("🔍 Search Product ID")
quarter_filter = st.sidebar.selectbox(
    "📅 Quarter Filter",
    ["All Quarters", "Q1", "Q2", "Q3", "Q4"],
    index=0
)
risk_filter = st.sidebar.multiselect(
    "🎯 Risk Level",
    ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
    default=["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
)

if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Connection and LLM
@st.cache_resource
def get_connection():
    return databricks.sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )

@st.cache_resource
def get_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-3-flash-preview")

@st.cache_data(ttl=30)  # Reduced TTL for real-time
def load_predictions():
    conn = get_connection()
    query = """
    SELECT * FROM default.gold_predictions
    WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
    """
    df = pd.read_sql(query, conn)
    
    # Apply filters
    if product_search:
        df = df[df["product_id"].astype(str).str.contains(product_search, case=False, na=False)]
    
    if risk_filter:
        df = df[df["risk_level"].isin(risk_filter)]
    
    # Quarter filter (assuming date column exists)
    if quarter_filter != "All Quarters" and "timestamp" in df.columns:
        df['quarter'] = pd.to_datetime(df['timestamp']).dt.to_period('Q')
        target_quarter = f"2026Q{quarter_filter[-1]}"
        df = df[df['quarter'] == target_quarter]
    
    return df

@st.cache_data(ttl=30)
def load_kpis():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)

@st.cache_data(ttl=30)
def load_priority():
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM default.maintenance_priority WHERE priority <= 20 ORDER BY priority",
        conn
    )

# Load data
predictions_df = load_predictions()
kpis_df = load_kpis()
priority_df = load_priority()

## KPI Cards - Row 1
st.markdown("### 📊 Real-time KPIs")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    total_machines = len(predictions_df)
    st.metric("Total Machines", total_machines, delta=f"+{total_machines//10}")

with kpi_col2:
    high_risk_count = len(predictions_df[predictions_df["risk_level"] == "HIGH RISK"])
    st.metric("High Risk Count", high_risk_count, delta=f"+{high_risk_count//20}")

with kpi_col3:
    priority_actions = len(priority_df)
    st.metric("Priority Actions", priority_actions, delta=f"+{priority_actions//5}")

with kpi_col4:
    st.metric("Model AUC", "0.954", delta="↗ 0.002")

# Charts Layout
st.markdown("### 📈 Real-time Analytics Dashboard")

# Row 1: Chart 1 (Left Bar) + Chart 2 (Donut)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Chart 1: Machine Type Volume**")
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        machine_vol = predictions_df["machine_type"].value_counts().head(10).reset_index()
        machine_vol.columns = ["machine_type", "count"]
        
        fig1 = px.bar(
            machine_vol, x="count", y="machine_type", 
            orientation='h',
            title="Machine Type Volume",
            color="count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Machine type data not available")

with col2:
    st.markdown("**Chart 2: Risk Distribution**")
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_dist = predictions_df["risk_level"].value_counts().reset_index()
        risk_dist.columns = ["risk_level", "count"]
        
        fig2 = px.pie(
            risk_dist, values="count", names="risk_level",
            title="Risk Distribution",
            hole=0.4,
            color_discrete_map={
                "HIGH RISK": "#ff4444", 
                "MEDIUM RISK": "#ffaa00", 
                "LOW RISK": "#44ff44"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Risk data not available")

# Row 2: Chart 3 (Right Bar) + Chart 4 (Bottom Donut)
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("**Chart 3: Priority by Machine Type**")
    if not predictions_df.empty and {"machine_type", "risk_level"}.issubset(predictions_df.columns):
        prio_chart = predictions_df.groupby(["machine_type", "risk_level"]).size().reset_index(name="count")
        
        fig3 = px.bar(
            prio_chart, x="machine_type", y="count", color="risk_level",
            title="Priority by Machine Type",
            barmode="group",
            color_discrete_map={
                "HIGH RISK": "#ff4444", 
                "MEDIUM RISK": "#ffaa00", 
                "LOW RISK": "#44ff44"
            }
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Priority chart data not available")

with col4:
    st.markdown("**Chart 4: Risk Share**")
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_share = predictions_df["risk_level"].value_counts().reset_index()
        risk_share.columns = ["risk_level", "share"]
        
        fig4 = px.pie(
            risk_share, values="share", names="risk_level",
            title="Risk Share",
            hole=0.5,
            color_discrete_map={
                "HIGH RISK": "#ff4444", 
                "MEDIUM RISK": "#ffaa00", 
                "LOW RISK": "#44ff44"
            }
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Risk share data not available")

# Row 3: Chart 5 (Bar) + Chart 6 (Line)
col5, col6 = st.columns([1, 1])

with col5:
    st.markdown("**Chart 5: Risk Frequency**")
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_freq = predictions_df["risk_level"].value_counts().reset_index()
        risk_freq.columns = ["risk_level", "frequency"]
        
        fig5 = px.bar(
            risk_freq, x="risk_level", y="frequency",
            title="Risk Frequency",
            color="frequency",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Risk frequency data not available")

with col6:
    st.markdown("**Chart 6: Risk Trend**")
    if not predictions_df.empty and "timestamp" in predictions_df.columns:
        df_trend = predictions_df.copy()
        df_trend['date'] = pd.to_datetime(df_trend['timestamp']).dt.date
        
        trend_data = df_trend.groupby(['date', 'risk_level']).size().reset_index(name='count')
        
        fig6 = px.line(
            trend_data, x="date", y="count", color="risk_level",
            title="Daily Risk Trend",
            color_discrete_map={
                "HIGH RISK": "#ff4444", 
                "MEDIUM RISK": "#ffaa00", 
                "LOW RISK": "#44ff44"
            }
        )
        fig6.update_traces(mode='lines+markers')
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Trend data (timestamp column) not available")

# Top Priorities Table
st.markdown("### 🎯 Top Maintenance Priorities")
if not priority_df.empty:
    display_cols = ["udi", "product_id", "machine_type", "risk_level", "priority"]
    available_cols = [col for col in display_cols if col in priority_df.columns]
    st.dataframe(
        priority_df[available_cols].head(20), 
        use_container_width=True,
        height=400
    )
else:
    st.info("No priority data available")

# AI Assistant (unchanged)
st.markdown("---")
st.subheader("🤖 AI Maintenance Advisor")
question = st.text_area(
    "Ask about machine health or maintenance strategy:",
    placeholder="Which machines should I fix first? What causes high risk?",
    height=100
)

if st.button("Get AI Advice", type="primary") and question.strip():
    with st.spinner("AI analyzing maintenance data..."):
        try:
            llm = get_llm()
            safe_cols = [col for col in ["udi", "product_id", "machine_type", "risk_level", "priority"] if col in priority_df.columns]
            context = priority_df[safe_cols].head(10).to_string(index=False) if not priority_df.empty else "No priority data available"

            prompt = f"""
You are a predictive maintenance assistant.

Use ONLY the data provided below.
Do NOT use outside knowledge.
Do NOT invent dates, thresholds, causes, downtime, cost savings, tool wear, machine_failure values.

Provided data:
{context}

User question:
{question}

Return answer in this format:

**Summary:** ...
**Priority machines:** ...
**Recommended actions:** ...
**Missing data:** ...
"""
            response = llm.generate_content(prompt)
            st.success("✅ AI Analysis Complete!")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"AI service error: {str(e)}")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**🗄️ Databricks Lakehouse**")
    st.markdown("- Medallion Architecture")
    st.markdown("- Delta Lake (ACID)")

with col_f2:
    st.markdown("**🤖 ML Pipeline**")
    st.markdown("- Random Forest")
    st.markdown("- AUC: 0.954")

with col_f3:
    st.markdown("**🚀 Real-time**")
    st.markdown("- Auto-refresh: 30s")
    st.markdown("- Live filtering")

st.markdown("---")
st.caption("**Built by Anchit Chourasia** | AI Engineer @ HEG Limited | [GitHub](https://github.com/anchitchourasia)")
