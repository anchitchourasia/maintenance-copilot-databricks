"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Databricks SQL Connector
import databricks.sql

# LLM (Gemini 3 Flash)
import google.generativeai as genai

# Page config
st.set_page_config(
    layout="wide",
    page_title="Maintenance Copilot",
    page_icon="🔧"
)

st.title("🔧 Predictive Maintenance Copilot")
st.markdown("**Databricks Lakehouse | Random Forest AUC 0.954 | Gemini 3 Flash**")

# Sidebar
st.sidebar.title("⚙️ Controls")
product_search = st.sidebar.text_input("🔍 Search Product ID")
risk_filter = st.sidebar.multiselect(
    "Filter Risk Level",
    ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
    default=["HIGH RISK", "MEDIUM RISK"]
)

# Connection function
@st.cache_resource
def get_connection():
    return databricks.sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )

# LLM Setup (Gemini 3 Flash)
@st.cache_resource
def get_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel('gemini-3-flash-preview')

# Load data functions
@st.cache_data(ttl=300)
def load_predictions():
    conn = get_connection()
    query = """
    SELECT * FROM default.gold_predictions 
    WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
    """
    return pd.read_sql(query, conn)

@st.cache_data(ttl=300)
def load_kpis():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)

@st.cache_data(ttl=300)
def load_priority():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.maintenance_priority WHERE priority_rank <= 20", conn)

# Main dashboard
col1, col2, col3, col4 = st.columns(4)
predictions_df = load_predictions()

with col1:
    st.metric(
        "Total Machines",
        len(predictions_df),
        delta=f"{len(predictions_df[predictions_df['risk_level']=='HIGH RISK'])} High Risk"
    )

with col2:
    high_risk_pct = len(predictions_df[predictions_df['risk_level']=='HIGH RISK']) / len(predictions_df)
    st.metric("High Risk %", f"{high_risk_pct:.1%}")

with col3:
    try:
        kpis_df = load_kpis()
        if not kpis_df.empty and 'failure_rate' in kpis_df.columns:
            avg_failure = kpis_df['failure_rate'].mean()
            delta_text = f"vs Actual {avg_failure:.1%}"
        else:
            delta_text = "KPI table empty/missing column"
    except:
        delta_text = "Table not ready"
    st.metric("Model AUC", "0.954", delta=delta_text)

with col4:
    priority_df = load_priority()
    st.metric("Priority Actions", len(priority_df))

# Charts
col1, col2 = st.columns(2)

with col1:
    # Risk distribution
    risk_counts = predictions_df['risk_level'].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values, 
        names=risk_counts.index, 
        title="Risk Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Machine type risk
    fig_bar = px.bar(
        predictions_df, 
        x="machine_type", 
        color="risk_level",
        title="Risk by Machine Type", 
        barmode="group"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Priority maintenance list
st.subheader("🎯 Top Maintenance Priorities")
priority_df = load_priority()
if not priority_df.empty:
    st.dataframe(
        priority_df[['udi', 'product_id', 'machine_type', 'risk_level', 'priority_rank']],
        use_container_width=True
    )
else:
    st.info("Run Notebook 06_kpi_tables.sql first to create priority table")

# AI Assistant
st.subheader("🤖 AI Maintenance Advisor")
question = st.text_area(
    "Ask about machine health or maintenance strategy:", 
    placeholder="Which machines should I fix first? What causes high risk?"
)

if st.button("Get AI Advice", type="primary") and question:
    with st.spinner("AI analyzing maintenance data..."):
        try:
            llm = get_llm()
            
            # Context for LLM
            context = priority_df.head(10).to_string() if not priority_df.empty else "No priority data available"
            prompt = f"""
            Predictive Maintenance Intelligence Report
            
            Data Summary:
            {context}
            
            Model: Random Forest (AUC 0.954)
            
            USER QUESTION: {question}
            
            Provide:
            1. Actionable maintenance recommendations  
            2. Priority machine list (UDI/Product ID specific)
            3. Risk mitigation steps
            4. Estimated impact (downtime/cost savings)
            
            Format as professional engineering report.
            """
            
            response = llm.generate_content(prompt)
            st.success("✅ AI Analysis Complete!")
            st.markdown("### **AI Maintenance Advisor**")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"AI service error: {str(e)}")
            st.info("Get free Gemini API key: https://makersuite.google.com/app/apikey")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🗄️ Databricks Lakehouse**")
    st.markdown("- Medallion Architecture (Bronze/Silver/Gold)")
    st.markdown("- Delta Lake (ACID + Time Travel)")
with col2:
    st.markdown("**🤖 ML Pipeline**")
    st.markdown("- Random Forest Classification")
    st.markdown("- AUC: **0.954**")
with col3:
    st.markdown("**🚀 Production**")
    st.markdown("- Batch Inference Pipeline")
    st.markdown("- Real-time Risk Dashboard")

st.markdown("---")
st.caption("**Built by Anchit Chourasia** | HEG Limited AI Engineer | [GitHub](https://github.com/anchitchourasia)")
