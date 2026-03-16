import os
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import databricks.sql
import google.generativeai as genai

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="Maintenance Copilot Dashboard",
    page_icon="📊"
)

# --- CUSTOM CSS (For the Pink/Purple Dashboard Look) ---
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #a445b2 0%, #d41872 50%, #ff0066 100%);
        color: white;
    }
    
    /* Card/Chart Container Styling */
    div[data-testid="stMetric"], .stPlotlyChart, div[data-testid="stDataFrameHost"] {
        background-color: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Text Colors */
    h1, h2, h3, h4, span, label, p {
        color: white !important;
    }

    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- CONNECTIONS ---
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
    return genai.GenerativeModel("gemini-1.5-flash") # Using latest stable flash

# --- DATA LOADING (Real-time 60s cache) ---
@st.cache_data(ttl=60)
def load_data():
    conn = get_connection()
    # Fetching predictions
    df = pd.read_sql("SELECT * FROM default.gold_predictions", conn)
    # Fetching KPIs
    kpi = pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)
    # Fetching Priorities
    prio = pd.read_sql("SELECT * FROM default.maintenance_priority WHERE priority <= 20", conn)
    return df, kpi, prio

# Sidebar
st.sidebar.title("⚙️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Real-time Auto-Refresh (60s)", value=True)
product_search = st.sidebar.text_input("🔍 Search Product ID")
risk_filter = st.sidebar.multiselect(
    "Risk Level", ["HIGH RISK", "MEDIUM RISK", "LOW RISK"], 
    default=["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
)

# Load Data
try:
    predictions_df, kpis_df, priority_df = load_data()
    
    # Filter Logic
    if product_search:
        predictions_df = predictions_df[predictions_df["product_id"].astype(str).str.contains(product_search)]
    if risk_filter:
        predictions_df = predictions_df[predictions_df["risk_level"].isin(risk_filter)]

    # --- TOP HEADER ROW ---
    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([3, 2, 1, 1, 1])
    
    with t_col1:
        st.markdown("# Predictive Maintenance")
    
    with t_col2:
        st.write("") # Spacer
        st.segmented_control("Timeline", options=["Qtr 1", "Qtr 2", "Qtr 3", "Qtr 4"], default="Qtr 1")

    with t_col3:
        st.metric("Total Units", len(predictions_df))
    
    with t_col4:
        st.metric("Avg Priority", round(priority_df["priority"].mean(), 1) if not priority_df.empty else 0)
        
    with t_col5:
        st.metric("High Risk", len(predictions_df[predictions_df["risk_level"]=="HIGH RISK"]))

    # --- MAIN DASHBOARD GRID ---
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        # Profit by Sub-Category (Mapped to Risk by Type)
        fig1 = px.bar(predictions_df, x="priority", y="machine_type", orientation='h', 
                      title="Priority by Machine Type", color_discrete_sequence=['#00D1FF'])
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig1, use_container_width=True)

    with m_col2:
        # Donut Chart
        fig2 = px.pie(predictions_df, names="risk_level", hole=0.5, title="Risk Distribution")
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with m_col3:
        # Repeating the Bar Chart look from the image
        st.plotly_chart(fig1, use_container_width=True)

    m2_col1, m2_col2, m2_col3 = st.columns(3)

    with m2_col1:
        # Double Ring Chart (Quantity/Amount)
        fig3 = px.pie(predictions_df, names="machine_type", hole=0.7, title="Type Composition")
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    with m2_col2:
        # Amount by Customer (Mapped to Priority by ID)
        fig4 = px.bar(priority_df.head(6), x="product_id", y="priority", title="Top Priority IDs")
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig4, use_container_width=True)

    with m2_col3:
        # Line Chart (Profit by Month)
        # Note: Using UDI as a proxy for 'time' if no date is in your schema
        fig5 = px.line(priority_df, y="priority", title="Maintenance Trend (Live)")
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig5, use_container_width=True)

    # --- AI ASSISTANT SECTION ---
    st.markdown("---")
    st.subheader("🤖 AI Maintenance Advisor (Gemini 3 Flash)")
    question = st.text_input("Ask about machine health:")
    
    if question:
        llm = get_llm()
        context = priority_df.head(10).to_string()
        response = llm.generate_content(f"Data: {context}\nQuestion: {question}")
        st.info(response.text)

except Exception as e:
    st.error(f"Waiting for Databricks Connection... {e}")

# Real-time Auto-refresh logic
if auto_refresh:
    time.sleep(60)
    st.rerun()
