"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import databricks.sql
import google.generativeai as genai

load_dotenv()

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
    default=["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
)

quarter_filter = st.sidebar.selectbox(
    "Select Quarter",
    ["Q1", "Q2", "Q3", "Q4", "All"]
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

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

@st.cache_data(ttl=60)
def load_predictions():
    conn = get_connection()
    query = """
    SELECT * FROM default.gold_predictions
    WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
    """
    df = pd.read_sql(query, conn)

    if product_search and "product_id" in df.columns:
        df = df[df["product_id"].astype(str).str.contains(product_search, case=False, na=False)]

    if risk_filter and "risk_level" in df.columns:
        df = df[df["risk_level"].isin(risk_filter)]

    return df

@st.cache_data(ttl=60)
def load_kpis():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)

@st.cache_data(ttl=60)
def load_priority():
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM default.maintenance_priority WHERE priority <= 20 ORDER BY priority",
        conn
    )

predictions_df = load_predictions()
kpis_df = load_kpis()
priority_df = load_priority()

def style_fig(fig, title):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827"),
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        height=200
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    return fig

# Quarter selector (top bar)
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
with col_q1:
    st.metric("Q1 Risk Count", "514", "37K", label_visibility="collapsed")
with col_q2:
    st.metric("Q2 Risk Count", "37K", "43K", label_visibility="collapsed")
with col_q3:
    st.metric("Q3 Risk Count", "43K", "+12%", label_visibility="collapsed")
with col_q4:
    st.metric("Q4 Risk Count", "Total", "Live", label_visibility="collapsed")

# Main 4x3 Grid
row1 = st.columns([1, 1, 1, 1])
row2 = st.columns([1, 1, 1, 1])
row3 = st.columns([1, 1, 1, 1])

# Row 1
with row1[0]:
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        type_counts = predictions_df["machine_type"].value_counts().head(5)
        fig1 = px.bar(type_counts.reset_index(), x="machine_type", y="count")
        style_fig(fig1, "Machines by Type")
        st.plotly_chart(fig1, use_container_width=True)

with row1[1]:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_counts = predictions_df["risk_level"].value_counts()
        fig2 = px.pie(risk_counts.reset_index(), values="count", names="risk_level", hole=0.4)
        style_fig(fig2, "Risk Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with row1[2]:
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        type_risk = predictions_df.groupby("machine_type", as_index=False).size()
        fig3 = px.bar(type_risk, x="machine_type", y="size")
        style_fig(fig3, "Risk by Type")
        st.plotly_chart(fig3, use_container_width=True)

with row1[3]:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        fig4 = px.histogram(predictions_df, x="risk_level")
        style_fig(fig4, "Risk Frequency")
        st.plotly_chart(fig4, use_container_width=True)

# Row 2
with row2[0]:
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        type_pct = predictions_df["machine_type"].value_counts(normalize=True) * 100
        fig5 = px.pie(type_pct.reset_index(), values="proportion", names="machine_type", hole=0.3)
        style_fig(fig5, "% by Machine Type")
        st.plotly_chart(fig5, use_container_width=True)

with row2[1]:
    total = len(predictions_df)
    if total > 0:
        fig6 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=len(predictions_df[predictions_df["risk_level"] == "HIGH RISK"]),
            number={"font": {"size": 28}},
            title={"text": "High Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"}
                ]
            }
        ))
        style_fig(fig6, "Risk Gauge")
        st.plotly_chart(fig6, use_container_width=True)

with row2[2]:
    if not priority_df.empty and "machine_type" in priority_df.columns:
        pri_type = priority_df["machine_type"].value_counts()
        fig7 = px.bar(pri_type.reset_index(), x="machine_type", y="count")
        style_fig(fig7, "Priority by Type")
        st.plotly_chart(fig7, use_container_width=True)

with row2[3]:
    if not predictions_df.empty and {"machine_type", "risk_level"}.issubset(predictions_df.columns):
        stacked = predictions_df.groupby(["machine_type", "risk_level"]).size().reset_index(name="count")
        fig8 = px.bar(stacked, x="machine_type", y="count", color="risk_level", barmode="stack")
        style_fig(fig8, "Stacked Risk by Type")
        st.plotly_chart(fig8, use_container_width=True)

# Row 3
with row3[0]:
    if not predictions_df.empty:
        fig9 = px.scatter(predictions_df.head(100), x="product_id", y="risk_level", color="machine_type")
        style_fig(fig9, "Product vs Risk")
        st.plotly_chart(fig9, use_container_width=True)

with row3[1]:
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        fig10 = px.treemap(predictions_df, path=["machine_type", "risk_level"], values="product_id")
        style_fig(fig10, "Risk Tree Map")
        st.plotly_chart(fig10, use_container_width=True)

with row3[2]:
    date_col = next((c for c in predictions_df.columns if "date" in c.lower() or "timestamp" in c.lower()), None)
    if date_col:
        trend = predictions_df.groupby(date_col).size().reset_index(name="count")
        fig11 = px.line(trend, x=date_col, y="count")
        style_fig(fig11, "Trend Over Time")
        st.plotly_chart(fig11, use_container_width=True)
    else:
        st.info("Date column not found for trend")

with row3[3]:
    if not kpis_df.empty:
        numeric_cols = kpis_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig12 = px.box(kpis_df[numeric_cols[:3]])
            style_fig(fig12, "KPI Distribution")
            st.plotly_chart(fig12, use_container_width=True)
        else:
            st.info("No numeric KPI columns")

st.subheader("🎯 Top Maintenance Priorities")
if not priority_df.empty:
    display_cols = ["udi
