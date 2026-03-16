"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
"""

import os
import pandas as pd
import plotly.express as px
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

    if product_search:
        df = df[df["product_id"].astype(str).str.contains(product_search, case=False, na=False)]

    if risk_filter:
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

col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk_count = len(predictions_df[predictions_df["risk_level"] == "HIGH RISK"]) if not predictions_df.empty else 0
    st.metric("Total Machines", len(predictions_df), delta=f"{high_risk_count} High Risk")

with col2:
    high_risk_pct = (
        len(predictions_df[predictions_df["risk_level"] == "HIGH RISK"]) / len(predictions_df)
        if not predictions_df.empty else 0
    )
    st.metric("High Risk %", f"{high_risk_pct:.1%}")

with col3:
    kpi_value = "0.954"
    kpi_note = None

    if not kpis_df.empty:
        possible_cols = ["failure_rate", "avg_failure_rate", "actual_failure_rate", "prediction"]
        found_col = next((c for c in possible_cols if c in kpis_df.columns), None)

        if found_col:
            try:
                avg_failure = pd.to_numeric(kpis_df[found_col], errors="coerce").dropna().mean()
                if pd.notna(avg_failure):
                    kpi_note = f"Actual failure: {avg_failure:.1%}"
            except Exception:
                kpi_note = None

    st.metric("Model AUC", kpi_value, delta=None)

    if kpi_note:
        st.caption(f"Observed KPI: {kpi_note}")
    else:
        st.caption("Observed KPI: Not mapped in gold_machine_kpis")

with col4:
    st.metric("Priority Actions", len(priority_df))

col1, col2 = st.columns(2)

with col1:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_counts = (
            predictions_df["risk_level"]
            .value_counts()
            .reset_index()
        )
        risk_counts.columns = ["risk_level", "count"]

        fig_pie = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            title="Risk Distribution",
            hole=0.15
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No prediction data available.")

with col2:
    if not predictions_df.empty and {"machine_type", "risk_level"}.issubset(predictions_df.columns):
        chart_df = (
            predictions_df.groupby(["machine_type", "risk_level"])
            .size()
            .reset_index(name="count")
        )

        fig_bar = px.bar(
            chart_df,
            x="machine_type",
            y="count",
            color="risk_level",
            title="Risk by Machine Type",
            barmode="group"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Machine type chart data not available.")

st.subheader("🎯 Top Maintenance Priorities")
if not priority_df.empty:
    display_cols = ["udi", "product_id", "machine_type", "risk_level", "priority"]
    available_cols = [col for col in display_cols if col in priority_df.columns]
    st.dataframe(priority_df[available_cols], use_container_width=True)
else:
    st.info("No priority data available.")

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

            safe_cols = [col for col in ["udi", "product_id", "machine_type", "risk_level", "priority"] if col in priority_df.columns]
            context = priority_df[safe_cols].head(10).to_string(index=False) if not priority_df.empty else "No priority data available"

            prompt = f"""
You are a predictive maintenance assistant.

Use ONLY the data provided below.
Do NOT use outside knowledge.
Do NOT invent dates, thresholds, causes, downtime, cost savings, tool wear, machine_failure values, or any field not explicitly present.
If something is not explicitly present in the data, say: Not available in provided data.

Provided data:
{context}

User question:
{question}

Return the answer in exactly this plain-text format:

Summary:
- ...

Priority machines:
- udi: ..., product_id: ..., machine_type: ..., risk_level: ..., priority: ...

Recommended actions:
- ...
- ...

Missing data:
- ...
- ...

Rules:
- Stay fully grounded in the provided data.
- Only mention columns explicitly present in the provided data.
- Keep the response concise and professional.
"""

            response = llm.generate_content(prompt)
            answer = response.text.strip()

            formatted = (
                answer.replace("Summary:", "#### Summary")
                      .replace("Priority machines:", "#### Priority machines")
                      .replace("Recommended actions:", "#### Recommended actions")
                      .replace("Missing data:", "#### Missing data")
            )

            st.success("✅ AI Analysis Complete!")
            st.markdown("### AI Maintenance Advisor")
            st.markdown(formatted)

        except Exception as e:
            st.error(f"AI service error: {str(e)}")

st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**🗄️ Databricks Lakehouse**")
    st.markdown("- Medallion Architecture (Bronze/Silver/Gold)")
    st.markdown("- Delta Lake (ACID + Time Travel)")

with c2:
    st.markdown("**🤖 ML Pipeline**")
    st.markdown("- Random Forest Classification")
    st.markdown("- AUC: 0.954")

with c3:
    st.markdown("**🚀 Production**")
    st.markdown("- Batch Inference Pipeline")
    st.markdown("- Real-time Risk Dashboard")

st.markdown("---")
st.caption("**Built by Anchit Chourasia** | Aspiring AI Engineer | [GitHub](https://github.com/anchitchourasia)")
