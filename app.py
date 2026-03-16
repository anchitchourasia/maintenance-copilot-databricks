"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
Updated: Fixes product search in AI context + expands priority query
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
def load_priority(product_search):
    conn = get_connection()
    
    if product_search:
        # GUARANTEE searched product appears (top 20 OR searched)
        query = f"""
        SELECT 
          udi, product_id, machine_type, tool_wear_min, machine_failure,
          prediction as predicted_failure, risk_level,
          ROW_NUMBER() OVER (ORDER BY 
            CASE WHEN prediction = 1 THEN 1 WHEN risk_level='HIGH RISK' THEN 2 ELSE 3 END,
            tool_wear_min DESC
          ) as display_priority
        FROM default.gold_predictions 
        WHERE product_id LIKE '%{product_search}%' 
           OR display_priority <= 20  -- Dynamic top 20
        ORDER BY display_priority
        LIMIT 25
        """
    else:
        query = """
        SELECT udi, product_id, machine_type, tool_wear_min, machine_failure,
               prediction as predicted_failure, risk_level,
               ROW_NUMBER() OVER (ORDER BY prediction DESC, tool_wear_min DESC) as display_priority
        FROM default.maintenance_priority 
        LIMIT 25
        """
    
    df = pd.read_sql(query, conn)
    return df

# Load data
predictions_df = load_predictions()
kpis_df = load_kpis()
priority_df = load_priority(product_search)  # Pass search term

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
    st.metric("Model AUC", kpi_value)
    st.caption("Random Forest Classifier")

with col4:
    st.metric("Priority Actions", len(priority_df))

col1, col2 = st.columns(2)

with col1:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_counts = predictions_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_pie = px.pie(
            risk_counts, values="count", names="risk_level",
            title="Risk Distribution", hole=0.15
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    if not predictions_df.empty and {"machine_type", "risk_level"}.issubset(predictions_df.columns):
        chart_df = (
            predictions_df.groupby(["machine_type", "risk_level"])
            .size().reset_index(name="count")
        )
        fig_bar = px.bar(
            chart_df, x="machine_type", y="count", color="risk_level",
            title="Risk by Machine Type", barmode="group"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("🎯 Top Maintenance Priorities")
if not priority_df.empty:
    display_cols = ["udi", "product_id", "machine_type", "risk_level", "priority", "failure_probability"]
    available_cols = [col for col in display_cols if col in priority_df.columns]
    st.dataframe(priority_df[available_cols], use_container_width=True)
else:
    st.info("No priority data available.")

# AI Assistant - FIXED CONTEXT
st.subheader("🤖 AI Maintenance Advisor")
question = st.text_area(
    "Ask about machine health or maintenance strategy:",
    placeholder="Which machines should I fix first? What about L47181?"
)

if st.button("Get AI Advice", type="primary") and question:
    with st.spinner("AI analyzing maintenance data..."):
        try:
            llm = get_llm()

            # Enhanced context: priorities + search matches
            safe_cols = ["udi", "product_id", "machine_type", "risk_level", "priority", "failure_probability"]
            safe_cols = [col for col in safe_cols if col in priority_df.columns]
            
            priority_context = priority_df[safe_cols].head(15).to_string(index=False) if not priority_df.empty else "No priorities"
            
            # Add search matches from predictions
            search_context = ""
            if product_search and not predictions_df.empty:
                search_matches = predictions_df[
                    predictions_df['product_id'].astype(str).str.contains(product_search, na=False)
                ][safe_cols[:5]].head(5)  # Limit cols for safety
                if not search_matches.empty:
                    search_context = search_matches.to_string(index=False)

            context = f"""
Top 15 Priorities:
{priority_context}

Search '{product_search}' matches (if any):
{search_context}
"""

            prompt = f"""
You are a predictive maintenance assistant.

Use ONLY the data provided below. Do NOT use outside knowledge or invent values.

Provided data:
{context}

User question: {question}

Respond in this EXACT format:

**Summary:** [1-2 sentences based ONLY on provided data]

**Priority Machines:** 
- UDI: [value], Product: [value], Type: [value], Risk: [value], Priority: [value], Probability: [value]

**Recommended Actions:**
• [Action 1 grounded in data]
• [Action 2 grounded in data]

**Data Notes:** [What was searched vs found]

Keep concise, professional, data-driven.
"""

            response = llm.generate_content(prompt)
            st.success("✅ AI Analysis Complete!")
            st.markdown(response.text.strip())

        except Exception as e:
            st.error(f"AI service error: {str(e)}")

st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**🗄️ Databricks Lakehouse**")
    st.markdown("- Medallion: Bronze→Silver→Gold")
    st.markdown("- Delta Lake ACID + Time Travel")

with c2:
    st.markdown("**🤖 ML Pipeline**")
    st.markdown("- Random Forest (AUC 0.954)")
    st.markdown("- PySpark ML + Batch Inference")

with c3:
    st.markdown("**🚀 Production**")
    st.markdown("- Streamlit + Databricks Apps")
    st.markdown("- Gemini AI Advisor")

st.markdown("---")
st.caption("**Built by Anchit Chourasia** | Codebasics Databricks Challenge | [GitHub](https://github.com/anchitchourasia/maintenance-copilot-app)")
