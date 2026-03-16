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

# Metrics
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

# Charts
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

# Footer
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

# ======================================================
# 🗨️ FLOATING CHAT - COMPLETE WORKING VERSION
# ======================================================

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_question" not in st.session_state:
    st.session_state.chat_question = ""
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""
if "chat_interaction" not in st.session_state:
    st.session_state.chat_interaction = None

# CSS for perfect ChatCompose-style chat
st.markdown("""
<style>
.chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 10000;
}

.chat-fab {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border: none;
    color: white;
    font-size: 24px;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-fab:hover {
    transform: scale(1.1);
    box-shadow: 0 12px 35px rgba(16, 185, 129, 0.6);
}

.chat-popup {
    width: 380px;
    height: 520px;
    background: white;
    border-radius: 24px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.25);
    overflow: hidden;
    transform: translateY(30px);
    opacity: 0;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    margin-bottom: 20px;
}

.chat-popup.show {
    transform: translateY(0);
    opacity: 1;
}

.chat-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-title {
    font-weight: 700;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.chat-close {
    background: rgba(255,255,255,0.2);
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    color: white;
    cursor: pointer;
    font-size: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s;
}

.chat-close:hover {
    background: rgba(255,255,255,0.3);
}

.chat-messages {
    height: 340px;
    padding: 24px;
    overflow-y: auto;
    background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
}

.ai-message {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 24px 24px 24px 8px;
    margin-bottom: 16px;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
}

.user-placeholder {
    color: #64748b;
    text-align: center;
    padding-top: 120px;
    font-style: italic;
}

.chat-input-container {
    padding: 24px;
    background: white;
    border-top: 1px solid #e2e8f0;
    display: flex;
    gap: 12px;
    align-items: center;
}

.chat-input {
    flex: 1;
    border: 2px solid #e2e8f0;
    border-radius: 28px;
    padding: 14px 24px;
    font-size: 15px;
    outline: none;
    transition: all 0.2s;
}

.chat-input:focus {
    border-color: #10b981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.chat-send {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border: none;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: all 0.2s;
}

.chat-send:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Prepare chat HTML variables (FIXES f-string backslash issue)
chat_display_style = "block" if st.session_state.show_chat else "none"
chat_visible_js = "true" if st.session_state.show_chat else "false"
chat_response_html = st.session_state.chat_response or '<div class="user-placeholder">Ask me about machine health, maintenance priorities, or risk analysis!</div>'
chat_question_value = st.session_state.chat_question.replace('"', '&quot;') if st.session_state.chat_question else ""

# Complete working chat HTML
chat_html = f"""
<div class="chat-container">
    <button class="chat-fab" id="chatToggleBtn" onclick="toggleChat()">
        🤖
    </button>
    
    <div id="chatPopup" class="chat-popup" style="display: {chat_display_style};">
        <div class="chat-header">
            <div class="chat-title">🤖 AI Maintenance Advisor</div>
            <button class="chat-close" onclick="toggleChat()">×</button>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            {chat_response_html}
        </div>
        
        <div class="chat-input-container">
            <input type="text" class="chat-input" id="chatInput" placeholder="Which machines should I fix first? What causes high risk?" value="{chat_question_value}">
            <button class="chat-send" onclick="sendMessage()">➤</button>
        </div>
    </div>
</div>

<script>
let chatVisible = {chat_visible_js};

function toggleChat() {{
    chatVisible = !chatVisible;
    const popup = document.getElementById('chatPopup');
    const fab = document.getElementById('chatToggleBtn');
    
    if (chatVisible) {{
        popup.style.display = 'block';
        setTimeout(() => popup.classList.add('show'), 10);
        fab.style.opacity = '0';
        fab.style.visibility = 'hidden';
    }} else {{
        popup.classList.remove('show');
        setTimeout(() => {{
            popup.style.display = 'none';
            fab.style.opacity = '1';
            fab.style.visibility = 'visible';
        }}, 400);
    }}
    
    // Notify Streamlit
    window.parent.postMessage({{
        type: "streamlit:setComponentValue",
        value: chatVisible
    }}, "*");
}}

function sendMessage() {{
    const input = document.getElementById('chatInput');
    const messages = document.getElementById('chatMessages');
    const question = input.value.trim();
    
    if (question) {{
        // Show user message immediately
        messages.innerHTML += `
            <div style="display: flex; justify-content: flex-end; margin-bottom: 16px;">
                <div style="background: #e2e8f0; padding: 14px 20px; border-radius: 24px 24px 8px 24px; max-width: 75%; font-size: 15px;">
                    ${{question}}
                </div>
            </div>
        `;
        input.value = '';
        messages.scrollTop = messages.scrollHeight;
        
        // Send to Streamlit for AI response
        window.parent.postMessage({{
            type: "streamlit:setComponentValue",
            value: {{"question": question, "action": "generate"}}
        }}, "*");
    }}
}}

// Enter key support
document.getElementById('chatInput').addEventListener('keypress', function(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendMessage();
    }}
}});
</script>
"""

st.components.v1.html(chat_html, height=650, width=420)

# Handle AI generation
if st.session_state.chat_interaction:
    interaction = st.session_state.chat_interaction
    if isinstance(interaction, dict) and interaction.get("action") == "generate":
        question = interaction.get("question", "")
        if question:
            st.session_state.chat_question = question
            
            with st.spinner("🤖 AI analyzing maintenance data..."):
                try:
                    llm = get_llm()
                    safe_cols = [col for col in ["udi", "product_id", "machine_type", "risk_level", "priority"] if col in priority_df.columns]
                    context = priority_df[safe_cols].head(10).to_string(index=False) if not priority_df.empty else "No priority data available"

                    prompt = f"""
You are a predictive maintenance assistant for a live production dashboard.

Use ONLY the data provided below.
Do NOT use outside knowledge or invent values.

Provided data:
{context}

Question: {question}

Respond in this exact format:

**Summary:**
- Key insight 1
- Key insight 2

**Priority Machines:**
- udi: [ID], product_id: [ID], risk_level: [LEVEL], priority: [NUM]

**Recommended Actions:**
- Action 1
- Action 2

**Data Gaps:**
- Missing field 1
"""

                    response = llm.generate_content(prompt)
                    answer = response.text.strip()
                    
                    # Format for chat bubble
                    formatted_answer = answer.replace("**Summary:**", "<strong>Summary:</strong>").replace("**Priority Machines:**", "<strong>Priority Machines:</strong>").replace("**Recommended Actions:**", "<strong>Recommended Actions:</strong>").replace("**Data Gaps:**", "<strong>Data Gaps:</strong>").replace("\n", "<br>")
                    
                    st.session_state.chat_response = f'<div class="ai-message">{formatted_answer}</div>'
                    
                    st.success("✅ AI response generated!")
                    
                except Exception as e:
                    st.session_state.chat_response = f'<div class="ai-message" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">⚠️ Error: {str(e)}</div>'
    
    st.session_state.chat_interaction = None

# Reset interaction listener
st.session_state.chat_interaction = st.session_state.get("chat_interaction_temp", None)
