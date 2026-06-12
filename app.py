import streamlit as st
import pandas as pd
import json
from datetime import datetime
from eco_budget_agent import run_agent

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Budgeting Agent Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Slate Theme custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Panel Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics panel styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Expanders styling */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        margin-bottom: 5px !important;
        color: #c9d1d9 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 15px !important;
        color: #8b949e !important;
    }
    
    /* Custom compliance cards */
    .compliance-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid;
    }
    
    .compliance-success {
        background-color: rgba(46, 160, 67, 0.15);
        border-color: #2ea043;
        color: #56d364;
    }
    
    .compliance-warning {
        background-color: rgba(210, 153, 34, 0.15);
        border-color: #d29922;
        color: #e3b341;
    }
    
    .compliance-danger {
        background-color: rgba(248, 81, 73, 0.15);
        border-color: #f85149;
        color: #ff7b72;
    }
    
    /* Citations custom formatting */
    .citation-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 10px 15px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=60)
st.sidebar.title("Procurement Config")
st.sidebar.markdown("Configure the initial procurement request and run What-If scenarios.")

st.sidebar.subheader("Supplier Profile")
supplier_id = st.sidebar.text_input("Supplier ID", "SUP-A-007")
supplier_name = st.sidebar.text_input("Supplier Name", "TechParts Inc.")
category = st.sidebar.selectbox("Category", ["electronics", "logistics", "packaging"])
quantity = st.sidebar.number_input("Order Quantity", min_value=1, value=500)
quoted_price_delta_pct = st.sidebar.number_input("Quoted Price Delta (%)", value=8.0)

st.sidebar.subheader("Historical Footprint")
current_quarter_kg_co2 = st.sidebar.number_input("Current Quarter CO₂ (kg)", value=3800.0)
months_remaining = st.sidebar.number_input("Months Remaining in Quarter", min_value=1, value=2)
supplier_concentration_pct = st.sidebar.slider("Current Concentration (% orders to Supplier A)", 0.0, 100.0, 45.0)

trend_str = st.sidebar.text_input("Monthly Trend (CO₂ kg, comma separated)", "900.0,1100.0,1300.0,1500.0")
try:
    monthly_trend = [float(x.strip()) for x in trend_str.split(",")]
except Exception:
    monthly_trend = [900.0, 1100.0, 1300.0, 1500.0]

st.sidebar.subheader("🌿 What-If Scenario Simulator")
st.sidebar.info("Alternative Supplier B (GreenCore Materials) has a carbon footprint of **1.1 kg CO₂/unit** compared to Supplier A's **2.0 kg CO₂/unit**.")
shift_pct = st.sidebar.slider("Shift Orders to Supplier B (%)", 0, 100, 0, step=5)

# Assemble request dict
request = {
    "supplier_id": supplier_id,
    "supplier_name": supplier_name,
    "category": category,
    "quantity": quantity,
    "quoted_price_delta_pct": quoted_price_delta_pct,
    "current_quarter_kg_co2": current_quarter_kg_co2,
    "monthly_trend": monthly_trend,
    "months_remaining": months_remaining,
    "supplier_concentration_pct": supplier_concentration_pct
}

# ─────────────────────────────────────────────
# RUN AGENT & SIMULATE LIVE DATA
# ─────────────────────────────────────────────
# 1. Run the core agent to get the baseline decision & CoT reasoning log
with st.spinner("Agent running reasoning loops & checking guardrails..."):
    result = run_agent(request)

# 2. Recalculate carbon footprint and concentration live based on the What-If slider
# Carbon logic:
# Supplier A footprint = 2.0 kg/unit, Supplier B footprint = 1.1 kg/unit.
original_projected_co2 = result["projected_quarterly_co2_kg"]
supplier_a_rate = 2.0
supplier_b_rate = 1.1
carbon_savings = quantity * (shift_pct / 100.0) * (supplier_a_rate - supplier_b_rate)
adjusted_projected_co2 = round(original_projected_co2 - carbon_savings, 2)
cap = result["sustainability_cap_kg"]

# Concentration logic:
adjusted_concentration_a = round(supplier_concentration_pct * (1 - shift_pct / 100.0), 2)
concentration_b = round(float(shift_pct), 2)

# Decision Overrides / Safety Gates based on live simulation
carbon_ok = adjusted_projected_co2 <= cap
diversity_ok = adjusted_concentration_a <= 40.0 and concentration_b <= 40.0

# ─────────────────────────────────────────────
# MAIN PANEL DISPLAY
# ─────────────────────────────────────────────
st.title("🌿 Eco-Budgeting Agent Dashboard")
st.markdown("### Stateful LangGraph CoT reasoning engine for carbon cap & diversity compliance")
st.markdown("---")

# Compliance Banners
if carbon_ok and diversity_ok:
    st.markdown(
        f'<div class="compliance-card compliance-success">'
        f'<h4>✓ FULLY COMPLIANT SCENARIO</h4>'
        f'Carbon emissions projection is below the <strong>{cap} kg</strong> quarterly cap, and no supplier concentration exceeds the <strong>40%</strong> safety limit.'
        f'</div>',
        unsafe_allow_html=True
    )
elif carbon_ok and not diversity_ok:
    breached = []
    if adjusted_concentration_a > 40.0:
        breached.append(f"{supplier_name} ({adjusted_concentration_a}%)")
    if concentration_b > 40.0:
        breached.append(f"GreenCore Materials ({concentration_b}%)")
    st.markdown(
        f'<div class="compliance-card compliance-warning">'
        f'<h4>⚠ SUPPLIER DIVERSITY BREACH</h4>'
        f'Emissions are within the cap, but concentration exceeds 40.0% on: {", ".join(breached)}. '
        f'Adjust the What-If slider to balance order distributions.'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    reasons = []
    if adjusted_projected_co2 > cap:
        reasons.append(f"Carbon emissions ({adjusted_projected_co2} kg) exceed cap of {cap} kg")
    if not diversity_ok:
        reasons.append("Supplier concentration exceeds 40% cap")
    st.markdown(
        f'<div class="compliance-card compliance-danger">'
        f'<h4>❌ NON-COMPLIANT SCENARIO</h4>'
        f'{" & ".join(reasons)}. Shift order percentages to compliant alternative suppliers.'
        f'</div>',
        unsafe_allow_html=True
    )

# Visual Metrics Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Color code decision card
    decision = result["decision"]
    if decision == "APPROVE" and carbon_ok and diversity_ok:
        st.metric("Agent Decision", "APPROVE")
    elif decision == "APPROVE_ALTERNATIVE" or (carbon_ok and diversity_ok):
        st.metric("Agent Decision", "APPROVE ALT")
    else:
        st.metric("Agent Decision", "REJECT / PIVOT")

with col2:
    st.metric("Quarterly Carbon Cap", f"{cap} kg")

with col3:
    delta_co2 = adjusted_projected_co2 - original_projected_co2
    st.metric(
        "Projected CO₂", 
        f"{adjusted_projected_co2} kg", 
        delta=f"{delta_co2:.1f} kg" if shift_pct > 0 else None, 
        delta_color="inverse"
    )

with col4:
    st.metric("Diversity Index (Max Share)", f"{max(adjusted_concentration_a, concentration_b)}%")

st.markdown("---")

# Chart Layouts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Carbon Projections vs. Cap")
    chart_data = pd.DataFrame({
        "Emissions (kg)": [cap, original_projected_co2, adjusted_projected_co2]
    }, index=["Quarterly Cap", "Original Projection", "What-If Adjusted Projection"])
    st.bar_chart(chart_data, color="#2ea043")

with col_chart2:
    st.subheader("Supplier Share of Wallet (Diversity Index)")
    concentration_data = pd.DataFrame({
        "Concentration (%)": [supplier_concentration_pct, adjusted_concentration_a, concentration_b]
    }, index=[f"Original {supplier_name}", f"Adjusted {supplier_name}", "GreenCore Materials (Supplier B)"])
    st.bar_chart(concentration_data, color="#58a6ff")

st.markdown("---")

# ─────────────────────────────────────────────
# CHAIN OF THOUGHT LOGS (EXPANDERS)
# ─────────────────────────────────────────────
st.subheader("🕵 Chain-of-Thought Auditing Logs")
st.markdown("Deep-dive trace mapping reasoning steps through the stateful LangGraph.")

for step in result["cot_log"]:
    step_num = step["step"]
    title = step["title"]
    detail = step["detail"]
    data = step["data"]
    
    # Custom expander title with emojis for aesthetics
    icon = "🔍"
    if "Policy" in title or "ESG" in title:
        icon = "📋"
    elif "Predictive" in title:
        icon = "📈"
    elif "Anomaly" in title:
        icon = "🚨" if "ANOMALY DETECTED" in detail else "✓"
    elif "Compare" in title:
        icon = "⚖️"
    elif "Critic" in title:
        icon = "⚖️" if "PASSED" in detail else "🛡️"
    elif "Safety" in title:
        icon = "🚦"
    elif "recommendation" in title or "reviewer" in title:
        icon = "🎯"
        
    with st.expander(f"Step {step_num}: {icon} {title}"):
        st.markdown(f"**Reasoning Details:** {detail}")
        if data:
            st.markdown("**Structured Node Data:**")
            st.json(data)

st.markdown("---")

# ─────────────────────────────────────────────
# CITATIONS PANEL
# ─────────────────────────────────────────────
st.subheader("📚 Audit Citations & Verification Sources")
for citation in result.get("citations", []):
    if citation:
        st.markdown(f'<div class="citation-box">🗎 {citation}</div>', unsafe_allow_html=True)
