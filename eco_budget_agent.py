"""
Eco-Budgeting Agent — Microsoft Agents League Hackathon
=======================================================
Architecture: LangGraph stateful graph with CoT logging at every node.
Tools: FoundryIQ (policy + ESG), DataAnalyst (carbon projection), SupplierBenchmark.
Explainability: Every node appends to `cot_log` → consumable by Streamlit dashboard.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI  # swap for your preferred LLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """Central state passed between every node in the graph."""
    messages: Annotated[list, add_messages]

    # Procurement context
    request: dict[str, Any]                  # raw incoming procurement request

    # CoT explainability log — list of dicts, one per reasoning step
    cot_log: list[dict[str, Any]]

    # Outputs from tools
    policy_data: dict[str, Any]              # from Foundry IQ policy tool
    esg_data: dict[str, Any]                 # from Foundry IQ ESG tool
    projection: dict[str, Any]               # from DataAnalyst tool
    alternatives: list[dict[str, Any]]       # from SupplierBenchmark tool

    # Decision flags
    exceeds_cap: bool
    critic_passed: bool
    escalate: bool

    # Final output
    recommendation: dict[str, Any]


# ─────────────────────────────────────────────
# 2. CHAIN-OF-THOUGHT LOGGER
# ─────────────────────────────────────────────

def log_step(state: AgentState, step: int, title: str, detail: str, data: Any = None) -> list:
    """Append a structured CoT step to the log. Returns updated cot_log."""
    entry = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(),
        "title": title,
        "detail": detail,
        "data": data,
    }
    return state["cot_log"] + [entry]


# ─────────────────────────────────────────────
# 3. TOOL DEFINITIONS
# ─────────────────────────────────────────────

@tool
def foundry_iq_policy_tool(category: str, supplier_id: str) -> dict:
    """
    Fetch ESG policy documents and compliance rules from Foundry IQ.
    Returns policy thresholds, sustainability cap, and a document citation.

    Args:
        category: Procurement category (e.g. 'electronics', 'logistics')
        supplier_id: Supplier identifier string
    """
    # ── REPLACE with real Foundry IQ API call ──
    # Example: response = requests.get(FOUNDRY_IQ_URL, params={...}, headers={...})
    return {
        "policy_id": "POL-2024-ECO-001",
        "sustainability_cap_kg_co2_per_quarter": 5000.0,
        "min_esg_score": 65,
        "cost_variance_threshold_pct": 15.0,
        "citation": "Foundry IQ Policy DB › ECO Procurement Policy v3.2 (2024-Q1)",
        "supplier_esg_score": 72,          # live ESG score for supplier
        "supplier_esg_rating": "B+",
    }


@tool
def foundry_iq_esg_tool(supplier_id: str) -> dict:
    """
    Retrieve real-time ESG scores and risk flags for a given supplier from Foundry IQ.

    Args:
        supplier_id: Supplier identifier string
    """
    # ── REPLACE with real Foundry IQ ESG endpoint ──
    return {
        "supplier_id": supplier_id,
        "esg_score": 72,
        "environmental_score": 68,
        "social_score": 75,
        "governance_score": 73,
        "risk_flags": [],                   # e.g. ["carbon_intensive_supply_chain"]
        "citation": "Foundry IQ ESG › Supplier ESG Dashboard (live)",
        "last_updated": datetime.utcnow().isoformat(),
    }


@tool
def data_analyst_tool(
    current_spend_kg_co2: float,
    monthly_trend: list[float],
    months_remaining_in_quarter: int,
) -> dict:
    """
    Predictive carbon footprint model: projects quarterly CO₂ from current trend.
    Uses linear regression on historical monthly data to forecast end-of-quarter.

    Args:
        current_spend_kg_co2: CO₂ already emitted this quarter (kg)
        monthly_trend: List of CO₂ values for past N months (kg)
        months_remaining_in_quarter: Months left in the current quarter
    """
    if len(monthly_trend) < 2:
        avg_rate = current_spend_kg_co2
    else:
        # Simple linear velocity: average month-on-month delta
        deltas = [monthly_trend[i+1] - monthly_trend[i] for i in range(len(monthly_trend)-1)]
        avg_rate = sum(deltas) / len(deltas)

    projected_total = current_spend_kg_co2 + avg_rate * months_remaining_in_quarter

    return {
        "current_kg_co2": current_spend_kg_co2,
        "avg_monthly_rate_kg": avg_rate,
        "months_remaining": months_remaining_in_quarter,
        "projected_quarterly_kg_co2": round(projected_total, 2),
        "model": "linear_velocity",
        "citation": "Eco-Budgeting Agent › DataAnalyst module (linear projection)",
    }


@tool
def supplier_benchmark_tool(category: str, exclude_supplier_id: str) -> list[dict]:
    """
    Find and rank alternative suppliers by ESG score, cost delta, and diversity index from Foundry IQ.

    Args:
        category: Procurement category
        exclude_supplier_id: Supplier to exclude (the one being reviewed)
    """
    # ── REPLACE with real Foundry IQ supplier search ──
    return [
        {
            "supplier_id": "SUP-B-042",
            "name": "GreenCore Materials",
            "esg_score": 88,
            "cost_delta_pct": +5.2,       # 5.2% more expensive
            "carbon_kg_per_unit": 1.1,
            "share_of_wallet_pct": 25.0,  # 25% of orders
            "citation": "Foundry IQ › Supplier Catalogue (category: electronics)",
        },
        {
            "supplier_id": "SUP-C-019",
            "name": "EcoLink Logistics",
            "esg_score": 81,
            "cost_delta_pct": +2.8,
            "carbon_kg_per_unit": 1.4,
            "share_of_wallet_pct": 15.0,  # 15% of orders
            "citation": "Foundry IQ › Supplier Catalogue (category: electronics)",
        },
    ]


# ─────────────────────────────────────────────
# 4. GRAPH NODES
# ─────────────────────────────────────────────

def node_analyze_request(state: AgentState) -> AgentState:
    """CoT Step 1 — Parse and validate the incoming procurement request."""
    req = state["request"]
    cot = log_step(
        state, step=1,
        title="Analyzing request",
        detail=f"Procurement request received for supplier '{req.get('supplier_id')}', "
               f"category '{req.get('category')}', quantity {req.get('quantity')} units.",
        data=req,
    )
    return {**state, "cot_log": cot}


def node_fetch_policy(state: AgentState) -> AgentState:
    """CoT Step 2 — Retrieve sustainability policy + ESG data from Foundry IQ."""
    req = state["request"]

    policy = foundry_iq_policy_tool.invoke({
        "category": req["category"],
        "supplier_id": req["supplier_id"],
    })
    esg = foundry_iq_esg_tool.invoke({"supplier_id": req["supplier_id"]})

    cot = log_step(
        state, step=2,
        title="Retrieving policy + ESG",
        detail=f"Policy fetched: cap={policy['sustainability_cap_kg_co2_per_quarter']} kg CO₂/quarter. "
               f"Supplier ESG score: {esg['esg_score']} ({esg.get('supplier_esg_rating', 'N/A')}). "
               f"Citation: {policy['citation']}",
        data={"policy": policy, "esg": esg},
    )
    return {**state, "policy_data": policy, "esg_data": esg, "cot_log": cot}


def node_predictive_model(state: AgentState) -> AgentState:
    """
    CoT Step 3 — THE EDGE: Run the DataAnalyst tool to project quarterly carbon footprint.
    Determines whether a Strategic Pivot is needed.
    """
    req = state["request"]
    policy = state["policy_data"]

    projection = data_analyst_tool.invoke({
        "current_spend_kg_co2": req.get("current_quarter_kg_co2", 3200.0),
        "monthly_trend": req.get("monthly_trend", [800.0, 950.0, 1100.0, 1450.0]),
        "months_remaining_in_quarter": req.get("months_remaining", 2),
    })

    cap = policy["sustainability_cap_kg_co2_per_quarter"]
    exceeds = projection["projected_quarterly_kg_co2"] > cap

    pivot_msg = (
        f"⚠ STRATEGIC PIVOT TRIGGERED — projected {projection['projected_quarterly_kg_co2']} kg "
        f"exceeds cap of {cap} kg."
        if exceeds else
        f"✓ Projection ({projection['projected_quarterly_kg_co2']} kg) within cap ({cap} kg)."
    )

    cot = log_step(
        state, step=3,
        title="Predictive modeling",
        detail=f"Quarterly CO₂ projection: {projection['projected_quarterly_kg_co2']} kg "
               f"(cap: {cap} kg). {pivot_msg} Citation: {projection['citation']}",
        data={"projection": projection, "exceeds_cap": exceeds},
    )
    return {**state, "projection": projection, "exceeds_cap": exceeds, "cot_log": cot}


def node_anomaly_detector(state: AgentState) -> AgentState:
    """
    CoT Step 4 — Anomaly Detector: computes a Z-score on the monthly trend.
    A spike > 2σ triggers an immediate warning in the CoT log before the critic even runs.
    """
    req = state["request"]
    trend = req.get("monthly_trend", [])
    
    z_score = 0.0
    is_anomaly = False
    
    if len(trend) >= 3:
        history = trend[:-1]
        latest = trend[-1]
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        std_dev = math.sqrt(variance)
        if std_dev > 0:
            z_score = (latest - mean) / std_dev
            if z_score > 2.0:
                is_anomaly = True
                
    warning_msg = (
        f" ⚠ ANOMALY DETECTED — Latest monthly footprint is a spike! Z-score = {z_score:.2f}σ (> 2σ threshold)."
        if is_anomaly else
        f" ✓ Trend is stable. Z-score = {z_score:.2f}σ (within ±2σ threshold)."
    )
    
    cot = log_step(
        state, step=4,
        title="Anomaly detection",
        detail=f"Calculated Z-score for monthly CO₂ trend: {z_score:.2f}σ. {warning_msg}",
        data={"z_score": round(z_score, 2), "is_anomaly": is_anomaly, "trend": trend},
    )
    return {**state, "cot_log": cot}


def node_compare_alternatives(state: AgentState) -> AgentState:
    """CoT Step 5 — Benchmark supplier alternatives from Foundry IQ."""
    req = state["request"]

    alternatives = supplier_benchmark_tool.invoke({
        "category": req["category"],
        "exclude_supplier_id": req["supplier_id"],
    })

    cot = log_step(
        state, step=5,
        title="Comparing alternatives",
        detail=f"Found {len(alternatives)} alternative supplier(s). "
               f"Top pick: {alternatives[0]['name']} (ESG: {alternatives[0]['esg_score']}, "
               f"cost delta: +{alternatives[0]['cost_delta_pct']}%). "
               f"Citation: {alternatives[0]['citation']}",
        data=alternatives,
    )
    return {**state, "alternatives": alternatives, "cot_log": cot}


def node_critic_loop(state: AgentState) -> AgentState:
    """
    CoT Step 6 — THE CRITIC: Evaluate the draft recommendation against guardrails.
    If it fails (poor ESG, cap breach, cost spike, or supplier concentration), self-correct to best alternative.
    """
    esg = state["esg_data"]
    policy = state["policy_data"]
    projection = state["projection"]
    alternatives = state.get("alternatives", [])
    req = state["request"]

    # Guardrail checks
    esg_fail = esg["esg_score"] < policy["min_esg_score"]
    cap_fail = state["exceeds_cap"]
    
    # Diversity concentration check: cap concentration at 40%
    current_concentration = req.get("supplier_concentration_pct", 45.0)
    concentration_fail = current_concentration > 40.0

    if esg_fail or cap_fail or concentration_fail:
        # Self-correct: select best alternative that fits concentration cap
        best_alt = None
        for alt in alternatives:
            if alt.get("share_of_wallet_pct", 0.0) <= 40.0:
                best_alt = alt
                break
        if not best_alt and alternatives:
            best_alt = alternatives[0]

        reason = []
        if esg_fail:
            reason.append(f"ESG score {esg['esg_score']} < minimum {policy['min_esg_score']}")
        if cap_fail:
            reason.append(f"Projected CO₂ exceeds quarterly cap")
        if concentration_fail:
            reason.append(f"Supplier concentration {current_concentration}% exceeds 40% cap")

        detail = (
            f"CRITIC FAILED guardrails: {'; '.join(reason)}. "
            f"Self-correcting to '{best_alt['name'] if best_alt else 'N/A'}' "
            f"(ESG: {best_alt['esg_score'] if best_alt else 'N/A'}, Concentration: {best_alt.get('share_of_wallet_pct', 'N/A')}%)."
        )
        critic_passed = False
    else:
        detail = (
            f"CRITIC PASSED: ESG score {esg['esg_score']} ≥ {policy['min_esg_score']} minimum. "
            f"Projection within cap. Supplier concentration {current_concentration}% meets 40% diversity cap. "
            f"Proceeding with original supplier."
        )
        critic_passed = True

    cot = log_step(
        state, step=6,
        title="Critic loop (self-evaluation)",
        detail=detail,
        data={
            "esg_fail": esg_fail,
            "cap_fail": cap_fail,
            "concentration_fail": concentration_fail,
            "critic_passed": critic_passed,
            "current_concentration_pct": current_concentration
        },
    )
    return {**state, "critic_passed": critic_passed, "cot_log": cot}


def node_safety_guardrail(state: AgentState) -> AgentState:
    """CoT Step 7 — Check for critical ESG or cost variance. Flag for human if needed."""
    esg = state["esg_data"]
    policy = state["policy_data"]
    req = state["request"]

    cost_variance = abs(req.get("quoted_price_delta_pct", 0))
    critical_esg = esg["esg_score"] < (policy["min_esg_score"] - 10)  # 10-pt buffer
    high_cost_variance = cost_variance > policy["cost_variance_threshold_pct"]

    escalate = critical_esg or high_cost_variance

    detail = (
        f"ESCALATING to human: {'Critical ESG score ' if critical_esg else ''}"
        f"{'High cost variance ' + str(cost_variance) + '%' if high_cost_variance else ''}."
        if escalate else
        f"Safety check passed. ESG: {esg['esg_score']}, cost variance: {cost_variance}%."
    )

    cot = log_step(
        state, step=7,
        title="Safety guardrail check",
        detail=detail,
        data={"escalate": escalate, "critical_esg": critical_esg, "high_cost_variance": high_cost_variance},
    )
    return {**state, "escalate": escalate, "cot_log": cot}


def node_synthesize_recommendation(state: AgentState) -> AgentState:
    """CoT Step 8 — Build the final, explainable recommendation with citations."""
    policy = state["policy_data"]
    esg = state["esg_data"]
    projection = state["projection"]
    alternatives = state.get("alternatives", [])
    critic_passed = state["critic_passed"]
    req = state["request"]

    if critic_passed:
        chosen_supplier = req["supplier_id"]
        chosen_name = req.get("supplier_name", chosen_supplier)
        chosen_esg = esg["esg_score"]
        chosen_concentration = req.get("supplier_concentration_pct", 45.0)
        rationale = "Original supplier meets all ESG, carbon cap, and supplier diversity requirements."
        citations = [policy["citation"], esg["citation"], projection["citation"]]
    else:
        # Re-find the chosen alternative
        best = None
        for alt in alternatives:
            if alt.get("share_of_wallet_pct", 0.0) <= 40.0:
                best = alt
                break
        if not best and alternatives:
            best = alternatives[0]
        best = best or {}

        chosen_supplier = best.get("supplier_id", "N/A")
        chosen_name = best.get("name", "N/A")
        chosen_esg = best.get("esg_score", 0)
        chosen_concentration = best.get("share_of_wallet_pct", 0.0)

        rationale = (
            f"Switched to '{chosen_name}' after critic loop detected guardrail failure. "
            f"ESG improvement: {esg['esg_score']} → {chosen_esg}. "
            f"Diversity improvement: Concentration reduced from {req.get('supplier_concentration_pct', 45.0)}% to {chosen_concentration}%."
        )
        citations = [best.get("citation", ""), policy["citation"], projection["citation"]]

    recommendation = {
        "decision": "APPROVE" if critic_passed else "APPROVE_ALTERNATIVE",
        "chosen_supplier_id": chosen_supplier,
        "chosen_supplier_name": chosen_name,
        "esg_score": chosen_esg,
        "chosen_concentration_pct": chosen_concentration,
        "projected_quarterly_co2_kg": projection["projected_quarterly_kg_co2"],
        "sustainability_cap_kg": policy["sustainability_cap_kg_co2_per_quarter"],
        "rationale": rationale,
        "citations": citations,
        "cot_log": state["cot_log"],  # full reasoning trace for dashboard
    }

    cot = log_step(
        state, step=8,
        title="Synthesizing recommendation",
        detail=f"Decision: {recommendation['decision']} → {chosen_name}. "
               f"ESG: {chosen_esg}, Concentration: {chosen_concentration}%. "
               f"CO₂ projection: {projection['projected_quarterly_kg_co2']} kg. "
               f"Rationale: {rationale}",
        data=recommendation,
    )
    return {**state, "recommendation": recommendation, "cot_log": cot}


def node_escalate(state: AgentState) -> AgentState:
    """Safety escalation node — package the full CoT log for human reviewer."""
    esg = state["esg_data"]
    projection = state["projection"]

    cot = log_step(
        state, step=8,
        title="Escalating to human reviewer",
        detail=f"Request flagged. ESG: {esg['esg_score']}, "
               f"projected CO₂: {projection.get('projected_quarterly_kg_co2', 'N/A')} kg. "
               f"Full CoT log attached for audit.",
        data={"reason": "critical_esg_or_cost_variance"},
    )

    recommendation = {
        "decision": "ESCALATE",
        "reason": "Critical ESG score or excessive cost variance detected.",
        "cot_log": state["cot_log"] + cot[-1:],
        "esg_score": esg["esg_score"],
        "projected_quarterly_co2_kg": projection.get("projected_quarterly_kg_co2"),
    }
    return {**state, "recommendation": recommendation, "cot_log": cot}


# ─────────────────────────────────────────────
# 5. ROUTING FUNCTIONS
# ─────────────────────────────────────────────

def route_after_safety(state: AgentState) -> Literal["escalate", "synthesize"]:
    return "escalate" if state["escalate"] else "synthesize"


def route_after_critic(state: AgentState) -> Literal["safety", "compare"]:
    """
    If critic failed and we haven't fetched alternatives yet, fetch them.
    Otherwise proceed to safety check.
    """
    if not state["critic_passed"] and not state.get("alternatives"):
        return "compare"
    return "safety"


# ─────────────────────────────────────────────
# 6. GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("analyze_request",     node_analyze_request)
    graph.add_node("fetch_policy",        node_fetch_policy)
    graph.add_node("predictive_model",    node_predictive_model)
    graph.add_node("anomaly_detector",    node_anomaly_detector)
    graph.add_node("compare_alternatives", node_compare_alternatives)
    graph.add_node("critic_loop",         node_critic_loop)
    graph.add_node("safety_guardrail",    node_safety_guardrail)
    graph.add_node("synthesize",          node_synthesize_recommendation)
    graph.add_node("escalate",            node_escalate)

    # Define edges (the reasoning flow)
    graph.set_entry_point("analyze_request")
    graph.add_edge("analyze_request",     "fetch_policy")
    graph.add_edge("fetch_policy",        "predictive_model")
    graph.add_edge("predictive_model",    "anomaly_detector")
    graph.add_edge("anomaly_detector",    "compare_alternatives")
    graph.add_edge("compare_alternatives", "critic_loop")

    # Conditional: critic passes → safety; critic fails → re-compare (already done above) → safety
    graph.add_conditional_edges(
        "critic_loop",
        lambda s: "safety",          # always go to safety after critic in this simplified flow
        {"safety": "safety_guardrail"},
    )
    graph.add_conditional_edges(
        "safety_guardrail",
        route_after_safety,
        {"escalate": "escalate", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("escalate",   END)

    return graph.compile()


# ─────────────────────────────────────────────
# 7. RUNNER + COT DISPLAY
# ─────────────────────────────────────────────

def print_cot_log(cot_log: list[dict]) -> None:
    """Pretty-print the Chain of Thought log (mirrors what your Streamlit dashboard will show)."""
    # Safe console print on Windows by resolving encoding errors
    print("\n" + "="*60)
    try:
        print("  CHAIN OF THOUGHT — Eco-Budgeting Agent")
    except UnicodeEncodeError:
        print("  CHAIN OF THOUGHT - Eco-Budgeting Agent")
    print("="*60)
    for entry in cot_log:
        try:
            print(f"\n[Step {entry['step']}] {entry['title']}")
        except UnicodeEncodeError:
            title_safe = entry['title'].encode('ascii', errors='replace').decode('ascii')
            print(f"\n[Step {entry['step']}] {title_safe}")
            
        detail = entry['detail']
        try:
            print(f"  {detail}")
        except UnicodeEncodeError:
            # Replace common problematic characters for safety
            safe_detail = (
                detail.replace("₂", "2")
                .replace("—", "-")
                .replace("✓", "[OK]")
                .replace("⚠", "[WARN]")
                .replace("❌", "[FAIL]")
                .replace("🎉", "[SUCCESS]")
                .replace("🗎", "[DOC]")
                .replace("→", "->")
            )
            try:
                print(f"  {safe_detail}")
            except Exception:
                print(f"  {safe_detail.encode('ascii', errors='replace').decode('ascii')}")
    print("\n" + "="*60)


def run_agent(request: dict) -> dict:
    """
    Entry point. Pass a procurement request dict, get back a recommendation with
    a full CoT log for your Streamlit explainability dashboard.

    Example request:
    {
        "supplier_id": "SUP-A-007",
        "supplier_name": "TechParts Inc.",
        "category": "electronics",
        "quantity": 500,
        "quoted_price_delta_pct": 8.0,         # % vs. baseline price
        "current_quarter_kg_co2": 3800.0,      # CO₂ emitted so far this quarter
        "monthly_trend": [900.0, 1100.0, 1300.0, 1500.0],
        "months_remaining": 2,
    }
    """
    agent = build_agent()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=json.dumps(request))],
        "request": request,
        "cot_log": [],
        "policy_data": {},
        "esg_data": {},
        "projection": {},
        "alternatives": [],
        "exceeds_cap": False,
        "critic_passed": True,
        "escalate": False,
        "recommendation": {},
    }

    final_state = agent.invoke(initial_state)
    print_cot_log(final_state["cot_log"])
    return final_state["recommendation"]


# ─────────────────────────────────────────────
# 8. STREAMLIT DASHBOARD HINTS
# ─────────────────────────────────────────────
#
# In your Streamlit app, consume the cot_log like this:
#
#   result = run_agent(request)
#   st.title("Eco-Budgeting Agent — Explainability Dashboard")
#   st.metric("Decision", result["decision"])
#   st.metric("Projected CO₂ (kg)", result["projected_quarterly_co2_kg"])
#   for step in result["cot_log"]:
#       with st.expander(f"Step {step['step']}: {step['title']}"):
#           st.write(step["detail"])
#           if step["data"]:
#               st.json(step["data"])
#   st.subheader("Citations")
#   for c in result.get("citations", []):
#       st.markdown(f"- {c}")
#
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_request = {
        "supplier_id": "SUP-A-007",
        "supplier_name": "TechParts Inc.",
        "category": "electronics",
        "quantity": 500,
        "quoted_price_delta_pct": 8.0,
        "current_quarter_kg_co2": 3800.0,
        "monthly_trend": [900.0, 1100.0, 1300.0, 1500.0],
        "months_remaining": 2,
        "supplier_concentration_pct": 45.0,
    }
    result = run_agent(sample_request)
    print("\nFINAL RECOMMENDATION:")
    print(json.dumps({k: v for k, v in result.items() if k != "cot_log"}, indent=2))
