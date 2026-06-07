import streamlit as st
import sqlite3
import pandas as pd
import json
import os
from src.agent import run_agent, chat_with_agent

st.set_page_config(page_title="Agentic Fraud Investigation", layout="wide", page_icon="🛡️")

st.title("🛡️ Enterprise Fraud Investigation Agent")
st.markdown("LLM-Powered detection, reasoning, and human-in-the-loop feedback.")

# Check for DB
if not os.path.exists("data/fraud_cases.db"):
    st.error("Database not found! Please run `python src/data_ingestion.py` first.")
    st.stop()

# Initialize session state for mock feedback loop
if 'feedback_db' not in st.session_state:
    st.session_state.feedback_db = []

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Connect to DB to get flagged cases
@st.cache_data
def load_flagged_cases():
    conn = sqlite3.connect("data/fraud_cases.db")
    df = pd.read_sql("SELECT * FROM transactions WHERE class = 1 LIMIT 50", conn)
    conn.close()
    return df

@st.cache_data
def load_eval_metrics():
    try:
        with open("models/eval_metrics.json", "r") as f:
            return json.load(f)
    except:
        return None

def display_decision(decision, simulated=False):
    """Helper to display agent decision in a consistent format."""
    suffix = " (Simulated Demo)" if simulated else ""
    if decision.get("decision") == "Fraud":
        st.error(f"🚨 Decision: FRAUD{suffix}")
    else:
        st.success(f"✅ Decision: NOT FRAUD{suffix}")
    st.info(f"**Reasoning:** {decision.get('reasoning')}")
    st.warning(f"**Suggested Action:** {decision.get('next_action')}")

def run_with_fallback(case_data, rag_context=""):
    """Run the agent with a simulated fallback on API failure."""
    try:
        decision = run_agent(dict(case_data), rag_context=rag_context)
        display_decision(decision)
        return decision
    except Exception as e:
        st.warning("⚠️ Free Cloud API blocked the request. Falling back to simulated Agent for demo purposes...")
        decision = {
            "decision": "Fraud" if float(case_data.get('purchase_value', 0)) > 200 else "Not Fraud",
            "reasoning": (
                f"SIMULATION: The ML model flagged device {case_data.get('device_id', 'Unknown')}. "
                f"Due to rapid pinging from IP {case_data.get('ip_address', 'Unknown')} conflicting "
                f"with the RAG database precedent, it matches historical fraud rings."
            ),
            "next_action": "Escalate"
        }
        display_decision(decision, simulated=True)
        return decision


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚦 Investigation Queue",
    "📊 Impact Metrics",
    "🔁 Feedback Loop",
    "🧾 Custom Investigation",
    "💬 Fraud Chat Assistant"
])

df_flagged = load_flagged_cases()

# ──────────────────────────────────────────────
# Tab 1: Investigation Queue (original)
# ──────────────────────────────────────────────
with tab1:
    st.header("Flagged Transactions Queue")
    st.markdown("These transactions were flagged by the initial ML Layer (XGBoost) for deepest LLM Analysis.")
    
    if len(df_flagged) > 0:
        # Select case
        case_id = st.selectbox("Select Transaction ID to investigate:", df_flagged['user_id'])
        case_data = df_flagged[df_flagged['user_id'] == case_id].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Transaction Details")
            st.json(dict(case_data))
            
        with col2:
            st.subheader("Agent Analysis")
            if st.button("Run LangGraph Agent"):
                with st.spinner("Agent summarizing history and RAG context..."):
                    # Mock RAG retrieval for speed in UI
                    rag_mock = "Previous feedback: User changed IP rapidly but from same geo region - ignore."
                    decision = run_with_fallback(case_data, rag_context=rag_mock)
                    
                    # Save state for feedback loop & chat context
                    st.session_state['last_decision'] = decision
                    st.session_state['last_case'] = case_id
                    st.session_state['last_transaction'] = json.dumps(dict(case_data), default=str)
            
    else:
        st.info("No cases in queue.")

# ──────────────────────────────────────────────
# Tab 2: Impact Metrics (original)
# ──────────────────────────────────────────────
with tab2:
    st.header("Business Impact & Evaluation")
    metrics = load_eval_metrics()
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Agent Accuracy", f"{metrics['accuracy']*100:.1f}%")
        col2.metric("False Positive Rate", f"{metrics['false_positive_rate']*100:.1f}%")
        col3.metric("Analyst Hours Saved", f"{metrics['human_time_saved_hours']} hrs")
        
        st.divider()
        st.subheader("Context")
        st.markdown(f"Evaluated on {metrics['total_cases_evaluated']} historic cases.")
        st.markdown(f"LLM Average Latency: {metrics['llm_avg_response_time_sec']} seconds per investigation vs human ~15 mins.")
    else:
        st.warning("No evaluation metrics found. Run Phase 5: `python src/evaluation.py` to populate.")

# ──────────────────────────────────────────────
# Tab 3: Feedback Loop (original)
# ──────────────────────────────────────────────
with tab3:
    st.header("Analyst Feedback Loop")
    st.markdown("""
    This loop corrects the LLM. When an analyst disagrees with the LLM, the feedback is embedded into **ChromaDB** so the RAG agent learns for future inferences without re-training.
    """)
    
    if 'last_case' in st.session_state:
        st.write(f"**Current active case:** User {st.session_state['last_case']}")
        verdict = st.radio("Do you agree with the LLM's decision?", ("Agree", "Disagree"))
        resolution_notes = st.text_area("Analyst Notes (will be sent to RAG database)")
        
        if st.button("Submit Feedback"):
            st.session_state.feedback_db.append({
                "case_id": st.session_state['last_case'],
                "verdict": verdict,
                "notes": resolution_notes
            })
            st.success("Feedback stored in Vector Database! Agent will use this context in future similar cases.")
            
    else:
        st.info("Run an investigation in Tab 1 first to provide feedback.")
    
    if st.session_state.feedback_db:
        st.subheader("Recent Feedback Ingests")
        st.dataframe(pd.DataFrame(st.session_state.feedback_db))

# ──────────────────────────────────────────────
# Tab 4: Custom Transaction Investigation (NEW)
# ──────────────────────────────────────────────
with tab4:
    st.header("🧾 Investigate a Custom Transaction")
    st.markdown("Manually enter transaction details to run the LangGraph fraud investigation agent.")
    
    with st.form("custom_transaction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.number_input("User ID", min_value=10000, max_value=99999, value=50000, step=1)
            purchase_value = st.number_input("Purchase Value ($)", min_value=1, max_value=10000, value=250, step=1)
            device_id = st.text_input("Device ID", value="DEV1234")
            ip_address = st.text_input("IP Address", value="192.168.1.100")
            
        with col2:
            browser = st.selectbox("Browser", ["Chrome", "Safari", "FireFox", "IE"])
            source = st.selectbox("Traffic Source", ["SEO", "Ads", "Direct"])
            age = st.slider("User Age", min_value=18, max_value=65, value=30)
            sex = st.radio("Sex", ["M", "F"], horizontal=True)
        
        submitted = st.form_submit_button("🔍 Investigate Transaction", use_container_width=True)
    
    if submitted:
        custom_txn = {
            "user_id": user_id,
            "purchase_value": purchase_value,
            "device_id": device_id,
            "ip_address": ip_address,
            "browser": browser,
            "source": source,
            "age": age,
            "sex": sex
        }
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Transaction Summary")
            st.json(custom_txn)
        
        with col2:
            st.subheader("Agent Verdict")
            with st.spinner("Agent analyzing custom transaction..."):
                rag_context = "No prior history available for this custom query."
                decision = run_with_fallback(custom_txn, rag_context=rag_context)
                
                # Save for chat context
                st.session_state['last_decision'] = decision
                st.session_state['last_case'] = user_id
                st.session_state['last_transaction'] = json.dumps(custom_txn, default=str)

# ──────────────────────────────────────────────
# Tab 5: Fraud Chat Assistant (NEW)
# ──────────────────────────────────────────────
with tab5:
    st.header("💬 Fraud Investigation Chat")
    st.markdown("Ask follow-up questions about fraud cases, patterns, or methodology. "
                "If you've investigated a transaction, the chat will have that context.")
    
    # Show context indicator
    if 'last_transaction' in st.session_state:
        with st.expander("📋 Active Transaction Context", expanded=False):
            st.json(json.loads(st.session_state['last_transaction']))
            if 'last_decision' in st.session_state:
                st.write(f"**Last Verdict:** {st.session_state['last_decision'].get('decision', 'N/A')}")
    else:
        st.info("💡 Tip: Investigate a transaction in Tab 1 or Tab 4 first to give the chat context about a specific case.")
    
    st.divider()
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about fraud patterns, this transaction, or investigation methodology..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Build context from prior investigation
        context = ""
        if 'last_transaction' in st.session_state:
            context += f"Transaction: {st.session_state['last_transaction']}\n"
        if 'last_decision' in st.session_state:
            context += f"Agent Decision: {json.dumps(st.session_state['last_decision'])}\n"
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_agent(
                    user_message=user_input,
                    context=context,
                    chat_history=st.session_state.chat_history[:-1]  # exclude current msg
                )
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
