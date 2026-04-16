<h1 align="center">🛡️ Agentic Fraud Investigation System</h1>
<h3 align="center">LLM-Powered detection, reasoning, and human-in-the-loop feedback.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agentic-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Backend-SQLite-lightgrey" alt="SQLite">
</p>

## 📌 Project Overview
This project mimics an enterprise-grade Risk & Trust infrastructure (similar to internal tooling at top tech companies). It bridges the gap between traditional anomaly detection (Big Data/ML) and deep investigative reasoning (Agentic LLMs).

When a transaction is flagged by a traditional Machine Learning model, the **LangGraph Agent** dynamically interrogates a Vector Database of past historical fraud, queries the user's past SQL history, interprets the ML feature importance, and makes an automated, explainable decision. 

## 🏗️ System Architecture

```mermaid
graph TD
    A[Real-time Transaction] --> B(LightGBM / XGBoost Model)
    B -- Feature Importance & Score --> C{Is Anomalous?}
    C -- No --> D[Approve]
    C -- Yes --> E[LangGraph Investigation Agent]
    
    subgraph Agentic Workflow
    E <--> F[(SQLite: User Transaction History)]
    E <--> G[(ChromaDB: RAG Historical Fraud Precedents)]
    E <--> H[ML Feature Explainer Tool]
    end
    
    E -- Final JSON Decision --> I[Streamlit Dashboard Queue]
    I --> J[Human Analyst Review]
    J -- Agrees/Disagrees --> K[Feedback Loop Injection]
    K -- Updates Memory --> G
```

## ✨ Key Features
- **Dual-Layer Analytics**: XGBoost for rapid, high-volume anomaly filtering, paired with Llama-3/Mistral via LangGraph for deep-dive investigation.
- **RAG-Enabled Precedent Search**: Agent recalls historical fraud topologies from a Chroma Vector Database to inform new decisions.
- **Automated ROI Evaluation**: Includes an evaluation suite (`src/evaluation.py`) calculating Agent decision accuracy, false-positive reduction rates, and total human analyst hours saved.
- **RL Feedback Loop**: Disagreements made via the Streamlit UI inject context back into the RAG engine to "steer" future predictions without massive DPO/PPO re-trainings.

## 🚀 Live Demo
**👉 Try the live interactive dashboard:** [Live Demo on Streamlit Cloud](https://llm-powered-fraud-investigation-agent-4yk3ytrzotyxjxqvx2ernu.streamlit.app/)

### Running Locally
1. Ensure your local LLM server (e.g. Ollama) is running:
`ollama run llama3`
2. Run the dashboard:
`streamlit run app.py`

## 🛠️ Repository Structure
- `data/` : SQLite Databases and simulated transaction CSVs.
- `models/` : XGBoost models, global feature artifacts, and Vector DB.
- `src/rag_setup.py` : Initializes RAG via `HuggingFaceEmbeddings`.
- `src/agent.py` : StateGraph containing Agent Node constraints & Tools.
- `src/tools.py` : SQL Retriever & ML JSON Explainers.
- `src/evaluation.py` : Evaluates LLM against ground truth.
- `app.py` : Multi-tab Streamlit UX.

## 🤝 Next Steps / Improvements
- True Multi-Agent collaboration (e.g., passing between a 'Network Graph Agent' and an 'IP Risk Agent').
- RAGAS implementation for Faithfulness and Context Precision scoring.
