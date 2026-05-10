# 🧠 Credit Risk AI — Multi-Agent Ensemble Scoring System

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![LangGraph](https://img.shields.io/badge/AI-LangGraph-blue?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203-orange?style=flat-square)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud-red?style=flat-square)](https://qdrant.tech/)

> An enterprise-grade credit risk assessment platform that bridges the gap between black-box machine learning and explainable AI. By combining a **Multi-Model ML Ensemble**, **Retrieval-Augmented Generation (RAG)**, and **Multi-Agent Orchestration**, this system delivers high-fidelity, policy-compliant, and fully justifiable lending decisions in real time.

---

## 🌟 Key Capabilities

| Feature | Description |
|---|---|
| **Multi-Model ML Ensemble** | Aggregates predictive probabilities from **Logistic Regression, Random Forest, and Gradient Boosting** models to generate a highly robust default risk score. |
| **Algorithmic Disagreement Detection** | Automatically flags high-variance predictions across models, triggering specialized deep-dive analysis by the LLM for edge-case applications. |
| **Agentic Workflow (LangGraph)** | Orchestrates specialized AI agents (Risk Analysis, Policy Retrieval, Decision Maker) in a deterministic, stateful pipeline. |
| **RAG-Powered Policy Compliance** | Utilizes **Qdrant Cloud** vector search to dynamically retrieve and enforce institutional lending policies (e.g., FOIR limits, LTV ratios). |
| **Explainable AI (XAI) Output** | Synthesizes numerical ML scores and retrieved policy text into natural language justifications, ensuring decisions are fully transparent to human underwriters. |
| **Real-Time Traceability** | Full step-by-step execution trace visible in the frontend, providing auditability for every algorithmic and AI-driven decision. |

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Client Application] -->|Applicant Data| B(FastAPI Backend)
    
    subgraph Machine Learning Pipeline
        B --> C{Ensemble Model Evaluator}
        C -->|w1| D[Logistic Regression]
        C -->|w2| E[Random Forest]
        C -->|w3| F[Gradient Boosting]
        D & E & F --> G[Weighted Probability Aggregation]
        G --> H[Disagreement Detection]
    end
    
    subgraph Agentic Orchestration (LangGraph)
        H --> I(Input Processor Node)
        I --> J(Risk Analysis Agent)
        
        subgraph RAG System
            J --> K(Policy Retrieval Agent)
            K <-->|Semantic Search| L[(Qdrant Vector DB)]
        end
        
        K --> M(Lending Decision Agent)
        M -->|Synthesize ML + Policies| N(Final Explainable Output)
    end
    
    N --> O[JSON API Response]
    O --> A
```

---

## 🔬 Machine Learning Pipeline

The system moves beyond simple rule-based or single-model scoring by employing a weighted ensemble approach:

- **Model Diversity:** Utilizes three distinct algorithms (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`) to capture both linear relationships and complex non-linear feature interactions.
- **Ensemble Weights:** Probabilities are combined using a tuned weighting strategy (e.g., 0.3 LR, 0.35 RF, 0.35 GB) to optimize the ROC-AUC and minimize false negatives in high-risk bands.
- **Disagreement Heuristic:** Calculates the maximum variance between model probabilities. If variance exceeds a predefined threshold (e.g., > 0.4), a `High Disagreement` flag is raised, signaling the Decision Agent to apply stricter scrutiny and manual review recommendations.

---

## 🤖 Agentic AI Implementation

Built on **LangGraph** and the **Groq API** (running `llama-3.1-8b-instant`), the orchestration layer mimics a human underwriting committee:

1. **ML Prediction Node:** Deterministically executes the ensemble pipeline.
2. **Risk Analysis Agent:** Interprets the financial ratios (DTI, FOIR) and ML scores.
3. **Policy Retrieval Agent:** Embeds the applicant's profile to query the Qdrant Vector DB for relevant internal banking guidelines.
4. **Decision Agent:** The final authoritative node. It receives the ensemble scores, the disagreement flag, and the retrieved policies to synthesize a final recommendation (`Approve`, `Reject`, or `Manual Review`) along with a detailed, human-readable justification.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Groq API Key](https://console.groq.com/keys)
- [Qdrant Cloud Account](https://cloud.qdrant.io/)

### 1. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

*Note: Ensure the pre-trained models (`logistic_model.pkl`, `rf_model.pkl`, `gb_model.pkl`) are present in `backend/models/`. You can generate them by running `python scripts/train_multi_models.py`.*

### 2. Frontend Setup

```bash
cd frontend
npm install
```

Optionally create `.env.local` inside `frontend/`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 3. Running Locally

**Terminal 1 — Backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** to launch the platform.

---

## 📂 Repository Structure

```
credit-risk-ai/
├── backend/
│   ├── app/
│   │   ├── graph/          # LangGraph multi-agent workflow
│   │   ├── routes/         # FastAPI endpoints
│   │   ├── services/       # ML Pipeline, Groq LLM, Qdrant RAG
│   │   └── main.py
│   ├── models/             # Serialized sklearn ensemble models
│   ├── scripts/            # Model training & data generation pipelines
│   └── notebooks/          # Exploratory Data Analysis (EDA) & Model Eval
├── frontend/
│   ├── app/                # Next.js App Router
│   ├── components/         # React components (Dashboard, Traces)
│   └── lib/                # API clients
├── docs/                   # Policy documents for Vector DB indexing
└── README.md
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.
