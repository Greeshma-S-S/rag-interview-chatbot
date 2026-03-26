"""
Streamlit frontend for the AI Engineer RAG Chatbot.
Supports both QA mode and Mock Interview mode.

Run with:
    streamlit run frontend/streamlit_app.py
"""

import sys
import time
from pathlib import Path
from typing import List

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Engineer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

WELCOME_QA = """
👋 **Welcome to the AI Engineer Chatbot!**

I'm your expert AI/ML knowledge assistant powered by RAG — grounded in the latest research papers on:
- 🔍 Retrieval-Augmented Generation (RAG)
- 🤖 Agentic AI & LLM Agents
- 🧠 Transformers & LLM architectures
- ⚙️ Fine-tuning (LoRA, QLoRA)
- 📊 Model evaluation & RAGAS
- 🚀 MLOps & production AI systems

Ask me anything — or switch to **Mock Interview Mode** to practice for your AI Engineer interview!
"""

WELCOME_INTERVIEW = """
🎯 **Mock Interview Mode Activated!**

I'll simulate a real AI Engineer technical interview. Here's how it works:
1. Tell me your **experience level** (Junior / Mid / Senior)
2. Choose a **topic area** or say "general"
3. I'll ask one question at a time and give you detailed feedback
4. After 5 questions, you'll get a full assessment

**Ready to begin?** Type your level and preferred topic!

*Example: "I'm a Mid-level engineer, let's focus on RAG pipelines"*
"""

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #a0c4ff; margin: 0.3rem 0 0; font-size: 0.9rem; }

    .chat-user {
        background: #e3f2fd;
        border-radius: 15px 15px 4px 15px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        border-left: 3px solid #1976d2;
    }
    .chat-assistant {
        background: #f8f9fa;
        border-radius: 15px 15px 15px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        border-left: 3px solid #00897b;
    }
    .source-badge {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #2e7d32;
        margin: 2px;
        display: inline-block;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stChatInput > div { border-radius: 20px !important; }
    .status-ready { color: #2e7d32; font-weight: bold; }
    .status-not-ready { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── State management ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "mode": "qa",
        "api_status": None,
        "doc_count": 0,
        "total_queries": 0,
        "avg_latency_ms": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def call_chat_api(question: str, history: List[dict], mode: str, top_k: int) -> dict | None:
    endpoint = "/chat" if mode == "qa" else "/interview"
    payload = {
        "question": question,
        "chat_history": history,
        "top_k": top_k,
    }
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the backend is running:\n`python app/main.py`")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Mode selector
    mode = st.radio(
        "🎯 Mode",
        options=["qa", "interview"],
        format_func=lambda x: "💬 Q&A Assistant" if x == "qa" else "🎤 Mock Interview",
        index=0 if st.session_state.mode == "qa" else 1,
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Retrieval settings
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=15, value=5)

    st.divider()

    # API status
    st.markdown("### 🔌 API Status")
    health = check_api_health()
    if health:
        status_class = "status-ready" if health.get("pipeline_ready") else "status-not-ready"
        status_text = "✅ Ready" if health.get("pipeline_ready") else "⚠️ Not Ready"
        st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
        st.caption(f"Docs: {health.get('doc_count', 0):,} chunks")
        st.caption(f"Model: {health.get('model', 'N/A')}")
    else:
        st.markdown('<span class="status-not-ready">❌ API Offline</span>', unsafe_allow_html=True)
        st.caption("Start API: `python app/main.py`")

    st.divider()

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Avg Latency", f"{st.session_state.avg_latency_ms:.0f}ms")

    st.divider()

    # Controls
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency_ms = 0.0
        st.rerun()

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────
mode_label = "💬 Q&A Assistant" if st.session_state.mode == "qa" else "🎤 Mock Interview"
st.markdown(f"""
<div class="main-header">
    <h1>🤖 AI Engineer Chatbot</h1>
    <p>RAG-powered assistant · {mode_label} · Powered by OpenAI GPT-4o</p>
</div>
""", unsafe_allow_html=True)

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        welcome = WELCOME_QA if st.session_state.mode == "qa" else WELCOME_INTERVIEW
        st.markdown(welcome)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
                    # Show sources if available
                    if msg.get("sources"):
                        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                            for src in msg["sources"]:
                                st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)
                            if msg.get("latency_ms"):
                                st.caption(f"⏱ Response time: {msg['latency_ms']:.0f}ms")

# ── Input ─────────────────────────────────────────────────────────────────────
placeholder = (
    "Ask about RAG, Transformers, LLMs, Agents, LoRA, RAGAS …"
    if st.session_state.mode == "qa"
    else "Start the interview or answer the current question …"
)

if prompt := st.chat_input(placeholder):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Build history for API
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    # Call API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Retrieving context and generating answer …"):
            result = call_chat_api(
                question=prompt,
                history=history,
                mode=st.session_state.mode,
                top_k=top_k,
            )

        if result:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            latency = result.get("latency_ms", 0.0)

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)
                    st.caption(f"⏱ Response time: {latency:.0f}ms")

            # Update session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency_ms": latency,
            })

            # Update stats
            n = st.session_state.total_queries
            prev_avg = st.session_state.avg_latency_ms
            st.session_state.total_queries = n + 1
            st.session_state.avg_latency_ms = (prev_avg * n + latency) / (n + 1)
        else:
            err_msg = "⚠️ Failed to get a response. Check that the API is running."
            st.error(err_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "sources": [],
            })

    st.rerun()


# ── Quick start guide (when API is offline) ───────────────────────────────────
if not health:
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Download AI research papers
python scripts/download_pdfs.py

# 4. Build the vector store
python scripts/ingest_data.py

# 5. Start the API backend
python app/main.py

# 6. In another terminal, start this UI
streamlit run frontend/streamlit_app.py
```
        """)
