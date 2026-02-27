"""
Document Q&A Assistant — Streamlit UI (main entry-point)
Run:  streamlit run app.py

Advanced RAG pipeline with:
  • Adaptive RAG Router  (auto-selects pipeline depth per query)
  • Self-RAG             (skips retrieval when not needed)
  • Streaming answers    (token-by-token display)
  • Conversation summarization (progressive memory)
  • Semantic / Parent-Child / Recursive chunking
  • Hybrid search (vector + BM25 → RRF fusion)
  • LLM re-ranking, Corrective RAG, Grounding checks
  • Agentic RAG mode (tool-calling agent)
"""

import os
import time
from datetime import datetime

import streamlit as st

from config import (
    OPENAI_API_KEY,
    CHAT_MODELS,
    CHUNKING_STRATEGIES,
    PIPELINE_MODES,
    SUMMARIZATION_THRESHOLD,
    SUMMARY_KEEP_RECENT,
)
from document_loader import (
    load_uploaded_file,
    load_wikipedia,
    get_file_hash,
)
from chunking import SemanticChunker, ParentChildChunker, chunk_recursive
from vector_store import create_vector_store, build_collection_name
from qa_chain import (
    ask_question,
    prepare_rag_context,
    generate_answer_stream,
    direct_answer_stream,
    highlight_sources,
    TokenTracker,
)
from retrieval import check_grounding
from router import route_query, ROUTE_CONFIGS
from memory import summarize_history, should_summarize
from agent import build_agent, run_agent
from logger import logger

# ────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
    .source-box{background:#fff3e0;padding:8px;border-radius:5px;margin:5px 0;font-size:.9em}
    .confidence-high{color:#2e7d32;font-weight:bold}
    .confidence-medium{color:#f57c00;font-weight:bold}
    .confidence-low{color:#d32f2f;font-weight:bold}
    .metric-card{background:#f8f9fa;padding:12px 16px;border-radius:8px;
                  border-left:4px solid #1976d2;margin:4px 0}
    .pipeline-step{background:#e8f5e9;padding:4px 8px;border-radius:4px;
                   margin:2px 0;font-size:.85em}
    .route-badge{display:inline-block;padding:4px 12px;border-radius:12px;
                 font-size:.85em;font-weight:600;margin:4px 0}
    .route-simple{background:#e3f2fd;color:#1565c0}
    .route-moderate{background:#fff3e0;color:#e65100}
    .route-complex{background:#fce4ec;color:#c62828}
    .route-direct{background:#e8f5e9;color:#2e7d32}
</style>
""",
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────
# Session state defaults
# ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "conversation_history": [],
    "conversation_summary": "",    # progressive summary of older messages
    "vector_store": None,
    "document_count": 0,
    "documents_loaded": [],
    "all_chunks": [],
    "parent_map": {},
    "retrieval_cache": {},
    "token_tracker": TokenTracker(),
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ────────────────────────────────────────────────────────────
# Sidebar — Settings
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    # --- API key (from st.secrets or env var — never exposed in UI) ---
    try:
        api_key = st.secrets.get("openai_api_key", "") or OPENAI_API_KEY
    except Exception:
        api_key = OPENAI_API_KEY
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.divider()
    # --- Pipeline mode ---
    st.subheader("🧪 Pipeline Mode")
    pipeline_mode = st.radio(
        "Mode",
        PIPELINE_MODES,
        index=0,
        help=(
            "**🧠 Adaptive** — Auto-classifies each query and selects "
            "the optimal pipeline depth. Simple questions get fast answers; "
            "complex ones get the full pipeline.\n\n"
            "**🤖 Agent** — Autonomous tool-calling agent that decides "
            "which tools to use (document search, Wikipedia).\n\n"
            "**🔧 Manual** — Full control over every pipeline stage."
        ),
    )

    is_adaptive = "Adaptive" in pipeline_mode
    is_agent = "Agent" in pipeline_mode
    is_manual = "Manual" in pipeline_mode

    st.divider()

    # --- Model config ---
    st.subheader("🤖 Model")
    model_name = st.selectbox("Model", CHAT_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    st.divider()

    # --- Chunking strategy ---
    st.subheader("🧩 Chunking")
    chunking_strategy = st.selectbox("Strategy", CHUNKING_STRATEGIES, index=1)
    chunk_size = st.slider("Chunk size", 200, 1500, 700, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 150, 50)

    st.divider()

    # --- Retrieval config ---
    st.subheader("🔍 Retrieval")
    retrieval_mode = st.selectbox("Vector strategy", ["similarity", "mmr"], index=0)
    k_docs = st.slider("Docs to retrieve (k)", 1, 10, 5)
    fetch_k = (
        st.slider("Fetch K (MMR)", 10, 30, 15)
        if retrieval_mode == "mmr"
        else k_docs * 2
    )

    # --- Manual-mode pipeline toggles ---
    if is_manual:
        st.divider()
        st.subheader("🔧 Manual Pipeline")
        use_hybrid = st.checkbox("Hybrid Search (BM25+Vector→RRF)", True,
                                 help="Fuses keyword and semantic search via RRF")
        use_reranking = st.checkbox("LLM Re-ranking", False,
                                    help="Re-score chunks with the LLM")
        use_crag = st.checkbox("Corrective RAG (CRAG)", True,
                               help="Grade retrieval; falls back to Wikipedia if poor")
        use_grounding = st.checkbox("Hallucination Check", False,
                                    help="Post-generation grounding verification")
        use_expansion = st.checkbox("Query Expansion", False,
                                    help="Rephrase question multiple ways")
        use_reformulation = st.checkbox("Question Reformulation", True,
                                        help="Rewrite follow-ups to be self-contained")

    st.divider()

    # --- Session controls ---
    st.subheader("🗂️ Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.conversation_history = []
            st.session_state.conversation_summary = ""
            st.session_state.retrieval_cache = {}
            st.session_state.token_tracker = TokenTracker()
            st.rerun()
    with c2:
        if st.button("🔄 Reset All"):
            for key in _DEFAULTS:
                st.session_state[key] = (
                    type(_DEFAULTS[key])()
                    if callable(type(_DEFAULTS[key]))
                    else _DEFAULTS[key]
                )
            st.session_state.token_tracker = TokenTracker()
            st.rerun()

    st.divider()

    # --- Stats ---
    st.subheader("📊 Statistics")
    st.metric("Documents loaded", st.session_state.document_count)
    st.metric("Chunks indexed", len(st.session_state.all_chunks))
    st.metric("Messages", len(st.session_state.conversation_history))
    st.metric("Cache entries", len(st.session_state.retrieval_cache))

    tracker: TokenTracker = st.session_state.token_tracker
    summary = tracker.summary(model_name)
    st.metric("Tokens used", summary["total_tokens"])
    st.metric("Est. cost (USD)", f"${summary['estimated_cost_usd']:.4f}")


# ────────────────────────────────────────────────────────────
# Title
# ────────────────────────────────────────────────────────────
st.title("📚 Document Q&A Assistant")
st.caption(
    "Upload documents (PDF · DOCX · TXT · MD · CSV) or search Wikipedia. "
    "**Adaptive RAG** auto-selects the optimal pipeline per query — "
    "with streaming answers, conversation memory, and intelligent routing."
)

# Show loaded documents
if st.session_state.documents_loaded:
    with st.expander(f"📁 Loaded documents ({len(st.session_state.documents_loaded)})"):
        for name in st.session_state.documents_loaded:
            st.markdown(f"- {name}")


# ────────────────────────────────────────────────────────────
# Chunking helper
# ────────────────────────────────────────────────────────────
def _chunk_docs(documents, strategy, size, overlap):
    """Apply the selected chunking strategy. Returns (chunks, parent_map)."""
    parent_map = {}
    if strategy == "Semantic (smart)":
        try:
            chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=size * 2)
            chunks = chunker.split_documents(documents)
        except Exception as exc:
            logger.warning("Semantic chunking failed (%s), falling back.", exc)
            chunks = chunk_recursive(documents, size, overlap)
    elif strategy == "Parent-Child (precise)":
        pc = ParentChildChunker(
            parent_chunk_size=size * 2,
            parent_overlap=overlap,
            child_chunk_size=size // 2,
            child_overlap=overlap // 2,
        )
        chunks, parent_map = pc.split_documents(documents)
    else:
        chunks = chunk_recursive(documents, size, overlap)
    return chunks, parent_map


# ────────────────────────────────────────────────────────────
# Document upload
# ────────────────────────────────────────────────────────────
col_up, col_wiki = st.columns([1, 1])

with col_up:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, MD, CSV)",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Process Documents", type="primary"):
        all_docs = []
        with st.spinner("Loading files…"):
            progress = st.progress(0)
            for idx, f in enumerate(uploaded_files):
                docs = load_uploaded_file(f)
                if docs:
                    all_docs.extend(docs)
                    if f.name not in st.session_state.documents_loaded:
                        st.session_state.documents_loaded.append(f.name)
                progress.progress((idx + 1) / len(uploaded_files))

        if all_docs:
            with st.spinner(f"Chunking ({chunking_strategy})…"):
                chunks, parent_map = _chunk_docs(
                    all_docs, chunking_strategy, chunk_size, chunk_overlap
                )
                st.info(f"📄 {len(chunks)} chunks ({chunking_strategy})")

            with st.spinner("Creating embeddings…"):
                col_name = build_collection_name(
                    "|".join(f.name for f in uploaded_files)
                )
                vs = create_vector_store(chunks, collection_name=col_name)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.all_chunks = chunks
                    st.session_state.parent_map = parent_map
                    st.session_state.document_count += len(uploaded_files)
                    st.session_state.retrieval_cache = {}
                    st.success(
                        f"✅ {len(uploaded_files)} file(s) → {len(chunks)} chunks"
                    )
                else:
                    st.error("Embedding creation failed.")

with col_wiki:
    st.header("🌐 Wikipedia Search")
    wiki_query = st.text_input("Enter a Wikipedia topic")
    wiki_articles = st.slider("Articles to fetch", 1, 5, 2)

    if wiki_query and st.button("🔍 Search Wikipedia", type="primary"):
        with st.spinner("Searching…"):
            wiki_docs = load_wikipedia(wiki_query, max_docs=wiki_articles)
            if wiki_docs:
                chunks, parent_map = _chunk_docs(
                    wiki_docs, chunking_strategy, chunk_size, chunk_overlap
                )
                col_name = build_collection_name(wiki_query)
                vs = create_vector_store(chunks, collection_name=col_name)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.all_chunks = chunks
                    st.session_state.parent_map = parent_map
                    st.session_state.document_count += len(wiki_docs)
                    doc_name = f"Wikipedia: {wiki_query}"
                    if doc_name not in st.session_state.documents_loaded:
                        st.session_state.documents_loaded.append(doc_name)
                    st.session_state.retrieval_cache = {}
                    st.success(f"✅ {len(wiki_docs)} article(s) → {len(chunks)} chunks")
            else:
                st.warning("No Wikipedia results found.")


# ────────────────────────────────────────────────────────────
# Chat history helper — uses summarization when history is long
# ────────────────────────────────────────────────────────────
def _build_chat_history(model: str) -> str:
    """
    Build chat history string with progressive summarization.
    When the conversation is long, older messages are summarized
    by the LLM instead of being truncated.
    """
    from langchain_openai import ChatOpenAI

    messages = st.session_state.conversation_history
    if not messages:
        return ""

    if should_summarize(messages, threshold=SUMMARIZATION_THRESHOLD):
        try:
            llm = ChatOpenAI(model=model, temperature=0.0)
            history_str = summarize_history(
                messages,
                llm,
                keep_recent=SUMMARY_KEEP_RECENT,
                existing_summary=st.session_state.get("conversation_summary", ""),
            )
            # Cache the summary for incremental use
            if "[Earlier conversation summary]" in history_str:
                summary_part = history_str.split("[Recent messages]")[0]
                st.session_state.conversation_summary = summary_part
            return history_str
        except Exception as exc:
            logger.warning("Summarization failed, using raw history: %s", exc)

    # Fallback: last messages
    return "\n".join(
        f"{'Q' if m['role'] == 'user' else 'A'}: {m.get('raw_answer', m['content'])}"
        for m in messages[-6:]
    )


# ────────────────────────────────────────────────────────────
# Q&A Section
# ────────────────────────────────────────────────────────────
st.divider()
st.header("💬 Ask Questions")

if st.session_state.vector_store and api_key:
    # Show current mode
    mode_label = pipeline_mode.split("(")[0].strip()
    st.markdown(f"**Active mode:** {mode_label}")

    question = st.text_area(
        "Your question:",
        placeholder="What would you like to know about the loaded documents?",
        height=80,
    )

    c1, c2 = st.columns([1, 5])
    with c1:
        ask = st.button("🤔 Ask", type="primary")

    if ask and question:
        start_time = time.time()

        # Build chat history with summarization
        history_str = _build_chat_history(model_name)

        try:
            # ═══════════════════════════════════════════════
            # 🧠 ADAPTIVE MODE — auto-routes per query
            # ═══════════════════════════════════════════════
            if is_adaptive:
                # Step 1: Route the query
                with st.spinner("🧠 Classifying query complexity…"):
                    route_config = route_query(question, model_name=model_name)
                    route = route_config["route"]
                    route_desc = route_config["description"]

                pipeline_steps = [f"🧭 Adaptive Router → **{route_desc}**"]

                # Step 2: Self-RAG — no retrieval needed
                if route == "no_retrieval":
                    st.markdown(
                        f"<span class='route-badge route-direct'>{route_desc}</span>",
                        unsafe_allow_html=True,
                    )

                    with st.chat_message("assistant"):
                        stream = direct_answer_stream(
                            question,
                            history_str,
                            model_name=model_name,
                            temperature=temperature,
                        )
                        raw_answer = st.write_stream(stream)

                    raw_answer = str(raw_answer)  # narrow type from write_stream
                    pipeline_steps.append("⚡ Self-RAG: answered directly (no retrieval)")
                    pipeline_steps.append("📡 Streamed token-by-token")

                    elapsed = round(time.time() - start_time, 2)

                    st.session_state.conversation_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": raw_answer,
                        "raw_answer": raw_answer,
                        "sources": [],
                        "confidence": 0.8,
                        "response_time": elapsed,
                        "retrieval_grade": "n/a",
                        "grounding_verdict": "n/a",
                        "pipeline_steps": pipeline_steps,
                        "effective_question": question,
                        "route": route,
                    })

                # Step 2b: Retrieval-based answer
                else:
                    # Use router-selected pipeline settings
                    r_cfg = route_config

                    # Show route badge
                    badge_class = {
                        "simple": "route-simple",
                        "moderate": "route-moderate",
                        "complex": "route-complex",
                    }.get(route, "route-moderate")
                    st.markdown(
                        f"<span class='route-badge {badge_class}'>{route_desc}</span>",
                        unsafe_allow_html=True,
                    )

                    # Phase 1: Retrieve (non-streaming, with spinner)
                    with st.spinner("🔍 Retrieving & processing…"):
                        ctx = prepare_rag_context(
                            question,
                            st.session_state.vector_store,
                            all_chunks=st.session_state.all_chunks,
                            parent_map=st.session_state.parent_map or None,
                            model_name=model_name,
                            temperature=temperature,
                            retrieval_mode=retrieval_mode,
                            k=r_cfg.get("k", k_docs),
                            fetch_k=fetch_k,
                            use_hybrid=r_cfg["use_hybrid"],
                            use_reranking=r_cfg["use_reranking"],
                            use_crag=r_cfg["use_crag"],
                            use_expansion=r_cfg["use_expansion"],
                            use_reformulation=r_cfg["use_reformulation"],
                            chat_history=history_str,
                            cache=st.session_state.retrieval_cache,
                        )

                    pipeline_steps.extend(ctx["pipeline_steps"])

                    # Phase 2: Stream answer
                    with st.chat_message("assistant"):
                        stream = generate_answer_stream(
                            ctx["effective_question"],
                            ctx["relevant_docs"],
                            history_str,
                            ctx["llm"],
                        )
                        raw_answer = st.write_stream(stream)

                    pipeline_steps.append("📡 Streamed token-by-token")

                    raw_answer = str(raw_answer)  # narrow type from write_stream

                    # Phase 3: Post-generation grounding check (complex only)
                    grounding_verdict = "n/a"
                    if r_cfg.get("use_grounding") and raw_answer:
                        with st.spinner("🛡️ Checking grounding…"):
                            grounding_verdict = check_grounding(
                                raw_answer, ctx["relevant_docs"], ctx["llm"]
                            )
                        pipeline_steps.append(f"🛡️ Grounding: {grounding_verdict}")
                        if grounding_verdict == "unsupported":
                            st.warning(
                                "⚠️ The answer may not be fully supported by "
                                "the retrieved documents."
                            )

                    answer_with_sources = highlight_sources(
                        raw_answer, ctx["relevant_docs"]
                    )

                    elapsed = round(time.time() - start_time, 2)

                    # Store in conversation
                    source_names = []
                    for doc in ctx["relevant_docs"][:5]:
                        src = doc.metadata.get("source", "Unknown")
                        ci = doc.metadata.get("chunk_index", "?")
                        source_names.append(f"{src} (chunk {ci})")
                    source_names = list(dict.fromkeys(source_names))

                    st.session_state.conversation_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": answer_with_sources,
                        "raw_answer": raw_answer,
                        "sources": source_names,
                        "confidence": 0.7,  # placeholder for streaming
                        "response_time": elapsed,
                        "retrieval_grade": ctx["retrieval_grade"],
                        "grounding_verdict": grounding_verdict,
                        "pipeline_steps": pipeline_steps,
                        "effective_question": ctx["effective_question"],
                        "route": route,
                    })

            # ═══════════════════════════════════════════════
            # 🤖 AGENT MODE
            # ═══════════════════════════════════════════════
            elif is_agent:
                with st.spinner("🤖 Agent reasoning…"):
                    agent_exec = build_agent(
                        st.session_state.vector_store,
                        st.session_state.all_chunks,
                        model_name=model_name,
                        temperature=temperature,
                        k=k_docs,
                    )
                    agent_result = run_agent(
                        question,
                        agent_exec,
                        chat_history=st.session_state.conversation_history,
                    )

                with st.chat_message("assistant"):
                    st.markdown(agent_result["answer"])

                elapsed = round(time.time() - start_time, 2)

                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": agent_result["answer"],
                    "raw_answer": agent_result["answer"],
                    "sources": [],
                    "confidence": 0.7,
                    "response_time": elapsed,
                    "retrieval_grade": "n/a",
                    "grounding_verdict": "n/a",
                    "pipeline_steps": [
                        f"🤖 Agent used tools: {', '.join(agent_result['tools_used']) or 'none'}",
                        f"   Steps taken: {agent_result['steps']}",
                    ],
                    "effective_question": question,
                    "route": "agent",
                })

            # ═══════════════════════════════════════════════
            # 🔧 MANUAL MODE
            # ═══════════════════════════════════════════════
            elif is_manual:
                # Phase 1: Retrieve
                with st.spinner("🔍 Retrieving…"):
                    ctx = prepare_rag_context(
                        question,
                        st.session_state.vector_store,
                        all_chunks=st.session_state.all_chunks if use_hybrid else None,
                        parent_map=st.session_state.parent_map or None,
                        model_name=model_name,
                        temperature=temperature,
                        retrieval_mode=retrieval_mode,
                        k=k_docs,
                        fetch_k=fetch_k,
                        use_hybrid=use_hybrid,
                        use_reranking=use_reranking,
                        use_crag=use_crag,
                        use_expansion=use_expansion,
                        use_reformulation=use_reformulation,
                        chat_history=history_str,
                        cache=st.session_state.retrieval_cache,
                    )

                pipeline_steps = list(ctx["pipeline_steps"])

                # Phase 2: Stream answer
                with st.chat_message("assistant"):
                    stream = generate_answer_stream(
                        ctx["effective_question"],
                        ctx["relevant_docs"],
                        history_str,
                        ctx["llm"],
                    )
                    raw_answer = st.write_stream(stream)

                raw_answer = str(raw_answer)  # narrow type from write_stream
                pipeline_steps.append("📡 Streamed token-by-token")

                # Phase 3: Grounding check
                grounding_verdict = "n/a"
                if use_grounding and raw_answer:
                    with st.spinner("🛡️ Checking grounding…"):
                        grounding_verdict = check_grounding(
                            raw_answer, ctx["relevant_docs"], ctx["llm"]
                        )
                    pipeline_steps.append(f"🛡️ Grounding: {grounding_verdict}")

                answer_with_sources = highlight_sources(
                    raw_answer, ctx["relevant_docs"]
                )

                elapsed = round(time.time() - start_time, 2)

                source_names = []
                for doc in ctx["relevant_docs"][:5]:
                    src = doc.metadata.get("source", "Unknown")
                    ci = doc.metadata.get("chunk_index", "?")
                    source_names.append(f"{src} (chunk {ci})")
                source_names = list(dict.fromkeys(source_names))

                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer_with_sources,
                    "raw_answer": raw_answer,
                    "sources": source_names,
                    "confidence": 0.7,
                    "response_time": elapsed,
                    "retrieval_grade": ctx["retrieval_grade"],
                    "grounding_verdict": grounding_verdict,
                    "pipeline_steps": pipeline_steps,
                    "effective_question": ctx["effective_question"],
                    "route": "manual",
                })

            # ── Log and update ──────────────────────────
            logger.info(
                "Q: %s | mode=%s | time=%.1fs",
                question[:60],
                pipeline_mode.split("(")[0].strip(),
                time.time() - start_time,
            )

        except Exception as exc:
            logger.exception("Question failed: %s", exc)
            st.error(f"Error: {exc}")
            st.info("Try adjusting settings or rephrasing your question.")

    # ── Conversation display ────────────────────────────
    if st.session_state.conversation_history:
        st.subheader("📝 Conversation")

        # Export
        if st.button("📥 Export conversation"):
            export = ""
            for m in st.session_state.conversation_history:
                prefix = "Q" if m["role"] == "user" else "A"
                export += f"{prefix}: {m['content']}\n"
                if "response_time" in m:
                    export += (
                        f"   (time: {m['response_time']:.1f}s | "
                        f"route: {m.get('route', 'n/a')} | "
                        f"retrieval: {m.get('retrieval_grade', 'n/a')} | "
                        f"grounding: {m.get('grounding_verdict', 'n/a')})\n"
                    )
                export += "\n"
            st.download_button(
                "Download",
                export,
                file_name=f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        # Render (newest first)
        for msg in reversed(st.session_state.conversation_history[-20:]):
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

                    # Route badge
                    route = msg.get("route", "")
                    if route and route != "n/a":
                        route_labels = {
                            "no_retrieval": ("⚡ Direct", "route-direct"),
                            "simple": ("🔍 Simple", "route-simple"),
                            "moderate": ("🧠 Moderate", "route-moderate"),
                            "complex": ("🔬 Complex", "route-complex"),
                            "agent": ("🤖 Agent", "route-moderate"),
                            "manual": ("🔧 Manual", "route-moderate"),
                        }
                        label, cls = route_labels.get(route, (route, "route-moderate"))
                        st.markdown(
                            f"<span class='route-badge {cls}'>{label}</span>",
                            unsafe_allow_html=True,
                        )

                    # Metadata
                    conf = msg.get("confidence", 0)
                    emoji = "✅" if conf > 0.7 else "⚠️" if conf > 0.4 else "❌"
                    rt = msg.get("response_time", 0)
                    grade = msg.get("retrieval_grade", "n/a")
                    ground = msg.get("grounding_verdict", "n/a")

                    meta_parts = [f"{emoji} Confidence: {conf:.0%}", f"⏱ {rt:.1f}s"]
                    if grade != "n/a":
                        meta_parts.append(f"Retrieval: {grade}")
                    if ground != "n/a":
                        meta_parts.append(f"Grounding: {ground}")
                    st.caption("  ·  ".join(meta_parts))

                    # Show reformulated question
                    eff_q = msg.get("effective_question", "")
                    if eff_q and eff_q != question:
                        st.caption(f"🔄 Searched for: _{eff_q}_")

                    # Pipeline steps (expandable)
                    steps = msg.get("pipeline_steps", [])
                    if steps:
                        with st.expander("🔬 Pipeline steps"):
                            for step in steps:
                                st.markdown(
                                    f"<div class='pipeline-step'>{step}</div>",
                                    unsafe_allow_html=True,
                                )

elif not api_key:
    st.error(
        "❌ OpenAI API key not found. Set it via **Streamlit Secrets** "
        "(`openai_api_key`) or the `OPENAI_API_KEY` environment variable."
    )
else:
    st.info("📤 Upload a document or search Wikipedia to get started.")


# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:.85em'>"
    "Advanced RAG Pipeline · Adaptive Router · Self-RAG · Streaming · "
    "Hybrid Search · CRAG · Agentic · Conversation Memory<br>"
    "Powered by LangChain · OpenAI · ChromaDB · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
