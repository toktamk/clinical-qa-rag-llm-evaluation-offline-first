from __future__ import annotations

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"


st.set_page_config(
    page_title="Offline Clinical RAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Offline-First Clinical RAG System")
st.caption(
    "Evaluation-driven, citation-grounded RAG with offline retrieval, "
    "verification, and abstention."
)

with st.sidebar:
    st.header("Configuration")

    api_url = st.text_input("API URL", value=API_URL)

    generation_mode = st.selectbox(
        "Generation mode",
        ["extractive", "llm", "extract_then_rewrite"],
        index=0,
    )

    retriever = st.selectbox(
        "Retriever",
        ["hybrid", "bm25", "dense"],
        index=0,
    )

    fusion_method = st.selectbox(
        "Fusion method",
        ["weighted", "rrf"],
        index=0,
    )

    top_k = st.slider("Top-k retrieved chunks", 1, 20, 5)

    max_context_chunks = st.slider("Max context chunks", 1, 10, 6)

    mock_llm = st.checkbox("Use MockLLM", value=False)

    st.markdown("---")
    st.warning("Research demo only. Not for clinical decision-making.")


query = st.text_area(
    "Clinical-style question",
    value="What monitoring is recommended after Therapy A?",
    height=100,
)

run_button = st.button("Run RAG Query", type="primary")


def post_query() -> dict:
    payload = {
        "query": query,
        "retriever": retriever,
        "fusion_method": fusion_method,
        "generation_mode": generation_mode,
        "top_k": top_k,
        "max_context_chunks": max_context_chunks,
        "mock_llm": mock_llm,
    }

    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


if run_button:
    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("Running retrieval, generation, and verification..."):
        try:
            result = post_query()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Start it with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error: {exc}")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. Try extractive mode or reduce top-k.")
            st.stop()

    generated = result.get("generated_answer", {})
    verification = result.get("verification", {})
    retrieved_chunks = result.get("retrieved_chunks", [])
    latency_ms = result.get("api_latency_ms")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Generation mode", generation_mode)
    col2.metric("Retriever", retriever)
    col3.metric("Verifier", verification.get("verification_status", "unknown"))
    col4.metric("Latency", f"{latency_ms:.2f} ms" if latency_ms else "N/A")

    st.subheader("Answer")

    if generated.get("abstained"):
        st.warning(generated.get("answer", "The system abstained."))
    else:
        st.success(generated.get("answer", "No answer returned."))

    citations = generated.get("citations") or []

    st.subheader("Citations")
    if citations:
        st.code(", ".join(citations))
    else:
        st.info("No citations returned.")

    st.subheader("Verification")

    verification_col1, verification_col2 = st.columns(2)

    with verification_col1:
        st.write("Invalid citations")
        st.json(verification.get("invalid_citations", []))

        st.write("Missing citations")
        st.json(verification.get("missing_citations", []))

    with verification_col2:
        st.write("Unsupported claims")
        st.json(verification.get("unsupported_claims", []))

        st.write("Safety warnings")
        st.json(verification.get("safety_warnings", []))

    st.subheader("Retrieved Evidence")

    if not retrieved_chunks:
        st.info("No retrieved chunks returned.")
    else:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = chunk.get("chunk_id", "unknown")
            title = chunk.get("title", "Untitled")
            section = chunk.get("section", "Unknown section")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")

            with st.expander(f"{idx}. {chunk_id} | {section} | score={score:.4f}"):
                st.markdown(f"**Document:** {title}")
                st.markdown(f"**Section:** {section}")
                st.write(text)

    with st.expander("Raw API response"):
        st.json(result)