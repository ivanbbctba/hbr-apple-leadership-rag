import streamlit as st
from src.hbr_apple_rag.response_engine import ResponseEngine
from src.hbr_apple_rag.prompts import QUESTIONS, PROMPT_ENGINEERING_SYSTEM_PROMPT

st.set_page_config(page_title="HBR RAG Comparison", layout="wide")
st.title("HBR Article Analysis")
st.markdown("### Raw LLM vs Prompt Engineering vs RAG")
st.markdown("---")


def run_comparison(question_text: str):
    """Run all three approaches and display them side by side."""

    col1, col2, col3 = st.columns(3)

    # =====================
    # 1. Raw LLM
    # =====================
    with col1:
        st.subheader("1. Raw LLM")
        with st.expander("System Prompt", expanded=False):
            st.code("No system prompt used", language="text")
        st.caption(f"**User Prompt:** {question_text}")

        with st.spinner("Running Raw LLM..."):
            raw_engine = ResponseEngine(mode="raw")
            raw_answer = raw_engine.respond(question_text)
        st.write(raw_answer)

    # =====================
    # 2. Prompt Engineering
    # =====================
    with col2:
        st.subheader("2. Prompt Engineering")
        with st.expander("System Prompt", expanded=False):
            st.code(PROMPT_ENGINEERING_SYSTEM_PROMPT, language="text")
        st.caption(f"**User Prompt:** {question_text}")

        with st.spinner("Running Prompt Engineering..."):
            prompt_engine = ResponseEngine(mode="prompt_eng")
            prompt_answer = prompt_engine.respond(question_text)
        st.write(prompt_answer)

    # =====================
    # 3. RAG (with context)
    # =====================
    with col3:
        st.subheader("3. RAG")
        with st.expander("System Prompt", expanded=False):
            st.code(PROMPT_ENGINEERING_SYSTEM_PROMPT, language="text")

        with st.spinner("Running RAG..."):
            rag_engine = ResponseEngine(mode="rag")
            rag_answer = rag_engine.respond(question_text)

            # Get retrieved context
            relevant_docs = rag_engine._retriever.invoke(question_text) if rag_engine._retriever else []

        # Show retrieved context (collapsed by default)
        with st.expander(f"Retrieved Context ({len(relevant_docs)} chunks)", expanded=False):
            if relevant_docs:
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                    st.markdown("---")
            else:
                st.info("No context retrieved.")

        st.caption(f"**User Prompt:** {question_text} + Retrieved Context")
        st.write(rag_answer)


# Tabs
tab1, tab2, tab3 = st.tabs([
    "Q1: Authors & Publisher",
    "Q2: Leadership Characteristics",
    "Q3: Apple's Leadership & Innovation"
])

with tab1:
    st.markdown(f"**Question:** {QUESTIONS['Q1']}")
    if st.button("Run Comparison", key="q1"):
        run_comparison(QUESTIONS["Q1"])

with tab2:
    st.markdown(f"**Question:** {QUESTIONS['Q2']}")
    if st.button("Run Comparison", key="q2"):
        run_comparison(QUESTIONS["Q2"])

with tab3:
    st.markdown(f"**Question:** {QUESTIONS['Q3']}")
    if st.button("Run Comparison", key="q3"):
        run_comparison(QUESTIONS["Q3"])