import streamlit as st
from src.hbr_apple_rag.comparison import ComparisonEngine
from src.hbr_apple_rag.prompts import QUESTIONS, PROMPT_ENGINEERING_SYSTEM_PROMPT

st.set_page_config(page_title="HBR RAG Comparison", layout="wide")
st.title("HBR Article Analysis")
st.markdown("### Raw LLM vs Prompt Engineering vs RAG")
st.markdown("---")


def render_column(col, title: str, mode: str, question: str, engine: ComparisonEngine):
    """Helper to render a single comparison column with metrics."""
    with col:
        st.subheader(title)

        sys_prompt = "No system prompt used" if mode == "raw" else PROMPT_ENGINEERING_SYSTEM_PROMPT
        with st.expander("System Prompt", expanded=False):
            st.code(sys_prompt, language="text")

        with st.spinner(f"Running {title}..."):
            result = engine.run_mode(mode, question)

        if mode == "rag":
            with st.expander(f"Retrieved Context ({len(result.context)} chunks)", expanded=False):
                if result.context:
                    for i, doc in enumerate(result.context, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:700] + "..." if len(doc.page_content) > 700 else doc.page_content)
                        st.markdown("---")
                else:
                    st.info("No context was retrieved.")
            st.caption(f"**User Prompt:** {question} + Retrieved Context")
        else:
            st.caption(f"**User Prompt:** {question}")

        st.write(result.answer)
        st.info(f"⏱️ Response time: {result.response_time:.2f}s")


def display_comparison_results(question_text: str):
    """Display the three-way comparison in columns (used by both fixed and custom questions)."""
    engine = ComparisonEngine()
    col1, col2, col3 = st.columns(3)

    render_column(col1, "1. Raw LLM", "raw", question_text, engine)
    render_column(col2, "2. Prompt Engineering", "prompt_eng", question_text, engine)
    render_column(col3, "3. RAG", "rag", question_text, engine)


# =====================
# PHASE 1 - Fixed Questions (Tabs)
# =====================
tab1, tab2, tab3 = st.tabs([
    "Q1: Authors & Publisher",
    "Q2: Leadership Characteristics",
    "Q3: Apple's Leadership & Innovation"
])

with tab1:
    st.markdown(f"**Question:** {QUESTIONS['Q1']}")
    if st.button("Run Comparison", key="q1"):
        display_comparison_results(QUESTIONS["Q1"])

with tab2:
    st.markdown(f"**Question:** {QUESTIONS['Q2']}")
    if st.button("Run Comparison", key="q2"):
        display_comparison_results(QUESTIONS["Q2"])

with tab3:
    st.markdown(f"**Question:** {QUESTIONS['Q3']}")
    if st.button("Run Comparison", key="q3"):
        display_comparison_results(QUESTIONS["Q3"])


# =====================
# PHASE 2 - Custom Question Section
# =====================
st.markdown("---")
st.subheader("Try Your Own Question")

st.markdown(
    "Ask any question about the article. It will use the **same system prompt** and "
    "retrieve context from the **same document** used above."
)

custom_question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is the main difference between Apple's and traditional leadership models?",
    height=100,
    max_chars=600  # Reasonable limit
)

if st.button("Run Comparison", key="custom"):
    if custom_question.strip():
        display_comparison_results(custom_question.strip())
    else:
        st.warning("Please enter a question before running the comparison.")