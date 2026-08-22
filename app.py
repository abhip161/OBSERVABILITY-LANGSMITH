import uuid
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Client builder
def build_clients(groq_key: str, gemini_key: str, serper_key:str) -> dict:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import GoogleSerperAPIWrapper

    return {
        "groq": ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            api_key=groq_key,
        ),
        "gemini": ChatOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
            model="gemini-2.5-flash-lite",
            temperature=0.3,
        ),
        "serper": GoogleSerperAPIWrapper(serper_api_key=serper_key),
    }


# ── Graph (cached — wiring never changes; clients passed per-invoke via config)
@st.cache_resource
def get_graph():
    from agent.graph import build_graph
    return build_graph()


graph = get_graph()

# Page
st.title("Document Intelligence Agent")
st.caption("LangGraph · LangSmith · Groq · Gemini · Google Serper")

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Sidebar
with st.sidebar:
    st.header("API Keys")
    st.caption("Keys are loaded from .env if present or entered here.")

    st.subheader("LLM Keys")
    groq_key   = st.text_input("Groq API Key *",   type="password", value=os.getenv("GROQ_API_KEY", ""), placeholder="gsk_...")
    gemini_key = st.text_input("Gemini API Key *", type="password", value=os.getenv("GEMINI_API_KEY", ""), placeholder="AIza...")
    serper_key = st.text_input("Serper API Key *", type="password", value=os.getenv("SERPER_API_KEY", ""), placeholder="...")

    st.subheader("LangSmith Observability")
    ls_key     = st.text_input("LangSmith API Key *",    type="password", value=os.getenv("LANGSMITH_API_KEY", ""), placeholder="ls__...")
    ls_project = st.text_input("LangSmith Project Name *", value=os.getenv("LANGSMITH_PROJECT", "OBSERVABILITY-LANGSMITH"), placeholder="my-observability-demo")

    keys_ready = all([groq_key, gemini_key, serper_key, ls_key, ls_project])

    # Auto-initialize session if keys exist from .env and not initialized yet
    if keys_ready and "clients" not in st.session_state:
        try:
            st.session_state.clients = build_clients(groq_key, gemini_key, serper_key)
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = ls_key
            os.environ["LANGSMITH_PROJECT"] = ls_project
            st.session_state.ls_enabled = True
            st.session_state.ls_project = ls_project
        except Exception:
            pass


    if st.button("Save Keys", type="primary", disabled=not keys_ready):
        try:
            st.session_state.clients = build_clients(groq_key, gemini_key, serper_key)

            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"]  = ls_key
            os.environ["LANGSMITH_PROJECT"]  = ls_project
            st.session_state.ls_enabled  = True
            st.session_state.ls_project  = ls_project

            st.success("Keys saved — ready to chat!")
        except Exception as e:
            st.error(f"Failed to initialise clients: {e}")

    st.divider()
    st.subheader("Agent pipeline")
    st.markdown(
        "1. **Planner** — refines your question  \n"
        "2. **Document Reader** — searches local guide  \n"
        "3. **Web Enricher** — Google Serper live search  \n"
        "4. **Synthesizer** — combines sources (Groq)  \n"
        "5. **Report Writer** — formats report (Gemini)"
    )

    st.divider()
    st.subheader("Try these")
    st.markdown(
        "_What are the biggest LLM security risks?_  \n"
        "_How does RAG reduce hallucinations?_  \n"
        "_What is LLM-as-Judge evaluation?_  \n"
        "_Best practices for deploying LLM agents?_"
    )

    if st.session_state.get("ls_enabled"):
        st.divider()
        st.caption(f"Project: `{st.session_state.get('ls_project', '')}`")
        st.link_button("View LangSmith traces →", "https://smith.langchain.com")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Pipeline details"):
                st.json(msg["meta"])


# Input
clients_ready = "clients" in st.session_state

if not clients_ready:
    st.info("Enter your API keys in the sidebar and click **Save Keys** to start.")

if prompt := st.chat_input("Ask anything about LLM production...", disabled=not clients_ready):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        project = st.session_state.get("ls_project", "")
        spinner_msg = f"Running pipeline (tracing to LangSmith project: {project})..."
        with st.spinner(spinner_msg):
            result = graph.invoke(
                {
                    "question":         prompt,
                    "session_id":       st.session_state.session_id,
                    "refined_question": "",
                    "doc_sections":     [],
                    "web_results":      "",
                    "synthesis":        "",
                    "final_report":     "",
                    "steps_taken":      [],
                },
                config={"configurable": {"clients": st.session_state.clients}},
            )

        st.markdown(result["final_report"])

        meta = {
            "refined_question":   result["refined_question"],
            "pipeline":           " → ".join(result["steps_taken"]),
            "doc_sections_found": len(result["doc_sections"]),
            "session_id":         result["session_id"],
        }
        with st.expander("Pipeline details"):
            st.json(meta)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["final_report"],
        "meta":    meta,
    })