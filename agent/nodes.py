"""
Sequential Document Intelligence Agent — nodes.
All LangChain / LangGraph calls are auto-traced to LangSmith via env vars.
No explicit tracing code needed in these nodes.
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

from .state import AgentState
from .tools import search_document

load_dotenv()

def _get_groq(config: RunnableConfig = None) -> ChatGroq:
    if config:
        client = config.get("configurable", {}).get("clients", {}).get("groq")
        if client:
            return client
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def _get_gemini(config: RunnableConfig = None) -> ChatOpenAI:
    if config:
        client = config.get("configurable", {}).get("clients", {}).get("gemini")
        if client:
            return client
    return ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0.3,
    )

def _get_serper(config: RunnableConfig = None) -> GoogleSerperAPIWrapper:
    if config:
        client = config.get("configurable", {}).get("clients", {}).get("serper")
        if client:
            return client
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        return GoogleSerperAPIWrapper(serper_api_key=api_key)
    return GoogleSerperAPIWrapper()

def planner(state: AgentState, config: RunnableConfig = None) -> dict:
    """Rewrites the user question for clarity and precision."""
    groq = _get_groq(config)
    response = groq.invoke([
        SystemMessage(content=(
            "You are a research question refiner. "
            "Rewrite the user question to be more specific and searchable. "
            "Return ONLY the rewritten question, nothing else."
        )),
        HumanMessage(content=state["question"]),
    ])
    return {
        "refined_question": response.content.strip(),
        "steps_taken": state.get("steps_taken", []) + ["planner"],
    }

def document_reader(state: AgentState) -> dict:
    """Searches the local knowledge-base document for relevant sections."""
    sections = search_document(state["refined_question"], top_k=3)
    return {
        "doc_sections": sections,
        "steps_taken": state.get("steps_taken", []) + ["document_reader"],
    }

def web_enricher(state: AgentState, config: RunnableConfig = None) -> dict:
    """Fetches the latest information from the web using Google Serper."""
    try:
        serper = _get_serper(config)
        web_text = serper.run(state["refined_question"])
    except Exception as exc:
        web_text = f"[Web search unavailable: {exc}]"
    return {
        "web_results": web_text,
        "steps_taken": state.get("steps_taken", []) + ["web_enricher"],
    }

def synthesizer(state: AgentState, config: RunnableConfig = None) -> dict:
    """Combines document knowledge and web results into a coherent analysis."""
    groq = _get_groq(config)
    doc_context = "\n\n---\n\n".join(state["doc_sections"])
    synthesis = groq.invoke([
        SystemMessage(content=(
            "You are a research synthesizer. Given knowledge from a document and "
            "from the web, combine both into a clear, structured analysis. "
            "Cite sources where possible. Use markdown formatting."
        )),
        HumanMessage(content=(
            f"Question: {state['refined_question']}\n\n"
            f"=== DOCUMENT KNOWLEDGE ===\n{doc_context}\n\n"
            f"=== WEB SEARCH RESULTS ===\n{state['web_results']}"
        )),
    ])
    return {
        "synthesis": synthesis.content,
        "steps_taken": state.get("steps_taken", []) + ["synthesizer"],
    }

def report_writer(state: AgentState, config: RunnableConfig = None) -> dict:
    """Formats the synthesis into a polished final report using Gemini."""
    try:
        gemini = _get_gemini(config)
        report = gemini.invoke([
            SystemMessage(content=(
                "You are a technical report writer. Format the given analysis into "
                "a clean, well-structured report with: a one-sentence TL;DR at the top, "
                "key findings as bullet points, and a brief conclusion. "
                "Keep it under 400 words."
            )),
            HumanMessage(content=state["synthesis"]),
        ])
        final = report.content
    except Exception:
        final = state["synthesis"]

    return {
        "final_report": final,
        "steps_taken": state.get("steps_taken", []) + ["report_writer"],
    }


