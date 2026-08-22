from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from .state import AgentState
from .nodes import planner, document_reader, web_enricher, synthesizer, report_writer


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("document_reader", document_reader)
    builder.add_node("web_enricher", web_enricher)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("report_writer", report_writer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "document_reader")
    builder.add_edge("document_reader", "web_enricher")
    builder.add_edge("web_enricher", "synthesizer")
    builder.add_edge("synthesizer", "report_writer")
    builder.add_edge("report_writer", END)

    return builder.compile()
    

graph = build_graph()
