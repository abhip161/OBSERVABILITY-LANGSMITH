from typing import TypedDict, List

class AgentState(TypedDict):
    question:str
    refined_question:str
    doc_sections:List[str]
    web_results:str
    synthesis:str
    final_report:str
    session_id:str
    steps_taken:List[str]