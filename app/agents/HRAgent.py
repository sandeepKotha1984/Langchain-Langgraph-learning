from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langgraph.constants import START, END, add_messages
from langgraph.graph import StateGraph

from llm.llm import get_llm
from tools.attendence import get_attendance
from app.tools.leave_balance import get_leave_balance
from pydantic import BaseModel
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def build_hr_agent():
    llm = get_llm()

    tools = [
        get_leave_balance,
        get_attendance
    ]

    tools_llm = llm.bind_tools(tools)




    def llm_node(state: AgentState):
        response1 = tools_llm.invoke(state["messages"])
        print(response1)
        return {
            "messages": [response1]
        }

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("tools",tool_node)
    graph.add_node("llm",llm_node)

    graph.add_edge(START,"llm")
    graph.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    graph.add_edge("tools","llm")

    app = graph.compile()
    return app











