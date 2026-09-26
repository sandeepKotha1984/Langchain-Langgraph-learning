from typing import TypedDict, Annotated
from urllib import response

from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from llm.llm import get_llm
from tools.employee import get_employee_details

def build_employee_agent():

    llm = get_llm()
    tools = [
        get_employee_details
    ]
    tools_llm = llm.bind_tools(tools)
    class AgentState(TypedDict):
        messages:Annotated[list,add_messages]

    def llm_node(state):
        user_messages = state["messages"][-1].content
        result = tools_llm.invoke(user_messages)
        print("llm node",result)

        return {
            "messages":[result]
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
