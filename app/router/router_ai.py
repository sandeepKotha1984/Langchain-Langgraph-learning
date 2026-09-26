from typing import Literal, TypedDict, Annotated

from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel

from app.rag.rag_query import rag_query
from app.agents.HRAgent import build_hr_agent
from app.tools.attendence import get_attendance
from app.tools.leave_balance import get_leave_balance


llm = ChatOllama(
    model="qwen3:4b-instruct",
    temperature=0,
    reasoning=False,
)


tools = [
    get_leave_balance,
    get_attendance
]


class RouteDecision(BaseModel):
    route: Literal["rag", "agent"]


llm_with_router = llm.with_structured_output(RouteDecision)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str


def router_node(state: AgentState):

    user_message = state["messages"][-1].content

    decision = llm_with_router.invoke(
        f"""
        Classify the following HR request.

        Use "rag" for:
        - HR policies
        - company rules
        - benefits
        - general HR knowledge

        Use "agent" for:
        - employee-specific information
        - leave balance
        - attendance
        - database queries
        - actions

        User request:
        {user_message}
        """
    )

    print("Selected route:", decision.route)

    return {
        "route": decision.route
    }


def rag_node(state: AgentState):

    user_message = state["messages"][-1].content

    rag_response = rag_query(user_message)


    return {
        "messages": [rag_response]
    }


def agent_node(state: AgentState):

    user_message = state["messages"][-1].content

    agent = build_hr_agent()

    answer = agent.invoke(user_message)

    return {
        "messages": [answer]
    }


graph = StateGraph(AgentState)


# Nodes
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("agent", agent_node)


# START → Router
graph.add_edge(START, "router")


# Router → RAG / Agent
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "rag": "rag",
        "agent": "agent",
    }
)


# RAG → END
graph.add_edge("rag", END)

# Agent → END
graph.add_edge("agent", END)


app = graph.compile()