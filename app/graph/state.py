from typing import Annotated
from langgraph.graph.message import add_messages


def AgentState(TypedDict):
    messages:Annotated[list,add_messages]