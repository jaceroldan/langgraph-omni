# Import general libraries
from typing import Literal, TypedDict

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

# Import utility functions
from utils.configuration import Configuration
from utils.models import models

# Import subgraphs
from graphs.scalema_web3 import scalema_web3_subgraph


# Tools
class ToolCall(TypedDict):
    """
        Decision on which tool to use
    """
    tool_type: Literal["proposal"]


def choose_tool(state: MessagesState) -> Literal["scalema_web3_subgraph", END]:  # type: ignore
    """
        Decide on which tool to use
    """

    tool_calls = state["messages"][-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END

    match (tool_calls[0]["args"]["tool_type"]):
        case "proposal":
            return "scalema_web3_subgraph"
        case _:
            return END

# Schemas


# Nodes
def agent(state: MessagesState, config=RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

    # access model name through config passed in the Backend
    model_name = Configuration.from_runnable_config(config).model_name
    node_model = models[model_name].bind_tools([ToolCall], parallel_tool_calls=False)
    response = node_model.invoke([SystemMessage(content=MODEL_SYSTEM_MESSAGE)] + state["messages"])
    return {"messages": [response]}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = (
    "You are Scalema, a helpful chatbot that assists clients with their business queries. "
    "If this is your first time interacting with a client, introduce yourself and inform them of your role. "
    "You have the ability to choose the appropriate tools to handle client requests."
    "\n\nGuidelines for tool usage:"
    "\n\t- If a user requests assistance with a proposal or provides details for one, "
    "always call ToolCall with the 'proposal' argument."
    "\n\t- Do not provide examples unless explicitly asked."
    "\n\nWhen using tools, do not inform the user that a tool has been called. Instead, "
    "respond naturally as if the action was performed seamlessly."
)


# Initialize Graph
builder = StateGraph(MessagesState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node("scalema_web3_subgraph", scalema_web3_subgraph)


builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", choose_tool)
builder.add_edge("scalema_web3_subgraph", "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
