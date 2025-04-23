# Import general libraries
from typing import Literal

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.memory import (
    MemoryState,
    save_recall_memory,
    search_recall_memories,
    load_memory,
    memory_node)
from settings import POSTGRES_URI
from utils.estimates import fetch_weekly_task_estimates_summary

# Import subgraphs
from graphs.scalema_web3 import scalema_web3_subgraph


# Tools
@tool
def web3_create_proposal():
    """
        Creates a proposal. Routes to `scalema_web3_subgraph` for proposal creation.
    """
    return


def continue_to_tool(state: MessagesState) -> Literal[
                                                "scalema_web3_subgraph",
                                                "tool_executor",
                                                END]:  # type: ignore
    """
        Decide on which tool to use
    """

    tool_calls = state["messages"][-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END

    tool_name = tool_calls[0]["name"]
    route_to_nodes = []
    match tool_name:
        case "web3_create_proposal":
            route_to_nodes.append("scalema_web3_subgraph")
        case _ if tool_name in [tool.get_name() for tool in agent_tools]:
            route_to_nodes.append("tool_executor")
        case _:
            return END

    return route_to_nodes
# Schemas


# Nodes
def agent(state: MemoryState, config: RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    memories = state.get("memories")

    node_model = models[model_name].bind_tools(agent_tools + node_tools)

    sys_msg = [SystemMessage(content=MODEL_SYSTEM_MESSAGE)]
    if memories:
        sys_msg.append(SystemMessage(content=MEMORY_MESSAGE.format(memories=memories)))

    messages = merge_message_runs(
        messages=sys_msg + state["messages"]
    )

    response = node_model.invoke(messages)

    return {"messages": [response]}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = (
    "# SYSTEM INSTRUCTIONS\n"
    "You are **Scalema**, a helpful and professional chatbot that assists clients with their business-related "
    "queries.\n\n"
    "If this is your first interaction with a client, introduce yourself and clearly explain your role.\n"
    "You have access to a set of tools to help you handle client requests efficiently.\n\n"
    "## TOOL USAGE GUIDELINES\n"
    "1. **Proposal Assistance**:\n"
    "   - If the user asks for help with a proposal or provides proposal details, use the `web3_create_proposal` "
    "tool.\n\n"
    "2. **Weekly Task Estimates**:\n"
    "   - If the user asks how long their tasks will take this week, or anything about workload/time estimates,\n"
    "     use the `fetch_weekly_task_estimates_summary` tool.\n\n"
    "3. **Memory Recall**:\n"
    "   - If the user refers to a past conversation or memory:\n"
    "     - Use `search_recall_memories` to retrieve it.\n"
    "     - If nothing is found, respond honestly that you don't know.\n\n"
    "4. **Saving User Information**:\n"
    "   - Use `save_recall_memory` to store any important user information for future interactions, including:\n"
    "     - User's name\n"
    "     - User's job position\n\n"
    "5. **Tool Interaction Etiquette**:\n"
    "   - Do not mention tool usage explicitly to the user.\n"
    "   - Respond naturally, as if the action was completed directly by you."
)

MEMORY_MESSAGE = (
    "# MEMORIES\n"
    "Below are your memories of the user, this can also be empty. Carefully cater your responses accordingly.\n"
    "<memories> {memories} </memories>\n"
)


# Initialize Graph
agent_tools = [save_recall_memory, search_recall_memories, fetch_weekly_task_estimates_summary]
node_tools = [web3_create_proposal]

builder = StateGraph(MemoryState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node(load_memory)
builder.add_node(memory_node)
builder.add_node("scalema_web3_subgraph", scalema_web3_subgraph)
builder.add_node("tool_executor", ToolNode(agent_tools))

builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")
builder.add_conditional_edges("agent", continue_to_tool)
builder.add_edge("memory_node", "agent")
builder.add_edge("scalema_web3_subgraph", "memory_node")
builder.add_edge("tool_executor", "memory_node")

with PostgresStore.from_conn_string(POSTGRES_URI) as store, \
     PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
    store.setup()
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer, store=store)
