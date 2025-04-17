# Import general libraries
from typing import Literal, TypedDict

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
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
class CreateProposal(TypedDict):
    """
        Creates a proposal. Routes to `scalema_web3_subgraph` for proposal creation.
    """


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
        case "CreateProposal":
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

    node_model = models[model_name].bind_tools(agent_tools + node_tools)
    sys_msg = MODEL_SYSTEM_MESSAGE

    memories = state.get("memories", None)
    if memories:
        sys_msg = MODEL_SYSTEM_MESSAGE + MEMORY_MESSAGE.format(memories=memories)

    messages = [SystemMessage(content=sys_msg)] + state["messages"]
    response = node_model.invoke(messages)

    return {"messages": [response]}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = (
    "# INSTRUCTIONS:\n"
    "You are Scalema, a helpful chatbot that assists clients with their business queries. If this is your first time "
    "interacting with a client, introduce yourself and inform them of your role. You have the ability to choose the "
    "appropriate tools to handle client requests.\n"
    "## Guidelines for tool usage:\n"
    "\t1. If a user requests assistance with a proposal or provides details for a proposal, always call the "
    "`CreateProposal` tool.\n"
    "\t2. If the user asks for an estimate of the total hours required for their tasks this week, call the "
    "`fetch_weekly_task_estimates_summary` tool. This applies whenever the user inquires "
    "about their workload, the time needed to complete their tasks, or any similar phrasing related to task estimates "
    "for the week.\n"
    "\t3. Determine if the user is referring to some memory, use `search_recall_memories` to retrieve those memories"
    "if you do not have them. If there is no such memory, simply respond that you do not know.\n"
    "\t4. Always use `save_recall_memory` to save any relevant information that the user shares with you. This will "
    "help you remember important details for future conversations. This includes the following:\n"
    "\t\t- User's name\n"
    "\t\t- User's job position\n"
    "\t5. When using tools, do not inform the user that a tool has been called. Instead, respond naturally as if the "
    "action was performed seamlessly.\n"
)

MEMORY_MESSAGE = (
    "# MEMORIES\n"
    "Below are your memories of the user, this can also be empty. Carefully cater your responses accordingly.\n"
    "<memories> {memories} </memories>\n"
)


# Initialize Graph
agent_tools = [save_recall_memory, search_recall_memories, fetch_weekly_task_estimates_summary]
node_tools = [CreateProposal]

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
