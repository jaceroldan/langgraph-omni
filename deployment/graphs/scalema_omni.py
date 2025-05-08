# Import general libraries
from typing import Literal
from datetime import datetime

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode
from langgraph.pregel import RetryPolicy

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.memory import (
    MemoryState,
    save_recall_memory,
    search_recall_memories,
    memory_summarizer)
from settings import POSTGRES_URI
from utils.estimates import fetch_weekly_task_estimates_summary

# Import subgraphs
from graphs.scalema_web3 import scalema_web3_subgraph
from graphs.card_creator import bposeats_card_creator_subgraph
from graphs.initialization import init_graph


# Tools
@tool
def web3_create_proposal():
    """Creates a proposal. Routes to `scalema_web3_subgraph` for proposal creation."""
    return


@tool
def bposeats_create_card():
    """Creates a card. Routes to `bposeats_card_creator_subgraph` for card creation."""
    return


def continue_to_tool(state: MessagesState) -> Literal[
                                                "scalema_web3_subgraph",
                                                "bposeats_card_creator_subgraph",
                                                "tool_executor",
                                                "memory_executor",
                                                "__end__"]:
    """
        Decide on which tool to use
        @TODO: Transfer to a handoff_tool
        https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/#implement-a-handoff-tool
    """

    last_message = state["messages"][-1]
    # If there is no function call, then finish
    if not (hasattr(last_message, "tool_calls") and len(last_message.tool_calls)):
        return END

    tool_name = last_message.tool_calls[0]["name"]
    match tool_name:
        case "web3_create_proposal":
            return "scalema_web3_subgraph"
        case "bposeats_create_card":
            return "bposeats_card_creator_subgraph"
        case _ if tool_name in [tool.get_name() for tool in memory_tools]:
            return "memory_executor"
        case _ if tool_name in [tool.get_name() for tool in agent_tools]:
            return "tool_executor"
        case _:
            return END

    return END
# Schemas


# Nodes
def agent(state: MemoryState, config: RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    memories = state.get("memories")

    tools = memory_tools + agent_tools + node_tools
    node_model = models[model_name].bind_tools(tools=tools, parallel_tool_calls=False)

    sys_msg = [
        SystemMessage(content=MODEL_SYSTEM_MESSAGE.format(
            memories=memories,
            timestamp=datetime.now()
        ))
    ]

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
    "You don't need to introduce yourself if you have memories or already have a conversation with the user.\n\n"

    "## MEMORIES\n"
    "Below are your past long-term memories of the user, this can also be empty.\n"
    "<memories> {memories} </memories>\n"
    "Base your responses on the memories above and the conversation history.\n"
    "If you don't have any memories, respond naturally and ask the user for more information.\n"
    "If you have memories, use them to provide a more personalized response.\n\n"

    "## TOOL USAGE GUIDELINES\n"
    "You have access to a set of tools to help you handle client requests efficiently.\n"
    "1. **Creating Proposal Assistance**:\n"
    "   - If the user asks for help with or to create a proposal or provides proposal details, use the "
    "`web3_create_proposal` tool.\n\n"
    "2. **Creating Cards**:\n"
    "   - If the user instead wants to create a card, simply call the `bposeats_create_card` tool. No "
    "need to ask for more details.\n\n"
    "3. **Fetching Weekly Task Estimates**:\n"
    "   - If the user asks for their weekly task estimates summary or anything of the sort, use the "
    "`fetch_weekly_task_estimates_summary` tool.\n"
    "   - Call the tool even if past memories indicate that the user has no remaining tasks for the week"
    " as there might be updates to the user's tasks.\n\n"
    "4. **Memory Recall**:\n"
    "   - If the user refers to a past conversation or memory:\n"
    "     - Use `search_recall_memories` to retrieve it.\n"
    "     - If nothing is found, respond honestly that you don't know.\n\n"
    "     - Don't mention data records, respond as if you were recalling the memory personally.\n\n"
    "5. **Saving User Information**:\n"
    "   - Use `save_recall_memory` to store any important user information for future interactions, including:\n"
    "     - User's name\n"
    "     - User's job position\n\n"
    "6. **Tool Interaction Etiquette**:\n"
    "   - You are only allowed to use one tool. Do not call more than one tool in one response.\n"
    "   - Do not mention tool usage explicitly to the user.\n"
    "   - Always try to confirm information being asked of you using tools over relying on memories.\n"
    "   - Respond naturally, as if the action was completed directly by you.\n\n"

    "**Current Time**: {timestamp}"
)

# Initialize Graph
memory_tools = [save_recall_memory, search_recall_memories]
agent_tools = [fetch_weekly_task_estimates_summary]
node_tools = [web3_create_proposal, bposeats_create_card]

builder = StateGraph(MemoryState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node(memory_summarizer, retry=RetryPolicy(max_attempts=3))
builder.add_node("initialization", init_graph)
builder.add_node("scalema_web3_subgraph", scalema_web3_subgraph)
builder.add_node("bposeats_card_creator_subgraph", bposeats_card_creator_subgraph)
builder.add_node("tool_executor", ToolNode(agent_tools))
builder.add_node("memory_executor", ToolNode(memory_tools))

builder.add_edge(START, "initialization")
builder.add_edge("initialization", "agent")
builder.add_conditional_edges("agent", continue_to_tool)
builder.add_edge("memory_summarizer", "agent")
builder.add_edge("scalema_web3_subgraph", "memory_summarizer")
builder.add_edge("bposeats_card_creator_subgraph", "memory_summarizer")
builder.add_edge("tool_executor", "memory_summarizer")
builder.add_edge("memory_executor", "agent")  # We don't want to process memories here

with PostgresStore.from_conn_string(POSTGRES_URI) as store, \
     PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
    store.setup()
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer, store=store)
