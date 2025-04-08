# Import general libraries
from typing import Literal, TypedDict

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from utils.api_caller import fetch_weekly_task_estimates
from utils.tools import estimate_tasks_duration

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.memory import load_memory, MemoryState, save_recall_memory, search_recall_memories

# Import subgraphs
from graphs.scalema_web3 import scalema_web3_subgraph


# Tools
class CreateProposal(TypedDict):
    """
        Creates a proposal. Redirects to the next step in the proposal process.
    """


class FetchWeeklyTaskEstimates(TypedDict):
    """
        Fetch weekly task estimates
    """


def choose_tool(state: MessagesState) -> Literal["scalema_web3_subgraph",
                                                 "fetch_weekly_task_estimates_summary",
                                                 "tool_executor",
                                                 END]:  # type: ignore
    """
        Decide on which tool to use
    """

    tool_calls = state["messages"][-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END

    match (tool_calls[0]["name"]):
        case "CreateProposal":
            return "scalema_web3_subgraph"
        case "FetchWeeklyTaskEstimates":
            return "fetch_weekly_task_estimates_summary"
        case "save_recall_memory" | "search_recall_memories":
            return "tool_executor"
        case _:
            return END

# Schemas


# Nodes
def fetch_weekly_task_estimates_summary(
        state: MessagesState, config: RunnableConfig):
    """
        Provides a summary of the estimated hours required for
        the user's tasks for the week. Use this tool whenever the
        user asks about their weekly task estimates. Call
        "fetch_weekly_task_estimates_summary" to retrieve
        this information.
    """

    configuration = Configuration.from_runnable_config(config)
    auth_token = configuration.auth_token
    job_position = configuration.job_position
    user_profile_pk = configuration.user_profile_pk
    x_timezone = configuration.x_timezone
    workforce_id = configuration.workforce_id

    model_name = configuration.model_name
    node_model = models[model_name]

    response = fetch_weekly_task_estimates(
        auth_token, workforce_id, user_profile_pk, x_timezone)

    if response:
        response = response['data']
        ai_estimation_hours = estimate_tasks_duration(
            node_model,
            response['target_task_names'],
            response['similar_task_names'],
            job_position,
            response['years_of_experience'],
        )
    else:
        ai_estimation_hours = 0

    response = """
        Below is the estimated number of hours required to complete the tasks
        the system has generated for the user:
        {ai_estimation_hours}

        Discuss with them how many hours are needed for the current week's
        tasks. If there are no tasks remaining, congratulate them on
        completing their work for the week and encourage them to relax.
    """.format(ai_estimation_hours=ai_estimation_hours)

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{
        "role": "tool",
        "content": response,
        "tool_call_id": tool_calls[0]['id']}]}


def agent(state: MemoryState, config: RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    model_history_length = configuration.model_history_length

    tools = [CreateProposal, FetchWeeklyTaskEstimates] + memory_tools

    node_model = models[model_name].bind_tools(tools, parallel_tool_calls=False)

    messages = [SystemMessage(content=MODEL_SYSTEM_MESSAGE)] + state["messages"][-model_history_length:]
    response = node_model.invoke(messages)

    return {"messages": [response]}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = (
    "# INSTRUCTIONS:\n"
    "You are Scalema, a helpful chatbot that assists clients with their business queries. If this is your first time "
    "interacting with a client, introduce yourself and inform them of your role. ""You have the ability to choose the "
    "appropriate tools to handle client requests.\n"
    "## Guidelines for tool usage:\n"
    "\t1. If a user requests assistance with a proposal or provides details for a proposal, always call the "
    "`CreateProposal` tool.\n"
    "\t2. If the user asks for an estimate of the total hours required for their tasks this week, call the "
    "`FetchWeeklyTaskEstimates` tool. This applies whenever the user inquires about their workload, the time ""needed "
    "to complete their tasks, or any similar phrasing related to task estimates for the week.\n"
    "\t3. Else, use `save_recall_memory` to save any relevant information that the user shares with you. This will "
    "help you remember important details for future conversations. This includes the following:\n"
    "\t\t- User's name\n"
    "\t\t- User's job position\n"
    "\t\t- The tools the user has used\n"
    "When using tools, do not inform the user that a tool has been called. Instead, respond naturally as if the action "
    "was performed seamlessly."
)


# Initialize Graph
memory_tools = [save_recall_memory, search_recall_memories]

builder = StateGraph(MemoryState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node("scalema_web3_subgraph", scalema_web3_subgraph)
builder.add_node("fetch_weekly_task_estimates_summary", fetch_weekly_task_estimates_summary)
builder.add_node("load_memory", load_memory)
builder.add_node("tool_executor", ToolNode(memory_tools))


builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")
builder.add_conditional_edges("agent", choose_tool)
builder.add_edge("scalema_web3_subgraph", "agent")
builder.add_edge("fetch_weekly_task_estimates_summary", "agent")
builder.add_edge("tool_executor", "agent")

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
