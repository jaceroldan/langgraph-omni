# Import general libraries
from typing import Literal, TypedDict

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from utils.api_caller import fetch_weekly_task_estimates
from utils.tools import estimate_tasks_duration

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.memory import load_memory

# Import subgraphs
from graphs.scalema_web3 import scalema_web3_subgraph


# Tools
class ToolCall(TypedDict):
    """
        Decision on which tool to use
    """
    tool_type: Literal["proposal", "weekly_tasks_summary"]


def choose_tool(state: MessagesState) -> Literal["scalema_web3_subgraph",
                                                 "fetch_weekly_task_estimates_summary",
                                                 END]:  # type: ignore
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
        case "weekly_tasks_summary":
            return "fetch_weekly_task_estimates_summary"
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


def agent(state: MessagesState, config: RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

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
    "\n\t1. If a user requests assistance with a proposal or provides details for one, "
    "always call ToolCall with the 'proposal' argument."
    "\n\t2. If the user asks for an estimate of the total hours required for their tasks this week, call ToolCall "
    "with the 'weekly_tasks_summary' argument. This applies whenever the user inquires about their workload, "
    "the time needed to complete their tasks, or any similar phrasing related to task estimates for the week."
    "\n\t3. Do not provide examples unless explicitly asked."
    "\n\nWhen using tools, do not inform the user that a tool has been called. Instead, "
    "respond naturally as if the action was performed seamlessly."
)


# Initialize Graph
builder = StateGraph(MessagesState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node("scalema_web3_subgraph", scalema_web3_subgraph)
builder.add_node("fetch_weekly_task_estimates_summary", fetch_weekly_task_estimates_summary)
builder.add_node("load_memory", load_memory)


builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")
builder.add_conditional_edges("agent", choose_tool)
builder.add_edge("scalema_web3_subgraph", "agent")
builder.add_edge("fetch_weekly_task_estimates_summary", "agent")

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
