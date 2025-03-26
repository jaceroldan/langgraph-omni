# Import general libraries
from pydantic import BaseModel, Field

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from utils.api_caller import fetch_weekly_task_estimates
from utils import estimate_tasks_duration

# Import utility functions
from utils.configuration import Configuration
from utils.models import models


class HumanQuery(BaseModel):  # Used to structure data
    """
        This contains the arguments of the User's query that you are chatting with.
        Always call this tool whenever the User asks anything.
    """

    query: str = Field(description="This contains the user's query")


def should_continue(state: MessagesState):
    """
        Decide whether to continue or not
    """

    messages = state["messages"]

    tool_calls = messages[-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END
    # if tool_calls[0]["name"] == "HumanQuery":
    #     return "input_node"
    if tool_calls[0]["name"] == "fetch_weekly_task_estimates_summary":
        return "fetch_weekly_task_estimates_summary"
    else:
        return END


def input_node(state: MessagesState):
    """
        Facilitates HITL and assists in collecting user input
    """

    tool_call = state["messages"][-1].tool_calls[0]
    prompt = tool_call["args"]["query"]
    formatted_question = (
        "You asked: {question}, provide an answer for this."
    )

    user_input = interrupt(formatted_question.format(question=prompt))
    content_message = (
        "The user responded with: "
        "<input>{input}</input>"
        "\nReview the user's input in relation to the question. Correct it if "
        "is wrong and provide an explanation."
    )  # This is temporary until a use case is created

    tool_message = [{
        "tool_call_id": tool_call["id"],
        "type": "tool",
        "content": content_message.format(input=user_input)
    }]
    return {"messages": tool_message}


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
    employment_id = configuration.employment_id
    job_position = configuration.job_position
    user_profile_pk = configuration.user_profile_pk
    x_timezone = configuration.x_timezone

    model_name = configuration.model_name
    node_model = models[model_name]

    response = fetch_weekly_task_estimates(
        auth_token, employment_id, user_profile_pk, x_timezone)

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


def agent(state: MessagesState, config=RunnableConfig):
    """
        Helps personalizes chatbot messages
    """

    model_name = Configuration.from_runnable_config(config).model_name
    tools = [fetch_weekly_task_estimates_summary]
    node_model = models[model_name].bind_tools(tools)
    # access model name through config passed in the Backend
    response = node_model.invoke([SystemMessage(content=MODEL_SYSTEM_MESSAGE)] + state["messages"])
    return {"messages": [response]}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = """
        You are Scalema, a helpful chatbot that helps clients with their
        business queries. If it's your first time talking with a client,
        be sure to inform them this. Here are your instructions for
        reasoning about the user's messages:

        Reason carefully about the user's messages as presented below.

        1. If the user asks for an estimate of the total hours required for
        their tasks this week, call the fetch_weekly_task_estimates_summary
        tool. This applies whenever the user inquires about their workload,
        the time needed to complete their tasks, or any similar phrasing
        related to task estimates for the week.
    """

# Build the graph
builder = StateGraph(MessagesState, config_schema=Configuration)

builder.add_node(agent)
builder.add_node(fetch_weekly_task_estimates_summary)

# builder.add_node(input_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
# builder.add_edge("input_node", "agent")
builder.add_edge("fetch_weekly_task_estimates_summary", "agent")

# Compile the graph
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
