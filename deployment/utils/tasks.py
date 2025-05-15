# Import Langgraph
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


# Import utils
from utils.bposeats import fetch_tasks_due
from utils.configuration import Configuration
from utils.models import models


@tool
def fetch_most_urgent_task(config: RunnableConfig) -> str:
    """Fetches the most urgent task for the current user"""

    configuration = Configuration.from_runnable_config(config)
    user_profile_pk = configuration.user_profile_pk
    workforce_id = configuration.workforce_id
    node_model = models["tool-calling-model"]

    form_data = {
        "workforce_id": workforce_id,
        "user_profile_pk": user_profile_pk,
        "due_date_flag": "Today"
    }

    api_response = fetch_tasks_due(form_data)
    tasks = [{
        "task": res["task"]["title"],
        "current_duration_worked": res["total_duration"],
        "is_scheduled_task": res["task"]["is_scheduled_task"],
        "is_meeting": res["task"]["is_meeting"]
    } for res in api_response["data"]]

    FORMATTED_TOOL_MESSAGE = (
        "You are an assistant that helps a user determine which tasks to do first. "
        "The following are tasks assigned to the user which can also be empty:\n"
        "{tasks}\n"
        "These tasks are due today so you must determine which tasks are the most "
        "urgent given the information."
    ).format(tasks=tasks)
    response = node_model.invoke(FORMATTED_TOOL_MESSAGE)

    return response


@tool
def fetch_tasks_to_complete_this_week(config: RunnableConfig) -> str:
    """Fetches tasks that are due this week"""

    configuration = Configuration.from_runnable_config(config)
    user_profile_pk = configuration.user_profile_pk
    workforce_id = configuration.workforce_id
    node_model = models["tool-calling-model"]

    form_data = {
        "workforce_id": workforce_id,
        "user_profile_pk": user_profile_pk,
        "due_date_flag": "Week"
    }

    api_response = fetch_tasks_due(form_data)
    tasks = [res["task"]["title"] for res in api_response["data"]]

    FORMATTED_TOOL_MESSAGE = (
        "You are an assistant that helps inform the user of tasks that are due this week. "
        "The following are tasks assigned to the user which can also be empty:\n"
        "{tasks}\n"
        "If there are more than 20 tasks, simply pick out which ones sound the most important "
        "and then omit the rest. Also, simply inform the user of the tasks and then add a small"
        " comment at the end."
    ).format(tasks=tasks)
    response = node_model.invoke(FORMATTED_TOOL_MESSAGE)

    return response
