from typing import List, Dict
from decimal import Decimal
import re

# Import Langgraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Import utils
from utils.models import models
from deployment.api import fetch_weekly_task_estimates
from utils.configuration import Configuration


def generate_completion(
        model, system_prompt: str, user_prompt: str) -> Decimal:
    """
        Performs the external API call to OpenAI and returns an integer
        value for estimated hours
    """
    try:
        response = model.invoke(
                [SystemMessage(content=system_prompt)] +
                [HumanMessage(content=user_prompt)]
            )

        response_content = response.content.strip()

        pattern = r"-?\d*\.\d+|\d+"
        matches = re.findall(pattern, response_content)

        if matches:
            last_match = matches[-1]

        return Decimal(last_match)

    except Exception as e:
        print(e)


def estimate_tasks_duration(
    model,
    task_names: List[str],
    similar_tasks: List[Dict],
    job_position: str,
    years_of_experience: int,
) -> Decimal:
    """
        Main util function for the API endpoint that generates AI estimation
        of the total hours required to complete multiple tasks.
    """

    system_template = (
        "You are an expert in estimating hours needed to complete any task. I"
        + " want you to estimate the total hours required to complete the"
        + " following tasks: {task_names} for a {job_position} with"
        + " {years_of_experience}  years of experience. Assume that more"
        + " years of experience means faster task completion."
    )

    system_prompt = system_template.format(
        task_names=", ".join(task_names),
        job_position=job_position,
        years_of_experience=str(years_of_experience),
    )

    user_prompt = (
        "Use the following similar tasks as a guide in estimating the"
        + " approximate hours needed to complete the tasks (Note that the"
        + " similar tasks are ordered from most similar to least):\n"
    )

    for task in similar_tasks:
        user_prompt += (
            " - "
            + task["name"]
            + " completed within "
            + str(task["duration"])
            + " hours\n"
        )

    user_prompt += (
        "I want you to strictly return a decimal with a maximum of only"
        + " 2 decimal places that represents the estimated total number"
        + " of hours needed to complete all the given tasks."
    )

    ai_estimated_hours = generate_completion(
        model, system_prompt, user_prompt)

    return ai_estimated_hours


@tool
def fetch_weekly_task_estimates_summary(config: RunnableConfig) -> str:
    """
        Provides a summary of the estimated hours required for
        the user's tasks for the week. Use this tool whenever the
        user asks about their weekly task estimates.
    """

    configuration = Configuration.from_runnable_config(config)
    auth_token = configuration.auth_token
    job_position = configuration.job_position
    user_profile_pk = configuration.user_profile_pk
    x_timezone = configuration.x_timezone
    workforce_id = configuration.workforce_id
    model_name = configuration.model_name
    node_model = models[model_name]
    is_web = configuration.is_web
    response = fetch_weekly_task_estimates(
        auth_token, workforce_id, user_profile_pk, x_timezone, is_web)

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

    # Temporary workaround: force the LLM to format its reply cleanly by
    # injecting a “system_message” instruction into the tool output.
    # Because LangGraph currently auto-prepends the raw tool return value
    return {
        "ai_estimation_hours": ai_estimation_hours,
        "system_message": (
            "If the user has tasks, start your reply with a blank space and the word 'Hours' "
            "right after. Proceed to construct your response."
            "Example: ' Hours. *Insert LLM Response*'"
            "If the user doesn't have any tasks, just send your response immediately."
        )
    }
