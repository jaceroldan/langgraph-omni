from typing import List, Dict
from decimal import Decimal
import re

# Import Langgraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


# Import utils
from utils.models import models
from utils.bposeats import fetch_weekly_task_estimates
from utils.configuration import Configuration


def generate_completion(model, system_prompt: str, user_prompt: str) -> Decimal:
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
    job_position = configuration.job_position
    user_profile_pk = configuration.user_profile_pk
    workforce_id = configuration.workforce_id
    node_model = models["tool-calling-model"]

    form_data = {
        "workforce_id": workforce_id,
        "user_profile_pk": user_profile_pk,
    }

    response = fetch_weekly_task_estimates(form_data)

    if response:
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

    return response
