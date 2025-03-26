
import re
from decimal import Decimal
from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0)


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
        system_prompt, user_prompt)

    return ai_estimated_hours


def generate_completion(
        system_prompt: str, user_prompt: str) -> Decimal:
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
