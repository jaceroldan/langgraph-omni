from typing import List, Dict
from decimal import Decimal
import re
import pandas as pd
import numpy as np

# Import Langgraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import InjectedState
from scipy.spatial.distance import cosine

# Import utils
from utils.models import models
from utils.api_caller import fetch_weekly_task_estimates
from utils.configuration import Configuration

from typing_extensions import Annotated


embeddings = OpenAIEmbeddings()


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

    return response


def get_embedding(text: str) -> np.array:
    # OpenAIEmbeddings.embed_query returns a list of floats.
    vector = embeddings.embed_query(text)
    return np.array(vector)


# Utility function to compute cosine similarity between two embedding vectors.
def cosine_similarity(a: np.array, b: np.array) -> float:
    return 1 - cosine(a, b)


@tool
def fetch_page_url(
        config: RunnableConfig, state: Annotated[dict, InjectedState]):
    """
    Receives a user query regarding a page, retrieves a  file containing page
    questions and URLs, converts both the  page questions and the user query to
    embeddings, performs a similarity search, and returns the URL of the page
    in question. If no URL meets the similarity threshold, inform the user
    that the link cannot be found.

    Expected columns: "description", "path"
    """
    user_query = state["messages"][-2].content.strip()

    try:
        pages_df = pd.read_csv("./graphs/sitemap.csv")
    except Exception as e:
        response = f"Error retrieving the file: {str(e)}"
        return response

    # Ensure the required columns are present.
    if not {"path", "description"}.issubset(pages_df.columns):
        response = "The file does not have the required columns"
        return response

    query_embedding = get_embedding(user_query)
    pages_df["embedding"] = pages_df["description"].apply(get_embedding)
    pages_df["similarity"] = pages_df["embedding"].apply(
        lambda emb: cosine_similarity(emb, query_embedding))

    best_match_idx = pages_df["similarity"].idxmax()
    best_match = pages_df.loc[best_match_idx]

    SIMILARITY_THRESHOLD = 0.6
    print(pages_df["similarity"])
    if best_match["similarity"] < SIMILARITY_THRESHOLD:
        response = f"You are currently unable to find the link for '{
            user_query}'."
    else:
        response = f"The page URL is hqzen.com{best_match['path']}"

    return response
