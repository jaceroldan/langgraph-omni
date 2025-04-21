
import pandas as pd
import numpy as np

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig

from scipy.spatial.distance import cosine
from typing_extensions import Annotated


embeddings = OpenAIEmbeddings()


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
