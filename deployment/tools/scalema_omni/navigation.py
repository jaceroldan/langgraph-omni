
import pandas as pd
import numpy as np

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
import xml.etree.ElementTree as ET

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
        config: RunnableConfig,
        state: Annotated[dict, InjectedState]
        ):
    user_query = state["messages"][-2].content.strip()

    try:
        tree = ET.parse("./graphs/sitemap.xml")
        root = tree.getroot()

        records = []
        for route in root.findall("route"):
            path_el = route.find("path")
            desc_el = route.find("description")
            if path_el is not None and desc_el is not None:
                records.append({
                    "path":    path_el.text.strip(),
                    "description": desc_el.text.strip()
                })

        pages_df = pd.DataFrame(records)
    except Exception as e:
        return f"Error loading or parsing sitemap.xml: {e}"

    if pages_df.empty:
        return "The sitemap.xml did not contain any entries."

    pages_df = pages_df.drop_duplicates(subset="path", keep="first")

    query_embedding = get_embedding(user_query)
    pages_df["embedding"] = pages_df["description"].apply(get_embedding)
    pages_df["similarity"] = pages_df["embedding"].apply(
        lambda emb: cosine_similarity(emb, query_embedding)
    )

    best_idx = pages_df["similarity"].idxmax()
    best = pages_df.loc[best_idx]

    SIMILARITY_THRESHOLD = 0.6
    if best["similarity"] < SIMILARITY_THRESHOLD:
        return f"Sorry, I couldn't find a page matching “{user_query}.”"
    else:
        return f"The page URL is https://hqzen.com{best['path']}"