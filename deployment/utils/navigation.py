from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.postgres import PostgresStore

from utils.environ import replace_postgres_hostname
import settings


POSTGRES_URI = replace_postgres_hostname(settings.POSTGRES_URI, "localhost")
HQZEN_URL = settings.SITE_DOMAINS["hqzen.com"]

RETURN_MESSAGE = ("The following links are available to the user:\n"
                  "{links}\n\n"
                  "Provide the user with the appropriate link.")


@tool
def get_navigation_links(state: MessagesState, config: RunnableConfig, store: PostgresStore):
    nav_links = [
        {"url": f"{HQZEN_URL}/profile/user-profile",
         "key": "user profile"},
    ]
    stringified_links = f"{"\n".join([f"  - {link["key"]}: {link["url"]}" for link in nav_links])}"

    return RETURN_MESSAGE.format(links=stringified_links)
