from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from utils.environ import replace_postgres_hostname
from utils.configuration import Configuration
import settings


POSTGRES_URI = replace_postgres_hostname(settings.POSTGRES_URI, "localhost")
HQZEN_URL = settings.SITE_DOMAINS["hqzen.com"]

RETURN_MESSAGE = ("The following links are available to the user:\n"
                  "{links}\n\n"
                  "Provide the user with the appropriate link.")


@tool
def get_navigation_links(*args, config: RunnableConfig) -> str:
    """
        Fetches links for HQZEN.com

        Returns:
            A formatted context-aware string containing links for the user to navigate to.
    """

    configurable = Configuration.from_runnable_config(config)
    company_id = configurable.company_id
    workforce_id = configurable.workforce_id
    employment_id = configurable.employment_id
    payroll_id = configurable.payroll_id

    prepend_URL = f"{HQZEN_URL}/company/{company_id}/workforce/{workforce_id}"

    nav_links = [
        {"url": f"{HQZEN_URL}/profile/user-profile",
         "key": "user profile"},
        {"url": f"{HQZEN_URL}/help-center",
         "key": "help center"},
        {"url": (f"{prepend_URL}/career/{employment_id}/timelog/{payroll_id}/logs?tab=timelogs"),
         "key": "time logs"},
        {"url": f"{prepend_URL}/board/-1",
         "key": "boards"},
        {"url": f"{prepend_URL}/career/{employment_id}/leaves",
         "key": "leave requests"},
        {"url": f"{prepend_URL}/career/{employment_id}/about",
         "key": "career employment"},
        {"url": f"{prepend_URL}/milestones",
         "key": "milestones"},
        {"url": f"{prepend_URL}/forms",
         "key": "forms"},
        {"url": f"{prepend_URL}/overview/{employment_id}",
         "key": "company overview"}
    ]
    stringified_links = f"{"\n".join([f"  - {link["key"]}: {link["url"]}" for link in nav_links])}"

    return RETURN_MESSAGE.format(links=stringified_links)
