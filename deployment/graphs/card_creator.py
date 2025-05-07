from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from utils.bposeats import create_new_card
from utils.configuration import Configuration


def card_creation_node(state: MessagesState, config: RunnableConfig) -> str:
    configuration = Configuration.from_runnable_config(config)
    user_profile = configuration.user_profile_pk

    form_data = {
        "creator": user_profile,
        "assignees": user_profile,
        "title": "Some kind of title here",
    }

    try:
        create_new_card(form_data)
    except Exception as e:
        return f"An error has occurred! {str(e)}"

    return "A card has been created"
