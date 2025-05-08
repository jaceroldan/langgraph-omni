from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from utils.bposeats import create_new_card
from utils.configuration import Configuration
from utils.schemas import CardState
from utils.nodes import fake_node, tool_handler


def card_creation_node(state: CardState, config: RunnableConfig) -> str:
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


builder = StateGraph(CardState, config_schema=Configuration)

builder.add_node(fake_node)
builder.add_node("initial_tool_handler", tool_handler)
builder.add_edge(START, "initial_tool_handler")
builder.add_edge("initial_tool_handler", "fake_node")
builder.add_edge("fake_node", END)

bposeats_card_creator_subgraph = builder.compile()
