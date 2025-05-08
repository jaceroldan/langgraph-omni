from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage  # , merge_message_runs, trim_messages
# from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, END, MessagesState

from utils.bposeats import create_new_card
from utils.configuration import Configuration
from utils.schemas import CardState, Card
from utils.nodes import tool_handler, input_helper
from utils.models import models

# import settings


# TODO: will need to formalize the standard structure for subgraphs that
#       require looping user input


# Tools
@tool
def finish_process():
    """Fake tool to finish the card creation process."""
    return


def continue_to_tool(state: MessagesState) -> Literal["exit_tool_handler", "input_helper"]:
    last_message = state["messages"][-1]
    if not (hasattr(last_message, "tool_calls") and len(last_message.tool_calls)):
        # Assume that: if there is no tool called, the agent is asking for
        #  an input from the user
        return "input_helper"

    tool_name = last_message.tool_calls[0]["name"]
    match tool_name:
        # Only leave the subgraph once the agent manually finishes the process
        case "finish_process":
            return "exit_tool_handler"

    return "input_helper"


def card_extractor_helper(state: CardState, config: RunnableConfig) -> CardState:

    test_card = Card()
    test_card.creator = "15434"
    test_card.assignees = ["15434"]
    test_card.title = "Test Card by UserProfile#15434"
    test_card.is_public = True

    return {"card_details": test_card}


def card_agent(state: CardState, config: RunnableConfig) -> CardState:
    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    model = models[model_name]
    card_agent_model = model.bind_tools(node_tools, parallel_tool_calls=False)

    card_details = state["card_details"]
    FORMATTED_MESSAGE = AGENT_SYSTEM_MESSAGE.format(card_details=card_details)

    response = card_agent_model.invoke(
        [SystemMessage(content=FORMATTED_MESSAGE)] + state["messages"])

    return {"messages": [response]}


def card_creation_caller_node(state: CardState, config: RunnableConfig) -> CardState:
    configuration = Configuration.from_runnable_config(config)
    user_profile = configuration.user_profile_pk

    card = state["card_details"]
    title = card.title
    assignees = card.assignees
    is_public = card.is_public

    form_data = {
        "creator": user_profile,
        "assignees": assignees,
        "title": title,
        "is_public": is_public
    }

    try:
        api_response = create_new_card(form_data)
    except Exception as e:
        api_response = f"An Exception has occurred! {str(e)}"

    tool_call = state["messages"][-1].tool_calls[0]
    tool_message = {
        "content": f"The server has responded with: {api_response}",
        "role": "tool",
        "tool_call_id": tool_call["id"]
    }

    return tool_message


AGENT_SYSTEM_MESSAGE = (
    "# SYSTEM INSTRUCTIONS:\n"
    "You are an Assistant AI that is tasked on creating Board Cards for the user.\n"
    "You must follow the given instructions below step-by-step to successfully create "
    "a Board Card for the user:\n"
    "  1. Ask the user for the card's title.\n"
    "  2. Ask the user if they want to assign it to themselves or just leave it without "
    "assignees.\n"
    "  3. Ask them if they would like to make the card visible for everyone.\n"
    "  4. Once the 'title', 'assignee', and 'is_public' have been asked, make sure to "
    "reiterate everything and confirm with the user that this is correct.\n"
    "  5. When the user says that it's correct or confirms, call `finish_process` to "
    "end the creation process.\n\n"
    "Below is also the current state of Card, use it as a reference for the rules above:\n"
    "<details> {card_details} <details>"
)


agent_tools = []
node_tools = [finish_process]


subgraph_builder = StateGraph(CardState, config_schema=Configuration)

subgraph_builder.add_node(card_agent)
subgraph_builder.add_node(input_helper)
subgraph_builder.add_node(card_extractor_helper)
subgraph_builder.add_node("create_card_tool_handler", tool_handler)
subgraph_builder.add_node("initial_tool_handler", tool_handler)
subgraph_builder.add_node("exit_tool_handler", tool_handler)

subgraph_builder.add_edge(START, "initial_tool_handler")
subgraph_builder.add_edge("initial_tool_handler", "card_agent")
subgraph_builder.add_conditional_edges("card_agent", continue_to_tool)

subgraph_builder.add_edge("create_card_tool_handler", "input_helper")
subgraph_builder.add_edge("input_helper", "card_extractor_helper")
subgraph_builder.add_edge("card_extractor_helper", "card_agent")
subgraph_builder.add_edge("exit_tool_handler", END)


bposeats_card_creator_subgraph = subgraph_builder.compile()
