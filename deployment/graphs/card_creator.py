from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, merge_message_runs
from langgraph.graph import StateGraph, START, END, MessagesState
from trustcall import create_extractor

from api.bposeats import create_new_card
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


@tool
def cancel_process():
    """Fake tool to cancel the card creation process."""
    return


def continue_to_tool(state: MessagesState) -> Literal["create_card_tool_handler",
                                                      "input_helper",
                                                      "cancel_tool_handler"]:
    """Determine the next node to go to."""

    last_message = state["messages"][-1]
    if not (hasattr(last_message, "tool_calls") and len(last_message.tool_calls)):
        # Assume that: if there is no tool called, the agent is asking for
        #  an input from the user
        return "input_helper"

    tool_name = last_message.tool_calls[0]["name"]
    match tool_name:
        # Only leave the subgraph once the agent manually finishes the process
        case "cancel_process":
            return "cancel_tool_handler"
        case "finish_process":
            return "create_card_tool_handler"

    return "input_helper"


def card_extractor_helper(state: CardState, config: RunnableConfig) -> CardState:
    """Handles in extracting Card information from user responses"""

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    user_profile_pk = configurable.user_profile_pk
    tool_name = "Card"

    card_details = state.get("card_details", None)

    detail_extractor = create_extractor(
        models[model_name],
        tools=[Card],
        tool_choice=tool_name,
        enable_inserts=True,
        enable_deletes=True
    )

    FORMATTED_EXTRACTOR_MESSAGE = EXTRACTOR_MESSAGE.format(user_profile_pk=user_profile_pk)

    merged_messages = list(merge_message_runs(
            messages=[SystemMessage(content=FORMATTED_EXTRACTOR_MESSAGE)] + state["messages"]
    ))

    # TODO: Implement extracting specific Board/Column of user
    #       currently only creates cards on the Development Board
    #       and on the To Do column specifically. Can also extend
    #       this to card creation on different Workforces

    result = detail_extractor.invoke({"messages": merged_messages, "existing": {tool_name: card_details}})
    extracted_card_details = result["responses"][0]

    return {"card_details": extracted_card_details}


def card_agent(state: CardState, config: RunnableConfig) -> CardState:
    """
        Facilitates the Board Card creation process by responding to the user and
        calling the correct tools.
    """

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    model = models[model_name]
    card_agent_model = model.bind_tools(node_tools, parallel_tool_calls=False)

    card_details = state.get("card_details", None)
    FORMATTED_MESSAGE = AGENT_SYSTEM_MESSAGE.format(card_details=card_details)

    response = card_agent_model.invoke(
        [SystemMessage(content=FORMATTED_MESSAGE)] + state["messages"])

    return {"messages": [response]}


def card_creation_caller_node(state: CardState, config: RunnableConfig) -> CardState:
    """Attempts to create a Card by calling an API endpoint"""

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

    FORMATTED_API_RESPONSE = API_RESPONSE_MESSAGE.format(api_response=api_response)

    return {"messages": SystemMessage(content=FORMATTED_API_RESPONSE)}


EXTRACTOR_MESSAGE = (
    "# SYSTEM INSTRUCTIONS:\n"
    "Your only job is to extract details from the current conversation to aid in creating "
    "cards. You will be required to follow specific steps for each field on the Card model:\n\n"
    "  1. title (str) - this can be anything the user says.\n"
    "  2. creator (str) - this is the current user's UserProfile PK which is {user_profile_pk}.\n"
    "  3. assignees (list[str]) - if the user assigns it to themselves, use their UserProfile PK, else"
    " you can leave it blank. For example: ['15434'].\n"
    "  4. is_public (boolean) - depends on if the user wants the card to be publicly available or not.\n"
    "  5. column (str) - this always defaults to '213'.\n\n"
)


API_RESPONSE_MESSAGE = (
    "The user attempted to create a card and the server has responded with: {api_response}.\n"
    "If 'pk' was returned by the server, consider the creation successful and inform the user,"
    " else, inform the user that the card creation has failed and to try again later. "
    "Do not mention anything about the server and its response, but respond simply and act "
    "as if you were the one that created the card for the user."
)

AGENT_SYSTEM_MESSAGE = (
    "# SYSTEM INSTRUCTIONS:\n"
    "You are an Assistant AI that is tasked on creating Board Cards for the user. "
    "You must follow the given instructions below to successfully create a Board "
    "Card for the user. You should only follow the instructions one-by-one, do not"
    "immediately ask the user everything at once.\n"
    "  1. Ask the user for the card's title.\n"
    "  2. Ask the user if they want to assign it to themselves or just leave it without "
    "assignees.\n"
    "  3. Ask them if they would like to make the card visible for everyone.\n"
    "  4. Once the 'title', 'assignee', and 'is_public' have been asked, make sure to "
    "reiterate everything and confirm with the user that this is correct.\n"
    "  5. When the user says that it's correct or confirms, call `finish_process` to "
    "end the creation process.\n\n"
    "Below is also the current state of Card, use it as a reference for the rules above:\n"
    "<details> {card_details} <details>\n\n"
    "Lastly, if the user does not want to continue, call `cancel_process` to end the "
    "creation process."
)


agent_tools = []
node_tools = [finish_process, cancel_process]


subgraph_builder = StateGraph(CardState, config_schema=Configuration)

subgraph_builder.add_node(card_agent)
subgraph_builder.add_node(input_helper)
subgraph_builder.add_node(card_extractor_helper)
subgraph_builder.add_node(card_creation_caller_node)
subgraph_builder.add_node("create_card_tool_handler", tool_handler)
subgraph_builder.add_node("initial_tool_handler", tool_handler)
subgraph_builder.add_node("cancel_tool_handler", tool_handler)

subgraph_builder.add_edge(START, "initial_tool_handler")
subgraph_builder.add_edge("initial_tool_handler", "card_agent")
subgraph_builder.add_conditional_edges("card_agent", continue_to_tool)
subgraph_builder.add_edge("input_helper", "card_extractor_helper")
subgraph_builder.add_edge("card_extractor_helper", "card_agent")
subgraph_builder.add_edge("create_card_tool_handler", "card_creation_caller_node")
subgraph_builder.add_edge("cancel_tool_handler", END)
subgraph_builder.add_edge("card_creation_caller_node", END)


bposeats_card_creator_subgraph = subgraph_builder.compile()
