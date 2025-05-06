from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from trustcall import create_extractor

from utils.schemas import InputState, Choices
from utils.configuration import Configuration
from utils.models import models


def fake_node():
    """Fake node that does nothing. Used for testing."""
    return {}


def tool_handler(state: MessagesState):
    """
        Fake node to used whenever there is a tool call without
        a tool message.
    """

    tool_calls = state["messages"][-1].tool_calls

    tool_call_messages = []
    if tool_calls is not None and isinstance(tool_calls, list):
        for tool_call in tool_calls:
            tool_call_message = {
                "content": "",
                "role": "tool",
                "tool_call_id": tool_call["id"]
            }
            tool_call_messages.append(tool_call_message)

    return {"messages": tool_call_messages}


def input_helper(state: InputState) -> MessagesState:
    """
        Helper node used for receiving the User's response for HITL.
    """

    choices = state["extra_data"].get("choices")
    value = {}

    if choices:
        value["choices"] = choices

    user_response = interrupt(value=value)
    return {"messages": [HumanMessage(content=user_response)]}


def choice_extractor_helper(state: InputState, config: RunnableConfig) -> InputState:
    """
        Currently handles extraction of choices from the previous node.
    """

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name

    choice_extractor = create_extractor(
        models[model_name],
        tools=[Choices],
        tool_choice="Choices",
        enable_inserts=True
    )
    result = choice_extractor.invoke([SystemMessage(content=CHOICE_EXTRACTOR_MESSAGE)] + state["messages"][-2:])
    dump = result["responses"][0].model_dump(mode="python").get("choice_selection", [])

    extra_data = state.get("extra_data", {})
    if dump:
        extra_data["choices"] = dump

    return {"extra_data": extra_data}


CHOICE_EXTRACTOR_MESSAGE = (
    "# INSTRUCTIONS:\n"
    "Given an input text, perform the following steps:\n"
    "1. If the text is asking for a name, title, number, or value of something, return an empty list.\n"
    "2. Carefully check if the text contains a question that is answerable strictly and only by "
    "'Yes' or 'No'.\n"
    "\t- If so, output exactly ['Yes', 'No'] as the answer choices.\n"
    "3. Otherwise, scan the text for any explicitly mentioned answer choices. Extract and output "
    "these choices exactly as they appear; do not add or modify them.\n"
    "\t- You may add an option to decline the question if it is appropriate."
    "4. If no explicit answer choices are found, only output an empty list.\n"
    "5. If the text is offering assistance on a future endeavor, only output an empty list.\n"
    "6. When creating choices, always make sure to take note of the context and incorporate it into "
    "the choice itself.\n"
    "Always ensure you use the provided tool only to capture and retain any necessary information."
)
