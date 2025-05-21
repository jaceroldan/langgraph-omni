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

    extra_data = state.get("extra_data", {})
    choices = extra_data.get("choices")
    value = {}

    if choices:
        value["choices"] = choices

    user_response = interrupt(value=value)

    return {
        "extra_data": {**extra_data, "choices": []},
        "messages": [HumanMessage(content=user_response)]
    }


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
    result = choice_extractor.invoke([SystemMessage(content=CHOICE_EXTRACTOR_MESSAGE)] + state["messages"][-3:])
    dump = result["responses"][0].model_dump(mode="python").get("choice_selection", [])

    extra_data = state.get("extra_data", {})
    if dump:
        extra_data["choices"] = dump

    return {"extra_data": extra_data}


CHOICE_EXTRACTOR_MESSAGE = (
    "You are a choice extraction tool. Given an input text, output a list of answer choices "
    "following these rules:\n\n"
    "1. If the text asks for a name, title, number, or specific value — return an empty list.\n"
    "2. If the question is strictly answerable with 'Yes' or 'No', return ['Yes', 'No'].\n"
    "3. If explicit answer options are mentioned in the text, extract them exactly as written.\n"
    "  - For example: If the message asked the user to pick between Oranges, Apples, and Watermelons,"
    "you may return ['Oranges', 'Apples', 'Watermelons'].\n"
    "  - You may include an option to decline if it's contextually appropriate.\n"
    "4. Prefer explicit answer choices over 'Yes' and 'No'.\n"
    "5. If the question is asking the user to provide data (ex. 'Could you provide...'), you do not need to"
    " output 'Yes' or 'No' for confirmation. Do not return anything in this context.\n"
    "5. If no valid choices are found, return an empty list.\n"
    "6. If the text offers assistance or advice for the future, return an empty list.\n"
    "7. Always ensure extracted choices reflect the context of the text.\n\n"
    "Output only a list of choices. Do not include explanations or any other text."
)
