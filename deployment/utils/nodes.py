from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt


def fake_node(state):
    """Fake node that does nothing. Used for testing."""
    return state


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

    return {**state,  "messages": tool_call_messages}


def input_helper(state: MessagesState) -> MessagesState:
    """
        Helper node used for receiving the User's response for HITL.
    """
    user_response = interrupt("")
    return {**state, "messages": [HumanMessage(content=user_response)]}
