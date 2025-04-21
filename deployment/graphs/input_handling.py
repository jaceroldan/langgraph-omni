# Import general libraries
from typing import Callable, List

# Import Langgraph
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt
from langchain_core.messages.utils import count_tokens_approximately

# Import utility functions
from utils.configuration import Configuration
from utils.models import models


# Schema
class InputState(MessagesState):
    tools: List[Callable]
    handler_message: str
    extra_data: dict


# Nodes
def input_helper(state: InputState) -> InputState:
    """
        Helper node used for receiving the User's response for HITL.
    """
    user_response = interrupt("")
    return {**state, "messages": [HumanMessage(content=user_response)]}


def interrupt_handler(state: InputState, config: RunnableConfig) -> MessagesState:
    """
        Handles the previous user input and provides instructions for the next tool call.
    """
    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    tools = state["tools"]
    handler_message = state["handler_message"]
    messages = state["messages"]

    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=500,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    node_model = models[model_name].bind_tools(tools, parallel_tool_calls=False)

    response = node_model.invoke(
        [SystemMessage(content=handler_message)] + trimmed_messages)
    return {"messages": [response]}


# Initialize Graph
handler_builder = StateGraph(InputState, config_schema=Configuration)
handler_builder.add_node("input", input_helper)
handler_builder.add_node("handler", interrupt_handler)

handler_builder.add_edge(START, "input")
handler_builder.add_edge("input", "handler")
handler_builder.add_edge("handler", END)

input_handler_subgraph = handler_builder.compile()
