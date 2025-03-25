# Import general libraries
from typing import Literal, TypedDict
from datetime import datetime
from uuid import uuid4

# Import Langgraph
from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from trustcall import create_extractor

# Import utility functions
from utils.configuration import Configuration
from utils.trustcall import Spy, extract_tool_info
from utils.models import models
from utils.nodes import exit_handler
from utils.schemas import Project, ProjectState

# Import subgraphs
from graphs.input_handling import input_handler_subgraph, InputState


# Tools
class ToolCall(TypedDict):
    """
        Decision on which tool to use
    """
    tool_type: Literal["retry_proposal", "finalize_proposal"]


def handler_decision(state: MessagesState) -> Literal["project_helper", "exit", END]:  # type: ignore
    """
        Contains decisions for the next node to be called in the project proposal process.
    """

    tool_calls = state["messages"][-1].tool_calls

    # WARNING: this should never be returned unless a use case wasn't handled by the handler.
    if not tool_calls:
        return END

    match (tool_calls[0]["args"]["tool_type"]):
        case "retry_proposal":
            return "project_helper"
        case "finalize_proposal":
            return "exit"
        case _:
            return "exit"


# Nodes
def project_helper_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
        Handles the tool call from the parent graph.
    """
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    user_profile_pk = configurable.user_profile_pk

    namespace = ("proposal", user_profile_pk)

    # Retrieve memories from the store
    existing_items = store.search(namespace)
    tool_name = "Project"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items][:1]) if existing_items else None

    TRUSTCALL_FORMATTED_MESSAGE = TRUSTCALL_SYSTEM_MESSAGE.format(time=datetime.now().isoformat())
    updated_messages = list(
        merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_FORMATTED_MESSAGE)] + state["messages"][:-1]))

    spy = Spy()

    proposal_extractor = create_extractor(
        models[model_name],
        tools=[Project],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    result = proposal_extractor.invoke({"messages": updated_messages, "existing": existing_memories})

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid4())),
            r.model_dump(mode="json"),
        )

    # Confirm the tool call made in the parent graph
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]

    # Extract the changes made by Trustcall and add them to the message.
    memory_update_msg = extract_tool_info(spy.called_tools, tool_name)
    message = {"content": memory_update_msg, "role": "tool", "tool_call_id": tool_call_id}

    return {"messages": [message]}


def project_agent(state: MessagesState, config: RunnableConfig, store: BaseStore) -> InputState:
    """
        Handles in summarizing the information from the project schema. Processes that information
        using the agent and responds accordingly.
    """
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    node_model = models[model_name]
    user_profile_pk = configurable.user_profile_pk

    namespace = ("proposal", user_profile_pk)

    # Retrieve memory from the store
    existing_items = store.search(namespace)
    current_proposal_details = [existing_item.value for existing_item in existing_items][0] if existing_items else None

    FORMATTED_MESSAGE = PROPOSAL_AGENT_MESSAGE.format(proposal_details=current_proposal_details)
    response = node_model.invoke([SystemMessage(content=FORMATTED_MESSAGE)] + state["messages"])
    return {"messages": [response], "tools": [ToolCall], "handler_message": INTERRUPT_HANDLER_MESSAGE}


TRUSTCALL_SYSTEM_MESSAGE = (
    "Your only instruction is to reflect on the interaction and call the appropriate tool. "
    "Always use the provided tool to retain any necessary information. "
    "Use parallel tool calls to handle updates and insertions simultaneously. "
    "Lastly, do not provide any new information other than what was provided by the user."
    "\nSystem Time: {time}"
    "\n\nYou will be given existing memories of the user's previous interactions below. "
    "However, only the first instance should be used for reference on extraction."
)

PROPOSAL_AGENT_MESSAGE = (
    "The following is the current state of the proposal:"
    "\n\n<details> {proposal_details} </details>"
    "Don't forget to always show the details of the proposal if all required information has been provided."
    "If the details have already been confirmed, ask the user if they would like you to refine it further."
)

INTERRUPT_HANDLER_MESSAGE = (
    "Your only instruction is to reflect on the interaction and call ToolCall with the appropriate argument. "
    "\n\nGuidelines for tool usage:"
    "\n\t- If a user accepts the changes call ToolCall with the 'finalize_proposal' argument."
    "\n\t- If a user does not wish to continue, end the conversation by calling "
    "ToolCall with the 'finalize_proposal' argument."
    "\n\t- If a user requests to make changes or provides new information, call "
    "ToolCall with the 'retry_proposal' argument."
    "\n\t- If a user asks you to refine or provide information for the project, "
    "call ToolCall with the 'retry_proposal' argument."
    "\n\nDepending on the user's message, always call ToolCall with the appropriate argument."
)


# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)

subgraph_builder.add_node("project_helper", project_helper_node)
subgraph_builder.add_node("project_agent", project_agent)
subgraph_builder.add_node("input_handler", input_handler_subgraph)
subgraph_builder.add_node("exit", exit_handler)

subgraph_builder.add_edge(START, "project_helper")
subgraph_builder.add_edge("project_helper", "project_agent")
subgraph_builder.add_edge("project_agent", "input_handler")
subgraph_builder.add_edge("exit", END)
subgraph_builder.add_conditional_edges("input_handler", handler_decision)

scalema_web3_main_subgraph = subgraph_builder.compile()
