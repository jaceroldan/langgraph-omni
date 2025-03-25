# Import general libraries
from typing import TypedDict, Literal

# Import Langgraph
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

# Import utility functions
from utils.configuration import Configuration
from utils.schemas import ProjectState
from utils.models import models
from utils.nodes import exit_handler

# Import subgraphs
from graphs.scalema_web3_project_first import scalema_web3_main_subgraph


# Tools
class RouteDecision(TypedDict):
    """
        Determines on whether to continue to the next part of project creation or not.
    """
    route: Literal["continue", "end"]


def routing_decision(state: MessagesState) -> Literal["continue_handler", "exit_handler", END]:  # type: ignore
    """
        Contains decisions for the next node to be called in the project proposal process.
    """

    tool_calls = state["messages"][-1].tool_calls

    # WARNING: this should never be returned unless a use case wasn't handled by the handler.
    if not tool_calls:
        return END

    match (tool_calls[0]["args"]["route"]):
        case "continue":
            return "continue_handler"
        case "end":
            return "exit_handler"
        case _:
            return "exit_handler"


# Nodes
def node(state: MessagesState):
    return {"messages": state["messages"]}


def web3_router_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
        Handles the previous user input and provides instructions for the next tool call.
    """
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    node_model = models[model_name]
    user_profile_pk = configurable.user_profile_pk

    namespace = ("proposal", user_profile_pk)

    # Retrieve memory from the store
    existing_items = store.search(namespace)
    current_proposal_details = [existing_item.value for existing_item in existing_items][0] if existing_items else None

    FORMATTED_MESSAGE = PIPELINE_AGENT_MESSAGE.format(project_details=current_proposal_details)

    node_model = models[model_name].bind_tools([RouteDecision], parallel_tool_calls=False)
    response = node_model.invoke([SystemMessage(content=FORMATTED_MESSAGE)])
    return {"messages": [response]}


PIPELINE_AGENT_MESSAGE = (
    "Your only instruction is to reflect on the interaction and call RouteDecision with the appropriate argument. "
    "\n\nGuidelines for tool usage:"
    "\n\t- A proposal is considered semi-complete if it has a title, a project_type, and description."
    "\n\t- If the proposal is semi-complete, call RouteDecision with the 'continue' argument."
    "\n\t- If the proposal is not semi-complete or missing information, call RouteDecision with the 'end' argument."
    "\n\nBelow is the current state of the project proposal: "
    "<proposal>{project_details}</proposal>"
    "\nUse this information in determining which argument to use."
)


# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)
subgraph_builder.add_node("project_creation_subgraph_A", scalema_web3_main_subgraph)
subgraph_builder.add_node("router_agent", web3_router_agent)
subgraph_builder.add_node("project_creation_subgraph_B", node)
subgraph_builder.add_node("exit_handler", exit_handler)
subgraph_builder.add_node("continue_handler", exit_handler)

subgraph_builder.add_edge(START, "project_creation_subgraph_A")
subgraph_builder.add_edge("project_creation_subgraph_A", "router_agent")
subgraph_builder.add_conditional_edges("router_agent", routing_decision)
subgraph_builder.add_edge("continue_handler", "project_creation_subgraph_B")
subgraph_builder.add_edge("project_creation_subgraph_B", END)
subgraph_builder.add_edge("exit_handler", END)

scalema_web3_subgraph = subgraph_builder.compile()
