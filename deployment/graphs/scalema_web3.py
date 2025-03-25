# Import general libraries
# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END

# Import utility functions
from utils.configuration import Configuration
from utils.schemas import ProjectState

# Import subgraphs
from graphs.scalema_web3_project_first import scalema_web3_main_subgraph


def node(state: MessagesState):
    return {"messages": state["messages"]}


# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)
subgraph_builder.add_node("project_creation_subgraph_A", scalema_web3_main_subgraph)
subgraph_builder.add_node("project_creation_subgraph_B", node)

subgraph_builder.add_edge(START, "project_creation_subgraph_A")
subgraph_builder.add_edge("project_creation_subgraph_A", "project_creation_subgraph_B")
subgraph_builder.add_edge("project_creation_subgraph_B", END)

scalema_web3_subgraph = subgraph_builder.compile()
