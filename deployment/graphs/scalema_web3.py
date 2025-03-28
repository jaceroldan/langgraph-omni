# Import general libraries

# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END

# Import utility functions
from utils.configuration import Configuration

# Import subgraphs
from graphs.scalema_web3_project_first import scalema_web3_main_subgraph
from graphs.scalema_web3_project_second import scalema_web3_second_subgraph


# Tools


# Nodes
def connector_node(state: MessagesState):
    # Fake node used to properly display the graph.
    return state


# Initialize Graph
subgraph_builder = StateGraph(MessagesState, config_schema=Configuration)
subgraph_builder.add_node("project_creation_subgraph_A", scalema_web3_main_subgraph)
subgraph_builder.add_node("project_creation_subgraph_B", scalema_web3_second_subgraph)
subgraph_builder.add_node("connector", connector_node)

subgraph_builder.add_edge(START, "project_creation_subgraph_A")
subgraph_builder.add_edge("project_creation_subgraph_A", "connector")
subgraph_builder.add_edge("connector", "project_creation_subgraph_B")
subgraph_builder.add_edge("project_creation_subgraph_B", END)

scalema_web3_subgraph = subgraph_builder.compile()
