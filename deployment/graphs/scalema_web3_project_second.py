# Import Langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from utils.configuration import Configuration


# Nodes
def node(state: MessagesState):
    return {"messages": state["messages"]}


# Initialize Graph
subgraph_builder = StateGraph(MessagesState, config_schema=Configuration)
subgraph_builder.add_node("node", node)

subgraph_builder.add_edge(START, "node")
subgraph_builder.add_edge("node", END)

scalema_web3_second_subgraph = subgraph_builder.compile()
