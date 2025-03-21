
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

from utils.configuration import Configuration


def proposal_helper_node(state: MessagesState, config: RunnableConfig):
    message = AIMessage(content="This is a proposal node.")
    return {"messages": [message]}


# Build the graph
subgraph_builder = StateGraph(MessagesState, config_schema=Configuration)

subgraph_builder.add_node("proposal_helper", proposal_helper_node)

subgraph_builder.add_edge(START, "proposal_helper")
subgraph_builder.add_edge("proposal_helper", END)

# checkpointer = MemorySaver()
scalema_web3_subgraph = subgraph_builder.compile()
