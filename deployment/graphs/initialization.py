# Import Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# Utils
from utils.configuration import Configuration
from utils.memory import MemoryState, load_memory

# Lib
from lib.sileo.restmodel import Defaults

import settings


def initialize(state: MemoryState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    auth_token = configuration.auth_token

    # Set Default properties
    Defaults.headers = {
        'Authorization': auth_token,
        'Content-Type': 'application/json',
        'X-App-Version': '1.0.0'
    }
    Defaults.base_url = f"{settings.API_URL}/"

    return {**state}


builder = StateGraph(MemoryState, config_schema=Configuration)
builder.add_node(initialize)
builder.add_node(load_memory)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "load_memory")
builder.add_edge("load_memory", END)

init_graph = builder.compile()
