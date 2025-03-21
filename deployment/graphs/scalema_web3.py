# Import general libraries
from typing import Optional
from pydantic import BaseModel, Field
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

# Tools


# Schema
class Project(BaseModel):
    """
        This is the schema format for a project.
    """
    title: Optional[str] = Field(description="Title of the project. (eg. The Residences at Greenbelt)")
    project_type: Optional[str] = Field(description="Type of project. (eg. Residential - Condominium)")
    description: Optional[str] = Field(
        description="Description of the project itself. This can include the location, amenities, and other details."
    )


# Nodes
def proposal_helper_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
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
                          for existing_item in existing_items] if existing_items else None)

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


def project_agent(state: MessagesState, config: RunnableConfig):
    """
        Handles in summarizing the information from the project schema. Processes that information
        using the agent and responds accordingly.

        CURRENTLY EMPTY
    """
    return {"messages": state["messages"]}


TRUSTCALL_SYSTEM_MESSAGE = (
    "Your only instruction is to reflect on the interaction and call the appropriate tool. "
    "Always use the provided tool to retain any necessary information. "
    "Use parallel tool calls to handle updates and insertions simultaneously. "
    "\nSystem Time: {time}"
)


# Initialize Graph
subgraph_builder = StateGraph(MessagesState, config_schema=Configuration)

subgraph_builder.add_node("proposal_helper", proposal_helper_node)
subgraph_builder.add_node("project_agent", project_agent)

subgraph_builder.add_edge(START, "proposal_helper")
subgraph_builder.add_edge("proposal_helper", "project_agent")
subgraph_builder.add_edge("project_agent", END)

scalema_web3_subgraph = subgraph_builder.compile()
