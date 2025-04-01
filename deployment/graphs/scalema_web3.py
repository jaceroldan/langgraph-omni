# Import general libraries
from typing import Literal, TypedDict
from datetime import datetime

# Import Langgraph
from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from trustcall import create_extractor

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.nodes import tool_handler
from utils.schemas import Project, ProjectState
from utils.tools import calculator

# Import subgraphs
from graphs.input_handling import input_handler_subgraph


# Tools
class ToolCall(TypedDict):
    """
        Decision on which tool to use
    """
    tool_type: Literal["retry", "finalize"]


def handler_decision(state: MessagesState) -> Literal["project_helper", "tool_handler", END]:  # type: ignore
    """
        Contains decisions for the next node to be called in the project proposal process.
    """

    tool_calls = state["messages"][-1].tool_calls

    # WARNING: this should never be returned unless a use case wasn't handled by the handler.
    if not tool_calls:
        return END

    match (tool_calls[0]["args"]["tool_type"]):
        case "retry":
            return "project_helper"
        case "finalize":
            return "tool_handler"
        case _:
            return "tool_handler"


def agent_tool_decision(state: MessagesState) -> Literal["input_handler", "execute_tool"]:  # type: ignore
    """
        Contains decisions for if the agent needs to use one of its tools or continue with asking
        for user inputs.
    """
    tool_calls = state["messages"][-1].tool_calls

    if not tool_calls:
        return "input_handler"

    for item in tool_calls:
        if item["name"] == "calculator":
            return "execute_tool"
    return "input_handler"


# Nodes
def project_helper_node(state: ProjectState, config: RunnableConfig) -> ProjectState:
    """
        Handles the tool call from the parent graph.
    """
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    tool_name = "Project"

    TRUSTCALL_FORMATTED_MESSAGE = TRUSTCALL_SYSTEM_MESSAGE.format(
        time=datetime.now().isoformat(),
        proposal_details=state.get("project_details", None)
    )
    merged_messages = list(merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_FORMATTED_MESSAGE)] + state["messages"][:-1]
    ))

    proposal_extractor = create_extractor(
        models[model_name],
        tools=[Project],
        tool_choice=tool_name,
        enable_inserts=True
    )

    result = proposal_extractor.invoke({"messages": merged_messages})

    extracted_project_details = result["responses"][0]

    # Confirm the tool call made in the parent graph
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]

    # Extract the changes made by Trustcall and add them to the message.
    message = {
        "content": '',
        "role": "tool",
        "tool_call_id": tool_call_id
    }

    return {"messages": [message], "project_details": extracted_project_details.model_dump(mode="python")}


def project_agent(state: ProjectState, config: RunnableConfig) -> ProjectState:
    """
        Handles in summarizing the information from the project schema. Processes that information
        using the agent and responds accordingly.
    """

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    node_model = models[model_name].bind_tools(agent_tools)
    project_details = state["project_details"]

    FORMATTED_MESSAGE = PROPOSAL_AGENT_MESSAGE.format(proposal_details=project_details)
    FORMATTED_HANDLER_MESSAGE = INTERRUPT_HANDLER_MESSAGE.format(proposal_details=project_details)

    response = node_model.invoke([SystemMessage(content=FORMATTED_MESSAGE)] + state["messages"])
    return {"messages": [response], "tools": [ToolCall], "handler_message": FORMATTED_HANDLER_MESSAGE}


TRUSTCALL_SYSTEM_MESSAGE = (
    "Your only instruction is to reflect on the interaction and call the appropriate tool. "
    "Always use the provided tool to retain any necessary information. "
    "Use parallel tool calls to handle updates and insertions simultaneously. "
    "Never provide data about the proposal that are not from the user's own messages."
    "\nSystem Time: {time}"
    "The following is the current state of the proposal: \n{proposal_details}"
)

PROPOSAL_AGENT_MESSAGE = (
    "\n# PROPOSAL GUIDELINES"
    "\nYour only goal is to complete a proposal form using inputs from a user. You will be given the "
    "current state of the proposal and you must base which instruction to follow using that information. "
    "The following is the current state of the proposal:"
    "\n{proposal_details}"
    "\n## MISSING FIELDS INSTRUCTIONS"
    "\n\nFollow the instructions below for missing fields:"
    "\n\t- Always ask for the title first, if possible."
    "\n\t- If the title is provided, provide some suggestions on the what kind of project (project_type) "
    "it is depending on the title. Ask the user for their input."
    "\n\t-Ask for information about each missing field individually until no more fields are empty."
    "\nThe fields should be filled by the user sequentially, the order being:"
    "\n\t\t1. title"
    "\n\t\t2. project_type"
    "\n\t\t3. description"
    "\n\t-After 1-3 has been filled, show only the current title, project_type, and description to the user. "
    "Ask them if the information is correct and if they would like to refine it further."
    "Once they confirm that it is correct and have not explicitly said to finish, continue with filling "
    "the next fields sequentially:"
    "\n\t\t4. location"
    "\n\t\t5. funding_goal"
    "\n\t\t6. available_shares"
    "\n\t-After 5-6 has been filled, calculate the per-share price by calling `calculator`. After "
    "calculating, inform the user about this and reassure them that they can readjust if needed."
    " Do not ask to continue unless they are alright about the share price then you can ask to continue."
    "\n\t\t7. minimum_viable_fund"
    "\n\t\t8. funding_date_completion"
    "\n\t-For 9-11, always consider the context of the proposal and suggest possible important items for the "
    "user to provide."
    "\n\t\t9. key_milestone_dates"
    "\n\t\t10. financial_documents"
    "\n\t\t11. legal_documents"
    "\nIf the user asks to see the current progress, only show fields that are filled, "
    "do not show fields that are missing. Once everything has been filled, show the complete "
    "proposal to the user and ask if it is correct or if they want to add more information or "
    "for you to refine the proposal."
    "\nLastly, before finishing, always ask the user if they would like to save it as a draft and refine "
    "it later or to submit it for review by Scalema Admins."
    "\nAlways be friendly and include a short comment on the user's response but be professional. "
)

INTERRUPT_HANDLER_MESSAGE = (
    "The following is the current state of the proposal:"
    "\n\n{proposal_details}"
    "\nYour only instruction is to carefully reason out the user's messages and react accordingly. "
    "You are equipped with a tool, you are not allowed to respond to the user and you can only use the tool."
    "\n\nGuidelines for tool usage:"
    "\n\t- If the user explicitly states not to continue, end the conversation by calling "
    "ToolCall with the `finalize` argument."
    "\n\t- If the user explicitly states to save it as a draft, end the conversation by calling "
    "ToolCall with the `finalize` argument."
    "\n\t- If the user explicitly states to submit it and the fields are completely filled, end the "
    "conversation by calling ToolCall with the `finalize` argument."
    "\n\t- If some fields of the proposal are still None or empty, call ToolCall with the `retry` argument."
    "\n\t- As default response, always call ToolCall with the `retry` argument."
)


agent_tools = [calculator]

# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)

subgraph_builder.add_node("project_helper", project_helper_node)
subgraph_builder.add_node("project_agent", project_agent)
subgraph_builder.add_node("input_handler", input_handler_subgraph)
subgraph_builder.add_node("tool_handler", tool_handler)
subgraph_builder.add_node("execute_tool", ToolNode(agent_tools))

subgraph_builder.add_edge(START, "project_helper")
subgraph_builder.add_edge("project_helper", "project_agent")
subgraph_builder.add_conditional_edges("project_agent", agent_tool_decision)
subgraph_builder.add_edge("execute_tool", "project_agent")
subgraph_builder.add_conditional_edges("input_handler", handler_decision)
subgraph_builder.add_edge("tool_handler", END)

scalema_web3_subgraph = subgraph_builder.compile()
