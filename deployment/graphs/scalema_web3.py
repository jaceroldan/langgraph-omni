# Import general libraries
from typing import Literal, TypedDict

# Import Langgraph
from langchain_core.messages import SystemMessage, merge_message_runs, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages.utils import count_tokens_approximately
from trustcall import create_extractor

# Import utility functions
from utils.configuration import Configuration
from utils.models import models
from utils.nodes import tool_handler
from utils.schemas import Project, ProjectState
from utils.tools import calculator

# Import subgraphs
from graphs.input_handling import input_handler_subgraph

import settings


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


def agent_tool_decision(state: MessagesState) -> Literal["input_handler", "execute_tool"]:  # type: ignore
    """
        Contains decisions for if the agent needs to use one of its tools or continue with asking
        for user inputs.
    """
    tool_calls = state["messages"][-1].tool_calls

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

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT_LARGE,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    merged_messages = list(merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_SYSTEM_MESSAGE)] + trimmed_messages
    ))

    proposal_extractor = create_extractor(
        models[model_name],
        tools=[Project],
        tool_choice=tool_name,
        enable_inserts=True,
        enable_deletes=True
    )

    project_details = state.get("project_details", None)

    result = proposal_extractor.invoke({"messages": merged_messages, "existing": {"Project": project_details}})

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
    project_details = state.get("project_details", None)

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    FORMATTED_MESSAGE = PROPOSAL_AGENT_MESSAGE.format(proposal_details=project_details)
    FORMATTED_HANDLER_MESSAGE = INTERRUPT_HANDLER_MESSAGE.format(proposal_details=project_details)

    response = node_model.invoke([SystemMessage(content=FORMATTED_MESSAGE)] + trimmed_messages)
    return {"messages": [response], "tools": [ToolCall], "handler_message": FORMATTED_HANDLER_MESSAGE}


TRUSTCALL_SYSTEM_MESSAGE = (
    "# TRUSTCALL SYSTEM INSTRUCTIONS\n"
    "You are a tool-routing assistant. Your only role is to analyze user input and call the appropriate tools.\n\n"
    "## Tool Usage Guidelines:\n"
    "- Do **not** generate, guess, or infer any proposal details beyond what is explicitly provided by the user.\n"
    "- If a piece of information is not directly stated by the user, leave it as `None`, `null`, or blank.\n"
    "- Start each interaction with an empty state. Do **not** retain or carry forward any previous proposal details.\n"
    "- Use only the user's current message for context.\n"
    "- Always use the provided tool(s) to store or update proposal information.\n"
    "- When multiple pieces of information are provided, use **parallel tool calls** to handle them simultaneously.\n\n"
    "## What You Must Avoid:\n"
    "- Do **not** fabricate details.\n"
    "- Do **not** assume missing values.\n"
    "- Do **not** use world knowledge, prior experience, or assumptions to fill in blanks.\n\n"
    "## Best Practices:\n"
    "- Extract only what is explicitly stated.\n"
    "- Validate that each field has a clear mapping in the user's input.\n"
    "- Ask for clarification if required, or proceed with blanks if the instruction is to do so.\n"
)

PROPOSAL_AGENT_MESSAGE = (
    "# PROPOSAL GUIDELINES\n"
    "Your primary objective is to help the user complete a proposal form, starting from a blank state.\n"
    "You will collect information from the user, field by field, and may ask clarifying questions as needed.\n"
    "Always try to suggest the most relevant options based on the user's input.\n\n"
    "The current state of the proposal (which may be empty) is shown below:\n"
    "<details> {proposal_details} </details>\n\n"
    "## INSTRUCTIONS FOR MISSING FIELDS\n"
    "- Begin by requesting the **title** of the proposal.\n"
    "- Once the title is provided:\n"
    "  - Suggest possible **project_type** options based on the title.\n"
    "  - Ask the user to select or confirm the appropriate type.\n"
    "- Proceed to gather each missing field individually, following this sequence:\n"
    "    1. title\n"
    "    2. project_type\n"
    "    3. description\n\n"
    "- After steps 1-3 are filled:\n"
    "  - Show the user only the current **title**, **project_type**, and **description**.\n"
    "  - Ask for confirmation and whether they want to refine anything.\n"
    "  - If confirmed, proceed to the next fields:\n"
    "    4. location\n"
    "    5. funding_goal\n"
    "    6. available_shares\n\n"
    "- Once steps 4-6 are completed:\n"
    "  - Call `calculator` to compute the per-share price.\n"
    "  - Inform the user of the calculated share price and let them know they can adjust values if needed.\n"
    "  - Do not proceed until the user confirms they're okay with the share price.\n\n"
    "- Continue with:\n"
    "    7. minimum_viable_fund\n"
    "    8. funding_date_completion\n\n"
    "- For the final fields (9-11), consider the context of the proposal and suggest useful additions:\n"
    "    9. key_milestone_dates\n"
    "    10. financial_documents\n"
    "    11. legal_documents\n\n"
    "- If the user requests to review progress:\n"
    "  - Show only the fields that have been filled.\n"
    "  - Do not display missing or incomplete fields.\n\n"
    "- Once all fields are completed:\n"
    "  - Present the full proposal to the user for final review.\n"
    "  - Ask if they would like to:\n"
    "    - Refine or add any more information\n"
    "    - Save it as a draft\n"
    "    - Submit it for review by Scalema Admins\n\n"
    "- Maintain a friendly and professional tone throughout.\n"
    "- Include short, encouraging comments when responding to user inputs."
)

INTERRUPT_HANDLER_MESSAGE = (
    "# INSTRUCTIONS\n"
    "You are provided with the current state of the proposal:\n"
    "<details>{proposal_details}</details>\n\n"
    "Your role is to carefully evaluate the user's input and determine the appropriate action using a tool.\n"
    "**You are not allowed to respond to the user directly; only interact via the tool.**\n\n"
    "## Tool Usage Guidelines:\n"
    "- If the user explicitly states **not to continue**, call `ToolCall` with the `finalize` argument.\n"
    "- If the user explicitly requests to **save as a draft**, call `ToolCall` with the `finalize` argument.\n"
    "- If the user explicitly instructs to **submit** the proposal and **all required fields are filled**,"
    "call `ToolCall` with the `finalize` argument.\n"
    "- If **any required fields are missing or empty**, call `ToolCall` with the `retry` argument.\n"
    "- By default, call `ToolCall` with the `retry` argument unless an explicit finalize condition is met."
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
