# Import general libraries
from typing import Literal  # , TypedDict

# Import Langgraph
from langchain_core.messages import SystemMessage, merge_message_runs, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import ToolNode
from trustcall import create_extractor

# Import utility functions
from utils.configuration import Configuration
from utils.models import models, SilentHandler
from utils.nodes import tool_handler, input_helper
from utils.schemas import Project, ProjectState
from utils.tools import calculator

import settings


# Tools
@tool
def get_user_input():
    """
        Fake tool to get user input.
    """
    return


def continue_to_tool(state: MessagesState) -> Literal[
                                                "input_tool_handler",
                                                "tool_executor",
                                                "__end__"]:  # type: ignore
    """
        Contains decisions for if the agent needs to use one of its tools or continue with asking
        for user inputs.
    """

    tool_calls = state["messages"][-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END

    tool_name = tool_calls[0]["name"]
    match tool_name:
        case _ if tool_name in [tool.get_name() for tool in agent_tools]:
            return "tool_executor"
    return "input_tool_handler"


# Nodes
def project_helper(state: ProjectState, config: RunnableConfig) -> ProjectState:
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
        end_on="human",
        allow_partial=False,
        include_system=True
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

    return {"project_details": extracted_project_details.model_dump(mode="python")}


def project_agent(state: ProjectState, config: RunnableConfig) -> ProjectState:
    """
        Handles in summarizing the information from the project schema. Processes that information
        using the agent and responds accordingly.
    """

    silent_handler = SilentHandler()

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name
    project_agent_model = models[model_name].bind_tools(node_tools)
    completion_agent_model = models[model_name].bind_tools(agent_tools)

    project_details = state.get("project_details", None)

    input_trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT_SMALL,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    FORMATTED_ROUTER_MESSAGE = PROPOSAL_ROUTER_MESSAGE.format(proposal_details=project_details)
    FORMATTED_COMPLETION_MESSAGE = PROPOSAL_COMPLETION_MESSAGE.format(proposal_details=project_details)

    route_response = project_agent_model.invoke(
        [SystemMessage(content=FORMATTED_ROUTER_MESSAGE)] + input_trimmed_messages,
        config={"callbacks": [silent_handler]})
    agent_response = completion_agent_model.invoke(
        [SystemMessage(content=FORMATTED_COMPLETION_MESSAGE)] + trimmed_messages)

    print("\n\n")
    print("Agent Response: ", agent_response)
    print("\n\n")
    if agent_response:
        pass

    return {"messages": [route_response]}


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

PROPOSAL_ROUTER_MESSAGE = (
    "# SYSTEM INSTRUCTIONS\n"
    "You are assisting the user with completing a proposal form. Your behavior must follow these strict rules:\n\n"
    "## DO NOT RESPOND WITH ANY TEXT.\n"
    "- Never respond with natural language.\n"
    "- Your output must either be a tool call (e.g., `get_user_input`) or an empty string (`''`).\n\n"
    "## WHEN TO CALL TOOLS:\n"
    "- If **any required fields are missing or empty**, call `get_user_input`.\n\n"
    "## WHEN TO RETURN AN EMPTY STRING:\n"
    "- If the user explicitly says **not to continue**.\n"
    "- If the user explicitly asks to **save as a draft**.\n"
    "- If the user instructs to **submit** the proposal and **all required fields are complete**.\n\n"
    "Currently, the proposal is in the following state:\n"
    "<details> {proposal_details} </details>\n\n"
    "Repeat: **You must not generate any messages or explanations**. Respond ONLY using a tool call or return an "
    "empty response.\n"
)

PROPOSAL_COMPLETION_MESSAGE = (
    "**IMPORTANT**: If the user explicitly states to stop or not to continue, you must ignore all "
    "instructions below and respond with absolutely nothing — no tool calls, no text, no response.\n\n"
    "# PROPOSAL GUIDELINES\n"
    "Starting from a blank state, you will collect information from the user, field by field, and may ask "
    "clarifying questions as needed. Always try to suggest the most relevant options based on the user's input.\n\n"
    "The current state of the proposal (which may be empty) is shown below:\n"
    "<details> {proposal_details} </details>\n\n"

    "## INSTRUCTIONS FOR MISSING FIELDS\n"
    "- Begin by requesting the `title` of the proposal.\n"
    "- Once the title is provided:\n"
    "  - Suggest possible `project_type` options based on the title.\n"
    "  - If given a description, infer the `project_type` from it.\n"
    "  - Ask the user to select or confirm the appropriate type.\n"
    "- Proceed to gather each missing field individually, following this sequence:\n"
    "    1. title\n"
    "    2. project_type\n"
    "    3. description\n\n"
    "- After steps 1-3 are filled:\n"
    "  - Show the user only the current `title`, `project_type`, and `description`.\n"
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
    "- Include short, encouraging comments when responding to user inputs but always be professional.\n\n"
    "**IMPORTANT**: If the user explicitly states to stop or not to continue, you must ignore all previous "
    "instructions and respond with absolutely nothing — no tool calls, no text, no response.\n"
)


agent_tools = [calculator]
node_tools = [get_user_input]

# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)

subgraph_builder.add_node(project_helper)
subgraph_builder.add_node(project_agent)
subgraph_builder.add_node(input_helper)
subgraph_builder.add_node("initial_tool_handler", tool_handler)
subgraph_builder.add_node("input_tool_handler", tool_handler)
subgraph_builder.add_node("tool_executor", ToolNode(agent_tools))

subgraph_builder.add_edge(START, "initial_tool_handler")
subgraph_builder.add_edge("initial_tool_handler", "project_agent")
subgraph_builder.add_conditional_edges("project_agent", continue_to_tool)
subgraph_builder.add_edge("tool_executor", "project_agent")
subgraph_builder.add_edge("input_tool_handler", "input_helper")
subgraph_builder.add_edge("input_helper", "project_helper")
subgraph_builder.add_edge("project_helper", "project_agent")

scalema_web3_subgraph = subgraph_builder.compile()
