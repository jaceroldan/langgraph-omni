# Import general libraries
from typing import Literal  # , TypedDict
from datetime import datetime

# Import Langgraph
from langchain_core.messages import SystemMessage, merge_message_runs, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import ToolNode
from langgraph.pregel import RetryPolicy
from trustcall import create_extractor

# Import utility functions
from utils.configuration import Configuration
from utils.models import models, SilentHandler
from utils.nodes import tool_handler, input_helper
from utils.schemas import Project, ProjectState, Choices
from utils.tools import calculator

import settings


# Tools
@tool
def get_user_input():
    """Fake tool to get user input."""
    return


@tool
def finish_proposal():
    """Fake tool to finish proposal."""
    return


def continue_to_tool(state: MessagesState) -> Literal[
                                                "input_tool_handler",
                                                "tool_executor",
                                                "end_tool_handler",
                                                "__end__"]:
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
        case "finish_proposal":
            return "end_tool_handler"
        case _ if tool_name in [tool.get_name() for tool in agent_tools]:
            return "tool_executor"
        case "get_user_input":
            return "input_tool_handler"
    return END


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

    FORMATTED_MESSAGE = TRUSTCALL_SYSTEM_MESSAGE.format(time=datetime.now())
    merged_messages = list(merge_message_runs(
            messages=[SystemMessage(content=FORMATTED_MESSAGE)] + trimmed_messages
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
    project_agent_model = models[model_name]
    tool_caller_model = models["tool-calling-model"].bind_tools(agent_tools + node_tools)

    project_details = state.get("project_details", None)

    merged_message = merge_message_runs(state["messages"])

    input_trimmed_messages = trim_messages(
        merged_message,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT_SMALL,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    trimmed_messages = trim_messages(
        merged_message,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.TOKEN_LIMIT_SMALL,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False
    )

    FORMATTED_ROUTER_MESSAGE = PROPOSAL_ROUTER_MESSAGE.format(proposal_details=project_details)
    FORMATTED_COMPLETION_MESSAGE = PROPOSAL_COMPLETION_MESSAGE.format(
        proposal_details=project_details, time=datetime.now())

    # TODO: Figure out how to force a model to both output a Text Response and a Tool Call.
    #       Usually you get either a tool call or an AI response but not both. However,
    #       GPT-4o models sometimes are able to output both at random times for no reason at
    #       all. Just need to make them consistent.

    tool_caller_model_response = tool_caller_model.invoke(
        [SystemMessage(content=FORMATTED_ROUTER_MESSAGE)] + input_trimmed_messages,
        config={"callbacks": [silent_handler]})
    agent_response = project_agent_model.invoke(
        [SystemMessage(content=FORMATTED_COMPLETION_MESSAGE)] + trimmed_messages)

    return {"messages": [agent_response, tool_caller_model_response]}


def post_processor(state: ProjectState, config: RunnableConfig) -> ProjectState:
    """
        Currently handles extraction of choices from the previous node.
    """

    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.model_name

    choice_extractor = create_extractor(
        models[model_name],
        tools=[Choices],
        tool_choice="Choices",
        enable_inserts=True
    )
    result = choice_extractor.invoke([SystemMessage(content=CHOICE_EXTRACTOR_MESSAGE), state["messages"][-2]])
    dump = result["responses"][0].model_dump(mode="python").get("choice_selection", [])

    extra_data = state.get("extra_data", {})
    if dump:
        extra_data["choices"] = dump

    return {**state, "extra_data": extra_data}


TRUSTCALL_SYSTEM_MESSAGE = (
    "# TRUSTCALL SYSTEM INSTRUCTIONS\n"
    "You are a tool-routing assistant. Your only role is to analyze user input and call the appropriate tools.\n\n"
    "## Tool Usage Guidelines:\n"
    "- Do **not** generate, guess, or infer any proposal details beyond what is explicitly provided by the user.\n"
    "- If a piece of information is not directly stated by the user, leave it as `None`, `null`, or blank.\n"
    "- Start each interaction with an empty state. Do **not** retain or carry forward any previous proposal details.\n"
    "- Use only the user's current message for context.\n"
    "- Always use the provided tool(s) to store or update proposal information.\n"
    "- You must call **exactly one tool per turn**. Never call more than one tool at a time.\n\n"
    "## What You Must Avoid:\n"
    "- Do **not** fabricate details.\n"
    "- Do **not** assume missing values.\n"
    "- Do **not** use world knowledge, prior experience, or assumptions to fill in blanks.\n\n"
    "- Given the system time, do not accept dates in the past. If multiple are presented, do not"
    " only accept the dates in the future.\n\n"
    "## Best Practices:\n"
    "- Extract only what is explicitly stated.\n"
    "- Validate that each field has a clear mapping in the user's input.\n"
    "- Ask for clarification if required, or proceed with blanks if the instruction is to do so.\n"
    "\nSystem Time: {time}"
)

CHOICE_EXTRACTOR_MESSAGE = (
    "# INSTRUCTIONS:\n"
    "Given an input text, perform the following steps:\n"
    "1. If the text is asking for a name, title, number, or value of something, return an empty list.\n"
    "2. Carefully check if the text contains a question that is answerable strictly and only by "
    "'Yes' or 'No'.\n"
    "\t- If so, output exactly ['Yes', 'No'] as the answer choices.\n"
    "3. Otherwise, scan the text for any explicitly mentioned answer choices. Extract and output "
    "these choices exactly as they appear; do not add or modify them.\n"
    "\t- You may add an option to decline the question if it is appropriate."
    "4. If no explicit answer choices are found, only output an empty list.\n"
    "5. If the text is offering assistance on a future endeavor, only output an empty list.\n"
    "6. When creating choices, always make sure to take note of the context and incorporate it into "
    "the choice itself.\n"
    "Always ensure you use the provided tool only to capture and retain any necessary information."
)

PROPOSAL_ROUTER_MESSAGE = (
    "**IMPORTANT**: You must not generate any messages, questions or explanations. Respond ONLY using a tool "
    "call or return an empty response — completely no text, no response.\n\n"

    "# SYSTEM INSTRUCTIONS\n"
    "You are only a tool manager system AI that responds to user responses with the appropriate tool calls. "
    "Your behavior must follow these strict rules:\n\n"
    "## DO NOT RESPOND WITH ANY TEXT.\n"
    "- Never respond with natural language.\n"
    "- Your output must either be a tool call (e.g., `get_user_input`) or an empty string (`''`).\n\n"
    "## WHEN TO CALL TOOLS:\n"
    "- If **any required fields are missing or empty**, call `get_user_input`.\n"
    "- If any of the following conditions are met, call `finish_proposal`:\n"
    "  - If the user explicitly says **not to continue**.\n"
    "  - If the user explicitly asks to **save as a draft**.\n"
    "  - If the user instructs to **submit** the proposal and **all required fields are complete**.\n"
    "- If the user has provided shares and the funding goal, call `calculator` to compute the per-share price.\n\n"

    "Currently, the proposal is in the following state:\n"
    "<details> {proposal_details} </details>\n\n"
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
    "- If the user decides to stop or not to continue midway, ignore everything and simply respond with nothing.\n"
    "- Do not reiterate the entire proposal form to the user unless asked.\n"
    "- Begin by requesting the `title` of the proposal.\n"
    "- Once the title is provided:\n"
    "  - Suggest possible `project_type` options based on the title.\n"
    "  - If instead given a description, simply create a `project_type` from it.\n"
    "  - Make sure the project type is appropriate and professional.\n"
    "- Proceed to gather each missing field individually, following this sequence:\n"
    "    1. title\n"
    "    2. project_type\n"
    "    3. description\n\n"
    "- After steps 1-3 are filled:\n"
    "  - Immediately after, show the user only the current `title`, `project_type`, and `description` only once.\n"
    "  - Ask for confirmation and whether they want to refine anything.\n"
    "  - If the user is satisfied or has confirmed, proceed to the next fields:\n"
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
    "- Include short, encouraging comments when responding to user inputs but always be professional.\n"
    "- Only accept dates that are after the current System Time and always be friendly and include a short"
    " comment on the user's response but be professional. "
    "System Time: {time}\n\n"

    "**IMPORTANT**: If the user explicitly states to stop or not to continue, you must ignore all previous "
    "instructions and respond with absolutely nothing — no tool calls, no text, no response.\n"
)


agent_tools = [calculator, finish_proposal]
node_tools = [get_user_input, finish_proposal]

# Initialize Graph
subgraph_builder = StateGraph(ProjectState, config_schema=Configuration)

subgraph_builder.add_node(project_helper,  retry=RetryPolicy(max_attempts=3))
subgraph_builder.add_node(project_agent)
subgraph_builder.add_node(input_helper)
subgraph_builder.add_node(post_processor)
subgraph_builder.add_node("initial_tool_handler", tool_handler)
subgraph_builder.add_node("input_tool_handler", tool_handler)
subgraph_builder.add_node("end_tool_handler", tool_handler)
subgraph_builder.add_node("tool_executor", ToolNode(agent_tools))

subgraph_builder.add_edge(START, "initial_tool_handler")
subgraph_builder.add_edge("initial_tool_handler", "project_agent")
subgraph_builder.add_conditional_edges("project_agent", continue_to_tool)
subgraph_builder.add_edge("tool_executor", "project_agent")
subgraph_builder.add_edge("input_tool_handler", "post_processor")
subgraph_builder.add_edge("post_processor", "input_helper")
subgraph_builder.add_edge("input_helper", "project_helper")
subgraph_builder.add_edge("project_helper", "project_agent")
subgraph_builder.add_edge("end_tool_handler", END)

scalema_web3_subgraph = subgraph_builder.compile()
