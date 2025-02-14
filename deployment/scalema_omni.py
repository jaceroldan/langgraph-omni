from datetime import datetime
from typing import Literal, Optional, TypedDict, get_origin, get_args
from pydantic import BaseModel, Field
from trustcall import create_extractor

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import SystemMessage, AIMessage


import configuration
import api_caller


# SYSTEM MESSAGES
ASSISTANT_MODEL_MESSAGE = """You are a helpful chatbot.

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide which action is being asked and respond accordingly:
- if the user wants to create anything, call the `ChooseTask` tool with type `input_helper`

3. Err on the side of calling tools. No need to ask for explicit permission.

4. Respond naturally to user as if no tool call was made."""




RESPOND_MESSAGE = """Directly respond to the user with the following message: {msg}"""


class GraphState(MessagesState):
    meta: dict
class ContinueState(GraphState):
    to_continue: bool



class ChooseTask(TypedDict):
    """ Decision on what tool to use """
    task_type: Literal['input_helper']

class Form(BaseModel):
    validated: bool = Field(description="Checks whether the card is acceptable or not.")

class CardForm(Form):
    title: str = Field(description="The card's title"),
    assignee: Optional[str] = Field(description="Person assigned to complete the task")
    is_public: Optional[bool] = Field(description="Visibility of the card to other users")

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Nodes
def assistant(state: MessagesState) -> MessagesState:

    """Intermediary node to personalize the chatbot's responses"""

    system_msg = ASSISTANT_MODEL_MESSAGE

    response = model.bind_tools([ChooseTask], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+ state["messages"])
    return {"messages": [response]}

def input_helper(state: MessagesState) -> ContinueState:

    """Ensures that the data about to be sent is formatted correctly."""

    INPUT_HELPER_MESSAGE = """Make sure the data is provided by the user, if not ask them for the data and assign `validated` as false.
    If the data is Optional, it can be ignored. Do not provide default data for the user.

    In this scenario, a Card is a task and it needs the following data:
    - title
    - assignee (Optional)
    - is_public (Optional)

    Based on the data, determine if the data is complete by assigning `validated` a value.
    If all Non-Optional data is provided, assign `validated` as true.

    System Time: {time}"""

    card_data_extractor = create_extractor(
        model.bind_tools([], parallel_tool_calls=False),
        tools=[CardForm],
        tool_choice="CardForm"
    )

    system_msg = INPUT_HELPER_MESSAGE.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs([SystemMessage(content=system_msg)] + state["messages"][:-1]))

    result = card_data_extractor.invoke({"messages": updated_messages})
    args = result["messages"][-1].tool_calls[0]["args"]
    print("\n\nargs: ", str(args))

    form = {}
    to_continue = False
    if args:
        to_continue = args["validated"]
        form = {
            "title": args["title"],
            "assignee": args["assignee"],
            "is_public": args["is_public"]
        }

    tool_calls = state['messages'][-1].tool_calls

    message_content = "" if to_continue else "card creation was incomplete, inform the user about this."
    message = {
        "role": "tool",
        "content": message_content,
        "tool_call_id":tool_calls[0]['id']
    }


    return {"messages":[message], "to_continue": to_continue, "meta": {"form":form}}

def create_card(state: GraphState, config: RunnableConfig) -> MessagesState:

    """Only accessed when creating a card"""

    auth_token = config["configurable"]["auth_token"]
    user_profile_id = config["configurable"]["user_profile_id"]

    args = state["meta"]["form"]

    # CardForm
    data = {
        "user_profile_id": user_profile_id,
        "title": args["title"],
        "assignee": args["assignee"],
        "is_public": args["is_public"]
    }


    response = api_caller.create_card(auth_token, data)

    tool_calls = state['messages'][-2].tool_calls

    CREATE_CARD_MESSAGE = """The server returned the following: {response}
    Briefly inform the user about the creation considering the received data.
    """

    return {"messages": SystemMessage(content=CREATE_CARD_MESSAGE.format(response=response), tool_call_id=tool_calls[0]['id'])}

# Routers
def route_message(state: MessagesState) -> Literal[ "input_helper", END]:
    """Decide which appropriate tool to call using the user's prompt."""

    message = state['messages'][-1]

    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['task_type'] == "input_helper":
            return "input_helper"
        raise ValueError

def continue_to_create_card(state: ContinueState) -> Literal["assistant", "create_card"]:
    """ Decides whether to continue or not based on the data provided """

    if state["to_continue"] == True:
        return "create_card"
    return "assistant"

# Build the graph
builder = StateGraph(GraphState, config_schema=configuration.Configuration)

builder.add_node(assistant)
builder.add_node(input_helper)
builder.add_node(create_card)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_message)
builder.add_conditional_edges("input_helper", continue_to_create_card)
builder.add_edge("create_card", "assistant")

# Compile the graph
graph = builder.compile()
