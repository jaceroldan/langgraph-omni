from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from langgraph.types import interrupt

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

import configuration


class HumanQuery(BaseModel):  # Used to structure data

    """
    This contains the arguments of the User's query that you are chatting with.
    Always call this tool whenever the User asks anything.
    """

    query: str = Field(description="This contains the user's query")


def should_continue(state: MessagesState):

    """Decide whether to continue or not"""

    messages = state["messages"]

    tool_calls = messages[-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END

    if tool_calls[0]["name"] == "HumanQuery":
        return "input_node"
    else:
        return END


def input_node(state: MessagesState):

    """Facilitates HITL and assists in collecting user input"""

    tool_call = state["messages"][-1].tool_calls[0]
    prompt = tool_call["args"]["query"]
    formatted_question = (
        "You asked: {question}, provide an answer for this."
    )

    user_input = interrupt(formatted_question.format(question=prompt))
    content_message = (
        "The user responded with: "
        "<input>{input}</input>"
        "\nReview the user's input in relation to the question. Correct it if "
        "is wrong and provide an explanation."
    )  # This is temporary until a use case is created

    tool_message = [{
        "tool_call_id": tool_call["id"],
        "type": "tool",
        "content": content_message.format(input=user_input)
    }]
    return {"messages": tool_message}


def agent(state: MessagesState):

    """Helps personalizes chatbot messages"""

    response = model.invoke([SystemMessage(content=MODEL_SYSTEM_MESSAGE)] + state["messages"])
    return {"messages": response}


# System Messages for the Model
MODEL_SYSTEM_MESSAGE = (
    "You are Scalema, a helpful chatbot that helps clients with their business queries. "
    "If it's your first time talking with a client, be sure to inform them this."
)


# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools([HumanQuery])

# Build the graph
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

builder.add_node(agent)
builder.add_node(input_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("input_node", "agent")

# Compile the graph
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
