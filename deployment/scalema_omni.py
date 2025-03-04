from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langgraph.types import interrupt, Command

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import configuration
import api_caller


# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# EXAMPLES
# def assistant(state: MessagesState, configuration=RunnableConfig) -> MessagesState:
#     return {"messages": [AIMessage(content="This is an AI message")] + state["messages"]}
# def human_node(state: MessagesState, configuration=RunnableConfig) -> MessagesState:
#     user_input = interrupt(value="Ready for user input.")
#     return Command(goto="end_node", update={"messages": [HumanMessage(content=user_input)]})
# def end_node(state: MessagesState, configuration=RunnableConfig) -> MessagesState:
#     last_message = state["messages"][-1].content
#     return {"messages": [AIMessage(content=f"Goodbye! You said: {last_message}")]}


# TOOLS
@tool
def call_api(query: str):
    """Calls an api"""
    return f"I called {query}. Result: Say hello to the user."


tools = [call_api]
tool_node = ToolNode(tools)


class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str


model = model.bind_tools(tools + [AskHuman])


def should_continue(state):
    """Decides whether to continue or not"""
    messages = state["messages"]

    tool_calls = messages[-1].tool_calls
    # If there is no function call, then finish
    if not tool_calls:
        return END
    elif tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    else:
        return "action"


def ask_human(state):
    """Internal node to ask the user"""

    tool_call = state["messages"][-1].tool_calls[0]
    question = tool_call["args"]["question"]

    user_input = interrupt(question)
    content_message = """The user responded with:
    {input}
    Respond only with a rating of this answer from 1 to 10."""

    tool_message = [{
        "tool_call_id": tool_call["id"],
        "type": "tool",
        "content": content_message.format(input=user_input)
    }]
    return {"messages": tool_message}


def agent(state):
    """Node to call the model"""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Build the graph
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

builder.add_node(agent)
builder.add_node("action", tool_node)
builder.add_node(ask_human)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("action", "agent")
builder.add_edge("ask_human", "agent")

# Compile the graph
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
