import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

import configuration
from api_caller import fetch_task_counts, fetch_shift_logs

## Utilities 

# Inspect the tool calls for Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g. "Profile")
    """
    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Check if there are any patches
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    # Handle case where no changes were needed
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)

## Schema definitions

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)

## Initialize the model and tools

# Update memory tool
class ChooseTask(TypedDict):
    """ Decision on what tool to use """
    update_type: Literal['user', 'fetch_task_count', 'create_shift_summary']

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

## Create the Trustcall extractors for updating the user profile and ToDo list
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

## Prompts

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot.

You have a long term memory which keeps track of:
1. The user's profile (general information about them)

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling ChooseTask tool with type `user`
- If the user is asking for the amount of tasks they have, provide information by calling ChooseTask tool with type `fetch_task_count`
- If the user wants a summary of their tasks for the day, provide it by calling ChooseTask tool with type `create_shift_summary`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile

4. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""


## Node definitions

def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

   # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile)

    # Respond using memory as well as the chat history
    response = model.bind_tools([ChooseTask], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def fetch_task_count(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Returns the current number of tasks assigned to the user"""

    workforce_id = config["configurable"]["workforce_id"]
    auth_token = config["configurable"]["auth_token"]

    task_count = fetch_task_counts(auth_token, workforce_id)

    response = """The server returned:
    {task_number}
    Relay this information to the user""".format(task_number=task_count)

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": response, "tool_call_id":tool_calls[0]['id']}]}

def create_shift_summary(state:MessagesState, config: RunnableConfig, store: BaseStore):

    """Returns a short summary of the tasks worked on the current shift"""

    auth_token = config["configurable"]["auth_token"]
    employment_id = config["configurable"]["employment_id"]
    shift_start = config["configurable"]["shift_start"]

    shifts = fetch_shift_logs(auth_token, employment_id, shift_start)

    response = """Below are the titles of the tasks that the user has done:
    {shifts}
    Return the exact tasks that the user has done with: \"Today, you did \".
    Provide a short precise summary for the tasks done afterward.""".format(shifts=shifts)

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": response, "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[ "update_profile", "fetch_task_count", "create_shift_summary", END]:

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "fetch_task_count":
            return "fetch_task_count"
        elif tool_call['args']['update_type'] == "create_shift_summary":
            return "create_shift_summary"
        else:
            raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(task_mAIstro)
builder.add_node(update_profile)
builder.add_node(fetch_task_count)
builder.add_node(create_shift_summary)

# Define the flow 
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("fetch_task_count", "task_mAIstro")
builder.add_edge("create_shift_summary", "task_mAIstro")

# Compile the graph
graph = builder.compile()