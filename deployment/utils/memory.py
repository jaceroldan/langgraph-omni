from typing import List, Optional
import uuid
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_postgres import PGEngine, PGVectorStore
from langgraph.graph import MessagesState
from langchain_core.messages import get_buffer_string, RemoveMessage, SystemMessage
from trustcall import create_extractor

from utils.configuration import Configuration, RunnableConfig
from utils.tokenizer import get_tokenizer
from utils.models import models

# import settings
import settings


class MemoryState(MessagesState):
    memories: List[str]


class Memory(BaseModel):
    memory: Optional[str] = Field(default=None, description="Memory to be stored in the vectorstore.")


def memory_node(state: MemoryState, config: RunnableConfig) -> MemoryState:
    """
        Processes previous messages and optimizes them to reduce token usage. Also
        summarizes the messages and appends them to the thread memory.
    """

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    node_model = models[model_name]

    messages = state["messages"][:-settings.MODEL_HISTORY_LENGTH]
    memories = state["memories"]

    # Commented out for future summarization
    memory_extractor = create_extractor(
        node_model,
        tools=[Memory],
        tool_choice="Memory"
    )
    result = memory_extractor.invoke(
        {"messages": [SystemMessage(content=SUMMARY_MESSAGE.format(memories=memories))] + messages})
    extracted = result['responses'][-1].memory

    # Delete all previous messages since action has already been summarized
    removed_messages = [RemoveMessage(id=m.id) for m in messages]
    saved_memores = save_recall_memory.invoke(extracted, config)

    return {
        "messages": removed_messages,
        "memories": saved_memores
    }


def load_memory(state: MemoryState, config: RunnableConfig) -> MemoryState:
    """
        Loads memories for the current conversation. Only loads memories when
        the `memories` in the state is empty.
    """

    memories = state.get("memories", [])

    if memories:
        return {"memories": memories}

    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.model_name
    tokenizer = get_tokenizer(model_name)

    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "memories": recall_memories
    }


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """
        Save memory to vectorstore for later semantic retrieval.
    """

    configuration = Configuration.from_runnable_config(config)
    user_profile_pk = configuration.user_profile_pk
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_profile_pk": user_profile_pk}
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """
        Search for relevant memories.
    """
    configuration = Configuration.from_runnable_config(config)
    user_profile_pk = configuration.user_profile_pk

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_profile_pk") == user_profile_pk

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]


recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings())

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Strings
SUMMARY_MESSAGE = (
    "# SYSTEM INSTRUCTIONS\n"
    "Your only task is to create a short but comprehensive summary of the previous conversations "
    "for you to remember later. If there are no conversations, then return an empty array.\n"
    "Always make sure to include the following in the memory sentence:\n"
    "\t- what the User asked.\n"
    "\t- what the outcome was of the conversation.\n"
    "\t- if there were any other notable interactions, include it.\n"
    "Below is the current list of memories:\n"
    "<memories> {memories} </memories>\n"
    "Below is the conversation history:\n"
)
