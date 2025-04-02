from typing import List
import uuid

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

from utils.configuration import Configuration, RunnableConfig


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
