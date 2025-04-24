from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


models = {
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "tool-calling-model": ChatOpenAI(model="gpt-4o", temperature=0, tool_choice="required"),
}


class SilentHandler(BaseCallbackHandler):
    """A callback handler that does nothing. Prevents models from streaming tokens."""
    def on_llm_new_token(self, token, **kwargs): pass
