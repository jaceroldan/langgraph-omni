from langchain_openai import ChatOpenAI

models = {
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0)
}
