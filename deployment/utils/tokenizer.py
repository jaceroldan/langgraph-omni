import tiktoken


def get_tokenizer(model_name: str = "gpt-4o"):
    """
        Returns a tokenizer.
    """
    return tiktoken.encoding_for_model(model_name)
