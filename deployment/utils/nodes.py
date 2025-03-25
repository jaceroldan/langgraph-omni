from schemas import ProjectState


def exit_handler(state: ProjectState):
    """
        Fake node to used for exiting the subgraph.
        Only use this node whenever there is a tool call without a tool message.
    """
    # Confirm the tool call made in the parent graph
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    tool_call_message = {"content": "", "role": "tool", "tool_call_id": tool_call_id}

    return {**state,  "messages": [tool_call_message]}
