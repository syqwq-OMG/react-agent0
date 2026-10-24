def build_initial_messages(
    question: str,
    tool_descriptions: str,
    max_steps: int,
    max_tool_calls: int,
) -> list[dict]:
    system_content = f"""
You are a retrieval agent.

Respond with exactly one JSON object and no Markdown.

Tool call format:
{{"type":"tool_call","tool":"search","args":{{"query":"...","k":3}}}}

Final answer format:
{{"type":"final","answer":"...","evidence":[["title",0]]}}

Available tools:
{tool_descriptions}

Maximum model calls: {max_steps}
Maximum tool calls: {max_tool_calls}
"""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def append_observation(
    messages: list[dict],
    raw_action: str,
    observation: str,
) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": raw_action},
        {"role": "user", "content": f"Observation: {observation}"},
    ]
