import inspect
import json
from .retrieval import run_search,run_lookup

def execute_tool(action: dict, tools: dict, max_observation_chars: int = 8000) -> dict:
    tool_name = action.get("tool")
    args = action.get("args")
    if tool_name not in tools:
        return {
            "ok": False,
            "observation": f"ERROR: tool {tool_name} not found",
        }

    func = tools[tool_name]
    
    try:
        inspect.signature(func).bind(**args)
    except TypeError as exc:
        return {
            "ok": False,
            "observation": f"ERROR: invalid arguments: {exc}",
        }
        
    try:
        res=func(**args)
        if isinstance(res, str):
            observation = res
        else:
            observation=json.dumps(res, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return {
            "ok": False,
            "observation": f"ERROR: tool failed: {exc}",
        }

    if len(observation) > max_observation_chars:
        marker = "ERROR: output truncated: "
        bounded = (marker + observation)[:max_observation_chars]

        return {
            "ok": False,
            "observation": bounded,
        }

    return {
        "ok": True,
        "observation": observation,
    }
    
def make_tools(context:list)->dict:
    def search(query:str, k:int=3):
        return run_search(context, query, k)
    def lookup(title:str):
        return run_lookup(context, title)
    return {
        "search": search,
        "lookup": lookup,
    }