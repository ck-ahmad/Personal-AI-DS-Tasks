"""Week 2 Day 1 — Agent Foundations using Ollama + Qwen2.5.

Install:
    pip install ollama
    ollama pull qwen2.5:7b
Run:
    python agent_foundations_ollama.py
"""
import os, json, ast, operator
from ollama import chat

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
WORKING_MEMORY = []

TOOLS = [
    {"type":"function","function":{"name":"calculator","description":"Evaluate basic arithmetic using numbers, +, -, *, /, %, ** and parentheses. Use this for numeric calculations.","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Arithmetic expression, e.g. (3 + 4) * 2"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"get_weather","description":"Get demo weather for Lahore, Karachi, Faisalabad, Toronto, or London. Do not guess unsupported cities.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    {"type":"function","function":{"name":"read_text_file","description":"Read a small local .txt file when the user explicitly refers to it.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"record_finding","description":"Save a short factual finding in working memory for later use.","parameters":{"type":"object","properties":{"note":{"type":"string"}},"required":["note"]}}},
]

_WEATHER_DB = {"lahore":{"temp_c":34,"condition":"sunny"},"karachi":{"temp_c":30,"condition":"humid"},"faisalabad":{"temp_c":33,"condition":"hazy"},"toronto":{"temp_c":18,"condition":"cloudy"},"london":{"temp_c":16,"condition":"rainy"}}


def execute_tool(name, args):
    try:
        if name == "calculator":
            expr = args["expression"]
            allowed = set("0123456789+-*/%(). ")
            if not set(expr) <= allowed: return {"ok":False,"error":"Unsupported characters in expression."}
            tree = ast.parse(expr, mode="eval")
            ops = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul, ast.Div:operator.truediv, ast.Mod:operator.mod, ast.Pow:operator.pow, ast.USub:operator.neg, ast.UAdd:operator.pos}
            def calc(n):
                if isinstance(n, ast.Constant) and isinstance(n.value,(int,float)): return n.value
                if isinstance(n, ast.UnaryOp) and type(n.op) in ops: return ops[type(n.op)](calc(n.operand))
                if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](calc(n.left),calc(n.right))
                raise ValueError("Only basic arithmetic is supported")
            return {"ok":True,"result":calc(tree.body)}
        if name == "get_weather":
            city=args["city"].strip().lower()
            if city not in _WEATHER_DB: return {"ok":False,"error":f"No weather data for '{args['city']}'. Supported cities: {list(_WEATHER_DB)}"}
            return {"ok":True,"result":_WEATHER_DB[city]}
        if name == "read_text_file":
            path=args["path"]
            if not path.endswith(".txt"): return {"ok":False,"error":"Only .txt files are supported."}
            if not os.path.exists(path): return {"ok":False,"error":f"File not found: {path}"}
            with open(path,encoding="utf-8") as f: return {"ok":True,"result":f.read()}
        if name == "record_finding":
            WORKING_MEMORY.append(args["note"])
            return {"ok":True,"result":f"Recorded. Scratchpad now has {len(WORKING_MEMORY)} note(s)."}
        return {"ok":False,"error":f"Unknown tool: {name}"}
    except Exception as e:
        return {"ok":False,"error":str(e)}


def run_agent(user_task, max_iterations=6, verbose=True):
    WORKING_MEMORY.clear()
    messages=[{"role":"user","content":user_task}]
    for iteration in range(1,max_iterations+1):
        if verbose: print(f"\n--- Iteration {iteration} ---")
        response=chat(model=MODEL,messages=messages,tools=TOOLS)
        msg=response["message"]
        if verbose and msg.get("content"): print("[reason]",msg["content"])
        tool_calls=msg.get("tool_calls",[])
        if not tool_calls:
            answer=msg.get("content","").strip()
            if verbose: print("[final]",answer)
            return answer
        messages.append(msg)
        for call in tool_calls:
            name=call["function"]["name"]; args=call["function"].get("arguments",{})
            if verbose: print(f"[act] calling `{name}` with input {args}")
            result=execute_tool(name,args)
            if verbose: print("[observe]",result)
            messages.append({"role":"tool","content":json.dumps(result)})
    return f"[Agent stopped after {max_iterations} iterations. Working memory: {WORKING_MEMORY}]"


def demo_ambiguous_request(): return run_agent("What's the weather like today?")
def demo_unsupported_tool_input(): return run_agent("What's the weather in Islamabad?")
def demo_task_needing_undefined_tool(): return run_agent("Email the Faisalabad weather to my manager.")

if __name__ == "__main__":
    print("="*70); print("TASK 2 DEMO — single tool call"); print("="*70)
    print(run_agent("What's 12.5% of 480?"))
    print("\n"+"="*70); print("TASK 3 DEMO — multi-step agent loop"); print("="*70)
    print(run_agent("Look up the weather in Faisalabad and Toronto, record each finding, then tell me which city is warmer right now and by how much."))
    print("WORKING MEMORY AT END:",WORKING_MEMORY)
    print("\nTASK 5 DEMOS")
    print("Ambiguous:",demo_ambiguous_request())
    print("Unsupported city:",demo_unsupported_tool_input())
    print("Undefined tool:",demo_task_needing_undefined_tool())
