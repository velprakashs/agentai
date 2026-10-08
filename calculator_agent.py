"""
Agentic AI Assistant following the ReAct (Reasoning + Acting) Pattern.
Equipped with 4 Tools: Calculator, Web Search, Date/Time, and Document Search.
Built without heavy agent frameworks to demonstrate pure Tool Calling & ReAct loops.
"""

import os
import sys
import json
import math
import html
import re
import time
import csv
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure Windows terminal outputs UTF-8 cleanly without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
load_dotenv()


# ==============================================================================
# 1. TOOL IMPLEMENTATIONS (Python Functions)
# ==============================================================================

def execute_calculator(operation: str, a: float, b: float = 0.0, **kwargs) -> Dict[str, Any]:
    """
    Executes a basic mathematical operation and returns the result in JSON format.
    """
    op = operation.lower().strip()
    try:
        if op in ["add", "+", "addition"]:
            res = a + b
            return {"status": "success", "operation": "add", "a": a, "b": b, "result": res}
        elif op in ["subtract", "-", "subtraction"]:
            res = a - b
            return {"status": "success", "operation": "subtract", "a": a, "b": b, "result": res}
        elif op in ["multiply", "*", "multiplication"]:
            res = a * b
            return {"status": "success", "operation": "multiply", "a": a, "b": b, "result": res}
        elif op in ["divide", "/", "division"]:
            if b == 0:
                return {"status": "error", "error_message": "Division by zero is not allowed.", "a": a, "b": b}
            res = a / b
            return {"status": "success", "operation": "divide", "a": a, "b": b, "result": res}
        elif op in ["power", "^", "pow", "exponentiation"]:
            res = math.pow(a, b)
            return {"status": "success", "operation": "power", "base": a, "exponent": b, "result": res}
        elif op in ["modulus", "%", "mod", "remainder"]:
            if b == 0:
                return {"status": "error", "error_message": "Modulus by zero is not allowed.", "a": a, "b": b}
            res = a % b
            return {"status": "success", "operation": "modulus", "a": a, "b": b, "result": res}
        elif op in ["square_root", "sqrt", "root"]:
            if a < 0:
                return {"status": "error", "error_message": "Square root of a negative number is not a real number.", "a": a}
            res = math.sqrt(a)
            return {"status": "success", "operation": "square_root", "a": a, "result": res}
        else:
            return {
                "status": "error",
                "error_message": f"Unsupported operation '{operation}'. Supported operations: add, subtract, multiply, divide, power, modulus, square_root."
            }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def execute_web_search(query: str, max_results: int = 3, **kwargs) -> Dict[str, Any]:
    """
    Performs a web search to fetch relevant facts, summaries, and information.
    Uses Wikipedia Search and DuckDuckGo without requiring extra paid keys.
    """
    query_clean = query.strip()
    results = []

    try:
        wiki_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query_clean,
            "utf8": "",
            "format": "json"
        }
        headers = {"User-Agent": "AgenticAssistant/1.0"}
        resp = requests.get(wiki_url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            search_items = data.get("query", {}).get("search", [])
            for item in search_items[:max_results]:
                raw_snippet = item.get("snippet", "")
                clean_snippet = html.unescape(re.sub(r"<.*?>", "", raw_snippet))
                results.append({
                    "title": item.get("title", ""),
                    "snippet": clean_snippet,
                    "source": f"https://en.wikipedia.org/wiki/{item.get('title', '')}"
                })
    except Exception:
        pass

    if len(results) < max_results:
        try:
            ddg_url = "https://api.duckduckgo.com/"
            ddg_params = {
                "q": query_clean,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            ddg_resp = requests.get(ddg_url, params=ddg_params, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if ddg_resp.status_code == 200:
                data = ddg_resp.json()
                if data.get("AbstractText"):
                    results.insert(0, {
                        "title": data.get("Heading", "Instant Answer"),
                        "snippet": data.get("AbstractText"),
                        "source": data.get("AbstractURL", "DuckDuckGo")
                    })
        except Exception:
            pass

    if not results:
        return {
            "status": "not_found",
            "query": query_clean,
            "message": f"No web search results found for query: '{query_clean}'."
        }

    return {
        "status": "success",
        "query": query_clean,
        "results": results[:max_results]
    }


def execute_date_time(*args, **kwargs) -> Dict[str, Any]:
    """
    Returns current date, time, day of the week, year, and timezone in JSON format.
    """
    try:
        tz_val = kwargs.get("tz") or kwargs.get("timezone") or (args[0] if args else "local")
        tz_val = str(tz_val).strip()

        if tz_val.upper() == "UTC":
            now = datetime.now(timezone.utc)
            tz_str = "UTC"
        else:
            now = datetime.now()
            tz_str = "Local"

        return {
            "status": "success",
            "formatted_datetime": now.strftime("%A, %B %d, %Y %I:%M:%S %p"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "timezone": tz_str,
            "iso_timestamp": now.isoformat()
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def execute_document_search(query: str = "", **kwargs) -> Dict[str, Any]:
    """
    Searches the internal Student Database document (students.csv) for student records,
    marks, attendance, departments, and contact details.
    """
    csv_path = "students.csv"
    if not os.path.exists(csv_path):
        return {"status": "error", "error_message": f"Document file '{csv_path}' not found."}

    records = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    q = str(query).lower().strip()
    
    # If query is broad / aggregation, return all records
    broad_keywords = ["all", "highest", "lowest", "how many", "list", "who scored", "database", "students", "overall", "attendance", "below", "above", ""]
    if not q or any(k in q for k in broad_keywords):
        return {
            "status": "success",
            "query": query,
            "total_records_found": len(records),
            "records": records
        }

    # Filter by query terms
    matched = []
    for r in records:
        text = " ".join(r.values()).lower()
        if any(term in text for term in q.split() if len(term) > 1):
            matched.append(r)

    return {
        "status": "success",
        "query": query,
        "total_records_found": len(matched),
        "records": matched if matched else records
    }


# Tool Dispatcher mapping tool names to actual Python functions
TOOL_REGISTRY = {
    "calculator": execute_calculator,
    "web_search": execute_web_search,
    "date_time": execute_date_time,
    "get_datetime": execute_date_time,
    "get_current_datetime": execute_date_time,
    "current_time": execute_date_time,
    "document_search": execute_document_search,
    "search_student_database": execute_document_search
}


# ==============================================================================
# 2. TOOL DECLARATIONS (JSON Schema format for the LLM)
# ==============================================================================

CALCULATOR_TOOL_DECLARATION = types.FunctionDeclaration(
    name="calculator",
    description="Performs basic arithmetic calculations: addition, subtraction, multiplication, division, power, modulus, and square root.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "operation": types.Schema(
                type=types.Type.STRING,
                description="The arithmetic operation to perform: 'add', 'subtract', 'multiply', 'divide', 'power', 'modulus', or 'square_root'.",
                enum=["add", "subtract", "multiply", "divide", "power", "modulus", "square_root"]
            ),
            "a": types.Schema(
                type=types.Type.NUMBER,
                description="The first operand (or the number for square_root)."
            ),
            "b": types.Schema(
                type=types.Type.NUMBER,
                description="The second operand (optional for square_root, required for other operations)."
            )
        },
        required=["operation", "a"]
    )
)

WEB_SEARCH_TOOL_DECLARATION = types.FunctionDeclaration(
    name="web_search",
    description="Searches the web for up-to-date facts, figures, statistics, data, and general knowledge.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="The search query string (e.g., 'population of Tokyo', 'height of Mount Everest in meters')."
            ),
            "max_results": types.Schema(
                type=types.Type.INTEGER,
                description="Maximum number of search results to retrieve (default is 3)."
            )
        },
        required=["query"]
    )
)

DATETIME_TOOL_DECLARATION = types.FunctionDeclaration(
    name="date_time",
    description="Gets the current date, current time, day of the week, month, year, and timezone from the system clock. Call this whenever the user asks what today is, what the time is, or for any time/date-sensitive queries.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "tz": types.Schema(
                type=types.Type.STRING,
                description="Timezone: 'local' (default) or 'UTC'.",
                enum=["local", "UTC"]
            )
        }
    )
)

DOCUMENT_SEARCH_TOOL_DECLARATION = types.FunctionDeclaration(
    name="document_search",
    description="Searches the internal Student Database document (students.csv) for student academic records, marks (Python, DSA, AI, Overall), attendance percentages, departments (CSE, IT, AIDS, ECE, EEE), placement status, phone numbers, and emails.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search query: student name (e.g., 'Priya Sharma'), student ID (e.g., 'STU001'), department, placement, or question criteria."
            )
        },
        required=["query"]
    )
)

AGENT_TOOLS = [
    types.Tool(
        function_declarations=[
            CALCULATOR_TOOL_DECLARATION,
            WEB_SEARCH_TOOL_DECLARATION,
            DATETIME_TOOL_DECLARATION,
            DOCUMENT_SEARCH_TOOL_DECLARATION
        ]
    )
]


# ==============================================================================
# 3. REACT (REASONING + ACTING) AGENTIC LOOP
# ==============================================================================

FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]


class ReActAgent:
    """
    Implements the ReAct (Reasoning + Acting) loop:
    1. Thought: Reason explicitly about the next step needed.
    2. Action: Invoke a registered tool with structured JSON arguments.
    3. Observation: Receive the JSON observation from the tool execution.
    4. Repeat until reaching the final answer.
    """

    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        self.client = genai.Client(api_key=api_key)
        self.system_instruction = (
            "You are an AI agent following the ReAct (Reasoning + Acting) architecture.\n"
            "You have access to 4 powerful tools:\n"
            "1. 'document_search': MUST be used to look up student records, marks (Python, DSA, AI, Overall), attendance, departments, emails, and phone numbers from the internal student database document (students.csv).\n"
            "2. 'calculator': MUST be used for any arithmetic calculations, averages, totals, percentages, or differences.\n"
            "3. 'web_search': MUST be used to lookup real-time web facts, news, and world entities.\n"
            "4. 'date_time': MUST be used to check the current date, time, year, day of the week, and timezone.\n\n"
            "ReAct Instructions:\n"
            "- In every step where you need to take an action, first write a concise 'Thought:' explaining your reasoning (what you know, what is missing, and why you are calling a specific tool).\n"
            "- Then execute the 'Action' by invoking the tool with JSON arguments.\n"
            "- You will receive an 'Observation' with the tool's JSON output.\n"
            "- You can chain tools (e.g. use 'document_search' to get marks, then 'calculator' to compute averages).\n"
            "- Finally, formulate your verified final answer clearly."
        )

    def _generate_with_retry(self, contents: list, max_retries: int = 5):
        """
        Calls Gemini API with automatic model fallback on 429 quota limits and retry on 503 errors.
        """
        delays = [2, 4, 8, 12, 16]
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        tools=AGENT_TOOLS,
                        temperature=0.0
                    )
                )
            except Exception as e:
                err_str = str(e)
                # Check for 429 Quota exhaustion -> Failover to another active model
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    for fallback in FALLBACK_MODELS:
                        if fallback != self.model_name:
                            print(f"\n[Quota exceeded on {self.model_name}. Automatically switching to fallback model '{fallback}'...]")
                            self.model_name = fallback
                            break
                    time.sleep(1)
                    continue

                is_transient = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str
                if is_transient and attempt < max_retries - 1:
                    wait_time = delays[min(attempt, len(delays) - 1)]
                    print(f"\n[Model experiencing high traffic (503). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})]")
                    time.sleep(wait_time)
                else:
                    raise e

    def run(self, user_query: str, verbose: bool = True) -> str:
        """
        Executes the ReAct Loop (Thought -> Action -> Observation -> Final Answer).
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_query)]
            )
        ]

        max_turns = 10
        current_turn = 0

        if verbose:
            print("\n" + "-" * 60)
            print(f"[User Goal]: {user_query}")
            print("-" * 60)

        while current_turn < max_turns:
            current_turn += 1

            # Request LLM response with tool declarations (with automatic retry)
            response = self._generate_with_retry(contents)

            candidate = response.candidates[0]
            model_parts = candidate.content.parts

            # Add model response to conversation history
            contents.append(candidate.content)

            # Separate text (Thought) from function calls (Action)
            thought_text = "".join([part.text for part in model_parts if part.text is not None]).strip()
            function_calls = [part.function_call for part in model_parts if part.function_call is not None]

            # If no tool calls were requested, the model provided its final answer
            if not function_calls:
                if verbose:
                    print(f"\n[ReAct Complete] Finished in {current_turn} turn(s).")
                return thought_text

            # Execute the ReAct Action(s)
            response_parts = []
            for call in function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}

                # Display the ReAct Step
                if verbose:
                    print(f"\n[Turn {current_turn}]")
                    if thought_text:
                        print(f"Thought: {thought_text}")
                    else:
                        print(f"Thought: Calling tool '{tool_name}' with arguments {json.dumps(tool_args)}")

                    print(f"Action: Call tool `{tool_name}`")
                    print(f"   Payload (JSON): {json.dumps(tool_args, indent=2)}")

                # Dispatch tool
                if tool_name in TOOL_REGISTRY:
                    tool_fn = TOOL_REGISTRY[tool_name]
                    tool_result = tool_fn(**tool_args)
                else:
                    tool_result = {"status": "error", "error_message": f"Unknown tool '{tool_name}'"}

                if verbose:
                    # Truncate preview if records are large
                    preview = json.dumps(tool_result, indent=2)
                    if len(preview) > 600:
                        preview = preview[:600] + "\n   ... [truncated for display] ..."
                    print(f"Observation:")
                    print(f"   Result (JSON): {preview}")

                # Create function response part (Observation fed back to LLM)
                response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result}
                    )
                )

            # Append Observations to conversation history and continue loop
            contents.append(
                types.Content(
                    role="user",
                    parts=response_parts
                )
            )

        return "Error: Agent reached maximum reasoning turns without completing the answer."


# ==============================================================================
# 4. INTERACTIVE CLI INTERFACE
# ==============================================================================

def print_banner():
    print("=" * 75)
    print("      ReAct AGENTIC AI (Calculator + Web Search + Date/Time + Document)")
    print("=" * 75)
    print(" Pattern:")
    print("  Thought     -> Agent reasons about what information is needed")
    print("  Action      -> Agent invokes a tool with JSON arguments")
    print("  Observation -> Tool executes and returns JSON result")
    print("  Answer      -> Agent synthesizes final response")
    print("-" * 75)
    print(" Available Tools:")
    print("  1. 'document_search' -> Student Database records, marks, attendance, etc.")
    print("  2. 'calculator'      -> Arithmetic operations (add, multiply, sqrt, etc.)")
    print("  3. 'web_search'      -> Real-time web knowledge and data")
    print("  4. 'date_time'       -> Real-time system date, time, year, and timezone")
    print("-" * 75)
    print(" Commands: 'exit' to quit | 'schema' for tool JSON | 'clear' to clear")
    print("=" * 75 + "\n")


def print_tool_schema():
    schemas = [
        {
            "name": DOCUMENT_SEARCH_TOOL_DECLARATION.name,
            "description": DOCUMENT_SEARCH_TOOL_DECLARATION.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "The search query, student name, ID, or criteria"}
                },
                "required": ["query"]
            }
        },
        {
            "name": CALCULATOR_TOOL_DECLARATION.name,
            "description": CALCULATOR_TOOL_DECLARATION.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "operation": {
                        "type": "STRING",
                        "description": "The arithmetic operation",
                        "enum": ["add", "subtract", "multiply", "divide", "power", "modulus", "square_root"]
                    },
                    "a": {"type": "NUMBER", "description": "First operand / number"},
                    "b": {"type": "NUMBER", "description": "Second operand (optional for square_root)"}
                },
                "required": ["operation", "a"]
            }
        },
        {
            "name": WEB_SEARCH_TOOL_DECLARATION.name,
            "description": WEB_SEARCH_TOOL_DECLARATION.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "The search query string"},
                    "max_results": {"type": "INTEGER", "description": "Maximum results (default 3)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": DATETIME_TOOL_DECLARATION.name,
            "description": DATETIME_TOOL_DECLARATION.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "tz": {
                        "type": "STRING",
                        "description": "Timezone: 'local' (default) or 'UTC'",
                        "enum": ["local", "UTC"]
                    }
                }
            }
        }
    ]
    print("\n--- Registered Tools (JSON Schema Format) ---")
    print(json.dumps(schemas, indent=2))
    print("-----------------------------------------------\n")


def main():
    print_banner()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[Notice] GEMINI_API_KEY is not set in your .env file.")
        api_key = input("Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("Error: API Key is required to run the agent. Exiting.")
            sys.exit(1)

    try:
        agent = ReActAgent(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        sys.exit(1)

    print("ReAct Agent initialized successfully with all 4 tools!\n")
    print("Example Queries:")
    print(" - 'What is Priya Sharma\\'s attendance and overall mark from the document?'")
    print(" - 'Find Arjun Kumar\\'s Python and DSA marks and calculate their sum.'")
    print(" - 'Who has the highest overall mark in the document?'")
    print(" - 'What is today\\'s date and what is sqrt(144)?'\n")

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "schema":
                print_tool_schema()
                continue
            elif user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue

            # Run the ReAct agent
            answer = agent.run(user_input, verbose=True)
            print(f"\nFinal Answer:\n{answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error during execution]: {e}")


if __name__ == "__main__":
    main()
