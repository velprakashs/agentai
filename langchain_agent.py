"""
LangChain Multi-Task Agent (Reasoning + Acting + Concurrent Execution)
Performs single tasks and multiple concurrent tasks in parallel using LangChain & Google Gemini.
"""

import os
import sys
import json
import math
import html
import re
import time
import asyncio
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Ensure Windows terminal outputs UTF-8 cleanly without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# Set up API keys for LangChain
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


# ==============================================================================
# 1. LANGCHAIN TOOLS
# ==============================================================================

@tool
def calculator(operation: str, a: float, b: float = 0.0) -> str:
    """Performs arithmetic calculations: add, subtract, multiply, divide, power, modulus, square_root."""
    op = operation.lower().strip()
    try:
        if op in ["add", "+", "addition"]:
            return json.dumps({"status": "success", "result": a + b})
        elif op in ["subtract", "-", "subtraction"]:
            return json.dumps({"status": "success", "result": a - b})
        elif op in ["multiply", "*", "multiplication"]:
            return json.dumps({"status": "success", "result": a * b})
        elif op in ["divide", "/", "division"]:
            if b == 0:
                return json.dumps({"status": "error", "error_message": "Division by zero is not allowed."})
            return json.dumps({"status": "success", "result": a / b})
        elif op in ["power", "^", "pow", "exponentiation"]:
            return json.dumps({"status": "success", "result": math.pow(a, b)})
        elif op in ["modulus", "%", "mod"]:
            if b == 0:
                return json.dumps({"status": "error", "error_message": "Modulus by zero is not allowed."})
            return json.dumps({"status": "success", "result": a % b})
        elif op in ["square_root", "sqrt", "root"]:
            if a < 0:
                return json.dumps({"status": "error", "error_message": "Square root of a negative number is undefined for real numbers."})
            return json.dumps({"status": "success", "result": math.sqrt(a)})
        else:
            return json.dumps({"status": "error", "error_message": f"Unsupported operation '{operation}'."})
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date facts, statistics, numbers, and general knowledge."""
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
        headers = {"User-Agent": "LangChainAgent/1.0"}
        resp = requests.get(wiki_url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("query", {}).get("search", [])[:3]:
                raw_snippet = item.get("snippet", "")
                clean_snippet = html.unescape(re.sub(r"<.*?>", "", raw_snippet))
                results.append({"title": item.get("title", ""), "snippet": clean_snippet})
    except Exception:
        pass

    if not results:
        return json.dumps({"status": "not_found", "message": f"No web search results for '{query_clean}'."})

    return json.dumps({"status": "success", "query": query_clean, "results": results})


@tool
def date_time(tz: str = "local") -> str:
    """Gets the current real-time system date, time, day of the week, year, and timezone."""
    try:
        if str(tz).strip().upper() == "UTC":
            now = datetime.now(timezone.utc)
            tz_str = "UTC"
        else:
            now = datetime.now()
            tz_str = "Local"

        return json.dumps({
            "status": "success",
            "formatted_datetime": now.strftime("%A, %B %d, %Y %I:%M:%S %p"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "year": now.year,
            "timezone": tz_str
        })
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})


@tool
def document_search(query: str = "") -> str:
    """Searches the internal Student Database document (students.csv) for student records, academic marks (Python, DSA, AI, Overall), attendance, departments, and placement status."""
    csv_path = "students.csv"
    if not os.path.exists(csv_path):
        return json.dumps({"status": "error", "error_message": f"Document file '{csv_path}' not found."})

    import csv
    records = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        return json.dumps({"status": "error", "error_message": str(e)})

    q = str(query).lower().strip()
    broad_keywords = ["all", "highest", "lowest", "how many", "list", "who scored", "database", "students", "overall", "attendance", "below", "above", ""]
    if not q or any(k in q for k in broad_keywords):
        return json.dumps({"status": "success", "total_records": len(records), "records": records})

    matched = []
    for r in records:
        text = " ".join(r.values()).lower()
        if any(term in text for term in q.split() if len(term) > 1):
            matched.append(r)

    return json.dumps({"status": "success", "total_records": len(matched), "records": matched if matched else records})


LANGCHAIN_TOOLS = [calculator, web_search, date_time, document_search]


# ==============================================================================
# 2. LANGCHAIN MULTI-TASK AGENT
# ==============================================================================

class LangChainMultiTaskAgent:
    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        
        # Initialize Gemini LLM with LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.0
        )
        
        # Create ReAct agent graph with tools
        self.agent = create_react_agent(
            model=self.llm,
            tools=LANGCHAIN_TOOLS,
            prompt=(
                "You are an expert AI assistant capable of multi-tasking. "
                "You have access to 3 tools: 'calculator', 'web_search', and 'date_time'. "
                "Always use tools when needed to verify calculations, retrieve live facts, or check the clock. "
                "Provide accurate, clear, and concise answers."
            )
        )

    def run_single_task(self, task: str, task_id: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes a single task with automatic error recovery and retry.
        """
        start_time = time.time()
        max_retries = 5
        delays = [2, 4, 8, 12, 16]

        for attempt in range(max_retries):
            try:
                if verbose:
                    print(f"\n⚡ [Task {task_id} Started]: '{task}'")
                
                inputs = {"messages": [HumanMessage(content=task)]}
                result = self.agent.invoke(inputs)
                
                # Extract final assistant message
                final_message = result["messages"][-1].content
                elapsed = round(time.time() - start_time, 2)

                if verbose:
                    print(f"✅ [Task {task_id} Completed in {elapsed}s]")

                return {
                    "task_id": task_id,
                    "task": task,
                    "status": "success",
                    "answer": final_message,
                    "elapsed_seconds": elapsed
                }
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                    wait_sec = delays[attempt]
                    if verbose:
                        print(f"⚠️ [Task {task_id}] High traffic. Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    elapsed = round(time.time() - start_time, 2)
                    return {
                        "task_id": task_id,
                        "task": task,
                        "status": "error",
                        "answer": f"Error during execution: {err_str}",
                        "elapsed_seconds": elapsed
                    }

    def run_multiple_tasks_parallel(self, tasks: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Executes multiple tasks concurrently in parallel using ThreadPoolExecutor.
        """
        print(f"\n🚀 Launching {len(tasks)} tasks concurrently in parallel using LangChain...\n" + "=" * 70)
        start_total = time.time()

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks at the same time
            futures = [
                executor.submit(self.run_single_task, task, idx + 1, True)
                for idx, task in enumerate(tasks)
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"status": "error", "answer": str(e), "elapsed_seconds": 0})

        total_elapsed = round(time.time() - start_total, 2)
        print("=" * 70)
        print(f"🎉 All {len(tasks)} tasks completed in parallel in {total_elapsed}s total!")
        return results


# ==============================================================================
# 3. INTERACTIVE CLI WITH MULTI-TASK & BATCH SUPPORT
# ==============================================================================

def print_banner():
    print("=" * 75)
    print("     LANGCHAIN MULTI-TASK AGENTIC AI (Concurrent Parallel Execution)")
    print("=" * 75)
    print(" Features:")
    print(" - Framework: LangChain + LangGraph")
    print(" - Model: Google Gemini (gemini-3.8-flash)")
    print(" - Tools: 1. calculator | 2. web_search | 3. date_time")
    print(" - Multi-tasking: Runs single queries OR multiple concurrent tasks in parallel!")
    print("-" * 75)
    print(" Commands:")
    print("  • Type 'batch' -> Run a preset batch of 4 concurrent tasks simultaneously")
    print("  • Separate multiple tasks with ';' -> e.g.:")
    print("    'What is 45*12?; What is today\\'s date?; Who is the CEO of Tesla?'")
    print("  • Type 'exit' to quit | 'clear' to clear")
    print("=" * 75 + "\n")


def display_results_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 75)
    print("               📊 MULTI-TASK EXECUTION SUMMARY")
    print("=" * 75)
    for res in results:
        status_icon = "✅" if res.get("status") == "success" else "❌"
        print(f"\n{status_icon} [Task {res.get('task_id')}]: {res.get('task')} ({res.get('elapsed_seconds')}s)")
        print(f"   Answer: {res.get('answer')}")
    print("\n" + "=" * 75 + "\n")


def main():
    print_banner()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("Error: API Key is required. Exiting.")
            sys.exit(1)

    try:
        agent = LangChainMultiTaskAgent(api_key=api_key)
        print("LangChain Multi-Task Agent initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing LangChain agent: {e}")
        sys.exit(1)

    preset_batch = [
        "Calculate sqrt(144) multiplied by 25",
        "What is the current date and day of the week?",
        "Who founded Microsoft and what year was it founded?",
        "What is 2 raised to the power 12 divided by 16?"
    ]

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            elif user_input.lower() == "batch":
                print("\n[Running 4 Sample Multi-Tasks in Parallel simultaneously...]")
                results = agent.run_multiple_tasks_parallel(preset_batch)
                display_results_table(results)
                continue

            # Check if user entered multiple tasks separated by semicolon ';'
            if ";" in user_input:
                tasks = [t.strip() for t in user_input.split(";") if t.strip()]
                if len(tasks) > 1:
                    results = agent.run_multiple_tasks_parallel(tasks)
                    display_results_table(results)
                    continue

            # Single task execution
            res = agent.run_single_task(user_input, task_id=1, verbose=True)
            print(f"\n🎯 [Agent Answer]:\n{res['answer']}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Execution Error]: {e}")


if __name__ == "__main__":
    main()
