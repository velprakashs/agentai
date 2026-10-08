import os
import sys
import json
import csv
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Ensure Windows terminal outputs UTF-8 cleanly without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "").strip()

from google import genai
from google.genai import types


# ==============================================================================
# 1. DOCUMENT LOADER & RETRIEVER
# ==============================================================================

class StudentDatabaseRetriever:
    """
    Loads and retrieves student records from the database using keyword and attribute indexing.
    """
    def __init__(self, csv_path: str = "students.csv"):
        self.csv_path = csv_path
        self.students: List[Dict[str, Any]] = []
        self.documents: List[Dict[str, Any]] = []
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Database file '{self.csv_path}' not found.")

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.students.append(row)
                text_content = (
                    f"Student ID: {row['StudentID']} | Name: {row['Name']} | Department: {row['Department']} | "
                    f"Year: {row['Year']} | Email: {row['Email']} | Phone: {row['Phone']} | "
                    f"Attendance: {row['Attendance']} | Python: {row['Python']} | DSA: {row['DSA']} | "
                    f"AI: {row['AI']} | Overall: {row['Overall']} | Placement: {row['Placement']}"
                )
                self.documents.append({"content": text_content, "metadata": row})

    def retrieve(self, query: str, top_k: int = 15) -> List[str]:
        """
        Retrieves relevant student documents based on terms and query context.
        """
        q = query.lower()
        scored_docs = []

        for doc in self.documents:
            score = 0
            text = doc["content"].lower()
            meta = doc["metadata"]

            # Direct name/ID match (high weight)
            if meta["Name"].lower() in q:
                score += 50
            if meta["StudentID"].lower() in q:
                score += 50

            # Department match
            dept = meta["Department"].lower()
            if dept in q or (dept == "aids" and ("ai & ds" in q or "artificial intelligence" in q)):
                score += 15

            # Placement match
            if "placement" in q and meta["Placement"].lower() in q:
                score += 10

            # Token matches
            tokens = [t for t in q.replace("?", "").replace(",", "").split() if len(t) > 2]
            for t in tokens:
                if t in text:
                    score += 2

            scored_docs.append((score, doc["content"]))

        # For aggregation / ranking queries, return full document for complete grounding
        broad_keywords = ["highest", "all", "how many", "list", "who scored", "overall", "attendance", "below", "above", "lowest"]
        if any(w in q for w in broad_keywords):
            return [d["content"] for d in self.documents]

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [content for score, content in scored_docs[:top_k]]


# ==============================================================================
# 2. RAG AGENT
# ==============================================================================

class StudentRAGAgent:
    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        self.retriever = StudentDatabaseRetriever()
        self.client = genai.Client(api_key=api_key)

        self.system_prompt = (
            "You are an accurate, reliable Document QA / RAG Agent for the Student Database.\n"
            "Answer the user's questions strictly using the provided student records context below.\n"
            "Format your answers with bullet points, names, and exact numbers/percentages when listing multiple students.\n"
            "If asked about rankings (e.g., highest attendance or highest marks), check all relevant records carefully before answering."
        )

    def ask(self, query: str, verbose: bool = True) -> str:
        """
        RAG workflow: Retrieve context -> Augment prompt -> Generate Answer
        """
        start_time = time.time()
        
        # 1. Retrieve relevant records
        relevant_contents = self.retriever.retrieve(query)
        context_text = "\n".join([f"- {c}" for c in relevant_contents])

        if verbose:
            print(f"\n🔍 [RAG Retrieval]: Retrieved {len(relevant_contents)} student record(s) from document context.")

        prompt = (
            f"STUDENT DATABASE CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        # 2. Generate grounded answer with retry for 503
        max_retries = 5
        delays = [2, 4, 8, 12, 16]

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.0
                    )
                )
                elapsed = round(time.time() - start_time, 2)
                if verbose:
                    print(f"⏱️ [Completed in {elapsed}s]")
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                    wait_sec = delays[attempt]
                    print(f"⏳ High traffic (503). Retrying in {wait_sec}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_sec)
                else:
                    return f"Error retrieving answer: {err_str}"


# ==============================================================================
# 3. INTERACTIVE CLI & PRESET QUESTIONS
# ==============================================================================

SUGGESTED_QUESTIONS = [
    "1. What is Priya Sharma's attendance and overall mark?",
    "2. Who has the highest overall mark?",
    "3. Which students belong to the CSE department?",
    "4. List students whose attendance is below 75%.",
    "5. Who scored above 90 in AI?",
    "6. How many students are placement eligible?",
    "7. What is Arjun Kumar's DSA mark?",
    "8. Which student has the highest attendance?",
    "9. Find all AIDS students with an overall mark above 80.",
    "10. What is the phone number of Fathima N?"
]


def print_banner():
    print("=" * 75)
    print("     STUDENT DATABASE RAG AGENT (Retrieval-Augmented Generation)")
    print("=" * 75)
    print(" Document: Dummy Student Database (25 Records)")
    print(" Fields: Student ID, Name, Dept, Attendance, Python, DSA, AI, Overall, Placement")
    print("-" * 75)
    print(" Commands:")
    print("  • Type 'questions' -> View all 10 suggested questions from the document")
    print("  • Type 'test_all'  -> Run all 10 suggested questions automatically")
    print("  • Type 1-10        -> Run that specific suggested question")
    print("  • Type 'exit' to quit | 'clear' to clear")
    print("=" * 75 + "\n")


def main():
    print_banner()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("Error: API Key is required. Exiting.")
            sys.exit(1)

    try:
        agent = StudentRAGAgent(api_key=api_key)
        print("RAG Agent loaded with 'students.csv' database!\n")
    except Exception as e:
        print(f"Error initializing RAG agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nRAG User > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            elif user_input.lower() == "questions":
                print("\n--- 10 Suggested RAG Questions ---")
                for q in SUGGESTED_QUESTIONS:
                    print(q)
                print("----------------------------------\n")
                continue
            elif user_input.lower() == "test_all":
                print("\n🚀 Executing all 10 suggested questions from the document...\n")
                for q in SUGGESTED_QUESTIONS:
                    clean_q = q.split(". ", 1)[1]
                    print(f"\n❓ [Question]: {clean_q}")
                    ans = agent.ask(clean_q, verbose=False)
                    print(f"💡 [Answer]:\n{ans}")
                    print("-" * 60)
                continue

            # Check if user typed a number between 1 and 10
            if user_input.isdigit() and 1 <= int(user_input) <= 10:
                selected_q = SUGGESTED_QUESTIONS[int(user_input) - 1].split(". ", 1)[1]
                print(f"\n❓ [Question {user_input}]: {selected_q}")
                answer = agent.ask(selected_q, verbose=True)
                print(f"\n💡 [RAG Answer]:\n{answer}\n")
                continue

            # Free-form question
            answer = agent.ask(user_input, verbose=True)
            print(f"\n💡 [RAG Answer]:\n{answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[RAG Error]: {e}")


if __name__ == "__main__":
    main()
