#!/usr/bin/env python3
"""
Simple LLM + SearXNG Bridge with Instructor
============================================
Uses instructor for structured LLM outputs to ensure reliable tool calling.

Features:
- Web search via SearXNG
- SQLite-only fact storage with salience tracking
- Validated structured outputs (no more malformed tool calls)

Dependencies: pip install openai requests instructor pydantic
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, Union
from enum import Enum

import requests

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    exit(1)

try:
    import instructor
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: instructor and pydantic required. Install with: pip install instructor pydantic")
    exit(1)


# =============================================================================
# Pydantic Models for Structured Tool Calls
# =============================================================================

class ActionType(str, Enum):
    """Available actions the LLM can take."""
    SEARCH = "search"
    REMEMBER = "remember"
    RECALL = "recall"
    RESPOND = "respond"


class SearchAction(BaseModel):
    """Request a web search."""
    action: Literal["search"] = "search"
    query: str = Field(description="The search query to look up")


class RememberAction(BaseModel):
    """Store a fact in memory."""
    action: Literal["remember"] = "remember"
    fact: str = Field(description="The fact or information to remember")


class RecallAction(BaseModel):
    """Retrieve information from memory."""
    action: Literal["recall"] = "recall"
    topic: str = Field(description="The topic to search for in memory")


class RespondAction(BaseModel):
    """Provide a final response to the user."""
    action: Literal["respond"] = "respond"
    message: str = Field(description="The response message to the user")


class LLMResponse(BaseModel):
    """
    Structured response from the LLM.
    The LLM must choose exactly one action to take.
    """
    thinking: Optional[str] = Field(
        default=None,
        description="Optional brief reasoning about what action to take"
    )
    action: Union[SearchAction, RememberAction, RecallAction, RespondAction] = Field(
        description="The action to take. Use 'search' for web lookups, 'remember' to store facts, 'recall' to retrieve memories, or 'respond' to answer the user."
    )


class FactExtraction(BaseModel):
    """A single extracted fact."""
    subject: str = Field(description="The subject of the fact")
    predicate: str = Field(description="The relationship or verb")
    object: str = Field(description="The object or value")


class ExtractedFacts(BaseModel):
    """List of facts extracted from text."""
    facts: list[FactExtraction] = Field(
        default_factory=list,
        description="List of extracted facts. Empty if no clear facts found."
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    llamacpp_url: str = "http://localhost:8080"
    searxng_url: str = "http://localhost:8888"
    db_path: Path = None
    search_results: int = 5
    max_retries: int = 3  # instructor retry attempts

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = Path.home() / ".llm_memory" / "simple_memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Memory Storage (SQLite)
# =============================================================================

class SimpleMemory:
    """SQLite-based memory with salience tracking."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    salience REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(subject, predicate, object)
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_facts_salience
                ON facts(salience DESC);
            """)

    def add_fact(self, subject: str, predicate: str, obj: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO facts (subject, predicate, object)
                VALUES (?, ?, ?)
                ON CONFLICT(subject, predicate, object) DO UPDATE SET
                    access_count = access_count + 1,
                    salience = MIN(1.0, salience + 0.1),
                    last_accessed = CURRENT_TIMESTAMP
            """, (subject.lower(), predicate.lower(), obj))

    def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            words = query.lower().split()
            if not words:
                return []
            conditions = " OR ".join(
                ["subject LIKE ? OR object LIKE ?" for _ in words]
            )
            params = []
            for w in words:
                params.extend([f"%{w}%", f"%{w}%"])

            cursor = conn.execute(f"""
                SELECT subject, predicate, object, salience
                FROM facts
                WHERE {conditions}
                ORDER BY salience DESC
                LIMIT ?
            """, (*params, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_top_facts(self, limit: int = 15) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT subject, predicate, object, salience
                FROM facts
                WHERE salience >= 0.4
                ORDER BY salience DESC, access_count DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def cache_search(self, query: str, results: list):
        h = hashlib.md5(query.lower().encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache (query_hash, results)
                VALUES (?, ?)
            """, (h, json.dumps(results)))

    def get_cached_search(self, query: str) -> Optional[list]:
        h = hashlib.md5(query.lower().encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT results FROM search_cache WHERE query_hash = ?", (h,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def decay_all(self, factor: float = 0.95):
        """Apply decay to old facts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE facts SET salience = salience * ?
                WHERE last_accessed < datetime('now', '-1 day')
            """, (factor,))


# =============================================================================
# Web Search
# =============================================================================

def search_web(config: Config, query: str, memory: SimpleMemory) -> list[dict]:
    """Search via SearXNG with caching."""
    cached = memory.get_cached_search(query)
    if cached:
        return cached

    try:
        resp = requests.get(
            f"{config.searxng_url}/search",
            params={"q": query, "format": "json"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        results = [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")
        } for r in data.get("results", [])[:config.search_results]]

        memory.cache_search(query, results)
        return results
    except Exception as e:
        return [{"error": str(e)}]


# =============================================================================
# Instructor-Powered Chat
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant with memory and web search capabilities.

You MUST respond with a structured action. Choose one:

1. **search**: Use when you need current/recent information, facts you're unsure about, or real-time data.
   - Use for: news, current events, technical documentation, anything time-sensitive

2. **remember**: Use to store important facts for future reference.
   - Use for: user preferences, important decisions, key information

3. **recall**: Use to retrieve previously stored information.
   - Use for: checking what you know about a topic

4. **respond**: Use to provide your final answer to the user.
   - Use for: answering questions, completing requests

Always prefer to search for current information rather than relying on potentially outdated knowledge."""


def create_instructor_client(config: Config) -> instructor.Instructor:
    """Create an instructor-patched OpenAI client."""
    base_client = OpenAI(
        base_url=f"{config.llamacpp_url}/v1",
        api_key="not-needed"
    )
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def extract_facts(client: instructor.Instructor, text: str) -> list[FactExtraction]:
    """Use LLM to extract facts from text with validation."""
    try:
        result = client.chat.completions.create(
            model="local-model",
            response_model=ExtractedFacts,
            messages=[{
                "role": "user",
                "content": f"Extract clear factual statements from this text. Only include definite facts, not opinions or uncertain information.\n\nText: {text}"
            }],
            temperature=0.1,
            max_tokens=500,
            max_retries=2
        )
        return result.facts
    except Exception as e:
        print(f"Fact extraction failed: {e}")
        return []


def chat(config: Config, memory: SimpleMemory, client: instructor.Instructor,
         history: list, user_msg: str) -> str:
    """Process a chat message with structured tool calling."""

    # Build context from top facts
    top_facts = memory.get_top_facts(10)
    context = ""
    if top_facts:
        facts_str = "\n".join(
            f"- {f['subject']} {f['predicate']} {f['object']}"
            for f in top_facts
        )
        context = f"\n\nKnown facts:\n{facts_str}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + context},
        *history,
        {"role": "user", "content": user_msg}
    ]

    for iteration in range(5):  # Max action loops
        try:
            response = client.chat.completions.create(
                model="local-model",
                response_model=LLMResponse,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                max_retries=config.max_retries
            )
        except Exception as e:
            return f"Error getting structured response: {e}"

        action = response.action

        # Handle search action
        if isinstance(action, SearchAction):
            print(f"\n[Searching: {action.query}]")
            results = search_web(config, action.query, memory)

            if results and "error" not in results[0]:
                results_text = "\n".join(
                    f"{i+1}. {r['title']}\n   {r['content']}"
                    for i, r in enumerate(results)
                )
                messages.append({
                    "role": "assistant",
                    "content": f"[Searched for: {action.query}]"
                })
                messages.append({
                    "role": "user",
                    "content": f"Search results:\n{results_text}\n\nNow provide your response using the 'respond' action."
                })
                continue
            else:
                error = results[0].get("error", "Unknown error") if results else "No results"
                messages.append({
                    "role": "assistant",
                    "content": f"[Search failed: {error}]"
                })
                messages.append({
                    "role": "user",
                    "content": f"Search failed: {error}. Please respond with what you know or explain the issue."
                })
                continue

        # Handle remember action
        elif isinstance(action, RememberAction):
            print(f"\n[Remembering: {action.fact}]")
            extracted = extract_facts(client, action.fact)
            for f in extracted:
                memory.add_fact(f.subject, f.predicate, f.object)

            if not extracted:
                # Store as a simple fact
                memory.add_fact("info", "is", action.fact)

            messages.append({
                "role": "assistant",
                "content": f"[Remembered: {action.fact}]"
            })
            messages.append({
                "role": "user",
                "content": "Got it, fact stored. Now continue with your response."
            })
            continue

        # Handle recall action
        elif isinstance(action, RecallAction):
            print(f"\n[Recalling: {action.topic}]")
            facts = memory.search_facts(action.topic)

            if facts:
                facts_text = "\n".join(
                    f"- {f['subject']} {f['predicate']} {f['object']}"
                    for f in facts
                )
                messages.append({
                    "role": "assistant",
                    "content": f"[Recalled information about: {action.topic}]"
                })
                messages.append({
                    "role": "user",
                    "content": f"Recalled facts:\n{facts_text}\n\nNow provide your response."
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"[No memories found for: {action.topic}]"
                })
                messages.append({
                    "role": "user",
                    "content": "No stored memories found for that topic. Please respond with what you know."
                })
            continue

        # Handle respond action (final answer)
        elif isinstance(action, RespondAction):
            # Auto-extract facts from conversation
            combined = f"User: {user_msg}\nAssistant: {action.message}"
            extracted = extract_facts(client, combined)
            for f in extracted:
                memory.add_fact(f.subject, f.predicate, f.object)

            return action.message

    return "Error: Maximum action loops reached"


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple LLM + SearXNG Chat (Instructor)")
    parser.add_argument("--llm", default="http://localhost:8080")
    parser.add_argument("--searxng", default="http://localhost:8888")
    parser.add_argument("--retries", type=int, default=3, help="Max instructor retries")
    args = parser.parse_args()

    config = Config(
        llamacpp_url=args.llm,
        searxng_url=args.searxng,
        max_retries=args.retries
    )
    memory = SimpleMemory(config.db_path)
    client = create_instructor_client(config)

    print(f"LLM: {config.llamacpp_url}")
    print(f"SearXNG: {config.searxng_url}")
    print(f"Memory: {config.db_path}")
    print(f"Mode: Instructor (structured outputs)")
    print("-" * 50)
    print("Commands: /facts, /clear, /quit")
    print("-" * 50)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/facts":
            facts = memory.get_top_facts(20)
            for f in facts:
                print(f"  [{f['salience']:.2f}] {f['subject']} {f['predicate']} {f['object']}")
            continue
        elif user_input == "/clear":
            history = []
            print("History cleared.")
            continue

        reply = chat(config, memory, client, history, user_input)
        print(f"\nAssistant: {reply}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        # Keep history manageable
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
