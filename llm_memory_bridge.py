#!/usr/bin/env python3
"""
LLM Memory Bridge with Web Search (Instructor Edition)
======================================================
Connects llama.cpp to:
1. SearXNG for web search
2. ChromaDB for persistent vector memory
3. SQLite for structured fact storage

Uses instructor for validated structured outputs - no more malformed tool calls!

Memory Architecture:
- Working Memory: Current conversation context
- Episodic Memory: Past conversations (vector search)
- Semantic Memory: Extracted facts and entities (structured)
- Web Memory: Cached search results
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, Union
from enum import Enum
import requests

# Optional imports - graceful degradation
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Vector memory disabled.")
    print("Install with: pip install chromadb")

try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    print("Warning: openai package not installed.")
    print("Install with: pip install openai")

try:
    import instructor
    from pydantic import BaseModel, Field
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    print("Warning: instructor/pydantic not installed. Structured outputs disabled.")
    print("Install with: pip install instructor pydantic")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using ChromaDB's default embeddings.")
    print("Install with: pip install sentence-transformers")


# =============================================================================
# Pydantic Models for Structured Tool Calls
# =============================================================================

if INSTRUCTOR_AVAILABLE:
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
        confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in this fact (0-1)")

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
    # LLM Backend
    llamacpp_url: str = "http://localhost:8080"
    model_name: str = "local-model"

    # SearXNG
    searxng_url: str = "http://localhost:8888"
    search_results_count: int = 5

    # Memory Storage
    data_dir: Path = field(default_factory=lambda: Path.home() / ".llm_memory")

    # Memory Settings
    max_working_memory_tokens: int = 4000
    memory_decay_days: int = 30
    salience_threshold: float = 0.5

    # Embedding Model (if using local embeddings)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Instructor settings
    max_retries: int = 3

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Structured Memory Store (SQLite)
# =============================================================================

class StructuredMemory:
    """
    Stores extracted facts, entities, and relationships.
    Good for: user preferences, learned facts, entity relationships
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    salience REAL DEFAULT 0.5,
                    UNIQUE(subject, predicate, object)
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT,
                    description TEXT,
                    attributes JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
                CREATE INDEX IF NOT EXISTS idx_facts_salience ON facts(salience DESC);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            """)

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, source: str = None) -> bool:
        """Add or update a fact. Returns True if new, False if updated."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO facts (subject, predicate, object, confidence, source)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(subject, predicate, object) DO UPDATE SET
                        confidence = MAX(confidence, excluded.confidence),
                        last_accessed = CURRENT_TIMESTAMP,
                        access_count = access_count + 1,
                        salience = MIN(1.0, salience + 0.1)
                """, (subject.lower(), predicate.lower(), obj, confidence, source))
                return conn.total_changes > 0
            except Exception as e:
                print(f"Error adding fact: {e}")
                return False

    def get_facts_about(self, subject: str, limit: int = 10) -> list[dict]:
        """Retrieve facts about a subject, ordered by salience."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT subject, predicate, object, confidence, salience
                FROM facts
                WHERE subject LIKE ? OR object LIKE ?
                ORDER BY salience DESC, last_accessed DESC
                LIMIT ?
            """, (f"%{subject.lower()}%", f"%{subject.lower()}%", limit))

            # Update access timestamps
            results = [dict(row) for row in cursor.fetchall()]
            if results:
                conn.execute("""
                    UPDATE facts SET last_accessed = CURRENT_TIMESTAMP,
                                    access_count = access_count + 1
                    WHERE subject LIKE ? OR object LIKE ?
                """, (f"%{subject.lower()}%", f"%{subject.lower()}%"))

            return results

    def get_high_salience_facts(self, limit: int = 20) -> list[dict]:
        """Get the most important facts for context injection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT subject, predicate, object, confidence, salience
                FROM facts
                WHERE salience >= 0.5
                ORDER BY salience DESC, access_count DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def decay_salience(self, decay_factor: float = 0.95):
        """Apply time-based decay to all facts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE facts
                SET salience = salience * ?
                WHERE last_accessed < datetime('now', '-1 day')
            """, (decay_factor,))

    def add_entity(self, name: str, entity_type: str = None,
                   description: str = None, attributes: dict = None):
        """Add or update an entity."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO entities (name, entity_type, description, attributes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    entity_type = COALESCE(excluded.entity_type, entity_type),
                    description = COALESCE(excluded.description, description),
                    attributes = COALESCE(excluded.attributes, attributes),
                    last_accessed = CURRENT_TIMESTAMP
            """, (name, entity_type, description,
                  json.dumps(attributes) if attributes else None))

    def cache_search(self, query: str, results: list, ttl_hours: int = 24):
        """Cache search results."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache
                (query_hash, query, results, expires_at)
                VALUES (?, ?, ?, ?)
            """, (query_hash, query, json.dumps(results), expires_at))

    def get_cached_search(self, query: str) -> Optional[list]:
        """Get cached search results if not expired."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT results FROM search_cache
                WHERE query_hash = ? AND expires_at > CURRENT_TIMESTAMP
            """, (query_hash,))
            row = cursor.fetchone()
            return json.loads(row['results']) if row else None


# =============================================================================
# Vector Memory Store (ChromaDB)
# =============================================================================

class VectorMemory:
    """
    Stores conversation chunks and documents for semantic search.
    Good for: finding relevant past conversations, document retrieval
    """

    def __init__(self, persist_dir: Path, embedding_model: str = None):
        if not CHROMA_AVAILABLE:
            self.client = None
            self.collection = None
            return

        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Use custom embeddings if sentence-transformers is available
        if EMBEDDINGS_AVAILABLE and embedding_model:
            self.embedder = SentenceTransformer(embedding_model)
            self.collection = self.client.get_or_create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Use ChromaDB's default embeddings
            self.collection = self.client.get_or_create_collection(
                name="conversations"
            )
            self.embedder = None

    def add_memory(self, text: str, metadata: dict = None, memory_id: str = None):
        """Add a memory chunk to the vector store."""
        if not self.collection:
            return

        if memory_id is None:
            memory_id = hashlib.md5(
                f"{text}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

        metadata = metadata or {}
        metadata["timestamp"] = datetime.now().isoformat()

        if self.embedder:
            embedding = self.embedder.encode(text).tolist()
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
        else:
            self.collection.add(
                ids=[memory_id],
                documents=[text],
                metadatas=[metadata]
            )

    def search(self, query: str, n_results: int = 5,
               where: dict = None) -> list[dict]:
        """Search for similar memories."""
        if not self.collection:
            return []

        if self.embedder:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )

        memories = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        return memories

    def add_conversation_turn(self, role: str, content: str,
                              conversation_id: str = None):
        """Add a conversation turn with appropriate metadata."""
        self.add_memory(
            text=f"{role}: {content}",
            metadata={
                "type": "conversation",
                "role": role,
                "conversation_id": conversation_id or "default"
            }
        )


# =============================================================================
# Web Search (SearXNG)
# =============================================================================

class WebSearch:
    """Interface to SearXNG for web searches."""

    def __init__(self, base_url: str, structured_memory: StructuredMemory = None):
        self.base_url = base_url.rstrip('/')
        self.structured_memory = structured_memory

    def search(self, query: str, num_results: int = 5,
               use_cache: bool = True) -> list[dict]:
        """
        Search the web via SearXNG.
        Results are cached in structured memory if available.
        """
        # Check cache first
        if use_cache and self.structured_memory:
            cached = self.structured_memory.get_cached_search(query)
            if cached:
                return cached[:num_results]

        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general"
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("results", [])[:num_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "engine": r.get("engine", ""),
                    "score": r.get("score", 0)
                })

            # Cache results
            if self.structured_memory and results:
                self.structured_memory.cache_search(query, results)

            return results

        except requests.exceptions.RequestException as e:
            return [{"error": f"Search failed: {str(e)}"}]


# =============================================================================
# Fact Extraction (Instructor-powered)
# =============================================================================

def extract_facts_with_llm(client, text: str, model: str = "local-model",
                           max_retries: int = 2) -> list[dict]:
    """Use the LLM to extract structured facts from text with validation."""
    if not INSTRUCTOR_AVAILABLE or not client:
        return []

    try:
        result = client.chat.completions.create(
            model=model,
            response_model=ExtractedFacts,
            messages=[{
                "role": "user",
                "content": f"""Extract key facts from this text as subject-predicate-object triples.
Focus on:
- User preferences and personal information
- Important decisions or statements
- Named entities and their relationships

Only include clear factual statements, not opinions or uncertain information.

Text: {text}"""
            }],
            temperature=0.1,
            max_tokens=1000,
            max_retries=max_retries
        )

        # Convert to dict format for compatibility
        return [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence
            }
            for f in result.facts
        ]

    except Exception as e:
        print(f"Fact extraction failed: {e}")
        return []


# =============================================================================
# Memory-Augmented Chat (Instructor Edition)
# =============================================================================

class MemoryAugmentedChat:
    """
    Main class that orchestrates:
    - LLM chat with llama.cpp (via instructor for structured outputs)
    - Web search via SearXNG
    - Memory storage and retrieval
    - Fact extraction and salience tracking
    """

    SYSTEM_PROMPT = """You are a helpful assistant with persistent memory and web search capabilities.

You MUST respond with a structured action. Choose one:

1. **search**: Use when you need current/recent information, facts you're unsure about, or real-time data.
   - Use for: news, current events, technical documentation, anything time-sensitive

2. **remember**: Use to store important facts for future reference.
   - Use for: user preferences, important decisions, key information the user shares

3. **recall**: Use to retrieve previously stored information from your memory.
   - Use for: checking what you know about a topic before answering

4. **respond**: Use to provide your final answer to the user.
   - Use for: answering questions, completing requests, having conversations

## Guidelines
- Always prefer to search for current information rather than relying on potentially outdated knowledge
- Use recall to check your memory before answering questions about things the user has told you
- Use remember to store important information the user shares
- Be concise and helpful in your responses
- Cite sources when using search results"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Initialize storage
        self.structured_memory = StructuredMemory(
            self.config.data_dir / "facts.db"
        )

        self.vector_memory = VectorMemory(
            self.config.data_dir / "vectors",
            self.config.embedding_model if EMBEDDINGS_AVAILABLE else None
        )

        self.web_search = WebSearch(
            self.config.searxng_url,
            self.structured_memory
        )

        # Initialize LLM client with instructor
        if OPENAI_CLIENT_AVAILABLE and INSTRUCTOR_AVAILABLE:
            base_client = OpenAI(
                base_url=f"{self.config.llamacpp_url}/v1",
                api_key="not-needed"
            )
            self.llm_client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
        elif OPENAI_CLIENT_AVAILABLE:
            # Fallback to regular client (will use legacy tag parsing)
            self.llm_client = OpenAI(
                base_url=f"{self.config.llamacpp_url}/v1",
                api_key="not-needed"
            )
            print("Warning: Running without instructor. Tool calls may be unreliable.")
        else:
            self.llm_client = None

        self.conversation_history = []

    def _build_context(self, user_message: str) -> str:
        """Build context from memories relevant to the current message."""
        context_parts = []

        # Get high-salience facts
        important_facts = self.structured_memory.get_high_salience_facts(limit=10)
        if important_facts:
            facts_text = "\n".join([
                f"- {f['subject']} {f['predicate']} {f['object']}"
                for f in important_facts
            ])
            context_parts.append(f"## Known Facts\n{facts_text}")

        # Search vector memory for relevant past conversations
        if self.vector_memory.collection:
            relevant_memories = self.vector_memory.search(user_message, n_results=3)
            if relevant_memories:
                memories_text = "\n".join([
                    f"- {m['text']}" for m in relevant_memories
                ])
                context_parts.append(f"## Relevant Past Conversations\n{memories_text}")

        return "\n\n".join(context_parts) if context_parts else ""

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results for injection into context."""
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r['title']}**\n"
                f"   URL: {r['url']}\n"
                f"   {r['content']}"
            )
        return "\n\n".join(formatted)

    def chat(self, user_message: str) -> str:
        """
        Process a user message with memory augmentation and structured tool calling.
        """
        if not self.llm_client:
            return "Error: LLM client not available. Install with: pip install openai instructor pydantic"

        if not INSTRUCTOR_AVAILABLE:
            return "Error: instructor not available. Install with: pip install instructor pydantic"

        # Build context from memories
        context = self._build_context(user_message)

        # Prepare messages
        system_content = self.SYSTEM_PROMPT
        if context:
            system_content += f"\n\n## Current Context\n{context}"

        messages = [
            {"role": "system", "content": system_content},
            *self.conversation_history,
            {"role": "user", "content": user_message}
        ]

        # Store user message in vector memory
        self.vector_memory.add_conversation_turn("user", user_message)

        # Chat loop (handles continuation for search/recall)
        max_iterations = 5

        for iteration in range(max_iterations):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.config.model_name,
                    response_model=LLMResponse,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    max_retries=self.config.max_retries
                )
            except Exception as e:
                return f"Error getting structured response: {e}"

            action = response.action

            # Handle search action
            if isinstance(action, SearchAction):
                print(f"\n[Searching: {action.query}]")
                results = self.web_search.search(action.query, self.config.search_results_count)

                if results and "error" not in results[0]:
                    search_text = self._format_search_results(results)
                    messages.append({
                        "role": "assistant",
                        "content": f"[Searched for: {action.query}]"
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Search results:\n{search_text}\n\nNow provide your response using the 'respond' action."
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

                # Extract structured facts using LLM
                extracted = extract_facts_with_llm(
                    self.llm_client, action.fact,
                    self.config.model_name, self.config.max_retries
                )
                for f in extracted:
                    self.structured_memory.add_fact(
                        f.get("subject", "unknown"),
                        f.get("predicate", "is"),
                        f.get("object", action.fact),
                        f.get("confidence", 0.8),
                        source="conversation"
                    )

                # Also store in vector memory
                self.vector_memory.add_memory(
                    action.fact,
                    {"type": "remembered", "source": "explicit"}
                )

                # If no structured facts extracted, store as simple fact
                if not extracted:
                    self.structured_memory.add_fact(
                        "info", "is", action.fact,
                        confidence=0.8, source="conversation"
                    )

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

                # Search both structured and vector memory
                facts = self.structured_memory.get_facts_about(action.topic)
                memories = self.vector_memory.search(action.topic, n_results=3)

                recall_text = ""
                if facts:
                    recall_text += "Facts:\n" + "\n".join([
                        f"- {f['subject']} {f['predicate']} {f['object']}"
                        for f in facts
                    ])
                if memories:
                    recall_text += "\nMemories:\n" + "\n".join([
                        f"- {m['text']}" for m in memories
                    ])

                if recall_text:
                    messages.append({
                        "role": "assistant",
                        "content": f"[Recalled information about: {action.topic}]"
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Recalled information:\n{recall_text}\n\nNow provide your response."
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
                processed_message = action.message

                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": processed_message
                })

                # Store in vector memory
                self.vector_memory.add_conversation_turn("assistant", processed_message)

                # Extract and store facts from the conversation
                combined = f"User: {user_message}\nAssistant: {processed_message}"
                facts = extract_facts_with_llm(
                    self.llm_client, combined,
                    self.config.model_name, self.config.max_retries
                )
                for f in facts:
                    if f.get("confidence", 0) >= self.config.salience_threshold:
                        self.structured_memory.add_fact(
                            f.get("subject", ""),
                            f.get("predicate", ""),
                            f.get("object", ""),
                            f.get("confidence", 0.5),
                            source="auto_extracted"
                        )

                return processed_message

        return "Error: Too many continuation loops"

    def get_memory_stats(self) -> dict:
        """Get statistics about stored memories."""
        stats = {
            "facts": 0,
            "high_salience_facts": 0,
            "entities": 0,
            "vector_memories": 0
        }

        with sqlite3.connect(self.structured_memory.db_path) as conn:
            stats["facts"] = conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
            stats["high_salience_facts"] = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE salience >= 0.5"
            ).fetchone()[0]
            stats["entities"] = conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]

        if self.vector_memory.collection:
            stats["vector_memories"] = self.vector_memory.collection.count()

        return stats


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Interactive chat interface."""
    print("=" * 60)
    print("LLM Memory Bridge (Instructor Edition)")
    print("=" * 60)

    # Parse command line args or use defaults
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-url", default="http://localhost:8080",
                       help="llama.cpp server URL")
    parser.add_argument("--searxng-url", default="http://localhost:8888",
                       help="SearXNG server URL")
    parser.add_argument("--data-dir", default=str(Path.home() / ".llm_memory"),
                       help="Directory for memory storage")
    parser.add_argument("--retries", type=int, default=3,
                       help="Max instructor retries on validation failure")
    args = parser.parse_args()

    config = Config(
        llamacpp_url=args.llm_url,
        searxng_url=args.searxng_url,
        data_dir=Path(args.data_dir),
        max_retries=args.retries
    )

    print(f"LLM: {config.llamacpp_url}")
    print(f"SearXNG: {config.searxng_url}")
    print(f"Memory: {config.data_dir}")
    print(f"Mode: Instructor (structured outputs)")
    print("-" * 60)

    chat = MemoryAugmentedChat(config)

    # Show memory stats
    stats = chat.get_memory_stats()
    print(f"Loaded {stats['facts']} facts, {stats['vector_memories']} memories")
    print("-" * 60)
    print("Commands: /stats, /facts, /clear, /quit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/stats":
                stats = chat.get_memory_stats()
                print(f"\nMemory Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif cmd == "/facts":
                facts = chat.structured_memory.get_high_salience_facts(20)
                print("\nHigh-salience facts:")
                for f in facts:
                    print(f"  [{f['salience']:.2f}] {f['subject']} {f['predicate']} {f['object']}")
            elif cmd == "/clear":
                chat.conversation_history = []
                print("Conversation history cleared.")
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Chat
        response = chat.chat(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
