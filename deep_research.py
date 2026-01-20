#!/usr/bin/env python3
"""
Deep Research Protocol with Task Anchoring
==========================================
An enhanced agentic search system that prevents premature termination,
enforces minimum research thresholds, and maintains task adherence.

Key Features:
- Multi-phase research protocol (Planning -> Researching -> Synthesizing)
- Task decomposition into sub-questions with coverage tracking
- Minimum search thresholds before synthesis is allowed
- Mandatory progress reports every N iterations
- Self-critique validation before final response
- Parallel search capability for throughput
- Configurable research depth modes

Dependencies: pip install openai requests instructor pydantic
"""

import json
import sqlite3
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, Union, Annotated
from enum import Enum
from contextlib import contextmanager

import requests

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    exit(1)

try:
    import instructor
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    print("Error: instructor and pydantic required. Install with: pip install instructor pydantic")
    exit(1)


# =============================================================================
# Research Protocol Configuration
# =============================================================================

class ResearchMode(str, Enum):
    """Configurable research depth modes."""
    QUICK = "quick"           # Fast answers, minimal research
    STANDARD = "standard"     # Balanced depth and speed
    THOROUGH = "thorough"     # Comprehensive research
    EXHAUSTIVE = "exhaustive" # Maximum depth, no shortcuts


@dataclass
class ResearchModeSettings:
    """Settings for each research mode."""
    min_searches: int
    max_iterations: int
    require_progress_reports: bool
    progress_report_interval: int
    require_self_critique: bool
    max_subquestions: int

    @classmethod
    def for_mode(cls, mode: ResearchMode) -> "ResearchModeSettings":
        settings = {
            ResearchMode.QUICK: cls(
                min_searches=1,
                max_iterations=5,
                require_progress_reports=False,
                progress_report_interval=0,
                require_self_critique=False,
                max_subquestions=3
            ),
            ResearchMode.STANDARD: cls(
                min_searches=3,
                max_iterations=10,
                require_progress_reports=True,
                progress_report_interval=3,
                require_self_critique=True,
                max_subquestions=5
            ),
            ResearchMode.THOROUGH: cls(
                min_searches=5,
                max_iterations=15,
                require_progress_reports=True,
                progress_report_interval=3,
                require_self_critique=True,
                max_subquestions=7
            ),
            ResearchMode.EXHAUSTIVE: cls(
                min_searches=8,
                max_iterations=25,
                require_progress_reports=True,
                progress_report_interval=4,
                require_self_critique=True,
                max_subquestions=10
            ),
        }
        return settings[mode]


@dataclass
class DeepResearchConfig:
    """Configuration for deep research sessions."""
    llamacpp_url: str = "http://localhost:8080"
    searxng_url: str = "http://localhost:8888"
    db_path: Path = None
    search_results_per_query: int = 5
    max_retries: int = 3
    mode: ResearchMode = ResearchMode.STANDARD

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = Path.home() / ".llm_memory" / "deep_research.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode_settings = ResearchModeSettings.for_mode(self.mode)


# =============================================================================
# Research Phase Definitions
# =============================================================================

class ResearchPhase(str, Enum):
    """The three phases of the research protocol."""
    PLANNING = "planning"         # Decompose query into sub-questions
    RESEARCHING = "researching"   # Execute searches, gather evidence
    SYNTHESIZING = "synthesizing" # Validate and produce final answer


# =============================================================================
# Pydantic Models - Sub-Questions and Planning
# =============================================================================

class SubQuestionPriority(str, Enum):
    """Priority levels for sub-questions."""
    REQUIRED = "required"   # Must be answered
    HELPFUL = "helpful"     # Should be answered if possible
    OPTIONAL = "optional"   # Nice to have


class SubQuestion(BaseModel):
    """A decomposed aspect of the original query."""
    question: str = Field(description="Specific sub-question to investigate")
    priority: SubQuestionPriority = Field(description="How important is this sub-question")
    search_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Suggested search queries for this sub-question"
    )


class ResearchPlan(BaseModel):
    """
    Generated at the start of research - locks in the scope.
    This prevents divergence by establishing clear goals upfront.
    """
    original_query: str = Field(description="The exact user query, verbatim")
    query_intent: str = Field(description="What the user is actually trying to learn/accomplish")
    sub_questions: list[SubQuestion] = Field(
        min_length=2,
        description="Decomposed questions that together answer the original query"
    )
    minimum_searches_recommended: int = Field(
        ge=1, le=10,
        description="How many searches are needed for a thorough answer"
    )

    @field_validator('sub_questions')
    @classmethod
    def at_least_one_required(cls, v):
        required = [sq for sq in v if sq.priority == SubQuestionPriority.REQUIRED]
        if not required:
            raise ValueError("At least one sub-question must be marked as REQUIRED")
        return v


class PlanningResponse(BaseModel):
    """Response model for the planning phase - ONLY ResearchPlan allowed."""
    thinking: str = Field(description="Reasoning about how to decompose this query")
    plan: ResearchPlan


# =============================================================================
# Pydantic Models - Research Actions
# =============================================================================

class SearchAction(BaseModel):
    """Execute a single web search."""
    action: Literal["search"] = "search"
    query: str = Field(description="The search query to execute")
    targets_subquestion: int = Field(
        ge=0,
        description="Index of the sub-question this search addresses"
    )
    reasoning: str = Field(description="Why this search helps answer the sub-question")


class ParallelSearchAction(BaseModel):
    """Execute multiple searches in one iteration for efficiency."""
    action: Literal["parallel_search"] = "parallel_search"
    searches: list[SearchAction] = Field(
        min_length=2,
        max_length=4,
        description="Multiple searches to execute in parallel"
    )


class RememberAction(BaseModel):
    """Store important information for later recall."""
    action: Literal["remember"] = "remember"
    fact: str = Field(description="The fact or information to remember")
    related_subquestion: Optional[int] = Field(
        default=None,
        description="Index of sub-question this fact relates to"
    )


class RecallAction(BaseModel):
    """Retrieve previously stored information."""
    action: Literal["recall"] = "recall"
    topic: str = Field(description="The topic to search for in memory")


# =============================================================================
# Pydantic Models - Progress Tracking
# =============================================================================

class SubQuestionStatus(BaseModel):
    """Status of a single sub-question."""
    index: int
    has_evidence: bool
    evidence_quality: Literal["none", "weak", "moderate", "strong"]
    notes: str = Field(default="", description="Brief notes on what was found")


class ProgressReport(BaseModel):
    """
    Mandatory self-assessment during research.
    Forces the model to reflect on progress and stay on track.
    """
    action: Literal["progress_report"] = "progress_report"

    # Sub-question coverage
    subquestion_statuses: list[SubQuestionStatus]

    # Overall assessment
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that we can answer the original query well"
    )

    # Decision
    should_continue_researching: bool = Field(
        description="True if more research would significantly improve the answer"
    )
    next_steps: str = Field(
        description="What to do next - which sub-questions need more work"
    )

    # Task adherence check
    still_on_track: bool = Field(
        description="Are we still addressing the original query, not tangents?"
    )
    drift_concerns: Optional[str] = Field(
        default=None,
        description="If drifting, what are we drifting toward?"
    )


class ProgressReportResponse(BaseModel):
    """Response model when progress report is required."""
    thinking: str = Field(description="Reflection on research progress")
    report: ProgressReport


# =============================================================================
# Pydantic Models - Synthesis Phase
# =============================================================================

class SynthesisAction(BaseModel):
    """
    Final response action - only allowed after minimum research threshold.
    Includes mandatory self-validation fields.
    """
    action: Literal["synthesize"] = "synthesize"

    # Task adherence validation (model must explicitly confirm)
    addresses_original_query: bool = Field(
        description="Does this response directly answer what was originally asked?"
    )

    # Coverage tracking
    subquestions_addressed: list[int] = Field(
        description="Indices of sub-questions this response covers"
    )
    subquestions_not_addressed: list[int] = Field(
        default_factory=list,
        description="Indices of sub-questions we couldn't answer"
    )

    # Quality assessment
    evidence_quality: Literal["weak", "moderate", "strong"] = Field(
        description="Overall quality of evidence supporting this response"
    )

    # Gaps and limitations
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Known gaps or limitations in this response"
    )

    # The actual response
    final_response: str = Field(
        min_length=50,
        description="The comprehensive response to the user"
    )

    @field_validator('addresses_original_query')
    @classmethod
    def must_address_query(cls, v):
        if not v:
            raise ValueError(
                "Cannot synthesize a response that doesn't address the original query. "
                "Continue researching or reformulate your response."
            )
        return v

    @field_validator('subquestions_addressed')
    @classmethod
    def must_address_some_questions(cls, v):
        if len(v) == 0:
            raise ValueError(
                "Response must address at least one sub-question. "
                "Continue researching if no sub-questions can be answered."
            )
        return v


class SelfCritique(BaseModel):
    """
    Second-pass validation before returning final response.
    The model critiques its own synthesis.
    """
    answers_original_question: bool = Field(
        description="Does the response actually answer what was asked?"
    )
    addresses_required_subquestions: bool = Field(
        description="Are all REQUIRED sub-questions addressed?"
    )
    evidence_is_cited: bool = Field(
        description="Does the response cite sources/evidence appropriately?"
    )
    no_hallucinations_detected: bool = Field(
        description="Is the response grounded in search results, not made up?"
    )
    response_is_complete: bool = Field(
        description="Is this a complete answer, not cut off or partial?"
    )

    overall_quality: Literal["poor", "acceptable", "good", "excellent"]

    recommended_action: Literal["approve", "revise", "continue_research"] = Field(
        description="What should happen with this response?"
    )

    critique_notes: str = Field(
        description="Specific feedback on what's good or needs improvement"
    )


class SelfCritiqueResponse(BaseModel):
    """Response model for self-critique phase."""
    thinking: str = Field(description="Critical analysis of the proposed response")
    critique: SelfCritique


# =============================================================================
# Pydantic Models - Main Research Response
# =============================================================================

# Union of all valid research actions
ResearchActionType = Union[
    SearchAction,
    ParallelSearchAction,
    RememberAction,
    RecallAction,
    ProgressReport,
    SynthesisAction,
]


class ResearchResponse(BaseModel):
    """
    Main response model during the research phase.
    The model must provide thinking and choose exactly one action.
    """
    thinking: str = Field(
        min_length=20,
        description="Reasoning about current state, progress, and next action"
    )
    action: ResearchActionType = Field(
        description="The action to take next"
    )


# =============================================================================
# Research State Machine
# =============================================================================

@dataclass
class ResearchState:
    """
    Tracks the complete state of a research session.
    Injected into every prompt to maintain continuity.
    """
    phase: ResearchPhase = ResearchPhase.PLANNING
    plan: Optional[ResearchPlan] = None

    # Counters
    searches_completed: int = 0
    iterations_completed: int = 0

    # Evidence tracking
    search_results: list[dict] = field(default_factory=list)
    evidence_by_subquestion: dict[int, list[str]] = field(default_factory=dict)

    # Progress reports
    progress_reports: list[ProgressReport] = field(default_factory=list)

    # Settings (from config)
    min_searches: int = 3
    max_iterations: int = 10
    require_progress_reports: bool = True
    progress_report_interval: int = 3
    require_self_critique: bool = True

    def can_synthesize(self) -> bool:
        """Check if synthesis is allowed based on thresholds."""
        return (
            self.searches_completed >= self.min_searches and
            self.iterations_completed >= 3 and
            self.phase == ResearchPhase.RESEARCHING
        )

    def needs_progress_report(self) -> bool:
        """Check if a progress report is due."""
        if not self.require_progress_reports:
            return False
        if self.progress_report_interval == 0:
            return False
        if self.iterations_completed == 0:
            return False
        return (
            self.iterations_completed % self.progress_report_interval == 0 and
            len(self.progress_reports) < self.iterations_completed // self.progress_report_interval
        )

    def get_unanswered_required_subquestions(self) -> list[int]:
        """Get indices of required sub-questions without strong evidence."""
        if not self.plan:
            return []
        unanswered = []
        for i, sq in enumerate(self.plan.sub_questions):
            if sq.priority == SubQuestionPriority.REQUIRED:
                evidence = self.evidence_by_subquestion.get(i, [])
                if len(evidence) < 2:  # Need at least 2 pieces of evidence
                    unanswered.append(i)
        return unanswered

    def to_context_string(self) -> str:
        """Format state for injection into prompts."""
        if not self.plan:
            return "Research has not started yet."

        lines = [
            "=" * 60,
            "RESEARCH STATE (DO NOT IGNORE)",
            "=" * 60,
            "",
            f"## ORIGINAL QUERY (YOUR PRIMARY OBJECTIVE)",
            f'"{self.plan.original_query}"',
            "",
            f"## QUERY INTENT",
            f"{self.plan.query_intent}",
            "",
            "## SUB-QUESTIONS TO ADDRESS",
        ]

        for i, sq in enumerate(self.plan.sub_questions):
            evidence_count = len(self.evidence_by_subquestion.get(i, []))
            status = "ANSWERED" if evidence_count >= 2 else "NEEDS RESEARCH"
            lines.append(
                f"  [{i}] [{sq.priority.value.upper()}] [{status}] {sq.question}"
            )
            if evidence_count > 0:
                lines.append(f"      Evidence pieces: {evidence_count}")

        lines.extend([
            "",
            "## PROGRESS",
            f"  Phase: {self.phase.value}",
            f"  Searches completed: {self.searches_completed}/{self.min_searches} minimum",
            f"  Iterations: {self.iterations_completed}/{self.max_iterations} maximum",
            f"  Can synthesize: {'YES' if self.can_synthesize() else 'NO - need more research'}",
        ])

        unanswered = self.get_unanswered_required_subquestions()
        if unanswered:
            lines.append(f"  Required sub-questions needing work: {unanswered}")

        if self.needs_progress_report():
            lines.extend([
                "",
                ">>> PROGRESS REPORT REQUIRED <<<",
                "You must submit a progress_report action before continuing.",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Memory Storage (SQLite) - Extended for Research Sessions
# =============================================================================

class ResearchMemory:
    """Extended memory with research session tracking."""

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
                    source TEXT,
                    UNIQUE(subject, predicate, object)
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS research_sessions (
                    id INTEGER PRIMARY KEY,
                    original_query TEXT NOT NULL,
                    plan_json TEXT,
                    state_json TEXT,
                    final_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'in_progress'
                );

                CREATE INDEX IF NOT EXISTS idx_facts_salience ON facts(salience DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_status ON research_sessions(status);
            """)

    def add_fact(self, subject: str, predicate: str, obj: str, source: str = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO facts (subject, predicate, object, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject, predicate, object) DO UPDATE SET
                    access_count = access_count + 1,
                    salience = MIN(1.0, salience + 0.1),
                    last_accessed = CURRENT_TIMESTAMP
            """, (subject.lower(), predicate.lower(), obj, source))

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
                FROM facts WHERE {conditions}
                ORDER BY salience DESC LIMIT ?
            """, (*params, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_top_facts(self, limit: int = 15) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT subject, predicate, object, salience
                FROM facts WHERE salience >= 0.4
                ORDER BY salience DESC, access_count DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def cache_search(self, query: str, results: list):
        h = hashlib.md5(query.lower().encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache (query_hash, query, results)
                VALUES (?, ?, ?)
            """, (h, query, json.dumps(results)))

    def get_cached_search(self, query: str) -> Optional[list]:
        h = hashlib.md5(query.lower().encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT results FROM search_cache WHERE query_hash = ?", (h,)
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def save_research_session(self, state: ResearchState, final_response: str = None):
        """Save or update a research session."""
        with sqlite3.connect(self.db_path) as conn:
            if final_response:
                conn.execute("""
                    INSERT INTO research_sessions
                    (original_query, plan_json, state_json, final_response, completed_at, status)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 'completed')
                """, (
                    state.plan.original_query if state.plan else "",
                    state.plan.model_dump_json() if state.plan else None,
                    json.dumps({
                        "searches_completed": state.searches_completed,
                        "iterations_completed": state.iterations_completed,
                    }),
                    final_response
                ))


# =============================================================================
# Web Search
# =============================================================================

def search_web(config: DeepResearchConfig, query: str, memory: ResearchMemory) -> list[dict]:
    """Execute a web search via SearXNG with caching."""
    cached = memory.get_cached_search(query)
    if cached:
        return cached

    try:
        resp = requests.get(
            f"{config.searxng_url}/search",
            params={"q": query, "format": "json"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        results = [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "query": query
        } for r in data.get("results", [])[:config.search_results_per_query]]

        memory.cache_search(query, results)
        return results
    except Exception as e:
        return [{"error": str(e), "query": query}]


def execute_parallel_searches(
    config: DeepResearchConfig,
    queries: list[str],
    memory: ResearchMemory
) -> list[list[dict]]:
    """Execute multiple searches (sequentially for now, async possible)."""
    return [search_web(config, q, memory) for q in queries]


# =============================================================================
# Fact Extraction
# =============================================================================

class FactExtraction(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractedFacts(BaseModel):
    facts: list[FactExtraction] = Field(default_factory=list)


def extract_facts(client: instructor.Instructor, text: str) -> list[FactExtraction]:
    """Extract facts from text using LLM."""
    try:
        result = client.chat.completions.create(
            model="local-model",
            response_model=ExtractedFacts,
            messages=[{
                "role": "user",
                "content": f"Extract clear factual statements from this text. Only definite facts, not opinions.\n\nText: {text}"
            }],
            temperature=0.1,
            max_tokens=500,
            max_retries=2
        )
        return result.facts
    except Exception:
        return []


# =============================================================================
# Deep Research Engine
# =============================================================================

class DeepResearchEngine:
    """
    The main research engine that orchestrates the multi-phase protocol.
    """

    PLANNING_PROMPT = """You are a research planning assistant. Your job is to analyze the user's query and create a comprehensive research plan.

CURRENT TIMESTAMP: {timestamp}

TASK: Decompose the following query into a structured research plan.

USER QUERY: "{query}"

You must:
1. Identify the core intent behind this query
2. Break it down into 2-{max_subquestions} specific sub-questions
3. Mark at least one sub-question as REQUIRED
4. Suggest search queries for each sub-question
5. Estimate how many searches will be needed

Be thorough - a good plan leads to a good answer."""

    RESEARCH_PROMPT = """You are a thorough research assistant conducting deep research on a topic.

CURRENT TIMESTAMP: {timestamp}

{state_context}

## AVAILABLE ACTIONS

1. **search**: Execute a web search
   - Must specify which sub-question it targets
   - Must explain why this search helps

2. **parallel_search**: Execute 2-4 searches at once
   - Use when multiple searches are clearly needed
   - Each search must target a sub-question

3. **remember**: Store important facts for later

4. **recall**: Retrieve previously stored facts

5. **progress_report**: Assess your progress (REQUIRED every {interval} iterations)

6. **synthesize**: Provide final answer (ONLY when can_synthesize is YES)

## RULES

1. You MUST complete at least {min_searches} searches before synthesizing
2. You CANNOT synthesize until "Can synthesize" shows YES
3. Your final response MUST address the original query
4. Stay focused on the sub-questions - don't go on tangents
5. If progress_report is required, you MUST submit one

## RECENT SEARCH RESULTS

{recent_results}

What action will you take next?"""

    PROGRESS_PROMPT = """You are assessing your research progress.

CURRENT TIMESTAMP: {timestamp}

{state_context}

## SEARCH RESULTS SO FAR

{all_results}

Provide a thorough progress report:
1. Status of each sub-question (has evidence? quality?)
2. Your overall confidence in answering the original query
3. Whether more research is needed
4. Whether you're still on track or drifting off-topic"""

    SELF_CRITIQUE_PROMPT = """You are a critical reviewer evaluating a research response.

## ORIGINAL QUERY
"{original_query}"

## RESEARCH PLAN
{plan_summary}

## PROPOSED RESPONSE
{response}

## EVIDENCE GATHERED
{evidence_summary}

Critically evaluate this response:
1. Does it actually answer the original question?
2. Are all REQUIRED sub-questions addressed?
3. Is evidence cited appropriately?
4. Are there any hallucinations (claims not supported by search results)?
5. Is the response complete?

Be honest and critical. If the response is inadequate, say so."""

    def __init__(self, config: DeepResearchConfig):
        self.config = config
        self.memory = ResearchMemory(config.db_path)

        base_client = OpenAI(
            base_url=f"{config.llamacpp_url}/v1",
            api_key="not-needed"
        )
        self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def research(self, query: str) -> str:
        """
        Execute the full research protocol on a query.
        Returns the final synthesized response.
        """
        # Initialize state with mode settings
        state = ResearchState(
            min_searches=self.config.mode_settings.min_searches,
            max_iterations=self.config.mode_settings.max_iterations,
            require_progress_reports=self.config.mode_settings.require_progress_reports,
            progress_report_interval=self.config.mode_settings.progress_report_interval,
            require_self_critique=self.config.mode_settings.require_self_critique,
        )

        # Phase 1: Planning
        print(f"\n{'='*60}")
        print(f"PHASE 1: PLANNING")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Mode: {self.config.mode.value}")

        plan = self._execute_planning_phase(query, state)
        if not plan:
            return "Error: Failed to create research plan"

        state.plan = plan
        state.phase = ResearchPhase.RESEARCHING

        print(f"\nPlan created with {len(plan.sub_questions)} sub-questions:")
        for i, sq in enumerate(plan.sub_questions):
            print(f"  [{i}] [{sq.priority.value}] {sq.question}")

        # Phase 2: Research Loop
        print(f"\n{'='*60}")
        print(f"PHASE 2: RESEARCHING")
        print(f"{'='*60}")

        messages = self._build_initial_messages(state)

        while state.iterations_completed < state.max_iterations:
            state.iterations_completed += 1
            print(f"\n--- Iteration {state.iterations_completed}/{state.max_iterations} ---")

            # Check if progress report is required
            if state.needs_progress_report():
                print("[Progress report required]")
                report = self._execute_progress_report(state)
                if report:
                    state.progress_reports.append(report)
                    if not report.should_continue_researching and state.can_synthesize():
                        print("[Progress report indicates ready to synthesize]")
                        break
                continue

            # Get next action
            action_result = self._get_next_action(state, messages)
            if not action_result:
                print("[Failed to get valid action, retrying...]")
                continue

            action = action_result.action

            # Handle different action types
            if isinstance(action, SearchAction):
                self._handle_search(state, action, messages)

            elif isinstance(action, ParallelSearchAction):
                self._handle_parallel_search(state, action, messages)

            elif isinstance(action, RememberAction):
                self._handle_remember(state, action, messages)

            elif isinstance(action, RecallAction):
                self._handle_recall(state, action, messages)

            elif isinstance(action, ProgressReport):
                state.progress_reports.append(action)
                print(f"[Progress: confidence={action.overall_confidence:.0%}, continue={action.should_continue_researching}]")
                if not action.should_continue_researching and state.can_synthesize():
                    break

            elif isinstance(action, SynthesisAction):
                if not state.can_synthesize():
                    print(f"[Synthesis blocked: need {state.min_searches - state.searches_completed} more searches]")
                    messages.append({
                        "role": "user",
                        "content": f"You cannot synthesize yet. You need at least {state.min_searches} searches (currently {state.searches_completed}). Continue researching."
                    })
                    continue

                # Phase 3: Synthesis with optional self-critique
                print(f"\n{'='*60}")
                print(f"PHASE 3: SYNTHESIS")
                print(f"{'='*60}")

                state.phase = ResearchPhase.SYNTHESIZING

                if state.require_self_critique:
                    critique = self._execute_self_critique(state, action)
                    if critique and critique.recommended_action == "continue_research":
                        print(f"[Self-critique rejected response: {critique.critique_notes}]")
                        state.phase = ResearchPhase.RESEARCHING
                        messages.append({
                            "role": "user",
                            "content": f"Your response was critiqued and needs improvement: {critique.critique_notes}\n\nContinue researching to address these issues."
                        })
                        continue

                # Save and return
                self.memory.save_research_session(state, action.final_response)

                print(f"\nResearch complete!")
                print(f"  Searches: {state.searches_completed}")
                print(f"  Iterations: {state.iterations_completed}")
                print(f"  Sub-questions addressed: {action.subquestions_addressed}")

                return action.final_response

        # Max iterations reached - force synthesis
        print(f"\n[Max iterations reached, forcing synthesis]")
        return self._force_synthesis(state, messages)

    def _execute_planning_phase(self, query: str, state: ResearchState) -> Optional[ResearchPlan]:
        """Execute the planning phase to decompose the query."""
        prompt = self.PLANNING_PROMPT.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query=query,
            max_subquestions=self.config.mode_settings.max_subquestions
        )

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                response_model=PlanningResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                max_retries=self.config.max_retries
            )
            return response.plan
        except Exception as e:
            print(f"Planning failed: {e}")
            return None

    def _build_initial_messages(self, state: ResearchState) -> list[dict]:
        """Build the initial message list for the research phase."""
        # Get context from memory
        top_facts = self.memory.get_top_facts(10)
        facts_context = ""
        if top_facts:
            facts_context = "\n## KNOWN FACTS\n" + "\n".join(
                f"- {f['subject']} {f['predicate']} {f['object']}"
                for f in top_facts
            )

        system_prompt = self.RESEARCH_PROMPT.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state_context=state.to_context_string(),
            interval=state.progress_report_interval,
            min_searches=state.min_searches,
            recent_results="No searches yet."
        ) + facts_context

        return [{"role": "system", "content": system_prompt}]

    def _get_next_action(self, state: ResearchState, messages: list[dict]) -> Optional[ResearchResponse]:
        """Get the next action from the LLM."""
        # Update system message with current state
        recent_results = "No searches yet." if not state.search_results else "\n".join(
            f"- [{r.get('query', 'unknown')}] {r.get('title', 'No title')}: {r.get('content', '')[:200]}..."
            for r in state.search_results[-5:]
        )

        messages[0] = {
            "role": "system",
            "content": self.RESEARCH_PROMPT.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                state_context=state.to_context_string(),
                interval=state.progress_report_interval,
                min_searches=state.min_searches,
                recent_results=recent_results
            )
        }

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                response_model=ResearchResponse,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                max_retries=self.config.max_retries
            )
            return response
        except Exception as e:
            print(f"Failed to get action: {e}")
            return None

    def _handle_search(self, state: ResearchState, action: SearchAction, messages: list[dict]):
        """Handle a single search action."""
        print(f"[Search: {action.query}] (targets sub-question {action.targets_subquestion})")

        results = search_web(self.config, action.query, self.memory)
        state.searches_completed += 1

        if results and "error" not in results[0]:
            state.search_results.extend(results)

            # Track evidence by sub-question
            if action.targets_subquestion not in state.evidence_by_subquestion:
                state.evidence_by_subquestion[action.targets_subquestion] = []
            state.evidence_by_subquestion[action.targets_subquestion].extend(
                [r.get("content", "")[:500] for r in results if r.get("content")]
            )

            results_text = "\n".join(
                f"{i+1}. {r['title']}\n   {r['content']}"
                for i, r in enumerate(results)
            )
            messages.append({"role": "assistant", "content": f"[Searched: {action.query}]"})
            messages.append({"role": "user", "content": f"Search results:\n{results_text}"})
        else:
            error = results[0].get("error", "Unknown error") if results else "No results"
            print(f"  Search failed: {error}")
            messages.append({"role": "assistant", "content": f"[Search failed: {error}]"})
            messages.append({"role": "user", "content": f"Search failed. Try a different query."})

    def _handle_parallel_search(self, state: ResearchState, action: ParallelSearchAction, messages: list[dict]):
        """Handle parallel search action."""
        queries = [s.query for s in action.searches]
        print(f"[Parallel search: {len(queries)} queries]")
        for s in action.searches:
            print(f"  - {s.query} (targets sub-question {s.targets_subquestion})")

        all_results = execute_parallel_searches(self.config, queries, self.memory)

        combined_text = []
        for i, (search_action, results) in enumerate(zip(action.searches, all_results)):
            state.searches_completed += 1

            if results and "error" not in results[0]:
                state.search_results.extend(results)

                # Track evidence
                sq_idx = search_action.targets_subquestion
                if sq_idx not in state.evidence_by_subquestion:
                    state.evidence_by_subquestion[sq_idx] = []
                state.evidence_by_subquestion[sq_idx].extend(
                    [r.get("content", "")[:500] for r in results if r.get("content")]
                )

                combined_text.append(f"\n### Results for: {search_action.query}")
                combined_text.extend(
                    f"{j+1}. {r['title']}: {r['content']}"
                    for j, r in enumerate(results)
                )

        messages.append({"role": "assistant", "content": f"[Executed {len(queries)} parallel searches]"})
        messages.append({"role": "user", "content": "\n".join(combined_text)})

    def _handle_remember(self, state: ResearchState, action: RememberAction, messages: list[dict]):
        """Handle remember action."""
        print(f"[Remember: {action.fact[:50]}...]")

        facts = extract_facts(self.client, action.fact)
        for f in facts:
            self.memory.add_fact(f.subject, f.predicate, f.object, source="research")

        if not facts:
            self.memory.add_fact("info", "is", action.fact, source="research")

        messages.append({"role": "assistant", "content": f"[Remembered: {action.fact}]"})
        messages.append({"role": "user", "content": "Fact stored. Continue with your research."})

    def _handle_recall(self, state: ResearchState, action: RecallAction, messages: list[dict]):
        """Handle recall action."""
        print(f"[Recall: {action.topic}]")

        facts = self.memory.search_facts(action.topic)

        if facts:
            facts_text = "\n".join(
                f"- {f['subject']} {f['predicate']} {f['object']}"
                for f in facts
            )
            messages.append({"role": "assistant", "content": f"[Recalled: {action.topic}]"})
            messages.append({"role": "user", "content": f"Recalled facts:\n{facts_text}"})
        else:
            messages.append({"role": "assistant", "content": f"[No memories for: {action.topic}]"})
            messages.append({"role": "user", "content": "No stored facts found. Continue with web search."})

    def _execute_progress_report(self, state: ResearchState) -> Optional[ProgressReport]:
        """Force a progress report."""
        all_results = "\n".join(
            f"- [{r.get('query', '?')}] {r.get('title', 'No title')}: {r.get('content', '')[:200]}"
            for r in state.search_results
        ) if state.search_results else "No search results yet."

        prompt = self.PROGRESS_PROMPT.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state_context=state.to_context_string(),
            all_results=all_results
        )

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                response_model=ProgressReportResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500,
                max_retries=self.config.max_retries
            )
            return response.report
        except Exception as e:
            print(f"Progress report failed: {e}")
            return None

    def _execute_self_critique(self, state: ResearchState, synthesis: SynthesisAction) -> Optional[SelfCritique]:
        """Execute self-critique on the proposed response."""
        plan_summary = "\n".join(
            f"- [{sq.priority.value}] {sq.question}"
            for sq in state.plan.sub_questions
        ) if state.plan else "No plan"

        evidence_summary = "\n".join(
            f"Sub-question {i}: {len(evidence)} pieces of evidence"
            for i, evidence in state.evidence_by_subquestion.items()
        )

        prompt = self.SELF_CRITIQUE_PROMPT.format(
            original_query=state.plan.original_query if state.plan else "Unknown",
            plan_summary=plan_summary,
            response=synthesis.final_response,
            evidence_summary=evidence_summary
        )

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                response_model=SelfCritiqueResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                max_retries=self.config.max_retries
            )

            print(f"[Self-critique: {response.critique.overall_quality}, recommends: {response.critique.recommended_action}]")
            return response.critique
        except Exception as e:
            print(f"Self-critique failed: {e}")
            return None

    def _force_synthesis(self, state: ResearchState, messages: list[dict]) -> str:
        """Force synthesis when max iterations reached."""
        prompt = f"""Maximum iterations reached. You MUST provide a final response now.

{state.to_context_string()}

Based on all the research conducted, provide your best answer to the original query.
Even if incomplete, synthesize what you have learned."""

        messages.append({"role": "user", "content": prompt})

        try:
            # Try to get a proper synthesis
            response = self.client.chat.completions.create(
                model="local-model",
                response_model=ResearchResponse,
                messages=messages,
                temperature=0.7,
                max_tokens=3000,
                max_retries=self.config.max_retries
            )

            if isinstance(response.action, SynthesisAction):
                self.memory.save_research_session(state, response.action.final_response)
                return response.action.final_response
        except Exception:
            pass

        # Fallback: just get a plain response
        base_client = OpenAI(
            base_url=f"{self.config.llamacpp_url}/v1",
            api_key="not-needed"
        )

        try:
            response = base_client.chat.completions.create(
                model="local-model",
                messages=messages + [{"role": "user", "content": "Provide your final answer now, summarizing what you learned."}],
                temperature=0.7,
                max_tokens=2000
            )
            answer = response.choices[0].message.content
            self.memory.save_research_session(state, answer)
            return answer
        except Exception as e:
            return f"Research incomplete. Error: {e}"


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Deep Research Protocol - Thorough agentic search with task anchoring"
    )
    parser.add_argument("--llm", default="http://localhost:8080", help="llama.cpp URL")
    parser.add_argument("--searxng", default="http://localhost:8888", help="SearXNG URL")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "thorough", "exhaustive"],
        default="standard",
        help="Research depth mode"
    )
    parser.add_argument("--retries", type=int, default=3, help="Max retries on validation failure")
    parser.add_argument("query", nargs="?", help="Research query (or enter interactive mode)")
    args = parser.parse_args()

    config = DeepResearchConfig(
        llamacpp_url=args.llm,
        searxng_url=args.searxng,
        max_retries=args.retries,
        mode=ResearchMode(args.mode)
    )

    engine = DeepResearchEngine(config)

    print("=" * 60)
    print("Deep Research Protocol")
    print("=" * 60)
    print(f"LLM: {config.llamacpp_url}")
    print(f"SearXNG: {config.searxng_url}")
    print(f"Mode: {config.mode.value} (min {config.mode_settings.min_searches} searches, max {config.mode_settings.max_iterations} iterations)")
    print("=" * 60)

    if args.query:
        # Single query mode
        result = engine.research(args.query)
        print("\n" + "=" * 60)
        print("FINAL RESPONSE")
        print("=" * 60)
        print(result)
    else:
        # Interactive mode
        print("\nEnter your research questions (or 'quit' to exit)")
        print("Use /mode <quick|standard|thorough|exhaustive> to change depth")
        print("-" * 60)

        while True:
            try:
                query = input("\nResearch query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue

            if query.lower() == "quit":
                break

            if query.startswith("/mode "):
                mode_name = query.split(" ", 1)[1].strip().lower()
                try:
                    config.mode = ResearchMode(mode_name)
                    config.mode_settings = ResearchModeSettings.for_mode(config.mode)
                    print(f"Mode changed to: {config.mode.value}")
                    print(f"  Min searches: {config.mode_settings.min_searches}")
                    print(f"  Max iterations: {config.mode_settings.max_iterations}")
                except ValueError:
                    print(f"Unknown mode: {mode_name}")
                continue

            result = engine.research(query)
            print("\n" + "=" * 60)
            print("FINAL RESPONSE")
            print("=" * 60)
            print(result)


if __name__ == "__main__":
    main()
