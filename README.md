# SearXNG LLM Tools

Local LLM integration with web search and persistent memory via llama.cpp and SearXNG.

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| **Search with Memory** | Chat with web search and SQLite & Vector memory | [MEMORY.md](MEMORY.md) |
| **Deep Research** | Multi-phase agentic research with task anchoring | [DEEP_RESEARCH.md](DEEP_RESEARCH.md) |

## Prerequisites

- llama.cpp server running with your model
- SearXNG + Valkey via Docker
- Python 3.10+

## Quick Install

```bash
pip install openai requests instructor pydantic
```

## Quick Start

```bash
# Simple conversational search
python simple_llm_search.py  --llm http://localhost:8080 --searxng http://localhost:8888

# Full search with vector memory
python llm_memory_bridge.py --llm-url http://localhost:8080 --searxng-url http://localhost:8888

# Deep multi-step research
python deep_research.py --mode thorough "your research question"
```

## Services

```bash
# Start SearXNG
docker compose up -d

# Start llama.cpp (adjust model path)
llama-server -m /path/to/model.gguf --ctx-size 16384 --port 8080
```
