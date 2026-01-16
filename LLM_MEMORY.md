# LLM Memory Bridge - Quick Start Guide

A Python bridge that gives your local LLM (via llama.cpp) persistent memory and web search capabilities.

## What It Does

- **Web Search**: Your LLM can search the internet via your local SearXNG instance
- **Fact Memory**: Automatically extracts and stores facts from conversations
- **Recall**: Retrieves relevant memories when needed
- **Salience Tracking**: Important/frequently-accessed facts bubble up; unused ones fade

---

## Prerequisites

1. **llama.cpp server running** with your model loaded
2. **SearXNG + Valkey running** (from the Docker setup)
3. **Python 3.10+**

---

## Installation

### Minimal Setup (SQLite only)

```powershell
pip install openai requests
```

### Full Setup (with vector memory)

```powershell
pip install openai requests chromadb

# Optional: Better embeddings (downloads ~90MB model)
pip install sentence-transformers
```

---

## Running

### Start your services first

```powershell
# Terminal 1: Start llama.cpp server (adjust path/model as needed)
llama-server -m D:\containers\sear-xng\gemma-3-12b-it-abliterated-q8_0.gguf -c 16384 --port 8080

# Terminal 2: Start SearXNG (if not already running)
cd D:\containers\sear-xng
docker compose up -d
```

### Run the memory bridge

**Simple version** (SQLite only):

```powershell
python simple_llm_search.py --llm http://localhost:8080 --searxng http://localhost:8888
```

**Full version** (with vector memory):

```powershell
python llm_memory_bridge.py --llm-url http://localhost:8080 --searxng-url http://localhost:8888
```

---

## How to Use

Once running, just chat normally. The LLM has been instructed to use special tags when it needs to search or remember things.

### Automatic Behaviors

The LLM will automatically:

- Use `<search>` when it needs current information
- Use `<remember>` when you tell it something important
- Use `<recall>` when it needs to retrieve past information

You'll see indicators when these happen:

```
[🔍 Searching: current weather in Denver]
[💾 Remembering: user prefers Python over JavaScript]
[🧠 Recalling: user preferences]
```

### Example Conversation

```
You: My name is Jake and I live in Fort Collins.

[💾 Remembering: user's name is Jake, lives in Fort Collins]
