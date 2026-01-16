# SearXNG + Valkey Setup for MCP Tool Calls

This configuration sets up a local SearXNG instance with Valkey (Redis fork) for use with MCP (Model Context Protocol) tool calls in Claude Desktop, Claude Code, Cursor, or other MCP-compatible clients.

## Prerequisites

1. **Docker Desktop for Windows** - [Download here](https://www.docker.com/products/docker-desktop/)
2. **Node.js** (for npx) - [Download here](https://nodejs.org/) (LTS recommended)

## Quick Start

### 1. Generate a Secure Secret Key

Open PowerShell in this directory and run:

```powershell
.\setup.ps1 -GenerateSecretKey
```

Or manually generate and replace the secret key in `searxng/settings.yml`:

```powershell
$randomBytes = New-Object byte[] 32
(New-Object Security.Cryptography.RNGCryptoServiceProvider).GetBytes($randomBytes)
$secretKey = -join ($randomBytes | ForEach-Object { "{0:x2}" -f $_ })
Write-Output $secretKey
```

### 2. Start the Containers

```powershell
.\setup.ps1 -Start
# Or directly:
docker compose up -d
```

### 3. Verify It's Working

```powershell
.\setup.ps1 -Test
# Or manually test the JSON API:
curl "http://localhost:8080/search?q=test&format=json"
```

You should see JSON search results returned.

### 4. Configure Your MCP Client

#### For Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "searxng": {
      "command": "npx",
      "args": ["-y", "mcp-searxng"],
      "env": {
        "SEARXNG_URL": "http://localhost:8080"
      }
    }
  }
}
```

Then restart Claude Desktop.

#### For Claude Code

```bash
claude mcp add --transport stdio searxng -- npx -y mcp-searxng
# Set the environment variable
set SEARXNG_URL=http://localhost:8080
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "searxng": {
      "command": "npx",
      "args": ["-y", "mcp-searxng"],
      "env": {
        "SEARXNG_URL": "http://localhost:8080"
      }
    }
  }
}
```

## Directory Structure

```
searxng-mcp/
├── docker-compose.yml      # Docker services configuration
├── setup.ps1               # PowerShell management script
├── mcp-config-examples.json # MCP client configuration examples
├── README.md               # This file
└── searxng/
    └── settings.yml        # SearXNG configuration (JSON API enabled)
```

## Configuration Details

### SearXNG Settings (`searxng/settings.yml`)

Key configurations for MCP usage:

- **JSON format enabled**: Required for API access
- **Rate limiting disabled**: For local development use
- **Multiple search engines**: Google, DuckDuckGo, Bing, Brave, Wikipedia, GitHub, Stack Overflow, arXiv, Google Scholar

### Valkey (Redis Fork)

- Provides caching for search results
- Persistence enabled (saves every 30 seconds if data changed)
- Health checks ensure SearXNG waits for Valkey to be ready

## Management Commands

```powershell
# Start containers
.\setup.ps1 -Start

# Stop containers
.\setup.ps1 -Stop

# Restart containers
.\setup.ps1 -Restart

# View logs
.\setup.ps1 -Logs

# Test API
.\setup.ps1 -Test

# Check status
.\setup.ps1 -Status
```

Or use Docker Compose directly:

```powershell
docker compose up -d        # Start
docker compose down         # Stop
docker compose logs -f      # Follow logs
docker compose ps           # Status
```

## API Usage

### Search Endpoint

```
GET http://localhost:8080/search?q=YOUR_QUERY&format=json
```

Parameters:
- `q` - Search query (required)
- `format` - Output format: `html` or `json` (use `json` for MCP)
- `categories` - Comma-separated: `general`, `images`, `news`, `videos`, `it`, `science`
- `engines` - Comma-separated engine names: `google`, `bing`, `duckduckgo`
- `language` - Language code: `en`, `es`, `de`, etc.
- `time_range` - Filter by time: `day`, `week`, `month`, `year`
- `safesearch` - Safe search level: `0` (off), `1` (moderate), `2` (strict)

### Example API Calls

```powershell
# Basic search
curl "http://localhost:8080/search?q=python+programming&format=json"

# Search with time filter
curl "http://localhost:8080/search?q=AI+news&format=json&time_range=week"

# Search specific engines
curl "http://localhost:8080/search?q=docker+tutorial&format=json&engines=google,stackoverflow"

# Search with language
curl "http://localhost:8080/search?q=machine+learning&format=json&language=en"
```

## Troubleshooting

### "Connection refused" errors

1. Ensure Docker Desktop is running
2. Check containers are up: `docker compose ps`
3. Verify port 8080 isn't used: `netstat -ano | findstr :8080`

### "403 Forbidden" on JSON endpoint

The JSON format isn't enabled. Verify `searxng/settings.yml` contains:

```yaml
search:
  formats:
    - html
    - json
```

Then restart: `docker compose restart searxng`

### Slow or no search results

Some search engines may rate-limit or block requests. Try:
1. Different search engines in the query
2. Adding delays between searches
3. Checking `docker compose logs searxng` for errors

### MCP server not connecting

1. Verify SearXNG is accessible: `curl http://localhost:8080/search?q=test&format=json`
2. Ensure Node.js/npx is installed and in PATH
3. Check the MCP client logs for errors
4. Try running the MCP server manually: `npx -y mcp-searxng`

## Security Notes

For production use:
1. Always generate a unique secret key
2. Consider enabling rate limiting (`limiter: true`)
3. Use a reverse proxy with HTTPS for external access
4. Restrict network access to trusted clients

## Available MCP Server Options

| Package | Install | Notes |
|---------|---------|-------|
| `mcp-searxng` | `npx -y mcp-searxng` | Most popular, TypeScript-based |
| `mcp-searxng` (Python) | `uvx mcp-searxng` | Python-based alternative |
| `@jharding_npm/mcp-server-searxng` | `npx -y @jharding_npm/mcp-server-searxng` | Enhanced error messages |
| `mcp-searxng-public` | `npx mcp-searxng-public` | Uses public instances (no local setup) |

## Resources

- [SearXNG Documentation](https://docs.searxng.org/)
- [SearXNG Docker Repository](https://github.com/searxng/searxng-docker)
- [Valkey Project](https://valkey.io/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [mcp-searxng GitHub](https://github.com/ihor-sokoliuk/mcp-searxng)
