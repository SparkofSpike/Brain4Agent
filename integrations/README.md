# Integrations

Ways to connect Brain4Agent to an agent runtime. All of them reuse the same
library; none of them adds a dependency to the core.

## MCP server

The default integration. Any MCP client can load it:

```json
{"mcpServers": {"brain4agent": {
    "command": "python",
    "args": ["-m", "brain4agent.mcp_server"],
    "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
}}}
```

Tools: `b4a_remember`, `b4a_recall`, `b4a_session`, `b4a_stats`. The prefix
keeps them from colliding with another memory server in the same client.

Requires `pip install -e ".[precise]"` (or `pip install Brain4Agent[precise]`
once published).

## CodeWhale / Claude Code skill

A skill carries no protocol — it is a document the agent reads and then calls the
CLI. Useful when changing `mcp.json` and restarting the client is not worth it,
or when the agent should decide *when* to consult memory rather than having the
tools always present.

```sh
# personal skills
cp -r integrations/skills/brain4agent ~/.agents/skills/

# or project-scoped
cp -r integrations/skills/brain4agent .agents/skills/
```

The skill assumes `brain4agent` is importable. Install it first:

```sh
pip install -e ".[precise]"
```

Both paths are the same underlying code, so a fix in `src/` reaches both.

## Which one to use

| | MCP | skill |
|---|---|---|
| Setup | edit config, restart client | copy a directory |
| Invocation | tool call, always available | agent reads and decides |
| Token cost | tool schemas always in context | skill body only when loaded |
| Best for | frequent use in one client | occasional use, many clients |
