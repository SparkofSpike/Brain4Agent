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

## A2A (Agent2Agent)

A2A 1.0 models every interaction as a message to an agent. A memory lookup is the
direct-answer case, so the server returns a Message and never creates a Task.

`sh
python integrations/a2a/server.py --port 8080 --db brain.db
curl -s localhost:8080/.well-known/agent-card.json
`

`json
{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {"message": {
    "role": "ROLE_USER",
    "parts": [{"text": "recall: 中文查询的 bug"}]
  }}
}
`

The first text part carries a one-word prefix: 
ecall: searches, 
emember:
stores, and a bare message is treated as 
ecall:.

**Implemented:** agent card, SendMessage.

**Not implemented:** streaming, task lifecycle (GetTask, ListTasks,
CancelTask), push notifications, auth, TLS. A memory read finishes in
milliseconds; wrapping it in a task lifecycle would be ceremony that buys
nothing. If you need task semantics you want an agent, not a memory store.

Bind address defaults to loopback. Put a reverse proxy in front if anything
outside this machine should reach it.

## Which one to use

| | MCP | skill | A2A |
|---|---|---|---|
| Setup | edit config, restart client | copy a directory | run a process |
| Invocation | tool call, always available | agent reads and decides | HTTP message |
| Token cost | tool schemas always in context | skill body only when loaded | none until called |
| Cross-language | needs an MCP client | needs an agent | any HTTP client |
| Best for | frequent use in one client | occasional use, many clients | another agent or service |
