## Running alongside another memory server

**Do not enable two memory MCP servers in the same agent.** Measured (D-031):
with `neural-memory` and `b4a` both enabled, an agent asked about code-style
conventions queried the other server, got nothing, and reported "memory has no
record of this" — while five relevant memories sat in `b4a`. The store cannot
detect this from inside; it looks exactly like an empty memory.

A direct MCP handshake proved `b4a` was healthy, so the failure was at selection,
not retrieval. One memory server per agent, or names that make the choice
unambiguous.

