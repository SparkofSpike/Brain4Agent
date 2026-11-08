# Hook: prepend relevant memory to a user prompt before the model sees it

MCP is **pull-only**. A server cannot volunteer a memory, so an agent only sees
one when it decides to call `b4a_recall`. That covers "the user asked about
something from before" and misses "the task is about to touch something we have
context on".

Closing that gap is a client-side job: run this before handing the prompt to the
model, and prepend its output. This file is the Windows wrapper that makes the
piping survive PowerShell's encoding.

```
$prompt | .\recall-before-prompt.ps1
```

Environment:

- `BRAIN4AGENT_DB` — path to the memory file. Unset means the hook does nothing.

Behaviour worth knowing:

- **Silent when nothing is relevant.** That is the common case, and a hook that
  prints a header on every turn trains the model to skip the block.
- **Never blocks.** Retrieval is BM25, single-digit milliseconds. If the memory
  file is missing the hook exits 0 without output.
- **Never decides.** It injects evidence with a coverage number and a warning
  that it may be about a different task. Whether to use it is the model's call.

The `$OutputEncoding` lines are not decoration. PowerShell 5.1 pipes strings as
ANSI, which turns Chinese into `?` before Python ever sees it — measured, the
byte stream was `3f3f3f...`. Without those two lines this hook silently matches
nothing on a Chinese prompt.
