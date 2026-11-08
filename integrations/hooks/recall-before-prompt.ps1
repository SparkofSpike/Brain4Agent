# Hook: prepend relevant memory to a user prompt before the model sees it.
#
# MCP is pull-only -- brain4agent cannot volunteer a memory, so proactive recall
# has to happen client-side. Register this as a "user prompt submitted" hook in
# your agent client and prepend its output to the prompt.
#
#   $prompt | .\recall-before-prompt.ps1
#
# Env: BRAIN4AGENT_DB points at the memory file. Unset = the hook does nothing.
#
# Silent when nothing is relevant, which is the common case. Silent is correct:
# a block on every turn trains the model to skip it.

$ErrorActionPreference = 'Stop'

# PowerShell 5.1 pipes strings as ANSI, which turns CJK into '?' before Python
# ever sees it -- measured, the byte stream was 3f3f3f... Without these two lines
# a Chinese prompt matches nothing and the hook looks broken rather than silent.
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:BRAIN4AGENT_DB) { exit 0 }

$prompt = [Console]::In.ReadToEnd()
if (-not $prompt -or -not $prompt.Trim()) { exit 0 }

$prompt | python -X utf8 -m brain4agent --db $env:BRAIN4AGENT_DB recall-context
