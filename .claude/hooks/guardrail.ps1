<#
  guardrail.ps1 — Claude Code hook for the autonomous loop (LOOP.md).

  Two jobs, selected by -HookEvent:
    PreToolUse   : hard-deny actions the agent must never take unattended —
                   `git push` (the user owns pushes + CI) and any write under legacy/ (read-only).
    SessionStart : re-inject the loop frontier (open [!]-blocked tasks + a pointer to LOOP.md)
                   so the "what task am I on" thread survives auto-compaction / resume.

  Reads the hook payload as JSON on stdin; emits a JSON decision on stdout; exit 0 always
  (a hook error must never silently corrupt a decision — on any parse failure we allow).
#>
param([string]$HookEvent = "PreToolUse")

$raw = [Console]::In.ReadToEnd()
try { $payload = $raw | ConvertFrom-Json } catch { exit 0 }  # malformed input -> do not block

function Deny([string]$reason) {
    $out = @{
        hookSpecificOutput = @{
            hookEventName            = "PreToolUse"
            permissionDecision       = "deny"
            permissionDecisionReason = $reason
        }
    }
    Write-Output ($out | ConvertTo-Json -Depth 6 -Compress)
    exit 0
}

if ($HookEvent -eq "PreToolUse") {
    $tool = [string]$payload.tool_name
    $cmd  = [string]$payload.tool_input.command
    $path = [string]$payload.tool_input.file_path

    # 1) Never push — the user owns pushes and CI verification (CLAUDE.md, LOOP.md).
    if ($cmd -match '(?i)\bgit\b[^\n;|&]*\bpush\b') {
        Deny "Guardrail: 'git push' is blocked - the user owns pushes and CI. Commit locally and continue; surface the push at the phase boundary (LOOP.md)."
    }

    # 2) legacy/ is read-only (CLAUDE.md doc map).
    if ($tool -eq "Edit" -or $tool -eq "Write" -or $tool -eq "NotebookEdit") {
        $p = ($path -replace '\\', '/')
        if ($p -match '(?i)(^|/)legacy/') {
            Deny "Guardrail: legacy/ is read-only (CLAUDE.md). Do not modify legacy artifacts."
        }
    }
    exit 0
}

if ($HookEvent -eq "SessionStart") {
    $tasks = Join-Path (Get-Location) "TASKS.md"
    $blocked = @()
    if (Test-Path $tasks) {
        # Only real blocked task headers (e.g. "### T3.1 ... `[!]`"), not the legend line.
        $blocked = Select-String -Path $tasks -Pattern '^\s*###\s+T[0-9X].*\[!\]' | ForEach-Object { $_.Line.Trim() }
    }
    $blockedText = if ($blocked.Count) { ($blocked -join "`n") } else { "none" }
    $ctx = @"
AUTONOMOUS LOOP CONTEXT (injected by .claude/hooks/guardrail.ps1).
If a /loop is active, follow LOOP.md: re-read TASKS.md and select the next dependency-satisfied [ ] task that is not [!]-blocked; re-read HANDOFF.md for open human blockers; STOP and ask on any design call; run the full local gate before each local commit; pause at phase boundaries.
Currently [!]-blocked tasks in TASKS.md:
$blockedText
"@
    $out = @{
        hookSpecificOutput = @{
            hookEventName     = "SessionStart"
            additionalContext = $ctx
        }
    }
    Write-Output ($out | ConvertTo-Json -Depth 6 -Compress)
    exit 0
}

exit 0
