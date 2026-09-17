# Workflow 2 Autorun Prompt

Continue the autonomous ARIS research loop for this repository.

Read current state from these files first:

- `research-workflow/RESEARCH_BRIEF.md`
- `research-workflow/refine-logs/AUTONOMOUS_LOOP_STATUS.md`
- `research-workflow/refine-logs/EXPERIMENT_TRACKER.md`
- `research-workflow/refine-logs/EXPERIMENT_RESULTS.md`
- `research-workflow/refine-logs/NEXT_CANDIDATE_REPAIR_PLAN.md`

Then do the next best action toward the real project objective:

- match or beat ARAA on `TSRS_RSNA-Epiphysis_clean_test_v2`
- preserve thin epiphysis and narrow gap quality
- keep experiment outputs isolated and documented

Execution policy for this autorun pass:

1. If a remote experiment is still running, monitor it and collect any finished results.
2. If a run has finished, update the ARIS files with the verdict and evidence.
3. If the current decision gate says to launch the next candidate, implement any needed code changes and launch it.
4. Keep all state persisted into the repo so the next non-interactive pass can continue from files, not memory.
5. If blocked by missing external state, record the blocker precisely and leave the loop in a resumable state.

Do not stop at analysis if a concrete next action can be taken.
Do not invent a new research direction unless the current loop has clearly failed and the repo state supports a pivot.

At the end of this pass:

- write or update the relevant `research-workflow/refine-logs/*.md` files
- keep `MANIFEST.md` in sync
- summarize what changed and what the next autorun pass should do
