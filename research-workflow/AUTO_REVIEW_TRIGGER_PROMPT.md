# Auto Review Trigger Prompt

Use the **upstream ARIS skill flow** for Workflow 2.

Read these first:

- `.agents/skills/auto-review-loop/SKILL.md`
- `research-workflow/RESEARCH_BRIEF.md`
- `research-workflow/refine-logs/EXPERIMENT_PLAN.md`
- `research-workflow/refine-logs/EXPERIMENT_TRACKER.md`
- `research-workflow/refine-logs/EXPERIMENT_RESULTS.md`
- `research-workflow/refine-logs/AUTONOMOUS_LOOP_STATUS.md`

Then continue the project using the upstream `/auto-review-loop` logic, not a custom shortcut.

Important constraints:

1. Treat `experiment-bridge` as already finished for the current candidate set only if the tracker and results files support that.
2. Do not launch new bridge experiments in this pass unless the review loop explicitly routes back to implementation.
3. Create and use `review-stage/` artifacts according to the upstream skill definition.
4. Keep `MANIFEST.md` synchronized.

Current intent of this trigger:

- The monitored bridge candidate has completed.
- The next action should be the upstream ARIS Workflow 2 review loop over the current results and project state.

At the end of this pass:

- update `review-stage/AUTO_REVIEW.md`
- update `review-stage/REVIEW_STATE.json`
- summarize whether the review loop ended in `ready`, `almost`, or `not ready`
