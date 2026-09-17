# Research Workflow

This directory is the project-local ARIS workspace for `YOLO+SAM`.

## What goes here

- `RESEARCH_BRIEF.md`: current research direction and constraints
- `refine-logs/EXPERIMENT_PLAN.md`: executable experiment plan
- `NARRATIVE_REPORT.md`: writing handoff after experiments
- future ARIS outputs such as idea reports, review logs, and pipeline summaries

## Recommended startup

1. Install ARIS Codex skills into this project:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_aris_yolosam.ps1
```

2. Initialize research memory in the project root:

```text
/research-wiki init
```

3. Fill in:

- `research-workflow/RESEARCH_BRIEF.md`
- `research-workflow/refine-logs/EXPERIMENT_PLAN.md`

4. Then use one of:

```text
/research-pipeline "improve YOLO+SAM epiphysis segmentation"
/idea-discovery "new direction for boundary-aware epiphysis refinement"
/experiment-bridge "research-workflow/refine-logs/EXPERIMENT_PLAN.md"
```

## Project conventions

- New ideas should compare against the current strongest baseline already present in `outputs/`.
- New training or evaluation runs must use isolated output folders.
- Dataset filtering and relabel-cleaning experiments must use separate manifests and separate dataset variants.
