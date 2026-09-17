# YOLO+SAM Research Agent Guide

This project is configured to use an ARIS-style research workflow on top of the existing `YOLO+SAM` codebase.

## Scope

- Treat this repository as an active medical image segmentation research project.
- Prefer project-local skills under `.agents/skills/` when they are installed.
- Do not modify the original dataset structure unless the task explicitly asks for a new isolated variant.
- Keep `TSRS_RSNA-Epiphysis` and `TSRS_RSNA-Articular-Surface` experiments, outputs, and reports separated.

## Project Layout That Matters

- `scripts/`: training, inference, evaluation, analysis, and dataset-variant utilities
- `configs/`: dataset configs and filtering manifests
- `outputs/`: experiment outputs and analysis artifacts
- `run_*.sh`: reproducible Linux-side experiment launchers
- `research-workflow/`: ARIS-oriented research workspace for plans, briefs, logs, and summaries

## Recommended ARIS Entry Points

After installing ARIS Codex skills into this project, the most useful commands for this repo are:

- `/research-wiki init`
- `/research-lit "bone age epiphysis segmentation"`
- `/idea-discovery "improve YOLO+SAM epiphysis segmentation beyond current refiner baselines"`
- `/experiment-plan`
- `/experiment-bridge`
- `/auto-review-loop`
- `/research-pipeline "new segmentation direction"`

## Project-Specific Research Rules

- Prefer reading current experiment records from `outputs/` before proposing a new direction.
- Any new experiment should define:
  - target dataset
  - baseline to beat
  - exact metrics
  - output folder isolation
- When proposing code changes, anchor them to existing entry scripts in `scripts/` and existing launcher conventions in `run_*.sh`.
- When proposing dataset cleaning or filtering, create new manifests and new dataset variant directories rather than editing the original raw dataset in place.

## GPU / Server Assumptions

- Main remote workspace commonly used for experiments:
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- Common environment:
  - `conda activate yolo-sam-gpu`
- If a workflow proposes deployment, it should reuse the project's existing shell launcher style where possible.

## Research Workspace

Use the following files first when available:

- `research-workflow/RESEARCH_BRIEF.md`
- `research-workflow/refine-logs/EXPERIMENT_PLAN.md`
- `research-workflow/NARRATIVE_REPORT.md`

If `research-wiki/` exists, treat it as persistent project memory and keep new papers, ideas, experiments, and claims synchronized there.
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS skills installed in this project: 80 entries.
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `G:\gutou\YOLO+SAM\.tmp\aris_repo`
Project skill path: `.agents/skills/<skill-name>`
For ARIS workflows, prefer the project-local skills under `.agents/skills/`.
Do not edit or delete junctioned skills in place; update upstream or rerun:
`powershell -NoProfile -ExecutionPolicy Bypass -File "G:\gutou\YOLO+SAM\.tmp\aris_repo\tools\install_aris.ps1" "G:\gutou\YOLO+SAM" -Platform codex -Reconcile`
<!-- ARIS-CODEX:END -->
