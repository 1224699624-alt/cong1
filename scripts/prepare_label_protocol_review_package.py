#!/usr/bin/env python3
"""Prepare a reviewer-facing work package for R134 label/protocol review.

This script is read-only with respect to the manifest decisions. It creates
focused CSV/Markdown review aids so a human can fill enough train decisions to
pass the later isolated-variant gate without using clean-test-v2 as training or
tuning signal.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/r134_label_protocol_review_manifest/review_package")
CONFIRMED_DECISIONS = {"label_ok_protocol_clear", "label_needs_correction"}
DECISION_LABELS = {
    "": "空白，暂不决定",
    "label_ok_protocol_clear": "标签可用，协议清楚",
    "label_boundary_ambiguous": "边界不清，需要谨慎",
    "label_needs_correction": "标签需要修正",
    "exclude_from_training_variant": "排除，不放入训练变体",
    "needs_second_review": "需要二次复核",
}
METRIC_LABELS = {
    "components": "连通域",
    "fg frac": "前景占比",
    "gap": "最近间距",
    "contrast": "前景背景对比",
    "dice": "Dice",
    "boundary": "边界 IoU",
    "component delta": "连通域差值",
    "score": "难例分数",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare R134 human review package.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-reviewed-train", type=int, default=20)
    parser.add_argument("--min-confirmed-train", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed(row: dict[str, Any]) -> bool:
    return bool(str(row.get("human_decision", "")).strip())


def confirmed(row: dict[str, Any]) -> bool:
    return str(row.get("human_decision", "")).strip() in CONFIRMED_DECISIONS


def row_sort_key(row: dict[str, Any]) -> tuple[int, float, int]:
    priority = str(row.get("priority", ""))
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "reference_only": 9}.get(priority, 8)
    score = row.get("hardcase_similarity_score", 999)
    try:
        score_float = float(score)
    except (TypeError, ValueError):
        score_float = 999.0
    return priority_rank, score_float, int(row.get("rank", 9999))


def make_review_row(row: dict[str, Any], *, group: str) -> dict[str, Any]:
    return {
        "group": group,
        "image": row.get("image", ""),
        "split": row.get("split", ""),
        "priority": row.get("priority", ""),
        "primary_tag": row.get("primary_tag", ""),
        "rank": row.get("rank", ""),
        "hardcase_similarity_score": row.get("hardcase_similarity_score", ""),
        "component_count": row.get("component_count", ""),
        "fg_frac": row.get("fg_frac", ""),
        "nearest_center_gap": row.get("nearest_center_gap", ""),
        "fg_bg_contrast": row.get("fg_bg_contrast", ""),
        "panel": row.get("panel", ""),
        "review_action": row.get("review_action", ""),
        "allowed_use": row.get("allowed_use", ""),
        "human_decision": row.get("human_decision", ""),
        "human_notes": row.get("human_notes", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "image",
        "split",
        "priority",
        "primary_tag",
        "rank",
        "hardcase_similarity_score",
        "component_count",
        "fg_frac",
        "nearest_center_gap",
        "fg_bg_contrast",
        "panel",
        "review_action",
        "allowed_use",
        "human_decision",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def html_escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def decision_display(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return DECISION_LABELS.get(text, text or DECISION_LABELS[""])


def metric_display(value: str) -> str:
    return METRIC_LABELS.get(value, value)


def rel_asset(from_file: Path, asset_text: Any) -> str:
    asset = Path(str(asset_text))
    if not str(asset_text):
        return ""
    try:
        return Path("../../../../") / asset
    except TypeError:
        return asset


def md_table(rows: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    lines = [
        "| # | Image | Split | Priority | Tag | Panel | Suggested Check | Decision |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(rows[:limit] if limit else rows, start=1):
        panel = str(row.get("panel", "")).replace("\\", "/")
        panel_link = f"[panel](../../../../{panel})" if panel else ""
        lines.append(
            f"| {idx} | {row.get('image', '')} | {row.get('split', '')} | "
            f"`{row.get('priority', '')}` | `{row.get('primary_tag', '')}` | "
            f"{panel_link} | {row.get('review_action', '')} |  |"
        )
    return lines


def write_markdown(path: Path, payload: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    allowed = payload.get("human_decision_schema", {}).get("human_decision_allowed", [])
    train_rows = [row for row in review_rows if row["split"] == "train"]
    val_rows = [row for row in review_rows if row["split"] == "val"]
    clean_rows = [make_review_row(row, group="clean_test_reference") for row in payload.get("clean_test_reference", [])]
    lines = [
        "# R134 人工标签/协议审查包",
        "",
        "用途：人工填写 train/val 的标签质量与协议判断；clean-test-v2 只能作诊断参考，不能作为训练或调参信号。",
        "",
        "## 决策选项",
        "",
    ]
    lines.extend(f"- `{item}`：{decision_display(item)}" for item in allowed)
    lines.extend(
        [
            "",
            "通过 gate 的最低要求：至少审查 `20` 条 train 样本，并且至少 `8` 条 train 样本标为 `label_ok_protocol_clear` 或 `label_needs_correction`。",
            "",
            "下面的 clean-test-v2 样本仅供参考。不要把 clean-test-v2 的判断复制进 train/val gate。",
            "",
            "## Train 审查队列",
            "",
        ]
    )
    lines.extend(md_table(train_rows))
    lines.extend(["", "## Val 协议审计队列", ""])
    lines.extend(md_table(val_rows))
    lines.extend(["", "## Clean-Test-v2 诊断参考", ""])
    lines.extend(md_table(clean_rows, limit=None))
    lines.extend(
        [
            "",
            "## 浏览器审查",
            "",
            "打开 `R134_HUMAN_REVIEW_WORK_PACKAGE.html`，在浏览器里选择决策，然后点击“导出已审查 CSV”下载 `r134_train_val_review_worklist_reviewed.csv`。",
            "",
            "## 审查后预览并应用",
            "",
            "```powershell",
            "E:\\360Downloads\\anaconda3\\envs\\YOLOSAM\\python.exe .\\scripts\\run_label_protocol_review_pipeline.py --decisions-csv .\\outputs\\analysis\\r134_label_protocol_review_manifest\\review_package\\r134_train_val_review_worklist_reviewed.csv",
            "# 如果 dry-run 摘要正常，再加 --apply-in-place --backup 重新运行。只有明确要创建隔离数据集变体时才加 --create-variant。",
            "E:\\360Downloads\\anaconda3\\envs\\YOLOSAM\\python.exe .\\scripts\\preview_label_protocol_review_decisions.py --decisions-csv .\\outputs\\analysis\\r134_label_protocol_review_manifest\\review_package\\r134_train_val_review_worklist_reviewed.csv --output-json .\\outputs\\analysis\\r134_label_protocol_review_manifest\\review_package\\r134_review_preview_status.json",
            "E:\\360Downloads\\anaconda3\\envs\\YOLOSAM\\python.exe .\\scripts\\apply_label_protocol_review_decisions.py --decisions-csv .\\outputs\\analysis\\r134_label_protocol_review_manifest\\review_package\\r134_train_val_review_worklist_reviewed.csv --backup",
            "E:\\360Downloads\\anaconda3\\envs\\YOLOSAM\\python.exe .\\scripts\\check_label_protocol_review_manifest.py --csv .\\outputs\\analysis\\r134_label_protocol_review_manifest\\r134_label_protocol_review_manifest.csv --output-json .\\outputs\\analysis\\r134_label_protocol_review_manifest\\r134_review_gate_status.json",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, payload: dict[str, Any], review_rows: list[dict[str, Any]]) -> None:
    allowed = payload.get("human_decision_schema", {}).get("human_decision_allowed", [])
    train_rows = [row for row in review_rows if row["split"] == "train"]
    val_rows = [row for row in review_rows if row["split"] == "val"]
    clean_rows = [make_review_row(row, group="clean_test_reference") for row in payload.get("clean_test_reference", [])]
    reviewed_train = [row for row in train_rows if row.get("human_decision")]
    confirmed_train = [row for row in train_rows if row.get("human_decision") in CONFIRMED_DECISIONS]

    def row_card(row: dict[str, Any], index: int, *, reference: bool = False) -> str:
        panel = str(rel_asset(path, row.get("panel", ""))).replace("\\", "/")
        image = html_escape(row.get("image", ""))
        split = html_escape(row.get("split", ""))
        priority = html_escape(row.get("priority", ""))
        tag = html_escape(row.get("primary_tag", ""))
        action = html_escape(row.get("review_action", ""))
        decision = html_escape(row.get("human_decision", ""))
        notes = html_escape(row.get("human_notes", ""))
        raw_image = html_escape(row.get("image", ""))
        raw_split = html_escape(row.get("split", ""))
        raw_group = html_escape(row.get("group", ""))
        raw_priority = html_escape(row.get("priority", ""))
        raw_tag = html_escape(row.get("primary_tag", ""))
        raw_rank = html_escape(row.get("rank", ""))
        raw_score = html_escape(row.get("hardcase_similarity_score", ""))
        raw_component_count = html_escape(row.get("component_count", ""))
        raw_fg_frac = html_escape(row.get("fg_frac", ""))
        raw_nearest_center_gap = html_escape(row.get("nearest_center_gap", ""))
        raw_fg_bg_contrast = html_escape(row.get("fg_bg_contrast", ""))
        raw_panel = html_escape(row.get("panel", ""))
        raw_action = html_escape(row.get("review_action", ""))
        raw_allowed_use = html_escape(row.get("allowed_use", ""))
        score = html_escape(row.get("hardcase_similarity_score", ""))
        metrics = [
            ("components", row.get("component_count", "")),
            ("fg frac", row.get("fg_frac", "")),
            ("gap", row.get("nearest_center_gap", "")),
            ("contrast", row.get("fg_bg_contrast", "")),
        ]
        if reference:
            metrics = [
                ("dice", row.get("dice", "")),
                ("boundary", row.get("boundary_iou", "")),
                ("component delta", row.get("component_delta", "")),
            ]
        metric_html = "".join(
            f"<span><b>{html_escape(metric_display(label))}</b>{html_escape(value)}</span>"
            for label, value in metrics
            if str(value) != ""
        )
        img_html = f'<img loading="lazy" src="{html_escape(panel)}" alt="{image} 的审查面板">' if panel else '<div class="missing">没有可用面板</div>'
        reference_note = '<span class="badge locked">仅参考</span>' if reference else ""
        decision_options = [f'<option value="">{html_escape(DECISION_LABELS[""])}</option>'] + [
            f'<option value="{html_escape(item)}" {"selected" if item == row.get("human_decision") else ""}>{html_escape(decision_display(item))}</option>'
            for item in allowed
        ]
        editor_html = (
            '<div class="review-editor">'
            '<label>人工决策'
            f'<select class="decision-select">{"".join(decision_options)}</select>'
            '</label>'
            '<label>备注'
            f'<input class="notes-input" type="text" value="{notes}" placeholder="简短说明，可留空">'
            '</label>'
            '</div>'
            if not reference
            else '<div class="review-editor locked-note">仅作诊断参考。不要把这一行导出为 train/val 决策。</div>'
        )
        decision_text = html_escape(decision_display(row.get("human_decision", "")))
        return f"""
        <article class="case-card" data-group="{raw_group}" data-image="{raw_image}" data-split="{raw_split}" data-priority="{raw_priority}" data-tag="{raw_tag}" data-rank="{raw_rank}" data-score="{raw_score}" data-component-count="{raw_component_count}" data-fg-frac="{raw_fg_frac}" data-nearest-center-gap="{raw_nearest_center_gap}" data-fg-bg-contrast="{raw_fg_bg_contrast}" data-panel="{raw_panel}" data-review-action="{raw_action}" data-allowed-use="{raw_allowed_use}" data-decision="{decision or 'blank'}">
          <div class="panel">{img_html}</div>
          <div class="case-body">
            <div class="case-top">
              <span class="rank">#{index}</span>
              <h3>{image}</h3>
              <span class="badge">{split}</span>
              <span class="badge priority">{priority}</span>
              <span class="badge tag">{tag}</span>
              {reference_note}
            </div>
            <div class="metrics">{metric_html}<span><b>{html_escape(metric_display("score"))}</b>{score}</span></div>
            <p>{action}</p>
            <div class="decision-line"><b>决策</b><code>{decision_text}</code><b>备注</b><span>{notes or '&nbsp;'}</span></div>
            {editor_html}
          </div>
        </article>
        """

    cards = "\n".join(row_card(row, idx) for idx, row in enumerate(train_rows + val_rows, start=1))
    reference_cards = "\n".join(row_card(row, idx, reference=True) for idx, row in enumerate(clean_rows, start=1))
    options = "\n".join(f"<code>{html_escape(item)}：{html_escape(decision_display(item))}</code>" for item in allowed)
    export_name = "r134_train_val_review_worklist_reviewed.csv"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R134 标签/协议人工审查</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2528;
      --muted: #617074;
      --line: #cad4d7;
      --paper: #f6f8f7;
      --panel: #ffffff;
      --accent: #006b5f;
      --accent-2: #9a4d1b;
      --warn: #b3261e;
      --ok: #1d6b34;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 248, 247, 0.96);
      backdrop-filter: blur(8px);
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 18px 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 720; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    p {{ margin: 0; line-height: 1.45; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
      margin-top: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      min-height: 64px;
    }}
    .stat b {{ display: block; font-size: 20px; color: var(--ink); }}
    .stat span {{ font-size: 12px; color: var(--muted); }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
    }}
    .toolbar button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      min-height: 34px;
      padding: 0 10px;
      cursor: pointer;
    }}
    .toolbar button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    .export-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
    }}
    .export-row button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fff;
      min-height: 36px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 700;
    }}
    .export-row span {{ color: var(--muted); font-size: 13px; }}
    .decision-options {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
    code {{
      border: 1px solid var(--line);
      background: #eef3f2;
      padding: 2px 6px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
    }}
    .case-grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .case-card {{
      display: grid;
      grid-template-columns: minmax(260px, 36%) 1fr;
      gap: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
      min-height: 220px;
    }}
    .panel {{
      background: #dce3e2;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .panel img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    .missing {{ color: var(--muted); font-size: 13px; }}
    .case-body {{ padding: 14px 14px 12px 0; }}
    .case-top {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .case-top h3 {{ margin: 0 4px 0 0; font-size: 18px; min-width: 86px; }}
    .rank {{ color: var(--muted); font-size: 12px; width: 34px; }}
    .badge {{
      border: 1px solid var(--line);
      padding: 2px 7px;
      font-size: 12px;
      background: #f9fbfa;
    }}
    .priority {{ color: var(--accent-2); border-color: #d6a47c; }}
    .tag {{ color: var(--accent); border-color: #8cbdb6; }}
    .locked {{ color: var(--warn); border-color: #e5a6a1; }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }}
    .metrics span {{
      min-width: 86px;
      border-left: 3px solid var(--line);
      padding-left: 7px;
      color: var(--muted);
      font-size: 12px;
    }}
    .metrics b {{ display: block; color: var(--ink); font-weight: 700; }}
    .decision-line {{
      display: grid;
      grid-template-columns: auto minmax(120px, auto) auto 1fr;
      gap: 8px;
      align-items: center;
      margin-top: 14px;
      font-size: 13px;
    }}
    .review-editor {{
      display: grid;
      grid-template-columns: minmax(220px, 300px) 1fr;
      gap: 8px;
      margin-top: 12px;
    }}
    .review-editor label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .review-editor select,
    .review-editor input {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 0 8px;
      font: inherit;
    }}
    .review-editor.locked-note {{
      color: var(--warn);
      font-size: 13px;
      border-left: 3px solid #e5a6a1;
      padding-left: 8px;
    }}
    .reference-section {{
      border-top: 2px solid var(--line);
      margin-top: 32px;
      padding-top: 12px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 820px) {{
      .wrap {{ padding: 14px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .case-card {{ grid-template-columns: 1fr; }}
      .case-body {{ padding: 12px; }}
      .decision-line {{ grid-template-columns: 1fr; }}
      .review-editor {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>R134 标签/协议人工审查</h1>
      <p>这里用于审查 train/val 标签质量。Clean-test-v2 只显示为诊断参考，不能用于训练、调参或 gate 决策。</p>
      <div class="summary">
        <div class="stat"><b>{len(train_rows)}</b><span>train 待审查</span></div>
        <div class="stat"><b>{len(val_rows)}</b><span>val 待审计</span></div>
        <div class="stat"><b>{len(clean_rows)}</b><span>clean-test 参考</span></div>
        <div class="stat"><b>{len(reviewed_train)}</b><span>已审查 train</span></div>
        <div class="stat"><b>{len(confirmed_train)}</b><span>可用于变体 train</span></div>
        <div class="stat"><b>{'已满足' if len(reviewed_train) >= 20 and len(confirmed_train) >= 8 else '未满足'}</b><span>变体 gate</span></div>
      </div>
      <div class="toolbar" aria-label="filters">
        <button class="active" data-filter="all">全部</button>
        <button data-filter="train">Train</button>
        <button data-filter="val">Val</button>
        <button data-filter="P0">P0</button>
        <button data-filter="P1">P1</button>
        <button data-filter="blank">未决策</button>
      </div>
      <div class="decision-options">{options}</div>
      <div class="export-row">
        <button id="exportCsv" type="button">导出已审查 CSV</button>
        <span id="exportStatus">页面里的改动只有导出后才会保存。应用前先运行 preview。</span>
      </div>
    </div>
  </header>
  <main class="wrap">
    <h2>Train / Val 审查清单</h2>
    <section class="case-grid" id="worklist">
      {cards}
    </section>
    <section class="reference-section">
      <h2>Clean-Test-v2 诊断参考</h2>
      <section class="case-grid">
        {reference_cards}
      </section>
    </section>
  </main>
  <script>
    const buttons = document.querySelectorAll('.toolbar button');
    const cards = document.querySelectorAll('#worklist .case-card');
    const exportButton = document.getElementById('exportCsv');
    const exportStatus = document.getElementById('exportStatus');
    const csvHeaders = ['group','image','split','priority','primary_tag','rank','hardcase_similarity_score','component_count','fg_frac','nearest_center_gap','fg_bg_contrast','panel','review_action','allowed_use','human_decision','human_notes'];

    function csvEscape(value) {{
      const text = String(value ?? '');
      return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
    }}

    function updateCardDecision(card) {{
      const select = card.querySelector('.decision-select');
      const notesInput = card.querySelector('.notes-input');
      if (!select || !notesInput) return;
      card.dataset.decision = select.value || 'blank';
      const code = card.querySelector('.decision-line code');
      const note = card.querySelector('.decision-line span');
      const labels = {{
        '': '空白，暂不决定',
        'label_ok_protocol_clear': '标签可用，协议清楚',
        'label_boundary_ambiguous': '边界不清，需要谨慎',
        'label_needs_correction': '标签需要修正',
        'exclude_from_training_variant': '排除，不放入训练变体',
        'needs_second_review': '需要二次复核',
      }};
      if (code) code.textContent = labels[select.value || ''] || select.value || labels[''];
      if (note) note.textContent = notesInput.value || '';
    }}

    document.querySelectorAll('.decision-select, .notes-input').forEach((control) => {{
      control.addEventListener('input', () => updateCardDecision(control.closest('.case-card')));
      control.addEventListener('change', () => updateCardDecision(control.closest('.case-card')));
    }});
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        cards.forEach((card) => {{
          const show = filter === 'all'
            || card.dataset.split === filter
            || card.dataset.priority === filter
            || card.dataset.decision === filter;
          card.classList.toggle('hidden', !show);
        }});
      }});
    }});

    exportButton.addEventListener('click', () => {{
      const rows = Array.from(cards).map((card) => {{
        const select = card.querySelector('.decision-select');
        const notesInput = card.querySelector('.notes-input');
        return {{
          group: card.dataset.group || '',
          image: card.dataset.image || '',
          split: card.dataset.split || '',
          priority: card.dataset.priority || '',
          primary_tag: card.dataset.tag || '',
          rank: card.dataset.rank || '',
          hardcase_similarity_score: card.dataset.score || '',
          component_count: card.dataset.componentCount || '',
          fg_frac: card.dataset.fgFrac || '',
          nearest_center_gap: card.dataset.nearestCenterGap || '',
          fg_bg_contrast: card.dataset.fgBgContrast || '',
          panel: card.dataset.panel || '',
          review_action: card.dataset.reviewAction || '',
          allowed_use: card.dataset.allowedUse || '',
          human_decision: select ? select.value : '',
          human_notes: notesInput ? notesInput.value : '',
        }};
      }});
      const csv = [csvHeaders.join(','), ...rows.map((row) => csvHeaders.map((key) => csvEscape(row[key])).join(','))].join('\\r\\n') + '\\r\\n';
      const blob = new Blob(['\\ufeff' + csv], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = '{export_name}';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      const reviewedTrain = rows.filter((row) => row.split === 'train' && row.human_decision).length;
      const confirmedTrain = rows.filter((row) => row.split === 'train' && ['label_ok_protocol_clear', 'label_needs_correction'].includes(row.human_decision)).length;
      exportStatus.textContent = `已导出 {export_name}。Train 已审查：${{reviewedTrain}} / 20；可用于变体：${{confirmedTrain}} / 8。`;
    }});
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_manifest(args.manifest)
    train = sorted(payload.get("train_review_queue", []), key=row_sort_key)
    val = sorted(payload.get("val_audit_queue", []), key=row_sort_key)
    train_rows = [make_review_row(row, group="train_review_queue") for row in train]
    val_rows = [make_review_row(row, group="val_audit_queue") for row in val]
    review_rows = train_rows + val_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)

    worklist_csv = args.output_dir / "r134_train_val_review_worklist.csv"
    reference_csv = args.output_dir / "r134_clean_test_reference_only.csv"
    summary_json = args.output_dir / "r134_review_package_summary.json"
    readme_md = args.output_dir / "R134_HUMAN_REVIEW_WORK_PACKAGE.md"
    review_html = args.output_dir / "R134_HUMAN_REVIEW_WORK_PACKAGE.html"

    write_csv(worklist_csv, review_rows)
    write_csv(reference_csv, [make_review_row(row, group="clean_test_reference") for row in payload.get("clean_test_reference", [])])
    write_markdown(readme_md, payload, review_rows)
    write_html(review_html, payload, review_rows)

    reviewed_train = [row for row in train if reviewed(row)]
    confirmed_train = [row for row in train if confirmed(row)]
    summary = {
        "manifest": str(args.manifest),
        "worklist_csv": str(worklist_csv),
        "reference_csv": str(reference_csv),
        "readme_md": str(readme_md),
        "review_html": str(review_html),
        "browser_export_csv_name": "r134_train_val_review_worklist_reviewed.csv",
        "counts": {
            "train_review_queue": len(train),
            "val_audit_queue": len(val),
            "clean_test_reference": len(payload.get("clean_test_reference", [])),
            "reviewed_train": len(reviewed_train),
            "confirmed_train_for_variant": len(confirmed_train),
            "train_priority_counts": dict(Counter(str(row.get("priority", "")) for row in train)),
            "train_tag_counts": dict(Counter(str(row.get("primary_tag", "")) for row in train)),
            "val_tag_counts": dict(Counter(str(row.get("primary_tag", "")) for row in val)),
        },
        "gate": {
            "min_reviewed_train": args.min_reviewed_train,
            "min_confirmed_train": args.min_confirmed_train,
            "ready": len(reviewed_train) >= args.min_reviewed_train and len(confirmed_train) >= args.min_confirmed_train,
            "next_action": "human_review_train_val_worklist",
        },
        "clean_test_policy": "reference_only_do_not_train_or_tune",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
