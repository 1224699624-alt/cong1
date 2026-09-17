#!/usr/bin/env python3
"""Prepare train/val-only review package for original-vs-reannotated labels."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_AUDIT = Path("outputs/analysis/r163_reannotated_pair_shift.json")
DEFAULT_ORIGINAL = Path("data/raw/TSRS_RSNA-Epiphysis")
DEFAULT_REANNOTATED = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1")
DEFAULT_OUTPUT = Path("outputs/analysis/r168_reannotation_protocol_review_package")
DECISIONS = [
    "original_label_correct",
    "reannotated_label_correct",
    "both_wrong_needs_correction",
    "exclude_from_training_variant",
    "uncertain_second_review",
]
DECISION_LABELS = {
    "": "未决定",
    "original_label_correct": "原标签更可信",
    "reannotated_label_correct": "新标注更可信",
    "both_wrong_needs_correction": "两者都不对，需要修正PNG",
    "exclude_from_training_variant": "排除，不放入训练变体",
    "uncertain_second_review": "不确定，需要二审",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--reannotated-root", type=Path, default=DEFAULT_REANNOTATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--panel-width", type=int, default=360)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_image(split_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def fit(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image.copy()
    scale = width / float(image.width)
    return image.resize((width, max(1, int(round(image.height * scale)))), Image.Resampling.BILINEAR)


def overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    resized_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(base.size, Image.Resampling.NEAREST)
    ) > 0
    arr = np.asarray(base).copy()
    arr[resized_mask] = ((1.0 - alpha) * arr[resized_mask] + alpha * np.array(color)).astype(np.uint8)
    return Image.fromarray(arr)


def diff_overlay(base: Image.Image, orig: np.ndarray, reann: np.ndarray) -> Image.Image:
    orig_rs = np.asarray(
        Image.fromarray(orig.astype(np.uint8) * 255).resize(base.size, Image.Resampling.NEAREST)
    ) > 0
    reann_rs = np.asarray(
        Image.fromarray(reann.astype(np.uint8) * 255).resize(base.size, Image.Resampling.NEAREST)
    ) > 0
    arr = np.asarray(base).copy()
    both = orig_rs & reann_rs
    orig_only = orig_rs & ~reann_rs
    reann_only = reann_rs & ~orig_rs
    arr[both] = ((0.78 * arr[both]) + 0.22 * np.array([40, 210, 80])).astype(np.uint8)
    arr[orig_only] = ((0.45 * arr[orig_only]) + 0.55 * np.array([60, 120, 255])).astype(np.uint8)
    arr[reann_only] = ((0.45 * arr[reann_only]) + 0.55 * np.array([255, 70, 70])).astype(np.uint8)
    return Image.fromarray(arr)


def title_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 48 if subtitle else 30
    canvas = Image.new("RGB", (image.width, image.height + header), (255, 255, 255))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((8, 27), subtitle, fill=(70, 70, 70))
    return canvas


def hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), (244, 246, 246))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def priority_reason(row: dict[str, Any]) -> str:
    dice = float(row.get("label_dice", 1.0))
    ratio = float(row.get("fg_frac_ratio", 1.0))
    comp_delta = float(row.get("component_count_delta", 0.0))
    reann_components = float(row.get("reann_component_count", 0.0))
    if reann_components == 0:
        return "新标注为空"
    if ratio >= 2.25:
        return "新标注面积明显膨胀"
    if ratio <= 0.45:
        return "新标注面积明显缩小"
    if abs(comp_delta) >= 12:
        return "连通域数量变化很大"
    if dice < 0.65:
        return "原标签和新标注重合度低"
    return "协议差异候选"


def priority_score(row: dict[str, Any]) -> float:
    dice = float(row.get("label_dice", 1.0))
    ratio = float(row.get("fg_frac_ratio", 1.0))
    comp_delta = abs(float(row.get("component_count_delta", 0.0)))
    reann_components = float(row.get("reann_component_count", 0.0))
    ratio_penalty = 0.0
    if ratio > 0:
        ratio_penalty = abs(np.log2(ratio))
    empty_bonus = 4.0 if reann_components == 0 else 0.0
    return (1.0 - dice) * 4.0 + ratio_penalty + min(comp_delta / 12.0, 3.0) + empty_bonus


def select_rows(audit: dict[str, Any], split: str, count: int) -> list[dict[str, Any]]:
    rows = list(audit["paired"][split]["per_image"])
    rows = sorted(rows, key=priority_score, reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row["image"])
        if name in seen:
            continue
        selected.append(row)
        seen.add(name)
        if len(selected) >= count:
            break
    return selected


def make_panel(
    *,
    split: str,
    row: dict[str, Any],
    original_root: Path,
    reannotated_root: Path,
    output_dir: Path,
    panel_width: int,
    rank: int,
) -> str | None:
    image_name = str(row["image"])
    image_path = find_image(reannotated_root / split, image_name) or find_image(original_root / split, image_name)
    if image_path is None:
        return None
    image = load_rgb(image_path)
    size = image.size
    orig_mask = load_mask(original_root / f"{split}_labels" / image_name, size)
    reann_mask = load_mask(reannotated_root / f"{split}_labels" / image_name, size)
    base = fit(image, panel_width)
    panels = [
        title_panel(base, "原图"),
        title_panel(overlay(base, orig_mask, (60, 120, 255), 0.20), "原标签", "蓝色，低透明度"),
        title_panel(overlay(base, reann_mask, (255, 70, 70), 0.20), "新标注", "红色，低透明度"),
        title_panel(diff_overlay(base, orig_mask, reann_mask), "差异", "绿=重合 蓝=仅原 红=仅新"),
    ]
    canvas = hstack(panels)
    panel_dir = output_dir / "panels" / split
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{rank:03d}_{Path(image_name).stem}_r168.jpg"
    canvas.save(panel_dir / out_name, quality=95)
    return str((Path("panels") / split / out_name).as_posix())


def review_record(split: str, rank: int, row: dict[str, Any], panel: str | None) -> dict[str, Any]:
    return {
        "run_id": "R168",
        "split": split,
        "rank": rank,
        "image": row.get("image", ""),
        "priority_reason": priority_reason(row),
        "priority_score": f"{priority_score(row):.6f}",
        "label_dice": f"{float(row.get('label_dice', 0.0)):.6f}",
        "orig_fg_frac": f"{float(row.get('orig_fg_frac', 0.0)):.8f}",
        "reann_fg_frac": f"{float(row.get('reann_fg_frac', 0.0)):.8f}",
        "fg_frac_ratio": f"{float(row.get('fg_frac_ratio', 0.0)):.6f}",
        "orig_component_count": f"{float(row.get('orig_component_count', 0.0)):.0f}",
        "reann_component_count": f"{float(row.get('reann_component_count', 0.0)):.0f}",
        "component_count_delta": f"{float(row.get('component_count_delta', 0.0)):.0f}",
        "panel": panel or "",
        "human_decision": "",
        "human_notes": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "split",
        "rank",
        "image",
        "priority_reason",
        "priority_score",
        "label_dice",
        "orig_fg_frac",
        "reann_fg_frac",
        "fg_frac_ratio",
        "orig_component_count",
        "reann_component_count",
        "component_count_delta",
        "panel",
        "human_decision",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    options = "".join(f'<option value="{esc(v)}">{esc(DECISION_LABELS[v])}</option>' for v in DECISIONS)
    cards = []
    for row in rows:
        panel = row["panel"].replace("\\", "/")
        img = f'<img loading="lazy" src="{esc(panel)}" alt="{esc(row["image"])}">' if panel else "<div>没有可用面板</div>"
        cards.append(
            f"""
            <article class="card" data-split="{esc(row['split'])}" data-reason="{esc(row['priority_reason'])}">
              <div class="panel">{img}</div>
              <div class="body">
                <div class="top"><b>#{esc(row['rank'])}</b><h3>{esc(row['image'])}</h3><span>{esc(row['split'])}</span><span>{esc(row['priority_reason'])}</span></div>
                <div class="metrics">
                  <span><b>标注 Dice</b>{esc(row['label_dice'])}</span>
                  <span><b>面积比</b>{esc(row['fg_frac_ratio'])}</span>
                  <span><b>原连通域</b>{esc(row['orig_component_count'])}</span>
                  <span><b>新连通域</b>{esc(row['reann_component_count'])}</span>
                  <span><b>连通域差</b>{esc(row['component_count_delta'])}</span>
                </div>
                <div class="editor">
                  <label>人工判断<select class="decision"><option value="">未决定</option>{options}</select></label>
                  <label>备注<input class="notes" type="text" placeholder="可留空"></label>
                </div>
              </div>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R168 新旧标注协议审查</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f7f8; color: #20292c; }}
    header {{ position: sticky; top: 0; z-index: 10; background: rgba(245,247,248,.96); border-bottom: 1px solid #ccd6d9; }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 16px 22px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    p {{ margin: 0; color: #607075; line-height: 1.45; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    button {{ border: 1px solid #b8c6ca; background: white; min-height: 34px; padding: 0 10px; cursor: pointer; }}
    button.active, #exportCsv {{ border-color: #006b5f; background: #006b5f; color: white; }}
    .grid {{ display: grid; gap: 12px; }}
    .card {{ display: grid; grid-template-columns: minmax(420px, 58%) 1fr; gap: 14px; background: white; border: 1px solid #ccd6d9; }}
    .panel {{ background: #dfe7e9; display: flex; align-items: center; justify-content: center; overflow: auto; }}
    .panel img {{ width: 100%; height: auto; display: block; }}
    .body {{ padding: 14px 14px 14px 0; }}
    .top {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
    h3 {{ margin: 0; font-size: 18px; }}
    .top span {{ border: 1px solid #ccd6d9; padding: 2px 7px; font-size: 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 8px; margin: 14px 0; }}
    .metrics span {{ border-left: 3px solid #ccd6d9; padding-left: 8px; color: #607075; font-size: 13px; }}
    .metrics b {{ display: block; color: #20292c; }}
    .editor {{ display: grid; grid-template-columns: minmax(180px, 260px) 1fr; gap: 8px; }}
    label {{ display: grid; gap: 4px; color: #607075; font-size: 12px; font-weight: 700; }}
    select, input {{ min-height: 34px; border: 1px solid #b8c6ca; padding: 0 8px; font: inherit; }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      .card {{ grid-template-columns: 1fr; }}
      .body {{ padding: 12px; }}
      .editor {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>R168 新旧标注协议审查</h1>
      <p>只包含 train/val，不包含 clean-test-v2，也不包含新标注 test。蓝色是原标签，红色是新标注，差异图中绿色表示重合。这里的判断用于决定后续是否构建新的隔离训练变体。</p>
      <div class="toolbar">
        <button class="active" data-filter="all">全部</button>
        <button data-filter="train">Train</button>
        <button data-filter="val">Val</button>
        <button id="exportCsv" type="button">导出已审查 CSV</button>
      </div>
    </div>
  </header>
  <main class="wrap grid" id="cards">
    {''.join(cards)}
  </main>
  <script>
    const headers = {json.dumps(list(rows[0].keys()) if rows else [], ensure_ascii=False)};
    const sourceRows = {json.dumps(rows, ensure_ascii=False)};
    const cards = Array.from(document.querySelectorAll('.card'));
    document.querySelectorAll('button[data-filter]').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('button[data-filter]').forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        cards.forEach((card) => card.classList.toggle('hidden', !(filter === 'all' || card.dataset.split === filter)));
      }});
    }});
    function csvEscape(value) {{
      const text = String(value ?? '');
      return /[",\\r\\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
    }}
    document.getElementById('exportCsv').addEventListener('click', () => {{
      const edited = sourceRows.map((row, idx) => {{
        const card = cards[idx];
        return {{
          ...row,
          human_decision: card.querySelector('.decision').value,
          human_notes: card.querySelector('.notes').value,
        }};
      }});
      const csv = [headers.join(','), ...edited.map((row) => headers.map((key) => csvEscape(row[key])).join(','))].join('\\r\\n') + '\\r\\n';
      const blob = new Blob(['\\ufeff' + csv], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'r168_reannotation_protocol_review_reviewed.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }});
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# R168 Reannotation Protocol Review Package",
        "",
        "This package is train/val only. It excludes clean-test-v2 and excludes the reannotated test split.",
        "",
        "## Summary",
        "",
        f"- Total review rows: `{len(rows)}`",
        f"- Train rows: `{summary['counts']['train']}`",
        f"- Val rows: `{summary['counts']['val']}`",
        f"- Output HTML: `{summary['review_html']}`",
        f"- Worklist CSV: `{summary['worklist_csv']}`",
        "",
        "## Decision Values",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in DECISION_LABELS.items() if key)
    lines.extend(
        [
            "",
            "## Review Notes",
            "",
            "- Original label overlay is blue with low opacity.",
            "- Reannotated label overlay is red with low opacity.",
            "- Difference panel uses green for overlap, blue for original-only, red for reannotated-only.",
            "- Exported reviewed CSV should be used to decide whether a corrected isolated train/val variant is worth building.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = load_json(args.audit_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str, int, dict[str, Any]]] = []
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        for rank, row in enumerate(select_rows(audit, split, count), start=1):
            selected.append((split, rank, row))

    records: list[dict[str, Any]] = []
    for split, rank, row in selected:
        panel = make_panel(
            split=split,
            row=row,
            original_root=args.original_root,
            reannotated_root=args.reannotated_root,
            output_dir=args.output_dir,
            panel_width=args.panel_width,
            rank=rank,
        )
        records.append(review_record(split, rank, row, panel))

    worklist_csv = args.output_dir / "r168_reannotation_protocol_review_worklist.csv"
    review_html = args.output_dir / "R168_REANNOTATION_PROTOCOL_REVIEW.html"
    readme = args.output_dir / "R168_REANNOTATION_PROTOCOL_REVIEW.md"
    summary_json = args.output_dir / "r168_reannotation_protocol_review_summary.json"
    write_csv(worklist_csv, records)
    write_html(review_html, records)
    summary = {
        "run_id": "R168",
        "audit_json": str(args.audit_json),
        "original_root": str(args.original_root),
        "reannotated_root": str(args.reannotated_root),
        "output_dir": str(args.output_dir),
        "worklist_csv": str(worklist_csv),
        "review_html": str(review_html),
        "readme": str(readme),
        "reviewed_export_name": "r168_reannotation_protocol_review_reviewed.csv",
        "counts": dict(Counter(row["split"] for row in records)),
        "priority_reason_counts": dict(Counter(row["priority_reason"] for row in records)),
        "policy": {
            "uses_clean_test_v2": False,
            "uses_reannotated_test": False,
            "mutates_dataset": False,
            "next_gate": "human_decisions_or_corrected_pngs_required_before_training",
        },
    }
    write_markdown(readme, records, summary)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
