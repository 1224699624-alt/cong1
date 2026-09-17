#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import paramiko


BASELINES = {
    "araa_clean": {
        "dice": 0.9117660066557425,
        "iou": 0.8384360930797206,
        "precision": 0.9101767600852156,
        "recall": 0.9146908686504701,
        "boundary_iou": 0.2352651559145258,
    },
    "keepbone_v2_clean": {
        "dice": 0.9100290053191697,
        "iou": 0.8354338498823685,
        "precision": 0.9131011545806722,
        "recall": 0.9082009857713057,
        "boundary_iou": 0.22848859791261497,
    },
    "keepbone_v2_test": {
        "dice": 0.8936935131682259,
        "iou": 0.810590948170741,
        "precision": 0.8992150550562691,
        "recall": 0.8915754953748375,
        "boundary_iou": 0.2159435973764147,
    },
    "aic_test": {
        "dice": 0.8897098247956451,
        "iou": 0.8043424421111297,
        "precision": 0.8803708058316431,
        "recall": 0.9019376122732686,
        "boundary_iou": 0.22374453240000397,
    },
    "aic_clean": {
        "dice": 0.9058689991490124,
        "iou": 0.8288071109243607,
        "precision": 0.8964336332990822,
        "recall": 0.9169009242293409,
        "boundary_iou": 0.23809236134009293,
    },
    "r010_test": {
        "dice": 0.8936940564822826,
        "iou": 0.8108160964986327,
        "precision": 0.8983233164112061,
        "recall": 0.8925827704012629,
        "boundary_iou": 0.21888265557189615,
    },
    "r010_clean": {
        "dice": 0.9106792045956249,
        "iou": 0.8365417998363095,
        "precision": 0.9126588865271129,
        "recall": 0.9099110885735171,
        "boundary_iou": 0.23261050948537884,
    },
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def read_json_if_exists(sftp: paramiko.SFTPClient, path: str):
    try:
        with sftp.open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def remote_text(client: paramiko.SSHClient, cmd: str) -> str:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out + (("\n" + err) if err else "")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_sync_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Step Sync Log\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def fmt_metric(v: float) -> str:
    return f"{v:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--remote-host", required=True)
    ap.add_argument("--remote-user", required=True)
    ap.add_argument("--remote-password", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    refine = repo_root / "research-workflow" / "refine-logs"
    status_file = refine / "AUTONOMOUS_LOOP_STATUS.md"
    tracker_file = refine / "EXPERIMENT_TRACKER.md"
    results_file = refine / "EXPERIMENT_RESULTS.md"
    summary_file = refine / "PROGRESS_SUMMARY_CN.md"
    sync_log = refine / "STEP_SYNC_LOG.md"

    remote_repo = "/home/shenzeyu/workspace/YOLO_SAM_generic_src"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.remote_host, username=args.remote_user, password=args.remote_password, timeout=20)
    sftp = client.open_sftp()

    r011_test = read_json_if_exists(
        sftp,
        f"{remote_repo}/outputs/ablations/allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam/TSRS_RSNA-Epiphysis/test/metrics.json",
    )
    r011_clean = read_json_if_exists(
        sftp,
        f"{remote_repo}/outputs/ablations_variants/allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/metrics.json",
    )

    tmux_out = remote_text(client, "tmux ls 2>/dev/null | grep r011_keepbone_v3 || true")
    client.close()

    date_text = now_text()
    r011_status = "DONE" if r011_clean else "RUNNING"

    if r011_clean:
        r011_status_note = (
            f"原始 test 与 clean-test-v2 都已完成；clean-test-v2 Dice `{fmt_metric(r011_clean['mean']['dice'])}`，"
            f"低于当前内部最强主切片基线 `{fmt_metric(BASELINES['keepbone_v2_clean']['dice'])}`"
        )
    else:
        r011_status_note = (
            "原始 test 已完成；clean-test-v2 尚未落盘。"
            " 如果后台为空，说明 clean-test-v2 还没启动或需要重新挂起。"
        )

    status_md = f"""# 自动循环状态

**日期**: {date_text.split()[0]}

## 当前 ARIS 阶段

- `result-to-claim`: 已完成，用于判断 `aic_refiner` 这条线当前支持什么结论
- `research-refine`: 已完成，用于整理下一轮主候选修补方向
- `experiment-bridge`: 当前这一轮仍在推进或刚完成状态同步

## 上一轮已完成结果

### `R010 = allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`

- bridge 状态：`DONE`
- 主切片 `clean-test-v2`：
  - Dice: `{fmt_metric(BASELINES['r010_clean']['dice'])}`
  - Precision: `{fmt_metric(BASELINES['r010_clean']['precision'])}`
  - Recall: `{fmt_metric(BASELINES['r010_clean']['recall'])}`
  - Boundary IoU: `{fmt_metric(BASELINES['r010_clean']['boundary_iou'])}`

### R010 结论

- 相比 `keepbone_cutgap_v2`，主切片是小幅提升
- 但 `1433` 明显退化，`3040` 也有轻微回落
- 所以还不能结束 `experiment-bridge`

## 当前关注实验

### `R011 = allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam`

#### 原始 test
"""
    if r011_test:
        m = r011_test["mean"]
        status_md += f"""
- Dice: `{fmt_metric(m['dice'])}`
- Precision: `{fmt_metric(m['precision'])}`
- Recall: `{fmt_metric(m['recall'])}`
- Boundary IoU: `{fmt_metric(m['boundary_iou'])}`
"""
    else:
        status_md += "\n- 尚未落盘\n"

    status_md += "\n#### clean-test-v2\n"
    if r011_clean:
        m = r011_clean["mean"]
        status_md += f"""
- Dice: `{fmt_metric(m['dice'])}`
- Precision: `{fmt_metric(m['precision'])}`
- Recall: `{fmt_metric(m['recall'])}`
- Boundary IoU: `{fmt_metric(m['boundary_iou'])}`
"""
    else:
        status_md += "\n- 尚未落盘\n"

    status_md += f"""
#### 当前状态说明

- `R011` 当前状态：`{r011_status}`
- 说明：{r011_status_note}
- 远端 tmux 检查结果：

```text
{tmux_out.strip() or '(no active tmux session)'}
```

## 下一步判定门槛

- 如果 `R011` 主切片强于当前内部最强，且 hard-case 足够干净，再进入 `/auto-review-loop`
- 否则继续留在 bridge 阶段，准备下一条候选
"""

    tracker_md = f"""# 实验跟踪表

说明：
- `Status` 保留英文 token，方便脚本识别
- 备注尽量用中文，方便直接读

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 冻结外部基线 | ARAA `epoch_98` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Boundary IoU | MUST | DONE | 已记录 ARAA 在主切片上的基线指标 |
| R002 | M0 | 冻结当前内部最强主切片基线 | `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Boundary IoU | MUST | DONE | 已记录当前内部最强主切片结果 |
| R003 | M0 | 冻结当前内部最强原始 test 基线 | `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` | `TSRS_RSNA-Epiphysis` test | Dice, IoU, Precision, Recall, Boundary IoU | MUST | DONE | 已记录当前内部最强原始 test 结果 |
| R004 | M1 | 复盘 `aic_refiner` 训练与阈值结果 | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | train/val | val Dice-led threshold selection | MUST | DONE | 远端已有 checkpoint 和 threshold sweep |
| R005 | M1 | 复盘 `aic_refiner` 原始 test 结果 | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | `TSRS_RSNA-Epiphysis` test | Dice, IoU, Precision, Recall, Specificity, Boundary IoU | MUST | DONE | 原始 test 不如当前内部最强基线 |
| R006 | M1 | 复盘 `aic_refiner` 主切片结果 | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Specificity, Boundary IoU | MUST | DONE | clean-test-v2 Dice `0.905869`，失败但提供了 recall / boundary 线索 |
| R007 | M3 | `aic_refiner` 的 hard-case 验证 | baseline vs current best vs candidate | `2982/1475/3040/1433` | qualitative gap / thin-epiphysis preservation | MUST | DONE | 结果是 mixed，不足以升级主线 |
| R008 | M4 | 过滤数据控制实验（trainval filtered） | keepbone-cutgap filtered trainval v1 | original test + clean-test-v2 if available | same core metrics | NICE | SKIPPED | 因 `aic_refiner` 主切片失败而跳过 |
| R009 | M4 | 过滤数据控制实验（badlabel filtered） | keepbone-cutgap badlabel filtered v1 | original test + clean-test-v2 if available | same core metrics | NICE | SKIPPED | 因 `aic_refiner` 主切片失败而跳过 |
| R010 | M1 | 低风险 precision 修补版 | `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam` | train/val + original test + clean-test-v2 | Dice, IoU, Precision, Recall, Boundary IoU | MUST | DONE | clean-test-v2 Dice `0.910679`，主切片略赢，但 hard-case 仍不够干净 |
| R011 | M2 | 结构修补主候选 `v3` | `allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam` | original test + clean-test-v2 | Dice, IoU, Precision, Recall, Boundary IoU | MUST | {r011_status} | {r011_status_note} |
"""

    results_md = f"""# 实验结果总览

## 目标基线

### 外部目标：ARAA

- 主切片：`TSRS_RSNA-Epiphysis_clean_test_v2`
- Dice: `{fmt_metric(BASELINES['araa_clean']['dice'])}`
- IoU: `{fmt_metric(BASELINES['araa_clean']['iou'])}`
- Precision: `{fmt_metric(BASELINES['araa_clean']['precision'])}`
- Recall: `{fmt_metric(BASELINES['araa_clean']['recall'])}`
- Boundary IoU: `{fmt_metric(BASELINES['araa_clean']['boundary_iou'])}`

### 当前内部最强：`keepbone_cutgap_v2`

- clean-test-v2 Dice: `{fmt_metric(BASELINES['keepbone_v2_clean']['dice'])}`
- clean-test-v2 Boundary IoU: `{fmt_metric(BASELINES['keepbone_v2_clean']['boundary_iou'])}`
- 原始 test Dice: `{fmt_metric(BASELINES['keepbone_v2_test']['dice'])}`

## `aic_refiner` 的结论

- clean-test-v2 Dice: `{fmt_metric(BASELINES['aic_clean']['dice'])}`
- clean-test-v2 Precision: `{fmt_metric(BASELINES['aic_clean']['precision'])}`
- clean-test-v2 Recall: `{fmt_metric(BASELINES['aic_clean']['recall'])}`
- clean-test-v2 Boundary IoU: `{fmt_metric(BASELINES['aic_clean']['boundary_iou'])}`

结论：

- recall 和 boundary 更高
- 但 precision 掉太多
- 整体 Dice 不够，所以不能升级主线

## `R010 precision_tune`

- clean-test-v2 Dice: `{fmt_metric(BASELINES['r010_clean']['dice'])}`
- clean-test-v2 Precision: `{fmt_metric(BASELINES['r010_clean']['precision'])}`
- clean-test-v2 Recall: `{fmt_metric(BASELINES['r010_clean']['recall'])}`
- clean-test-v2 Boundary IoU: `{fmt_metric(BASELINES['r010_clean']['boundary_iou'])}`

结论：

- 主切片小幅正向
- 但 hard-case 仍然不够干净

## `R011 v3`
"""

    if r011_test:
        m = r011_test["mean"]
        results_md += f"""
### 原始 test

- Dice: `{fmt_metric(m['dice'])}`
- IoU: `{fmt_metric(m['iou'])}`
- Precision: `{fmt_metric(m['precision'])}`
- Recall: `{fmt_metric(m['recall'])}`
- Boundary IoU: `{fmt_metric(m['boundary_iou'])}`
"""
    if r011_clean:
        m = r011_clean["mean"]
        delta = m["dice"] - BASELINES["keepbone_v2_clean"]["dice"]
        results_md += f"""
### clean-test-v2

- Dice: `{fmt_metric(m['dice'])}`
- IoU: `{fmt_metric(m['iou'])}`
- Precision: `{fmt_metric(m['precision'])}`
- Recall: `{fmt_metric(m['recall'])}`
- Boundary IoU: `{fmt_metric(m['boundary_iou'])}`

### 对比当前内部最强主切片基线

- 基线 Dice: `{fmt_metric(BASELINES['keepbone_v2_clean']['dice'])}`
- `R011` Dice: `{fmt_metric(m['dice'])}`
- 差值: `{delta:+.6f}`

### 结论

- `R011` 已完整跑完
- 主切片仍低于当前内部最强基线
- 因此这条线目前也不够，不能进入 `/auto-review-loop`
"""
    else:
        results_md += """
### clean-test-v2

- 尚未落盘

### 结论

- `R011` 还没拿到主切片完整结果
- 继续等待 bridge 完成
"""

    results_md += """
## 当前阶段判断

- 还在 `experiment-bridge`
- 只有当某条候选完整跑完，并且主切片与 hard-case 都足够强，才会进入 `/auto-review-loop`
"""

    summary_md = f"""# 项目进度总结（中文易读版）

**更新时间**: {date_text}

## 一、我们现在在做什么

当前目标很明确：

- 任务：儿童掌骨骨骺分割
- 主线：`YOLO+SAM` + error refiner
- 最终目标：在 `TSRS_RSNA-Epiphysis_clean_test_v2` 上追平或超过 `ARAA`

当前要追的目标分数：

- `ARAA` clean-test-v2 Dice = `{fmt_metric(BASELINES['araa_clean']['dice'])}`

当前内部最强基线：

- `keepbone_cutgap_v2` clean-test-v2 Dice = `{fmt_metric(BASELINES['keepbone_v2_clean']['dice'])}`

## 二、前面做完了什么

### `aic_refiner`

- 已完整跑完
- 结论：有高 recall / 高 boundary 信号，但 precision 崩得太多，不能升主线

### `R010 = v2_precision_tune`

- clean-test-v2 Dice = `{fmt_metric(BASELINES['r010_clean']['dice'])}`
- 主切片小幅正向
- 但 hard-case 仍不够干净，所以不能结束 bridge

## 三、最新结果：`R011 = keepbone_cutgap_v3`
"""

    if r011_clean:
        m = r011_clean["mean"]
        summary_md += """
### `R011` 已经完整跑完
"""
        if r011_test:
            summary_md += f"""
- 原始 test Dice = `{fmt_metric(r011_test['mean']['dice'])}`
"""
        summary_md += f"""
- clean-test-v2 Dice = `{fmt_metric(m['dice'])}`
- clean-test-v2 Precision = `{fmt_metric(m['precision'])}`
- clean-test-v2 Recall = `{fmt_metric(m['recall'])}`
- clean-test-v2 Boundary IoU = `{fmt_metric(m['boundary_iou'])}`

### 这说明什么

- `R011` 低于当前内部最强主切片基线 `{fmt_metric(BASELINES['keepbone_v2_clean']['dice'])}`
- 所以这条线目前不够好
- 现在看不到后台，不是没启动，而是因为它已经跑完退出了
"""
    else:
        summary_md += """
### `R011` 还没有完整结束

- 原始 test 已完成
- clean-test-v2 还没出完整结果
- 当前仍停在 bridge 阶段
"""

    summary_md += """
## 四、现在停在哪一步

当前仍停在：

- `/experiment-bridge`

没有进入：

- `/auto-review-loop`

原因：

- 当前候选结果还不够强，bridge 阶段还没成功

## 五、现在最该记住的一句话

`R011` 的最新结果已经说明这条线目前也不够，因此 bridge 还没有成功，`/auto-review-loop` 还不能开始。
"""

    write_text(status_file, status_md)
    write_text(tracker_file, tracker_md)
    write_text(results_file, results_md)
    write_text(summary_file, summary_md)

    append_sync_log(
        sync_log,
        f"- [{date_text}] sync complete: R011 status={r011_status}, "
        + (f"clean-test-v2 dice={fmt_metric(r011_clean['mean']['dice'])}" if r011_clean else "clean-test-v2 pending"),
    )
    print(f"Synced ARIS state at {date_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
