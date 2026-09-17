# R255 Refinement Report

**Date**: 2026-07-13  
**Rounds**: 3  
**Final score**: 9.07/10  
**Verdict**: READY FOR GATE A AND IMPLEMENTATION

## Method evolution

1. 从 generic adjacent-pair PCR 收缩为腕骨/掌骨近端 close-gap false-merge。
2. 从跨病例骨编号改为图内无身份局部描述符。
3. 骨龄/性别只作为 training-only privileged coefficient。
4. 从 hard same-component 阈值改为局部 capsule 内 continuous widest-path bridge risk。
5. 删除造成前景膨胀的 core BCE 和可能过切的 rank loss。
6. 窄缝可靠性由 y=0、X-ray valley/double-edge、增强存活和 annotation audit 联合决定。
7. 训练前设置 Gate A；失败时停止，不能放宽规则制造 anchor。

## Output files

- Final proposal: `FINAL_PROPOSAL.md`
- Review summary: `REVIEW_SUMMARY.md`
- Score history: `score-history.md`
- Round reviews/refinements: `round-*.md`

## Remaining implementation cautions

- widest-path 的 hop constraint 必须正确实现或由 capsule 几何替代；
- X-ray profile 参数只能在 train-only Gate A 冻结；
- 当前 READY 只授权 Gate A 和代码实现，不授权直接正式训练或 clean-test-v2。

