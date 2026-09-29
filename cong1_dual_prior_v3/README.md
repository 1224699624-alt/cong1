# cong1 双先验 Codex 交付包 · V3 (SUPERSEDED / 历史协议)

> **协议状态（2026-09-29）**：`SUPERSEDED`。
> 本包（2026-09-19）记录了双先验多任务耦合联合训练的历史探索。后续实验已证实多先验直接强耦合会导致梯度冲突与负向迁移。
> **当前唯一有效研究入口请参见**：
> - [`research-wiki/experiments/ram_so_heatmap_20260929.md`](../research-wiki/experiments/ram_so_heatmap_20260929.md)
> - [`codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md`](../codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md)

**日期：2026-09-19。历史范围：TSRS + RAM，nnU-Net。**

## 主线

两个数据集都建立/适配骨缝先验与骨重叠先验：**骨缝先验作用于模型输入端，骨重叠先验作用于模型输出端，以统一局部损失和梯度协调减轻冲突。** 单张图可同时包含两类问题。TSRS 侧重骨缝、RAM 侧重骨重叠，是用户的研究设定；真实占比、特别是 TSRS 真实 overlap 占比尚不能据现有记录量化。

本包替代此前 `TSRS_Dual_Prior_V2_1_Codex.zip` 及其中 TASK、配置、启动提示词。旧包的“输出补全 / 强制 OOF + crowding”核心定义不再执行；不要混用两个版本。

## 阅读顺序

1. `REPO_AUDIT.md`：已核查的仓库事实、问题及证据边界。
2. `TASK.md`：给 Codex 的实施要求、局部监督和阶段验收。
3. `ACCEPTANCE_TESTS.md`：必须实现/运行的测试。
4. `config/dual_prior_v3.template.yaml`：双数据集协议模板。
5. `CODEX_START_PROMPT.txt`：直接交给 Codex 的入口提示词。

`SOURCE_INDEX.md` / `SOURCE_INDEX.json` 记录固定 commit 下实际读取的文件与范围。`evidence/` 保存本轮确实运行过的 CPU 梯度反例及对应只读源码快照。`tools/` 只提供协议检查和回归探针，不是完成的训练器。

## 本轮实际完成与未完成

完成：GitHub 代码/记录静态审计、已校验源码副本上的 CPU 合成梯度反例、交付文件与配置自检。

未完成：服务器文件存在性检查、真实数据标签审计、checkpoint 张量加载、真实 batch、训练或评估。仓库记录的指标不是本轮重新运行的结果。公开仓库与用户本地/服务器工作树可能不同。

## 使用

把本目录放入项目的隔离任务目录，阅读项目 `AGENTS.md`，再把启动提示词交给 Codex。

```bash
python tools/check_protocol.py --config config/dual_prior_v3.template.yaml --stage preflight
python tools/probe_gradient_none_case.py --repo /path/to/cong1
```

`check_protocol.py` 依赖 PyYAML；梯度探针使用现有 PyTorch 环境，不要求升级或替换你的训练环境。本轮探针在 CPU PyTorch 2.10.0 上运行，历史训练环境另见 provenance。配置中的相对数据路径按 `--repo`（默认当前目录）解析。

第一条对空模板只验证设计约束，输出的 `TEMPLATE_VALID_WITH_BLOCKERS` **不是训练已获准**。第二条会导入指定仓库的 R349 路由函数并运行两个标量参数的 CPU 测试；旧缺陷未修复时应返回非零退出码。它不读数据，不训练模型，不写仓库。

本包不含模型权重、医疗数据、登录凭据或新的实验成绩，不会自动提交 GitHub。

## 验证与完整性

`PACKAGE_VALIDATION.json` 记录本轮交付文件/配置与 CPU 探针检查；它不是模型验收。`CHECKSUMS.sha256` 为包内文件的 SHA256（不包含校验表自身）。训练相关真实验证状态仍保留为 NOT_RUN/待核，不因包检查通过而置为 PASS。
