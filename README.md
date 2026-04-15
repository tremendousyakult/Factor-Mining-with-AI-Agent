# Beyond Prompting 本地复现包

这是一个**可本地运行**的 Python 包，用来复现论文 **Beyond Prompting: An Autonomous Framework for Systematic Factor Investing via Agentic AI** 的核心系统设计：受约束的候选因子生成、固定规则评估与 gatekeeping、memory-guided 闭环搜索、线性与 LightGBM 组合、decile 排序、turnover 与交易成本扣减，以及可审计的因子库与实验日志。

## 与论文高度一致的部分
- 生成 -> 执行 -> 评估 -> gate -> memory update 的闭环流程
- 日频股票面板、严格 IS/OOS 时间切分
- rank IC / t-stat / long-short Sharpe 评估协议
- 线性组合与 LightGBM 非线性组合
- 交易成本与 turnover 诊断

## 本包新增的稳健性增强（相对基础版）
- 评估 gate 支持最小覆盖天数、最小日截面样本约束，减少偶然样本通过
- agentic 模式支持 promoted 因子的相关性去重（防止同质化因子库）

## 论文未完全公开、因此本包采用透明近似的部分
- 论文 Table IX 给出名称和经济解释，但没有公布 12 个最终因子的逐字符公式；本包提供 `paper_seed` 模式做可审计近似实现
- 论文没有公开完整 LLM prompt 与所有内部搜索策略；本包默认使用**离线可运行**的 heuristic symbolic agent，不需要远程 API
- 论文 Appendix C 描述了 LightGBM 与时间序列验证，但没有给出完整超参数表；本包提供可配置默认值

## 安装
```bash
cd beyond_prompting_repro
pip install -e .
```

## 最快跑通方式
```bash
agentic-factor make-demo-data --out data/demo_panel.parquet --n-assets 120 --start 2018-01-01 --end 2024-12-31
agentic-factor run --config configs/demo_small.yaml --data data/demo_panel.parquet --out runs/demo --report
```

## 三种 discovery mode
- `agentic`：默认闭环搜索
- `paper_seed`：直接加载根据 Table IX 推断的近似 12 因子
- `traditional_baseline`：静态 one-shot baseline，不做 memory update

## 输出物
- `factor_metrics_is.csv`：完整筛选表
- `round_logs.jsonl`：审计日志
- `promoted_library.jsonl`：最终因子库
- `linear/` 与 `lgbm/`：组合层 decile、spread 与季度成本诊断
- `report.md` 与 PNG 图：自动报告

## 免责声明
这是**论文方法复现包**，不是作者官方代码镜像，也不是投资建议。它的目标是忠实复现论文的系统结构与实验协议，并在论文未公开全部实现细节处使用透明、可配置、可审计的近似实现。


## 附带样例

仓库里已经附带一个 `sample_panel.csv` 和一个成功跑完的 `example_run/`，你也可以直接执行：

```bash
./run_sample.sh
```
