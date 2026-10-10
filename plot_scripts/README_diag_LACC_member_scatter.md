# LACC 成员散点与协方差稳定性诊断

脚本：`diag_LACC_member_scatter.py`。需要同目录已有的
`diag_LACC_lag_cov_innovation.py`；输入解析、成员编号、F-order、中心时刻和空间采样校验复用该脚本。
本次开发仅检查了服务器目录结构；功能、绘图与回归测试全部使用本地合成数据。

## 配置与运行

编辑脚本顶部的 `INPUTS` 与 `CONFIG`，然后直接运行脚本，无命令行参数。
依赖：NumPy、SciPy、netCDF4、Matplotlib；不需要交互图形界面。

```bash
python plot_scripts/diag_LACC_member_scatter.py
```

已填入核对过的实验路径：

| 输入 | 默认路径或约定 |
| --- | --- |
| 项目根目录 | `/share/home/lililei1/kcfu/tc_mangkhut` |
| 中心海洋背景 | `/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART/EAKF/obs_seq111/preassim_member_0001_d01.nc` 等成员文件 |
| 成员 Hx | 项目下 `3create_obs/hx_rttov/4ens_BT_LACC/mem001/AMSUA/BT_10_00_00/obs_d01_ch4_totalline.txt` 等成员/时次文件 |
| Hx 矩阵校验 | 对应目录 `obs_d01_ch4.txt`，要求 F-order 展平与列文件一致 |
| NR 海洋 | 项目下 `NR_wrfout/2domain/wrfout_d02_2018-09-10_00:00:00` |
| 中心时刻与成员 | `2018-09-10_00:00:00`，成员 1–50 |
| 观测与状态 | 沿用原脚本的 66、98 号观测，中心时刻 `OM_TMP`，第 0 层 |
| 输出 | 脚本目录下 `figs/LACC_member_scatter/` |

可在 `INPUTS` 设置 `obs_targets`、`member_start/member_end`、`state_lat/state_lon`、
`state_level`、`output_dir` 及原诊断的路径覆盖参数。默认抽样 2000 次配对 bootstrap、
500 次随机半集合，固定种子 `20261010`。`figure_formats=()` 可只导出数值。
真实同化窗口从 LACC times 文件确认；本脚本要求诊断包含完整窗口。
额外时次可以单独画面板，但不会进入窗口平均；缺少同化时次会终止。

## 数学定义与图形

每个观测分别画各滞后时次和完整窗口。横轴为成员 Hx，纵轴为同一组成员的
**中心时刻海洋背景**；各面板的海洋成员值保持不变。蓝点为成员，空心点为集合均值，
红星为 NR；成员旁编号是逐成员剔除时对协方差影响最大的实际成员编号。
晴空状态仅作标记，不自动剔除观测或时次。

令 x 为 Hx、o 为海洋状态，样本协方差采用分母 N−1：

- `c = cov(o, x)`，`r = corr(o, x)`。
- 回归海洋对 Hx：`slope = c / var(x)`，`o_fit(x) = mean(o) + slope × (x − mean(x))`。
- NR 垂直残差：`o_NR − o_fit(x_NR)`，单位 K。回归线只画成员 Hx 范围；
  `nr_hx_outside_member_range` 会标明 NR 是否涉及外推。
- `d_NR = x_NR − mean(x)`，`e_o = o_NR − mean(o)`；
  `alignment_NR = c × d_NR × e_o`，负值表示该无噪声方向背离 NR。
  这是方向诊断，不是实际含噪观测的分析增量或分析误差。
- 完整窗口先对每个成员的同化时次 Hx 等权平均，再计算统计量；
  NR Hx 也平均同一窗口。不会平均相关系数，也不会用各时次 `c_j × d_j` 的和替代窗口方向。

`regression_residual_std_K` 使用回归残差平方和除以 N−2。
`nr_vertical_residual_over_residual_std` 是描述性比值，不是显著性检验、后验标准差或预测区间。
Hx 方差为零时回归不可定义；任一变量方差为零时相关系数不可定义，JSON 写为 `null`。
NR 残差较大表明 NR 的偏差组合与当前集合回归关系不一致；不能据此单独判定协方差的物理机制。

## 稳定性检查

- **逐成员剔除（LOO）**：每次删掉一个成员，重算协方差和相关系数；报告
  `n_sign_flips`、`sign_flip_members`、协方差范围及影响最大的成员。
  翻转要求两次协方差严格异号；变为零单独记录为 `n_sign_zero` 和 `sign_zero_members`。
- **配对 bootstrap**：有放回抽 N 个成员；海洋与所有 Hx 时次使用相同成员下标。
- **随机子集合**：无放回抽取 `ceil(subset_fraction × N)` 个成员，最少 2 个，最多 N−1 个。
  默认 50 成员取 25 个；小集合实际大小写入输出。
- 各面板和观测使用相同抽样下标，便于比较时次；随机种子和实际成员编号均可追溯。
- `same_sign_fraction_valid` 是与完整集合协方差同号的比例；
  `negative_fraction_valid` 是负协方差比例。它们不代表正确更新或提高预报技巧的概率。
  分母包含所有协方差有限的抽样（`n_valid`），包括协方差为零、相关系数未定义的常数抽样；
  非有限协方差数量由 `n_invalid` 记录，相关系数未定义数量由 `n_correlation_undefined` 另行记录并显示在图上。
  相关系数的分位区间仅使用 `n_correlation_valid` 个可定义的抽样。
  完整集合协方差为零或无有效抽样时，同号比例为 `null`。
- 默认 95% 分位区间描述这组成员的重采样分布；没有证明成员独立，也不是跨独立试验的可信度保证。

建议先看成员形态和 NR 位置，再看 LOO 是否翻转、半集合是否仍同号、退化样本是否很多。
负协方差稳定只能排查“少数成员主导”，不能证明它会把当前海洋背景推向 NR。

## 输出与输入校验

| 输出 | 内容 |
| --- | --- |
| `lacc_member_scatter_obs*.png/.pdf` | 每观测各时次及窗口散点图 |
| `lacc_scatter_members_*.csv` | 每观测、面板、实际成员编号的 Hx 和中心海洋值 |
| `lacc_scatter_summary_*.csv` | 回归、NR 残差、方向与稳定性摘要 |
| `lacc_scatter_leave_one_out_*.csv` | 每个被剔除成员对应的协方差变化 |
| `lacc_scatter_resamples_*.csv` | 每次抽样的下标、成员编号、c、r、d_NR、e_o 与方向 |
| `lacc_member_scatter_summary_*.json` | 完整参数、统计、采样位置、文件路径及所有输入校验 |

输入不有限、成员编号重复、成员/NR 时刻错误、成员网格不一致或 F-order 不匹配都会终止。
默认 `strict_external_fo=True`：若观测序列中的 external_FO 可读但与载入成员窗口均值不一致，会终止。
无法读取该交叉校验时，在 JSON 留下 `unavailable`；不会猜测成员映射。
原脚本允许显式放宽矩阵缺失校验，本脚本会继承该设置并在警告和 JSON 中记录未验证项。
已知历史参考值不吻合会警告，并保留具体差异。

## 本地合成数据测试

在仓库根目录运行；测试生成临时目录与小型 NetCDF/Hx 文件，不连接服务器：

```bash
MPLCONFIGDIR=/tmp/lacc-mpl-local python -B -m unittest plot_scripts.tests.test_diag_LACC_member_scatter
```

覆盖回归方向、NR 残差、窗口平均与额外时次隔离、成员配对、影响成员识别、
退化抽样、成员子集编号、文件错序拒绝，以及 PNG/PDF/CSV/JSON 的端到端导出。
这些测试验证代码行为；真实实验结果需要在用户部署后读取。
