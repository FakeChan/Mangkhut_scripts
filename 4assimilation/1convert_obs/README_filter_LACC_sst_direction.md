# 等权 LACC 的 NR 辅助 SST 更新方向筛选

脚本：`filter_LACC_sst_direction.py`；驱动接入：`run_single_assimilation_test.sh`
（新增开关 `USE_NR_SST_DIRECTION_FILTER`，默认 0）。

只处理**等权** LACC 观测集，不涉及自适应（`OptimizWeight=1`）与生产 cyclingDA。
输入解析、F-order、中心时刻与空间采样校验复用同仓库已有的
`plot_scripts/diag_LACC_lag_cov_innovation.py`（该文件未被修改）。

**本次开发只使用本地合成数据验证，未运行服务器真实数据，也没有复现真实观测编号的筛选结果。**

## 判据（数学定义）

对原始观测编号 i、集合成员 m：

| 符号 | 含义 |
| --- | --- |
| `x_m` | 中心时刻背景成员 m 的 `OM_TMP[0, 0, :, :]` 在观测 i 位置的插值值 |
| `h_m` | 成员 m 已生成的**最终等权 LACC Hx 文件**中的第 i 条值 |
| `y` | 已经生成、已经加噪声的最终 LACC 观测文件的第 i 条值 |
| `x_NR` | 中心时刻 NR 的 `OM_TMP[0, 0, :, :]` 在同一观测位置的插值值 |
| `M` | 成员数，至少为 2 |

```text
x_bar = mean(x_m)
h_bar = mean(h_m)
c     = sum((x_m - x_bar) * (h_m - h_bar)) / (M - 1)
d     = y - h_bar
e     = x_NR - x_bar
score = c * d * e

score <  0  -> 反方向，删除整条观测
score >= 0  -> 不反方向，保留
final_mask  = original_clear_sky_mask AND direction_keep
```

实现细节（固定，不提供开关）：

1. 这是**方向判据**，不是“更新后误差是否减小”的判据；没有过冲筛选、相关系数阈值或显著性检验。
2. `c = 0`、`d = 0`、`e = 0` 全部保留（状态 `kept_zero_direction`）。
3. 成员值完全相同时，显式令离差与协方差为**精确 0**：相同的十进制开尔文值在二进制浮点里求平均并不精确，舍入残差会凭空造出集合并不存在的协方差。
4. 先做有限性检查，再用 `c`、`d`、`e` 的**符号组合**（奇数个负号且无零因子）判定反方向，避免三者乘积下溢为 0 而漏删；`score` 仅作诊断字段输出。
5. 全程 float64；`M >= 2`（ddof = 1）。
6. 成员严格按编号配对：背景 `mem001` 对 Hx `mem001`，依次类推；两侧都不排序、也不跳过缺失成员。
7. 不重新平均各滞后 Hx、不重新生成观测噪声、不用 NR 无噪声亮温替代 `y`；不需要增量幅度，不引入 R 或局地化参数。
8. 使用共同的同化前背景做**一次静态筛选**；不实现串行同化过程中重新计算筛选。
9. 均值、协方差、创新、NR 误差及诊断乘积计算后再次检查有限性；溢出或 NaN 明确报错，不进入符号判断。乘积下溢为 0 仍按各因子符号判断。

### 适用范围（不要过度解读）

- 单观测海洋增量近似为 `Delta T_o ~ c / (sigma_hx^2 + R) * d`，分母为正，所以只有**符号**是可声称的：该判据是**单观测线性均值增量**的方向代理。
- **不保证**多观测串行同化后实际增量的方向；**不宣称**等价于任意非线性或分位数/自适应滤波器的实际更新。
- 方向正确 ≠ 最终误差更小；过冲但方向正确的观测会被保留，脚本不作任何技巧性声明。

## 在驱动中启用

```bash
USE_NR_SST_DIRECTION_FILTER=1 ./run_single_assimilation_test.sh
```

开启时必须同时满足（否则在**任何数据生成阶段之前**报错退出）：

```text
EXPERIMENT_MODE=LACC   OptimizWeight=0   USE_CLEAR_SKY_MASK=1
RUN_OBS_CONVERT=1      RUN_HX_MERGE=1    RUN_VALIDATION=1
```

理由：新掩膜必须与同一次运行中重新生成的 `obs_input`、`merged_FO` 和计数校验配套，否则会出现“新掩膜 + 旧产物”。

接入位置：真值与集合 Hx 生成之后、stage 4 观测文本转换之前（stage 3b）。流程为

1. 保留原始 `CLEAR_SKY_MASK_FILE` 路径（每次都作为筛选输入，**不会**把上次合并掩膜当原始输入）；
2. 调用筛选器，输出合并掩膜；
3. 把驱动的 `CLEAR_SKY_MASK_FILE` 指向合并掩膜并 export；
4. 之后的 `obs2DART_LACC.py`、`merge_FO_in_one_file.py` 和计数校验全部使用这份掩膜。

开启时驱动器工作目录与最终 obs_seq 名都追加 `_nr_sst_direction`（已带该后缀时不重复追加），
后缀在计算最终路径和覆盖检查之前处理，`OVERWRITE_OBS_SEQ` 的语义不变；
关闭时路径、文件名与输出与基线完全一致。

驱动侧只有一个很短的 Python 桥接段：它把已解析的驱动参数（通过环境变量传递，不把 shell 路径拼进源码）
显式构造成 `Config` 再调用 `run(config)`；`importlib` 以**绝对路径**加载模块，计算逻辑全部在模块里。

## 输入（服务器默认路径）

| 输入 | 默认路径或约定 |
| --- | --- |
| 项目根目录 `ROOT` | `/share/home/lililei1/kcfu/tc_mangkhut`（驱动中的 `BASE_DIR`） |
| 最终带噪声 LACC 观测 | `ROOT/3create_obs/hx_rttov/3obs_BT_LACC/AMSUA/BT_LACC_10_00_00/obs_d01_ch4_totalline_withpert.txt`（严格 `(676, 1)`，取 `[:, 0]`） |
| 观测文件来源 | 驱动解析一次的 `OBS_INPUT_FILE`；筛选器与 `obs2DART_LACC.py` 读同一个文件 |
| 最终成员 LACC Hx | `ROOT/3create_obs/hx_rttov/4ens_BT_LACC/memNNN/AMSUA/BT_LACC_10_00_00/obs_d01_ch4_totalline.txt`（676 条）与其矩阵 `obs_d01_ch4.txt`（26×26） |
| 原始晴空掩膜 | `.../BT_LACC_10_00_00/clear_sky_mask.txt`（676 行 0/1，只读，不修改） |
| 观测经纬度 | `ROOT/3create_obs/hx_rttov/profile/profile_d01_LACC_10_00_00/prof10_00:00.dat`，用现有 `read_profile_coordinates()` 解析，保持原始顺序 |
| LACC 时次 | `.../BT_LACC_10_00_00/LACC_times.txt`，校验中心时次、时次不重复、与驱动窗口一致（不硬编码为 3 个时次） |
| 背景海洋集合 | `ROOT/4assimilation/0mem_all_time/cyclingDA/10_00_00/firstguess_d01.memNNN`（成员模板 `firstguess_{domain}.mem{mem:03d}`） |
| NR 海洋 | `ROOT/NR_wrfout/2domain/wrfout_d02_2018-09-10_00:00:00` |

背景与 NR 都只读取 `OM_TMP` 的**第一个时间、第 0 层**，不读完整 30 层变量。
背景目录、成员模板、NR 文件与中心时刻时间串都有独立配置项（`NR_SST_BG_DIR`、
`NR_SST_BG_MEMBER_PATTERN`、`NR_SST_NR_FILE`、`NR_SST_CENTER_TIME`），**不复用** ADAPTIVE 命名的字段。

> **背景集合必须与生成当前 Hx 的成员对应。** 文件存在与格式正确并不能证明它一定是本次实验的同化前背景；
> 代码只使用用户明确配置的目录，不会自动寻找其他 firstguess/preassim 文件替代。

## 采样与输入检查

- 复用现有 `read_level_zero_field()` / `sample_point()` / `validate_background_consistency()`；
- 线性插值、水点约束开启、局部窗口 1°、最大 4°，**不外推**、不自动回退最近邻；
- 水陆判定优先 `LANDMASK`（0 为水），缺失时用 `XLAND`（2 为水）；
- 背景与 NR **分别**在同一观测经纬度插值；NR 与背景网格尺寸不同，禁止用相同数组下标对应；
- 每个成员文件只打开一次用于读取该层场，随后对该成员完成**所有**待筛选观测的采样后关闭（不对每条观测重新打开）；NR 文件同样只读一次。为检查 `OM_TMP:units` 兼容 K，会对同一文件再做一次**只读属性**的打开（不触碰数据）；
- 检查所有成员与 NR 的 `Times` 等于中心时刻、成员之间网格与水陆掩膜一致、经纬度有限且纬度合法、温度单位兼容 K；
- 背景成员文件必须互不相同：拒绝重复路径，以及软链接或硬链接指向同一文件的成员，错误信息列出成员号和文件路径；
- 在有局部缺失值时，先保持水点几何不变，检查最近海洋格点及实际插值支撑点；缺失点承担正插值权重时终止。远处或不参与该点插值的缺失格点不导致失败，也不会通过删点重新构造插值支撑；
- `state_var="OM_TMP"`、`state_level=0`、`sample_mode="linear"`、`sample_require_water=True`、`require_center_time_match=True` 是强制定义，配置校验拒绝其他值；
- 对原始掩膜为 1 的观测，若遇到非海洋点、无法插值、非有限输入、成员文件/矩阵缺失、时刻不符、网格不一致或 F-order 不一致，报出**路径、成员号与原始观测号**并终止；不会静默保留、删除或用其他点代替；
- F-order 检查（列文件必须是本成员本目录矩阵的 F-order 展平）**无条件强制执行**，本任务不增加放宽开关。

## 输出（独立筛选目录 `.../nr_sst_direction_filter/`）

```text
clear_sky_and_nr_sst_direction_mask.txt   固定 nobs 行、每行 0/1、结尾换行
nr_sst_direction_audit.csv                对全部原始观测逐条输出
nr_sst_direction_summary.json             严格 JSON（allow_nan=False，缺失为 null）
```

三份文件都是先写临时文件再原子替换，且只在**输入检查与计算全部成功之后**发布；失败必须非零退出，驱动立即停止。
原始观测文件、成员 Hx、原始晴空掩膜与所有 NetCDF 只读不写。

CSV 列（固定顺序，全部原始观测逐条、按原始顺序）：

```text
raw_obs_id_1based  base_filtered_id_1based  final_filtered_id_1based  latitude  longitude
original_mask  status  sst_background_mean  sst_nr  sst_error_nr_minus_background
hx_mean  actual_observation  innovation  cov_sst_hx  direction_score  keep_final
```

`status` 只有四种取值：`excluded_by_original_mask`、`rejected_reverse`、
`kept_nonreverse`、`kept_zero_direction`。原掩膜为 0 的行不计算统计量（统计字段留空），
没有对应的筛选后编号时留空（不用 0 冒充有效编号）。

`nr_sst_direction_summary.json` 记录：全部实际输入路径与配置、中心时刻与窗口时次、成员编号、
SST 变量与层号、采样方式、三类计数（原始排除 / 反方向删除 / 最终保留）、
F-order、时间与成员网格检查结果、静态方向筛选的数学定义与适用范围、掩膜与审计输出路径。

**所有观测都被删除时**：审计、汇总 JSON 和（全 0 的）掩膜仍会写出，随后 `run()` 抛出
`NoObservationSurvivesError`，驱动在 `text_to_obs` 之前明确终止——空观测集是筛选的失败，不是结果。

## 本地测试

只使用本地合成数据，不连接服务器、不读取真实数据、不运行 RTTOV/MPI/DART；
驱动调用测试用本地桩替代 `text_to_obs`。

```bash
/Users/kcfu/miniforge3/bin/python -m unittest \
    4assimilation/1convert_obs/tests/test_filter_LACC_sst_direction.py -v
```

覆盖内容：`c/d/e` 全 27 种符号组合（含 c<0 且 d<0 的正确方向）、零因子与相同小数成员不产生假协方差、
`c*d*e` 下溢仍按符号删除、同方向过冲保留、使用带噪声 `y`（构造无噪声与带噪声创新符号相反）、
只使用最终 Hx（滞后数据相差约 1000 K 以暴露重算）、按成员号配对、F-order/单列形状/成员缺失失败、
时间错误/网格不一致/非有限值/非单位 K 失败、背景与 NR 不同网格下线性温度场的解析值、
非海洋点与无法插值失败、原掩膜排除点不可复活与三种编号映射、仅剩 1 条观测、
全部删除的审计输出、两个转换器逐行对应、开关关闭时路径不变、开启时路径隔离、
不兼容开关组合的拒绝、失败后驱动不会继续。

## 已知限制

- `nobs = 1`（即 `matrix_side = 1`）会被拒绝：1×1 矩阵文件经 `np.loadtxt` 读回是 0 维数组，
  复用的 F-order 检查无法确认编号，而本任务不增加放宽该检查的开关。服务器规模固定为 676（26×26），不受影响。
- 筛选器只读取中心时刻的 `OM_TMP` 第 0 层；若未来改用 `TSK` 或其他 SST 变量，需要显式修改配置并重新验证，
 需修改脚本的固定定义校验并重新验证，不能仅修改配置或环境变量替换。
