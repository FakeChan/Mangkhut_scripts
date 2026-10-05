# verify_diag — 强弱耦合海温-通量误差诊断套件(SYNTHETIC 测试版)

本目录包含三个独立诊断脚本及其公共模块、合成数据生成器和冒烟测试,用于在
已有 WRF 输出上核验"海洋温度误差减小,但感热和潜热通量误差没有同步减小"
这一现象。**当前状态:仅完成代码实现与本地合成数据测试;真实数据接口、
物理实现与数学归因均未经核实**(见 `MATH_REVIEW_CHECKLIST.md`、`TEST_RECORD.md`)。

## 一、文件清单

| 文件 | 作用 |
|---|---|
| `verify_common.py` | 公共模块:配置 dataclass、纯数组诊断函数、NR 配准、成员汇总、CSV/绘图输出、真实模式数据提供者(未测试) |
| `verify_01_skill_timeseries.py` | 诊断一:完整误差时间序列 + 海温-通量逐点空间分类 |
| `verify_02_flux_error_budget.py` | 诊断二:通量误差变化的精确分解(交叉项/平方项/闭合残差) |
| `verify_03_fixed_atmosphere_flux.py` | 诊断三:固定大气海温替换(阶段 A 重建一致性 + 阶段 B 条件替换) |
| `verify_synthetic.py` | 合成数据生成器(SYNTHETIC ONLY,固定种子,内存数据,不触任何真实文件) |
| `verify_04_initial_handoff.py` | 诊断四:初值传递核验(B→A→I,及独立 F0;d_assim/d_handoff/d_pair 三组差值严格分名);默认真实模式,B/A/I/F0 路径已于 2026-10-05 服务器只读核验并填入配置(绑定 10_00_00 循环) |
| `verify04_readers.py` | 小型 NetCDF 读取适配:按维度名称与 Times 属性定时间、网格一致性校验、无插值 |
| `caliber_link.py` | 模块 B:pathway vs verify_diag 口径对照表与连接兼容性判定(unknown 默认) |
| `REVIEW_SOURCES_AND_CALIBER.md` | 代码层审查:四阶段来源关系(含证据行号)、update_tsk_from_omtmp 审查、口径差异 |
| `run_verify_diag.sh` | LSF 串行作业提交脚本(RUN_VERIFY01/02/03 开关;verify_04 未纳入,单独运行) |
| `verify_smoke_test.py` | 两层冒烟测试 + 三个诊断的合成全流程测试 |
| `outputs/SYNTHETIC_*/` | 本次合成测试的全部输出(明细 CSV、汇总 CSV、图件、逐点场 npz、列说明) |
| `TEST_RECORD.md` | 本次测试记录(解释器、实际测试项、通过项、未验证项) |
| `MATH_REVIEW_CHECKLIST.md` | 供数学审查的清单(公式↔函数对应;待核实的物理与数据约定) |

依赖:Python(netCDF4、numpy、pandas、matplotlib;诊断三另需能导入
`Mangkhut_scripts/plot_scripts/omtmp_flux_decomposition.py`)。
本次测试使用解释器:`/Users/kcfu/miniforge3/envs/wrf/bin/python`(Python 3.14.5)。

## 二、运行顺序(Mac 本地,合成模式)

```bash
cd /Users/kcfu/works/nju/tc_mangkhut/Mangkhut_scripts/plot_scripts/verify_diag
PY=/Users/kcfu/miniforge3/envs/wrf/bin/python

# 1) 冒烟测试(第一层纯数组函数、第二层真实物理函数接口、三个诊断合成全流程)
PYTHONDONTWRITEBYTECODE=1 $PY verify_smoke_test.py

# 2) 三个诊断正式合成运行(各自写入 outputs/SYNTHETIC_<名称>/)
PYTHONDONTWRITEBYTECODE=1 $PY verify_01_skill_timeseries.py
PYTHONDONTWRITEBYTECODE=1 $PY verify_02_flux_error_budget.py
PYTHONDONTWRITEBYTECODE=1 $PY verify_03_fixed_atmosphere_flux.py
```

所有脚本无命令行参数、无环境变量依赖(仅可选 `PYTHONDONTWRITEBYTECODE=1`
抑制 pyc 缓存);所有可编辑参数集中在每个脚本顶部的"可编辑配置区"
(使用 dataclass / 模块级常量)。诊断三运行时间约数十秒,其余为秒级。

## 三、符号与约定(三个诊断一致)

### 误差指标(verify_common.paired_error_metrics)

```
bias_exp   = mean(X_exp - X_NR)                     弱/强平均偏差
MSE_exp    = mean((X_exp - X_NR)^2)                 等权有效格点平均
RMSE_exp   = sqrt(MSE_exp)
mse/rmse_change_strong_minus_weak = 强 - 弱         正值 = 强耦合恶化
rmse_improvement_pct = 100*(RMSE_weak - RMSE_strong)/RMSE_weak
                                                    正值 = 改善;RMSE_weak=0 时置 NaN(status=zero_rmse_weak)
```

### 空间分类(verify_01)

逐点 `dSE_X = (X_strong - X_NR)^2 - (X_weak - X_NR)^2`,正值 = 强耦合该点恶化。
对 (tsk, hfx)、(tsk, lh) 两组按容差交叉分类:

| 代码 | 含义 |
|---|---|
| 1 | 海温改善 & 通量改善 |
| 2 | 海温改善 & 通量恶化 |
| 3 | 海温恶化 & 通量改善 |
| 4 | 海温恶化 & 通量恶化 |
| 0 | 双侧未变化(\|dSE\| ≤ 容差,单独处理) |
| -1 | 单侧未变化(单独处理) |

判定容差在配置区(`sst_se_tol = 1e-4 K^2`、`flux_se_tol = 1.0 (W m-2)^2`),
是可调的判定参数,非物理常数。容差直接作用于 dSE;由于 dSE = 2e·dF + dF²
含交叉项,**不能把平方容差开方解释为普适的海温/通量增量阈值**。
**区域平均的海温改善不能直接解释为同一格点上的通量恶化**——逐点分类即为此设计。

### 误差分解(verify_02)

```
e  = F_weak - F_NR          (未去均值的误差乘积;不得替换为协方差)
dF = F_strong - F_weak
dSE(逐点)   = 2 e dF + (dF)^2
delta_mse_direct      = MSE_strong - MSE_weak = <dSE>
cross_term            = 2<e dF>          (逐点乘积的平均;不是 <e><dF>)
increment_square_term = <(dF)^2>
closure_residual      = delta_mse_direct - (cross_term + increment_square_term)
```

解释标签保持克制:净误差减小 = 通量误差改善;交叉项非负 = 区域整体变化
方向不利;交叉项为负但净误差增大 = 存在纠错倾向但净效果不利;净变化在
容差内(`BUDGET_UNCHANGED_TOL`,可配置)= unchanged。这只是误差预算描述,
不自动等同于物理因果结论。改善率的可用性单列于 `improvement_status`
(零基准 RMSE 时置 NaN,其余绝对误差指标仍然有效);主汇总表含
mean_cross_term / mean_increment_square_term / mean_closure_residual,
可直接复核闭合。

### 固定大气替换(verify_03)

```
F_ww = f(T_weak,  A_weak)      F_sw = f(T_strong, A_weak)
F_ss = f(T_strong, A_strong)   F_ws = f(T_weak,  A_strong)   (可配置第四组合)

dF_SST           = F_sw - F_ww      只替换海温的响应
dF_rest          = F_ss - F_sw      替换其余输入后的响应(不直接宣称纯大气反馈)
dF_reconstructed = F_ss - F_ww     = dF_SST + dF_rest(恒等式,残差应≈0)
actual(实际模式通量差) = 强 - 弱,统一经 vc.strong_minus_weak 计算
```

A = 除被替换表面温度之外的全部重建输入,不仅包括温湿风,还包括摩擦速度
初值(UST)、动力粗糙度初值(由 UST 诊断)、离地高度、气压等辅助输入。
固定其余输入只替换表面温度时,物理函数按自身算法重新计算稳定度和交换系数,
不人为固定函数内部计算量。正反两条路径的 dF_SST 一般不同(`path_split_difference`),
结果具有路径依赖,单一路径的结果不是唯一贡献。

归因可信度:阶段 A 的质量标记(主判据为残差 RMS 与模式强弱差 RMS 之比,
辅以最大绝对差对 RMS 之比,均用 RMS 尺度避免正负抵消)。阶段 B 的
`attribution_flag` 在**每个区域的最终比较掩膜**上用三个必要条件逐一评估:
弱重建残差比、强重建残差比、重建差-实际差比(分母均为该区域模式强弱差
RMS;判据数值保存在 `attr_*_ratio` 列)。只检查弱耦合会漏掉"弱准强错"
的情形,三条件全部满足才给 residual_small_vs_explained_diff;图件标题附
标记分布。阶段 B 另报告逐点 `(F_ss-F_ww)-(模式强-弱)` 的 RMS 与最大绝对差。
阶段 B 的比较在"重建输出 + 模式通量 + NR 通量"的共同有限掩膜上进行,
非有限输出点数**按区域**统计(`n_nonfinite_output`),状态
`status=nonfinite_reconstruction_output` 只影响实际存在无效点的区域;
区域掩膜为空时 `improvement_status`/`attribution_flag` 等状态字段保持显式
取值(不被 NaN 默认值覆盖)。重建输入先做物理有效范围预检(气压/高度/
摩擦速度/粗糙度为正),失败记录为 `reconstruction_failed`;阶段 A 中一组
重建无效导致差值共同失效时,状态记为 `nonfinite_reconstruction_diff`。

阶段 A(重建一致性)的 `quality_flag` 只是可信度标记(阈值可配置),
**不构成"物理验证通过"**;真实运行若残差相对强弱通量差不可忽略,
归因结果必须标记待核实。表面温度默认取试验自身 `TSK`
(`SURFACE_TEMP_SOURCE="TSK"`);用 OM_TMP 第一层代替 TSK 的做法未经核实、默认禁用。

### 条件子集预算(verify_02 新增)

dSE_T = (T_s-T_NR)^2 - (T_w-T_NR)^2(默认 T = TSK,阈值 = sst_se_tol,单位 K^2)。
在"区域 & 海洋 & 强/弱/NR 的 TSK 与目标通量共同有效"掩膜内构造
`all_common` / `sst_improved`(dSE_T<-tol)/ `sst_worsened`(dSE_T>tol)/
`sst_unchanged` 四个子集(**只按海温误差变化筛选,不同时要求通量变化**)。
对每个子集、HFX/LH 复用同一预算公式(实际弱误差 e_model=F_weak_model-F_NR、
实际增量 dF_actual、C=2⟨e·dF⟩、S=⟨dF²⟩、ΔMSE=C+S)。
子集比例分母 = 同案例/区域/变量的 all_common 点数。
交叉项方向用独立容差 `CROSS_TERM_TOL`(单位 (W m-2)^2):接近零为
`cross_term_neutral`,不归入"方向不利"。**新增条件表独立成文件,
不替换原有全区域预算结果**(原结果的掩膜不要求 TSK 有效)。

### 联合诊断:离线 SST 响应 vs 实际通量变化(verify_03 新增)

在"区域 & 海洋 & mask_compare & 强/弱/NR 的 TSK 共同有限"的同一最终掩膜上
(全部指标同掩膜,不各自删点;含启用时的 Fws),逐子集计算:

- `cross_model_error_sst_response = 2⟨e_model·dF_SST⟩`,
  e_model = F_weak_model − F_NR(**实际弱误差**,不是离线重建弱误差
  e_recon = F_ww − F_NR;现有 `sst_only_rmse_improvement_pct` 用的
  是离线基线,两者不可互相替代);
- `rms_sst_response`、`rms_actual_response`、`rms_sst_minus_actual`、
  `sst_to_actual_rms_ratio`(实际响应严格为零 → NaN +
  actual_response_zero;低于分母下限 `SST_RATIO_DENOMINATOR_FLOOR`
  (W m-2,与符号一致率阈值分开配置)→ NaN + actual_response_below_floor,
  避免近零分母的极大比率主导平均);
- 符号一致率(仅双方 |增量| > `FLUX_INCREMENT_TOL`(W m-2)的格点参与,
  输出参与点数与占子集有效点数比例);
- 同掩膜上的实际通量 C/S/ΔMSE/闭合残差(actual_ 前缀列,与诊断二
  all_common 同掩膜行逐位一致);
- 点数三级口径:`n_input_valid`(输入掩膜)/`n_common`(重建输出后)/
  `n_tsk_common`(再加 TSK),重建与 TSK 造成的样本损失可分别恢复;
  `n_nonfinite_output` 按区域承载重建输出无效点数(部分失效时行仍为
  status=ok,全失败行为 nonfinite_reconstruction_output 而非 empty_mask);
- 子集质量的唯一标记是 `joint_status`(每个子集掩膜上重算三判据,空子集/
  不可评估行不继承全区域标记);`attribution_flag_region_ref` 仅为全域对照;
  质量不足时数值仍保存,但标记为不支持物理解释;
- 联合汇总附 `n_times_quality_ok`/`n_times_quality_failed`/
  `quality_coverage_ratio`(合格时次占数值有效时次的比例),
  质量不合格的时次不从统计中静默删除。

**解释边界**:`cross_model_error_sst_response` 只是离线响应相对于实际误差
方向的诊断,不是"实际 WRF ΔMSE 的 SST 因果贡献",不输出因果贡献百分比,
也不假定它与实际增量平方项相加后能闭合实际 ΔMSE。

### 通用口径

- 强、弱耦合与 NR 按有效时间严格匹配;NR 与试验网格通过统一配准入口
  `vc.register_nr_fields` 连接(三个诊断共用,策略一致):物理场用双线性
  (合成:规则网格;真实:默认 wrf.ll_to_xy 投影插值,已用 mock Mercator
  文件纳入冒烟测试,真实数据上待核实;可选 numpy_nearest);
  掩膜在真实模式默认用同一投影索引的 order=0 重采样,
  其余情形用**可复用空间索引最近邻**(cKDTree,按网格指纹缓存,复杂度
  O(N log N),不做全点对搜索)。覆盖判断 = 源网格足迹(规则网格包围盒/
  曲线网格凸包,容差半个格距)且距离不超过覆盖半径;"贴着边界但在域外"
  的目标点为 NaN,不自动套用边缘最近邻。
- 每个时刻、每个配对中,强弱试验使用相同 NR 中心(NR PSFC 海面最低点)
  和验证格点;区域与海洋掩膜逐时次固定(单时次上下文由
  `vc.build_time_context` 统一构造)。NR 与试验文件读取均校验文件内部
  Times 属性(兼容 WRF 的 S1 字符数组)与网格一致性。
- 海洋掩膜 = 试验 LANDMASK<0.5 且 NR(最近邻配准)LANDMASK<0.5;
  每项比较另取参与全部数组的共同有效值掩膜(|x|<1e30 且有限)。
- 主结果为**等权有效格点平均**(CSV `weight_method=equal_weight_valid_points`)。
- 区域:`r000_075` [0,75)、`r075_150` [75,150)、`r150_300` [150,300](闭)、
  `r000_300` 完整 0-300 km;边界不重复计数。
- EAKF 与 QCF_RHF 分开汇总;每成员先做强弱配对计算,再跨成员汇总。
- 方法级主汇总(`summary_by_method.csv`)只用完整 0-300 km 区域,
  不把并集与子环带混合平均;逐区域结果在 `summary_by_region_window.csv`
  保留。改善成员数在选定区域/窗口内按唯一成员计数(≤ n_members)。
- 覆盖率三级分列:已读取时次(mean/min/max_member_times_read)、
  有效绝对误差时次(status=ok 或 zero_rmse_weak,零基准的绝对误差视为有效)、
  有效改善率时次(改善率有限);分母显式给出(expected_times、
  expected_members),`member_abs_time_coverage_ratio` 与
  `member_coverage_ratio` 分别对配置时刻数与配置成员数。
- 两种改善口径严格区分(汇总 CSV 列名):
  - `mean_member_rmse_improvement_pct`:各成员相对改善率的平均值
    (成员先做窗口内时间平均,再等权跨成员);
  - `pooled_rmse_improvement_pct`:平均 RMSE 的相对变化。
- 空掩膜、全 NaN、零分母、缺失成员/时次均显式记录
  (结果行 `status` 列 + `verify0X_run_status.csv`),不静默跳过;
  诊断三的非有限重建输出点数按区域统计,只影响实际存在无效点的区域。
- 每个输出目录含 `overall_status.txt`:有效结果行为零时显式标记
  `no_valid_cases`,避免"程序正常结束"被误读为"诊断有有效结果"。
- 不做把格点或相邻时次视为独立样本的显著性检验。

## 四、输入输出

### 合成模式(默认)

输入:仅 `verify_synthetic.py` 内存生成的合成数据(固定种子 20260930,
33×33 试验网格 + 17×17 NR 网格,2 方法 × 6 成员 × 13 时次;含陆地、NaN 补丁、
缺失时次 QCF_RHF/044/t=2.5h)。生成方式与限制:合成分析场 = 解析真值 +
统计误差形态,合成低层大气输入按解析廓线加小扰动(强弱差 ~0.5 K / ~4% 风速,
成员间稳定度形态有别)生成;两组数据之间**没有通量方案层面的公式联系**,
只用于接口与流程测试,不能用于检验通量物理机制或重建一致性
(阶段 A 的残差天然不可忽略)。被测物理函数与生成器完全独立,无循环论证。

输出(每个诊断一个 `outputs/SYNTHETIC_<名称>/` 目录,全部行带 `mode` 列):

| 诊断 | 明细 | 汇总 | 图件 | 中间数组 |
|---|---|---|---|---|
| 01 | `verify01_member_metrics.csv`、`verify01_member_classification_fractions.csv` | `verify01_summary_by_method.csv`(仅并集)、`verify01_summary_by_region_window.csv`(逐区域) | RMSE/改善率时序(缺失时次断线)、分类地图 | `fields/verify01_dSE_*.npz` |
| 02 | `verify02_member_budget.csv`、`verify02_conditional_budget.csv`(TSK 改善/恶化/未变化子集的条件预算,独立表) | `verify02_summary_by_method.csv`(仅并集,含预算闭合列)、`verify02_summary_by_region_window.csv`、`verify02_conditional_summary_by_member/_method.csv` | 预算时序、闭合残差、分量柱状(逐区域,成员先均) | `fields/verify02_dSE_points_*.npz` |
| 03 | `verify03_phaseA_reconstruction.csv`、`verify03_phaseB_replacement.csv`(含 `attribution_flag`)、`verify03_joint_response_actual.csv`(离线 SST 响应 vs 实际通量变化联合统计) | (见各表内 `quality_flag`/`attribution_flag`/`joint_status` 等)、`verify03_joint_summary_by_member/_method.csv` | 阶段 A 残差时序(逐方法逐试验)、阶段 B 分量柱状(标题附归因标记) | `fields/verify03_replacement_*.npz`(含 Fww/Fsw/Fss/Fws、NR 通量、共同掩膜) |

每个输出目录另有 `config_snapshot.json`(配准方式、时间匹配方式等配置快照)。

每个目录另有 `columns_readme.txt`(列含义)与 `verify0X_run_status.csv`
(缺失/异常状态)。诊断一 `MAP_CASES`、诊断二 `POINT_FIELD_CASES`、
诊断三 `POINT_FIELD_CASES` 控制保存逐点场的案例,可按需增删。

### 真实模式(本次未运行;切换方法)

1. 在脚本顶部配置区把 `CONFIG` 的 `mode` 改为 `"real"`,例如:

   ```python
   CONFIG = dataclasses.replace(CONFIG, mode="real")
   ```

2. 同时把 `real=RealPathConfig(acknowledge_real_mode=True, ...)` 显式写入 CONFIG
   (防呆开关,缺省 False 会拒绝运行)。
3. 逐项核对 `RealPathConfig` 与 `MATH_REVIEW_CHECKLIST.md` 中全部 TODO(待核实)
   条目(路径结构、NR 半小时输出、OM_TMP 维度、dx、isftcflx、sfclay 方案版本等)。
4. 真实模式的读取与配准代码位于 `verify_common.RealWrfProvider`
   (路径结构、NR 时间插值、ll_to_xy 配准,口径取自 omtmp_skill_extract.py),
   **从未在真实数据上执行**;首次真实运行务必先用单成员单时次小样例试跑,
   并人工核对中间数值。

真实路径线索(本次禁止访问,仅作配置记录):预报
`/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/{试验}/{方法}/{成员}/wrfout_d02_时间`;
NR `/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/wrfout_d02_时间`。

### 初值传递核验(verify_04)

核验"同化前背景 B → 同化后分析 A → WRF 实际启动状态 I"的来源关系与数值差异,
F0(0h 预报输出)为独立阶段只做描述性比较。三组差值字段名严格区分:

    d_assim   = A - B   (stage_pair=A-B;同化增量)
    d_handoff = I - A   (stage_pair=I-A;交接差,或 I - M(A))
    d_pair    = X_strong - X_weak(逐阶段;initial_pair 表,stage 列)

- 传递映射 M = `ncks_updateDARTvar.sh` 变量清单(MU,OM_TMP,OM_U,OM_V,OM_S,
  P,PH,QVAPOR,THM,U,V,W):仅这些变量把分析值带入启动状态;M 外变量
  (如 TSK)的 d_handoff 标注 `mapping_unverified`,数值照常输出,
  不判通过/失败——"OM_TMP 改变而 TSK 未变"可能是设计行为(update_tsk
  流程未启用),不是错误;
- 读取按维度名称与 Times 属性定位(缺失时才回退显式记录号);形状或
  XLAT/XLONG 不一致即 shape_mismatch/grid_mismatch,**无插值掩盖**;
- 每项输出差异均值/RMS(差值均方根,非标准差)/最大绝对差/超容差计数与
  比例/有效与缺测点数/来源与阶段/单位/映射依据;
- 真实模式 B/A/I 路径默认未配置(文件名推断见
  REVIEW_SOURCES_AND_CALIBER.md,confirmed=False):使用前逐项确认并填
  `STAGE_SOURCE_DEFAULTS`,未配置的阶段只跳过依赖它的比较;
- 输出:`verify04_source_manifest.csv`(来源清单)、
  `verify04_pairwise_checks.csv`(阶段差)、`verify04_initial_pair_checks.csv`
  (强弱初值差)、`verify04_coverage.csv`(预期/实际/缺失)、
  `verify04_status.csv`、`caliber_compat_table.csv`(口径判定)。

### 口径衔接(caliber_link)

`caliber_compat_table.csv` 逐维对照 pathway 与 verify_diag(中心定义、空间
支持、时次集合、海洋掩膜、离线通量的 SST/大气来源、指标语义等),判定
`value_by_value / parallel_evidence / not_comparable / unknown` 四档,默认
unknown。`check_joinable()` 对两张结果表按键逐组判定:口径全同才可逐值
比较;块平均与逐点、不同中心、不同 SST 来源一律 not_comparable;元数据
缺失不默认 compatible;重复键标记 duplicate_keys;成员前导零不做静默归一。
两套离线通量结果(pathway 旧口径与 verify03)是**两个不同的替换试验**,
只能并列展示。

## 五、本次合成测试结论的边界

合成测试只覆盖:Python 语法与导入、无副作用导入、数组维度与基本接口、
输出路径与文件生成、合成流程能否运行完成,以及若干解析小数组的恒等式检查
(闭合残差 ~1e-12 量级)。它**不构成**对物理实现、数学归因或真实数据接口
正确性的验证;相关未决问题见 `MATH_REVIEW_CHECKLIST.md`。
