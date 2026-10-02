# verify_diag 测试记录(2026-09-30,本地 Mac)

## 一、环境与解释器

- 机器:本地 Mac(darwin 27.0.0,arm64)
- 解释器:`/Users/kcfu/miniforge3/envs/wrf/bin/python`
  (Python 3.14.5;numpy 2.4.6、pandas 3.0.3、matplotlib 3.10.9、netCDF4 1.7.4、
  scipy 1.17.1、wrf 1.4.2)
- 未安装/升级任何依赖;未连接任何服务器;未读取任何真实 WRF/NR/观测文件或
  既有缓存、诊断 CSV、NetCDF。
- 工作目录:测试时为 `/Users/kcfu/works/nju/tc_mangkhut/verify_diag`(项目根下);
  测试完成后整个目录移至
  `/Users/kcfu/works/nju/tc_mangkhut/Mangkhut_scripts/plot_scripts/verify_diag`,
  并在新位置复跑冒烟测试与三个入口验证通过(输出确定性一致)。全部输出写入
  本目录下的 `outputs/SYNTHETIC_*/`,并带 `mode=SYNTHETIC` 标记。
- 未修改任何原有脚本、物理模块、缓存与图件;`Mangkhut_scripts/plot_scripts/`
  的 `__pycache__` 中未出现新文件(导入复用了既有 pyc)。

## 二、实际执行的测试

### 第一层:纯数组诊断函数与输出流程(`verify_smoke_test.py`)

| 检查项 | 内容 | 结果 |
|---|---|---|
| paired_error_metrics 手工解析值 | 2×2 手工数组的 MSE/RMSE/偏差/改善率逐项对照解析值;掩膜版 | PASS |
| dSE 逐点恒等式 | `(s-t)^2-(w-t)^2 == 2e·dF+dF^2`(rtol 1e-9);NaN 传播 | PASS |
| flux_error_budget 闭合残差 | 分量与直接计算一致;cross_term 为 2⟨e·dF⟩ 而非协方差;残差 ≤ 1e-8·max(1,\|ΔMSE\|) | PASS |
| classify_change_categories | 7 个手工点覆盖类别 1-4、单侧/双侧未变化;NaN→-99;五类占比和为 1 | PASS |
| 零 RMSE 保护与空掩膜 | RMSE_weak=0 → 改善率 NaN + status=zero_rmse_weak;全 NaN → empty_mask(指标、预算、分类三处) | PASS |
| region_masks 边界与划分 | 75 km 归入第二环带、150 km 归入第三环带、300 km 含入并集、300.001 排除;三环带之并=并集、无重复计数 | PASS |
| resample_bilinear_regular | 线性场跨网格双线性精确复原(atol 1e-10);域外 NaN | PASS |
| derive_lowest_level_inputs | 常数场解析对照:温度 302 K、气压合成、离地高度 16 m(0/32 m 中点)、去交错风、诊断式 z0m=1.7522e-4 m | PASS |
| 输出流程 | CSV/npz/PNG 写出与回读 | PASS |
| 无副作用导入 | 子进程导入全部 5 个新模块(PYTHONDONTWRITEBYTECODE=1),父进程目录前后快照无新增文件 | PASS |
| 合成数据集行为 | NR PSFC 低压中心被精确识别于预置中心 (14.0N,138.0E);缺失时次识别;NaN 补丁;陆地通量 NaN/TSK 有效;推导输入形状/正性 | PASS |

### 第二层:现有真实物理函数接口冒烟(合成输入)

| 检查项 | 内容 | 结果 |
|---|---|---|
| reconstruct_ocean_fluxes 接口 | 导入真实模块 `Mangkhut_scripts/plot_scripts/omtmp_flux_decomposition.py`(无修改),以合成海洋格点输入(~1000 点,真实量级)调用:qfx/lh/hfx 全有限、量级合理(≥-250、<5000 W m-2)、同输入两次调用逐元素相等(equal_nan) | PASS |
| 方向观察(仅记录,非断言、不算验证) | 海温 +1 K 时 mean dLH=+12.3、dHFX=+0.99 W m-2(详见 smoke_test_report.txt) | 记录 |

### 三个诊断入口的合成全流程

| 检查项 | 内容 | 结果 |
|---|---|---|
| 完整流程 verify_01/02/03 | 各自 `run()` 在合成模式跑完并生成全部明细 CSV、汇总 CSV、run_status、columns_readme、图件、逐点场 npz | PASS |
| 合成明细覆盖 | 每方法 6 成员、13 个时次全覆盖(缺失时次显式记录于 run_status:QCF_RHF/044/t=2.5 → missing_member_or_time) | PASS |
| 阶段 B 通路 | 合成数据 A_weak≠A_strong → dF_rest 非零;拆分恒等式 dF_SST+dF_rest=dF_reconstructed 闭合(max 残差 ~1e-13);正反路径分裂差异非零(路径依赖可见) | PASS |

冒烟测试最终结果:**17/17 项通过**,报告见
`outputs/SYNTHETIC_smoke_test/smoke_test_report.txt`。

### 独立入口运行

三个脚本以 `python verify_0X_*.py` 方式独立运行成功,分别写入:
`outputs/SYNTHETIC_verify01_skill_timeseries/`(2480 行指标 + 1240 行分类占比)、
`outputs/SYNTHETIC_verify02_flux_error_budget/`(1240 行预算;max |closure_residual|
= 4.5e-12)、`outputs/SYNTHETIC_verify03_fixed_atmosphere_flux/`(2480 行阶段 A +
1240 行阶段 B;max |拆分恒等式残差| ≈ 1e-13)。

## 三、通过项(验收范围)

- Python 语法与导入;无副作用导入;
- 数组维度与基本接口;
- 输出路径与文件生成(CSV/图/npz/列说明/状态表);
- 合成流程能否运行完成;
- 若干解析恒等式残差(闭合残差 ~1e-12,仅供审查)。

## 四、未验证项(本次测试不能宣称的内容)

1. **物理实现正确性**:`reconstruct_ocean_fluxes()` / `wrf41_sfclayrev.py` 的
   物理公式未经本次测试验证(本次未修改、仅接口冒烟调用)。
2. **数学归因正确性**:分解、分类、替换归因的数学解释需后续审查者核实;
   本次仅验证恒等式在浮点意义上闭合。
3. **真实数据接口**:`RealWrfProvider`(路径结构、NR 时间插值、ll_to_xy 配准)
   的读取适配层已用本地合成 NetCDF 文件做过契约测试(入口方法齐全、时间/
   网格一致性校验生效),但**从未在真实数据上执行**;NR 半小时输出是否存在、
   OM_TMP 维度顺序、dx=1500 m、isftcflx=0、sfclay 方案版本等全部待核实
   (见 `MATH_REVIEW_CHECKLIST.md`)。
4. **科学结论**:合成输出中的任何数值(如 om 改善 ~72%、通量改善率、分类占比、
   阶段 A 质量标记分布)都是合成数据的表现形态,不代表真实试验结果,
   不应被引用为科研结论。
5. 阶段 A 的 `quality_flag` 阈值与分类容差均为可配置的判定参数,
   其取值是否合理需在真实数据量级下重新评估。

---

# 第二轮:响应外部代码审查(2026-09-30)

针对 `/private/tmp/SYNTHETIC_verify_diag_review_d8fco7ir/REVIEW.md` 指出的问题,
逐条核实并修复如下;全部修改仅以本地合成数据验收,未连接服务器、未读取真实
科研数据、未修改原有物理模块。

## 已修复的必改项(按审查编号)

1. **[P1] 诊断三实际模式通量差符号写反**:`model_flux`/`recon_flux` 按
   (strong, weak) 存储但差值误用 `[1]-[0]`(weak-strong),字段名却是
   strong_minus_weak。修复:阶段 A/B 全部差值统一改经
   `vc.strong_minus_weak([0], [1])`(强-弱);新增不依赖替换闭合式的符号测试
   (助手级:110/100→+10)与端到端校验(冒烟测试按统一掩膜独立重算 actual,
   与 CSV 逐例对照)。修复前合成输出 actual=+28.39(反向),修复后 -29.55
   (与偏置结构 -28 一致)。
2. **[P1] 汇总把三个环带与并集混合平均、改善成员数翻倍**:主汇总改为仅筛选
   `r000_300`(并集),逐区域结果保留在 `summary_by_region_window.csv`;
   `n_members_improved` 在选定区域/窗口内按唯一成员计数(≤ n_members);
   汇总不再对跨区域的改善成员数求和。诊断二柱状图改为成员先均、后跨成员,
   并按区域分别绘制。
3. **[P1] RealWrfProvider 缺少入口方法**:补齐 `nr_static()`、`exp_fields()`
   (含文件 Times 属性校验与强/弱网格经纬度一致性校验,不按维度相同直接
   逐下标比较),并用本地合成 NetCDF 文件(mock,位于冒烟输出目录,非真实
   数据)做契约测试:入口方法齐全、读取/配准走通、时间与网格校验生效。
4. **[P1] 配准未接入/不一致/最近邻内存不可扩展**:新建统一配准入口
   `vc.register_nr_fields` 与单时次上下文 `vc.build_time_context`,三个诊断
   共用(策略一致);真实模式 ll_to_xy 路径接入 provider;
   `resample_nearest` 重写为目标/源双向分块流式实现(内存 O(chunk×tile),
   不再做全点对广播),并加覆盖半径(2×源网格间距估计),域外目标置 NaN。
5. **[P2] 质量标记未传递到阶段 B**:阶段 A 质量判据改用 RMS 尺度
   (residual_rmse_ratio 为主,residual_maxabs_to_rms_ratio 为辅),在阶段 B
   共同掩膜上重新评估并写入 `attribution_flag` 列,图件标题附标记分布;
   阶段 A 另报告两差残差的 RMS/最大绝对差。
6. **[P2] 重建输出缺共同有限掩膜**:新增 `evaluate_reconstruction_outputs`,
   对重建输出(四组合)、模式通量、NR 通量建共同有限掩膜;n_valid/status
   反映输出有效性(非有限输出 → `nonfinite_reconstruction_output` +
   `n_nonfinite_output`),阶段 A 每行单一统计掩膜;重建输入先做物理有效
   范围预检(失败记 `reconstruction_failed`)。
7. **[P2] 降序坐标插值未翻转数据**:`_extract_regular_axis` 返回反转标志,
   `resample_bilinear_regular` 同步翻转数据场;新增纬度/经度分别与同时
   降序的合成测试(线性场精确复原)。
8. **[P2] 全部案例缺失时在写出状态前崩溃**:`run_status` 改为最先落盘;
   `summarize_member_metrics` 空输入返回带完整列结构的空表;绘图与汇总
   兼容零有效案例;新增"全缺失运行"冒烟检查(不崩溃、状态表/空表正确)。
   时序图对缺失时刻补 NaN 断线(`vc.align_times_to_hours`),不跨接前后时次。

## 一并调整的其他事项

- 容差注释修正:dSE 容差不能开方解释为普适增量阈值(交叉项 2e·dF 之故)。
- 诊断二 `interpretation` 增加 `unchanged` 类别(`BUDGET_UNCHANGED_TOL` 可配置);
  诊断二/三新增独立的 `improvement_status`/`sst_only_improvement_status` 列,
  零基准时其余绝对误差指标保留。
- 诊断二汇总表纳入交叉项/平方项/闭合残差的窗口汇总(可直接复核闭合)。
- 诊断三阶段 A 图按方法×试验分线;阶段 B 柱状图标题附归因标记分布。
- 图题/打印/npz/run_status 的 SYNTHETIC 硬编码改为由 `config.mode` 生成;
  npz 增加 mode/method/member/time 元数据。
- 诊断三逐点场保存补齐 Fww/Fsw/Fss(/Fws)、NR 通量、共同掩膜,可离线重算
  各组 RMSE;actual_model_diff 的保存范围与重建量统一(同一 compare_mask)。
- 汇总表更名为 `summary_by_region_window.csv`(按区域×窗口),主汇总
  `summary_by_method.csv` 仅用并集;汇总新增覆盖率列(实际/期望时次数)。
- 合成生成器:修复 NaN 补丁条件(此前误用调用参数,补丁泄漏到其他方法/成员);
  强弱大气差异从 ×1.06(约 +18 K)改为 ~+0.5 K 的小扰动,并加入成员相关的
  稳定度形态;交错风注释改为如实描述(去交错 = 邻点平均,仅常数场严格相等);
  生成方式说明改为与实际一致(真值+统计误差,与低层大气无通量公式联系)。
- 冒烟测试:强制合成模式(即使入口 CONFIG 被改为 real);每次运行使用独立
  临时输出目录并使用 run() 返回的确切路径(不读旧文件);新增防呆开关、
  mock NetCDF 契约、非有限重建输出、全缺失运行、NaN 补丁隔离、actual 独立
  重算、降序网格、覆盖半径、汇总上界、成员等权等回归检查。

## 第二轮测试结果

- 解释器同前(`/Users/kcfu/miniforge3/envs/wrf/bin/python`,未安装新依赖)。
- 冒烟测试扩至 **27 项,全部通过**(报告见
  `outputs/SYNTHETIC_smoke_test/smoke_test_report.txt`)。
- 三个诊断入口独立重跑成功;抽查确认:actual 强-弱方向正确、主汇总
  n_members=6 且 n_members_improved≤6、覆盖率列反映缺失时次(0.99)、
  归因标记已传递到阶段 B 表与图。
- 仍然未验证:物理实现与 WRF Fortran 的逐公式等价性、真实 namelist、
  NR 时间插值物理误差、真实网格与变量维度(见 MATH_REVIEW_CHECKLIST.md)。

---

# 第三轮:响应复核报告(2026-09-30)

针对 `/private/tmp/SYNTHETIC_verify_diag_recheck_z6pquz4e/REVIEW.md` 的 R1–R6
及低优先级项,逐条修复如下;全部以本地合成数据验收,未触真实数据。

## R1 [P1] WRF 字符数组时间格式

`Times(time, DateStrLen)` 的 S1 字符数组此前经 `str()` 变成 `[b'2' b'0' ...]`,
正常时间也被拒绝。新增 `_decode_wrf_time`(兼容 VLEN 字符串、bytes、S1 数组),
试验与 NR 读取路径均使用;NR 的精确时次与上/下整点插值路径现在同样校验文件
内部时间。冒烟测试的 mock 写入器支持两种 Times 格式,并新增 S1 端到端测试
(正常时间通过、错误时间拒绝、整点插值校验)。

## R2 [P1] 归因标记只查弱耦合且跨区域复制

`attribution_flag` 改为在**每个区域的最终比较掩膜**(含启用时的 Fws 输出)
上,用三个必要条件逐一评估:弱重建残差比、强重建残差比、重建差-实际差比
(分母均为该区域模式强弱差 RMS);判据数值保存在 `attr_weak_rmse_ratio`/
`attr_strong_rmse_ratio`/`attr_diff_agree_ratio`,标记由纯函数
`attribution_quality_flag` 生成。审查反例(模式 100/201、重建 100/101,
弱准强错)在单测中不再被放行,对称情形同样拦截;输出抽查确认判据数值
随区域不同(r000_075: 4.41/4.80,r150_300: 0.64/0.49)。

## R3 [P1] 最近邻全点对搜索

最近邻重采样改为**可复用空间索引**:源网格按指纹构建 cKDTree(球面三维坐标),
按目标点查询,复杂度 O(N log N);索引按网格指纹缓存(几何固定时跨时次复用)。
真实模式的海陆掩膜默认走 ll_to_xy 投影索引 order=0
(`RealWrfProvider.register_mask_lltoxy`),不做全点对搜索。无 scipy 时保留
分块全对搜索作为回退并明确标注其复杂度限制。

## R4 [P2] 间距估计与足迹覆盖

`estimate_source_spacing_km` 改为按**相邻索引对** ((i,j)-(i±1,j)/(i,j±1))
的真实球面距离中位数估计(规则与曲线结构网格都适用,消除抽样高估);
覆盖判断增加**源网格足迹**:规则网格用包围盒、曲线网格用凸包(容差半个
格距),"边界外但距最近点很近"的目标点被排除。新增近边界测试
(0.01 度网格:域内/边界有效,域外 0.05 度排除;间距估计约 1.1 km)。

## R5 [P2] 覆盖率统计口径

汇总区分三级时次计数:已读取(`mean/min/max_member_times_read`)、
有效绝对误差(status=ok 或 zero_rmse_weak,零基准的绝对误差视为有效)、
有效改善率(改善率有限);分母显式输出 `expected_times` 与
`expected_members`,`member_abs_time_coverage_ratio` 与
`member_coverage_ratio` 分别对应;完全缺失的成员计入成员分母。
单测覆盖:空掩膜行不计入有效、零基准计入绝对但不计入改善、缺失成员分母。

## R6 [P2] 状态被 NaN 默认值覆盖、非有限计数跨区域

阶段 B 空掩膜分支改为"先填数值 NaN、后设置状态字段"(`improvement_status`/
`attribution_flag`/`sst_only_improvement_status` 不再被覆盖;空区域
attribution_flag=empty_mask);非有限输出点数按区域统计
(`region_output_status` 抽成可测函数),300 km 外的无效点不再影响其他区域;
阶段 A 中一组重建无效导致差值共同失效时,状态记为
`nonfinite_reconstruction_diff` 以记录共同失效原因。单测覆盖三种场景:
组合全 NaN、仅一个区域 NaN、仅 300 km 外 NaN。

## 低优先级项

- 汇总表重复表头(rmse_weak/rmse_strong)已消除,输出列唯一(冒烟测试断言)。
- 诊断二/三时序图对缺失时次补 NaN 断线(与诊断一同口径)。
- NR 各时次网格一致性显式断言(`_check_nr_grid`,与缓存参考逐点核对);
  试验侧网格一致性校验此前已有。
- 批次全部失败时写 `overall_status.txt`(no_valid_cases)并打印警告。

## 第三轮测试结果

- 解释器同前;冒烟测试扩至 **32 项,全部通过**
  (`outputs/SYNTHETIC_smoke_test/smoke_test_report.txt`),
  覆盖复核报告"下一轮最小验收集合"1–7 全部条目。
- 三个诊断入口独立重跑成功;抽查确认:归因判据数值按区域不同且与标记
  一致、覆盖率列分母显式(缺失时次 0.987)、总状态文件 ok。
- 仍然未验证:物理实现与 WRF Fortran 的逐公式等价性、真实 namelist、
  NR 时间插值物理误差、真实网格与变量维度。

---

# 第四轮:自查补充(2026-09-30)

针对复核报告遗留缺口与自查发现的问题:

1. **ll_to_xy 投影路径从未被测试执行(复核报告修复状态表遗留项)**:
   新增 `check_lltoxy_registration`——用 Mercator 反解构造等 ψ 纬度网格,
   DX/DY 与网格间距严格一致,使 `wrf.ll_to_xy` 能在 mock 投影文件上运行;
   断言双线性配准的线性场复现(atol 0.5)、域外目标为 NaN、掩膜 order=0
   取最近格点原值。至此真实模式的两条配准路径(投影插值与空间索引最近邻)
   都有合成测试覆盖。
2. **归因残差与比较掩膜的严格一致(自查)**:阶段 B 的归因三判据此前用
   阶段 A 掩膜上的重建输出构建残差;改为直接用阶段 B 自身的替换输出
   (`flux_ww`/`flux_ss`)构建,与最终比较掩膜严格一致。
3. **索引缓存指纹加固(自查)**:`_grid_fingerprint` 增加加权校验和,
   排除不同网格在角点/中点采样值相同时的缓存碰撞可能。
4. **清理**:移除未使用的 `import dataclasses`、run() 内未使用的 `hours`
   变量与一处测试占位残留(AST 静态检查无未使用 import)。
5. 冒烟测试扩至 **33 项,全部通过**;三个诊断入口再次独立重跑成功。

仍然未验证:物理实现与 WRF Fortran 的逐公式等价性、真实 namelist、
NR 时间插值物理误差、真实网格与变量维度;ll_to_xy 路径的 mock 投影
(Mercator)与真实兰勃特投影的公式差异仍属真实数据核验范围。


---

# 第五轮:功能补充与基础修复(2026-09-30,响应第二轮任务书)

## 新增功能

1. **诊断二条件子集预算表**(`verify02_conditional_budget.csv` +
   `verify02_conditional_summary_by_member/_method.csv`):按
   dSE_T(TSK,阈值 sst_se_tol,单位 K^2)构造 all_common/sst_improved/
   sst_worsened/sst_unchanged 四个子集(只按海温误差变化筛选),对 HFX/LH
   复用同一预算公式(实际弱误差、逐点乘积后平均);输出子集点数、分母
   all_common 点数、子集比例、C/S/ΔMSE/闭合残差、净变化与交叉项方向状态
   (独立容差 CROSS_TERM_TOL,接近零为中性)。独立成表,不替换原全区域
   预算结果。合成输出中验证子集拆分守恒(improved+worsened+unchanged
   = all_common)。
2. **诊断三联合诊断表**(`verify03_joint_response_actual.csv` +
   `verify03_joint_summary_by_member/_method.csv`):在"区域&海洋&比较掩膜
   &强/弱/NR TSK 共同有限"的同一最终掩膜上(含 Fws),逐子集计算
   cross_model_error_sst_response(用实际弱误差)、rms_sst_response/
   rms_actual_response/rms_sst_minus_actual/sst_to_actual_rms_ratio、
   符号一致率(FLUX_INCREMENT_TOL,单位 W m-2,输出参与点数与比例)、
   同掩膜实际通量 C/S/ΔMSE/闭合残差(actual_ 前缀列)、按子集重评的重建
   质量(joint_status + attr_*_ratio 数值)。全部配置案例(2 方法×6 成员
   ×13 时次×4 区域×4 子集×2 变量 = 4960 行)都计算,POINT_FIELD_CASES
   只控制逐点 npz 保存。与诊断二条件预算在同掩膜行逐位一致
   (max diff = 0),掩膜更严的行点数更小(显式断言)。
3. **配置快照** `config_snapshot.json`(三入口均输出):配准方式、
   NR 时间匹配方式、阈值、时刻表等。

## 基础问题核查与修复(任务书第五节)

1. `_inside_footprint()` 凸包分支:确认对多查询点会抛 ValueError
   (`bool(array)`)。已修:逐点布尔并 reshape,多点/单点/零点查询回归通过。
2. 网格指纹:采样点+加权校验和仍可能被互抵内部几何欺骗
   (构造反例确认)。已修:缓存命中后做**完整经纬度逐点校验**
   (`_grids_identical`,equal_nan),不一致按碰撞处理重建索引;
   回归测试确认"互抵内部几何不错误复用、同网格命中缓存、
   缓存结果与独立计算一致"。
3. 诊断三零有效输出分支:`region_output_status` 已区分
   `nonfinite_reconstruction_output`(输入有效但输出全无效,含 nonfinite
   计数)与新增的 `empty_input`(该区域本来没有有效输入);
   空掩膜行的状态字段此前会被 NaN 默认值覆盖,已在状态字段后置;
   `response_vs_actual_stats` 非 ok 分支此前缺符号列键导致下游 KeyError,
   已补全(所有分支返回完整键)。

## 合成验收(全部通过,40/40)

新增手算预期测试(预期值全部来自显式构造/手算,不调用被测函数生成):
- e=1,dF=1 → C=2、S=1、ΔMSE=3;e=1,dF=−3 → C=−6、S=9、ΔMSE=3;
- 交叉项为零(error 与 dF 正交)而平方项非零;
- 改善子集 C=−6 与恶化子集 C=+2 反号,全区域 C=−2(手算对照);
- dF_SST=[1,−1]、dF_actual=[−1,1]:均值均零、RMS 差=2、符号一致率=0;
- cross_model_error_sst_response 用实际弱误差(实际 +1 vs 离线 −1 的
  符号相反构造,C=+2 而非 −2);
- 不同输入场在不同位置含 NaN:全部联合指标同一最终掩膜
  (n_valid 只数共同有限点,e/dF_SST/dF_actual 任一 NaN 即剔除);
- 空子集/全输出无效/实际响应为零(ratio NaN + actual_response_zero)/
  离线响应为零(ratio=0 有效)/阈值边界(等于阈值不参与);
- 全区域与子集质量标记可不同(残差在子集内的反例);
- 端到端:读取实际写出的新表,检查列、状态、点数、子集比例、闭合、
  跨表一致性、全案例覆盖(6 成员×2 方法)。
- 三处基础问题的回归测试(凸包多点、指纹碰撞、状态区分)。

原有 33 项测试全部保留并通过(无回归)。

## 解释边界(重申)

- 联合表的 cross_model_error_sst_response 只是离线响应相对实际误差方向的
  诊断,不是因果贡献,不与实际增量平方项相加闭合实际 ΔMSE;
- e_model(实际弱误差)与 e_recon(离线重建弱误差)是两个基线,
  `sst_only_rmse_improvement_pct` 用离线基线,不替代实际误差方向指标;
- 两表 all_common 掩膜定义不同(联合表额外要求重建输出有限),
  跨表对照仅在点数相同的行逐位一致;
- 合成通量并非由同一物理重建器生成,以上为程序与代数测试,
  未通过调整质量阈值伪装成物理验证。


---

# 第六轮:响应数学逻辑复核(2026-10-02)

针对 `/private/tmp/SYNTHETIC_verify_diag_mathreview_5m6aig4y/REVIEW.md` 的
6 项 P2 与次要事项,逐条修复;全部以本地合成数据验收。

1. **[P2] 重建质量未贯通到联合汇总**:联合明细的质量标记唯一化——
   `joint_status` 为子集质量的唯一依据(每个子集掩膜上重算三判据,
   空子集/指标不可用行记自身状态,不继承全区域合格标记);原全区域标记
   改名 `attribution_flag_region_ref` 仅供对照。联合汇总新增
   `n_times_quality_ok`/`n_times_quality_failed`/`quality_coverage_ratio`,
   全样本数值保留供描述,质量不合格时次不从统计中静默删除。
2. **[P2] rms_sst_minus_actual 用旧掩膜**:`response_vs_actual_stats` 内
   改用 `joint_mask`(审查反例 e=[1,NaN]、dSST=[1,100]、actual=[1,0] 的
   数值逐项断言通过:rms 差=0 而非 70.71);回归测试检查全部指标的数值
   而不仅是 n_valid。
3. **[P2] 无效重建失败信息丢失/输入计数不正确**:Phase A 空分支改为
   "先定失效原因再判空"(empty_input / nonfinite_reconstruction_output /
   nonfinite_reconstruction_diff);Phase B 空分支使用
   `region_output_status` 的真实状态与 `n_nonfinite_output` 计数,
   不再一律写 empty_mask;联合表新增三级点数
   `n_input_valid`(输入掩膜)/`n_common`(重建输出后)/
   `n_tsk_common`(再加 TSK)与 `n_nonfinite_output` 列,
   重建造成的样本损失可单独恢复,部分失效信息传入联合表。
4. **[P2] 新增汇总缺少期望分母**:诊断二条件汇总新增 `expected_times`/
   `expected_members`/`member_time_coverage_ratio`/`member_coverage_ratio`
   (含 min/max 成员有效时次);诊断三联合汇总含
   `mean_n_input_valid`/`mean_n_tsk_common` 与质量覆盖率。
5. **[P2] 诊断二全缺失崩溃**:条件汇总键与期望分母移到分支前,
   空表保留完整列结构;新增诊断二全缺失入口测试
   (run_status 先落盘、空表列完整、overall_status=no_valid_cases)。
6. **[P2] 新表验收顺序错误**:端到端检查移到全流程之后,使用本轮
   run() 返回的确切目录(check_full_flow 返回目录映射),不再跨运行
   glob 猜测;子集守恒检查全部组(不提前 break)。

次要事项:`cross_term_stats` 全 NaN 输入的 `n_valid` 契约修正为 0
(附测试);比率分母下限新增独立配置 `SST_RATIO_DENOMINATOR_FLOOR`
(W m-2,默认 0.5,与符号一致率阈值分开;低于下限比率置 NaN 并标记
actual_response_below_floor,只影响比率不改原始 RMS);
`flux_increment_tol_w2` 更名 `flux_increment_tol_wm2` 并在 columns_readme
逐列说明单位。

**本轮测试:44/44 通过**(40 项原有 + 4 项新增回归:联合统计 NaN 数值
反例、三级点数与全失败状态、子集质量标记唯一化、诊断二全缺失运行)。
三个入口独立重跑成功;抽查确认联合表三级点数与质量分层列、
条件汇总覆盖率列(expected_times=13、member_coverage_ratio=1.0)。

仍未验证:物理实现与 WRF Fortran 逐公式等价性、真实 namelist、
NR 时间插值物理误差、真实网格与变量维度。
