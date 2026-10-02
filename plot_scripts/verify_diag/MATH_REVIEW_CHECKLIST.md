# verify_diag 数学审查清单(供后续审查者使用)

本清单列出三个诊断中关键公式与实现函数的对应关系,以及所有尚未确认、
需要审查者或在真实数据上核实的物理与数据约定。审查范围:公式实现、
数值口径、掩膜与权重、汇总统计;不含代码风格。

更新(2026-09-30 第二轮):已按外部审查报告修复符号反转(诊断三实际模式差)、
汇总区域混合、真实 provider 契约缺失、配准未接入与最近邻内存、质量标记未传递、
重建输出掩膜、降序轴插值、全缺失崩溃等缺陷(见 TEST_RECORD.md 第二轮)。

更新(2026-09-30 第三轮):已按复核报告修复 S1 字符数组时间解析、归因标记
(逐区域三判据:弱+强+差一致)、最近邻空间索引与足迹覆盖、覆盖率三级口径、
状态字段被 NaN 覆盖与跨区域非有限计数等缺陷(见 TEST_RECORD.md 第三轮)。

更新(2026-09-30 第四轮,自查):ll_to_xy 投影配准路径已用 mock Mercator
文件纳入冒烟测试(此前从未被执行);诊断三归因残差改用阶段 B 自身替换输出;
索引缓存指纹加固。审查者请以当前代码与本清单为准。

更新(2026-09-30 第五轮):新增条件子集预算(诊断二)与离线响应-实际变化
联合诊断(诊断三),公式映射见 4b 节;修复凸包足迹多点布尔、索引缓存
完整几何校验、区域状态 empty_input 区分。两基线(e_model/e_recon)与
解释边界见 4b 节审查要点。

更新(2026-10-02 第六轮):按数学逻辑复核修复——联合表子集质量标记唯一化
(joint_status;attribution_flag_region_ref 仅对照)、汇总质量分层
(n_times_quality_ok/failed、quality_coverage_ratio)、rms_sst_minus_actual
改用联合掩膜、三级点数(n_input_valid/n_common/n_tsk_common)、
Phase A/B 空分支真实状态(empty_input 与输出失效分开)、比率分母下限
(SST_RATIO_DENOMINATOR_FLOOR,与符号一致率阈值分开)、
cross_term_stats 全 NaN 时 n_valid=0、诊断二全缺失不崩溃、
新表验收顺序(全流程后用本轮确切目录)。

## 一、关键公式 ↔ 实现函数

### 1. 配对误差指标(诊断一)

| 公式 | 实现位置 |
|---|---|
| `bias_exp = mean(X_exp - X_NR)` | `verify_common.paired_error_metrics`(weak_bias/strong_bias) |
| `MSE_exp = mean((X_exp-X_NR)^2)`,`RMSE = sqrt(MSE)` | 同上(mse_weak/mse_strong/rmse_weak/rmse_strong) |
| `mse/rmse_change = 强 - 弱`(正值=恶化) | 同上(mse/rmse_change_strong_minus_weak) |
| `改善率 = 100*(RMSE_weak-RMSE_strong)/RMSE_weak`(正值=改善;RMSE_weak=0 → NaN+status) | 同上(rmse_improvement_pct);零分母保护在此函数内 |

审查要点:
- 掩膜 = 区域 & 海洋 & strong/weak/NR 三数组共同有限(|x|<1e30),
  见 `verify_common.common_valid_mask`/`finite_mask`;等权格点平均。
- 口径与 `omtmp_skill_extract.paired_skill_metrics` 一致(改善率同号),
  该旧实现本身也在审查范围内。

### 2. 海温-通量逐点分类(诊断一)

| 公式 | 实现位置 |
|---|---|
| `dSE_X = (X_strong-X_NR)^2 - (X_weak-X_NR)^2` | `verify_common.se_change` |
| 分类:改善 dSE<-tol、恶化 dSE>+tol、\|dSE\|≤tol 未变化;组合 1-4;单侧未变化=-1、双侧=0 | `verify_common.classify_change_categories`;占比 `category_fractions` |

审查要点:
- 容差 `sst_se_tol=1e-4 K^2`、`flux_se_tol=1.0 (W m-2)^2` 为判定参数,
  单位含义见配置区;阈值取值是否合理由研究目标决定。
- 分类配对按用户口径为 (tsk,hfx)、(tsk,lh);海温侧变量可换 om(OM_TMP 表层),
  在 `verify_01.CLASSIFICATION_PAIRS` 修改。
- 分类统计的分母为共同有效格点数(含 NR 场有限)。

### 3. 通量误差预算(诊断二)

| 公式 | 实现位置 |
|---|---|
| `e = F_weak - F_NR`,`dF = F_strong - F_weak` | `verify_common.flux_error_budget` |
| `dSE = 2 e dF + (dF)^2`(逐点) | 逐点场在 `verify_02` 内由 `se_change` 生成并存 npz;预算用其区域平均 |
| `delta_mse_direct = MSE_strong - MSE_weak = <dSE>` | `flux_error_budget` |
| `cross_term = 2<e dF>`(未去均值乘积的平均;非协方差、非 <e><dF>) | 同上 |
| `increment_square_term = <(dF)^2>` | 同上 |
| `closure_residual = delta_mse_direct - (cross+square)` | 同上(合成全流程实测 max≈4.5e-12) |

审查要点:
- 确认 cross_term 的定义是逐点乘积的平均(代码 `np.mean(2.0*error*increment)`),
  满足"不得替换为协方差、不得用 <e><dF> 代替"。
- 解释标签(`interpret_budget`)仅为描述,审查其措辞是否与数值逻辑一致。

### 4. 固定大气海温替换(诊断三)

| 公式 | 实现位置 |
|---|---|
| 最低层输入推导:温度 `theta0*(p/p0)^(Rd/cp)`、高度 `0.5*((PH+PHB)_0+(PH+PHB)_1)/9.81-HGT`、U/V 去交错、诊断式 z0m | `verify_common.derive_lowest_level_inputs`、`initial_momentum_roughness`(口径取自 `omtmp_pathway_extract.py` 224-245 行,该口径本身待核实) |
| `F_ww/F_sw/F_ss/F_ws = reconstruct_ocean_fluxes(...)`,统一掩膜 (1,n) 调用、下限钳制 ust≥1e-4、z0m≥1.27e-7 | `verify_03.call_reconstruction`;掩膜 `verify_common.unified_reconstruction_mask` |
| 阶段 A:重建-模式偏差/RMSE/maxabs;模式与重建的强减弱差(经 `vc.strong_minus_weak`,强-弱方向);两差残差的均值/RMS/最大绝对差;`residual_rmse_ratio`(残差RMS/模式强弱差RMS,主判据)与 `residual_maxabs_to_rms_ratio` 及 quality_flag;每行单一统计掩膜 | `verify_03._process_case` 阶段 A 段;`_phase_a_quality` |
| 阶段 B:`dF_SST=F_sw-F_ww`、`dF_rest=F_ss-F_sw`、`dF_reconstructed=F_ss-F_ww`;`actual=强-弱`(`vc.strong_minus_weak`)并与 dF_total 逐点对照(RMS/最大绝对差);F_ww/F_sw/F_ss 相对 NR 通量误差;`sst_only_rmse_improvement_pct=100*(RMSE(F_ww)-RMSE(F_sw))/RMSE(F_ww)`(分母状态单列) | `verify_03._process_case` 阶段 B 段 |
| 比较掩膜:重建输出(Fww/Fsw/Fss/Fws)+ 模式通量 + NR 通量的共同有限掩膜;非有限输出点数按区域统计 → `nonfinite_reconstruction_output` + `n_nonfinite_output`,不静默删点 | `verify_03.region_output_status`、`evaluate_reconstruction_outputs` |
| 归因可信度:在**每个区域**的最终比较掩膜上用三必要条件评估——弱重建残差比、强重建残差比、重建差-实际差比(分母=该区域模式强弱差 RMS;数值存 `attr_*_ratio` 列),全部满足才给有利标记 | `verify_03.attribution_quality_flag`(`_process_case` 阶段 B 段调用) |
| 反向路径:`dF_SST_rev=F_ss-F_ws`、`path_split_difference=dF_SST-dF_SST_rev` | 同上(INCLUDE_REVERSE_PATH 开关) |

审查要点:
- 拆分恒等式 `dF_SST+dF_rest=dF_reconstructed` 应精确闭合(合成实测 ~1e-13)。
- **符号契约**:所有 strong_minus_weak 语义的差值必须经 `vc.strong_minus_weak`
  (强-弱;修复前阶段 A/B 曾因下标误用呈反向,已修复并以独立重算回归覆盖)。
- 阶段 A/B 掩膜不同(A 不含 NR 通量、B 含),审查该选择是否合理;
  attribution_flag 只取自 B 掩膜上的重评结果。
- "重建通量强减弱差"使用各试验**自身**大气输入分别重建,非单一大气态。

### 4b. 条件子集预算(诊断二新增)与联合诊断(诊断三新增)

| 公式 | 实现位置 |
|---|---|
| dSE_T = (T_s-T_NR)^2 - (T_w-T_NR)^2;子集 all_common/sst_improved(<-tol)/sst_worsened(>tol)/sst_unchanged(\|dSE_T\|<=tol),只按海温侧筛选 | `verify_common.sst_subset_masks` |
| 实际通量预算:e_model=F_weak_model-F_NR、dF_actual、C=2⟨e·dF⟩、S=⟨dF²⟩、ΔMSE=C+S(逐点乘积后平均,空掩膜 NaN) | `verify_common.cross_term_stats`(复用 flux_error_budget 口径) |
| 交叉项/净变化方向状态(独立容差;接近零=中性) | `verify_common.cross_term_direction`、`net_error_direction` |
| 联合统计:cross_model_error_sst_response=2⟨e_model·dF_SST⟩(**实际弱误差**);rms_sst/actual/rms_sst_minus_actual/sst_to_actual_rms_ratio;符号一致率(增量阈值,单位 W m-2);同掩膜实际 C/S/ΔMSE | `verify_common.response_vs_actual_stats`(全部指标同一最终掩膜) |
| 子集重评的重建质量(attr_*_ratio 数值 + joint_status) | `verify_03._ratio_on_mask`、`_joint_quality_flag` |

审查要点:
- 两个基线不可混用:e_model(实际弱误差)用于实际预算与新增方向交叉项;
  e_recon=F_ww-F_NR(离线)用于现有 sst_only_rmse_improvement_pct。
- cross_model_error_sst_response 不是因果贡献,不与实际增量平方项相加
  闭合实际 ΔMSE;不输出因果贡献百分比。
- 两表 all_common 掩膜定义不同(联合表额外要求重建输出有限),
  跨表对照仅在点数相同的行逐位一致(端到端测试已断言)。
- 子集质量标记按子集重算,不继承全区域标记。

### 5. 汇总统计(两个改善口径)

| 定义 | 实现位置 |
|---|---|
| 各成员相对改善率的平均值:成员先做窗口内时间平均,再跨成员等权平均 | `verify_common.summarize_member_metrics`(mean_member_rmse_improvement_pct;n_members_improved,选定区域/窗口内唯一成员数) |
| 平均 RMSE 的相对变化:`100*(mean_rmse_weak-mean_rmse_strong)/mean_rmse_weak` | 同上(pooled_rmse_improvement_pct) |
| 主汇总仅用并集区域 r000_300;逐区域结果保留 `summary_by_region_window.csv`;并集不再与子环带混合平均 | 各入口 `run()` 汇总段(修复项:此前把 4 个区域平均、改善数按区域求和) |
| 覆盖率:成员窗口内实际时次数(mean/min/max)对期望时次数之比 | `summarize_member_metrics`(expected_times 参数) |

审查要点:两口径数值不可混用;窗口定义(0-6/0.5-2/3-6 h)衔接
`analyze_omtmp_skill.py` 的 early/late 口径;主汇总不含区域混合平均;
覆盖率三级分列(已读取/有效绝对误差/有效改善率),分母显式
(expected_times、expected_members);空掩膜行不计入有效,零基准 RMSE
计入绝对误差有效但不计入改善率有效。

## 二、已采用的本地源码口径(来源可信但建议复核)

1. `paired_skill_metrics` 的有限值掩膜与 1e30 阈值(omtmp_skill_extract.py)。
2. 台风中心 = NR PSFC 在搜索框(10-25N,135-155E)海洋最低点
   (omtmp_skill_extract.py);注意 `omtmp_pathway_extract.py` 用强弱平均 PSFC
   定中心,两个旧脚本口径不同,本套件默认 NR PSFC。
3. 海洋掩膜 = 试验 LANDMASK<0.5 且 NR 配准后 LANDMASK<0.5(omtmp_skill_extract.py)。
4. NR→试验网格配准:统一入口 `vc.register_nr_fields`——物理场用
   wrf.ll_to_xy + scipy.map_coordinates 双线性(omtmp_skill_extract.py 口径;
   合成模式用规则网格 numpy 双线性替代,可选 numpy_nearest=空间索引最近邻);
   掩膜在真实模式默认用同一投影索引 order=0,其余情形用 cKDTree 空间索引
   最近邻(按网格指纹缓存,复杂度 O(N log N),无全点对搜索)。覆盖判断 =
   源网格足迹(规则网格包围盒/曲线网格凸包,容差半个格距)且不超过覆盖半径
   (2×相邻索引对间距估计);"边界外但距最近点很近"的目标点为 NaN。
5. 最低层输入推导公式(omtmp_pathway_extract.py 224-245 行,含 Rd/cp=287/1004、
   T+300、9.81、z0m 诊断式 CZO=0.0185、0.11*1.5e-5、上限 2.85e-3)。
6. 调用 reconstruct 前的钳制:ust≥1e-4、z0m≥1.27e-7(analyze_omtmp_pathway.py)。
7. NR 半小时真值由相邻整点线性插值(omtmp_skill_extract.py 的 NR_TIME_BRACKETS)。

## 三、待核实事项(真实数据运行前必须逐项确认)

### 数据与路径

| # | 事项 | 现状/缺省 |
|---|---|---|
| 1 | 预报目录结构 `BASE/{试验}/{方法}/{成员}/wrfout_d02_时间` | 由本地源码推断;未实测 |
| 2 | 成员目录名为三位零填充字符串(006/015/029/037/043/044) | 同上 |
| 3 | NR 目录文件命名与可得时次;**NR 是否有半小时输出** | 缺省按相邻整点插值 |
| 4 | `OM_TMP` 维度顺序 (time, ocean_layer, y, x) 及表层=第 0 层 | 由 `[0,0]` 用法推断 |
| 5 | d02 网格距 DX_M=1500 m | 取自 analyze_omtmp_pathway.py,未对照 namelist |
| 6 | NR 与 d02 投影一致性(wrf.ll_to_xy 可用性);配准方法选择 | 默认 ll_to_xy_bilinear;读取适配层已用本地合成 NetCDF 做契约测试,从未在真实数据执行 |
| 7 | 13 个时刻(0-6 h 半小时)文件是否齐全 | 缺省逐时次检查,缺失记 run_status |

### 物理与模式配置

| # | 事项 | 现状/缺省 |
|---|---|---|
| 8 | 表面通量方案为 WRF v4.1 Revised MM5(sfclayrev),与 wrf41_sfclayrev.py 对应版本一致 | 假设;需对照 namelist 物理方案 |
| 9 | `isftcflx=0` | 取自 analyze_omtmp_pathway.py;需对照 namelist |
| 10 | 最低模式层输入定义(T+300、P+PB、QVAPOR、离地高度公式、U/V 去交错) | 复用旧口径;未与模式内 sfclay 调用逐项核对 |
| 11 | UST 输出能否作为初始摩擦速度;诊断式 z0m 与模式内 z0m 的差异(模式内 ustar/z0m 会迭代,输出的是终值,本重建用输出值作"初值") | 口径来自旧脚本;未核实 |
| 12 | **TSK 与 OM_TMP 第一层的衔接关系**;禁止未经核实用 OM_TMP 第 0 层代替通量方案的 TSK(接口保留、默认禁用) | SURFACE_TEMP_SOURCE="TSK" |
| 13 | HFX/LH 在预报 0 时次可能是初始化值(旧脚本 README 提示),解释时需谨慎 | 已在旧脚本记录;本套件未做特殊处理 |
| 14 | NR 的 HFX/LH 是否与试验同方案同输出频率(作为"真值"的可用性) | 待核实 |
| 15 | 阶段 A 质量阈值(0.5/2.0)的合理量级 | 可配置;需真实数据评估 |
| 16 | 分类容差(1e-4 K²、1 (W m-2)²)的合理性 | 可配置;研究目标决定 |

### 统计与解释

| # | 事项 |
|---|---|
| 17 | 12 个配对(2 方法×6 成员)不是独立天气样本;两方法使用对应成员,汇总时按方法分开,不合并为 12 独立样本 |
| 18 | 不做把格点/相邻时次当独立样本的显著性检验(本套件未实现任何显著性检验) |
| 19 | 交叉项/平方项/替换分解只是误差预算与条件替换描述,不构成物理因果结论 |
| 20 | 诊断三 `dF_rest` 标注为"替换其余输入后的响应",不是纯大气反馈;正反路径结果不同即存在路径依赖 |
