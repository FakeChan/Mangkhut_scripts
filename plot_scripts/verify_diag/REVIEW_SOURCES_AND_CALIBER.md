# verify_diag 初值传递与口径衔接审查(2026-10-03)

本文档回答两项审查要求:(一)本地代码中 background/analysis/forecast_initial/
forecast_output 四阶段的来源关系;(二)pathway 与 verify_diag 的统计口径对照。
所有结论均给出文件与行号证据;**"本地存在某脚本"不等于"服务器当前试验执行过
该脚本"**;路径规则来自脚本阅读,属文件名推断,未经真实运行确认。

## 一、四个阶段对应的本地文件(6mem cycling 试验,证据:5cyclingDA/run_driver_6mem_cyclingDA.sh)

| 阶段 | 本地名称 | 证据 | 说明 |
|---|---|---|---|
| B 背景 | `firstguess_d0{dom}.{member}`(ensmem_dir) | driver_DART_cyclingDA.sh:83 将其符号链接为 `wrfinput_d0{dom}` 供 DART 读取("DART need") | 即 DART 的先验状态,格式为 wrfinput;首个循环来自 1icbc/IC 的 `wrfinput_201809DDHH_memNN`(sub_cycle_wrf.sh:33) |
| A 分析 | `output_d0{dom}.{member}`(2DART/run_dir) | driver_DART_cyclingDA.sh:131 `mpirun ./filter`;update_ocean=1 时先用 inflatedOcean 的 OM_TMP/OM_S/OM_U/OM_V 覆盖 output(run_driver_6mem_cyclingDA.sh:208-209) | DART 后验;注意 output 与 firstguess 为**不同文件** |
| I 启动 | `firstguess_d0{dom}.{member}`(ncks 合并后)→ `mv` 为 `run_wrf/<time>/<mem>/wrfinput_d0{dom}` | run_driver_6mem_cyclingDA.sh:222-230(cp firstguess→run_dir,ncks.sh 合并,注释掉的 mv 行)与 307-309(`mv ... ./wrfinput_d0${dom}`) | 合并后经 `./update_wrf_bc`(更新侧边界)与 `ad_omini_d01/d02.py`(见下)后启动 wrf.exe |
| F0 预报0h | `wrfout_d0{dom}_2018-09-10_00:00:00` 等 | run_driver_6mem_cyclingDA.sh 的 history_interval=30 输出 | **仅为模式启动后的输出,不能替代 B/A/I**;0h wrfout 与 I 的差异是 WRF 初始化的效果,属描述性检查 |

**传递映射 M(有代码依据)**:`0necessay_files/ncks_updateDARTvar.sh`(被复制为
run_dir/ncks.sh,run_driver_6mem_cyclingDA.sh:225-228 调用):

```
common_vars="MU,OM_TMP,OM_U,OM_V,OM_S,P,PH,QVAPOR,THM,U,V,W"
ncks -A -v $common_vars ${updated_file} ${base_file}
```

即**只有这 12 个变量**把分析值带入启动状态;其余变量在 I 中保持背景值。
另有三个后处理会修改 I:

- `update_wrf_bc`(driver 内调用):DART 侧边界更新,作用于 wrfbdy;
- `ad_omini_d01/d02.py`(0necessay_files):从 `wrfinput_d0X_gfs`(real.exe 的
  GFS 初始场)**中心点垂线**复制 `OM_TINI,OM_SINI` 廓线写入 wrfinput;
  **注意**:这只证明 OM_TINI/OM_SINI 两个变量被重置,是否等于重置了实际
  预报的 OM_TMP 廓线,需要模式初始化代码证据,仅凭变量名不能下此结论
  (OM_TINI 与 OM_TMP 的关系本身待核实);
- `0necessay_files/ncks_AssignPostaasim2Wrfinput.sh`(仅 sub_cycle_wrf.sh 的
  0mem 流程引用):只移植 `OM_TINI,OM_SINI`,**TSK 行被注释掉**。

**变量命名注意**:DART/firstguess 文件中海洋变量名为 `OM_TMP`(ncdump 已确认),
而分析移植脚本使用 `OM_TINI/OM_SINI`(wrfinput 内变量);两者的关系(是否同一
变量的别名)未在本地代码中说明,列为待核实。

## 二、`update_tsk_from_omtmp.py` 审查

位置:`4assimilation/0mem_all_time/cyclingDA/10_00_00/update_tsk_from_omtmp.py`。

- 输入对象:`firstguess_{domain}.mem{member:03d}`(d01+d02,50 成员,
  member_count=50;**与当前 6mem 试验的 6 成员配置不同**);
- 修改变量:`TSK` 就地改写(`nc.Dataset(path,"r+")`),取 `OM_TMP` 第 0 层;
- 掩膜:`XLAND == 2`(WRF 水体掩膜);注意 verify_diag 的海洋掩膜用
  `LANDMASK < 0.5`(pathway 同),两者语义等价但变量不同;
- 调用位置:**仓库内未发现任何调用者**(grep update_tsk 仅命中自身);
  属于 0mem_all_time/10_00_00 试验目录,是否被手动执行、是否适用于当前
  6mem 试验均未知;
- 含义:若该脚本在某试验中执行,则该试验的 B/A/I 中 TSK(海洋点)已被
  OM_TMP 表层替换——"OM_TMP 改变而 TSK 不变"因此可能是设计行为而非错误,
  verify_04 对此类差异只记录数值与状态,不判为错误。

相关但独立的脚本:`0necessay_files/correct_bias.py`(VAR_NAME='OM_TMP',以
真值文件集合平均为目标对成员做偏差订正),仅在 `profile_matlab_flag=1` 时于
首个同化时次调用(driver_DART_cyclingDA.sh:120-129);`inflate_ocean.sh/.py`
(0mem_all_time):对 firstguess 的 OM_TMP 做 NCO 集合膨胀(λ=5)。

## 三、来源关系的证据强度分级

| 关系 | 证据强度 | 依据 |
|---|---|---|
| filter 读 firstguess 写 output | 调用代码支持 | driver_DART_cyclingDA.sh:83,112-131 |
| ncks 把 output 的 12 变量并入 firstguess 副本 | 调用代码支持 | run_driver_6mem_cyclingDA.sh:222-228;ncks_updateDARTvar.sh |
| 合并后的 firstguess 直接 mv 为 wrfinput | 调用代码支持 | run_driver_6mem_cyclingDA.sh:307-309 |
| update_wrf_bc / ad_omini 修改启动状态 | 调用代码支持 | run_driver_6mem_cyclingDA.sh:313-320(调用存在;ad_omini 内容已读) |
| update_tsk_from_omtmp 曾被当前试验执行 | **未知** | 无调用者;目录属 0mem_all_time 试验 |
| firstguess 的原始来源(上一循环 wrfout 或 1icbc IC) | 文件名推断 | sub_cycle_wrf.sh:43(0mem 流程);6mem 首循环来源未在本地脚本中出现 |
| post_anal_dir、scratch 路径的实际布局 | 文件名推断 | 脚本变量,未经服务器确认 |

**推论边界**:I 与 A 的差异 = (1) 非 M 列表变量保持背景值;(2) update_wrf_bc/
ad_omini 的修改;(3) ncks 合并本身的精度(ncks -A 为精确拷贝,应为 0)。
因此 d_handoff 对 M 列表变量应≈0(除非后续步骤修改);非 M 变量在 I 中等于 B,
故 d_handoff(非M) = B − A = −d_assim。这两条是 verify_04 数值检查的预期
形态(仅作解释参考,不作通过/失败判据)。

**dom0 与同化增量的一般关系**(2026-10-04 更正,取代此前不正确的写法):
记强弱试验下标 s/w,一般地

    dom0 = A_s - A_w = (B_s - B_w) + [(A_s - B_s) - (A_w - B_w)]

即 dom0 = 强弱背景差 + 两试验同化增量之差(δ_s = A_s−B_s,δ_w = A_w−B_w)。
"共同背景 B_s = B_w 且弱试验该变量无分析更新(A_w = B_w)"是常用的
**充分条件**(此时 dom0 = δ_s),不是必要条件——严格等价条件为
B_s − B_w = δ_w(反例:B_s=1、B_w=0、δ_s=2、δ_w=1,则 dom0=2=δ_s,
但背景不同且弱试验有更新)。若交接或初始化另有改动,还需加入相应的
强弱残差。因此 pathway 的 dom0 首先对应 verify_04 的 d_pair@F0,
不能默认对应 B/I 阶段。

## 四、pathway 与 verify_diag 的功能覆盖对照

| 功能 | pathway 系 | verify_diag 系 | 状态 |
|---|---|---|---|
| 0h/后续 OM_TMP 表层强弱差(dom0/dom) | ✅ omtmp_pathway_extract.py | verify_04 将以 d_pair 重复该语义(仅 t=0 对齐) | 保留 pathway;verify_04 不重算后续时次 |
| TSK/通量/温湿风/PBLH 响应提取(块平均) | ✅ | ❌(verify01 只算相对 NR 误差) | 不重复建设 |
| 最低 10 层位温/水汽响应 + 相关/回归 | ✅ | ❌ | 不重复建设 |
| 空间相关、回归斜率、响应 RMS、合成对比 | ✅ finite_linear_stats/source_composite | ❌ | 不重复建设 |
| 相对 NR 的误差演变(bias/MSE/RMSE) | ❌ | ✅ verify01 | — |
| 误差预算分解、条件子集 | ❌ | ✅ verify02 | — |
| 离线条件替换 + 重建质量 | ✅(旧口径,见下) | ✅ verify03(新口径) | 并列证据,不合并 |
| 初值传递核验(B→A→I) | ❌ | ❌ → 本次 verify_04 | 新增最小实现 |

## 五、必须处理的口径差异(模块 B 的输入)

1. **块平均 vs 逐点**:pathway 用 10×10 块平均(BLOCK_SIZE=10,
   MIN_VALID_FRACTION=0.5);verify_diag 用逐格点等权误差。块均值无法恢复
   逐点 MSE/RMS/极值。
2. **中心定义**:pathway 用强弱 PSFC 平均场的海洋最低点
   (omtmp_pathway_extract.py:259-264);verify_diag 用 NR PSFC 最低点。
3. **时次集合**:pathway 8 个时次(0,0.5,1,1.5,2,3,4,6);verify_diag 13 个
   (0—6 每半小时)。
4. **dom0 ≠ A−B**:pathway 的 dom0 是 0h wrfout 的强弱 OM_TMP 差(预报初值
   差),不是同化增量;一般地 dom0 = (B_s−B_w) + (δ_s−δ_w)(见上文),
   与 verify_04 各阶段 d_pair 的对应关系需按真实数据核验。
5. **离线通量表面温度**:旧 pathway 离线计算用初始 OM_TMP 表层
   (om0_strong/om0_weak)作表面温度、大气取 t=0 弱试验状态;verify03 默认
   用各试验自身 TSK、大气取同时次状态。两套结果不是同一个替换试验。
6. **指标语义**:pathway 的 corr/slope/RMS 是块平均强弱差之间的空间统计,
   不是相对 NR 的预报技巧;verify01 的 improvement 才是技巧口径。
7. **海洋掩膜变量**:pathway/verify_diag 用 LANDMASK<0.5;
   update_tsk_from_omtmp.py 用 XLAND==2。
8. **成员前导零**:pathway 缓存 CSV 的 member 曾有不补零形式
   (analyze_omtmp_pathway.py:39-40 需 zfill);verify_diag 输出为三位零填充。
9. **成员集合是动态发现**:pathway 缓存按目录内容发现成员,不是固定的
   "6 paired members";任何连接判定前须逐侧核对实际成员集合。

兼容性判定工具(`caliber_link.py`)对以上逐项给出
`value_by_value / parallel_evidence / not_comparable / unknown` 四档结论;
默认 unknown,不默认 compatible。
