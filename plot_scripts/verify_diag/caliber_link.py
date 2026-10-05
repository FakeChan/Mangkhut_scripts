"""模块 B:pathway 与 verify_diag 的统计口径对照与连接检查。

两件事(对应 REVIEW_SOURCES_AND_CALIBER.md 的第五节):
1. `CALIBER_DIMENSIONS`:两张证据体系的口径对照表(逐维给出取值与判定,
   证据为脚本文件/行号);`write_caliber_table()` 写出 CSV。
2. `check_joinable()`:对两张结果表按键做兼容性判定,逐键输出
     value_by_value    键与全部口径元数据一致,可逐值比较;
     parallel_evidence 键一致但口径不同,只能并列展示、不能逐值比较;
     not_comparable    口径冲突且两侧取值均已知;
     unknown           任一口径元数据缺失/未知——默认 unknown,不判 compatible;
     duplicate_keys    键不唯一,无法安全连接。
   只做判定,不拼接、不重算、不重跑任何旧脚本。
"""

from __future__ import annotations

import pandas as pd

#: 判定档位
VERDICT_VALUE_BY_VALUE = "value_by_value"
VERDICT_PARALLEL = "parallel_evidence"
VERDICT_NOT_COMPARABLE = "not_comparable"
VERDICT_UNKNOWN = "unknown"
VERDICT_DUPLICATE_KEYS = "duplicate_keys"

#: 口径对照表(证据见 REVIEW_SOURCES_AND_CALIBER.md 第五节及其引用的脚本行号)
#: (dimension, pathway_value, verify_diag_value, verdict, reason)
CALIBER_DIMENSIONS = (
    (
        "experiments/methods/members",
        "6mem_oceanAssim{0,1}Run1; EAKF+QCF_RHF; members discovered dynamically per cache",
        "same experiments/methods (verify01-03 config); member set from config",
        VERDICT_UNKNOWN,
        "成员集合由缓存动态发现,不能把固定成员数当成已验证事实;"
        "连接前须逐侧核对实际成员与前导零",
    ),
    (
        "valid_time_set",
        "8 times: 0,0.5,1,1.5,2,3,4,6 h",
        "13 times: 0-6 h every 0.5 h",
        VERDICT_PARALLEL,
        "pathway 时次是 verify_diag 的子集;逐时值比较只能在共享时次上进行",
    ),
    (
        "spatial_support",
        "10x10 grid-point block means (BLOCK_SIZE=10, MIN_VALID_FRACTION=0.5)",
        "pointwise equal-weight valid points",
        VERDICT_NOT_COMPARABLE,
        "块平均丢失逐点信息;不能由块均值恢复逐点 MSE/RMS/极值",
    ),
    (
        "center_definition",
        "min of mean(strong PSFC, weak PSFC) over ocean in search box",
        "min NR PSFC over ocean in search box",
        VERDICT_NOT_COMPARABLE,
        "中心不同则环带与扇区格点集不同",
    ),
    (
        "ocean_mask",
        "LANDMASK<0.5 of experiment (weak t=0 file)",
        "LANDMASK<0.5 of experiment AND NR (nearest-registered)",
        VERDICT_NOT_COMPARABLE,
        "双重掩膜与单侧掩膜的格点集不同",
    ),
    (
        "ocean_variable_layer",
        "OM_TMP surface layer 0",
        "OM_TMP surface layer 0",
        VERDICT_VALUE_BY_VALUE,
        "同一定义(维度 ocean_layer_stag 第 0 层)",
    ),
    (
        "dom0_meaning",
        "0h wrfout strong-weak OM_TMP diff (forecast initial diff)",
        "verify04 d_pair@F0 corresponds first; d_pair@B/I are separate stages",
        VERDICT_UNKNOWN,
        "dom0 首先对应 d_pair@F0,不能默认对应 B/I;dom0 也不是 A-B 同化增量;"
        "对应关系需真实数据核验",
    ),
    (
        "offline_flux_sst_source",
        "OM_TMP surface layer at t=0 (om0_strong/om0_weak)",
        "TSK of each experiment at each forecast time (verify03)",
        VERDICT_NOT_COMPARABLE,
        "表面温度来源与时刻都不同,是两个不同的替换试验",
    ),
    (
        "offline_flux_atmosphere",
        "t=0 weak-experiment lowest-level state, fixed",
        "each experiment's own lowest-level state at each time",
        VERDICT_NOT_COMPARABLE,
        "大气状态时刻与来源不同",
    ),
    (
        "error_vs_skill",
        "block corr/slope/composite of strong-weak diffs (no NR)",
        "pointwise bias/MSE/RMSE vs NR-registered truth (verify01)",
        VERDICT_PARALLEL,
        "相关/斜率不是预报技巧;两者可作为不同证据并列",
    ),
    (
        "weights",
        "block means with min valid fraction 0.5; member equal weight",
        "equal-weight valid points; member inner-time mean then equal member",
        VERDICT_NOT_COMPARABLE,
        "空间支持与汇总权重均不同",
    ),
    (
        "registration",
        "no NR registration (strong-weak only)",
        "NR registered by wrf.ll_to_xy + bilinear (or numpy nearest)",
        VERDICT_PARALLEL,
        "pathway 不涉及 NR;verify_diag 的 NR 配准不构成口径冲突",
    ),
)

#: 对照表列
CALIBER_TABLE_COLUMNS = [
    "dimension", "pathway_value", "verify_diag_value", "verdict", "reason",
]

#: 常用元数据列(结果表自带时用于 check_joinable)
RECOMMENDED_META_COLUMNS = (
    "center_definition",
    "spatial_support",
    "time_set_id",
    "ocean_mask",
    "sst_source",
    "atmosphere_source",
    "metric_family",
    "weight_method",
)

#: 连接判定的必需语义(P2 #6):必须出现在键列或口径列中,
#: 缺失即判 unknown(不默认 compatible)。
#: - experiment_or_cycle / valid_time / variable / unit / vertical_layer:
#:   保证同键下比较的是同一物理量;
#: - metric_family / spatial_support:保证指标含义与空间支持可比。
REQUIRED_SEMANTIC_FIELDS = (
    "experiment_or_cycle",
    "valid_time",
    "variable",
    "unit",
    "vertical_layer",
    "metric_family",
    "spatial_support",
)


def _normalize_key(key) -> tuple:
    """把单列键统一为规范元组(P2 #7:字符串不被拆成首字符)。"""
    if isinstance(key, tuple):
        return key
    return (key,)


def unmatched_keys(left: pd.DataFrame, right: pd.DataFrame,
                   key_columns: tuple[str, ...]) -> pd.DataFrame:
    """报告单侧独有的键(P2 #7:未匹配样本不从检查结果中消失)。

    返回列:key_columns + side(left_only/right_only)。前导零等表示差异
    会体现为两侧不同的键,本函数不做静默归一。
    """
    key_columns = tuple(key_columns)

    def key_set(frame: pd.DataFrame) -> set:
        subset = frame[list(key_columns)]
        return {
            _normalize_key(row)
            for row in subset.itertuples(index=False, name=None)
        }

    left_keys = key_set(left)
    right_keys = key_set(right)
    rows = [
        {**dict(zip(key_columns, key)), "side": "left_only"}
        for key in sorted(left_keys - right_keys, key=repr)
    ]
    rows += [
        {**dict(zip(key_columns, key)), "side": "right_only"}
        for key in sorted(right_keys - left_keys, key=repr)
    ]
    return pd.DataFrame(rows, columns=list(key_columns) + ["side"])


def build_caliber_table() -> pd.DataFrame:
    """返回口径对照表(每行一个维度,verdict 为四档之一)。"""
    rows = [
        {
            "dimension": dimension,
            "pathway_value": pathway,
            "verify_diag_value": verify,
            "verdict": verdict,
            "reason": reason,
        }
        for dimension, pathway, verify, verdict, reason in CALIBER_DIMENSIONS
    ]
    return pd.DataFrame(rows, columns=CALIBER_TABLE_COLUMNS)


def write_caliber_table(out_dir) -> pd.DataFrame:
    """把口径对照表写出为 caliber_compat_table.csv。"""
    from pathlib import Path

    table = build_caliber_table()
    path = Path(out_dir) / "caliber_compat_table.csv"
    table.to_csv(path, index=False)
    print(f"[output] {path} ({len(table)} rows)", flush=True)
    return table


def check_joinable(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: tuple[str, ...],
    caliber_columns: tuple[str, ...],
    label_left: str = "left",
    label_right: str = "right",
) -> pd.DataFrame:
    """按键逐组判定两张结果表是否可连接。

    - 必需语义字段(REQUIRED_SEMANTIC_FIELDS)未全部出现在键列或口径列
      → 全部判 unknown(P2 #6:变量/单位/层等缺失时不判可比);
    - 键在任一侧重复 → duplicate_keys(不拼接);
    - 口径列在两侧均存在且逐列相等 → value_by_value;
    - 口径列均已知但存在冲突 → not_comparable(reason 列出冲突列);
    - 任一口径列缺失或取值缺失/标记为 unknown → unknown(不默认 compatible);
    - 单列键按标量处理(不被拆成字符;P2 #7);
    - 只在键同时出现的组上判定;单侧独有的键用 unmatched_keys() 查看。

    返回列:key_columns + verdict + conflicts + n_left + n_right。
    """
    key_columns = tuple(key_columns)
    caliber_columns = tuple(caliber_columns)
    covered = set(key_columns) | set(caliber_columns)
    missing_required = [
        field for field in REQUIRED_SEMANTIC_FIELDS if field not in covered
    ]
    rows: list[dict] = []

    def dup_check(frame: pd.DataFrame) -> set:
        duplicated = frame.duplicated(subset=list(key_columns), keep=False)
        keys = frame.loc[duplicated, list(key_columns)]
        return {
            _normalize_key(row)
            for row in keys.itertuples(index=False, name=None)
        }

    dup_left = dup_check(left)
    dup_right = dup_check(right)

    left_norm = left.copy()
    right_norm = right.copy()
    left_norm["__key__"] = [
        _normalize_key(row)
        for row in left_norm[list(key_columns)].itertuples(index=False, name=None)
    ]
    right_norm["__key__"] = [
        _normalize_key(row)
        for row in right_norm[list(key_columns)].itertuples(index=False, name=None)
    ]
    left_index = left_norm.set_index("__key__", drop=False)
    right_index = right_norm.set_index("__key__", drop=False)
    shared = left_index.index.intersection(right_index.index).unique()

    def meta_values(frame: pd.DataFrame, column: str):
        if column not in frame.columns:
            return "unknown", "column_absent"
        values = frame[column]
        non_null = values.dropna().astype(str).str.strip()
        non_null = non_null[~non_null.str.lower().isin(("unknown", "nan", ""))]
        if non_null.empty:
            return "unknown", "value_missing_or_unknown"
        unique = sorted(non_null.unique())
        if len(unique) == 1:
            return unique[0], "ok"
        return "mixed:" + ",".join(unique), "multiple_values"

    if missing_required:
        # 必需语义缺失:逐共享键输出 unknown,不默认放行(P2 #6)
        reason = "missing_required_semantics:" + ",".join(missing_required)
        for key in shared:
            group_left = left_index.loc[[key]]
            group_right = right_index.loc[[key]]
            row = {column: key[i] for i, column in enumerate(key_columns)}
            row["verdict"] = VERDICT_UNKNOWN
            row["conflicts"] = reason
            row["n_left"] = len(group_left)
            row["n_right"] = len(group_right)
            rows.append(row)
        return pd.DataFrame(
            rows, columns=list(key_columns) + ["verdict", "conflicts",
                                                "n_left", "n_right",
                                                "verdict_scope"]
        )

    for key in shared:
        row: dict = {column: key[i] for i, column in enumerate(key_columns)}
        group_left = left_index.loc[[key]]
        group_right = right_index.loc[[key]]
        row["n_left"] = len(group_left)
        row["n_right"] = len(group_right)
        if key in dup_left or key in dup_right:
            side = label_left if key in dup_left else label_right
            if key in dup_left and key in dup_right:
                side = f"{label_left}+{label_right}"
            row["verdict"] = VERDICT_DUPLICATE_KEYS
            row["conflicts"] = f"duplicate keys in {side}"
            rows.append(row)
            continue
        conflicts: list[str] = []
        unknowns: list[str] = []
        # 必需语义逐键检查实际取值(R5):不因其放在键列而跳过未知检查;
        # 值为 unknown/缺失 → unknown,不作为逐值可比证据
        for field in REQUIRED_SEMANTIC_FIELDS:
            if field not in covered:
                continue  # 已由 missing_required 分支处理
            value_left, state_left = meta_values(group_left, field)
            value_right, state_right = meta_values(group_right, field)
            if state_left != "ok" or state_right != "ok":
                unknowns.append(
                    f"{field}({label_left}:{state_left},{label_right}:{state_right})"
                )
                continue
            if value_left != value_right:
                conflicts.append(f"{field}:{value_left}!={value_right}")
        for column in caliber_columns:
            if column in REQUIRED_SEMANTIC_FIELDS:
                continue  # 已在上面检查
            value_left, state_left = meta_values(group_left, column)
            value_right, state_right = meta_values(group_right, column)
            if state_left != "ok" or state_right != "ok":
                unknowns.append(f"{column}({label_left}:{state_left},{label_right}:{state_right})")
                continue
            if value_left != value_right:
                conflicts.append(f"{column}:{value_left}!={value_right}")
        if unknowns:
            row["verdict"] = VERDICT_UNKNOWN
            row["conflicts"] = ";".join(unknowns)
        elif conflicts:
            row["verdict"] = VERDICT_NOT_COMPARABLE
            row["conflicts"] = ";".join(conflicts)
        else:
            row["verdict"] = VERDICT_VALUE_BY_VALUE
            row["conflicts"] = ""
        # 范围声明:判定只对调用者列出的键/口径字段成立;中心/掩膜/权重等
        # 未列出的维度不在本判定的保证范围内(不能代替实际数据的全部口径核验)
        row["verdict_scope"] = "limited_to_listed_fields"
        rows.append(row)

    columns = list(key_columns) + ["verdict", "conflicts", "n_left", "n_right",
                                    "verdict_scope"]
    return pd.DataFrame(rows, columns=columns)
