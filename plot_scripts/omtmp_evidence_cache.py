"""Runtime cache helpers for the OM_TMP evidence figures."""

from __future__ import annotations

import os
import io
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FORECAST_BASE_DIR = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
NR_DIR = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")
STRONG_EXPERIMENT = "6mem_oceanAssim1Run1"
WEAK_EXPERIMENT = "6mem_oceanAssim0Run1"
METHODS = ("EAKF", "QCF_RHF")
CACHE_POLICIES = {"auto", "refresh", "reuse"}


def member_sort_key(member: str) -> tuple[int, int | str]:
    text = str(member)
    return (0, int(text)) if text.isdigit() else (1, text)


def parse_methods(value: str | None) -> tuple[str, ...]:
    if not value:
        return METHODS
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods:
        raise ValueError("At least one assimilation method must be configured.")
    return methods


def _member_dirs(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing member directory parent: {path}")
    return {
        item.name
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    }


def discover_common_members(
    forecast_base_dir: str | Path,
    *,
    strong_experiment: str,
    weak_experiment: str,
    methods: tuple[str, ...],
    max_members_per_method: int | None,
) -> dict[str, tuple[str, ...]]:
    """Return method-specific strong/weak member intersections."""
    base = Path(forecast_base_dir)
    result: dict[str, tuple[str, ...]] = {}
    for method in methods:
        strong_members = _member_dirs(base / strong_experiment / method)
        weak_members = _member_dirs(base / weak_experiment / method)
        members = tuple(sorted(strong_members & weak_members, key=member_sort_key))
        if max_members_per_method is not None:
            members = members[:max_members_per_method]
        if not members:
            raise ValueError(
                f"No paired members found for method {method!r} under {base}"
            )
        result[method] = members
    return result


def paired_case_count(members_by_method: dict[str, tuple[str, ...]]) -> int:
    return sum(len(members) for members in members_by_method.values())


def _env(
    *,
    forecast_base_dir: Path,
    nr_dir: Path,
    strong_experiment: str,
    weak_experiment: str,
    methods: tuple[str, ...],
    max_members_per_method: int | None,
    cache_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMTMP_FORECAST_BASE_DIR": str(forecast_base_dir),
            "OMTMP_NR_DIR": str(nr_dir),
            "OMTMP_STRONG_EXPERIMENT": strong_experiment,
            "OMTMP_WEAK_EXPERIMENT": weak_experiment,
            "OMTMP_METHODS": ",".join(methods),
            "OMTMP_CACHE_DIR": str(cache_dir),
        }
    )
    if max_members_per_method is not None:
        env["OMTMP_MAX_MEMBERS_PER_METHOD"] = str(max_members_per_method)
    else:
        env.pop("OMTMP_MAX_MEMBERS_PER_METHOD", None)
    return env


def _ensure_cache_from_scripts(
    *,
    cache_dir: Path,
    required_files: tuple[str, ...],
    extract_script: str,
    analyze_script: str,
    env: dict[str, str],
    cache_policy: str,
) -> None:
    missing = [name for name in required_files if not (cache_dir / name).exists()]
    if cache_policy == "reuse" and missing:
        raise FileNotFoundError(
            f"Cache is incomplete under {cache_dir}; missing: {', '.join(missing)}"
        )
    if cache_policy == "auto" and not missing:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_path = SCRIPT_DIR / extract_script
    analyze_path = SCRIPT_DIR / analyze_script
    if not extract_path.exists() or not analyze_path.exists():
        raise FileNotFoundError(
            f"Missing cache helper scripts beside the plotting script: "
            f"{extract_path.name}, {analyze_path.name}"
        )

    print(f"building cache in {cache_dir} with {extract_path.name}", flush=True)
    result = subprocess.run(
        [sys.executable, str(extract_path)],
        check=True,
        stdout=subprocess.PIPE,
        env=env,
    )
    with zipfile.ZipFile(io.BytesIO(result.stdout), "r") as bundle:
        bundle.extractall(cache_dir)

    print(f"summarizing cache with {analyze_path.name}", flush=True)
    subprocess.run([sys.executable, str(analyze_path)], check=True, env=env)


def _cache_is_complete(cache_dir: Path, required_files: tuple[str, ...]) -> bool:
    return all((cache_dir / name).exists() for name in required_files)


def validate_cache_policy(cache_policy: str) -> None:
    if cache_policy not in CACHE_POLICIES:
        raise ValueError("cache_policy must be 'auto', 'refresh', or 'reuse'")


def ensure_skill_cache(
    *,
    cache_dir: Path,
    forecast_base_dir: Path,
    nr_dir: Path,
    strong_experiment: str,
    weak_experiment: str,
    methods: tuple[str, ...],
    max_members_per_method: int | None,
    cache_policy: str,
) -> dict[str, tuple[str, ...]]:
    validate_cache_policy(cache_policy)
    required_files = (
        "omtmp_skill_metrics.csv",
        "omtmp_skill_summary.csv",
        "omtmp_skill_vertical_summary.csv",
        "omtmp_skill_intensity_summary.csv",
    )
    if cache_policy in {"auto", "reuse"} and _cache_is_complete(cache_dir, required_files):
        return {}
    members = discover_common_members(
        forecast_base_dir,
        strong_experiment=strong_experiment,
        weak_experiment=weak_experiment,
        methods=methods,
        max_members_per_method=max_members_per_method,
    )
    env = _env(
        forecast_base_dir=forecast_base_dir,
        nr_dir=nr_dir,
        strong_experiment=strong_experiment,
        weak_experiment=weak_experiment,
        methods=methods,
        max_members_per_method=max_members_per_method,
        cache_dir=cache_dir,
    )
    _ensure_cache_from_scripts(
        cache_dir=cache_dir,
        required_files=required_files,
        extract_script="omtmp_skill_extract.py",
        analyze_script="analyze_omtmp_skill.py",
        env=env,
        cache_policy=cache_policy,
    )
    return members


def ensure_pathway_cache(
    *,
    cache_dir: Path,
    forecast_base_dir: Path,
    nr_dir: Path,
    strong_experiment: str,
    weak_experiment: str,
    methods: tuple[str, ...],
    max_members_per_method: int | None,
    cache_policy: str,
) -> dict[str, tuple[str, ...]]:
    validate_cache_policy(cache_policy)
    required_files = (
        "omtmp_pathway_surface_blocks.csv",
        "omtmp_pathway_link_summary.csv",
        "omtmp_pathway_direct_flux_member.csv",
        "omtmp_pathway_vertical_summary.csv",
    )
    if cache_policy in {"auto", "reuse"} and _cache_is_complete(cache_dir, required_files):
        return {}
    members = discover_common_members(
        forecast_base_dir,
        strong_experiment=strong_experiment,
        weak_experiment=weak_experiment,
        methods=methods,
        max_members_per_method=max_members_per_method,
    )
    env = _env(
        forecast_base_dir=forecast_base_dir,
        nr_dir=nr_dir,
        strong_experiment=strong_experiment,
        weak_experiment=weak_experiment,
        methods=methods,
        max_members_per_method=max_members_per_method,
        cache_dir=cache_dir,
    )
    _ensure_cache_from_scripts(
        cache_dir=cache_dir,
        required_files=required_files,
        extract_script="omtmp_pathway_extract.py",
        analyze_script="analyze_omtmp_pathway.py",
        env=env,
        cache_policy=cache_policy,
    )
    return members
