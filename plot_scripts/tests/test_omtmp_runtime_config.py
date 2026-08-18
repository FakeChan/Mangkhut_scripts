import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "omtmp_evidence_cache.py"


def load_module():
    spec = importlib.util.spec_from_file_location("omtmp_evidence_cache", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch_member(root, experiment, method, member):
    path = root / experiment / method / member
    path.mkdir(parents=True)
    return path


def test_discover_common_members_uses_method_specific_intersections(tmp_path):
    module = load_module()
    for member in ("002", "010", "001"):
        touch_member(tmp_path, "strong", "EAKF", member)
    for member in ("010", "001", "999"):
        touch_member(tmp_path, "weak", "EAKF", member)
    for member in ("044", "006"):
        touch_member(tmp_path, "strong", "QCF_RHF", member)
        touch_member(tmp_path, "weak", "QCF_RHF", member)

    members = module.discover_common_members(
        tmp_path,
        strong_experiment="strong",
        weak_experiment="weak",
        methods=("EAKF", "QCF_RHF"),
        max_members_per_method=None,
    )

    assert members == {"EAKF": ("001", "010"), "QCF_RHF": ("006", "044")}


def test_discover_common_members_honors_member_limit(tmp_path):
    module = load_module()
    for member in ("006", "015", "029"):
        touch_member(tmp_path, "strong", "EAKF", member)
        touch_member(tmp_path, "weak", "EAKF", member)

    members = module.discover_common_members(
        tmp_path,
        strong_experiment="strong",
        weak_experiment="weak",
        methods=("EAKF",),
        max_members_per_method=2,
    )

    assert members == {"EAKF": ("006", "015")}
    assert module.paired_case_count(members) == 2


def test_pathway_extract_import_does_not_require_forecast_directories(monkeypatch):
    monkeypatch.setenv("OMTMP_FORECAST_BASE_DIR", "/path/that/does/not/exist")
    module_path = Path(__file__).resolve().parents[1] / "omtmp_pathway_extract.py"
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location("omtmp_pathway_extract_import_test", module_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.finite_linear_stats)
