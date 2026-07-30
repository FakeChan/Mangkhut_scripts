"""
Diagnose LACC ensemble correlations between satellite Hx and current OM_TMP.

The single-lag diagnostic is

    corr_member[OM_TMP(t), Hx(t - lag)]

and the leading-averaged diagnostic is

    corr_member[OM_TMP(t), mean(Hx(t), ..., Hx(t - lag))].

Correlations are calculated independently at every observation location and
then summarized spatially without pooling members and locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


# =========================
# User configuration
# =========================
@dataclass(frozen=True)
class Config:
    hx_dir: Path = Path(
        "/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/4ens_BT"
    )
    mem_dir: Path = Path(
        "/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00"
    )
    profile_path: Path = Path(
        "/share/home/lililei1/kcfu/tc_mangkhut/"
        "3create_obs/hx_rttov/profile/profile_d01/prof09_12:00.dat"
    )
    output_dir: Path = Path("./figs/LACC")
    current_time: str = "2018-09-10_00:00:00"
    max_lag_hours: int = 12
    lag_interval_hours: int = 3
    member_start: int = 1
    member_end: int = 50
    expected_obs_count: int = 676
    sensor: str = "AMSUA"
    channel: int = 4
    domain: str = "d01"
    omtmp_var: str = "OM_TMP"
    lat_var: str = "XLAT"
    lon_var: str = "XLONG"


CONFIG = Config()
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
PROFILE_COORDINATE_MARKER = (
    "! Elevation (km), latitude and longitude (degrees)"
)


def validate_config(config: Config) -> None:
    datetime.strptime(config.current_time, TIME_FORMAT)
    if config.max_lag_hours < 0:
        raise ValueError("max_lag_hours must be nonnegative.")
    if config.lag_interval_hours <= 0:
        raise ValueError("lag_interval_hours must be positive.")
    if config.max_lag_hours % config.lag_interval_hours != 0:
        raise ValueError(
            "max_lag_hours must be an integer multiple of lag_interval_hours."
        )
    if config.member_start < 1 or config.member_end < config.member_start:
        raise ValueError("Member range must satisfy 1 <= member_start <= member_end.")
    if config.expected_obs_count < 1:
        raise ValueError("expected_obs_count must be positive.")


def build_lag_hours(config: Config) -> list[int]:
    validate_config(config)
    return list(
        range(
            0,
            config.max_lag_hours + config.lag_interval_hours,
            config.lag_interval_hours,
        )
    )


def hx_valid_time(config: Config, lag_hours: int) -> datetime:
    if lag_hours < 0:
        raise ValueError("lag_hours must be nonnegative.")
    current = datetime.strptime(config.current_time, TIME_FORMAT)
    return current - timedelta(hours=int(lag_hours))


def hx_path(config: Config, member: int, lag_hours: int) -> Path:
    valid_time = hx_valid_time(config, lag_hours)
    time_dir = valid_time.strftime("BT_%d_%H_%M")
    filename = f"obs_{config.domain}_ch{config.channel}_totalline.txt"
    return (
        config.hx_dir
        / f"mem{member:03d}"
        / config.sensor
        / time_dir
        / filename
    )


def member_path(config: Config, member: int) -> Path:
    return config.mem_dir / f"firstguess_{config.domain}.mem{member:03d}"


def read_profile_coordinates(
    path: Path, expected_count: int = 676
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()
    rows: list[dict[str, float | int]] = []
    for line_index, line in enumerate(lines):
        if line.strip() != PROFILE_COORDINATE_MARKER:
            continue
        if line_index + 1 >= len(lines):
            raise ValueError(
                f"Missing coordinate values after marker at line {line_index + 1}."
            )
        values = lines[line_index + 1].split()
        if len(values) < 3:
            raise ValueError(
                f"Invalid coordinate row at line {line_index + 2}: "
                f"expected elevation, latitude, and longitude."
            )
        elevation, latitude, longitude = map(float, values[:3])
        if not np.all(np.isfinite([elevation, latitude, longitude])):
            raise ValueError(
                f"Non-finite coordinate at profile line {line_index + 2}."
            )
        rows.append(
            {
                "obs_index": len(rows) + 1,
                "elevation_km": elevation,
                "lat": latitude,
                "lon": longitude,
            }
        )

    if len(rows) != expected_count:
        raise ValueError(
            f"Profile {path} contains {len(rows)} coordinates; "
            f"expected {expected_count}."
        )
    return pd.DataFrame(
        rows,
        columns=["obs_index", "elevation_km", "lat", "lon"],
    )


def read_hx_file(path: Path, expected_count: int = 676) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Hx file not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != expected_count:
        raise ValueError(
            f"Hx file {path} contains {len(lines)} values; "
            f"expected {expected_count}."
        )

    values = np.empty(expected_count, dtype=float)
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(
                f"Hx file {path}, line {index + 1} must contain one value."
            )
        values[index] = float(fields[0])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Hx file {path} contains non-finite values.")
    return values


def pointwise_correlations(
    state_members: np.ndarray, hx_members: np.ndarray
) -> np.ndarray:
    state = np.asarray(state_members, dtype=float)
    hx = np.asarray(hx_members, dtype=float)
    if state.ndim != 2 or hx.ndim != 2 or state.shape != hx.shape:
        raise ValueError(
            "state_members and hx_members must have the same "
            "(member, observation) shape."
        )

    correlations = np.full(state.shape[1], np.nan, dtype=float)
    for obs_index in range(state.shape[1]):
        x = state[:, obs_index]
        y = hx[:, obs_index]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        if np.std(x_valid, ddof=1) == 0.0 or np.std(y_valid, ddof=1) == 0.0:
            continue
        correlations[obs_index] = float(
            np.corrcoef(x_valid, y_valid)[0, 1]
        )
    return correlations


def build_averaged_hx(hx_by_lag: np.ndarray) -> np.ndarray:
    hx = np.asarray(hx_by_lag, dtype=float)
    if hx.ndim != 3 or hx.shape[0] == 0:
        raise ValueError(
            "hx_by_lag must have shape (lag, member, observation) "
            "with at least one lag."
        )
    window_sizes = np.arange(1, hx.shape[0] + 1, dtype=float)[:, None, None]
    return np.cumsum(hx, axis=0) / window_sizes


def summarize_pointwise(correlations: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(correlations, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    summary: dict[str, float | int] = {
        "valid_point_count": int(finite.size),
        "total_point_count": int(values.size),
        "mean_corr": np.nan,
        "median_corr": np.nan,
        "q25_corr": np.nan,
        "q75_corr": np.nan,
        "min_corr": np.nan,
        "max_corr": np.nan,
    }
    if finite.size == 0:
        return summary

    summary.update(
        {
            "mean_corr": float(np.mean(finite)),
            "median_corr": float(np.median(finite)),
            "q25_corr": float(np.percentile(finite, 25.0)),
            "q75_corr": float(np.percentile(finite, 75.0)),
            "min_corr": float(np.min(finite)),
            "max_corr": float(np.max(finite)),
        }
    )
    return summary


def interpolate_to_observations(
    field: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    lat = np.asarray(grid_lat, dtype=float)
    lon = np.asarray(grid_lon, dtype=float)
    target_lat = np.asarray(obs_lat, dtype=float).reshape(-1)
    target_lon = np.asarray(obs_lon, dtype=float).reshape(-1)

    if values.ndim != 2 or lat.shape != values.shape or lon.shape != values.shape:
        raise ValueError("field, grid_lat, and grid_lon must have the same 2D shape.")
    if target_lat.shape != target_lon.shape:
        raise ValueError("obs_lat and obs_lon must have the same shape.")

    source_valid = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    if source_valid.sum() < 3:
        raise ValueError("At least three finite source grid points are required.")
    if not np.all(np.isfinite(target_lat) & np.isfinite(target_lon)):
        raise ValueError("Observation coordinates must be finite.")

    source_points = np.column_stack((lon[source_valid], lat[source_valid]))
    source_values = values[source_valid]
    target_points = np.column_stack((target_lon, target_lat))
    triangulation = Delaunay(source_points)
    simplices = triangulation.find_simplex(target_points, tol=1.0e-12)
    outside = np.flatnonzero(simplices < 0)
    if outside.size:
        display_indices = ", ".join(str(index + 1) for index in outside[:10])
        suffix = " ..." if outside.size > 10 else ""
        raise ValueError(
            f"{outside.size} observation points are outside the member domain "
            f"(1-based indices: {display_indices}{suffix})."
        )

    transforms = triangulation.transform[simplices, :2, :]
    offsets = target_points - triangulation.transform[simplices, 2, :]
    first_weights = np.einsum("nij,nj->ni", transforms, offsets)
    weights = np.column_stack(
        (first_weights, 1.0 - np.sum(first_weights, axis=1))
    )
    vertices = triangulation.simplices[simplices]
    return np.sum(source_values[vertices] * weights, axis=1)


def _as_float_array(data: np.ndarray) -> np.ndarray:
    return np.asarray(np.ma.filled(data, np.nan), dtype=float)


def _as_horizontal_field(data: np.ndarray, variable_name: str) -> np.ndarray:
    array = _as_float_array(data)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"{variable_name} must be 2D after removing Time; got {array.shape}."
        )
    return array


def _as_omtmp_level_zero(data: np.ndarray, variable_name: str) -> np.ndarray:
    array = _as_float_array(data)
    if array.ndim == 4:
        return array[0, 0, :, :]
    if array.ndim == 3:
        return array[0, :, :]
    raise ValueError(
        f"{variable_name} must have (Time, level, y, x) or (level, y, x) "
        f"dimensions; got {array.shape}."
    )


def load_current_omtmp(
    config: Config, coordinates: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    members = range(config.member_start, config.member_end + 1)
    obs_lat = coordinates["lat"].to_numpy(dtype=float)
    obs_lon = coordinates["lon"].to_numpy(dtype=float)
    n_members = config.member_end - config.member_start + 1
    values = np.full(
        (n_members, config.expected_obs_count),
        np.nan,
        dtype=float,
    )
    rows: list[dict[str, float | int]] = []

    for member_offset, member in enumerate(members):
        path = member_path(config, member)
        if not path.exists():
            raise FileNotFoundError(f"Current member file not found: {path}")
        with nc.Dataset(path) as dataset:
            missing = [
                name
                for name in (config.lat_var, config.lon_var, config.omtmp_var)
                if name not in dataset.variables
            ]
            if missing:
                raise KeyError(f"{path} is missing variables: {', '.join(missing)}")
            grid_lat = _as_horizontal_field(
                dataset.variables[config.lat_var][:],
                config.lat_var,
            )
            grid_lon = _as_horizontal_field(
                dataset.variables[config.lon_var][:],
                config.lon_var,
            )
            omtmp_level_zero = _as_omtmp_level_zero(
                dataset.variables[config.omtmp_var][:],
                config.omtmp_var,
            )

        interpolated = interpolate_to_observations(
            omtmp_level_zero,
            grid_lat,
            grid_lon,
            obs_lat,
            obs_lon,
        )
        values[member_offset, :] = interpolated
        for obs_offset, value in enumerate(interpolated):
            rows.append(
                {
                    "member": member,
                    "obs_index": int(coordinates.iloc[obs_offset]["obs_index"]),
                    "lat": float(obs_lat[obs_offset]),
                    "lon": float(obs_lon[obs_offset]),
                    "omtmp_level": 0,
                    "omtmp": float(value),
                }
            )

    return values, pd.DataFrame(rows)


def load_hx_ensemble(config: Config, lag_hours: list[int]) -> np.ndarray:
    members = range(config.member_start, config.member_end + 1)
    n_members = config.member_end - config.member_start + 1
    hx = np.full(
        (len(lag_hours), n_members, config.expected_obs_count),
        np.nan,
        dtype=float,
    )
    for lag_offset, lag_hour in enumerate(lag_hours):
        for member_offset, member in enumerate(members):
            hx[lag_offset, member_offset, :] = read_hx_file(
                hx_path(config, member, lag_hour),
                expected_count=config.expected_obs_count,
            )
    return hx


@dataclass(frozen=True)
class LaccResults:
    single_pointwise: pd.DataFrame
    single_summary: pd.DataFrame
    averaged_pointwise: pd.DataFrame
    averaged_summary: pd.DataFrame
    omtmp_interpolated: pd.DataFrame


def _pointwise_rows(
    coordinates: pd.DataFrame,
    correlations: np.ndarray,
    extra_fields: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for obs_offset, correlation in enumerate(correlations):
        coordinate = coordinates.iloc[obs_offset]
        rows.append(
            {
                **extra_fields,
                "obs_index": int(coordinate["obs_index"]),
                "lat": float(coordinate["lat"]),
                "lon": float(coordinate["lon"]),
                "corr": float(correlation),
            }
        )
    return rows


def calculate_lacc_correlations(config: Config) -> LaccResults:
    validate_config(config)
    lag_hours = build_lag_hours(config)
    coordinates = read_profile_coordinates(
        config.profile_path,
        expected_count=config.expected_obs_count,
    )
    hx = load_hx_ensemble(config, lag_hours)
    omtmp, omtmp_table = load_current_omtmp(config, coordinates)
    averaged_hx = build_averaged_hx(hx)

    single_pointwise_rows: list[dict[str, object]] = []
    single_summary_rows: list[dict[str, object]] = []
    averaged_pointwise_rows: list[dict[str, object]] = []
    averaged_summary_rows: list[dict[str, object]] = []

    for lag_offset, lag_hour in enumerate(lag_hours):
        valid_time = hx_valid_time(config, lag_hour).strftime(TIME_FORMAT)
        correlations = pointwise_correlations(omtmp, hx[lag_offset])
        single_fields = {
            "lag_hours": lag_hour,
            "hx_valid_time": valid_time,
        }
        single_pointwise_rows.extend(
            _pointwise_rows(coordinates, correlations, single_fields)
        )
        single_summary_rows.append(
            {
                **single_fields,
                **summarize_pointwise(correlations),
            }
        )

        included_lags = lag_hours[: lag_offset + 1]
        included_text = ",".join(str(value) for value in included_lags)
        window_size = lag_offset + 1
        averaged_correlations = pointwise_correlations(
            omtmp,
            averaged_hx[lag_offset],
        )
        averaged_fields = {
            "window_size": window_size,
            "window_label": f"Ave{window_size}",
            "max_lag_hours": lag_hour,
            "included_lag_hours": included_text,
        }
        averaged_pointwise_rows.extend(
            _pointwise_rows(
                coordinates,
                averaged_correlations,
                averaged_fields,
            )
        )
        averaged_summary_rows.append(
            {
                **averaged_fields,
                **summarize_pointwise(averaged_correlations),
            }
        )

    return LaccResults(
        single_pointwise=pd.DataFrame(single_pointwise_rows),
        single_summary=pd.DataFrame(single_summary_rows),
        averaged_pointwise=pd.DataFrame(averaged_pointwise_rows),
        averaged_summary=pd.DataFrame(averaged_summary_rows),
        omtmp_interpolated=omtmp_table,
    )


def plot_lacc_correlations(
    single_summary: pd.DataFrame,
    averaged_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
    with mpl.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.0, 4.4),
            sharey=True,
        )

        single = single_summary.sort_values("lag_hours")
        single_x = single["lag_hours"].to_numpy(dtype=float)
        single_mean = single["mean_corr"].to_numpy(dtype=float)
        single_q25 = single["q25_corr"].to_numpy(dtype=float)
        single_q75 = single["q75_corr"].to_numpy(dtype=float)
        axes[0].fill_between(
            single_x,
            single_q25,
            single_q75,
            color="#72A0C1",
            alpha=0.28,
            linewidth=0.0,
            label="Spatial IQR",
        )
        axes[0].plot(
            single_x,
            single_mean,
            color="#1F4E79",
            marker="o",
            linewidth=2.0,
            label="Mean signed correlation",
        )
        axes[0].set_title("Single-time lead-lag correlation")
        axes[0].set_xlabel("Lead time of Hx (hours)")
        axes[0].set_ylabel("Correlation across ensemble members")
        axes[0].set_xticks(single_x)

        averaged = averaged_summary.sort_values("window_size")
        averaged_x = averaged["max_lag_hours"].to_numpy(dtype=float)
        averaged_mean = averaged["mean_corr"].to_numpy(dtype=float)
        averaged_q25 = averaged["q25_corr"].to_numpy(dtype=float)
        averaged_q75 = averaged["q75_corr"].to_numpy(dtype=float)
        axes[1].fill_between(
            averaged_x,
            averaged_q25,
            averaged_q75,
            color="#DDA15E",
            alpha=0.28,
            linewidth=0.0,
            label="Spatial IQR",
        )
        axes[1].plot(
            averaged_x,
            averaged_mean,
            color="#A44A3F",
            marker="o",
            linewidth=2.0,
            label="Mean signed correlation",
        )
        axes[1].set_title("Leading-averaged Hx correlation")
        axes[1].set_xlabel("Maximum included lead time (hours)")
        axes[1].set_xticks(averaged_x)
        axes[1].set_xticklabels(
            [
                f"{label}\n[{lags}]"
                for label, lags in zip(
                    averaged["window_label"],
                    averaged["included_lag_hours"],
                )
            ]
        )

        for axis in axes:
            axis.axhline(0.0, color="0.3", linestyle="--", linewidth=0.9)
            axis.set_ylim(-1.05, 1.05)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")

        fig.suptitle("LACC: current OM_TMP level 0 versus leading satellite Hx")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def output_paths(config: Config) -> dict[str, Path]:
    tag = (
        f"{config.domain}_ch{config.channel}_"
        f"{config.current_time.replace('-', '').replace(':', '')}"
    )
    return {
        "single_pointwise": config.output_dir
        / f"single_lag_pointwise_{tag}.csv",
        "single_summary": config.output_dir / f"single_lag_summary_{tag}.csv",
        "averaged_pointwise": config.output_dir
        / f"averaged_window_pointwise_{tag}.csv",
        "averaged_summary": config.output_dir
        / f"averaged_window_summary_{tag}.csv",
        "omtmp_interpolated": config.output_dir
        / f"omtmp_interpolated_{tag}.csv",
        "figure": config.output_dir / f"lacc_hx_omtmp_corr_{tag}.png",
    }


def write_outputs(config: Config, results: LaccResults) -> dict[str, Path]:
    paths = output_paths(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.single_pointwise.to_csv(paths["single_pointwise"], index=False)
    results.single_summary.to_csv(paths["single_summary"], index=False)
    results.averaged_pointwise.to_csv(paths["averaged_pointwise"], index=False)
    results.averaged_summary.to_csv(paths["averaged_summary"], index=False)
    results.omtmp_interpolated.to_csv(paths["omtmp_interpolated"], index=False)
    plot_lacc_correlations(
        results.single_summary,
        results.averaged_summary,
        paths["figure"],
    )
    return paths


def run(config: Config = CONFIG) -> tuple[LaccResults, dict[str, Path]]:
    results = calculate_lacc_correlations(config)
    paths = write_outputs(config, results)
    return results, paths


def main() -> None:
    results, paths = run(CONFIG)
    print("LACC ensemble-correlation diagnostic completed.")
    print(
        "Lag hours:",
        ", ".join(str(value) for value in results.single_summary["lag_hours"]),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
