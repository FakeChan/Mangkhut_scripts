import netCDF4
import numpy as np


EARTH_RADIUS_KM = 6371.0

# Set these values before running this script on the remote server.
PREV_NC_PATH = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d01_2018-09-09_18:00:00'
CURR_NC_PATH = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/output_mean_d01.nc'
# CURR_NC_PATH = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d01_2018-09-10_00:00:00'
WAKE_LAG_KM = 200.0
WAKE_LENGTH_KM = 300.0
WAKE_HALF_WIDTH_KM = 100.0

def get_TClocation(file):
    with netCDF4.Dataset(file, 'r') as data:
        p = data.variables['P'][:]

    if p.ndim == 4:
        p_field = p[0, 0, :, :]
    elif p.ndim == 3:
        p_field = p[0, :, :]
    elif p.ndim == 2:
        p_field = p
    else:
        raise ValueError(f'Unsupported P dimensions: {p.shape}')

    min_idx = np.unravel_index(np.nanargmin(p_field), p_field.shape)
    jTC = min_idx[0]
    iTC = min_idx[1]
    return iTC,jTC

def _first_time_2d(var):
    """Return the first vertical level as a 2-D field."""
    data = var[:]

    if data.ndim == 4:
        return data[0, 0, :, :]
    if data.ndim == 3:
        return data[0, :, :]
    if data.ndim == 2:
        return data

    raise ValueError(f'Unsupported variable dimensions for {var.name}: {data.shape}')


def _latlon_2d(var):
    data = var[:]

    if data.ndim == 3:
        return data[0, :, :]
    if data.ndim == 2:
        return data

    raise ValueError(f'Unsupported lat/lon dimensions for {var.name}: {data.shape}')


def _local_xy_km(lat, lon, center_lat, center_lon):
    x = (
        EARTH_RADIUS_KM
        * np.deg2rad(lon - center_lon)
        * np.cos(np.deg2rad(center_lat))
    )
    y = EARTH_RADIUS_KM * np.deg2rad(lat - center_lat)
    return x, y


def _get_tc_center_latlon(nc_path):
    iTC, jTC = get_TClocation(nc_path)

    with netCDF4.Dataset(nc_path, 'r') as nc:
        lats = np.ma.filled(_latlon_2d(nc.variables['XLAT']), np.nan)
        lons = np.ma.filled(_latlon_2d(nc.variables['XLONG']), np.nan)

    return {
        'i': int(iTC),
        'j': int(jTC),
        'lat': float(lats[jTC, iTC]),
        'lon': float(lons[jTC, iTC]),
    }


def calculate_om_tmp_wake_front_diff(
    curr_nc_path,
    prev_nc_path,
    wake_lag_km,
    wake_length_km,
    wake_half_width_km,
):
    """
    Calculate the OM_TMP surface-layer cold-wake signal.

    Parameters
    ----------
    curr_nc_path : str
        Path to the current WRF NetCDF file.
    prev_nc_path : str
        Path to the previous WRF NetCDF file. It is used to define TC motion.
    wake_lag_km : float
        Distance from the current TC center to skip before calculating the wake.
    wake_length_km : float
        Along-track length of the rear/front corridor after the lag distance.
    wake_half_width_km : float
        Half width of the rear/front corridor.

    Returns
    -------
    dict
        Current/previous TC center, wake mean, front mean, and diff.
        diff = wake_mean - front_mean.
    """
    prev_center = _get_tc_center_latlon(prev_nc_path)
    curr_center = _get_tc_center_latlon(curr_nc_path)

    motion_x, motion_y = _local_xy_km(
        curr_center['lat'],
        curr_center['lon'],
        prev_center['lat'],
        prev_center['lon'],
    )
    motion_norm = np.hypot(motion_x, motion_y)
    if motion_norm == 0.0:
        raise ValueError('TC center did not move between previous and current files.')

    motion_unit_x = motion_x / motion_norm
    motion_unit_y = motion_y / motion_norm
    wake_unit_x = -motion_unit_x
    wake_unit_y = -motion_unit_y

    with netCDF4.Dataset(curr_nc_path, 'r') as nc:
        om_tmp_sfc = np.ma.filled(_first_time_2d(nc.variables['OM_TMP']), np.nan)
        lats = np.ma.filled(_latlon_2d(nc.variables['XLAT']), np.nan)
        lons = np.ma.filled(_latlon_2d(nc.variables['XLONG']), np.nan)

    grid_x, grid_y = _local_xy_km(
        lats,
        lons,
        curr_center['lat'],
        curr_center['lon'],
    )

    along_wake_km = grid_x * wake_unit_x + grid_y * wake_unit_y
    cross_wake_km = grid_x * (-wake_unit_y) + grid_y * wake_unit_x

    along_front_km = grid_x * motion_unit_x + grid_y * motion_unit_y
    cross_front_km = grid_x * (-motion_unit_y) + grid_y * motion_unit_x

    wake_mask = (
        (along_wake_km >= wake_lag_km)
        & (along_wake_km <= wake_lag_km + wake_length_km)
        & (np.abs(cross_wake_km) <= wake_half_width_km)
    )
    front_mask = (
        (along_front_km >= wake_lag_km)
        & (along_front_km <= wake_lag_km + wake_length_km)
        & (np.abs(cross_front_km) <= wake_half_width_km)
    )

    wake_mean = float(np.nanmean(np.where(wake_mask, om_tmp_sfc, np.nan)))
    front_mean = float(np.nanmean(np.where(front_mask, om_tmp_sfc, np.nan)))

    return {
        'prev_center_i': prev_center['i'],
        'prev_center_j': prev_center['j'],
        'prev_center_lat': prev_center['lat'],
        'prev_center_lon': prev_center['lon'],
        'curr_center_i': curr_center['i'],
        'curr_center_j': curr_center['j'],
        'curr_center_lat': curr_center['lat'],
        'curr_center_lon': curr_center['lon'],
        'motion_distance_km': float(motion_norm),
        'wake_lag_km': float(wake_lag_km),
        'wake_length_km': float(wake_length_km),
        'wake_half_width_km': float(wake_half_width_km),
        'wake_mean': wake_mean,
        'front_mean': front_mean,
        'diff': wake_mean - front_mean,
    }


def main():
    result = calculate_om_tmp_wake_front_diff(
        CURR_NC_PATH,
        PREV_NC_PATH,
        WAKE_LAG_KM,
        WAKE_LENGTH_KM,
        WAKE_HALF_WIDTH_KM,
    )
    for key, value in result.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()