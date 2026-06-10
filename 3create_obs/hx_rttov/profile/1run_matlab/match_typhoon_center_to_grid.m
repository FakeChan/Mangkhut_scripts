function [center_lat, center_lon, min_mslp_val, i_idx, j_idx] = match_typhoon_center_to_grid(center_file, target_file)
% MATCH_TYPHOON_CENTER_TO_GRID
% Locate the typhoon center in center_file, then find the nearest grid point
% in target_file. This keeps LACC observations fixed to the center-time NR
% storm position even when the target WRF domain moves with time.

    if ~isfile(center_file)
        error('center_file does not exist: %s', center_file);
    end
    if ~isfile(target_file)
        error('target_file does not exist: %s', target_file);
    end

    [center_lat, center_lon, min_mslp_val, ~, ~] = find_typhoon_center(center_file, 1);

    lat_data = ncread(target_file, 'XLAT');
    lon_data = ncread(target_file, 'XLONG');
    if ndims(lat_data) == 3
        lat_field = lat_data(:,:,1);
        lon_field = lon_data(:,:,1);
    else
        lat_field = lat_data;
        lon_field = lon_data;
    end

    dist_sq = (lat_field - center_lat).^2 + (lon_field - center_lon).^2;
    [~, min_idx_linear] = min(dist_sq(:));
    [i_idx, j_idx] = ind2sub(size(dist_sq), min_idx_linear);

    fprintf('------------------------------------------------\n');
    fprintf('LACC center-time NR location: Lat = %.4f, Lon = %.4f, MSLP = %.2f hPa\n', ...
        center_lat, center_lon, min_mslp_val);
    fprintf('Matched target grid point: i = %d, j = %d, Lat = %.4f, Lon = %.4f\n', ...
        i_idx, j_idx, lat_field(min_idx_linear), lon_field(min_idx_linear));
    fprintf('------------------------------------------------\n');
end
