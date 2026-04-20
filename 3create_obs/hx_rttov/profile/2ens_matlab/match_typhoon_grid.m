function [b_lat, b_lon, b_i_index, b_j_index] = match_typhoon_grid(file_A_path, file_B_path)
% MATCH_TYPHOON_GRID  在文件B中找到距离文件A台风中心最近的格点
%
%   输入:
%       file_A_path : 参考文件路径 (用于确定台风中心经纬度)
%       file_B_path : 目标文件路径 (需要在该文件中找对应格点)
%
%   输出:
%       b_lat      : 文件 B 中匹配点的纬度
%       b_lon      : 文件 B 中匹配点的经度
%       b_i_index  : 文件 B 中匹配点的 West-East 索引 (Grid X)
%       b_j_index  : 文件 B 中匹配点的 South-North 索引 (Grid Y)
%

    %% --- 步骤 1: 读取文件 A 并定位台风中心 ---
    fprintf('正在处理文件 A: %s ...\n', file_A_path);
    
    if ~isfile(file_A_path) || ~isfile(file_B_path)
        error('错误: 输入的文件路径不存在。');
    end

    % 1.1 获取气压变量 (优先找 SLP, 否则用 PSFC)
    var_info_A = ncinfo(file_A_path);
    var_names_A = {var_info_A.Variables.Name};
    
    if any(strcmpi(var_names_A, 'SLP'))
        p_var_name = 'SLP';  % 或者 'slp'
    elseif any(strcmp(var_names_A, 'PSFC'))
        p_var_name = 'PSFC';
    else
        error('文件 A 中未找到 SLP 或 PSFC 变量，无法定位台风。');
    end
    
    % 1.2 读取数据 (气压 + 经纬度)
    % 注意: ncread 返回维度通常为 [West-East, South-North, Time]
    p_data_A = ncread(file_A_path, p_var_name);
    lat_data_A = ncread(file_A_path, 'XLAT');
    lon_data_A = ncread(file_A_path, 'XLONG');
    
    % 处理时间维度 (默认取第1个时次)
    if ndims(p_data_A) == 3
        p_field_A = p_data_A(:,:,1);
        % 经纬度通常也有时间维，也取第1个时次
        lat_field_A = lat_data_A(:,:,1);
        lon_field_A = lon_data_A(:,:,1);
    else
        p_field_A = p_data_A;
        lat_field_A = lat_data_A;
        lon_field_A = lon_data_A;
    end
    
    % 1.3 找最低气压索引
    [min_p_val, min_idx_linear] = min(p_field_A(:));
    
    % 获取中心点的经纬度
    center_lat = lat_field_A(min_idx_linear);
    center_lon = lon_field_A(min_idx_linear);
    
    fprintf('  > 文件 A 台风中心定位成功: %.2f Pa\n', min_p_val);
    fprintf('  > 目标坐标: Lat = %.4f, Lon = %.4f\n', center_lat, center_lon);


    %% --- 步骤 2: 在文件 B 中寻找最近格点 ---
    fprintf('正在处理文件 B: %s ...\n', file_B_path);

    % 2.1 读取文件 B 的经纬度网格
    lat_data_B = ncread(file_B_path, 'XLAT');
    lon_data_B = ncread(file_B_path, 'XLONG');
    
    if ndims(lat_data_B) == 3
        lat_field_B = lat_data_B(:,:,1);
        lon_field_B = lon_data_B(:,:,1);
    else
        lat_field_B = lat_data_B;
        lon_field_B = lon_data_B;
    end
    
    % 2.2 计算距离 (欧几里得距离平方，寻找最小值即可)
    % 距离^2 = (Lat - TargetLat)^2 + (Lon - TargetLon)^2
    % 注意：如果跨越区域非常大，应使用 Haversine 公式，但在网格匹配中，平方差通常足够精确。
    dist_sq = (lat_field_B - center_lat).^2 + (lon_field_B - center_lon).^2;
    
    % 2.3 找到最小距离的索引
    [min_dist, min_idx_B_linear] = min(dist_sq(:));
    
    % 将线性索引转换为二维下标 [i, j]
    % size(dist_sq) 顺序为 [West-East, South-North]
    [i_idx, j_idx] = ind2sub(size(dist_sq), min_idx_B_linear);
    
    % 2.4 提取结果
    b_lat = lat_field_B(min_idx_B_linear);
    b_lon = lon_field_B(min_idx_B_linear);
    b_i_index = i_idx;
    b_j_index = j_idx;
    
    %% --- 输出结果 ---
    fprintf('------------------------------------------------\n');
    fprintf('匹配完成。\n');
    fprintf('文件 B 中最近格点:\n');
    fprintf('  > 位置: Lat = %.4f, Lon = %.4f\n', b_lat, b_lon);
    fprintf('  > 索引: i_index (WE) = %d, j_index (SN) = %d\n', b_i_index, b_j_index);
    fprintf('------------------------------------------------\n');

end
