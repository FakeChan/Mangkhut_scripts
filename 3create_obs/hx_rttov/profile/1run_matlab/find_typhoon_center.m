function [center_lat, center_lon, min_mslp_val, i_idx, j_idx] = find_typhoon_center(filename, time_idx)
    % FIND_TYPHOON_CENTER 定位WRF输出中的台风中心（基于最小MSLP）
    %
    % 输入:
    %   filename - wrfout 文件的路径 (字符串), 例如 'wrfout_d01_2025-11-24_00:00:00'
    %   time_idx - 时间步索引 (整数), 默认为 1
    %
    % 输出:
    %   center_lat   - 台风中心的纬度
    %   center_lon   - 台风中心的经度
    %   min_mslp_val - 中心的海平面气压值 (hPa)
    %   i_idx, j_idx - 中心在网格中的索引 (x, y)
    %
    % 算法:
    %   利用 PSFC (表面气压), T2 (2米气温), Q2 (2米比湿), HGT (地形高度)
    %   通过静力学方程将气压订正到海平面。
    
        if nargin < 2
            time_idx = 1;
        end
    
        % --- 1. 读取必要的 WRF 变量 ---
        % 注意：ncread 的读取范围是 [start, count]。WRF 维度通常为 [West-East, South-North, Time]
        
        % 读取经纬度
        lat = ncread(filename, 'XLAT', [1 1 time_idx], [Inf Inf 1]);
        lon = ncread(filename, 'XLONG', [1 1 time_idx], [Inf Inf 1]);
        
        % 读取气象变量
        try
            P_sfc = ncread(filename, 'PSFC', [1 1 time_idx], [Inf Inf 1]); % Pa
            T2    = ncread(filename, 'T2', [1 1 time_idx], [Inf Inf 1]);   % K
            HGT   = ncread(filename, 'HGT', [1 1 time_idx], [Inf Inf 1]);  % m
            try
                Q2 = ncread(filename, 'Q2', [1 1 time_idx], [Inf Inf 1]);  % kg/kg
            catch
                warning('变量 Q2 未找到，将忽略湿空气修正。');
                Q2 = zeros(size(T2));
            end
        catch ME
            error(['读取变量失败，请确认文件包含 PSFC, T2, HGT。错误信息: ', ME.message]);
        end
    
        % --- 2. 计算海平面气压 (MSLP) ---
        % 使用标准的气压高度公式进行订正
        % 公式: MSLP = Psfc * exp( (g * H) / (R * Tv) )
        
        % 常数定义
        R_d = 287.05;  % 干空气气体常数 (J/kg/K)
        g   = 9.81;    % 重力加速度 (m/s^2)
        
        % 计算虚温 (Virtual Temperature) Tv = T * (1 + 0.61 * q)
        Tv = T2 .* (1 + 0.61 .* Q2);
        
        % 为了防止除以0或极端低温导致的数值问题，做一个简单的平滑处理(可选)
        % 通常台风都在海面，HGT接近0，PSFC即为MSLP。但在陆地上需要此修正。
        
        % 计算 MSLP (单位保持为 Pa)
        exponent = (g .* HGT) ./ (R_d .* Tv);
        mslp_pa  = P_sfc .* exp(exponent);
        
        % 转换为 hPa
        mslp_hpa = mslp_pa / 100.0;
    
        % --- 3. 寻找最小值定位中心 ---
        
        % 找到最小 MSLP 的线性索引
        [min_mslp_val, linear_idx] = min(mslp_hpa(:));
        
        % 将线性索引转换为二维网格索引 (row, col) -> (West-East, South-North)
        [row, col] = ind2sub(size(mslp_hpa), linear_idx);
        
        % 提取对应的经纬度
        center_lat = lat(row, col);
        center_lon = lon(row, col);
        
        % 保存索引 (注意 MATLAB 是 1-based, WRF/NCView 也是 1-based，但在 Python 中是 0-based)
        i_idx = row; % 对应 WRF 的 west_east 维度
        j_idx = col; % 对应 WRF 的 south_north 维度
    
        % --- 4. (可选) 输出信息 ---
        % fprintf('Time Index: %d\n', time_idx);
        % fprintf('Typhoon Center: Lat = %.4f, Lon = %.4f\n', center_lat, center_lon);
        % fprintf('Min MSLP: %.2f hPa\n', min_mslp_val);
        % fprintf('Grid Index: (%d, %d)\n', i_idx, j_idx);
    
    end