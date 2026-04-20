function [Zenith_Angle, Azimuth_Angle] = calc_OSSE_satAngles(target_lat, target_lon, center_lat, center_lon)
    % 输入参数:
    % target_lat, target_lon: 目标网格点的经纬度 (可以是标量，也可以是矩阵)
    % center_lat, center_lon: 你的WRF区域中心点经纬度 (设定为卫星的星下点)
    % 
    % 输出参数:
    % Zenith_Angle: 局部天顶角 (单位: 度)
    % Azimuth_Angle: 方位角 (单位: 度，正北为0，顺时针)

    % --- 物理常量设定 ---
    Re = 6371.0;            % 地球平均半径 (km)
    satheight=str2num(getenv('satheight'));             % NOAA-18 卫星轨道高度 (km)
    rs = Re + satheight;            % 卫星到地心的距离
    
    % 将经纬度转换为弧度
    deg2rad = pi / 180.0;
    lat1 = center_lat * deg2rad; % 星下点(中心点)纬度
    lon1 = center_lon * deg2rad; % 星下点(中心点)经度
    lat2 = target_lat * deg2rad; % 目标点纬度
    lon2 = target_lon * deg2rad; % 目标点经度
    
    delta_lon = lon2 - lon1;

    % ==========================================
    % 1. 计算地心角 (Central Angle, gamma)
    % 使用更稳定的球面余弦/半正矢公式替代原有的计算
    % ==========================================
    cos_gamma = sin(lat1).*sin(lat2) + cos(lat1).*cos(lat2).*cos(delta_lon);
    % 防止浮点数误差导致 cos_gamma 略大于 1
    cos_gamma(cos_gamma > 1) = 1; 
    cos_gamma(cos_gamma < -1) = -1;
    
    gamma = acos(cos_gamma); % 地心角 (弧度)
    sin_gamma = sin(gamma);

    % ==========================================
    % 2. 计算局部天顶角 (Zenith Angle)
    % ==========================================
    % 根据平面三角形正弦/余弦定理推导
    tan_zenith = sin_gamma ./ (cos_gamma - (Re / rs));
    zenith_rad = atan(tan_zenith);
    
    % 如果计算出负值(在几何上可能发生于超越地平线的情况，虽WRF区域通常不会)，加上 pi
    zenith_rad(zenith_rad < 0) = zenith_rad(zenith_rad < 0) + pi;
    
    Zenith_Angle = zenith_rad ; % (输出结果)

    % ==========================================
    % 3. 计算方位角 (Azimuth Angle)
    % ==========================================
    y = sin(delta_lon) .* cos(lat1);
    x = cos(lat2) .* sin(lat1) - sin(lat2) .* cos(lat1) .* cos(delta_lon);
    
    azimuth_rad = atan2(y, x);
    
    % 将方位角调整到 0 到 360 度之间
    Azimuth_Angle = mod(azimuth_rad , 2*pi);
    
    % 如果目标点正好在星下点，天顶角为0，方位角无意义(置为0)
    center_idx = (Zenith_Angle < 1e-5);
    Azimuth_Angle(center_idx) = 0.0;

end