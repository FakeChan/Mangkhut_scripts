% 20201017再次检查
% 2021.1.10 检查发现这是superobs啊 
% 修改说明：已更改读取方式以适应跨行数据格式 (2025)

clc; clear

domain = getenv('domain');
% domain='d01'; % 调试用

obsdir = getenv('obs_dir');
instrument = getenv('instrument');

time_day = getenv('obs_day');
time_hour = getenv('obs_hour');
time_min = getenv('obs_min');
time = strcat(time_day, '_', time_hour, '_', time_min);

str1 = [obsdir '/' instrument '/NR_' domain '_output_'];
obs_all = [str1 time '.txt']; % all chnum*(point^2) obs

% 获取网格点数和通道数
point = sqrt(str2num(getenv('npoint')));
% point = 26; % 调试用: sqrt(676) = 26
chnum = str2num(getenv('chnum'));
% chnum = 15; % 调试用

obs_nr = [obsdir '/' instrument '/BT_' time '/'];

%% --- 核心修改部分 开始 ---

% 原代码:
% dataA=textread(obs_all);
% dataB=reshape(dataA',chnum,point*point);

% 新代码: 使用 fscanf 流式读取，忽略换行符的影响
fprintf('正在读取文件: %s ...\n', obs_all);
fid = fopen(obs_all, 'r');
if fid == -1
    error('无法打开文件: %s', obs_all);
end

% '%f' 会读取所有浮点数，忽略空格、换行和空行，返回一个长列向量
data_vector = fscanf(fid, '%f'); 
fclose(fid);

% 校验数据量是否正确
total_expected = chnum * point * point;
if length(data_vector) ~= total_expected
    warning('警告: 读取的数据量 (%d) 与预期 (%d) 不符！请检查 npoint 或 chnum 设置。', length(data_vector), total_expected);
end

% Reshape: 
% 文件中的顺序是: [Obs1_Ch1, Obs1_Ch2 ... Obs1_Ch15, Obs2_Ch1 ...]
% MATLAB 的 reshape 是先填充列 (Column-major)。
% 这里的目标形状是 (chnum, total_obs)，即每一列代表一个观测点的所有通道。
% reshape(vector, chnum, total_obs) 会将前 chnum 个数放入第一列，正是我们要的效果。
dataB = reshape(data_vector, chnum, point*point);

%% --- 核心修改部分 结束 ---


% dataB: row(通道) * col(观测点mem)

%% data_ch: row*mem*ch            
        
% for chnumi=1:chnum
for chnumi = 1:chnum
    % 提取第 chnumi 个通道的所有观测点数据
    bt_60_1d = dataB(chnumi, :); 
    
    % Reshape 回二维网格 (point x point)
    dataset2 = reshape(bt_60_1d, point, point);
    
    % 构建输出文件名并保存
    % 注意：使用了转置 dataset2 (MATLAB reshape 是列优先，如果原始数据是行优先可能需要转置，保持原逻辑不变)
    % 这里保留原代码逻辑，直接写出
    
    % 输出格式化矩阵文件
    output_mat_name = [obs_nr 'obs_' domain '_ch' num2str(chnumi) '.txt'];
    
    % 检查目录是否存在，不存在则创建
    if ~exist(obs_nr, 'dir')
        mkdir(obs_nr);
    end
    
    dlmwrite(output_mat_name, dataset2, 'precision', '%.4f', 'delimiter', '\t');
    
    % 输出单列文件
    output_line_name = [obs_nr 'obs_' domain '_ch' num2str(chnumi) '_totalline.txt'];
    dlmwrite(output_line_name, bt_60_1d', 'precision', '%.4f', 'delimiter', '\t');
    
    % 显示进度
    if mod(chnumi, 5) == 0
        fprintf('已处理通道: %d / %d\n', chnumi, chnum);
    end
end

fprintf('处理完成。\n');

%=========================================================================================
%fkc:注释掉了画图部分
% figure(chnumi)
% imagesc(flipud(dataset2))
% colormap(jet)
% axis([1,point,1,point]);% 坐标
% %axis([0,0,128,128])

%  colorbar('eastoutside')
% set(gca,'XLim',[1 point]);% X轴的数据显示范围 
% set(gca,'XTick',[1:6*ind:point] );% X轴的记号点
% set(gca,'XTicklabel',{-(12)*7.5,-6...
