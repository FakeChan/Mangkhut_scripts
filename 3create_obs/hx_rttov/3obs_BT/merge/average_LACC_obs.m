clc; clear

domain = getenv('domain');
obsdir = getenv('obs_dir');
instrument = getenv('instrument');
lacc_times = strsplit(strtrim(getenv('lacc_times')));
lacc_center_time = getenv('lacc_center_time');
obserr_std = str2num(getenv('obserr_std'));
point = sqrt(str2num(getenv('npoint')));
chnum = str2num(getenv('chnum'));

if isempty(lacc_times) || isempty(lacc_times{1})
    error('lacc_times is empty.');
end
if isempty(lacc_center_time)
    error('lacc_center_time is empty.');
end

base_dir = [obsdir '/' instrument '/'];
out_dir = [base_dir 'BT_LACC_' lacc_center_time '/'];
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
mask_sum = zeros(point * point, 1);
mask_count = 0;
hydrometeor_path_sum = zeros(point * point, 1);

fid = fopen([out_dir 'LACC_times.txt'], 'w');
fprintf(fid, 'center_time=%s\n', lacc_center_time);
for it = 1:length(lacc_times)
    fprintf(fid, 'lag_time=%s\n', lacc_times{it});
    mask_file = [base_dir 'BT_' lacc_times{it} '/clear_sky_mask.txt'];
    hydrometeor_path_file = [base_dir 'BT_' lacc_times{it} '/hydrometeor_path.txt'];
    if exist(mask_file, 'file')
        mask_sum = mask_sum + load(mask_file);
        mask_count = mask_count + 1;
    end
    if exist(hydrometeor_path_file, 'file')
        hydrometeor_path_sum = hydrometeor_path_sum + load(hydrometeor_path_file);
    end
end
fclose(fid);

if mask_count > 0
    if mask_count ~= length(lacc_times)
        error('Only %d/%d LACC lag times have clear_sky_mask.txt.', mask_count, length(lacc_times));
    end
    n_lacc_times = length(lacc_times);
    clear_sky_frac_env = getenv('CLEAR_SKY_MIN_FRACTION');
    if isempty(clear_sky_frac_env)
        clear_sky_frac = 2/3;                     % default: 2/3
    else
        clear_sky_frac = str2double(clear_sky_frac_env);
        if isnan(clear_sky_frac) || clear_sky_frac <= 0 || clear_sky_frac > 1
            error('CLEAR_SKY_MIN_FRACTION must be in (0,1], got %s.', clear_sky_frac_env);
        end
    end
    required_times = floor(clear_sky_frac * n_lacc_times);
    % guard: for N=1, floor(2/3) = 0 would let every point pass;
    % require at least 1 clear time so the filter still applies.
    required_times = max(1, min(required_times, n_lacc_times));
    clear_sky_mask = mask_sum >= required_times;
    hydrometeor_path_mean = hydrometeor_path_sum ./ n_lacc_times;
    dlmwrite([out_dir 'clear_sky_mask.txt'], clear_sky_mask, 'precision', '%d', 'delimiter', '\t');
    dlmwrite([out_dir 'hydrometeor_path.txt'], hydrometeor_path_mean, 'precision', '%.8f', 'delimiter', '\t');
    fprintf('LACC clear-sky mask written to %s; clear obs = %d / %d; requires >= %d of %d lag-time clear\n', ...
        [out_dir 'clear_sky_mask.txt'], sum(clear_sky_mask), length(clear_sky_mask), ...
        required_times, n_lacc_times);
end

for chnumi = 1:chnum
    bt_sum = zeros(point, point);
    bt_withpert_sum = zeros(point, point);
    randn('state', chnumi);
    for it = 1:length(lacc_times)
        in_dir = [base_dir 'BT_' lacc_times{it} '/'];
        in_file = [in_dir 'obs_' domain '_ch' num2str(chnumi) '.txt'];
        if ~exist(in_file, 'file')
            error('Missing LACC input file: %s', in_file);
        end
        bt_lag = load(in_file);
        bt_sum = bt_sum + bt_lag;

        rand = randn(point, point);
        obs_rand = obserr_std * (rand - mean(mean(rand)));
        bt_withpert_sum = bt_withpert_sum + bt_lag + obs_rand;
    end

    bt_mean = bt_sum ./ length(lacc_times);
    bt_1d = reshape(bt_mean, point * point, 1);
    bt_withpert_mean = bt_withpert_sum ./ length(lacc_times);

    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '.txt'], ...
        bt_mean, 'precision', '%.4f', 'delimiter', '\t');
    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '_totalline.txt'], ...
        bt_1d, 'precision', '%.4f', 'delimiter', '\t');

    bt_withpert_1d = reshape(bt_withpert_mean, point * point, 1);
    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '_totalline_withpert.txt'], ...
        bt_withpert_1d, 'precision', '%.4f', 'delimiter', '\t');

    fprintf('LACC obs channel %d/%d done\n', chnumi, chnum);
end

fprintf('LACC obs mean written to %s\n', out_dir);
