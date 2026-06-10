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

fid = fopen([out_dir 'LACC_times.txt'], 'w');
fprintf(fid, 'center_time=%s\n', lacc_center_time);
for it = 1:length(lacc_times)
    fprintf(fid, 'lag_time=%s\n', lacc_times{it});
end
fclose(fid);

for chnumi = 1:chnum
    bt_sum = zeros(point, point);
    for it = 1:length(lacc_times)
        in_dir = [base_dir 'BT_' lacc_times{it} '/'];
        in_file = [in_dir 'obs_' domain '_ch' num2str(chnumi) '.txt'];
        if ~exist(in_file, 'file')
            error('Missing LACC input file: %s', in_file);
        end
        bt_sum = bt_sum + load(in_file);
    end

    bt_mean = bt_sum ./ length(lacc_times);
    bt_1d = reshape(bt_mean, point * point, 1);

    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '.txt'], ...
        bt_mean, 'precision', '%.4f', 'delimiter', '\t');
    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '_totalline.txt'], ...
        bt_1d, 'precision', '%.4f', 'delimiter', '\t');

    randn('state', chnumi);
    rand = randn(point, point);
    obs_rand = obserr_std * (rand - mean(mean(rand)));
    bt_withpert = bt_mean + obs_rand;
    bt_withpert_1d = reshape(bt_withpert, point * point, 1);
    dlmwrite([out_dir 'obs_' domain '_ch' num2str(chnumi) '_totalline_withpert.txt'], ...
        bt_withpert_1d, 'precision', '%.4f', 'delimiter', '\t');

    fprintf('LACC obs channel %d/%d done\n', chnumi, chnum);
end

fprintf('LACC obs mean written to %s\n', out_dir);
