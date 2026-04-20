
%20201017再次检查
%2021.1.10 检查发现这是superobs啊 
clc;clear
domain=getenv('domain')
% domain='d02'
%domain='d02'
obs_mem_dir=getenv('obs_mem_dir');
member=getenv('member')
instrument=getenv('instrument')


time_day=getenv('obs_day')
time_hour=getenv('obs_hour')
time_min=getenv('obs_min')
time=strcat(time_day,'_',time_hour,'_',time_min)

str1=[obs_mem_dir '/' instrument '/ens' member '_' domain '_output_'];
obs_all=[str1 time '.txt'] %all chnum*(point^2) obs
% time='04_01_50'
%time='04_01_50'
point=sqrt(str2num(getenv('npoint')))
%point=127
ind=5
chnum=str2num(getenv('chnum'))
%chnum=1650
obs_nr=[obs_mem_dir '/' instrument '/BT_' time '/'];

dataA=textread(obs_all);%
dataB=reshape(dataA',chnum,point*point);
% data: row*mem

%% data_ch: row*mem*ch            
        
%data的顺序是一列一列的

%for chnumi=1:chnum
for chnumi=1:chnum
bt_60_1d=dataB(chnumi,:);%竖着一列一列读，因为其他 ens 的就是一列一列
dataset2=reshape(bt_60_1d,point,point);
dlmwrite([obs_nr 'obs_' domain '_ch' num2str(chnumi) '.txt'],dataset2,'precision', '%.4f', 'delimiter', '\t')
dlmwrite([obs_nr 'obs_' domain '_ch' num2str(chnumi) '_totalline.txt'],bt_60_1d','precision', '%.4f', 'delimiter', '\t')
%生成 obs 顺序：一列一列的，在 diagfile 中要与ens相减 
%%
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
% set(gca,'XTicklabel',{-(12)*7.5,-6*7.5,0,6*7.5,(12)*7.5});% X轴的记号

% set(gca,'YLim',[1 point]);% X轴的数据显示范围 
% set(gca,'YTick',[1:6*ind:point] );% X轴的记号点
% set(gca,'YTicklabel',{-(12)*7.5,-6*7.5,0,6*7.5,(12)*7.5});% X轴的记号

% set(gca,'FontSize',12); %只能同时改变x y轴显示的字体大小。
% colorbar('eastoutside')
% title(['ch' num2str(chnumi) '  time: ' time(1:2) ':' time(4:5) ],'FontSize',15)
% xlabel('  km','FontSize',12)
% ylabel('  km','FontSize',12)
% saveas(gcf,[obs_nr 'pics_obsnr/obsnr_d04_ch' num2str(chnumi) '.jpg'])
%=============================================================================================
zlf=['zlf ' num2str(chnumi) ' done']
end

