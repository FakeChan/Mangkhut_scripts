clear all;clc;
%2021.1.09 find error: randn('state',i);
obsdir=getenv('obs_dir');
time_day=getenv('obs_day')
time_hour=getenv('obs_hour')
time_min=getenv('obs_min')
obserr_std=str2num(getenv('obserr_std'))
time=strcat(time_day,'_',time_hour,'_',time_min)
% time='04_00_30'
chnum=str2num(getenv('chnum'));
point=sqrt(str2num(getenv('npoint'))); % 
domain=getenv('domain')
instrument=getenv('instrument')
ind=5;
% obs_nr=['./'];
obs_nr=[obsdir '/' instrument '/BT_' time '/'];
for i=1:1:chnum
% for i=1025
chmem_obs_d04=load([obs_nr 'obs_' domain '_ch' num2str(i) '.txt']);

randn('state',i);
rand=randn(point,point);
obs_rand =obserr_std*(rand - mean(mean(rand)));         

bt=chmem_obs_d04 + obs_rand;
bt_1d=reshape(bt,point*point,1);%Êú×ÅÒ»ÁÐÒ»ÁÐ¶Á£¬ÎªÁËºÍ ens µÄÊý¾Ý±£³ÖÒ»ÖÂ
%xx=[1:25]';x = kron(xx,ones(1,25));
%y=x;
%x_1d=reshape(x',25*25,1);%heng×Å
%y_1d=reshape(y,25*25,1);%Êú×ÅÒ»ÁÐÒ»ÁÐ¶Á

%scatter(x_1d,y_1d,100,bt_1d,'filled')
%caxis([230,255]);
%====================================================================================================
%fkc:注释掉了画图部分
% figure(i)
% imagesc(flipud(bt))
% colormap(jet)

% axis([1,point,1,point]);% ×ø±ê
% %axis([0,0,128,128])
% colorbar('eastoutside')
% set(gca,'XLim',[1 point]);% XÖáµÄÊý¾ÝÏÔÊ¾·¶Î§ 
% set(gca,'XTick',[1:6*ind:point] );% XÖáµÄ¼ÇºÅµã
% set(gca,'XTicklabel',{-(12)*7.5,-6*7.5,0,6*7.5,(12)*7.5});% XÖáµÄ¼ÇºÅ

% set(gca,'YLim',[1 point]);% XÖáµÄÊý¾ÝÏÔÊ¾·¶Î§ 
% set(gca,'YTick',[1:6*ind:point] );% XÖáµÄ¼ÇºÅµã
% set(gca,'YTicklabel',{-(12)*7.5,-6*7.5,0,6*7.5,(12)*7.5});% XÖáµÄ¼ÇºÅ
% %set(gcf,'unit','centimeters','position',[60 30 11 8]);

% set(gca,'FontSize',12); %Ö»ÄÜÍ¬Ê±¸Ä±äx yÖáÏÔÊ¾µÄ×ÖÌå´óÐ¡¡£
% colorbar('eastoutside')
% title(['ch' num2str(i) '  time: ' time(1:2) ':' time(4:5)],'FontSize',15);
% xlabel('  km','FontSize',12)
% ylabel('  km','FontSize',12)

% saveas(gcf,[obs_nr 'pics_obs_withpert/obs_d04_ch' num2str(i) '_' time(1:2)  time(4:5) 'BT_withpert.jpg'])
%======================================================================================================
%dlmwrite([obs_nr 'obs_d04_ch' num2str(i) '_withpert.txt'],bt,'precision', '%.4f', 'delimiter', '\t')
dlmwrite([obs_nr 'obs_' domain '_ch' num2str(i) '_totalline_withpert.txt'],bt_1d,'precision', '%.4f', 'delimiter', '\t')
addpert=['add obsch_'  num2str(i) ' done']
end
