%20201202 checkhx.m
clear all;clc;
addpath /share/home/lililei1/kcfu/matlab_utils/nctoolbox-master;


setup_nctoolbox();

% threshold for cloud top
qcloud_thresh = 0.0001; % kg/kg
cldfra_thresh=1.0e-6;
PI=3.1415926;
% read in wrf data
time_day=getenv('obs_day')
time_hour=getenv('obs_hour')
time_min=getenv('obs_min')
npoint=str2num(getenv('npoint'))
rttov_scatt=getenv('rttov_scatt')
use_total_ice=getenv('use_total_ice')
clear_sky_mode=strcmp(rttov_scatt,'0');
% time='10_00:00';
time=strcat(time_day,'_',time_hour,':',time_min)
lacc_mode=getenv('lacc_mode');
if strcmp(lacc_mode,'1')
    center_day=getenv('lacc_center_day');
    center_hour=getenv('lacc_center_hour');
    center_min=getenv('lacc_center_min');
else
    center_day=time_day;
    center_hour=time_hour;
    center_min=time_min;
end
if isempty(center_day); center_day=time_day; end
if isempty(center_hour); center_hour=time_hour; end
if isempty(center_min); center_min=time_min; end
center_time=strcat(center_day,'_',center_hour,':',center_min)
wrfdir=getenv('NR_wrfout_dir');
if strcmp(lacc_mode,'1')
    wrf_domain='d02';
    delta_x=1500;
else
    wrf_domain='d03';
    delta_x=300;
end
wrf_fname = ['wrfout_' wrf_domain '_2018-09-' time ':00'];
center_wrf_fname = ['wrfout_' wrf_domain '_2018-09-' center_time ':00'];
radius=(sqrt(npoint)/2-1); % a square of (2*radius+2)^2
delta=7500/delta_x;

%wrfdir = '/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/0les_nr/';

%=============================================
%update in 2025.2.21
% NR_location='/share/home/lililei1/kcfu/tc_mangkhut/0necessay_files'
% ilist=readmatrix(strcat(NR_location,'/','NR_d02_ilist.txt'));
% jlist=readmatrix(strcat(NR_location,'/','NR_d02_jlist.txt'));
% obsDay=str2num(time_day);
% obsHour=str2num(time_hour);

% startDay=09;
% startHour=06;

% tloc=((obsDay-startDay)*24+obsHour-startHour)/3+1;%calc the order of obs
% iloc=ilist(tloc)+1;
% jloc=jlist(tloc)+1;

wrffile=strcat(wrfdir,center_wrf_fname)
[center_lat, center_lon, min_mslp_val, iloc, jloc] = find_typhoon_center(wrffile, 1)

%=============================================
% wrfdir='/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/';

work_dir=getenv('prof_dir');

string=sprintf('wrfd01=ncgeodataset([''%s'',''%s'']);',wrfdir,wrf_fname);eval(string)
xlat=squeeze(wrfd01.data('XLAT'));xlon=squeeze(wrfd01.data('XLONG'));
t2=squeeze(wrfd01.data('T2'));
q2=squeeze(wrfd01.data('Q2')); %kg/kg
u10=squeeze(wrfd01.data('U10'));
v10=squeeze(wrfd01.data('V10'));
tsk=squeeze(wrfd01.data('TSK')); %surface skin temperature K
pbase=squeeze(wrfd01.data('PB'));pb=squeeze(wrfd01.data('P'));pres=pbase+pb; %pressure Pa
t=squeeze(wrfd01.data('T')); [nz ny nx]=size(t); t=t+repmat(300,nz,ny,nx); %potential temperature K
qvapor=squeeze(wrfd01.data('QVAPOR')); %water vapor mixing ratio kg/kg
qcloud=squeeze(wrfd01.data('QCLOUD')); %cloud water mixing ratio kg/kg
qice=squeeze(wrfd01.data('QICE'));
qgraup=squeeze(wrfd01.data('QGRAUP'));

psfc=squeeze(wrfd01.data('PSFC'));
hgt=squeeze(wrfd01.data('HGT')); %terrain height m
landmask=squeeze(wrfd01.data('LANDMASK'));
cldfra=squeeze(wrfd01.data('CLDFRA'));

qsnow = squeeze(wrfd01.data('QSNOW'));  % 用于计算 totalice
qrain = squeeze(wrfd01.data('QRAIN'));  % 用于 rain
znw   = squeeze(wrfd01.data('ZNW'));    % 垂直坐标 (eta levels)
mu    = squeeze(wrfd01.data('MU'));     % 扰动干空气质量
mub   = squeeze(wrfd01.data('MUB'));    % 基准干空气质量
% 获取 P_TOP (通常在 global attributes 中)
try
    p_top = wrfd01.getAttribute('P_TOP'); 
catch
    p_top = 5000; % 如果读取失败，根据你的 namelist 设置默认值 (例如 50hPa -> 5000Pa)
    disp('Warning: P_TOP not found, using default 5000 Pa');
end
if clear_sky_mode
    geopot=squeeze(wrfd01.data('PH')) + squeeze(wrfd01.data('PHB'));
    clear_sky_mask=zeros(npoint,1);
    hydrometeor_path=zeros(npoint,1);
    clear_sky_thresh=0.01; % kg m^-2 for RWP + SWP + GWP
end


%%set up ozone profile
% ozcons=[0.0555 0.0537 0.0512 0.0491 0.0471 0.0455 0.0429 0.0400 0.0372 0.0343 0.0315 0.0291 0.0269 0.0255];
% nozcons=length(ozcons);
% ozone_prof = zeros(nz,1);
% for kk=1:nozcons
%     ozone_prof(kk) = ozcons(nozcons-kk+1);
% end
% for kk=nozcons+1:nz
%     ozone_prof(kk) = ozone(nozcons); ozone(nozcons) not define!
% end

filename=[work_dir '/prof' time '.dat']
    fid=fopen(filename,'w');   
    fprintf(fid,'! Gas units (must be same for all profiles) \r\n');
           fprintf(fid,'! 0 => ppmv over dry air \r\n');
           fprintf(fid,'! 1 => kg/kg over moist air \r\n');
           fprintf(fid,'! 2 => ppmv over moist air \r\n');
           fprintf(fid,'%d \r\n',1);  % gas unit is kg/kg

  
%   for  xloc=38:125:3163
%   for  yloc=38:125:3163

obs_index=0;
for yloc=jloc-radius*delta:delta:jloc+(radius+1)*delta
    for xloc=iloc-radius*delta:delta:iloc+(radius+1)*delta
            obs_index=obs_index+1;
            if xloc<0 || yloc<0
                error('xloc or yloc invalid')
            end
            % === [Start] Modified Section: Write Loop matching RTTOV-SCATT Fortran ===

            % 1. 插值/准备单点廓线数据
            pres_prof   = interp_prof(pres,xloc,yloc,'P');      % Full level pressure (Pa)
            t_prof_raw  = interp_prof(t,xloc,yloc,'T');         % Potential Temp
            qvapor_prof = interp_prof(qvapor,xloc,yloc,'QVAPOR');
            cldfra_prof = interp_prof(cldfra,xloc,yloc,'CLDFRA');
            qcloud_prof = interp_prof(qcloud,xloc,yloc,'QCLOUD');
            qice_prof   = interp_prof(qice,xloc,yloc,'QICE');
            qsnow_prof  = interp_prof(qsnow,xloc,yloc,'QSNOW');
            qrain_prof  = interp_prof(qrain,xloc,yloc,'QRAIN');
            qgraup_prof = interp_prof(qgraup,xloc,yloc,'QGRAUP');
            if clear_sky_mode
                geopot_prof = interp_prof(geopot,xloc,yloc,'PH_PLUS_PHB');
                height_prof = geopot_prof / 9.8;
                dz_prof = abs(diff(height_prof));
            end
            % 2. 计算界面气压 (PH / Half-level Pressure)
            % MU 是 2D (y,x)，ZNW 是 1D (nz+1)
            mu_point  = interp_point(mu, xloc, yloc, 'MU');
            mub_point = interp_point(mub, xloc, yloc, 'MUB');
            % 公式: Ph = (MU + MUB) * ZNW + P_TOP
            % 注意: ZNW 长度为 nz+1 (31层)，从 1.0(地) 到 0.0(顶)
            ph_prof_pa = (mu_point + mub_point) .* znw + p_top; 
            
            % 3. 写入 Header (RTTOV 需要先读取 header 信息)
            % 注意：这里只写一次 Profile 数据的 Header 说明，或者根据你的 file format 不需要
            fprintf(fid, '! Cloud Profile Data: P(hPa) PH(hPa) T(K) Q(kg/kg) CC(0-1) CLW(kg/kg) TotIce(kg/kg) Rain(kg/kg) \r\n');
            
            % 4. 循环写入 (Top-Down: nz -> 1)
            % 严格对应 Fortran: READ(iup,*) p, ph, t, q, cc, clw, totalice, rain
            for kk = nz:-1:1
                
                % --- 准备变量 ---
                
                % (1) P: Full Level Pressure (hPa)
                p_hpa = pres_prof(kk) / 100.0;
                
                % (2) PH: Half Level Pressure (hPa)
                % WRF ZNW索引: 1是地面, nz+1是顶。
                % 当 kk=nz (顶层) 时，对应 ZNW(nz) 和 ZNW(nz+1)。
                % RTTOV Top-Down 循环通常期望 ph(ilev) 是该层的"下边界"(higher pressure side) 或一一对应。
                % 这里我们取 ZNW(kk) 作为该层的界面气压 (对应 WRF 的 Bottom interface of layer k)
                ph_hpa = ph_prof_pa(kk) / 100.0;
                
                % (3) T: Temperature (K)
                tk_val = wrf_tk(t_prof_raw(kk), pres_prof(kk));
                
                % (4) Q: Specific Humidity (kg/kg) (防止负值)
                q_val = max(qvapor_prof(kk), 1.0e-9);
                
                % (5) CC: Cloud Cover (0-1)
                cc_val = max(0, min(1, cldfra_prof(kk)));
                
                % (6) CLW: Liquid Water (kg/kg)
                clw_val = max(0, qcloud_prof(kk));
                
                
                
                % (8) Rain: Rain (kg/kg)
                rain_val = max(0, qrain_prof(kk));
                
                % --- 写入一行 ---
                %判断是否使用total ice
                if strcmp(use_total_ice,'1')
                    tot_ice_val = max(0, qice_prof(kk) + qsnow_prof(kk)+qgraup_prof(kk));
                    fprintf(fid, '%.4f %.4f %.4f %.4e %.4f %.4e %.4e %.4e \r\n', ...
                        p_hpa, ph_hpa, tk_val, q_val, cc_val, clw_val, tot_ice_val, rain_val);
                else
                    qice_val= max(0,qice_prof(kk))
                    FrozenPrecip_val=max(0, qsnow_prof(kk)+qgraup_prof(kk))
                    fprintf(fid, '%.4f %.4f %.4f %.4e %.4f %.4e %.4e %.4e %.4e \r\n', ...
                        p_hpa, ph_hpa, tk_val, q_val, cc_val, clw_val, qice_val, rain_val, FrozenPrecip_val);
                end
            end
            if clear_sky_mode
                tk_prof_for_mask = zeros(1,nz);
                for kk=1:nz
                    tk_prof_for_mask(kk) = wrf_tk(t_prof_raw(kk), pres_prof(kk));
                end
                hydro_mix = max(0,qrain_prof) + max(0,qsnow_prof) + max(0,qgraup_prof);
                rho_prof = pres_prof ./ (287.05 .* tk_prof_for_mask .* (1.0 + 0.61 .* qvapor_prof));
                hydrometeor_path(obs_index) = sum(rho_prof(:) .* hydro_mix(:) .* dz_prof(:));
                clear_sky_mask(obs_index) = hydrometeor_path(obs_index) < clear_sky_thresh;
            end
            
            % === [End] Modified Section ===
           %----------------------------------------------
           % ozone affected channels are not assimilated
           % so patch some "climate" values here
%           fprintf(fid,'! Ozone (ppmv) - currently not read in \r\n');
%           for kk=nz:-1:1
%               fprintf(fid,'%.4f \r\n',ozone_prof(kk));
%           end
           psfc_point = interp_point(psfc,xloc,yloc,'PSFC');
           t2_point = interp_point(t2,xloc,yloc,'T2');
           q2_point = interp_point(q2,xloc,yloc,'Q2');
           u10_point = interp_point(u10,xloc,yloc,'U10');
           v10_point = interp_point(v10,xloc,yloc,'V10'); 
           fprintf(fid,'! Near-surface variables: \r\n');
           fprintf(fid,'!  2m T (K)    2m q (kg/kg) 2m p (hPa) 10m wind u (m/s)  10m wind v (m/s)  wind fetch (m) \r\n');
           fprintf(fid,'%.4f   ',t2_point);
           fprintf(fid,'%.4f   ',q2_point);
           fprintf(fid,'%.4f   ',psfc_point/100.0);
           fprintf(fid,'%.4f   ',u10_point);
           fprintf(fid,'%.4f   ',v10_point);
           % wind fetch default value 100000.0 is used
           if strcmp(rttov_scatt,'0')
            fprintf(fid,'%.1f \r\n',100000.0); %Wind fetch
           end
           % salinity and FASTEM params are not considered, so patch default values
           tsk_point = interp_point(tsk,xloc,yloc,'TSK');
           fprintf(fid,'\r\n ! Skin variables: \r\n');
           fprintf(fid,'! Skin T (K)  Salinity   FASTEM parameters for land surfaces \r\n')
           fprintf(fid,'%.4f   ',tsk_point);
           fprintf(fid,'%.1f   ',34.4); %Salinity
           fprintf(fid,'%.1f   ',[3.0 5.0 15.0 0.1 0.3]); %FASTEM parameters for land surfaces
           fprintf(fid,'\r\n');

           % surface type use the nearest grid point landmask
           % water type choose ocean if '1' in wrf landmask
           fprintf(fid,'! Surface type (0=land, 1=sea, 2=sea-ice) and water type (0=fresh, 1=ocean) \r\n');
           %surf_type = landmask(round(yloc),round(xloc));
           %fprintf(fid,'%d   ', surf_type);
          % if ( surf_type == 1 ) 
              fprintf(fid,'%d   ',[1 1]);
          % end
           fprintf(fid,'\r\n');

           hgt_point = interp_point(hgt,xloc,yloc,'HGT');
%modified: need to interp_point 20201120
           xlat_point = interp_point(xlat,xloc,yloc,'XLAT');
           xlon_point = interp_point(xlon,xloc,yloc,'XLONG');
           fprintf(fid,'! Elevation (km), latitude and longitude (degrees) \r\n');
           fprintf(fid,'%.4f   ',hgt_point/1000.0);
           fprintf(fid,'%.4f   ',xlat_point);
           fprintf(fid,'%.4f   ',xlon_point);
           fprintf(fid,'\r\n');
	   
	       xx1 = double(xlat_point);
           %class(double(xx1))
           xx2 = double(xlon_point);
           fprintf(fid,'! Sat. zenith and azimuth angles, solar zenith and azimuth angles (degrees) \r\n');
          %需要在这里调用子程序 satAngle 与 sunAngle
           %[SatSeta,SatAlph]=satAngle(xx1,xx2);
           %假定卫星始终在台风中心正上方
           [SatSeta,SatAlph]=calc_OSSE_satAngles(xx1,xx2,center_lat,center_lon)
          % [Sunseta,SunAlph]=sunAngle(xx1,xx2);
          % mDateVec = datenum([2015,08,03,0,0,0]);
          % UTC = datestr(mDateVec,'yyyy/mm/dd HH:MM:SS');
          % [sAz,Ze] = SolarAzEl(UTC,xx1,xx2,0);
           fprintf(fid,'%.4f   ',SatSeta*180/PI);%卫星天顶角
           fprintf(fid,'%.4f   ',SatAlph*180/PI);%卫星方位角
           if strcmp(rttov_scatt,'0')
            fprintf(fid,'%.4f   ',45.0);%太阳天顶  无日变化，使用默认的
            fprintf(fid,'%.4f   ',30.0);%太阳方位
           end
           fprintf(fid,'\r\n');
           if strcmp(rttov_scatt,'0')
                qcloud_prof = interp_prof(qcloud,xloc,yloc,'QCLOUD');
                qice_prof   = interp_prof(qice,xloc,yloc,'QICE');
                qice_prof=qice_prof';qcloud_prof =qcloud_prof';
                
                    kcloud = 1;                                          
                for kk=nz:-1:1                                           
                    if ( qcloud_prof(kk) > qcloud_thresh );          
                            kcloud = kk        ;                         
                            break                                       
                    end                                              
                end                             
                    if ( qice_prof(kcloud)+ qcloud_prof(kcloud) > cldfra_thresh ) ;         
                                cldfra=1;   
                    else
                                cldfra=0;                                      
                    end                                              
                            
                fprintf(fid,'! Cloud top pressure (hPa) and cloud fraction for simple cloud scheme \r\n');
                fprintf(fid,'%.4f   ',pres_prof(kcloud)/100)
                fprintf(fid,'%.4f   ',cldfra)
                fprintf(fid,'\r\n');
            end
        %    % fprintf(fid,'! cloud liquid water (kg/kg) \r\n');
        %    % for kk=nz:-1:1
        %    %     fprintf(fid,'%.4f \r\n',qcloud_prof(kk));
        %    % end
        %    % fprintf(fid,'\r\n');

    end
end
if clear_sky_mode
    clear_sky_mask_file=[work_dir '/clear_sky_mask_' time '.txt'];
    hydrometeor_path_file=[work_dir '/hydrometeor_path_' time '.txt'];
    dlmwrite(clear_sky_mask_file, clear_sky_mask, 'precision', '%d', 'delimiter', '\t');
    dlmwrite(hydrometeor_path_file, hydrometeor_path, 'precision', '%.8f', 'delimiter', '\t');
    fprintf('Clear-sky mask written to %s; clear obs = %d / %d\n', ...
        clear_sky_mask_file, sum(clear_sky_mask), length(clear_sky_mask));
end
    % work on MW
