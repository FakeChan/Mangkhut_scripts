%20201202 checkhx.m
%fkc.2026.8: ensemble H(x) evaluated on the d02 ensemble grid with true
%horizontal interpolation at the NR truth observation locations. The NR
%prof.dat (whose observation locations, order and count must be used) is
%parsed and verified before any output is written.
clear all;clc;
addpath /share/home/lililei1/kcfu/matlab_utils/nctoolbox-master;


setup_nctoolbox();

% threshold for cloud top
qcloud_thresh = 0.0001; % kg/kg
cldfra_thresh=1.0e-6;
PI=3.1415926;
% read in wrf data
mem=getenv('member')
domain=getenv('domain')
i_parent_start=str2num(getenv('i_parent_start'))
j_parent_start=str2num(getenv('j_parent_start'))
time_day=getenv('obs_day')
time_hour=getenv('obs_hour')
time_min=getenv('obs_min')
npoint=str2num(getenv('npoint'))
wrf_name_flag=getenv('profile_matlab_flag')
cycle_flag=getenv('cycle_flag')
wrfdir=getenv('ens_mem_dir');
% time='10_00:00';
time=strcat(time_day,'_',time_hour,':',time_min)
lacc_mode=getenv('lacc_mode');
if strcmp(lacc_mode,'1')
    center_day=getenv('lacc_center_day');
    center_hour=getenv('lacc_center_hour');
    center_min=getenv('lacc_center_min');
    % fkc.2026.8: LACC center fields must all be complete
    if isempty(center_day) || isempty(center_hour) || isempty(center_min)
        error('LACC mode: lacc_center_day/lacc_center_hour/lacc_center_min must all be set (got %s/%s/%s).',...
              center_day,center_hour,center_min);
    end
    % normalized center time in DD_HH_MM (underscore) form
    lacc_center_time=sprintf('%s_%s_%s',center_day,center_hour,center_min);
    lacc_center_time_env=getenv('lacc_center_time');
    if ~isempty(lacc_center_time_env) && ~strcmp(lacc_center_time_env,lacc_center_time)
        error('LACC mode: environment variable lacc_center_time ''%s'' inconsistent with the reconstructed center time ''%s''.',...
              lacc_center_time_env,lacc_center_time);
    end
else
    center_day=time_day;
    center_hour=time_hour;
    center_min=time_min;
    lacc_center_time=sprintf('%s_%s_%s',center_day,center_hour,center_min);
end
center_time=strcat(center_day,'_',center_hour,':',center_min)
% fkc.2026.8: default ensemble domain is d02 (1.5 km)
if strcmp(cycle_flag,'1')
    if strcmp(wrf_name_flag,'1')
        wrf_fname=['wrfout_d02_2018-09-' time ':00']
    else
        wrf_fname = strcat('firstguess_d02.',mem)
    end
else
    wrf_fname=['wrfout_d02_2018-09-' time ':00']
end


%wrfdir = '/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/0les_nr/';

%=============================================
%update in 2025.2.21
work_dir=getenv('prof_mem_dir');
%=============================================
%fkc.2026.8: the NR truth profile defines the exact observation locations at
%which the ensemble H(x) must be evaluated. Parse and verify it BEFORE any
%output is written. The NR profile lives in the parent directory of the
%member profile output directory, e.g.
%   profile_d01/prof<obs_day>_<obs_hour>:<obs_min>.dat      (ordinary)
%   profile_d01_LACC_<lacc_center_time>/prof<lag_time>.dat  (LACC)
%where <lag_time> is the current obs time, NOT the LACC center time.
member_out_dir=work_dir;
while ~isempty(member_out_dir) && (member_out_dir(end)=='/' || member_out_dir(end)=='\')
    member_out_dir(end)=[];
end
[prof_parent_dir,~]=fileparts(member_out_dir);
[~,prof_parent_name]=fileparts(prof_parent_dir);
nr_prof_fname=sprintf('prof%s_%s:%s.dat',time_day,time_hour,time_min);
nr_prof_path=fullfile(prof_parent_dir,nr_prof_fname);
if strcmp(lacc_mode,'1')
    expected_parent_name=sprintf('profile_%s_LACC_%s',domain,lacc_center_time);
    if ~strcmp(prof_parent_name,expected_parent_name)
        error('LACC mode: NR profile parent directory ''%s'' does not match expected ''%s'' (lacc_center_time=%s).',...
              prof_parent_name,expected_parent_name,lacc_center_time);
    end
end
if ~exist(nr_prof_path,'file')
    error(['NR truth profile not found: %s\n'...
           '  obs_time=%s, lacc_center_time=%s, member profile output dir=%s\n'...
           '  Please run the truth profile generation first (run_rttov_TrueObs_driver.sh / run_rttov_TrueObs_LACC_driver.sh).'],...
          nr_prof_path,time,lacc_center_time,work_dir);
end
[obs_lat,obs_lon]=read_obs_locations_from_profile(nr_prof_path,npoint);
if strcmp(lacc_mode,'1')
    lacc_times_str=getenv('lacc_times');
    if ~isempty(lacc_times_str)
        current_lacc_time=sprintf('%s_%s_%s',time_day,time_hour,time_min);
        lacc_time_list=strsplit(strtrim(lacc_times_str));
        if ~ismember(current_lacc_time,lacc_time_list)
            error('LACC mode: current obs time %s is not in lacc_times (%s).',...
                  current_lacc_time,strtrim(lacc_times_str));
        end
    end
end
fprintf('LACC center time: %s\n',lacc_center_time);
fprintf('Current lag/observation time: %s\n',time);
fprintf('NR truth profile: %s\n',nr_prof_path);
fprintf('Ensemble profile output: %s\n',fullfile(work_dir,['prof' time '.dat']));
%=============================================
% wrfdir='/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/';



string=sprintf('wrfd01=ncgeodataset([''%s'',''%s'']);',wrfdir,wrf_fname);eval(string)
xlat=squeeze(wrfd01.data('XLAT'));xlon=squeeze(wrfd01.data('XLONG'));
%fkc.2026.8: map the NR truth observation locations onto the d02 ensemble
%grid as continuous floating-point coordinates for horizontal interpolation
[xloc_all,yloc_all]=latlon_to_fractional_ij(xlat,xlon,obs_lat,obs_lon);
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

psfc=squeeze(wrfd01.data('PSFC'));
hgt=squeeze(wrfd01.data('HGT')); %terrain height m
landmask=squeeze(wrfd01.data('LANDMASK'));
cldfra=squeeze(wrfd01.data('CLDFRA'));

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

filename=[work_dir 'prof' time '.dat']
    fid=fopen(filename,'w');
    fprintf(fid,'! Gas units (must be same for all profiles) \r\n');
           fprintf(fid,'! 0 => ppmv over dry air \r\n');
           fprintf(fid,'! 1 => kg/kg over moist air \r\n');
           fprintf(fid,'! 2 => ppmv over moist air \r\n');
           fprintf(fid,'%d \r\n',1);  % gas unit


%   for  xloc=38:125:3163
%   for  yloc=38:125:3163

for obs_index=1:npoint
            xloc=xloc_all(obs_index);
            yloc=yloc_all(obs_index);
            if xloc<0 || yloc<0
                error('xloc or yloc invalid')
            end
           pres_prof = interp_prof(pres,xloc,yloc,'P');
           psfc_point = interp_point(psfc,xloc,yloc,'PSFC');
           t2_point = interp_point(t2,xloc,yloc,'T2');
           q2_point = interp_point(q2,xloc,yloc,'Q2');
           u10_point = interp_point(u10,xloc,yloc,'U10');
           v10_point = interp_point(v10,xloc,yloc,'V10');
           fprintf(fid,'! Pressure levels (hPa) \r\n');
           for kk=nz:-1:1
               fprintf(fid,'%.4f \r\n',pres_prof(kk)/100.0);
           end

           t_prof = interp_prof(t,xloc,yloc,'T');
           tsk_point = interp_point(tsk,xloc,yloc,'TSK');
           for kk=1:nz
               tk_prof(kk) = wrf_tk(t_prof(kk),pres_prof(kk));
           end
           fprintf(fid,'! Temperature profile (K) \r\n');
           for kk=nz:-1:1
               fprintf(fid,'%.4f \r\n',tk_prof(kk));
           end

           qvapor_prof = interp_prof(qvapor,xloc,yloc,'QVAPOR');
           fprintf(fid,'! Water vapor profile (ppmv) \r\n');
           for kk=nz:-1:1
		        if (qvapor_prof(kk) < 0.000000001) ;
                    qvapor_prof(kk)  = 0.000000001;
                end
	   fprintf(fid,'%.9f \r\n',qvapor_prof(kk));
           end

           % ozone affected channels are not assimilated
           % so patch some "climate" values here
%           fprintf(fid,'! Ozone (ppmv) - currently not read in \r\n');
%           for kk=nz:-1:1
%               fprintf(fid,'%.4f \r\n',ozone_prof(kk));
%           end
        %    psfc_point = interp_point(psfc,xloc,yloc,'PSFC');
        %    t2_point = interp_point(t2,xloc,yloc,'T2');
        %    q2_point = interp_point(q2,xloc,yloc,'Q2');
        %    u10_point = interp_point(u10,xloc,yloc,'U10');
        %    v10_point = interp_point(v10,xloc,yloc,'V10');
           fprintf(fid,'! Near-surface variables: \r\n');
           fprintf(fid,'!  2m T (K)    2m q (kg/kg) 2m p (hPa) 10m wind u (m/s)  10m wind v (m/s)  wind fetch (m) \r\n');
           fprintf(fid,'%.4f   ',t2_point);
           fprintf(fid,'%.4f   ',q2_point);
           fprintf(fid,'%.4f   ',psfc_point/100.0);
           fprintf(fid,'%.4f   ',u10_point);
           fprintf(fid,'%.4f   ',v10_point);
           % wind fetch default value 100000.0 is used
           fprintf(fid,'%.1f \r\n',100000.0); %Wind fetch

           % salinity and FASTEM params are not considered, so patch default values
           tsk_point = interp_point(tsk,xloc,yloc,'TSK');
           fprintf(fid,'! Skin variables: \r\n');
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
%fkc.2026.8: write the NR truth observation location, not a d02 grid point
           xlat_point = obs_lat(obs_index);
           xlon_point = obs_lon(obs_index);
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
           [SatSeta,SatAlph]=satAngle(xx1,xx2);
          % [Sunseta,SunAlph]=sunAngle(xx1,xx2);
          % mDateVec = datenum([2015,08,03,0,0,0]);
          % UTC = datestr(mDateVec,'yyyy/mm/dd HH:MM:SS');
          % [sAz,Ze] = SolarAzEl(UTC,xx1,xx2,0);
           fprintf(fid,'%.4f   ',SatSeta*180/PI);%卫星天顶角
           fprintf(fid,'%.4f   ',SatAlph*180/PI);%卫星方位角
           % fprintf(fid,'%.4f   ',45.0);%太阳天顶  无日变化，使用默认的
           % fprintf(fid,'%.4f   ',30.0);%太阳方位
           fprintf(fid,'%.4f   ',70.5);%太阳天顶
           fprintf(fid,'%.4f   ',114.0);%太阳方位
           fprintf(fid,'\r\n');

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
           % fprintf(fid,'! cloud liquid water (kg/kg) \r\n');
           % for kk=nz:-1:1
           %     fprintf(fid,'%.4f \r\n',qcloud_prof(kk));
           % end
           % fprintf(fid,'\r\n');

    end
    % work on MW
