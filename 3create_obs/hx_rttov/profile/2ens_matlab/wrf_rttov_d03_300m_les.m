%20210628 1.5km obs 127point*127point 26:25:3176
%20210818 300m  obs 639*639point 6:5:3196
clear all;clc;
addpath /share/home/lililei1/lfzhou/scripts/matlab_utils/nctoolbox;

setup_nctoolbox();

% threshold for cloud top
qcloud_thresh = 0.0001; % kg/kg
cldfra_thresh=1.0e-6;
PI=3.1415926;
% read in wrf data
time='04_03_00'
%wrfdir = '/share/home/lililei1/lfzhou/hyperspectral_da/step1_obs_ensBT/step2_les_obs/0les_nr/';
wrfdir=getenv('wrf_data_dir');
scripts_dir='/share/home/lililei1/kcfu/hyperspectral_da/step1_obs_ensBT/step2_les_obs'
wrf_fname = ['wrfout_d04_0001-01-' time '_00'];
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

filename=[scripts_dir '/3profdata_d03/prof' time '.dat']
    fid=fopen(filename,'w');   
    fprintf(fid,'! Gas units (must be same for all profiles) \r\n');
           fprintf(fid,'! 0 => ppmv over dry air \r\n');
           fprintf(fid,'! 1 => kg/kg over moist air \r\n');
           fprintf(fid,'! 2 => ppmv over moist air \r\n');
           fprintf(fid,'%d \r\n',1);  % gas unit

  
  for  xloc=8:5:3198
  for  yloc=8:5:3198
 
           pres_prof = interp_prof(pres,xloc,yloc,'P');
           fprintf(fid,'! Pressure levels (hPa) \r\n');
           for kk=nz:-1:1
               fprintf(fid,'%.4f \r\n',pres_prof(kk)/100.0);
           end

           t_prof = interp_prof(t,xloc,yloc,'T');
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
           fprintf(fid,'%.1f \r\n',100000.0); %Wind fetch

           % salinity and FASTEM params are not considered, so patch default values
           tsk_point = interp_point(tsk,xloc,yloc,'TSK');
           fprintf(fid,'! Skin variables: \r\n');
           fprintf(fid,'! Skin T (K)  Salinity   FASTEM parameters for land surfaces \r\n')
           fprintf(fid,'%.4f   ',tsk_point);
           fprintf(fid,'%.1f   ',35.0); %Salinity
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
           [SatSeta,SatAlph]=satAngle(xx1,xx2);
          % [Sunseta,SunAlph]=sunAngle(xx1,xx2);
          % mDateVec = datenum([2015,08,03,0,0,0]);
          % UTC = datestr(mDateVec,'yyyy/mm/dd HH:MM:SS');
          % [sAz,Ze] = SolarAzEl(UTC,xx1,xx2,0);
           fprintf(fid,'%.4f   ',SatSeta*180/PI);%卫星天顶角
           fprintf(fid,'%.4f   ',SatAlph*180/PI);%卫星方位角
           fprintf(fid,'%.4f   ',45.0);%太阳天顶  无日变化，使用默认的
           fprintf(fid,'%.4f   ',30.0);%太阳方位
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
	if ( qice_prof(kcloud)+ qcloud_prof(kcloud) > cldfra_thresh )          
                   cldfra=1;   
              else
                   cldfra=0;                                      
	end                                              
                       
          fprintf(fid,'! Cloud top pressure (hPa) and cloud fraction for simple cloud scheme \r\n');
           fprintf(fid,'%.4f   ',pres_prof(kcloud)/100);
           fprintf(fid,'%.4f   ',cldfra);
           fprintf(fid,'\r\n');

    end
    end
    % work on MW






