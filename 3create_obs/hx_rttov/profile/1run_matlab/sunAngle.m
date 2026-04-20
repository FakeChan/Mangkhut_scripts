function [SunSeta,SunAlph]=sunAngle(xlat,xlon)

satlon=104.7;   %星下点经度，
satheight=35793;
earthRe=6371;
PI = 3.141592653589793;
%wrf_fname = 'wrfout_d02_2015-08-03_15_00_00';12:15 17:18 20:21

minute=00;dayofyear=1; %为了出值，自己假定了一下
       gmt = 15 + 00 / 60.0;            %为了出值，自己假定了一下
       tet=2.0 * PI* ( 100 -1) / 365.0; %为了出值，自己假定了一下    
       
%实际需要根据主程序中
%wrf_fname = 'wrfout_d02_2015-08-03_15_00_00';
%2015-08-03_15_00_00设置小时、分钟以及这一天在是这一年的第几天

       %gmt = float(hour) + float(minute) / 60.0%世界时 这两句是实际要使用的
       %tet=2.0 * PI* ( float(dayofyear) -1) / 365.0;

       
%function [Sunseta,SunAlph]=sunAngle(gmt,tet,xlat,xlon)
  a1=0.000075;a2=0.001868;a3=0.032077;a4=0.014615;a5=0.04089;
  b1=0.006918;b2=0.399912;b3=0.070257;b4=0.006758;b5=0.000907;b6=0.002697;b7=0.001480;
  PI = 3.141592653589793;
        %time equation
        et=(a1+a2*cos(tet)-a3*sin(tet)-a4*cos(2*tet)-a5*sin(2*tet)) * 180/ PI /15;
        
        %true solar time
        tst=gmt+(xlon/15)+et;
        
        %hour angle
        ah=(tst - 12)*15.0 * PI / 180;

        %solar declination(in radian)
        delta=b1-b2*cos(tet)+b3*sin(tet)-b4*cos(2*tet)+b5*sin(2*tet)-b6*cos(3*tet)+b7*sin(3*tet);

        %elevation,azimuth
        xlatAngle = xlat * PI /180;
        cosSeta=sin(xlatAngle)*sin(delta) +cos(xlatAngle)*cos(delta)*cos(ah);
        SunSeta=acos(cosSeta);

        sinSeta=sin(SunSeta);
        cosAlph=(sin(delta)-sin(xlatAngle)*cosSeta)/cos(xlatAngle)/sinSeta;
        if (cosAlph>1) 
            cosAlph=1;
        end
        if (cosAlph<-1)
            cosAlph=-1;
        end
        SunAlph=acos(cosAlph);
        if(ah>pi) 
           ah=ah-2* PI;
        end
        if(ah<-pi) 
            ah=ah+2*PI;
        end
        if( ah>0 ) 
           SunAlph=2* PI -SunAlph;
        end
        
end


        