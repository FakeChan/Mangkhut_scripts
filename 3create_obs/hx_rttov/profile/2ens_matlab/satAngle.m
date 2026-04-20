function [SatSeta,SatAlph]=satAngle(xlat,xlon)

satlon=str2num(getenv('satlon'))
satheight=str2num(getenv('satheight'))
earthRe=6371;
PI = 3.141592653589793;
heightradio=earthRe/(earthRe+satheight);
        cosdelta=cos(xlat*PI/180)*cos((satlon-xlon)*PI/180);
        sindelta=sqrt(1-cosdelta*cosdelta);
        constA=(cosdelta-heightradio)/sindelta;
        cosSeta=constA/sqrt(1+constA*constA);
        SatSeta=acos(cosSeta);
        sinAlph=sin((satlon-xlon)*PI/180)/sindelta;
        if (sinAlph>1) 
            sinAlph=1;
        end 
        if (sinAlph<-1) 
            sinAlph=-1;
        end
        SatAlph=asin(sinAlph);
        if(xlat>0) 
            SatAlph=PI-SatAlph;
        end 
end
        
        
      