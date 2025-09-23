%下面为波带片焦点处光斑入射
%N_number=D_zp*10^3/delta_r/4;
%N_number=1666;%波带片环数
%dalta_r %波带片最外环宽度
%NA=lamda/(2*dalta_r);
D_zp=-85;%um
delta_r=50;%nm
E_xray=12600;%%%%%%%%%%%%%%%
r_probe0=800; 
r_probe=150;
ap_rand=2;
h=6.626*10^(-34); %普朗克常数
c=3*10^8;
e=1.6*10^(-19);
lamda=10^9*h*c/(E_xray*e); %unit is nm
Xflux=1e8;
D_ccd=200;
D_sd=2400;%mm
t_frsnl_pi=1;
D_zp_in=35;
%D_fs=78;
r_ccd=floor(D_ccd/2);
ccdpixel=75;%um


theta=atan(r_ccd*ccdpixel/D_sd/1000);
objpixel=lamda/tan(theta)/2;


f=10^3*D_zp*delta_r*E_xray/1240.0; %ZP focus length,nm
k=2*pi/lamda;R30=delta_r*f/(D_zp*10^3);
rzp=D_zp/2;      %um
%theta=atan(rzp*10^3/f);  %sinapssrf
%zprob_0=r_probe*objpixel/tan(theta);    %sinapssrf  %unit is nm.
% D_fs=D_fs*10^3;   % unit is changed from um to nm.
S_zp=pi*rzp^2;
A=R30*sqrt(Xflux/S_zp);
%E_illum=zeros(pi_d,pi_d);

p_supp=zeros(D_ccd,D_ccd);
p_supp0=zeros(D_ccd,D_ccd);   %sinapssrf
p_supps=zeros(D_ccd,D_ccd);   %sinapssrf
for ss=1:D_ccd;
    x=ss-r_ccd-1;
    for tt=1:D_ccd;
        y=tt-r_ccd-1;
        r=sqrt(x*x+y*y);
        if(r<=r_probe+ceil(ap_rand))
           p_supp(ss,tt)=1;
        end
        if(r<=r_probe)        %sinapssrf
           p_supp0(ss,tt)=1;  %sinapssrf
        end                   %sinapssrf
        if(r<=r_probe-ceil(ap_rand)-1)
           p_supps(ss,tt)=1;
        end
    end
end

E_illum=zeros(D_ccd,D_ccd);
if(t_frsnl_pi)
    E0=sqrt(Xflux/S_zp);%入射光振幅
    rzp_in=D_zp_in/2;
    rzp=rzp*10^3;  %nm
    rzp_in=rzp_in*10^3;  %nm
    E_cen=-1i*E0/lamda/f*exp(1i*k*f)*(rzp^2-rzp_in^2);
    pre_cons=(-2*1i*E0)/lamda/f*exp(1i*k*f);
    krzp=k*rzp;
    krzpin=k*rzp_in;
    rzp2=rzp^2;
    rzpin2=rzp_in^2;
    kf2=k*f/2;

    
    for ss=1:D_ccd;
        x=ss-r_ccd-1;
        for tt=1:D_ccd;
            y=tt-r_ccd-1;
            r=sqrt(x*x+y*y);
            if r>r_probe0
                E_illum(ss,tt)=0;
            elseif r==0
                E_illum(ss,tt)=E_cen;  % SINAPSSRF
            else
                theta=r*objpixel/f;
                x1=krzp*theta;
                x2=krzpin*theta;
                %E_illum(ss,tt)=(-2*i*rzp*rzp*E0)/(lamda*f)*exp(i*k*f)*exp(i*k*theta*theta*f/2)*besselj(1,k*rzp*10^3*theta)/(k*rzp*10^3*theta);
                if(rzp_in==0)
                    E_illum(ss,tt)=pre_cons*exp(1i*kf2*theta*theta)*rzp2*besselj(1,x1)/x1;
                else
                    E_illum(ss,tt)=pre_cons*exp(1i*kf2*theta*theta)*(rzp2*besselj(1,x1)/x1-rzpin2*besselj(1,x2)/x2);
                end
            end
        end
    end
    %ccd(k1:k2,k1:k2)=E_illum;
    ccd=E_illum;

end

cm=0.01;um=1e-6;
M=200;N=200;                                           %全息图分辨率
%z=0.0003;                                                  %菲涅尔传输距离
lambda=lamda*0.001*um; 
k=2*pi/lambda;                                           %波数
hx=200*objpixel*0.001*um;hy=200*objpixel*0.001*um;                                     %原始图大小
dhx=hx/M;dhy=hy/N; 
probe_t= fresnel_angular(ccd ,M,N,dhx,dhy,-0.0005,lambda);
probe=probe_t/3600;
save('probe','probe','-v7.3');

%%%%%%%%%%%%%%scanning positions%%%%%%%%%%%%%%%%%%%
% [data1,data2,data3]=textread('2022-09-29-12-26-21-posi.txt','%n%n%n');
% data2=data2-mean(data2);
% data3=data3-mean(data3);
% offset_probe_a=round(data3*1000/objpixel);
% offset_probe_b=round(data2*1000/objpixel);
% save('offset_probe','offset_probe_a','offset_probe_b','-v7.3');
data1=0:0.2:0.2*40;
data2=0:0.2:0.2*40;
data1=data1-mean(data1);
data2=data2-mean(data2);
for ii=1:1681
    offset_probe_a(ii)=data1(floor((ii-1)/41)+1);
    offset_probe_b(ii)=data2(mod(ii,41)+1);
end
offset_probe_b=circshift(offset_probe_b,[0 1]);
% for ii=1:2:9
%     offset_probe_b(1+ii*10:10+ii*10)=-offset_probe_b(1+ii*10:10+ii*10);
% end 

offset_probe_a=round(offset_probe_a*1000/objpixel);
offset_probe_b=round(offset_probe_b*1000/objpixel);
 save('offset_probe','offset_probe_a','offset_probe_b','-v7.3');