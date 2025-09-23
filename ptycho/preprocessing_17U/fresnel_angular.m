function f1 = fresnel_angular(f0,M,N,dhx,dhy,z,lambda)
k=2*pi/lambda;
du=1./(M*dhx);
dv=1./(N*dhy);
u=ones(N,1)*[0:round(M/2)-1 -(M/2):-1]*du;                      %Note order of points for FFT
v=[0:round(N/2)-1 -N/2:-1]'*ones(1,M)*dv;
H=exp(i*2*pi*z*sqrt((1/lambda).^2-u.^2-v.^2));         %Fourier transform of kernel
f1=ifft2(fft2(f0).*H);                                 %Convolution
