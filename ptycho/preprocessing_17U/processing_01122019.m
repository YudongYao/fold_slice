
clear all
path_name = '/mnt/data2/ssrf17u/test/Target_4_small_0_2s/';
fnames = dir([path_name 'dp_*.mat']);
fsorted = {fnames.name};
for dd=1
%     importdata([path_name fsorted{dd}]);
%     load([path_name fsorted{dd}]);
    pattern=zeros(200,200,1681);
    for ii=1:1681
        temp=double(dp(:,:,ii));
        temp_cut=temp(714-99:714+100,437-99:437+100);
        pattern(:,:,ii)=fliplr(temp_cut);
    end
    pattern(pattern>500000)=0;
%     save(['pattern_',fsorted{dd}],'pattern','-v7.3');
    save('pattern','pattern','-v7.3');
end


% load('4T1_8_area1_710eV_3u_600n_200ms_18x22_2.mat')
% for ii=271:396
%     temp=ccdfs(:,:,ii-270);
%     temp_cut=temp(478-379:478+380,506-379:506+380);
%     pattern(:,:,ii)=temp_cut;
% end
%
% backg_sep= (sum(pattern(:,:,1:396),3)/396);
%
%
% for ii=1:396
%
%
%     pattern_a(:,:,ii)=pattern(:,:,ii)-backg_sep;
%
%
% end
% pattern_b=pattern-1400;
%
% a_ring=zeros(760);
% for mm=1:760
%     for nn=1:760
%         if sqrt((mm-380).^2+(nn-380).^2)<(480-380)
%             a_ring(mm,nn)=1;
%         end
%
%     end
% end
% for ii=1:396
%     temp_a= pattern_a(:,:,ii);
%     temp_b= pattern_b(:,:,ii);
%     temp_a(a_ring==1)=temp_b(a_ring==1);
%     pattern(:,:,ii)=temp_a;
%
%
% end
% a_ring_1=zeros(760);
% for mm=1:760
%     for nn=1:760
%         if sqrt((mm-380).^2+(nn-380).^2)<(404-380)
%             a_ring_1(mm,nn)=1;
%         end
%
%     end
% end
%
% for ii=1:396
%     temp= pattern(:,:,ii);
%
%     temp(a_ring_1==1)=3000;
%     pattern(:,:,ii)=temp;
%
%
% end
% pattern(pattern<0)=0;
%
% for ii=1:396
%     temp=pattern(:,:,ii);
%     temp=sqrt(temp);
%     temp=flipud(temp);
%     pattern(:,:,ii)=temp;
% end
% save('pattern_1','pattern','-v7.3');