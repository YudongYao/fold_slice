path_name = 'G:\SSRF17U\IC_2_large\';
fnames = dir([path_name 'IC*.tif']);
fsorted = {fnames.name};
 fsorted = natsort({fnames.name});
for ii =1:1681
    
    dp(:,:,ii) = importdata([path_name fsorted{ii}]);
end
% dp(dp>2000)=0;

      save('dp_0','dp','-v7.3');

