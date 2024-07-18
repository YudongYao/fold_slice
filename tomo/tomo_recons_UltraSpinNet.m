close all 
clear all
%% Add paths
addpath('utils')
addpath('tests')
% addpath(find_base_package())
addpath('../')

import plotting.*
import io.*
import math.*

%% File management
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

par.num_proj = 901;
par.scanstomo = 1:par.num_proj;

par.tomo_id = []; % Either scan numbers or tomo_id can be given, but not both, if not provided leave tomo_id=[]

% verbosity and online tomo 
par.verbose_level = 1; 
par.online_tomo = true;   % automatically run if called from externally

par.base_path = 'C:\Users\yudongyao\Work\Data\UltraSpinNet\1500Epochs\';
filename = 'projection_T_uncompressed.hdf5';
par.output_folder = 'C:\Users\yudongyao\Work\Data\UltraSpinNet\T\';

par.showrecons = par.verbose_level > 1;

% Other
par.showsorted = true;      % sort plotted projections by angles, it has not effect on the alignment itself 
par.windowautopos = true;
par.save_memory = false;        % try to limit use of RAM 
par.inplace_processing = par.save_memory; % process object_stack using inplace operations to save memory 
par.fp16_precision     = par.save_memory; % use 16-bit precision to store the complex-valued projections 
par.cache_stack_object = par.save_memory; % store stack_object to disk when no needed 
par.force_overwrite = par.online_tomo ;     % ask before overwritting data 

par.GPU_list = [1];     % number of the used GPU % If you want to check usage of GPU 
                        % > nvidia-smi
                        % Then in matlab use  par.GPU_list = 2  for example to use the second GPU 
par.Nworkers = min(10,feature('numcores'));  % number of workers for parfor. avoid starting too many workers

%%% reconstruction settings 
par.air_gap = [50,50];    % very roughly distance around sides where is assumed air, it is used only for initial guess       

% Geometry settings, important for laminography 
par.is_laminography = false;        % false -> standard tomography, true = allow some specific options for laminography reconstruction, ie circular field of view ... 

%%%% geometry settings %% 
par.lamino_angle = 90;              % laminography angle, for normal tomo set 90
par.tilt_angle = 0;                 % rotation of the camera around the beam direction 
par.vertical_scale = 1;             % relative pixel scale between the projection and reconstruction 
par.horizontal_scale = 1;           % relative pixel scale between the projection and reconstruction 

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

block_fun_cfg = struct('GPU_list', par.GPU_list, 'inplace', par.inplace_processing);   % instruction for stack_object processing
object_preprocess_fun = @(x)x;    % data preprocessing function, useful for laminography


%% Reads ptychographic reconstructions and stores them in stack_object
utils.verbose(-1,'Loading saved projections...')

datafile = [par.base_path, filename];

theta = h5read(datafile,'/theta');
theta = theta(1:par.num_proj);
theta = theta/2/pi*360;

par.pixel_size = 0.65e-6*2;
par.asize= [16,16];

par.lambda = 0.1e-9;

projs_data = h5read(datafile,'/data');
projs_data = projs_data(:,:,1:par.num_proj);
projs_data = permute(projs_data,[2,1,3]);

%% remove background
stack_object = zeros(size(projs_data));
for jj=1:size(stack_object,3)
    stack_object(:,:,jj) = projs_data(:,:,jj) - projs_data(1,1,jj);
end
utils.verbose(-1,'Done')

tomo.show_projections(stack_object, theta, par, ...
    'title', 'Full original projections before alignmnent') 


%% 
stack_object = exp(1j*log(stack_object+1));

%% 

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
exclude_scans = [];
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply some very preliminary phase ramp removal - has to be applied before the dataset is shifted to avoid ramp artefacts at the edges 
utils.verbose(-1,'Roughly remove phase-ramp and normalize amplitude')
    
par.dims_ob_loaded = [size(stack_object,1), size(stack_object,2)];  % load the sizes directly from the object, note that "object_preprocess_fun" can crop/rotate the image !!
par.illum_sum = ones(par.dims_ob_loaded-par.asize);
par.illum_sum = utils.crop_pad(par.illum_sum,par.dims_ob_loaded);
par.illum_sum = par.illum_sum ./ quantile(par.illum_sum(:), 0.9);  % normalize the values to keep maximum around 1

% stack_object = tomo.block_fun(@utils.stabilize_phase,stack_object, 'weight', par.illum_sum / max(par.illum_sum(:)),'normalize_amplitude',true, block_fun_cfg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%  RECONSTRUCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
utils.verbose(-1,'Preparing reconstruction parameters...')

[Nx,Ny,Nangles] = size(stack_object);

% Choose reconstructed region
% default is take as much as possible (safe)
vert_crop  = par.asize(1)/2;  % use default cropping equal to half of probe size removed from each side
horiz_crop = par.asize(2)/2;  % use default cropping equal to half of probe size removed from each side 
object_ROI = {ceil(1+vert_crop :Nx-vert_crop),...
              ceil(1+horiz_crop:Ny-horiz_crop)}; 
          

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% prevent rewritting "total_shift" value if user runs this section mutliple times 
total_shift = zeros(Nangles,2);  %% store all shifts for future use 

% Make data easily splitable for ASTRA, preferable size of blocks should be dividable by 32
width_sinogram = ceil(length(object_ROI{2})/32)*32;
Nlayers = floor(length(object_ROI{1})/32)*32;
Nroi = [length(object_ROI{1}),length(object_ROI{2})];
object_ROI = {object_ROI{1}(ceil(Nroi(1)/2))+[1-Nlayers/2:Nlayers/2],...
              object_ROI{2}(ceil(Nroi(2)/2))+[1-width_sinogram/2:width_sinogram/2]}; 

%% Display reconstructed phases
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
par.baraxis = 'auto';       % = 'auto'  or   = [-1 1]
par.windowautopos = true;  % automatic placemement of the plot
par.showsorted = true;      % sort the projections by angles 
plot_residua = false;      % if true, denote residua in projections by red circles. 

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
tomo.show_projections(stack_object, theta, par,  'fnct', @angle,...
    'title', 'Full original projections before alignmnent','plot_residua', plot_residua, ...
    'rectangle_pos', [object_ROI{2}(1), object_ROI{2}(end), object_ROI{1}(1), object_ROI{1}(end)]) 

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Manual removal of poor projections %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
which_remove =  [];         % list of projection indices or bool vector of projections to be removed  
                            % Examples: 
                            %      which_remove = [1,5,10]
                            %      which_remove = ismember(par.scanstomo, [264,1023])
                            %      which_remove = theta > 0.5 & theta < 15
                            %       
                            %      
plot_fnct = @angle;         % function used to preprocess the complex projections before plotting, use @(x)x to show raw projections 

[stack_object,theta,total_shift,par] = tomo.remove_projections(stack_object,theta,total_shift,par, which_remove, plot_fnct, object_ROI); 
[Nx,Ny,Nangles] = size(stack_object);


%% Cross-correlation alignment of raw data - only rough guess to ease the following steps 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
par.filter_pos = Nangles/4;        %  highpass filter on the evolution of the recovered positions applied in the following form:  X - smooth(X,par.filter_pos), it prevents accumulation of drifts in the reconstructed shifts 
par.filter_data = 0.01;    %  highpass filter on the sinograms to avoid effects of low spatial freq. errors like phase-ramp 
par.max_iter = 3;           %  maximal number of iterations 
par.precision = 0.1;          %  pixels; stopping criterion
par.binning = 4;           %  binning used to speed up the cross-correlation guess 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%


utils.verbose(-1,'Cross-correlation pre-alignment of raw data')
[xcorr_shift, variation_binned, variation_aligned]  = tomo.align_tomo_Xcorr(stack_object, theta, par, 'ROI', object_ROI);

% show prealigned projections 
tomo.show_projections(cat(1,variation_aligned, variation_binned), theta, par, 'title', sprintf('Preview of cross-correlation pre-aligned projections, binning %ix \n Top: original Bottom: aligned \n', par.binning), 'figure_id', 12)
axis off image 

clear variation_binned variation_aligned

%% 
utils.verbose(-1,'Apply shifts found by cross-correlation to projections...')
stack_object = tomo.block_fun(@utils.imshift_linear, stack_object, xcorr_shift(:,1),xcorr_shift(:,2), 'circ', block_fun_cfg);
total_shift = total_shift + xcorr_shift;
utils.verbose(-1,'Done \n')

%% Register (align) projections by vertical mass fluctuations and center of mass
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
par.alignment_invariant = 'phase_2D'; 
        % 'phase_1D' = vertical mass fluctuation with 1D unwrapping, 
        %                useful if sample has strongly scatteting layers, e.g. platinum on the top of the pillar 
        % 'phase_2D' = vertical mass fluctuation with 2D unwrapping, 
        %              more robust than 1D, more resistent againts phase jumps / residua 
        % 'phase_derivative' = use fluctuation of vertical derivative, more
        %                 robust agains phase residues, 
        %                 great for capilaries 
        % 'phase_goldstein'  = precise unwrapping method, slow and can be used
        %                only if no resiua are present , NOT RECOMMENDED 
        
par.data_filter = 0.01;              % High pass filter value for the vertical mass fluctuations curves, higher-> more filtering  
par.use_vertical_xcorr_guess = true;  % get initial guess though crosscorelation -> helps to escape from local minima, but in some cases may find wrong solution
par.phase_jumps_threshold  = 1.0;  % detection threshold for phase jumps 
par.vert_range = [];               % selected vertical layers for alignment, 

%RECOMMENDED PROCEDURE:
%start with full range ([] = full), 
%later you may refine for smaller field of view, 
% the alignment may fail is sample vertically moves more than half FOV or if partly drifts
% horizontally out of FOV 
% Try avoid highly scattering / residial features in the vert_range

% MASKING OF POOR REGIONS FOR "phase_2D" UNWRAPPING , weights = [] -> ignore mask and use whole region inside of object_ROI
weights = [];                      % (3D array) weights can be used to mask unwanted regions in projections, W==0 ignored pixel, W=1 fully considered pixel, 0<W<1 partly considered pixel
% thresh = 0.8;                    % regions where amplitude is below thresh * mean(absorbtion) will be masked 
% mask_erode = 3;                  % number of pixels to be removed around the masked features 
% %% example of an weights array  
% fun = @(x)(convn(abs(x) > thresh * mean(mean(abs(x(object_ROI{:},:)))), ones(mask_erode)/mask_erode^2, 'same')==1);  % ball tracking
% weights = tomo.block_fun(fun, stack_object, struct('use_GPU', true)); 

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

init_vshift = [];                % initial guess of the shift for the stack_object
utils.verbose(-1,'Initial prealignment')
% Tomography invariant (alignment)
init_vshift = tomo.align_tomo_initial(stack_object, init_vshift, theta, object_ROI, par, 'weights', weights);


% show preview 
preview = tomo.block_fun(@(x,shift)utils.imshift_fft_ax(utils.binning_2D(x,par.binning),shift/par.binning,1), stack_object, init_vshift, struct('ROI', {object_ROI}, 'use_GPU', false, 'use_fp16', false, 'weights', weights));
tomo.show_projections(preview, theta, par, 'fnct', @angle, 'title', sprintf('Vertically aligned projections, BINNED %ix', par.binning))

%% 

if par.online_tomo || debug() || ~strcmpi(input('Apply vertical shifts found by vertical mass fluctuation method: [Y/n]\n', 's'), 'n')
    % shift the complex objects 
    stack_object = tomo.block_fun(@utils.imshift_fft, stack_object, 0,init_vshift);
    % store the total shift
    total_shift(:,2) = total_shift(:,2) + init_vshift; 
else
    return
end

%% Show radiation damage with vertical derivative phasors
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
Nmodes = 1;        % Number of plotted SVD modes 
median_smoothing = 15;        % length of smoothing window along time axis 
invariant_method = 'derivative';   % 'phase' = standard vertical mass flustuation, 'derivative' = use fluctuation of vertical derivative  
vert_range = par.vert_range;  % vertical range used to evaluate the radiation damage, try avoid phase jumps and highly scattering structures 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
if par.online_tomo
    tomo.radiation_damage_estimation(stack_object(object_ROI{:},:), par,...
        'N_SVD_modes',  Nmodes, 'smoothing', median_smoothing, 'logscale', false, ...
        'invariant', invariant_method, 'vert_range', vert_range)
end


%% TOMOCONSISTENCY ALIGNMENT
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% RECONSTRUCTION PARAMETERS %%%%%%%%%%%%%%%%%%%%
Npix = ceil(1.0*width_sinogram/32)*32;  % for pillar it can be the same as width_sinogram
par.vert_range = 32:Nlayers-33; % selected vertical layers for alignment 

%%%%%%%%%% Other parameters %%%%%%%%
max_binning = 2^ceil(log2(max(Npix))-log2(100));   % prevent binning to make smaller object than 100 pixels, reconstruction gets unstable 
min_binning = min(max_binning, 2^par.online_tomo) ; 


%%%%%%%%%% Useful tuning parameters %%%%%%%%%%%%%%%
par.high_pass_filter = 0.01;        % remove effect of residuums/bad ptycho convergence , high value may get stack in local minima , low value increases sensitivity to residua / phase jumps 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% In case of problem you can also change %%%%
par.momentum_acceleration = false;   % accelerate convergence by momentum gradient descent method, set to false in case the the convergence is failing
par.apply_positivity = false;        % remove negative values in the reconstructed volume, helps with alignment especially for sparser samples, it can cause problems if the unwrapped phase contain low spatial frequency errors (-> positivity constraint is not valid anymore)

%%%%%%%%%% Other settings %%%%%%%%%%%%%%%%%%%%%%%%%
par.showsorted = true;              % selected angular order for alignment 
par.valid_angles = 1:Nangles > 0;   % use only this indices to do reconstruction (with respect to the time sorted angles)
par.center_reconstruction = false;   % keep the center of mass of the reconstructed sample in center of 3D field of view
par.align_horizontal = true;        % horizontal alignment 
par.align_vertical = false;         % vertical alignment, usually only a small correction of initial guess  
par.use_mask = false;               % apply support mask on the reconstructed sample volume 
par.mask_threshold = 0.001;         % []== Otsu thresholding, otherwise the value is in phase tomogram reconstruction 
par.use_localTV = false;            % apply local TV into the reconstructed volume, should help with convergence  
par.min_step_size  = 0.01;          % stopping criterion ( subpixel precision )
par.max_iter = 300;                 % maximal number of iterations
par.use_Xcorr_outlier_check = false; % in the first iteration check and remove outliers using Xcorr 
par.refine_geometry = false;        % try to find better geoometry to fit the data

%%%  Internal parameters, do not change if you are not sure %%%%%%%%
par.step_relaxation = 0.01;          % gradient decent step relaxation, (1 == full step), value <1 may be needed to avoid oscilations  
par.filter_type = 'ram-lak';        % FBP filter (ram-lak, hamming, ....)
par.freq_scale = 1;                 % Frequency cutoff for the FBP filters, 1 == no FBP filtering
par.unwrap_data_method = 'none';  % Unwrap method: fft_2d, fft_1d, none - assume that provided sinogram is already unwrapped 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Generate inputs for tomoconsistency alignment
utils.verbose(-1,'Get phase gradient')
% solve the function blockwise on GPU 
% sinogram = tomo.block_fun(@math.get_phase_gradient_1D, stack_object, 2, struct('GPU_list', par.GPU_list, 'ROI', {object_ROI}, 'use_fp16', false)); 
sinogram = tomo.unwrap2D_fft2_split(stack_object, par.air_gap ,1, par.GPU_list);

utils.verbose(-1,'Get mask')
% include the effect of high pass filter into the weights 
ker = gausswin(max(3,ceil(par.high_pass_filter*width_sinogram)))'.*hanning(3); 
% relevance weights -> remove effect of potential residues / phase jumps 
sino_weights = max(0,1-convn(single(abs(sinogram) > 2), single(ker), 'same')); 
utils.verbose(-1,'Done')

tomo.show_projections(sinogram, theta, par, 'title', 'Selected region')

%% CONSISTENCY-BASED ALIGNMENT 
shift = zeros(Nangles, 2);

binning = 2.^(log2(max_binning):-1:log2(min_binning));
 
utils.verbose(-1,'Alignment')


for jj = 1:length(binning)
    par.binning = binning(jj);
    utils.verbose(-1, 'Binning %i', par.binning)
    % self consitency based alignment procedure based on the ASTRA toolbox 
    [shift, par] = tomo.align_tomo_consistency_linear(sinogram,sino_weights, theta+0.1, Npix, shift, par); 
    % try to use xcorr to prevent trapping in local minima
    if  par.binning == min(8,max_binning) && par.use_Xcorr_outlier_check
        xcorr_shift = tomo.align_tomo_consistency_Xcorr(sinogram, sino_weights, theta+0.1, shift, par.binning, Npix, par);
        if par.online_tomo || all(abs(xcorr_shift(:)) < 2*par.binning) || strcmpi(input('Apply shifts found by Xcorr-consistent alignment: [y/N]\n', 's'), 'y')
            shift = shift + xcorr_shift ;
        end
    end
    
    % plot the estimated shifts 
    plotting.smart_figure(25)
    clf()
    [~,ind_sort] = sort(theta);
    plot(theta(ind_sort), shift(ind_sort,:), '.')
    legend({ 'Horizontal shift', 'Vertical shift'})
    xlabel('Angle')
    ylabel('Shift [px]')
    xlabel('Sorted angles')
    title('Total shift from self-consistency alignment')
    axis tight ; grid on 
    drawnow

    
end
utils.verbose(-1,'Alignment finished')


%%

% show aligned projections before shifting full stack_object
tomo.show_projections(tomo.block_fun(@utils.imshift_fft, sinogram, shift), theta, par,...
    'rectangle_pos', [1,length(object_ROI{2}),par.vert_range([1,end])], ...
    'title', 'Projections after self-consistent alignment')


%% 
clear sinogram rec sino_weights
if par.cache_stack_object && ~exist('stack_object', 'var')
    stack_object = load_stored_object('stack_object'); 
end
if par.online_tomo || debug() || ~strcmpi(input('Apply shifts found by self-consistent alignment: [Y/n]\n', 's'), 'n')
    % shift the complex objects  
    stack_object = tomo.block_fun(@(x,s)utils.imshift_fft(utils.imshear_fft(x,-par.skewness_angle,1),s), stack_object, shift, block_fun_cfg);
    % store the total shift
    total_shift = total_shift + shift; 
    par.skewness_angle = 0;
else
    return
end
if par.cache_stack_object
    utils.savefast_safe('stack_object', 'stack_object', 'par', 'theta','total_shift', true)
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of alignment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Phase ramp removal + amplitude calibration %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
Niter = 3;                  % number of iterations of phase ramp refinement 
binning = max_binning;      % bin data before phase ramp removal (make it faster)
ROI = object_ROI; 
ROI{1} = ROI{1}(par.vert_range); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

par.unwrap_data_method = 'fft_2D'; 
% remove phase ramp from data to using self consistency
stack_object = tomo.phase_ramp_removal_tomo(stack_object,object_ROI, theta, Npix,total_shift, par, 'binning', binning, 'Niter', Niter, ...
    'positivity', true, 'auto_weighting', true);


% show the sorted projection, check if all are ok. If not try to run
% alignment again. Maybe change vertical range or high_pass filter 
tomo.show_projections(stack_object, theta, par, 'fnct', @(x)angle(x(object_ROI{:},:)),  'plot_residua', false)


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of phase refinement 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save aligned complex projections for external processing 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
save_external = false; 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
if save_external && ~par.online_tomo && ~debug()
    utils.savefast_safe([par.output_folder,'/stack_object_external'], 'stack_object', 'theta', 'par', 'Npix', 'total_shift', 'object_ROI')
end

%% (EXPERIMENTAL FEATURE) Find optimal propagation (numerical refocusing), try to minimize amplitude in the complex projections 
%% It may slighly improve laminography reconstructions or sub-10nm tomograms 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
refocus_projections = false;
refocusing_range = linspace(-4,4,100)*1e-6;               % scanning range 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
     
if refocus_projections
    optimal_propagation = projection_propagation_optimization( stack_object, theta, refocusing_range, object_ROI, par); 
    if par.online_tomo || ~strcmpi(input(sprintf('Apply the found optimal_propagation of %3.2gum? Is the value from phase and amplitude consistent ? : [Y/n]\n', median(optimal_propagation)*1e6), 's'), 'n')
        stack_object = tomo.block_fun(@utils.prop_free_nf, stack_object, par.lambda, median(optimal_propagation), par.pixel_size,  struct('GPU_list',par.GPU_list)); 
    end
end

%% Full tomogram
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
par.usecircle = true;               % Use circle to mask out the corners of the tomogram
par.filter_type = 'ram-lak';        % FBP filter (ram-lak, hamming, ....)
par.freq_scale = 1;                 % Frequency cutoff
rec_ind = find(par.valid_angles);   % use only some angles 
apodize = 0;                        % axial apodization 
radial_smooth_apodize = Npix(1)/50; % smooths the circle function (circulo), which delimits the reconstruction, this to avoid sharp edges in the  reconstruction
par.phase_unwrapping = 'fft_2d';    % phase unwrapping methods: fft_1d, fft_2d, none (use phase difference), bootstrap (iterative unwrapping in case of strong phase residua)
vert_range = 'manual';              % vertical region to be reconstructed, options:  'manual'   - manual selection by user 
                                    %                                                [] - vert_range = object_ROI{1}
                                    %                                                indices - list of layers with respect to the stack_object file that is used for reconstruction
% MASKING OF POOR REGIONS FOR "phase_2D" UNWRAPPING , weights = [] -> ignore mask and use whole region inside of object_ROI
weights = [];                      % (3D array) weights can be used to mask unwanted regions in projections, W==0 ignored pixel, W=1 fully considered pixel, 0<W<1 partly considered pixel
% thresh = 0.8;                    % regions where amplitude is below thresh * mean(absorbtion) will be masked 
% mask_erode = 3;                  % number of pixels to be removed around the masked features 
% %% example of an weights array  
% fun = @(x)(convn(abs(x) > thresh * mean(mean(abs(x(object_ROI{:},:)))), ones(mask_erode)/mask_erode^2, 'same')==1);  % ball tracking
% weights = tomo.block_fun(fun, stack_object, struct('use_GPU', true)); 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

utils.verbose(struct('prefix', 'recons'))

if par.online_tomo || isempty(vert_range) || debug()
    reconstruct_ROI = object_ROI; 
elseif isnumeric(vert_range)
    reconstruct_ROI = {vert_range, object_ROI{2}};
else
    if ishandle(2); close(2); end  % force closing and reopening on the front
    plotting.smart_figure(2);
    plotting.imagesc3D(object_ROI{2},object_ROI{1},stack_object,'fnct',@(x)angle(fp16.get(x(object_ROI{:}))))
    title('Select reconstruction region')
    axis image xy
    colormap bone
    grid on 
    utils.verbose(-1,'Manually select vertical reconstruction region (horizontal range is used always full) ')
    rect = round(getrect);
    rect(1:2) = max([object_ROI{2}(1), object_ROI{1}(1)], rect(1:2)) ;
    rect(3:4) = min(ceil(rect(3:4)/16)*16,[length(object_ROI{2}),length(object_ROI{1})]); 
    close(2)
    reconstruct_ROI = {(rect(2):rect(2)+rect(4)-1), object_ROI{2}};
end

circulo = [];  % apodization function 

utils.verbose(-1,'Generating sinograms ...')
switch lower(par.phase_unwrapping)
    case 'fft_2d'  % 2D phase unwrapping 
        sinogram = tomo.unwrap2D_fft2_split(stack_object,par.air_gap,0,weights, par.GPU_list, reconstruct_ROI, []);
    case 'fft_1d'  % 1D phase unwrapping, similar results as phase gradient
        sinogram = -tomo.block_fun(@math.unwrap2D_fft,stack_object,2,par.air_gap, struct('GPU_list',par.GPU_list,'ROI',{reconstruct_ROI},'use_fp16',false));
    case 'none'  % get phase gradient 
        sinogram = tomo.block_fun(@math.get_phase_gradient_1D,stack_object,2,0,-1, struct('GPU_list',par.GPU_list,'ROI',{reconstruct_ROI},'use_fp16',false));
    otherwise, math('Missing unwrap method')
end
utils.verbose(-1,'Sinograms done')

%% 
% Get two reconstructions for FSC   %%%%%%%%%%%%%%
[~,ind_sort] = sort(theta); 
ind_sort = ind_sort(ismember(ind_sort, rec_ind));
ind_rec = {ind_sort(1:2:end), ind_sort(2:2:end)}; 

[Ny_sino,Nx_sino,~] = size(sinogram);
CoR = [Ny_sino,Nx_sino]/2+[0,0.5]*strcmpi(par.phase_unwrapping,'none');   % there is 0.5px shift between fft_1d/fft_2d vs none
[cfg, vectors] = astra.ASTRA_initialize([Npix,Npix, Ny_sino],[Ny_sino,Nx_sino],theta,par.lamino_angle,par.tilt_angle,1,CoR); 
% find optimal split of the dataset for given GPU 
split = astra.ASTRA_find_optimal_split(cfg, length(par.GPU_list), 2);


% remove artefacts around edges of tomogram 
if par.usecircle
    [~,circulo] = utils.apply_3D_apodization(ones(Npix), apodize, 0, radial_smooth_apodize); 
end

tomograms = {};
utils.verbose(-1,'Reconstructing ... ')
for ii = 1:2
    % FBP code 
    tomograms{ii} = tomo.FBP_zsplit(sinogram, cfg, vectors, split,'valid_angles',ind_rec{ii},...
        'GPU', par.GPU_list,'filter',par.filter_type, 'filter_value',par.freq_scale,...
        'use_derivative', strcmpi(par.phase_unwrapping, 'none'), 'mask', circulo);
end
utils.verbose(-1,'FBP reconstruction done')



%% 3D FSC 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
rad_apod = Npix(1)/5;           % Radial apodization
axial_apod = 20;         % Axial apodization
radial_smooth = (Npix/2-rad_apod)/10; % smooths the circle function (circulo), which delimits the reconstruction, this to avoid sharp edges in the  reconstruction
SNRt = 0.2071; %1/2 bit  % Threshold curve for FSC
thickring = 5;         % use thickring for smoothing 
FSC_vertical_range =1:Ny_sino;  % Choose  vertical region for FSC
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

par.online_tomo_path = par.output_folder;
% call "close(45)" if you want to plot FSC to clean window (without overlay with previous FSC curve)
[resolution FSC T freq n FSC_stats, fsc_path] = tomo.get_FSC_from_subtomos(tomograms, FSC_vertical_range, rad_apod,radial_smooth,axial_apod,SNRt,thickring,par);
utils.savefast_safe([fsc_path, '.mat'], 'FSC','T','freq','n','FSC_stats', 'rad_apod','axial_apod','circulo','theta', 'total_shift', 'par', par.force_overwrite || par.online_tomo);



%% Calculate delta tomogram 

% get full reconstruction (for FBP is sum already final tomogram)
% calculate complex refractive index 
% par.factor=par.lambda/(2*pi*par.pixel_size); 
par.factor = 1;
% par.factor_edensity = 1e-30*2*pi/(par.lambda^2*2.81794e-15);

par.rec_delta_info = ['FBP_',par.filter_type '_freqscl_' sprintf('%0.2f',par.freq_scale)]; 
tomogram_delta = ((tomograms{1} + tomograms{2})/2)*par.factor;
% clear the tomograms for FSC from RAM 
% clear tomograms

%%  SART  solver - helps mainly for sparse sample (with a lot of air gaps) and in case of angularly undersampled tomograms 
%%   - solver uses positivity constraint 
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% par.Niter_SART = 50;     % number of SART iteration 
% SART_block_size = 100;   % size of blocks solved in parallel
% par.usecircle = true;                 % Use circle to mask out the corners of the tomogram
% relax = 0.1 ;                       % SART relaxation, 0 = no relaxation
% %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%
% 
% if par.Niter_SART > 0
%     if par.usecircle
%         [~,circulo] = utils.apply_3D_apodization(ones(Npix), apodize, 0, radial_smooth_apodize);
%     else
%         circulo = 1; 
%     end
% 
%     utils.verbose(-1,'SART preparation', ii, par.Niter_SART)
%     [Ny_sino,Nx_sino,~] = size(sinogram);
%     CoR = [Ny_sino,Nx_sino]/2+[0,0.5]*strcmpi(par.phase_unwrapping,'none');   % there is 0.5px shift between fft_1d/fft_2d vs none
%     [cfg, vectors] = astra.ASTRA_initialize([Npix,Npix, Ny_sino],[Ny_sino,Nx_sino],theta,par.lamino_angle,par.tilt_angle,1,CoR); 
%     
%     split = astra.ASTRA_find_optimal_split(cfg, length(par.GPU_list), 1, 'both');
%     [cache_SART, cfg] = tomo.SART_prepare(cfg, vectors, SART_block_size, split);
% 
%     tomogram_SART = tomogram_delta/par.factor;
%     clear err_sart 
%     for ii = 1:par.Niter_SART
%         utils.verbose(-1,'SART iter %i/%i', ii, par.Niter_SART)
%         [tomogram_SART,err_sart(ii,:)] = tomo.SART(tomogram_SART, sinogram, cfg, vectors, cache_SART, split, ...
%             'relax',relax, 'constraint', @(x)(x .* (0.9+0.1*circulo))) ; 
%         plotting.smart_figure(1111)
%         ax(1)=subplot(2,2,1);
%         plotting.imagesc3D(tomogram_delta, 'init_frame', size(tomogram_delta,3)/2); axis off image; colormap bone; 
%         title('Original FBP reconstruction')
%         ax(2)=subplot(2,2,2);
%         plotting.imagesc3D(tomogram_SART, 'init_frame', size(tomogram_delta,3)/2); axis off image; colormap bone;  
%         title('Current SART reconstruction')
%         subplot(2,1,2)
%         plot(1:ii,err_sart)
%         hold all
%         plot(1:ii,median(err_sart,2),'k--', 'Linewidth',4)
%         hold off 
%         xlabel('Iteration #'); ylabel('Error'); set(gca, 'xscale', 'log'); set(gca, 'yscale', 'log')
%         grid on  
%         title('Evolution of projection-space error')
%         linkaxes(ax, 'xy')
%         plotting.suptitle('SART reconstruction')
%         drawnow
%     end
%     
%     figure
%     plotting.imagesc_tomo(tomogram_SART); 
%     plotting.suptitle('Tomogram delta - FBP reconstruction')
%     
% end

%% 
%Should the SART reconstruction be saved to tomogram_delta ? [Y/n]'), 'n')
% store SART into tomogram_delta
% tomogram_delta = tomogram_SART * par.factor; 
% par.rec_delta_info = sprintf('SART_Niter=%i', par.Niter_SART); 
% 
% clear tomogram_SART cache_SART
% clear sinogram 

% %% Show radiation damage using PCA estimation of the tomogram time evolution
% %%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Edit this section %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%
% estimate_evolution = true;        % Number of plotted SVD modes 
% reconstruct_ROI_svd = {object_ROI{1}(ceil(length(object_ROI{1})/2) + [-5:5]), object_ROI{2}};  % select region of the sinogram used for reconstruction 
% N_SVD_modes = 2; % number of recovered SVD modes, 2 is usually enough 
% Niter_SVD = 4;   % number of iter of the PCA SART
% %%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%
% if estimate_evolution  || debug()
%     % perform 2D FFT-based unwrapping
%     sinogram = -tomo.unwrap2D_fft2_split(stack_object,par.air_gap,0,[], par.GPU_list, reconstruct_ROI_svd, []);
%     % perform PCA-SART to 
%     subtomo_range = [0, find(abs(diff(par.subtomos)) > 0), Nangles]; % try to estimate subtomograms, if it does not work, provide start and end indices for each subtomo manually 
% 
%     if length(subtomo_range) > 2 && length(subtomo_range) < 20
%         subtomo_inf = cell(length(subtomo_range)-1,1);
%         for ii = 1:length(subtomo_range)-1
%             subtomo_inf{ii} = 1+subtomo_range(ii):subtomo_range(ii+1); 
%         end
% 
%         % perform SVD regularized reconstruction, if Niter_SVD == Nsubtomos, it
%         % is equivalent to N independend reconstructions 
%         [~,circulo] = utils.apply_3D_apodization(ones(Npix), apodize, 0, radial_smooth_apodize); 
%         %[U,S,V,rec_per_subtomo] = ...
%             nonrigid.SART_SVD(sinogram, theta, Npix, subtomo_inf, 'Niter_SVD', Niter_SVD, 'N_SVD_modes', N_SVD_modes, 'output_folder', par.output_folder, 'mask', circulo); 
%     end
% end



%% ROTATE THE TOMOGRAM 
% try to find a most significant direction in the sample and align along
% it, useful for chips :)
% 
% [rot_angle(1)] = utils.find_img_rotation_2D(utils.Garray(squeeze(tomogram_delta(end/2,:,:))));
% [rot_angle(2)] = utils.find_img_rotation_2D(utils.Garray(squeeze(tomogram_delta(:,end/2,:))));
% [rot_angle(3)] = utils.find_img_rotation_2D(utils.Garray(mean(tomogram_delta,3)));
% 
% utils.verbose(-1,'Rotating')
% tomogram_delta = permute(tomo.block_fun(@utils.imrotate_ax_fft,permute(tomogram_delta,[3,2,1]),-rot_angle(1)),[3,2,1]);
% tomogram_delta = permute(tomo.block_fun(@utils.imrotate_ax_fft,permute(tomogram_delta,[1,3,2]), rot_angle(2)),[1,3,2]);
% tomogram_delta = tomo.block_fun(@utils.imrotate_ax_fft,tomogram_delta, rot_angle(3));
% utils.verbose(-1,'Done')
% 
% % % show quick preview 
% figure; plotting.imagesc_tomo(tomogram_delta)



%% Display reconstructed volume by slices
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
figure(131)
plotting.imagesc_tomo(tomogram_delta); 

%% Save Tomogram (delta)
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
savedata = true;
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Note saving tomogram in delta (index of refraction)
% To get phase tomogram   = tomogram_delta/par.factor
% To get electron density tomogram in [e/A^3]   = tomogram_delta*par.factor_edensity
% If you saved before you may clear 

par.scans_string = '1500epochs_901projs';
par.rec_delta_info = 'FBP';

tomo.save_tomogram(tomogram_delta, par, 'delta', circulo, theta, par.rec_delta_info)

%% Save Tiff files for 3D visualization with external program
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edit this section %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
par.save_as_stack = true; 
par.tiff_compression = 'none';
par.tiff_subfolder_name = ['TIFF_delta_' par.rec_delta_info];
par.name_prefix = 'tomo_delta';
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
tomo.save_as_tiff(tomogram_delta, par, par.rec_delta_info)



%*-----------------------------------------------------------------------*
%|                                                                       |
%|  Except where otherwise noted, this work is licensed under a          |
%|  Creative Commons Attribution-NonCommercial-ShareAlike 4.0            |
%|  International (CC BY-NC-SA 4.0) license.                             |
%|                                                                       |
%|  Copyright (c) 2017 by Paul Scherrer Institute (http://www.psi.ch)    |
%|                                                                       |
%|       Author: CXS group, PSI                                          |
%*-----------------------------------------------------------------------*
% You may use this code with the following provisions:
%
% If the code is fully or partially redistributed, or rewritten in another
%   computing language this notice should be included in the redistribution.
%
% If this code, or subfunctions or parts of it, is used for research in a 
%   publication or if it is fully or partially rewritten for another 
%   computing language the authors and institution should be acknowledged 
%   in written form in the publication: “Data processing was carried out 
%   using the “cSAXS matlab package” developed by the CXS group,
%   Paul Scherrer Institut, Switzerland.” 
%   Variations on the latter text can be incorporated upon discussion with 
%   the CXS group if needed to more specifically reflect the use of the package 
%   for the published work.
%
% A publication that focuses on describing features, or parameters, that
%    are already existing in the code should be first discussed with the
%    authors.
%   
% This code and subroutines are part of a continuous development, they 
%    are provided “as they are” without guarantees or liability on part
%    of PSI or the authors. It is the user responsibility to ensure its 
%    proper use and the correctness of the results.
