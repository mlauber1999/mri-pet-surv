function process_files(suffix)
addpath(genpath('/home/mfromano/spm/spm12/')); %this generates a folder named /home/mfromano/spm/spm12/ and adds it to the search path 
%do I need to create a new path for my folder or will this work? 

%Orgininal:
%BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
%changed to:
BASE_DIR = ['/data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep];
%this creates a 1 row by 3 column array named BASE_DIR with the file path data2.. being the first value, suffix being the middle value, and filesep being the last value 

MRI_folder_new = [BASE_DIR 'ADNI_MRI_nii_recenter' suffix filesep];
%this creates a 1 row by 4 column array named MRI_folder_new with Base_dir being the first value, suffix being the middle value, and filesep being the last value 

fnames = dir(MRI_folder_new);
%defining the filenames as all files in the directory where MRI_folders_new is 
%when you use dir you get .name for name of the file 
fnames = {fnames.name};
%redefine fnames as the name of each file in folder
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
%regex tells it to match x with the specified conditions (0-9), 4 digits long, ending in .nii, fnames is the names of all the files within the directory mri_folder_new
rids = [rids{:}];
%creating a matrix of the rids, taking specific data types {:} 
%rids variable is now equal to all of the rids 
%{:}  means take all contents of a cell array 
rids = [rids{:}];
%redefining rids 
rids = [rids{:}];
%redefining rids 
%original: 
%mkdir(['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'ADNI_MRI_nii_recenter_amyloid' suffix]);
%changed to: 
mkdir(['/data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep 'ADNI_MRI_nii_recenter_amyloid' suffix]);\

%makes a new directory consisting of a 1x4 array, where the first value is /data2..filepath, the second is suffix, the third is Adni_mri filepath, and the fourth is suffix.
%I dont think thats exactly right though 

%Original: 
%mkdir(['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix]);
%changed to: 
mkdir(['/data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep 'brain_stripped' suffix]);
%not sure exactly what this is doing 

rand('state',10);
%older form of uniform random number generator, rand('state',10) for the integer 10 initalizes the generator to the 10th integer state 
maxNumCompThreads = 1;
%sets the current maximum number of computational threads to 1 

spm('defaults', 'PET');
%tells sprm to do the default settings for pet scans? 

spm_jobman('initcfg');
%command that allows you to batch process spm through the command line 
parpool(14);
%this creates a paralell pool of workers using the default cluster, parameter is poolsize (number of workers) in this case 14 

%executes for loop iterations in parallel on workers 
parfor i=1:length(rids) %specifies it should iterate through all the elements in rids 
        rand('state',10); %initalizes the generator to the 10th integer state 
        %original: 
        %curr_mri = ['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter' suffix filesep rids{i} '_mri.nii'];
        %changed to: 
        curr_mri = ['/data2/MRI_PET_DATA/ML/processed_images_final' suffix '/ADNI_MRI_nii_recenter' suffix filesep rids{i} '_mri.nii'];
        %creates an array named curr_mri in the filepath /data2.. with suffix /adni_mri_nii_recenter with the rids id number and ending in suffix _mri.nii 
        disp(['copying ' rids{i}]) %prints copying the rids number in the command line 
        %original:
        %system(['rsync -av ' curr_mri ' /data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_amyloid' suffix]);
        system(['rsync -av ' curr_mri ' /data2/MRI_PET_DATA/ML/processed_images_final' suffix '/ADNI_MRI_nii_recenter_amyloid' suffix]);
        %system tells it to interact with the os 
        %rysnc is copying and syncing data from one computer to another 
        %rsync -a means archive, syncs directories recursively, preserve symbolic links, modification times, groups, ownership, and permission
        %rsync -v means verbose, means it gives you information about the files being transferred and gives summary at the end 
%         system(['rsync -av ' curr_mri ' /data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_fdg' suffix filesep]);
%I think the above line is commented out bc were not doing the fdg pet, uncomment it if I do  
        jobs = batch_process_amyloidmri(rids{i}, suffix);
        %creates variable jobs as calling the function batch_process_amyloid_mri and include the suffix 
        spm_jobman('run_nogui', jobs);
        %tells spm to run without the gui on jobs (to batch process the amyloid mri) 
end
% 
% spm('defaults', 'PET');
% spm_jobman('initcfg');
% 
% parfor i=1:length(rids)
%         rand('state',10);
%         jobs = batch_process_fdg(rids{i}, suffix);
%         job = spm_jobman('run_nogui', jobs);
% end

% modality = {'amyloid','mri','fdg'};
modality = {'amyloid','mri'};
%creates variable to specify the modlaities are amyloid and mri 
for i=1:length(modality) %for loop to run the 
modal = modality{i};
%original: 
%cdir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped_' modal suffix];
%changed to:
cdir = ['/data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep 'brain_stripped_' modal suffix];
mkdir(cdir);
original: 
%system(['rsync -ar /data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep]);
%changed to: 
system(['rsync -ar /data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep]);
%original: 
%disp(['rsync -ar /data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep])
%changed to: 
disp(['rsync -ar /data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep 'brain_stripped' suffix filesep '*' modal '.nii ' cdir filesep])
end

end
%-----------------------------------------------------------------------
% Job saved on 22-Oct-2020 12:09:13 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function matlabbatch = batch_process_amyloidmri(rid, suffix) 
%coregistration steps: 
%you either have mri or pet as a template, doesn't matter which is the template, just that you coregister one with the other 
%mike thinks he used pet as template and coregistered mri to pet 
%get rigid cooridates from that 
%after coregistration you normalize the MRI and you get deformation field from that is a 4d array of splines 
%apply splines to mri to map it to the space
%also does a bias correction to get rid of low frequency inhomogeneties in mri 
%output is a normalized mri in the mni space 
%at that point, apply the deformation field to the pet scan 
%this brings pet scan into mri space, achieving coregistration 
%before pet scans are downloaded there have already been 4 processing steps performed in ADNI

%the reference image is the pet amyloid scan 
matlabbatch{1}.spm.spatial.coreg.estimate.ref = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_AMYLOID_nii_recenter' suffix '/' rid '_amyloid.nii,1']};
%the source image is the mri 
matlabbatch{1}.spm.spatial.coreg.estimate.source = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_amyloid' suffix '/' rid '_mri.nii,1']};
matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{2}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [0 1];
matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 0;
matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{2}.spm.spatial.preproc.warp.samp = 2;
matlabbatch{2}.spm.spatial.preproc.warp.write = [1 1];
matlabbatch{2}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{2}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
%mike put the pet scan as the standard space and transformed the mri to that and then applied ut 
matlabbatch{3}.spm.spatial.normalise.write.subj(1).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{3}.spm.spatial.normalise.write.subj(1).resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
%for subject 2 all he did was apply the forward transformations to it 
matlabbatch{3}.spm.spatial.normalise.write.subj(2).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
%then he resampled it
%resampling step, all the metadata gets scrambled
matlabbatch{3}.spm.spatial.normalise.write.subj(2).resample = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_AMYLOID_nii_recenter' suffix '/' rid '_amyloid.nii,1']};
matlabbatch{3}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                          NaN NaN NaN];
%this resampling gives it uniform voxels 
matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [1.5 1.5 1.5];
matlabbatch{3}.spm.spatial.normalise.write.woptions.interp = 4; %interpolate to get things into 1.5 voxels
matlabbatch{3}.spm.spatial.normalise.write.woptions.prefix = 'w';
%these next lines are doing calculations that mask everything 
%looks complicated because this is the vernacular that spm spits out 
matlabbatch{4}.spm.util.imcalc.input(1) = cfg_dep('Segment: wc1 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(2) = cfg_dep('Segment: wc2 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(3) = cfg_dep('Segment: wc3 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','wc', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.output = [rid '_brainmask_amyloidmri'];
matlabbatch{4}.spm.util.imcalc.outdir = {''};
matlabbatch{4}.spm.util.imcalc.expression = '(i1+i2+i3) > .1'; %gives you the probability of it being wm in this region
%i1+i2+i3 is adding the gm, wm, and csf masks and if at any voxel its >.1, keep it as a mask 
matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{4}.spm.util.imcalc.options.mask = 0;
matlabbatch{4}.spm.util.imcalc.options.interp = 1;
matlabbatch{4}.spm.util.imcalc.options.dtype = 768;
matlabbatch{5}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
matlabbatch{5}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_amyloidmri', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{5}.spm.util.imcalc.output = [rid '_brain_mri'];
matlabbatch{5}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
matlabbatch{5}.spm.util.imcalc.expression = 'i1.*i2'; %apply a mask to the mri 
matlabbatch{5}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{5}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{5}.spm.util.imcalc.options.mask = 0;
matlabbatch{5}.spm.util.imcalc.options.interp = 1;
matlabbatch{5}.spm.util.imcalc.options.dtype = 64;
matlabbatch{6}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 2)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{2}, '.','files'));
matlabbatch{6}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_amyloidmri', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.spm.util.imcalc.output = [rid '_brain_amyloid'];
matlabbatch{6}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
matlabbatch{6}.spm.util.imcalc.expression = 'i1.*i2'; %applys the mask to the pet scan
matlabbatch{6}.spm.util.imcalc.var = struct('name', {}, 'value', {}); 
matlabbatch{6}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{6}.spm.util.imcalc.options.mask = 0;
matlabbatch{6}.spm.util.imcalc.options.interp = 1;
matlabbatch{6}.spm.util.imcalc.options.dtype = 64;
end

%if you script it 
%do one, corgeister, then apply transformations to pet scan 

%ask if I need to uncomment these lines 
% function matlabbatch = batch_process_fdg(rid, suffix)
% matlabbatch{1}.spm.spatial.coreg.estimate.ref = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_FDG_nii_recenter' suffix '/' rid '_fdg.nii,1']};
% matlabbatch{1}.spm.spatial.coreg.estimate.source = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recenter_fdg' suffix '/' rid '_mri.nii,1']};
% matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
% matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
% matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
% matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
% matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
% matlabbatch{2}.spm.spatial.preproc.channel.write = [0 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
% matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
% matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
% matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
% matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
% matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
% matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
% matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
% matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 1];
% matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
% matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
% matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [0 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [1 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
% matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
% matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
% matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
% matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
% matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 0;
% matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
% matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
% matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
% matlabbatch{2}.spm.spatial.preproc.warp.samp = 2;
% matlabbatch{2}.spm.spatial.preproc.warp.write = [1 1];
% matlabbatch{2}.spm.spatial.preproc.warp.vox = NaN;
% matlabbatch{2}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
%                                               NaN NaN NaN];
% matlabbatch{3}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
% matlabbatch{3}.spm.spatial.normalise.write.subj.resample = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_FDG_nii_recenter' suffix '/' rid '_fdg.nii,1']};
% matlabbatch{3}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
%                                                           NaN NaN NaN];
% matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [1.5 1.5 1.5];
% matlabbatch{3}.spm.spatial.normalise.write.woptions.interp = 4;
% matlabbatch{3}.spm.spatial.normalise.write.woptions.prefix = 'w';
% matlabbatch{4}.spm.util.imcalc.input(1) = cfg_dep('Segment: wc1 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.input(2) = cfg_dep('Segment: wc2 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.input(3) = cfg_dep('Segment: wc3 Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','wc', '()',{':'}));
% matlabbatch{4}.spm.util.imcalc.output = [rid '_brainmask_fdg'];
% matlabbatch{4}.spm.util.imcalc.outdir = {''};
% matlabbatch{4}.spm.util.imcalc.expression = '(i1+i2+i3) > 0.1';
% matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{4}.spm.util.imcalc.options.mask = 0;
% matlabbatch{4}.spm.util.imcalc.options.interp = 1;
% matlabbatch{4}.spm.util.imcalc.options.dtype = 768;
% matlabbatch{5}.spm.util.imcalc.input(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
% matlabbatch{5}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: brainmask_fdg', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
% matlabbatch{5}.spm.util.imcalc.output = [rid '_brain_fdg'];
% matlabbatch{5}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped' suffix '/']};
% matlabbatch{5}.spm.util.imcalc.expression = 'i1.*i2';
% matlabbatch{5}.spm.util.imcalc.var = struct('name', {}, 'value', {});
% matlabbatch{5}.spm.util.imcalc.options.dmtx = 0;
% matlabbatch{5}.spm.util.imcalc.options.mask = 0;
% matlabbatch{5}.spm.util.imcalc.options.interp = 1;
% matlabbatch{5}.spm.util.imcalc.options.dtype = 64;
% end
