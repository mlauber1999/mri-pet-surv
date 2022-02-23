%-----------------------------------------------------------------------
% Job saved on 31-Mar-2021 16:27:45 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

function apply_mask_and_warp_job(nyrs)
addpath(genpath('/home/mfromano/spm/'));

% create folders
BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' nyrs '/'];
MRI_folder_init = [BASE_DIR 'ADNI_MRI_nii_recentered_cat12' nyrs '/'];
PET_folder_init = [BASE_DIR 'ADNI_AMYLOID_nii_recenter' nyrs '/'];

% get file names for raw PET scans
fnames = dir(PET_folder_init);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_pet = [rids{:}];

fnames = dir(MRI_folder_init);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_mri = [rids{:}];

% get rids in both folders
rids = intersect(rids_pet, rids_mri);

outdir = [BASE_DIR '/brain_stripped' nyrs '/'];
outdir_atlas = [BASE_DIR '/brain_stripped_atlas' nyrs '/'];
mkdir(outdir);
mkdir(outdir_atlas);

rand('state',10);
spm('defaults', 'PET');
spm_jobman('initcfg');
for i=1:numel(rids)
    batch = apply_skull_strip(rids{i}, PET_folder_init, MRI_folder_init, outdir);
    spm_jobman('run_nogui', batch);
end
system(['rsync ' MRI_folder_init 'mri/wp0*_mri.nii ' outdir])
system(['rsync ' MRI_folder_init 'mri_atlas/wneuromorphometrics_*_mri.nii ' outdir_atlas])
end

function matlabbatch = apply_skull_strip(rid, amyloid_folder, mri_folder, outdir)
matlabbatch{1}.spm.tools.cat.tools.defs.field1 = {[mri_folder 'mri/y_' rid '_mri.nii,1']};
matlabbatch{1}.spm.tools.cat.tools.defs.images = {[amyloid_folder rid '_amyloid.nii,1']};
matlabbatch{1}.spm.tools.cat.tools.defs.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{1}.spm.tools.cat.tools.defs.vox = [NaN NaN NaN];
matlabbatch{1}.spm.tools.cat.tools.defs.interp = 1;
matlabbatch{1}.spm.tools.cat.tools.defs.modulate = 0;
matlabbatch{2}.spm.util.imcalc.input = {[mri_folder '/mri/wp0' rid '_mri.nii,1']};
matlabbatch{2}.spm.util.imcalc.output = 'MRI_MASK';
matlabbatch{2}.spm.util.imcalc.outdir = {''};
matlabbatch{2}.spm.util.imcalc.expression = 'i1 > 0.001';
matlabbatch{2}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{2}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{2}.spm.util.imcalc.options.mask = 0;
matlabbatch{2}.spm.util.imcalc.options.interp = 1;
matlabbatch{2}.spm.util.imcalc.options.dtype = 4;
matlabbatch{3}.spm.util.imcalc.input(1) = cfg_dep('Apply deformations (many images): All Output Files', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','vfiles'));
matlabbatch{3}.spm.util.imcalc.input(2) = cfg_dep('Image Calculator: ImCalc Computed Image: MRI_MASK', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{3}.spm.util.imcalc.output = ['masked_amyloid_' rid];
matlabbatch{3}.spm.util.imcalc.outdir = {outdir};
matlabbatch{3}.spm.util.imcalc.expression = 'i1.*i2';
matlabbatch{3}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{3}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{3}.spm.util.imcalc.options.mask = 0;
matlabbatch{3}.spm.util.imcalc.options.interp = 1;
matlabbatch{3}.spm.util.imcalc.options.dtype = 4;
end