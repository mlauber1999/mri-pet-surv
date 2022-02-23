function process_files_cox_rigid(suffix, reg_exp, test)
addpath(genpath('/home/mfromano/spm/spm12/'));
BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
if nargin > 2 && test
    MRI_folder_old = [BASE_DIR 'ADNI_MRI_NACC_recenter' suffix filesep];
else
    MRI_folder_old = [BASE_DIR 'ADNI_MRI_nii_recenter' suffix filesep];
end

disp(MRI_folder_old)
if nargin < 2
    reg_exp = '^([0-9]{4}).*\.nii$';
end

fnames = dir(MRI_folder_old);
fnames = {fnames.name};
disp(fnames)
rids = arrayfun(@(x) regexp(x,reg_exp,'tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids = [rids{:}];
brain_stripped_dir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'brain_stripped_rigid' suffix];
new_mri_dir = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep 'mri_processed_rigid' suffix];
mkdir(new_mri_dir)
mkdir(brain_stripped_dir);
rand('state',10);
maxNumCompThreads = 1;
spm('defaults', 'PET');
spm_jobman('initcfg');
parpool(10);
cat12('expert')
parfor i=1:length(rids)
    rand('state',10);
    system(['rsync ' MRI_folder_old rids{i} '.nii ' new_mri_dir filesep]);
    jobs = batch_process_mri(rids{i}, suffix, new_mri_dir);
    try
        spm_jobman('run', jobs);
    catch
        disp(['Job failed for ' rids{i} '!']);
    end
end
new_coreg_dir = [new_mri_dir filesep 'coregistered_brains'];
mkdir(new_coreg_dir)
disp(['rsync -av ' new_mri_dir filesep 'coregistered*.nii ' new_coreg_dir filesep])
system(['rsync -av ' new_mri_dir filesep 'coregistered*.nii ' new_coreg_dir filesep]);

end
%-----------------------------------------------------------------------
% Job saved on 22-Oct-2020 12:09:13 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

function matlabbatch = batch_process_mri(rid, suffix, old_dir)
%-----------------------------------------------------------------------
% Job saved on 01-May-2021 18:29:22 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.preproc.channel.vols = {[old_dir '/' rid '.nii,1']};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,1'};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,2'};
matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,3'};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,4'};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,5'};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/spm/spm12/tpm/TPM.nii,6'};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{2}.spm.util.imcalc.input(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
matlabbatch{2}.spm.util.imcalc.input(2) = cfg_dep('Segment: c1 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','c', '()',{':'}));
matlabbatch{2}.spm.util.imcalc.input(3) = cfg_dep('Segment: c2 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','c', '()',{':'}));
matlabbatch{2}.spm.util.imcalc.input(4) = cfg_dep('Segment: c3 Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','c', '()',{':'}));
matlabbatch{2}.spm.util.imcalc.output = [rid '_native_masked.nii'];
matlabbatch{2}.spm.util.imcalc.outdir = {['/data2/MRI_PET_DATA/processed_images_final' suffix '/brain_stripped_rigid' suffix '/']};
matlabbatch{2}.spm.util.imcalc.expression = '((i2+i3+i4) > .2).*i1';
matlabbatch{2}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{2}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{2}.spm.util.imcalc.options.mask = 0;
matlabbatch{2}.spm.util.imcalc.options.interp = 1;
matlabbatch{2}.spm.util.imcalc.options.dtype = 64;
matlabbatch{3}.spm.spatial.coreg.estwrite.ref = {'/usr/local/spm/spm12/canonical/average305_t1_tal_lin_mask.nii,1'};
matlabbatch{3}.spm.spatial.coreg.estwrite.source(1) = cfg_dep(['Image Calculator: ImCalc Computed Image: ' rid '_native_masked.nii'], substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{3}.spm.spatial.coreg.estwrite.other = {''};
matlabbatch{3}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{3}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{3}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{3}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.prefix = 'coregistered_';
end