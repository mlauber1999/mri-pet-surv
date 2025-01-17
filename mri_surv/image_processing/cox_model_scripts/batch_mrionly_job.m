%this allows you to batch process mnris to segment them into 95 regions 
%mike looked at gray matter volumes, which corresponded to each brain region 
function batch_mrionly_job(suffix)
    addpath(genpath('/home/mfromano/spm/spm12/'));
    %do I need to change this? Or can I call spm from Mike's folder 
    %original: 
    %BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' suffix filesep];
    %changed to: 
    BASE_DIR = ['/data2/MRI_PET_DATA/ML/processed_images_final' suffix filesep];
    MRI_folder_new = [BASE_DIR 'ADNI_MRI_nii_recenter' suffix filesep];

    fnames = dir(MRI_folder_new);
    fnames = {fnames.name};
    maxNumCompThreads = 1;
    spm('defaults', 'PET'); 
    spm_jobman('initcfg');
    rand('state',10);
    modal = 'mri';
    %original: 
    %newdir = ['/data2/MRI_PET_DATA/processed_images_final' suffix '/ADNI_MRI_nii_recentered_cat12' suffix];
    %changed this to:
    newdir = ['/data2/MRI_PET_DATA/ML/processed_images_final' suffix '/ADNI_MRI_nii_recentered_cat12' suffix];
    
    mkdir(newdir); %makes new directory to put the mris in 
    system(['rsync -av ' MRI_folder_new  '*' modal '.nii ' newdir filesep]); %moves MRIs you just recentered into a new folder 
    disp(['rsync -av ' MRI_folder_new  '*' modal '.nii ' newdir filesep])
    %takes a list of mris and processes them in optimized chunks 
    %this is in cat12, takes the raw images, recenters them first and then runs them through cat12
    %most of these are the default options, only ones you might want to adjust are the ROIs
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {newdir};
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '^[0-9]{4}_mri.nii$';
    matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    matlabbatch{2}.spm.tools.cat.estwrite.data(1) = cfg_dep('File Selector (Batch Mode): Selected Files (^0974_mri.nii$)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{2}.spm.tools.cat.estwrite.data_wmh = {''};
    matlabbatch{2}.spm.tools.cat.estwrite.nproc = 5;
    matlabbatch{2}.spm.tools.cat.estwrite.useprior = '';
    matlabbatch{2}.spm.tools.cat.estwrite.opts.tpm = {'/usr/local/spm/spm12/tpm/TPM.nii'};
    matlabbatch{2}.spm.tools.cat.estwrite.opts.affreg = 'mni';
    matlabbatch{2}.spm.tools.cat.estwrite.opts.biasstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.opts.accstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.NCstr = -Inf;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.spm_kamap = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.BVCstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.WMHC = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.SLC = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.mrf = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.segmentation.restypes.optimal = [1 0.1];
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {'/usr/local/spm/spm12/toolbox/cat12/templates_volumes/Template_0_IXI555_MNI152_GS.nii'};
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.registration.shooting.regstr = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.vox = 1.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtres = 0.5;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtmethod = 'pbt2x';
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.pbtlas = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.collcorr = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.reduce_mesh = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.vdist = 1.33333333333333;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.scale_cortex = 0.7;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.add_parahipp = 0.1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.surface.close_parahipp = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.experimental = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.new_release = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.lazy = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.verb = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.extopts.admin.print = 2;
    matlabbatch{2}.spm.tools.cat.estwrite.output.surface = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.surf_measures = 1; %surface atlas, not as good as freesurfer but runs faster 
    %can play around with different ROIs
    %the lines represent different atlases you can use 
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 1; %used this one, because its set to 1 not 0 
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 1; %this ones good 
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.ibsr = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.aal3 = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.mori = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.anatomy = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.julichbrain = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_100Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_200Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_400Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_600Parcels_17Networks_order = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {''};
    %mike's abstract looked at GM, but could look at others depending on question asking 
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.native = 0; %native space is the intial space the image was in 
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.warped = 1; %it ouputs GM, .warped means it's been registered to the MNI space, ie warped 
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.GM.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.warped = 1; %it ouputs WM 
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WM.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.native = 0; 
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.warped = 1; %it ouputs CSF 
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.CSF.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.ct.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.pp.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.native = 0; 
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.WMH.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.SL.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.warped = 1; %set to 1 if you want it, 0 if not 
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.mod = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.TPMC.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.atlas.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.native = 1; % .label represents the labels for the atlases 
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.label.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.labelnative = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.native = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.warped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.bias.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.native = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.warped = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.las.dartel = 0;
    matlabbatch{2}.spm.tools.cat.estwrite.output.jacobianwarped = 1;
    matlabbatch{2}.spm.tools.cat.estwrite.output.warps = [1 1];
    matlabbatch{2}.spm.tools.cat.estwrite.output.rmat = 1;
    cat12('expert') %expert mode allows you to specify extra things you can use 
    jobs = matlabbatch; %specifies its a matlab batch job 
    spm('defaults', 'PET');
    spm_jobman('initcfg'); %automatically populates 
    spm_jobman('run_nogui', jobs); %tells it to run without a gui 
end
