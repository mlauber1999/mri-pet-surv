%script to move niftis to a new folder once they are realigned
function rids = realign_all_niftis(outputfolder, suffix)
%defining a function named realign_all_nifits with input parameters outputfolder and suffix and output parameter rids 
%maybe rids might be realigned ids 
%the input is the rIDs, but where are they stored? Do I have to be in the same folder as them 

%% main
if nargin < 1
%nargin is number of function input arguments 
%if the number of inputs is less than one, the output folder is /data2/....
    
    %original: 
    %outputfolder = '/data2/MRI_PET_DATA/processed_images_final';
    % changed to: 
    outputfolder = '/data2/MRI_PET_DATA/ML/processed_images_final';
    %creating a folder or just specifying which one to do? Ask mike 
end
%if nargin is less than 2, add suffix '', meaning defining an empty string so that when you call the function and specify a suffix, that gets added 
%second condition usually over rides first 
%
if nargin < 2
    suffix = '';
end

%defining variable of initialized mri folder, that is equal to path /data2... 
%in matlab comma and a space mean the same thing in brackets 
%3 parameters in this brackets, the string filename, the suffix, 

%original:
%MRI_folder_init = ['/data2/MRI_PET_DATA/raw_data/MRI_nii' suffix filesep]; %goes through this folder 
% changed to: 
MRI_folder_init = ['/data2/MRI_PET_DATA/ML/raw_data/MRI_nii' suffix filesep];

fnames = dir(MRI_folder_init); 
%defining variable fnames as all files in that directory 
%when you use dir you get .name for name of the file 
fnames = {fnames.name};
%redefine fnames as the name of each file in folder 
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
%arrayfun tells it to apply the function to each element in the array
%regex tells it to match x with the specified conditions (0-9), 4 digits long, ending in .nii, fnames is the names of all the files within the directory mri_folder_init  

rids = [rids{:}];
%creating a matrix of the rids, taking specific data types {:} 
%rids variable is now equal to all of the rids 
%{:}  means take all contents of a cell array 
rids = [rids{:}];
rids_mri = [rids{:}];
%assigning var named rids_mri to all the rids 
%renaming rids to rids_mri

%everytime you run rids=[rids{:} its slightly changing it? that's why you have to keep redefining it 

rids = rids_mri;

%would I change this too? Or is it good as is? 
MRI_folder_new = [outputfolder '/ADNI_MRI_nii_recenter' suffix];
%if im in my ML folder, don't need to respecify path name since it will create a new folder within ML by running this, but if I'm running it from the server I do need to give the absolute path 

% moves the niftis to this new /data2/MRI_PET_DATA/processed_images_final_2yr/ADNI_MRI_nii_recenter folder after realigning them 


if ~exist(outputfolder,'dir')
%conditional, if conditons are met in the output folder, make a new folder with the name of the value of output folder 
%~exist will return a boolean form of the matrix its called against, the result matrix is 1 for 0 in the og matrix and 0 otherwise 
%matlab converst boolean t/f to 1/0 
%if it doesnt exist, create a directory? 
    mkdir(outputfolder)
end
if ~exist(MRI_folder_new,'dir') %if mri_folder_new dir doesnt exist then make one 
%'dir' is the suffix defined earlier in the script 
    mkdir(MRI_folder_new);
end

realign_nifti(MRI_folder_init, MRI_folder_new, rids, 'mri');
%this is calling the function realign_nifti from another script 
%initiates the second step in the pipeline 
