function VOCopts = configureVOC(expDir, VOCRoot, testset) 
% configure the options required for VOC evaluations

dataDir = fullfile(VOCRoot,  'VOC2007') ;
devkitCode = fullfile(VOCRoot, 'VOCcode') ;

%if ~exist(VOCRoot, 'dir')
    %fprintf('VOC 2007 devkit not found at %s\n', VOCRoot) ;
    %res = input('Would you like to download it? y/n (n)\n', 's') ;
    %if strcmp(res, 'y')
        %fprintf('Downloading to %s\n', VOCRoot) ;
        %vocSetup('dataDir', VOCRoot, 'edition', '07') ;
    %else
        %error('Not downloading. Exiting..\n') ;
    %end
%end

addpath(devkitCode) ;

VOCinit ;

% override some of the defaults 
VOCopts.localdir = fullfile(VOCRoot, 'local/VOC2007') ;
VOCopts.testset = testset ;
VOCopts.datadir = dataDir ;
VOCopts.imgsetpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.imgpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.annopath = fullfile(dataDir, 'Annotations/%s.xml') ;
VOCopts.cacheDir = fullfile(expDir, 'VOC2007/Results/Cache') ;
VOCopts.devkitCode = devkitCode ;
VOCopts.drawAPCurve = false ;

% create detection directory if required
detDir = fullfile(expDir, 'VOCdetections') ;
if ~exist(detDir, 'dir') 
    mkdir(detDir) ;
end
VOCopts.detrespath = fullfile(detDir, sprintf('%%s_det_%s_%%s.txt', testset)) ;
