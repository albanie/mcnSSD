function imdb = getPascalImdb(opts, varargin)
% LOADIMDB loads Pascal VOC image database
%
% Inspiration ancestry for code:
%   A.Vedaldi -> R.Girshick -> S.Albanie

opts.excludeDifficult = false ;
opts = vl_argparse(opts, varargin) ;

% Although the 2012 data can be used during training, only 
% the 2007 test data is used for evaluation
opts.VOCRoot = fullfile(opts.dataOpts.dataRoot, 'VOCdevkit2007' ) ;
opts.devkitCode = fullfile(opts.VOCRoot, 'VOCcode') ;

imdb = loadImdb(opts) ;

opts.pascalOpts = loadPascalOpts(opts) ;

% add meta information (inlcuding background class)
imdb.meta.classes = {'background' opts.pascalOpts.classes{:}} ;
classIds = 1:numel(imdb.meta.classes) ;  
imdb.classMap = containers.Map(imdb.meta.classes, classIds) ;
imdb.images.ext = 'jpg' ;
imdb.meta.sets = {'train', 'val', 'test'} ;

%-----------------------------------------
function pascalOpts = loadPascalOpts(opts) 
%-----------------------------------------
% LOADPASCALOPTS Load the pascal VOC database options
%
% NOTE: The Pascal VOC dataset has a number of directories 
% and attributes. The paths to these directories are 
% set using the VOCdevkit code. The VOCdevkit initialization 
% code assumes it is being run from the devkit root folder, 
% so we make a note of our current directory, change to the 
% devkit root, initialize the pascal options and then change
% back into our original directory 

% check the existence of the required folders
assert(logical(exist(opts.VOCRoot, 'dir')), 'VOC root directory not found') ;
assert(logical(exist(opts.devkitCode, 'dir')), 'devkit code not found') ;

currentDir = pwd ;
cd(opts.VOCRoot) ;
addpath(opts.devkitCode) ;

% VOCinit loads database options into a variable called VOCopts
VOCinit ; 

% Note - loads the options 
pascalOpts = VOCopts ;  
cd(currentDir) ;

%-----------------------------
function imdb = loadImdb(opts) 
%-----------------------------

cache07 = '/tmp/07.mat' ;
if ~exist(cache07, 'file') 
    dataDir07 = fullfile(opts.dataOpts.dataRoot, 'VOCdevkit2007', '2007') ;
    imdb07 = vocSetup('edition', '07', ...
                    'dataDir', dataDir07, ...
                    'archiveDir', dataDir07, ...
                    'includeDevkit', true, ...
                    'includeTest', true, ...
                    'includeDetection', true, ...
                    'includeSegmentation', false) ;
    save(cache07, '-struct', 'imdb07') ;
else
    imdb07 = load(cache07) ;
end

cache12 = '/tmp/12.mat' ;
if ~exist(cache12, 'file') 
    dataDir12 = fullfile(opts.dataOpts.dataRoot, 'VOCdevkit2012', '2012') ;
    imdb12 = vocSetup('edition', '12', ...
                    'dataDir', dataDir12, ...
                    'archiveDir', dataDir12, ...
                    'includeTest', false, ...
                    'includeDetection', true, ...
                    'includeSegmentation', false) ;
    save(cache12, '-struct', 'imdb12') ;
else
    imdb12 = load(cache12) ;
end

imdb = combineImdbs(imdb07, imdb12, opts) ;

% ------------------------------------------------
function imdb = combineImdbs(imdb07, imdb12, opts) 
% ------------------------------------------------

imdb.images.name = horzcat(imdb07.images.name, ...
                           imdb12.images.name) ;

imdb.images.set = horzcat(imdb07.images.set, ...
                          imdb12.images.set) ;

imdb.images.year = horzcat(ones(1,numel(imdb07.images.name)) * 2007, ...
                           ones(1,numel(imdb12.images.name)) * 2012) ;

imageSizes = horzcat(imdb07.images.size, imdb12.images.size ) ;

paths = vertcat(repmat(imdb07.paths.image, numel(imdb07.images.name), 1), ...
                 repmat(imdb12.paths.image, numel(imdb12.images.name), 1)) ;
imdb.images.paths = arrayfun(@(x) paths(x,:), 1:size(paths, 1),'Uni', 0) ;

% for consistency, store in Height-Width order
imdb.images.imageSizes = arrayfun(@(x) imageSizes([2 1],x)', ...
                                    1:size(imageSizes, 2), 'Uni', 0) ;

annot07 = loadAnnotations(imdb07, opts) ;
annot12 = loadAnnotations(imdb12, opts) ;
imdb.annotations = horzcat(annot07, annot12) ;


% ------------------------------------------------
function annotations = loadAnnotations(imdb, opts) 
% ------------------------------------------------

annotations = cell(1, numel(imdb.images.name)) ;
for i = 1:numel(imdb.images.name)
    match = find(imdb.objects.image == i) ;

    % normalize annotation
    if opts.excludeDifficult
        keep = ~[imdb.objects.difficult(match)] ;
    else
        keep = 1:numel(match) ;
    end

    match = match(keep) ;
    boxes = imdb.objects.box(:,match) ;
    classes = imdb.objects.class(match) ;

    % normalize annotation
    imSize = repmat(imdb.images.size(:, i)', [1 2]) ;
    gt.boxes = bsxfun(@rdivide, boxes', single(imSize)) ;
    gt.classes = classes + 1 ; % add offset for background

    assert(all(2 <= gt.classes) && all(gt.classes <= 21), ...
                 'pascal class labels do not lie in the expected range') ;
    annotations{i} = gt ;
    fprintf('Loading annotaton %d \n', i) ;
end
