function printPascalResults(cacheDir, varargin)
%PRINTPASCALRESULTS prints out results as formatted tables
%   PRINTRESULTS(CACHEDIR) searches the cache directory of 
%   pascal VOC evaluations and prints out a formatted summary
%   CACHEDIR is a string specifying the absolute path of the 
%   directory holding the cached results
%
%   PRINTPASCALRESULTS(..., 'option', value, ...) takes 
%   the following options:
%   
%   `orientation`:: 'portrait'
%     The orientation in which the table is printed. Can be 
%     either 'landscape' or 'portrait'. Landscape is useful 
%     when you have a wide monitor.
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.orientation = 'portrait' ;
opts = vl_argparse(opts, varargin) ;

[valResList, testResList] = getResultsList(cacheDir) ;
valResults = cellfun(@(x) getResult(x), valResList, 'Uni', false) ;                     
testResults = cellfun(@(x) getResult(x), testResList, 'Uni', false) ;                     
valResults = cell2mat(valResults) ;
testResults = cell2mat(testResults) ;

if ~isempty(valResults)
    fprintf('\nVal Set Results:\n') ;
    prettyPrintTable(valResults, opts) ;
end

if ~isempty(testResults)
    fprintf('\nTest Set Results:\n') ;
    prettyPrintTable(testResults, opts) ;
end

% -------------------------------------------------------------------------
function [valList, testList] = getResultsList(cacheDir)
% -------------------------------------------------------------------------

files = ignoreSystemFiles(dir(fullfile(cacheDir, '*.mat'))) ;
names = {files.name} ;
[penSuffixes, suffixes] = cellfun(@(x) getSuffixes(x), names, 'Uni', false) ;
valList = fullfile(cacheDir, names(strcmp(suffixes, 'results') & ...
                                        strcmp(penSuffixes, 'val'))) ;
testList = fullfile(cacheDir, names(strcmp(suffixes, 'results') & ...
                                        strcmp(penSuffixes, 'test'))) ;

% -------------------------------------------------------------------------
function [penSuffix, suffix] = getSuffixes(filename) 
% -------------------------------------------------------------------------

[~,filename,~] = fileparts(filename) ;
tokens = strsplit(filename, '-') ;
penSuffix = tokens{end -1} ;
suffix = tokens{end} ;

% -------------------------------------------------------------------------
function result = getResult(resultFile)
% -------------------------------------------------------------------------

[~,fname,~] = fileparts(resultFile) ;
tokens = strsplit(fname,'-') ;
model = strjoin(tokens(1:end - 2), '-') ;
result.model= model;
result.subset = tokens{end - 1} ;
data = load(resultFile) ;
aps = data.results * 100;
pascalClasses = {'aeroplane', 'bicycle', 'bird', 'boat', ...
                 'bottle', 'bus', 'car', 'cat', 'chair', ...
                 'cow', 'diningtable', 'dog', 'horse', ...
                 'motorbike', 'person', 'pottedplant', ...
                 'sheep', 'sofa', 'train', 'tvmonitor'} ;
for i = 1:numel(pascalClasses)
    f = pascalClasses{i} ;
    result.(f) = aps(i) ;
end
result.mean = mean(aps) ;

%---------------------------------------
function prettyPrintTable(results, opts)
%---------------------------------------

switch opts.orientation
    case 'portrait'
        table = struct2table(results);
        numCols = size(table, 1) ;

        % format the column widths 
        leftCol = 14 ;
        if ischar(table.model)
            table.model = {table.model} ;
        end
        dataCol = max(cellfun(@length, table.model)) ;

        % print the table headers
        columnWidths = horzcat(leftCol, repmat(dataCol, 1, numCols)) ;
        dividerFormatter = strcat(repmat('%s ', 1, numCols + 1), '\n') ;
        dividerStrings = arrayfun(@(x) repmat('-', 1, x), columnWidths, ...
                                        'Uni', false) ;
        fprintf(dividerFormatter, dividerStrings{:}) ;

        headerFormatter = strcat(sprintf('%%-%ds', leftCol), ...
                                 repmat(sprintf(' %%-%ds', dataCol), ...
                                 1, numCols), '\n') ;
        headerStrings = horzcat(' ', table.model') ;
        fprintf(headerFormatter, headerStrings{:}) ; 
        fprintf(dividerFormatter, dividerStrings{:}) ;

        % print the main table
        categories = setdiff(fieldnames(table), ...
                            {'Properties', 'subset', 'model'}) ;
        dataFormatter = strcat(sprintf('%%-%ds', leftCol), ...
                                 repmat(sprintf(' %%-%d.2f', dataCol), ...
                                 1, numCols), '\n') ;
        for i = 1:numel(categories);
            c = categories{i} ;
            ap = table.(c) ;
            fprintf(dataFormatter, c, ap(:)) ;
        end
        fprintf(dividerFormatter, dividerStrings{:}) ;
        ap = table.('mean') ;
        fprintf(dataFormatter, 'mean', ap(:)) ;
        fprintf(dividerFormatter, dividerStrings{:}) ;

    case 'landscape'
        table = struct2table(results);
        disp(table) ;

    otherwise
        error('table orientation must be either landscape or portrait') ;
end
