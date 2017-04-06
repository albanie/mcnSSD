function files = ignoreSystemFiles(files)
% IGNORESYSTEMFILES removes system files
%   IGNORESYSTEMFILES removes files produced by the operating 
%   system from the cell array of file names contained in fileNames

systemFiles = {'.', '..', '.DS_Store'} ;
ignoreIdx = cellfun(@(x) ismember(x, systemFiles), {files.name}, ...
                                        'UniformOutput', true) ;
                                        files(ignoreIdx) = [] ;
