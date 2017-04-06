function varargout = pruneCheckpoints(expDir, varargin)
%PRUNECHECKPOINTS removes unnecessary checkpoint files
%   PRUNECHECKPOINTS(EXPDIR) evaluates the checkpoints
%   (the `net-epoch-%d.mat` files created during
%   training) in EXPDIR and removes all checkpoints except:
%
%       1. The checkpoint with the lowest validation error metric
%       2. The last checkpoint
%
%   If an output argument is provided, PRUNECHECKPOINTS returns the 
%   epoch with the lowest validation error.
%
%   PRUNECHECKPOINTS(..., 'option', value, ...) accepts the following
%   options:
%
%   `priorityMetric`:: 'classError'
%    Determines the highest priority metric by which to rank the 
%    checkpoints for pruning.

opts.priorityMetric = 'classError' ;
opts = vl_argparse(opts, varargin) ;

lastEpoch = findLastCheckpoint(expDir);

% return if no checkpoints were found
if ~lastEpoch
    return
end

bestEpoch = findBestCheckpoint(expDir, opts.priorityMetric);
preciousEpochs = [bestEpoch lastEpoch];
removeOtherCheckpoints(expDir, preciousEpochs);
fprintf('----------------------- \n');
fprintf('%s directory cleaned: \n', expDir);
fprintf('----------------------- \n');

if nargout == 1
    varargout{1} = bestEpoch ;
end

% -------------------------------------------------------------------------
function removeOtherCheckpoints(expDir, preciousEpochs)
% -------------------------------------------------------------------------
list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epochs = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
targets = ~ismember(epochs, preciousEpochs);
files = cellfun(@(x) fullfile(expDir, sprintf('net-epoch-%d.mat', x)), ...
        num2cell(epochs(targets)), 'UniformOutput', false);
cellfun(@(x) delete(x), files)
