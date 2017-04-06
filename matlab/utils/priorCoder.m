function y = priorCoder(x, pBoxes, pVar, to) 
%PRIORCODER encodes/decodes bboxes relative to a set of priors 
% Y = PRIORCODER(X, PBOXES, PVAR, TO) encodes/decodes
% a set of bounding box values to/from the `CenWH` and `targets`
% formats described below, relative to a set of prior boxes and
% prior variances.
%
%   INPUTS:
%   `X` is an array of dimensions H x 4 x 1 x N, while `PBOXES` 
%   and `PVAR` have dimensions H x 4 x 1 where H is the number 
%   of boxes and N is the batch size. 
%   The `PBOXES` are encoded as [P_CX P_CY P_WIDTH P_HEIGHT]
%   representing the center X,Y values, width and height of the
%   prior boxes.  The `PVAR` are scaling factors that are 
%   applied to each prior box. 
%   `TO` is a string denoting the direction of the coding. It can
%   take one of two values - `CenWH` or `targets`. The 
%   interpretation of X depends on the coding formats, described 
%   below.
%
%   'CenWH": [CX CY WIDTH HEIGHT] x 1 x N
%    X is an H x 4 x 1 x N array whose columns denote the X,Y
%    coordinates of the centers of each box, its 
%    width and its height. H is the number of bounding boxes 
%    and N is the batch size.
%
%   'targets' [TX TY TW TH] x 1 x N
%    X is an H x 4 x 1 x N array of values which are defined 
%    with repsect to the `CenWH` values, the `pBoxes` 
%    and the `priorVars` as folows:
%       TX = (CX - P_CX) / P_WIDTH * P_VAR_X
%       TY = (CY - P_CY) / P_HEIGHT * P_VAR_Y
%       TW = log(WIDTH / P_WIDTH)  / P_VAR_WIDTH
%       TH = log(HEIGHT / P_HEIGHT) / P_VAR_HEIGHT
%
% The target encoding follows the approach described for bounding 
% box regression described in the paper:
%
% "Rich feature hierarchies for accurate object detection and 
% semantic segmentation" (Girshick et al, 2014)
%
% Example:
%   
%    Consider a single input bbox (H = N = 1):
%   
%    x = [ 0.25 0.25 0.5 0.5 ] 
%    pBox = [ 0.3 0.3 0.4 0.4 ] 
%    pVar = [ 0.1 0.1 0.2 0.2 ] 
%    
%    Then the function call
%      y = priorCoder(x, pBox, pVar, 'targets') 
%   will produce the target enocoded output:
%      [            ]

switch to
    case 'CenWH'
        XY = bsxfun(@plus, bsxfun(@times, x(:,1:2,:,:),  ...
                          pBoxes(:, 3:4) .* pVar(:,1:2)), ...
                          pBoxes(:,1:2)) ;
        WH = bsxfun(@times, exp(bsxfun(@times, x(:, 3:4,:,:), pVar(:, 3:4))), ...
                    pBoxes(:,3:4)) ;
    case 'targets'
        XY = (x(:, 1:2) - pBoxes(:,1:2)) ./ (pBoxes(:, 3:4) .* pVar(:, 1:2)) ;
        WH = log(x(:, 3:4) ./ pBoxes(:,3:4)) ./ pVar(:,3:4) ;
    otherwise
        fprintf('coder `to` target %s not recognised', to) ;
end

y = [ XY WH ] ;
