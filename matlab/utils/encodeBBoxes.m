function [boxesMinWH, boxesCenterWH] = encodeBBoxes(boxes)
%ENCODEBBOXES provides alternative boudning box encodings
% [BOXESMINWH, BOXESCENTERWH] = ENCODEBBOXES(BOXES)
% re-enocdes an N x 4 array of bounding boxes in the form
%   [XMIN YMIN XMAX YMAX] in two alternative formats:
%
%   Format 1. (XMIN, YMIN, WIDTH, HEIGHT): "MinWH"
%   Format 2. (CENTERX, CENTERY, WIDTH, HEIGHT): "CenterWH"
%
% BOXESMINWH and BOXESCENTERWH have the same dimensions as
% BOXES

boxWidths = boxes(:,3) - boxes(:,1) ;
boxHeights = boxes(:,4) - boxes(:,2) ;
boxesMinWH = [ boxes(:,[1 2]) boxWidths boxHeights ] ;

boxesCenterWH = [ (boxes(:,3) - boxes(:,1)) / 2 ...
                  (boxes(:,4) - boxes(:,2)) / 2 ...
                  boxWidths boxHeights ] ;
