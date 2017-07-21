function [success, targets, labels] = updateAnnotations(patch, targetsWH, labels, opts) 
% TODO: docs

% keep gt annotations with centres that still lie in the patch
targetCenters = [ targetsWH(:,1) + targetsWH(:,3) / 2 ...
                   targetsWH(:,2) + targetsWH(:,4) / 2 ] ; 
                   
retain = patch(1) <= targetCenters(:,1) & ...
           targetCenters(:,1) <= patch(3) & ...
           patch(2) <= targetCenters(:,2) & ...
           targetCenters(:,2) <= patch(4) ;

% Patches for which there are no gt boxes whose center lies in the 
% patch are replaced with the original image. (This is slightly 
% different to in caffe, where the patch is passed with no gt boxes)
if ~any(retain)
    %patch = [ 0 0 1 1 ] ;
    %targets = bboxCoder(targetsWH, 'MinWH', 'MinMax') ;
    targets = [] ;
    success = false ;
    return
else
    success = true ;
end

% Patches for which  only a portion of the annotations' centers no 
% longer lie in the image are retained, but the annotations that 
% are outside are removed from the labels.   
if any(~retain)
    targetsWH(~retain,:) = [] ;
    labels(~retain) = [] ;
end

patchWH = bboxCoder(patch, 'MinMax', 'MinWH') ;

xScale = 1 / patchWH(3) ;
yScale = 1 / patchWH(4) ;

xmin = xScale * (targetsWH(:,1) - patchWH(1)) ;
xmax = xScale * (targetsWH(:,1) + targetsWH(:,3) - patchWH(1)) ;
ymin = yScale * (targetsWH(:,2) - patchWH(2)) ;
ymax = yScale * (targetsWH(:,2) + targetsWH(:,4) - patchWH(2)) ;

targets = [xmin ymin xmax ymax ] ;

if opts.clipTargets
  targets = min(max(targets, 0), 1) ;
end
