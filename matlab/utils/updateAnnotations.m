function [success, targets, labels] = updateAnnotations(patch, targetsWH, labels, opts) 
%UPDATEANNOTATIONS - update annotation coordinates to match patch
%   UPDATEANNOTATIONS(PATCH, TARGETSWH, LABELS, OPTS) updates the coordinate
%   frame of reference for a given set of annotations to match a PATCH.
%   TARGETSWH is an N x 4 array of bounding box locations described in the 
%   [0,1] coordinate system representing the entire input image dimensions.  
%
%   The goal is to rescale these coordinates so that they are given relative
%   to PATCH, a single box (also described in [0,1]) coordinates.  As a result
%   when the region specified by PATCH is cropped from the image, the updated 
%   annotation should then be a valid [0,1] coordinate description for the 
%   patch.
%
%   If the centre of the annotation does not lie inside the patch, it is 
%   dropped, along with its corresponding LABELS.  
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

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
    targets = [] ; success = false ; return
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
  xScale = 1 / patchWH(3) ; yScale = 1 / patchWH(4) ;
  xmin = xScale * (targetsWH(:,1) - patchWH(1)) ;
  xmax = xScale * (targetsWH(:,1) + targetsWH(:,3) - patchWH(1)) ;
  ymin = yScale * (targetsWH(:,2) - patchWH(2)) ;
  ymax = yScale * (targetsWH(:,2) + targetsWH(:,4) - patchWH(2)) ;
  targets = [xmin ymin xmax ymax ] ;
  if opts.clipTargets, targets = min(max(targets, 0), 1) ; end
