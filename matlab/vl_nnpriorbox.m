function y = vl_nnpriorbox(x, im, varargin)
%VL_NNPRIORBOX constructs prior boxes for a feature map.  
%   Y = VL_NNPRIORBOX(X, IM) produces a set of prior boxes as defined 
%   in the SSD paper.  The core idea is to generate a set of evenly
%   spaced boxes which tile the feature layer. At each position in the
%   tiling, boxes are created with multiple aspect ratios and sizes, where
%   X is the feature layer used to determine size and number of the 
%   priorboxes and IM is the original input image data.
%
%   The output Y is an array of prior boxes in which each prior box 
%   is stored as two columns of four numbers.  The first column is 
%   enocded as (xmin, ymin, xmax, ymax) where each number indicates 
%   a position relative to the original input image shape 
%   (e.g. (0,0,1,1) would define a box covering the entire
%   input image).
%
%   The second column contains the "variances" of these boxes - this is
%   way of efficiently storing boxes from the same location at multiple
%   sizes by providing the scalings that should applied to the coordinates
%   of the box: (xmin_variance, ymin_variance, xmax_variance, ymax_variance)
%   Each prior box corner will be decoded as:
%
%    xmin = xmin + xmin * xmin_variance
%    ...
%
%   VL_NNPRIORBOX does not support a backward pass.
%
%   VL_NNPRIORBOX(...,'OPT',VALUE,...) takes the following options:
%
%   `aspectRatios`:: [2]
%    The set of aspect ratios used to create each prior box. By default,
%    an extra prior box is also created with an aspect ratio of 1, so
%    for example, if `aspectRatios = [2 3]`, then prior boxes will be
%    created with aspect ratios 1, 2 and 3. 
%
%   `flip`:: true
%    If true, two prior boxes will be created for each aspect ratio. 
%    For instance, if the aspect ratios are [2, 3], then with `flip`
%    enabled, prior boxes will also be created with aspect ratios 
%    [1/2, 1/3].  
%
%   `minSize`:: 0.1
%    The smallest size used to create each prior box.
%
%   `maxSize`:: 0.2
%    The largest size used to create each prior box .
%
%   `offset`:: 0.5
%    The offset applied to each feature location to "centre" the boxes 
%
%   `variance`:: [0.1 0.1 0.2 0.2]
%     A 4x1 array used to scale prior boxes
%
%   `clip`:: false
%    Determines whether or not prior boxes overlapping the edge 
%    boundary of the feature map should be clipped to lie within it
%
%   `pixelStep`:: 1
%    Dictates how many pixels in the input image, IM correspond 
%    to a single pixel in the feature layer X

  opts.aspectRatios = 2 ;
  opts.flip = true ;
  opts.clip = false ;
  opts.offset = 0.5 ;
  opts.minSize = 0.1 ;
  opts.maxSize = 0.2 ;
  opts.pixelStep = 1 ;
  opts.variance = [0.1 0.1 0.2 0.2] ;

  opts = vl_argparse(opts, varargin, 'nonrecursive') ;

  % Each spatial element of the input layer `im` produces a corresponding
  % prior box in the input image. We assume that every image in the
  % batch is the same size so that the prior boxes can be duplicated
  % across all input images. 
  layerWidth = size(x, 2) ;
  layerHeight = size(x, 1) ;
  imgWidth = size(im, 2) ;
  imgHeight = size(im, 1) ;

  % There is one prior box by default, which has apsect ratio 1.
  % If flipping is enabled, each addtional aspect ratio generates
  % two extra prior boxes.
  aspectRatios = opts.aspectRatios ;
  if opts.flip
      aspectRatios = cat(1, aspectRatios, 1 ./ aspectRatios) ;
  end
  numAspectRatios = 1 + numel(aspectRatios) ;

  % An additional box is generated if the maxSize property is specified
  boxesPerPosition = numAspectRatios + logical(opts.maxSize) ;
  numBoxes = layerWidth * layerHeight * boxesPerPosition ; 
  boxes = zeros(numBoxes * 4, 1) ;

  % maintain compatibility
  if opts.pixelStep == 0
    opts.pixelStep = imgWidth / layerWidth ;
    heightStep = imgHeight / layerHeight ;
    assert(opts.pixelStep == heightStep, 'non-square input not supported using `old-style` caffe syntax') ;
  end

  idx = 1 ;
  for i = 1:layerHeight
      for j = 1:layerWidth

          centreX = (j - opts.offset) * opts.pixelStep ;
          centreY = (i - opts.offset) * opts.pixelStep ;
          
          boxWidth = opts.minSize ;
          boxHeight = opts.minSize ;
          
          % first prior box:
          %  aspect ratio 1, size = opts.minSize
          xMin = (centreX - boxWidth / 2)  / imgWidth ;
          yMin = (centreY - boxHeight / 2) / imgHeight ;
          xMax = (centreX + boxWidth / 2)  / imgWidth ;
          yMax = (centreY + boxHeight / 2) / imgHeight ;
          boxes(idx:idx + 3) = [xMin, yMin, xMax, yMax]' ;
          idx = idx + 4 ;
          
          if opts.maxSize > 0
              % second prior box:
              %   aspect ratio 1, size = sqrt(opts.minSize * opts.maxSize)
              length = sqrt(opts.minSize * opts.maxSize) ;
              boxWidth = length ;
              boxHeight = length ;
              
              xMin = (centreX - boxWidth / 2)  / imgWidth ;
              yMin = (centreY - boxHeight / 2) / imgHeight ;
              xMax = (centreX + boxWidth / 2)  / imgWidth ;
              yMax = (centreY + boxHeight / 2) / imgHeight ;
              boxes(idx:idx + 3) = [xMin, yMin, xMax, yMax]' ;
              idx = idx + 4 ;
          end
          
          for k = 1:numel(aspectRatios)
              if abs(aspectRatios(k) - 1) < 1e-6
                  continue ;
              end
              boxWidth = opts.minSize * sqrt(aspectRatios(k)) ;
              boxHeight = opts.minSize / sqrt(aspectRatios(k)) ;
              
              xMin = (centreX - boxWidth/2)  / imgWidth ;
              yMin = (centreY - boxHeight/2) / imgHeight ;
              xMax = (centreX + boxWidth/2)  / imgWidth ;
              yMax = (centreY + boxHeight/2) / imgHeight ;
              boxes(idx:idx + 3) = [xMin, yMin, xMax, yMax]' ;
              idx = idx + 4 ;            
          end
      end
  end

  % If clip is true, constrain all relative box coordinates to
  % lie in the interval [0,1].
  if opts.clip
    boxes(boxes(:) < 0) = 0 ;
    boxes(boxes(:) > 1) = 1 ;
  end

  variances = repmat(opts.variance, [numBoxes 1]) ;
  y = cast(cat(3, boxes, variances), 'like', x) ;
