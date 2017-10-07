function [y, dzdl] = vl_nnmultiboxloss(x, l, varargin)
%VL_NNMULTIBOXLOSS computes the Multibox Loss
%   Y = VL_NNMULTIBOXLOSS(X, L) computes the Multibox Loss which 
%   is a weighted combination of the class prediction loss and 
%   boudning box regression loss for a multibox object detector. The
%   inputs X and L are scalars where X is the class prediction loss and
%   L is the bounding box regression loss. The output Y is a SINGLE scalar.
%
%   The Multibox Loss produced by X and L is:
%
%     Loss = X + OPTS.LOCWEIGHT * L
%
%   where `OPTS.LOCWEIGHT` is a scalar weighting term (described below).
%
%   DERS = VL_NNMULTIBOXLOSS(X, L, DZDY) computes the derivatives 
%   with respect to inputs X and L, where DERS = {DZDX, DZDL}. Here
%   DZDX, DZDL and DZDY have the same dimensions as X, L and Y respectively.
%
%   VL_NNMULTIBOXLOSS(..., 'option', value, ...) takes the following option:
%
%   `locWeight`:: 1
%    A scalar which weights the loss contribution of the regression loss. 
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.locWeight = 1 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

  if isempty(dzdy)
    y = x + opts.locWeight * l ;
  else
    dzdx = dzdy{1} ; 
    dzdl = dzdx * opts.locWeight ;
    y = dzdx ;
  end
