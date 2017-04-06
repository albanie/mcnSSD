function y = vl_nnmultiboxloss(x, l, dzdy, varargin)
%VL_NNMULTIBOXLOSS computes the Multibox Loss
%   Y = VL_NNMULTIBOXLOSS(X, L, N) computes the Multibox Loss which 
%   is a weighted combination of the class prediction loss and 
%   boudning box regression loss for a multibox object detector. The
%   inputs X, L and N are scalars where X is the class prediction loss,
%   L is the bounding box regression loss and N is the number of matches
%   used to normalize the loss. The output Y is a SINGLE scalar.
%
%   The Multibox Loss produced by X, L and N is:
%
%     Loss = 1 / N * (X + LOCWEIGHT * L)
%
%   where `LOCWEIGHT` is a scalar weighting term (described below).
%
%   DERS = VL_NNMULTIBOXLOSS(X, L, N, DZDY) computes the derivatives 
%   with respect to inputs X and L, where DERS = {DZDX, DZDL}. Here
%   DZDX, DZDL and DZDY have the same dimensions as X, L and Y respectively.
%
%   VL_NNMULTIBOXLOSS(..., 'option', value, ...) takes the following option:
%
%   `locWeight`:: 1
%    A scalar which weights the loss contribution of the regression loss. 

opts.locWeight = 1 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if nargin <= 1 || isempty(dzdy)
    y = x + opts.locWeight * l ;
else
    dzdx = dzdy  ;
    dzdl = dzdy * opts.locWeight ;
    y = {dzdx, dzdl} ;
end
