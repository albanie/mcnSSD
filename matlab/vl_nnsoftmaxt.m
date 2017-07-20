function Y = vl_nnsoftmaxt(x, varargin)
%VL_NNSOFTMAXT CNN softmax transpose.
%   Y = VL_NNSOFTMAXT(X) applies the softmax operator the data X. X
%   has dimension H x W x D x N, packing N arrays of W x H
%   D-dimensional vectors.
%
%   D can be thought of as the number of possible classes.  The function 
%   and the function computes the softmax along the dimension specified
%   as an option. 
%
%   DZDX = VL_NNSOFTMAXT(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%  VL_NNSOFTMAXT(.., 'option', value, ...) accepts the following options:
%
%  `dim`:: 1
%   The dimension of X along which to compute the softmax.
%
%  This function is based on Andrea Vedaldi's vl_nnsoftmax function.
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

% NOTE: This is approach to parsing dzdy from varargin works
% without autonn, but is less readable.  Will be updated to 
% use vl_argparsepos when it is included in the main library
if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ; varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.dim = 1 ;
opts = vl_argparsepos(opts, varargin) ;

E = exp(bsxfun(@minus, x, max(x, [], opts.dim))) ;
L = sum(E, opts.dim) ;
Y = bsxfun(@rdivide, E, L) ;

if isempty(dzdy) ; return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, opts.dim)) ;
