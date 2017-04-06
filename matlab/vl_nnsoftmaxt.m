function Y = vl_nnsoftmaxt(x, dzdy, varargin)
%VL_NNSOFTMAXT CNN softmax transpose.
% TODO:           

opts.dim = 3 ;
opts = vl_argparse(opts, varargin) ;

E = exp(bsxfun(@minus, x, max(x, [], opts.dim))) ;
L = sum(E, opts.dim) ;
Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, dzdy = [] ; end
if isempty(dzdy) ; return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, opts.dim)) ;
