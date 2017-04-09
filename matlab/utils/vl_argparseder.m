function [dzdy, opts] = vl_argparseder(opts, args)
%VL_ARGPARSEDER wraps VL_ARGPARSE but peels off derivatives
%  [DZDY, OPTS] = VL_ARGPARSEDER(OPTS, ARGS) wraps a call
%  to VL_ARGPARSE, with the additional functionally of first pulling
%  out any derivative term found in args

if ~isempty(args) && ~ischar(args{1}) 
  dzdy = args{1} ;
  args(1) = [] ;
else
  dzdy = [] ;
end

assert(nargout == 2, 'two output arguments must be specified') ;
opts = vl_argparse(opts, args, 'nonrecursive') ;
