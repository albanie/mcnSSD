function compile_mcnSSD(varargin)
% compile the C++/CUDA components of the SSD Detector 

last_args_path = fullfile(vl_rootnn, 'matlab/mex/.build', ...
  'last_compile_opts.mat') ;
opts = {} ;

if exist(last_args_path, 'file')
  opts = {load(last_args_path)} ;
end

vl_compilenn(opts{:}, varargin{:}, 'preCompileFn', @preCompileFn) ;


% ------------------------------------------------------------------------
function [opts, mex_src, lib_src, flags] = preCompileFn(opts, mex_src, ...
                                                        lib_src, flags)
% ------------------------------------------------------------------------

mcn_root = vl_rootnn() ;
root = fullfile(fileparts(mfilename('fullpath')), 'matlab') ;

% Build inside the module path
flags.src_dir = fullfile(root, 'src') ;
flags.mex_dir = fullfile(root, 'mex') ;
flags.bld_dir = fullfile(flags.mex_dir, '.build') ;

if ~exist(fullfile(flags.bld_dir,'bits/impl'), 'dir')
  mkdir(fullfile(flags.bld_dir,'bits/impl')) ;
end

mex_src = {} ;
lib_src = {} ; 

if opts.enableGpu
  ext = 'cu' ; 
else 
  ext = 'cpp' ; 
end

% Add mcn dependencies
lib_src{end+1} = fullfile(mcn_root, 'matlab/src/bits', ['data.' ext]) ;
lib_src{end+1} = fullfile(mcn_root, 'matlab/src/bits', ['datamex.' ext]) ;
lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/impl/copy_cpu.cpp') ;

if opts.enableGpu
  lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/datacu.cu') ;
  lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/impl/copy_gpu.cu') ;
end

flags.cc{end+1} = sprintf('-I%s', fullfile(mcn_root,'matlab/src')) ;

% Add module files
lib_src{end+1} = fullfile(root,'src/bits',['nnmultiboxdetector.' ext]) ;
mex_src{end+1} = fullfile(root,'src',['vl_nnmultiboxdetector.' ext]) ;

% CPU-specific files
lib_src{end+1} = fullfile(root,'src/bits/impl/multiboxdetector_cpu.cpp') ;

% GPU-specific files
if opts.enableGpu
  lib_src{end+1} = fullfile(root,'src/bits/impl/multiboxdetector_gpu.cu') ;
end
