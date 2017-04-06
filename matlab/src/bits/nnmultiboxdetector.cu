// @file nnmultiboxdetector.cu
// @brief Multibox Detector block
// @author Samuel Albanie
// @author Andrea Vedaldi

/*
Copyright (C) 2017- Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnmultiboxdetector.hpp"
#include "impl/multiboxdetector.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <cstdio>
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                         multiboxdetector_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,T) \
error = vl::impl::multiboxdetector<deviceType,T>::forward \
(context, \
(T*) output.getMemory(), \
(T const*) locPreds.getMemory(), \
(T const*) confPreds.getMemory(), \
(T const*) priors.getMemory(), \
nmsTopK, \
keepTopK, \
numClasses, \
nmsThresh, \
confThresh, \
backgroundLabel, \
output.getHeight(), \
output.getWidth(), \
locPreds.getSize(), \
priors.getHeight()/4) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; \
break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; \
break ;) \
default: assert(false) ; \
return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnmultiboxdetector_forward(vl::Context& context,
                               vl::Tensor output,
                               vl::Tensor locPreds,
                               vl::Tensor confPreds,
                               vl::Tensor priors,
                               int nmsTopK,
                               int keepTopK,
                               int numClasses,
                               float nmsThresh,
                               float confThresh,
                               int backgroundLabel)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;
  
  // switch on locPreds type, since output is always on CPU
  switch (locPreds.getDeviceType())
  {
    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
    if (error == VLE_Cuda) {
      context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
    }
    break;
#endif

    default:
      assert(false);
      error = vl::VLE_Unknown ;
      break ;
  }
  return context.passError(error, __func__);
}
