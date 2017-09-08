// @file vl_multiboxdetector.cu
// @brief Multibox detector block MEX wrapper
// @author Samuel Albanie 
// @author Andrea Vedaldi
/*
Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <bits/mexutils.h>
#include <bits/datamex.hpp>
#include "bits/nnmultiboxdetector.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

/* option codes */
enum {
  opt_nms_top_k = 0,
  opt_keep_top_k,
  opt_num_classes,
  opt_nms_thresh,
  opt_conf_thresh,
  opt_background_label,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"nmsTopK",         1,   opt_nms_top_k        },
  {"keepTopK",        1,   opt_keep_top_k       },
  {"numClasses",      1,   opt_num_classes      },
  {"nmsThresh",       1,   opt_nms_thresh       },
  {"confThresh",      1,   opt_conf_thresh      },
  {"backgroundLabel", 1,   opt_background_label },
  {"Verbose",         0,   opt_verbose          },
  {0,                 0,   0                    }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_LOC_PREDS = 0, IN_CONF_PREDS, IN_PRIORS, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int nmsTopK = 400 ;
  int keepTopK = 200 ;
  int numClasses = 21 ;
  float nmsThresh = 0.45 ;
  float confThresh = 0.01 ;
  int backgroundLabel = 1 ;
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 3) {
    mexErrMsgTxt("There are less than three arguments.") ;
  }

  // backwards mode is not supported
  next = 3 ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_nms_top_k :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "NMSTOPK is not a scalar.") ;
        }
        nmsTopK = (int)mxGetPr(optarg)[0] ;
        break ;

      case opt_keep_top_k :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "KEEPTOPK is not a scalar.") ;
        }
        keepTopK = (int)mxGetPr(optarg)[0] ;
        break ;

      case opt_num_classes :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "NUMCLASSES is not a scalar.") ;
        }
        numClasses = (int)mxGetPr(optarg)[0] ;
        break ;

      case opt_nms_thresh :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "NMSTHRESH is not a scalar.") ;
        }
        nmsThresh = (float)mxGetPr(optarg)[0] ;
        break ;

      case opt_conf_thresh :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "CONFTHRESH is not a scalar.") ;
        }
        confThresh = (float)mxGetPr(optarg)[0] ;
        break ;

      case opt_background_label :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "BACKGROUNDLABEL is not a scalar.") ;
        }
        backgroundLabel = (float)mxGetPr(optarg)[0] ;
        break ;

      default: 
        break ;
    }
  }


  vl::MexTensor locPreds(context) ;
  vl::MexTensor confPreds(context) ;
  vl::MexTensor priors(context) ;

  locPreds.init(in[IN_LOC_PREDS]) ;
  locPreds.reshape(4) ;
  int batchSize = locPreds.getSize() ;

  confPreds.init(in[IN_CONF_PREDS]) ;
  confPreds.reshape(4) ;

  priors.init(in[IN_PRIORS]) ;
  priors.reshape(4) ;

  /* check for GPU/data class consistency */
  if (!vl::areCompatible(locPreds, confPreds)) {
    vlmxError(VLMXE_IllegalArgument, "LOCPREDS and CONFPREDS do not have compatible formats.") ;
  }

  /* check for GPU/data prior consistency */
  if (!vl::areCompatible(locPreds, priors)) {
    vlmxError(VLMXE_IllegalArgument, "LOCPREDS and PRIORS do not have compatible formats.") ;
  }

  /* check for appropriate number of prior predictions */
  int numPriors = priors.getHeight() / 4 ;
  if ((numPriors != (confPreds.getDepth() / numClasses)) | (numPriors != (locPreds.getDepth() / 4))) {
    vlmxError(VLMXE_IllegalArgument, "LOCPREDS and CONFPREDS do not match the given set of priors.") ;
  }

  /* check for the existence of prior variances */
  if (priors.getDepth() != 2) {
    vlmxError(VLMXE_IllegalArgument, "PRIORS dim 3 should have a depth of 2 (containing variances)") ;
  }

  /* Create output buffers */
  // Currently, the final output (after NMS) is on the CPU, regardless 
  // of the input data
  //vl::DeviceType deviceType = locPreds.getDeviceType() ;
  vl::MexTensor output(context) ;
  vl::DataType dataType = locPreds.getDataType() ;
  vl::TensorShape outputShape = vl::TensorShape(keepTopK, 6, 1, batchSize) ;
  output.initWithZeros(vl::VLDT_CPU, dataType, outputShape) ;


  if (verbosity > 0) {
    mexPrintf("vl_multiboxdetector: mode %s; %s\n",  
            (locPreds.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu", "forward") ;
        mexPrintf("vl_multiboxdetector: nmsTopK: %d\n", nmsTopK) ;
        mexPrintf("vl_multiboxdetector: keepTopK: %d\n", keepTopK) ;
        mexPrintf("vl_multiboxdetector: numClasses: %d\n", numClasses) ;
        mexPrintf("vl_multiboxdetector: nmsThresh: %d\n", nmsThresh) ;
        mexPrintf("vl_multiboxdetector: confThresh: %d\n", confThresh) ;
        mexPrintf("vl_multiboxdetector: backgroundLabel: %d\n", backgroundLabel) ;
        vl::print("vl_multiboxdetector: locPreds: ", locPreds) ;
        vl::print("vl_multiboxdetector: output: ", output) ;
      }
      /* -------------------------------------------------------------- */
      /*                                                    Do the work */
      /* -------------------------------------------------------------- */

      vl::ErrorCode error ;
      error = vl::nnmultiboxdetector_forward(context,
                                             output, 
                                             locPreds,
                                             confPreds,
                                             priors, 
                                             nmsTopK,
                                             keepTopK,
                                             numClasses,
                                             nmsThresh,
                                             confThresh,
                                             backgroundLabel) ;

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  out[OUT_RESULT] = output.relinquish() ;
}
