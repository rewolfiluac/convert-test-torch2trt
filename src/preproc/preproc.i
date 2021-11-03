%module preproc
%feature("autodoc", "1");
%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
#include "preproc.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {
    (float* inplaceArr, int inplaceDim1),
    (float* mean, int meanDim1),
    (float* std, int stdDim1)
};
%apply (unsigned char* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
    (unsigned char* inArr, int inDim1, int inDim2, int inDim3)
};
%apply (unsigned char* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
    (unsigned char* inplaceArr, int inplaceDim1, int inplaceDim2, int inplaceDim3)
};
%apply (unsigned char* INPLACE_ARRAY1, int DIM1) {
    (unsigned char* inplaceArr, int inplaceDim1)
};

%include "preproc.h"
