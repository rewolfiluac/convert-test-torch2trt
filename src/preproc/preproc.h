#include <string.h>
#include <iostream>
#include <cassert>

void padToSquare(
    unsigned char *inArr, int inDim1, int inDim2, int inDim3,
    unsigned char *inplaceArr, int inplaceDim1, int inplaceDim2, int inplaceDim3,
    unsigned char value = 0);

void normalize(
    unsigned char *inArr, int inDim1, int inDim2, int inDim3,
    float* inplaceArr, int inplaceDim1,
    float *mean, int meanDim1,
    float *std, int stdDim1,
    float devVal = 255);
