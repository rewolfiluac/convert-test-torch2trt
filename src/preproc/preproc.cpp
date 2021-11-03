#include "preproc.h"

using namespace std;

void padToSquare(
    unsigned char *inArr, int inDim1, int inDim2, int inDim3,
    unsigned char *inplaceArr, int inplaceDim1, int inplaceDim2, int inplaceDim3,
    unsigned char value)
{
    assert(inplaceDim1 == inplaceDim2);
    assert(inDim3 == 1 || inDim3 == 3);
    assert(inplaceDim3 == 1 || inplaceDim3 == 3);

    int orgImgPosX = 0;
    int orgImgPosY = 0;
    if (inDim1 < inplaceDim1)
    {
        orgImgPosY = (int)((inplaceDim1 - inDim1) / 2);
    }
    if (inDim2 < inplaceDim2)
    {
        orgImgPosX = (int)((inplaceDim2 - inDim2) / 2);
    }

    // memset(inplaceArr, 0, inplaceDim1 * inplaceDim2 * inplaceDim3);

    for (int j = 0; j < inDim1; j++)
    {
        memcpy(
            inplaceArr + ((inplaceDim2 * (orgImgPosY + j) + orgImgPosX) * inplaceDim3),
            inArr + ((inDim2 * j + orgImgPosX) * inDim3),
            sizeof(unsigned int) * inDim2);
    }
}

void normalize(
    unsigned char *inArr, int inDim1, int inDim2, int inDim3,
    float *inplaceArr, int inplaceDim1,
    float *mean, int meanDim1,
    float *std, int stdDim1,
    float devVal)
{
    assert(inDim3 == 3);
    assert(meanDim1 == 3);
    assert(stdDim1 == 3);

    for (int j = 0; j < inDim1; j++)
    {
        for (int i = 0; i < inDim2; i++)
        {
            int inIdx = (j * inDim2 + i) * inDim3;
            int inplaceIdx = j * inDim2 + i;

            inplaceArr[inplaceIdx] = (inArr[inIdx] / devVal - mean[0]) / std[0];
            inplaceArr[inplaceIdx + (inDim1 * inDim2)] = (inArr[inIdx + 1] / devVal - mean[1]) / std[1];
            inplaceArr[inplaceIdx + (inDim1 * inDim2) * 2] = (inArr[inIdx + 2] / devVal - mean[2]) / std[2];
        }
    }
    return;
}
