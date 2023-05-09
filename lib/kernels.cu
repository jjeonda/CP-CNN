/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "yolo.h"

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

// jdy add letterbox
__global__ void gpuYoloLayerV3(const float* input, float* output, const uint gridSize_1, const uint gridSize_2, const uint numOutputClasses,
                               const uint numBBoxes)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;


    //printf("xid : %d\n", x_id);
    if ((x_id >= gridSize_2) || (y_id >= gridSize_1) || (z_id >= numBBoxes))
//    if (x_id >= gridSize_1*gridSize_2*(numBBoxes))
    {
        return;
    }
    //printf("grid_1 %d , gird_2 %d, numbbox: %d\n", gridSize_1, gridSize_2, numBBoxes);
    const int numGridCells = gridSize_1 * gridSize_2;
//    int z_id = x_id/numGridCells;
//    int bbindex = x_id - z_id*numGridCells;
//    printf("bbindex : %d, zid : %d\n", bbindex, z_id);
    const int bbindex = y_id * gridSize_2 + x_id;

#ifdef MDN
    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 2)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 3)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 4)]
        = __expf(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 4)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 5)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 5)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 6)]
        = __expf(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 6)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 7)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 7)]);

    output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 8)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + 8)]);

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + (9 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (9 + numOutputClasses) + (9 + i))]);
    }
#else
    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
#endif
}

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize_1, const uint& gridSize_2,
                            const uint& numOutputClasses, const uint& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
 // jdy add letterbox
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize_2 / threads_per_block.x) + 1,
                          (gridSize_1 / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
//    dim3 threads_per_block(512, 1, 1);
//    dim3 number_of_blocks((gridSize_1*gridSize_2*numBBoxes / threads_per_block.x) + 1,1,1);
//    printf("threads/block x  : %d , y : %d, z : %d\n", threads_per_block.x, threads_per_block.y, threads_per_block.z);
    for (int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSize_1, gridSize_2, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}
