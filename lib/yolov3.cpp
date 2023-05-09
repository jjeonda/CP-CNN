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
#include "yolov3.h"
#include "yolo.h"
#include <math.h>

YoloV3::YoloV3(const uint batchSize, const NetworkInfo& networkInfo,
               const InferParams& inferParams) :
    Yolo(batchSize, networkInfo, inferParams){};

std::vector<BBoxInfo> YoloV3::decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                           const TensorInfo& tensor, const int stream)
{
    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    float xOffset = (m_InputW - scalingFactor * imageW) / 2;
    float yOffset = (m_InputH - scalingFactor * imageH) / 2;

    // jdy add dla
    const float* detections;
#ifdef Engine
    if (stream == 0) detections = &tensor.hostBuffer_1[imageIdx * tensor.volume];
    else if (stream == 1)  detections = &tensor.hostBuffer_1[imageIdx * tensor.volume];
#else
    if (stream == 0) detections = &tensor.hostBuffer[imageIdx * tensor.volume];
    else if (stream == 1)  detections = &tensor.hostBuffer[imageIdx * tensor.volume];
#endif
    // jdy end

    std::vector<BBoxInfo> binfo;
    int gridSize_1 = m_InputW/tensor.stride;
    int gridSize_2 = m_InputH/tensor.stride;
    const int numGridCells = gridSize_1 * gridSize_2;	// jdy add letterbox
    for (uint y = 0; y < gridSize_2; ++y)		// jdy add letterbox
    {
        for (uint x = 0; x < gridSize_1; ++x)	// jdy add letterbox
        {
            for (uint b = 0; b < tensor.numBBoxes; ++b)
            {
                const float pw = tensor.anchors[tensor.masks[b] * 2];
                const float ph = tensor.anchors[tensor.masks[b] * 2 + 1];

                const int bbindex = y * gridSize_1 + x;
	
#ifdef MDN
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 0)];

                const float by
                    = y + detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 2)];
                const float bw
                    = pw * (detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 4)]);
                const float bh
                    = ph * (detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 6)]);
		// jdy add mdn
                const float delta_x
                    = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 1)];

                const float delta_y
                    = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 3)];
                const float delta_w
                    = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 5)];
                const float delta_h
                    = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 7)];
		// jdy end
                const float objectness
                    = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + 8)];
#else
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0)];

                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 4)];
#endif
                float maxProb = 0.0f;
                int maxIndex = -1;

#ifdef MDN
#ifdef UC_NMS
		float max_uc_val = delta_x;
		int location = 0;
		for(int iter = 1; iter < 4; iter++)
		{
		    if(detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + (2 * iter + 1))] >= max_uc_val)
		    {
			max_uc_val = detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + (2 * iter + 1))];
			location = iter;
		    }
		}
		float uc_4;
		for(int iter = 0; iter < 4; iter++)
		{
		    if(iter !=location) uc_4 += detections[bbindex + numGridCells * (b * (9 + tensor.numClasses) + (2 * iter + 1))];
		}
		uc_4 = uc_4/3.0;

		
                for (uint i = 0; i < tensor.numClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (9 + tensor.numClasses) + (9 + i))])*(1.0-max_uc_val);
#else
                for (uint i = 0; i < tensor.numClasses; ++i)
                {
		    float uc_aver = (delta_x + delta_y + delta_w + delta_h)/4.0;
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (9 + tensor.numClasses) + (9 + i))])*(1.0-uc_aver);
#endif
#else
                for (uint i = 0; i < tensor.numClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + tensor.numClasses) + (5 + i))]);
#endif

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > m_ProbThresh)
                {
#ifdef UC_NMS
                    addBBoxProposal_UC(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb, binfo, uc_4);
#else
                    addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb, binfo);
#endif
		    //printf("(%d, %d)bx:%f, by:%f, bw:%f, bh:%f, tensor.stride:%d, xOffset:%f, yOffset:%f, maxProb: %f \n", y, x, bx, by, bw, bh, tensor.stride, xOffset, yOffset, maxProb);
                }
            }
        }
    }
    return binfo;
}
