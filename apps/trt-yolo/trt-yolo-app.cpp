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
#include "ds_image.h"
#include "dds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"
#include "image_opencv.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <set>
#include <pthread.h>    // jdy add multi-th
#include <atomic>       // jdy add multi-th

/* OpenCV headers */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//using std::thread;
static std::vector<DsImage> dsImages, dsImages_1, dsImages_2, dsImages_3;       // jdy add dla
static std::vector<std::string> imageList;
static int InputH, InputW;
cv::Mat trtInput, trtInput_1, trtInput_2, trtInput_3;
static uint idx, idx_1, idx_2, idx_3;
static cv::Mat buff[4];
static cv::Mat buff_resize[4];

// video processing
void *vpreImage(void *ptr)
{
    buff_resize[0] = letterbox_image(buff[0], InputH, InputW); 
    trtInput = blobFromDdsImages(buff_resize[0], InputH, InputW);
    buff_resize[0].release();
}

void *vpreImage_1(void *ptr)
{
    buff_resize[1] = letterbox_image(buff[1], InputH, InputW); 
    trtInput_1 = blobFromDdsImages(buff_resize[1], InputH, InputW);
    buff_resize[1].release();
}

void *vpreImage_2(void *ptr)
{
    buff_resize[2] = letterbox_image(buff[2], InputH, InputW); 
    trtInput_2 = blobFromDdsImages(buff_resize[2], InputH, InputW);
    buff_resize[2].release();
}

void *vpreImage_3(void *ptr)
{
    buff_resize[3] = letterbox_image(buff[3], InputH, InputW); 
    trtInput_3 = blobFromDdsImages(buff_resize[3], InputH, InputW);
    buff_resize[3].release();
}

/*
void vimageDecode(cv::Mat buff, int k, int w, int h, int i)
{ 
    auto curImage = buff;
    auto binfo = inferNet->decodeDetections(0, h, w, 0);
    auto remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
    for (auto b : remaining)
    {
        addbox(b, inferNet->getClassName(b.label), curImage);
    }
    
    k = show_image_cv(curImage, "Demo", 1);
    buff[i] = get_image_from_stream(cap);
}
*/

// image processing
void *preImage(void *ptr)
{
    dsImages.clear();
    dsImages.emplace_back(imageList.at(idx), InputH, InputW);
    trtInput = blobFromDsImages(dsImages, InputH, InputW);
}

void *preImage_1(void *ptr)
{
    dsImages_1.clear();
    dsImages_1.emplace_back(imageList.at(idx_1), InputH, InputW);
    trtInput_1 = blobFromDsImages(dsImages_1, InputH, InputW);
}

void *preImage_2(void *ptr)
{
    dsImages_2.clear();
    dsImages_2.emplace_back(imageList.at(idx_2), InputH, InputW);
    trtInput_2 = blobFromDsImages(dsImages_2, InputH, InputW);
}

void *preImage_3(void *ptr)
{
    dsImages_3.clear();
    dsImages_3.emplace_back(imageList.at(idx_3), InputH, InputW);
    trtInput_3 = blobFromDsImages(dsImages_3, InputH, InputW);
}


int main(int argc, char** argv)
{
    // Flag set in the command line overrides the value in the flagfile
    gflags::SetUsageMessage(
        "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    std::string testImages = getTestImages();
    std::string testImagesPath = getTestImagesPath();
    std::string demoImagesPath = getDemoImages();	// jdy add demo
    bool decode = getDecode();
    bool doBenchmark = getDoBenchmark();
    bool viewDetections = getViewDetections();
    bool saveDetections = getSaveDetections();
    std::string saveDetectionsPath = getSaveDetectionsPath();
    uint batchSize = getBatchSize();
    bool shuffleTestSet = getShuffleTestSet();
    std::string inferenceType = getInferenceType();	// jdy add demo
    uint cudaStreamSize = getCudaStreamSize();		// jdy add dla

    srand(unsigned(seed));

//    std::unique_ptr<Yolo> inferNet{nullptr};	
    std::shared_ptr<Yolo> inferNet{nullptr};		// jdy add multi-th
    if ((networkType == "yolov2") || (networkType == "yolov2-tiny"))
    {
        //inferNet = std::unique_ptr<Yolo>{new YoloV2(batchSize, yoloInfo, yoloInferParams)};
        inferNet = std::shared_ptr<Yolo>{new YoloV2(batchSize, yoloInfo, yoloInferParams)};
    }
    else if ((networkType == "yolov3") || (networkType == "yolov3-tiny"))
    {
        inferNet = std::shared_ptr<Yolo>{new YoloV3(batchSize, yoloInfo, yoloInferParams)};
//        inferNet = std::unique_ptr<Yolo>{new YoloV3(batchSize, yoloInfo, yoloInferParams)};	// jdy add multi-th
    }
    else
    {
        assert(false && "Unrecognised network_type. Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 and yolov3-tiny");
    }

    if (testImages.empty())
    {
        std::cout << "Enter a valid file path for test_images config param" << std::endl;
        return -1;
    }

    // jdy add demo
    if (inferenceType == "demo")	
    {
///*
	std::cout << "\n/////DEMO START/////\n" << std::endl;

	static void *cap;
	static int buff_index = 0;

	int cam_index = 0;
	int i,j,k;
	int count = 0;
	int prefix = 0;
    	InputH = inferNet->getInputH();     
	InputW = inferNet->getInputW();
	struct timeval inferStart, inferEnd;
	double inferElapsed = 0;

	// Camera Setup
	int w = 1280; int h = 720;	
	int frames = 40;

	if(demoImagesPath !="false")
	{
	    std::cout << "Video file: " << demoImagesPath << "\n" << std::endl;
	    cap = open_video_stream(demoImagesPath.c_str(), 0, 0, 0, 0);
	    //exit(0);
	}
	else 
	{
	    std::cout << "Camera Index: " << cam_index << "\n" << std::endl;
	    cap = open_video_stream(0, cam_index, w, h, frames);
	}
	if(!cap) assert(false && "Couldn't connect to webcam.\n");

	buff[0] = get_image_from_stream(cap);

        w = buff[0].cols;	
	h = buff[0].rows;

	//for (i=1; i<3; i++) buff[0].copyTo(buff[i]);
	
	if(!prefix) make_window("Demo", 640, 360, 0);

	pthread_t th, th1, th2, th3;

	buff_resize[0] = letterbox_image(buff[0], InputH, InputW);
	trtInput = blobFromDdsImages(buff_resize[0], InputH, InputW);
	buff[1] = get_image_from_stream(cap);
	gettimeofday(&inferStart, NULL);
	assert(!pthread_create(&th1, 0, vpreImage_1, 0));
	inferNet->doInference_1(trtInput.data, 1);
	trtInput.release();

	std::cout << "\nstart!!!\n" << std::endl;
	while(count < 500)
	{
	    std::cout << count << "\n" << std::endl;
	    gettimeofday(&inferStart, NULL);

	    buff[2] = get_image_from_stream(cap);
	    pthread_join(th1, 0);
	    assert(!pthread_create(&th2, 0, vpreImage_2, 0));
	    inferNet->doInference_3(trtInput_1.data, 1);
	    if (decode) inferNet->vimageDecode(buff[0], k, w, h);
	    trtInput_1.release();
	    buff[0].release();

	    buff[3] = get_image_from_stream(cap);
	    pthread_join(th2, 0);
	    assert(!pthread_create(&th3, 0, vpreImage_3, 0));
	    inferNet->doInference_3(trtInput_2.data, 1);
	    if (decode) inferNet->vimageDecode(buff[1], k, w, h);
	    trtInput_2.release();
	    buff[1].release();

	    buff[0] = get_image_from_stream(cap);
	    pthread_join(th3, 0);
	    assert(!pthread_create(&th, 0, vpreImage, 0));
	    inferNet->doInference_3(trtInput_3.data, 1);
	    if (decode) inferNet->vimageDecode(buff[2], k, w, h);
	    trtInput_2.release();
	    buff[2].release();

	    buff[1] = get_image_from_stream(cap);
	    pthread_join(th, 0);
	    assert(!pthread_create(&th1, 0, vpreImage_1, 0));
	    inferNet->doInference_3(trtInput.data, 1);
	    if (decode) inferNet->vimageDecode(buff[3], k, w, h);
	    trtInput.release();
	    buff[3].release();

	    count+=4;
            gettimeofday(&inferEnd, NULL);
            inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec)
                           + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
                           * 1000;
            std::cout << " \nInference FPS (per 4 images) : " << 4000 / inferElapsed << "\n" << std::endl;
	}

//	inferNet->doInference_2(1);
	//if (decode) inferNet->vimageDecode(buff[3], k, w, h);
//*/	
	//VideoCapture *cap;
	//cap = new VideoCapture(demoImagePath)
    return 0;

    }
    // jdy end demo

    if (inferenceType == "test")	// jdy add demo
    {

    imageList = loadImageList(testImages, testImagesPath);
    std::cout << "Total number of images used for inference : " << imageList.size() << std::endl;

    if (shuffleTestSet)
    {
        std::random_shuffle(imageList.begin(), imageList.end(), [](int i) { return rand() % i; });
    }
    const int barWidth = 70;
    double inferElapsed = 0;
    double inferElapsed_1 = 0;

    std::ofstream fout;
    bool written = false;
    if (doBenchmark)
    {
        size_t extIndex = testImages.find_last_of(".txt");
        fout.open(testImages.substr(0, extIndex - 3) + "_" + networkType + "_" + precision
                  + "_results.json");
        fout << "[";
    }
    // Batched inference loop

    InputH = inferNet->getInputH();     
    InputW = inferNet->getInputW();
    struct timeval inferStart, inferEnd, inferStart_1, inferEnd_1;
    int tmp = 0;
    decode = 1;
    idx_2 = 1;
    pthread_t th, th1, th2, th3;

    dsImages.emplace_back(imageList.at(0), InputH, InputW);     // for first image

    dsImages_1.emplace_back(imageList.at(0), InputH, InputW);
    trtInput_1 = blobFromDsImages(dsImages_1, InputH, InputW);
    gettimeofday(&inferStart_1, NULL);
    assert(!pthread_create(&th2, 0, preImage_2, 0));
    inferNet->doInference_1(trtInput_1.data, dsImages_1.size());        // jdy add engine

    for (uint loopIdx = 2; loopIdx < imageList.size()-3 ; loopIdx += 4 ) 	// jdy add multi-th
    {
        idx_3 = loopIdx, idx = loopIdx+1; idx_1 = loopIdx+2; idx_2 = loopIdx+3;

        pthread_join(th2, 0);
        assert(!pthread_create(&th3, 0, preImage_3, 0));
        inferNet->doInference_3(trtInput_2.data, dsImages_1.size());
        //printf("decode : %d\n", idx);
        if (decode) inferNet->imageDecode(dsImages_1, viewDetections, doBenchmark, fout, written);

        pthread_join(th3, 0);
        assert(!pthread_create(&th, 0, preImage, 0));
        inferNet->doInference_3(trtInput_3.data, dsImages_2.size());
        //printf("decode : %d\n", idx_1);
        if (decode) inferNet->imageDecode(dsImages_2, viewDetections, doBenchmark, fout, written);

        pthread_join(th, 0);
        assert(!pthread_create(&th1, 0, preImage_1, 0));
        inferNet->doInference_3(trtInput.data, dsImages_3.size());
        //printf("decode : %d\n", idx_2);
        if (decode) inferNet->imageDecode(dsImages_3,  viewDetections, doBenchmark, fout, written);

        pthread_join(th1, 0);
        assert(!pthread_create(&th2, 0, preImage_2, 0));
        inferNet->doInference_3(trtInput_1.data, dsImages.size());
        //printf("decode : %d\n", idx_3);
        if (decode) inferNet->imageDecode(dsImages, viewDetections, doBenchmark, fout, written);

        tmp += 1;

    }

    inferNet->doInference_2(dsImages_1.size());
    if (decode) inferNet->imageDecode(dsImages_1, viewDetections, doBenchmark, fout, written);

     gettimeofday(&inferEnd_1, NULL);
     inferElapsed_1 += ((inferEnd_1.tv_sec - inferStart_1.tv_sec)
                    + (inferEnd_1.tv_usec - inferStart_1.tv_usec) / 1000000.0)
                    * 1000;

    if (doBenchmark)
    {
        fout << std::endl << "]";
        fout.close();
        std::cout << std::endl;
    }
    std::cout << std::endl
              << "Network Type : " << inferNet->getNetworkType() << " Precision : " << precision
              << " Batch Size : " << batchSize
	      << " ImageList Size : " << imageList.size()
              << " \ntmp Size : " << tmp << " 4*tmp+1 : " << 4*tmp+1
              << " \nInference time per image : " << inferElapsed_1 / (4*tmp+1) << " ms"
              << std::endl;

    return 0;
    }	// jdy demo end
}
