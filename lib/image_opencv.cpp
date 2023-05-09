
#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image_opencv.h"

using namespace cv;
///////////////#information#jdy/////////////////
////////////////////////////////////////////////
//////					  //////
//////	          w, h : video	 	  //////
//////        width, height : input	  //////
//////					  //////
////////////////////////////////////////////////

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    //else cap = new VideoCapture(c);	// jdy mod for jetson
    else{
        printf("start capturing, width = %d\n", w);
	cap = new VideoCapture(c);
        //cap = new VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
    }
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, h);
    //if(fps) cap->set(CV_CAP_PROP_FPS, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, 30);
    return (void *) cap;
}

Mat make_empty_Mat(int w, int h, int c)
{
    Mat m(w, h, c);
    return m; 
}


Mat get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_Mat(0,0,0);
    return m;
}


int show_image_cv(Mat m, const char* name, int ms)
{
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

Mat letterbox_image(Mat m_OrigImage, const int inputH, const int inputW)
{
    Mat m_LetterboxImage;
//    m_OrigImage.copyTo(m_LetterboxImage);
    int m_Height = m_OrigImage.rows;
    int m_Width = m_OrigImage.cols;
    //printf("m_Height:%d, m_width:%d, inputH: %d, inputW:%d\n", m_Height, m_Width, inputH, inputW);
    
    int resizeH, resizeW;
    if (float(inputH)/m_Height > float(inputW)/m_Width)
    {
	resizeW = inputW;
	resizeH = m_Height * inputW / m_Width;
    }
    else
    {
	resizeH = inputH;
	resizeW = m_Width * inputH / m_Height;
    }

    if ((inputW - resizeW) % 2) resizeW--;
    if ((inputH - resizeH) % 2) resizeH--;
    assert((inputW - resizeW) % 2 == 0); 
    assert((inputH - resizeH) % 2 == 0); 

    float m_XOffset = (inputW - resizeW) / 2;
    float m_YOffset = (inputH - resizeH) / 2;

    assert(2 * m_XOffset + resizeW == inputW);
    assert(2 * m_YOffset + resizeH == inputH);

    // resizing
    resize(m_OrigImage, m_LetterboxImage, Size(resizeW, resizeH), 0, 0, INTER_CUBIC);
    copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                       m_XOffset, BORDER_CONSTANT, Scalar(128, 128, 128));
    cvtColor(m_LetterboxImage, m_LetterboxImage, CV_BGR2RGB);

   return m_LetterboxImage;
}

void addbox(BBoxInfo box, const std::string& labelName, Mat m_MarkedImage)
{
    RNG m_RNG = RNG(unsigned(std::time(0)));
    const int x = box.box.x1;
    const int y = box.box.y1;
    const int w = box.box.x2 - box.box.x1;
    const int h = box.box.y2 - box.box.y1;
    const Scalar color
        = Scalar(m_RNG.uniform(0, 255), m_RNG.uniform(0, 255), m_RNG.uniform(0, 255));

    rectangle(m_MarkedImage, Rect(x, y, w, h), color, 1); 
    const Size tsize
        = getTextSize(labelName, FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1, nullptr);
    rectangle(m_MarkedImage, Rect(x, y, tsize.width + 3, tsize.height + 4), color, -1);
    putText(m_MarkedImage, labelName.c_str(), Point(x, y + tsize.height),
                FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255, 255, 255), 1, CV_AA);

}
