#include "trt_utils.h"

using namespace cv;

void *open_video_stream(const char *f, int c, int w, int h, int fps);
Mat make_empty_Mat(int w, int h, int c);
Mat get_image_from_stream(void *p);
int show_image_cv(Mat m, const char* name, int ms);
void make_window(char *name, int w, int h, int fullscreen);
Mat letterbox_image(Mat m_OrigImage, const int inputH, const int inputW);
void addbox(BBoxInfo box, const std::string& labelName, Mat m_MarkedImage);
