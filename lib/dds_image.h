#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"

struct BBoxInfo;

class DdsImage
{
public:
    DdsImage();
    DdsImage(cv::Mat m, const int& inputH, const int& inputW, int count);
    int getImageHeight() const { return m_Height; }
    int getImageWidth() const { return m_Width; }
    cv::Mat getLetterBoxedImage() const { return m_LetterboxImage; }
    cv::Mat getOriginalImage() const { return m_OrigImage; }
    std::string getImageName() const { return m_ImageName; }
    void addBBox(BBoxInfo box, const std::string& labelName);
    void showImage() const;
    void saveImageJPEG(const std::string& dirPath) const;
    std::string exportJson() const;

private:
    int m_Height;
    int m_Width;
    int m_XOffset;
    int m_YOffset;
    float m_ScalingFactor;
    std::string m_ImagePath;
    cv::RNG m_RNG;
    std::string m_ImageName;
    std::vector<BBoxInfo> m_Bboxes;

    // unaltered original Image
    cv::Mat m_OrigImage;
    // letterboxed Image given to the network as input
    cv::Mat m_LetterboxImage;
    // final image marked with the bounding boxes
    cv::Mat m_MarkedImage;
};

#endif
