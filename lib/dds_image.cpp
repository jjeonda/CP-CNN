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
#include "dds_image.h"
#include <experimental/filesystem>

DdsImage::DdsImage() :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),	
    m_RNG(cv::RNG(unsigned(std::time(0)))),
    m_ImageName()
{
}

// jdy add count for fr. 
DdsImage::DdsImage(cv::Mat m, const int& inputH, const int& inputW, int count) :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),	
    m_RNG(cv::RNG(unsigned(std::time(0)))),
    m_ImageName()
{
    m_ImageName = std::to_string(count);

    //m_OrigImage = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    m_OrigImage = m;

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
    {
        std::cout << "Unable to open video frame : " << count << std::endl;
        assert(0);
    }

    if (m_OrigImage.channels() != 3)
    {
        std::cout << "Non RGB images are not supported video frame : " << count << std::endl;
        assert(0);
    }

    m_OrigImage.copyTo(m_MarkedImage);
    m_Height = m_OrigImage.rows;
    m_Width = m_OrigImage.cols;

    printf("m_Height:%d, m_width:%d, inputH: %d, inputW:%d\n", m_Height, m_Width, inputH, inputW); 	// jdy add debug

    // resize the DsImage with scale
    //float dim = std::max(m_Height, m_Width);
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
//    int resizeH = ((m_Height / dim) * inputH);
//    int resizeW = ((m_Width / dim) * inputW);

    // jdy add letterbox
 
    m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);
//    printf("resizeH : %d, resizeW : %d\n", resizeH, resizeW);

    // Additional checks for images with non even dims
    if ((inputW - resizeW) % 2) resizeW--;
    if ((inputH - resizeH) % 2) resizeH--;
    assert((inputW - resizeW) % 2 == 0);
    assert((inputH - resizeH) % 2 == 0);

    m_XOffset = (inputW - resizeW) / 2;
    m_YOffset = (inputH - resizeH) / 2;

    assert(2 * m_XOffset + resizeW == inputW);
    assert(2 * m_YOffset + resizeH == inputH);


    // resizing
    cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);
    //cv::imshow("resize", m_LetterboxImage); 
    //cv::waitKey(0);
    // letterboxing
    cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                       m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    //cv::imshow("letterboxomg", m_LetterboxImage); 
    //cv::waitKey(0);
//    m_LetterboxImage.copyTo(m_MarkedImage); 	// jdy add test
    // converting to RGB
    cv::cvtColor(m_LetterboxImage, m_LetterboxImage, CV_BGR2RGB);
    //cv::imshow("letterbox", m_LetterboxImage); 
    //cv::waitKey(0);
//    std::cout << "resizeH" << resizeH << "resizeW" << resizeW << std::endl;
//    int h_dist = inputH/(inputH*4/32); int w_dist = inputW/(inputW*4/32);
//    for(int i=0; i<inputH; i+=h_dist)
//        cv::line(m_MarkedImage, cv::Point(0,i), cv::Point(inputW,i), cv::Scalar(0,0,0));
//    for(int i=0; i<inputW; i+=w_dist)
//        cv::line(m_MarkedImage, cv::Point(i,0), cv::Point(i,inputH), cv::Scalar(0,0,0));
//    cv::Rect roi;
//    roi.x = m_XOffset; roi.y = m_YOffset; roi.width=resizeW; roi.height = resizeH;
//    m_MarkedImage = m_MarkedImage(roi);
//    cv::resize(m_MarkedImage, m_MarkedImage, cv::Size(m_Width, m_Height), 0, 0, cv::INTER_CUBIC);
}

void DdsImage::addBBox(BBoxInfo box, const std::string& labelName)
{

//    printPredictions(box, labelName);
    m_Bboxes.push_back(box);
    const int x = box.box.x1;
    const int y = box.box.y1;
    const int w = box.box.x2 - box.box.x1;
    const int h = box.box.y2 - box.box.y1;
    const cv::Scalar color
        = cv::Scalar(m_RNG.uniform(0, 255), m_RNG.uniform(0, 255), m_RNG.uniform(0, 255));

    cv::rectangle(m_MarkedImage, cv::Rect(x, y, w, h), color, 1);
    const cv::Size tsize
        = cv::getTextSize(labelName, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1, nullptr);
    cv::rectangle(m_MarkedImage, cv::Rect(x, y, tsize.width + 3, tsize.height + 4), color, -1);
    cv::putText(m_MarkedImage, labelName.c_str(), cv::Point(x, y + tsize.height),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, CV_AA);
}

void DdsImage::showImage() const
{
    cv::namedWindow(m_ImageName);
    cv::imshow(m_ImageName.c_str(), m_MarkedImage);
    cv::waitKey(0);
}

void DdsImage::saveImageJPEG(const std::string& dirPath) const
{
    cv::imwrite(dirPath + m_ImageName + ".jpeg", m_MarkedImage);
}
std::string DdsImage::exportJson() const
{
    if (m_Bboxes.size() == 0) return "";
    std::stringstream json;
    json.precision(2);
    json << std::fixed;
    for (uint i = 0; i < m_Bboxes.size(); ++i)
    {
        json << "\n{\n";
        json << "  \"image_id\"         : " << std::stoi(m_ImageName) << ",\n";
        json << "  \"category_id\"      : " << m_Bboxes.at(i).classId << ",\n";
        json << "  \"bbox\"             : ";
        json << "[" << m_Bboxes.at(i).box.x1 << ", " << m_Bboxes.at(i).box.y1 << ", ";
        json << m_Bboxes.at(i).box.x2 - m_Bboxes.at(i).box.x1 << ", "
             << m_Bboxes.at(i).box.y2 - m_Bboxes.at(i).box.y1 << "],\n";
        json << "  \"score\"            : " << m_Bboxes.at(i).prob << "\n";
        if (i != m_Bboxes.size() - 1)
            json << "},";
        else
            json << "}";
    }
    return json.str();
}
