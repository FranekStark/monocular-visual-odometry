#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>

#include <stdio.h>
#include <list>

class MVO
{
private:
    std::list<cv::Point> detectCorners(const cv::Mat &image,  int num);

    /*For Corner-Detector */
    int _blockSize;
    int _apertureSize;
    double _k;
    int _thresh;

public:
    MVO();
    ~MVO();
    void handleImage(const cv::Mat &image);
    void setCornerDetectotParams(int blockSize, int aperatureSize, double k, int thresh);
};
