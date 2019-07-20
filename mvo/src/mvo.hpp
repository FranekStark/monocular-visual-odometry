#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

#include <stdio.h>
#include <list>

#include "SlidingWindow.hpp"

class MVO
{
private:
  std::vector<cv::Point2f> detectCorners(const cv::Mat &image, int num);  // TODO: return als Param?
  void trackFeatures(const cv::Mat &nowImage, const cv::Mat &prevImage, const std::vector<cv::Point2f> &prevFeatures,
                     std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found);

  /*Fields */
  SlidingWindow _slidingWindow;

  /*For Corner-Detector */
  int _blockSize;
  int _apertureSize;
  double _k;
  int _thresh;

public:
  MVO();
  ~MVO();
  void handleImage(const cv::Mat &image);
  void setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh);
};
