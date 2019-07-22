#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

#include <image_geometry/pinhole_camera_model.h>

#include <stdio.h>
#include <list>

#include "SlidingWindow.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

class MVO
{
private:
  std::vector<cv::Point2f> detectCorners(const cv::Mat &image, int num);  // TODO: return als Param?
  void trackFeatures(const cv::Mat &nowImage, const cv::Mat &prevImage, const std::vector<cv::Point2f> &prevFeatures,
                     std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found);
  Eigen::Translation3d calculateBaseLine(const std::vector<Eigen::Vector2d> &mt,
                                         const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);

  Eigen::Translation3d calculateBaseLineMLESAC(const std::vector<Eigen::Vector2d> &mt,
                                         const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);
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
  void handleImage(const cv::Mat &image, const image_geometry::PinholeCameraModel &camerModel);
  void setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh);
};
