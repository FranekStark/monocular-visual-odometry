#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

#include <image_geometry/pinhole_camera_model.h>

#include <stdio.h>
#include <list>

#include "SlidingWindow.hpp"
#include "IterativeRefinement.hpp"
#include "CornerTracker.hpp"
#include "EpipolarGeometry.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#define PI 3.14159265

class MVO
{
private:
  //Eigen::Translation3d calculateBaseLine(const std::vector<Eigen::Vector2d> &mt,
  //                                       const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);

  //Eigen::Translation3d calculateBaseLineMLESAC(const std::vector<Eigen::Vector2d> &mt,
  //                                       const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);

  void reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                           const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r,
                           const cv::Vec3d &b);

  void sortOutSameFeatures(const std::vector<cv::Point2f> & beforeFeatures, std::vector<cv::Point2f> & newFeatures);
  void euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> & featuresE, const image_geometry::PinholeCameraModel & cameraModel);
  void drawDebugImage(const std::vector<cv::Point2f> points, const cv::Vec3d baseLine, cv::Mat & image);
  /*Fields */
  SlidingWindow _slidingWindow;

  unsigned int _frameCounter;

  bool checkEnoughDisparity(std::vector<cv::Point2f> & first, std::vector<cv::Point2f> & second);

   

public:
  CornerTracker _cornerTracker;
  IterativeRefinement _iterativeRefinement;
  EpipolarGeometry _epipolarGeometry;
  MVO();
  ~MVO();
  void handleImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel);
  
};
