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

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#define PI 3.14159265

class MVO
{
private:
  Eigen::Translation3d calculateBaseLine(const std::vector<Eigen::Vector2d> &mt,
                                         const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);

  Eigen::Translation3d calculateBaseLineMLESAC(const std::vector<Eigen::Vector2d> &mt,
                                         const std::vector<Eigen::Vector2d> &mhi, const Eigen::Quaterniond &rh);

  void reconstructDepth(std::vector<double> &depth, const std::vector<Eigen::Vector2d> &m2L,
                                         const std::vector<Eigen::Vector2d> &m1L,  const Eigen::Quaterniond &r, const Eigen::Translation3d &b);
  /*Fields */
  SlidingWindow _slidingWindow;

  unsigned int _frameCounter;



   

public:
  CornerTracker _cornerTracker;
  IterativeRefinement _iterativeRefinement;
  MVO();
  ~MVO();
  void handleImage(const cv::Mat &image, const image_geometry::PinholeCameraModel &camerModel);
  
};
