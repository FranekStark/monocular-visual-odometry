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
#include "OdomData.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#define PI 3.14159265

#define DEBUGIMAGES

class MVO
{
private:
  void reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                           const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r,
                           const cv::Vec3d &b);

  void euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> & featuresE, const image_geometry::PinholeCameraModel & cameraModel);
  void drawDebugPoints(const std::vector<cv::Point2f> & points, const cv::Scalar & color, cv::Mat & image);
  void drawDebugImage(const cv::Vec3d & baseLine, cv::Mat &image, const cv::Scalar &color, unsigned int index);
  void drawDebugScale(cv::Mat image, double scaleBefore, double scaleAfter);
 
  /*Fields */
  SlidingWindow _slidingWindow;

  unsigned int _frameCounter;

  unsigned int _numberOfFeatures;

  double _disparityThreshold;

  bool checkEnoughDisparity(const std::vector<cv::Vec3d> &first, const std::vector<cv::Vec3d> &second);

  void unrotateFeatures(const std::vector<cv::Vec3d> & features, std::vector<cv::Vec3d> & unrotatedFeatures, const cv::Matx33d & R);

   

public:
  CornerTracker _cornerTracker;
  IterativeRefinement _iterativeRefinement;
  EpipolarGeometry _epipolarGeometry;
  MVO();
  ~MVO();
  OdomData handleImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel, const cv::Matx33d &R);
  void setParameters(unsigned int numberOfFeatures, double disparityThreshold);
  
  cv::Mat _debugImage;
  cv::Mat _debugImage2;
};
