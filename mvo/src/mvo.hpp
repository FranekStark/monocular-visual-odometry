
#ifndef MVO_SRC_MVO_HPP_
#define MVO_SRC_MVO_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

#include <image_geometry/pinhole_camera_model.h>

#include <cstdio>
#include <list>

#include "algorithms/IterativeRefinement.hpp"
#include "algorithms/CornerTracking.hpp"
#include "algorithms/EpipolarGeometry.hpp"

#include "pipeline/TrackerDetector.hpp"
#include "pipeline/Merger.h"
#include "pipeline/BaselineEstimator.hpp"
#include "pipeline/Refiner.hpp"
#include "pipeline/PipelineBegin.hpp"
#include "pipeline/PipeLineEnd.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <functional>
#include <thread>
#include "Utils.hpp"

#define PI 3.14159265

class MVO : public PipelineBegin {
 private:
  /**
   * Positions
   */
  cv::Point3d _estimatedPosition;
  cv::Point3d _refined1Position;
  cv::Point3d _refined2Position;

  /**
   * Callbackfunction
   */
  std::function<void(cv::Point3d, cv::Matx33d, ros::Time timeStamp)> _estimatedCallbackFunction;
  std::function<void(cv::Point3d, cv::Matx33d, ros::Time timeStamp)> _refined1CallbackFunction;
  std::function<void(cv::Point3d, cv::Matx33d, ros::Time timeStamp)> _refined2CallbackFunction;

  /**
   * Algorithms
   */
  CornerTracking _cornerTracking;
  EpipolarGeometry _epipolarGeometry;
  IterativeRefinement _iterativeRefinement;

  /**
   * Pipelinestages
   */
  TrackerDetector _trackerDetector;
  Merger _merger;
  BaselineEstimator _baseLineEstimator;
  Refiner _refiner;
  PipeLineEnd _end;

  /**
   * Pipelinethreads
   */
  std::thread _trackerThread;
  std::thread _mergerThread;
  std::thread _estimatorThread;
  std::thread _refinerThread;
  std::thread _endThread;

  /**
   *Callbackthreads
   */
  std::thread _estimatedCallbackThread;
  std::thread _refined1CallbackThread;
  std::thread _refined2CallbackThread;

  Frame *_prevFrame;

 public:
  MVO(std::function<void(cv::Point3d, cv::Matx33d,ros::Time timeStamp)> estimatedPositionCallback,
      std::function<void(cv::Point3d, cv::Matx33d,ros::Time timeStamp)> refined1PositionCallback,
      std::function<void(cv::Point3d, cv::Matx33d,ros::Time timeStamp)> refined2PositionCallback);
  ~MVO() override;

  void newImage(const cv::Mat &image,
                const image_geometry::PinholeCameraModel &cameraModel,
                const cv::Matx33d &R,
                mvo::mvoConfig parameters, const ros::Time &timeStamp);

};

#endif //MVO_SRC_MVO_HPP_
