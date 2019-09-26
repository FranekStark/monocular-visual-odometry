#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

#include <image_geometry/pinhole_camera_model.h>

#include <stdio.h>
#include <list>

#include "sliding_window/SlidingWindow.hpp"
#include "algorithms/IterativeRefinement.hpp"
#include "algorithms/CornerTracking.hpp"
#include "algorithms/EpipolarGeometry.hpp"

#include "pipeline/TrackerDetector.hpp"
#include "pipeline/Merger.h"
#include "pipeline/BaselineEstimator.hpp"
#include "pipeline/Refiner.hpp"
#include "pipeline/PipelineBegin.hpp"
#include "pipeline/PipeLineEnd.cpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <functional>
#include <thread>


#define PI 3.14159265

#define DEBUGIMAGES
#define MEASURETIME

class MVO : public PipelineBegin {
 private:
  /**
   * Positions
   */
  cv::Point3d _estimatedPosition;
  cv::Point3d _refinedPosition;

  /**
   * Callbackfunction
   */
  std::function<void(cv::Point3d)> _estimatedCallbackFunction;
  std::function<void(cv::Point3d)> _refinedCallbackFunction;

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
   std::thread _refinedCallbackThread;

  Frame * _prevFrame;
  Frame * _lastFrame;

 public:
  MVO(std::function<void(cv::Point3d)> estimatedPositionCallback,
      std::function<void(cv::Point3d)> refinedPositionCallback);
  ~MVO();

  void newImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel, const cv::Matx33d &R);

};
