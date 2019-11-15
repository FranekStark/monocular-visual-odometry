#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <utility>

MVO::MVO(std::function<void(cv::Point3d, cv::Matx33d, ros::Time timeStamp)> estimatedPositionCallback,
         std::function<void(cv::Point3d, cv::Matx33d, ros::Time timeStamp)> refinedPositionCallback,
         mvo::mvoConfig startConfig) :
    _estimatedPosition(0, 0, 0),
    _refinedPosition(0, 0, 0),
    _estimatedCallbackFunction(estimatedPositionCallback),
    _refinedCallbackFunction(refinedPositionCallback),
    _trackerDetector(*this, 10, _cornerTracking),
    _merger(_trackerDetector, 10),
    _baseLineEstimator(_merger, 100, _epipolarGeometry),
    _scaler(_baseLineEstimator, 100),
    _refiner(_scaler, 4, _iterativeRefinement, startConfig.numberToRefine, startConfig.numberToNote),
    _end(_refiner),
    _trackerThread(std::ref(_trackerDetector)),
    _mergerThread(std::ref(_merger)),
    _estimatorThread(std::ref(_baseLineEstimator)),
    _scalerThread(std::ref(_scaler)),
    _refinerThread(std::ref(_refiner)),
    _endThread(std::ref(_end)),
    _estimatedCallbackThread([this]() {
      do {
        auto baseLine = _baseLineEstimator._baseLine.dequeue();
        _estimatedPosition = _estimatedPosition + cv::Point3d(baseLine.position);
        _estimatedCallbackFunction(_estimatedPosition, baseLine.orientation, baseLine.timeStamp);
      } while (ros::ok());
    }),
    _refinedCallbackThread([this]() {
      do {
        auto baseLine = _refiner._baseLine.dequeue();
        _refinedPosition = _refinedPosition + cv::Point3d(baseLine.position);
        _refinedCallbackFunction(_refinedPosition, baseLine.orientation, baseLine.timeStamp);
      } while (ros::ok());
    }),
    _prevFrame(nullptr) {
// Set Threadnames:
  Utils::SetThreadName(&_trackerThread,
                       "TrackerDetector");
  Utils::SetThreadName(&_mergerThread,
                       "Merger");
  Utils::SetThreadName(&_estimatorThread,
                       "Estimator");
  Utils::SetThreadName(&_refinerThread,
                       "Refiner");
  Utils::SetThreadName(&_endThread,
                       "Endthread");
  Utils::SetThreadName(&_estimatedCallbackThread,
                       "CallbackEstimator");
  Utils::SetThreadName(&_refinedCallbackThread,
                       "CallbackRefiner");

#ifdef DEBUGIMAGES
  cv::startWindowThread();
#endif
}

void MVO::newImage(const cv::Mat &image,
                   const image_geometry::PinholeCameraModel &cameraModel,
                   const cv::Matx33d &R,
                   mvo::mvoConfig parameters, const ros::Time &timeStamp) {
  auto pyramideImage = _cornerTracking.createPyramide(image,
                                                      cv::Size(parameters.windowSizeX, parameters.windowSizeY),
                                                      parameters.pyramidDepth);
  //Creates Frame:
  auto *frame = new Frame(pyramideImage, cameraModel, R, _prevFrame, parameters, timeStamp);
  _prevFrame = frame;
  pipeIn(frame);
  LOG_DEBUG("New Frame Created and Piped in: " << frame);
}

MVO::~MVO() = default;


