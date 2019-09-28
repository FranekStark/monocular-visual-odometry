//
// Created by franek on 24.09.19.
//

#include "TrackerDetector.hpp"

TrackerDetector::TrackerDetector(PipelineStage &precursor,
                                 unsigned int outGoingChannelSize,
                                 CornerTracking &cornerTracking,
                                 unsigned int number) :
    PipelineStage(&precursor, outGoingChannelSize),
    _prevFrame(nullptr),
    _cornerTracking(cornerTracking),
    _numberToDetect(number) {
#ifdef DEBUGIMAGES
  cv::namedWindow("TrackerImage", cv::WINDOW_NORMAL);
  cv::startWindowThread();
#endif

}

Frame *TrackerDetector::stage(Frame *newFrame) {
  //In Case the previous Frame is null only detect new Features
  if (_prevFrame == nullptr) {
    detect(*newFrame, _numberToDetect);
  } else { //Else track and then detect
    track(*newFrame);
    int numberToDetect = _numberToDetect - SlidingWindow::getNumberOfKnownFeatures(*newFrame);
    detect(*newFrame, numberToDetect);
  }
#ifdef DEBUGIMAGES
  cv::Mat image;
  cv::cvtColor(SlidingWindow::getImage(*newFrame), image, cv::COLOR_GRAY2BGR);
  VisualisationUtils::drawFeatures(*newFrame, image);
  cv::imshow("TrackerImage", image);
  cv::waitKey(10);
#endif
  //Pass through the new Frame
  _prevFrame = newFrame;
  return newFrame;
}

void TrackerDetector::track(Frame &newFrame) {
  std::vector<cv::Point2f> prevFeatures, trackedFeatures;
  /*Get previous Features*/
  SlidingWindow::getFeatures(*_prevFrame, prevFeatures);
  /*Calculate Difference rotation*/
  auto prevRotation = SlidingWindow::getRotation(*_prevFrame);
  auto nowRotation = SlidingWindow::getRotation(newFrame);
  auto diffRotation = prevRotation.t() * nowRotation;
  /*Estimate new location regarding to rotation*/
  std::vector<cv::Vec3d> prevFeaturesE, prevFeaturesERotated;
  //Project the Features to Rotate them
  FeatureOperations::euclidNormFeatures(prevFeatures, prevFeaturesE, SlidingWindow::getCameraModel(*_prevFrame));
  //Rotate them onto new Position
  FeatureOperations::unrotateFeatures(prevFeaturesE, prevFeaturesERotated, diffRotation.t());
  //Reproject the rotated Features
  FeatureOperations::euclidUnNormFeatures(prevFeaturesERotated, trackedFeatures, SlidingWindow::getCameraModel(newFrame));
  /*Track the Features (with guess from Rotation)*/
  std::vector<unsigned char> found;
  auto camInfo = SlidingWindow::getCameraModel(newFrame);
  auto shipMask = getShipMask(cv::Size2d(camInfo.rectifiedRoi().size()));
  _cornerTracking.trackFeatures(SlidingWindow::getImagePyramid(newFrame),
                                SlidingWindow::getImagePyramid(*_prevFrame),
                                prevFeatures,
                                trackedFeatures,
                                found,
                                shipMask);
  /*Project them to Euclidean*/
  std::vector<cv::Vec3d> trackedFeaturesE;
  FeatureOperations::euclidNormFeatures(trackedFeatures, trackedFeaturesE, SlidingWindow::getCameraModel(newFrame));
  /*Add them to the Frame*/
  SlidingWindow::addTrackedFeatures(trackedFeatures, trackedFeaturesE, found, newFrame);
}

void TrackerDetector::detect(Frame &newFrame, unsigned int number) {
  /*Detect new Features*/
  std::vector<cv::Point2f> existingFeatures, newFeatures;
  SlidingWindow::getFeatures(newFrame, existingFeatures); //get Current Existing Features
  auto camInfo = SlidingWindow::getCameraModel(newFrame);
  auto shipMask = getShipMask(cv::Size(camInfo.rectifiedRoi().size()));
  _cornerTracking.detectFeatures(newFeatures,
                                 std::vector<cv::Mat>(SlidingWindow::getImagePyramid(newFrame))[0],
                                 number,
                                 existingFeatures,
                                 shipMask,
                                 false);
  /*Convert them and Put them back*/
  std::vector<cv::Vec3d> newFeaturesE;
  FeatureOperations::euclidNormFeatures(newFeatures, newFeaturesE, SlidingWindow::getCameraModel(newFrame));
  SlidingWindow::addFeaturesToFrame(newFrame, newFeatures, newFeaturesE);
}
TrackerDetector::~TrackerDetector() {
  cv::destroyWindow("TrackerImage");
}
cv::Rect2d TrackerDetector::getShipMask(cv::Size imageSize) {

  /* Mask, where Ship is In Image*/
  cv::Rect2d shipMask((imageSize.width / 2) - (1.6 / 16.0) * imageSize.width,
                      imageSize.height - (6.5 / 16.0) * imageSize.height, (3.2 / 16.0) * imageSize.width,
                      (6.5 / 16.0) * imageSize.width);
  return shipMask;

}


