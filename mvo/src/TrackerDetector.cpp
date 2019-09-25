//
// Created by franek on 24.09.19.
//

#include "TrackerDetector.hpp"

TrackerDetector::TrackerDetector(SlidingWindow &slidingWindow,
                                 PipelineStage &precursor,
                                 unsigned int outGoingChannelSize,
                                 CornerTracking &cornerTracking,
                                 unsigned int number) :
                                 PipelineStage(slidingWindow, precursor, outGoingChannelSize),
                                 _cornerTracking(cornerTracking),
                                 _numberToDetect(number)
                                 {}
Frame* TrackerDetector::stage(Frame *newFrame) {
  //In Case the previous Frame is null only detect new Features
  if(_prevFrame == nullptr){
    detect(*newFrame, _numberToDetect);
  }else { //Else track and then detect
    track(*newFrame);
    int numberToDetect = _numberToDetect - SlidingWindow::getNumberOfKnownFeatures(*newFrame);
    detect(*newFrame, numberToDetect);
  }
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
  auto diffRotation = prevRotation * nowRotation;
  /*Estimate new location regarding to rotation*/
  std::vector<cv::Vec3d> prevFeaturesE;
  //Project the Features to Rotate them
  FeatureOperations::euclidNormFeatures(prevFeatures, prevFeaturesE, SlidingWindow::getCameraModel(*_prevFrame));
  //Rotate them onto new Position
  FeatureOperations::unrotateFeatures(prevFeaturesE, prevFeaturesE, diffRotation.t());
  //Reproject the rotated Features
  FeatureOperations::euclidUnNormFeatures(prevFeaturesE, trackedFeatures, SlidingWindow::getCameraModel(newFrame));
  /*Track the Features (with guess from Rotation)*/
  std::vector<unsigned char> found;
  auto shipMask = MVO::getShipMask(cv::Size());
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
  auto shipMask = MVO::getShipMask(cv::Size());
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


