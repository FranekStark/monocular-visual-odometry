//
// Created by franek on 27.09.19.
//
#include <iostream>
#include <ros/ros.h>
#include "Frame.hpp"

void Frame::lock() const{
  LOG_DEBUG("tries to lock Frame" << this);
  _lock.lock();
  LOG_DEBUG("locked Frame" << this);
}

void Frame::unlock() const{
  LOG_DEBUG("unlock Frame" << this);
  _lock.unlock();
}
void Frame::addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                               const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                               const std::vector<unsigned char> &found) {
  this->lock();
  assert(this->_preFrame != nullptr);
  assert(trackedFeaturesNow.size() == trackedFeaturesNowE.size());
  assert(found.size() == trackedFeaturesNowE.size());
  assert(trackedFeaturesNowE.size() == this->_preFrame->_features.size());
  /* Reserve some Memory */
  this->_features.reserve(found.size());
  /* Iterate trough the tracked Features and and them to current Frame*/
  auto trackedFeature = trackedFeaturesNow.begin();
  auto trackedFeatureE = trackedFeaturesNowE.begin();
  auto featureFound = found.begin();
  while (featureFound != found.end()) {
    if (*featureFound == 1) { //If the feature has been found
      int indexOfBeforeFeature = std::distance(found.begin(), featureFound);
      /*Push Back the tracked Feature */
      this->_features.push_back(Feature(*trackedFeature,
                                        *trackedFeatureE,
                                        indexOfBeforeFeature,
                                        0)); //Set it to 0, it will be correct set in Merger
    }
    trackedFeature++;
    trackedFeatureE++;
    featureFound++;
  }
  this->unlock();
}
void Frame::addFeaturesToFrame(const std::vector<cv::Point2f> &features, const std::vector<cv::Vec3d> &featuresE) {
  assert(features.size() == featuresE.size());
  this->lock();
  this->_features.reserve(features.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  while (featureIT != features.end()) {
    this->_features.push_back(Feature(*featureIT, *featureEIT, -1, 0));
    featureEIT++;
    featureIT++;
  }
  this->unlock();
}
const cv::Mat &Frame::getImage() {
  return (this->_imagePyramide)[0]; //TODO: Sync-Problems?
}
const cv::Matx33d &Frame::getRotation() {
  return this->_rotation; //TODO:: Lockproblem?
}
const image_geometry::PinholeCameraModel &Frame::getCameraModel() {
  return this->_cameraModel; //TODO: Lockproblem
}
unsigned int Frame::getNumberOfKnownFeatures() {
  this->lock();
  unsigned int size = this->_features.size();
  this->unlock();
  return size;
}
void Frame::disbandFeatureConnection(const std::vector<unsigned int> &indices) {
  this->lock();
  for (auto index : indices) {
    this->_features[index]._preFeature = -1;
    this->_features[index]._preFeatureCounter = 0;
  }
  this->unlock();
}
void Frame::updateFrame(Frame &targetFrame, Frame &sourceFrame) {
  targetFrame.lock();
  sourceFrame.lock();
  assert(&targetFrame == sourceFrame._preFrame);
  //Update the preIndexes
  for (auto sourceFeature = sourceFrame._features.begin(); sourceFeature != sourceFrame._features.end();
       sourceFeature++) {
    int &index = sourceFeature->_preFeature;
    if (index >= 0) {
      index = targetFrame._features[index]._preFeature;
    }
  }
  //Update the PreFrame
  sourceFrame._preFrame = targetFrame._preFrame;
  //Delte the leftover Frame
  targetFrame.unlock();
  sourceFrame.unlock();
  LOG_DEBUG("deleted Frame: "<< &targetFrame);
  delete &targetFrame;
}
void Frame::mergeFrame(Frame &targetFrame, Frame &sourceFrame) {
  //Lock Frames
  sourceFrame.lock();
  targetFrame.lock();
  //Calculate the difference
  int difference =  sourceFrame._features.size() - targetFrame._features.size();
  if (difference > 0) { //Only if there are further features run the algorithm
    //Iterate through the 'new' Features and add them to before Frame
    targetFrame._features.reserve(difference);
    for (auto feature = sourceFrame._features.begin(); feature != sourceFrame._features.end(); feature++) {
      if (feature->_preFeature == -1) { //If there is no precessor, it is a new Feature
        targetFrame._features.push_back(Feature(feature->_positionImage, feature->_positionEuclidian, -1, 0));
        //Set the connection right
        feature->_preFeature = (targetFrame._features.size() - 1); //The last Index
      }
    }
  }
  //Remove the Frames in between
  Frame *frame = &sourceFrame;
  while (frame->_preFrame != &targetFrame) {//Remove the Preframe if its not the target Frame
    Frame *deleteFrame = frame->_preFrame;
    //Connect the pre correct
    frame->_preFrame = deleteFrame->_preFrame;
    LOG_DEBUG("deleted Frame: "<< deleteFrame);
    delete deleteFrame;
  }
  targetFrame.unlock();
  sourceFrame.unlock();
}
void Frame::calculateFeaturePreCounter() {
  this->lock();
  assert(this->_preFrame != nullptr);
  this->_preFrame->lock();
  Frame &preFrame = *(this->_preFrame);
  for (auto feature = this->_features.begin(); feature != this->_features.end(); feature++) {
    if (feature->_preFeature >= 0) {
      feature->_preFeatureCounter = preFrame._features[feature->_preFeature]._preFeatureCounter + 1;
    }
    else{
      feature->_preFeatureCounter = 0;
      feature->_preFeature = -1;
    }
  }
  this->_preFrame->unlock();
  this->unlock();
}
cv::Vec3d Frame::getBaseLineToPrevious() {
  this->lock();
  auto baseLine = this->_baseLine;
  this->unlock();
  return baseLine;
}
const std::vector<cv::Mat> & Frame::getImagePyramid() {
  return this->_imagePyramide;
}
void Frame::setBaseLineToPrevious(const cv::Vec3d &baseLine) {
  this->lock();
  this->_baseLine = baseLine;
  this->unlock();
}
template<>
const cv::Point2f &Frame::getFeatureLocation(const Feature &f) {
  return f._positionImage;
}
bool Frame::isFirstFrame() {
  return this->_preFrame == nullptr;
}
Frame::Frame(std::vector<cv::Mat> imagePyramide,
             image_geometry::PinholeCameraModel camerModel,
             cv::Matx33d rotation,
             Frame *preFrame,
             mvo::mvoConfig params,
             ros::Time timeStamp
             ):
             _imagePyramide(imagePyramide),
             _cameraModel(camerModel),
             _scale(1.0),
             _rotation(rotation),
             _preFrame(preFrame),
             _parameters(params),
             _timeStamp(timeStamp)
             {
}
const mvo::mvoConfig &Frame::getParameters() {
  return _parameters;
}
double Frame::getScaleToPrevious() {
  this->lock();
  double scale = _scale;
  this->unlock();
  return scale;
}
void Frame::setScaleToPrevious(double scale) {
  this->lock();
  _scale = scale;
  this->unlock();
}
ros::Time Frame::getTimeStamp() {
  return _timeStamp;
}

template<>
const cv::Vec3d &Frame::getFeatureLocation(const Feature &f) {
  return f._positionEuclidian;
}