#include "SlidingWindow.hpp"

#include <ros/ros.h>

#include <fstream>
#include <iostream>


void SlidingWindow::addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                                       const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                                       const std::vector<unsigned char> &found,
                                       Frame &frame) {
  frame._lock.lock();
  assert(frame._preFrame != nullptr);
  assert(trackedFeaturesNow.size() == trackedFeaturesNowE.size());
  assert(found.size() == trackedFeaturesNowE.size());
  assert(trackedFeaturesNowE.size() == frame._preFrame->_features.size());
  /* Reserve some Memory */
  frame._features.reserve(found.size());
  /* Iterate trough the tracked Features and and them to current Frame*/
  auto trackedFeature = trackedFeaturesNow.begin();
  auto trackedFeatureE = trackedFeaturesNowE.begin();
  auto featureFound = found.begin();
  while (featureFound != found.end()) {
    if (*featureFound == 1) { //If the feature has been found
      int indexOfBeforeFeature = std::distance(found.begin(), featureFound);
      /*Push Back the tracked Feature */
      frame._features.push_back(Feature(*trackedFeature,
                                        *trackedFeatureE,
                                        indexOfBeforeFeature,
                                        0)); //Set it to 0, it will be correct set in Merger
    }
    trackedFeature++;
    trackedFeatureE++;
    featureFound++;
  }
  frame._lock.unlock();
}

void SlidingWindow::addFeaturesToFrame(Frame &frame,
                                       const std::vector<cv::Point2f> &features,
                                       const std::vector<cv::Vec3d> &featuresE) {
  assert(features.size() == featuresE.size());
  frame._lock.lock();
  frame._features.reserve(features.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  while (featureIT != features.end()) {
    frame._features.push_back(Feature(*featureIT, *featureEIT, -1, 0));
    featureEIT++;
    featureIT++;
  }
  frame._lock.unlock();
}

const cv::Mat &SlidingWindow::getImage(Frame &frame) {
  return std::vector<cv::Mat>(frame._image)[0]; //TODO: Sync-Problems?
}

template<typename T>
void SlidingWindow::getFeatures(const Frame &frame, std::vector<T> &features) {
  frame._lock.lock();
  //Check wether the Vector is empty
  assert(features.size() == 0);
  //Reserve needed memory
  features.reserve(frame._features.size());
  //Pushback all Features
  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    features.push_back(getFeatureLocation<T>(feature));
  }
  frame._lock.unlock();
}

template<typename T>
void SlidingWindow::getCorrespondingFeatures(const Frame &oldestFrame,
                                             const Frame &newestFrame,
                                             std::vector<std::vector<T> *> features) {

/*Calculate Depth and maximum Number Of Features  and check wether the params are okay*/
  //Lock Ingoing Frame:
  newestFrame._lock.lock();
  unsigned int depth = 0;
  const Frame *frame = &newestFrame;
  auto outFeatures = features.begin();
  while (frame != &oldestFrame) {
    //Lock the Frames:
    depth++;
    assert(*outFeatures != nullptr);
    assert((*outFeatures)->size() == 0);
    (*outFeatures)->reserve(frame->_features.size());
    frame = frame->_preFrame;
    frame->_lock.lock();
    assert(frame != nullptr);
  }
  assert(depth > 0);

/*Iterate through the Features and check wether are enough preFeatures and than get them*/

  for (auto newestFeature = newestFrame._features.begin(); newestFeature != newestFrame._features.end();
       newestFeature++) {
    if (newestFeature->_preFeatureCounter >= (depth)) {
      frame = &newestFrame;
      const Feature *feature = &(*newestFeature);
      unsigned int i = 0;
      do {
        outFeatures[i]->push_back(*feature);
        i++;
        frame = frame->_preFrame;
        feature = &frame->_features[feature->_preFeature];
      } while (i < depth);
      outFeatures[i]->push_back(*feature);
    }
  }

  //Unlock Frames;
  newestFrame._lock.unlock();
  frame = &newestFrame;
  while (frame != &oldestFrame) {
    //Lock the Frames:
    frame = frame->_preFrame;
    frame->_lock.unlock();
  }

}

const cv::Matx33d &SlidingWindow::getRotation(const Frame &frame) {
  return frame._rotation; //TODO:: Lockproblem?
}

const image_geometry::PinholeCameraModel &SlidingWindow::getCameraModel(const Frame &frame) {
  return frame._cameraModel; //TODO: Lockproblem
}

template<>
const cv::Point2f &SlidingWindow::getFeatureLocation(const Feature &f) {
  return f._positionImage;
}

template<>
const cv::Vec3d &SlidingWindow::getFeatureLocation(const Feature &f) {
  return f._positionEuclidian;
}

unsigned int SlidingWindow::getNumberOfKnownFeatures(Frame &frame) {
  frame._lock.lock();
  return frame._features.size();
  frame._lock.unlock();
}

void SlidingWindow::disbandFeatureConnection(const std::vector<unsigned int> &indices, Frame &frame) {
  frame._lock.lock();
  for (auto index : indices) {
    frame._features[index]._preFeature = -1;
    frame._features[index]._preFeatureCounter = 0;
  }
  frame._lock.unlock();
}

void SlidingWindow::updateFrame(Frame &targetFrame, Frame &sourceFrame) {
  targetFrame._lock.lock();
  sourceFrame._lock.lock();
  assert(&targetFrame == sourceFrame._preFrame);
  //Update the preIndexes and preCounter
  for (auto sourceFeature = sourceFrame._features.begin(); sourceFeature != sourceFrame._features.end();
       sourceFeature++) {
    int &index = sourceFeature->_preFeature;
    if (index != -1) {
      index = targetFrame._features[index]._preFeature;
    }
  }
  //Update the PreFrame
  sourceFrame._preFrame = targetFrame._preFrame;
  //Delte the leftover Frame
  targetFrame._lock.unlock();
  sourceFrame._lock.unlock();
  delete &targetFrame;
}

void SlidingWindow::mergeFrame(Frame &targetFrame, Frame &sourceFrame) {
  //Lock Frames
  sourceFrame._lock.lock();
  targetFrame._lock.lock();
  //Calculate the difference
  unsigned int difference = targetFrame._features.size() - sourceFrame._features.size();
  if (difference > 0) { //Only if there are further features run the algorithm
    //Iterate through the 'new' Features and add them to before Frame
    targetFrame._features.reserve(difference);
    for (auto feature = sourceFrame._features.begin(); feature != sourceFrame._features.end(); feature++) {
      if (feature->_preFeature == -1) { //If there is no precessor, it is a new Feature
        targetFrame._features.push_back(*feature);
        //Set the connection right
        feature->_preFeature = (targetFrame._features.size() - 1); //The las Index
      }
    }
  }
  //Remove the Frames in between
  Frame *frame = &sourceFrame;
  while (frame->_preFrame != &targetFrame) {//Remove the Preframe if its not the target Frame
    Frame *deleteFrame = frame->_preFrame;
    //Connect the pre correct
    frame->_preFrame = deleteFrame->_preFrame;
    delete deleteFrame;
  }
  targetFrame._lock.unlock();
  sourceFrame._lock.unlock();
}

void SlidingWindow::calculateFeaturePreCounter(Frame &frame) {
  frame._lock.lock();
  assert(frame._preFrame != nullptr);
  frame._preFrame->_lock.lock();
  Frame &preFrame = *(frame._preFrame);
  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    if (feature->_preFeature >= 0) {
      feature->_preFeatureCounter = preFrame._features[feature->_preFeature]._preFeatureCounter + 1;
    }
  }
  frame._preFrame->_lock.unlock();
  frame._lock.unlock();
}

cv::Vec3d SlidingWindow::getBaseLineToPrevious(const Frame &frame) {
  frame._lock.lock();
  auto baseLine = frame._baseLine;
  frame._lock.unlock();
  return baseLine;
}

const cv::Mat &SlidingWindow::getImagePyramid(Frame &frame) {
  return frame._image;
}

void SlidingWindow::setBaseLineToPrevious(Frame &frame, const cv::Vec3d &baseLine) {
  frame._lock.lock();
  frame._baseLine = baseLine;
  frame._lock.lock();
}

/*
  void SlidingWindow::exportMatlabData() {
    cv::Vec3d &st0 = this->getPosition(0);
    cv::Vec3d &st1 = this->getPosition(1);
    cv::Vec3d &st2 = this->getPosition(2);

    double n0 = cv::norm(st0 - st1);
    double n1 = cv::norm(st1 - st2);
    cv::Vec3d u0 = (st0 - st1) / n0;
    cv::Vec3d u1 = (st1 - st2) / n1;

    double a0 = 0.0;    // A0
    double b0 = 0.0;    // B0
    double t0 = 1.0;    // T0
    double x0 = u0(0);  // X0
    double y0 = u0(1);  // Y0
    double z0 = u0(2);  // Z0

    double a1 = 0.0;    // A1
    double b1 = 0.0;    // B1
    double t1 = 1.0;    // T1
    double x1 = u1(0);  // X1
    double y1 = u1(1);  // Y1
    double z1 = u1(2);  // Z1                                                                         // Z1

    cv::Matx33d R0 = this->getRotation(0);
    cv::Matx33d R1 = this->getRotation(1);
    cv::Matx33d R2 = this->getRotation(2);

    std::vector<cv::Vec3d> m0, m1, m2;
    std::vector<std::vector<cv::Vec3d> *> vectors{&(m0), &(m1), &(m2)};

    this->getCorrespondingFeatures(3 - 1, 0, vectors);

    // File
    std::ofstream myfile;
    myfile.open("/home/franek/Repos/ba/MATLAB2/dataTEMP.m");

    myfile << "R0 = ..." << std::endl;
    myfile << "[" << R0(0, 0) << " " << R0(0, 1) << " " << R0(0, 2) << ";" << std::endl;
    myfile << R0(1, 0) << " " << R0(1, 1) << " " << R0(1, 2) << ";" << std::endl;
    myfile << R0(2, 0) << " " << R0(2, 1) << " " << R0(2, 2) << "];" << std::endl;

    myfile << std::endl;
    myfile << std::endl;
    myfile << std::endl;

    myfile << "R1 = ..." << std::endl;
    myfile << "[" << R1(0, 0) << " " << R1(0, 1) << " " << R1(0, 2) << ";" << std::endl;
    myfile << R1(1, 0) << " " << R1(1, 1) << " " << R1(1, 2) << ";" << std::endl;
    myfile << R1(2, 0) << " " << R1(2, 1) << " " << R1(2, 2) << "];" << std::endl;

    myfile << std::endl;
    myfile << std::endl;
    myfile << std::endl;

    myfile << "R2 = ..." << std::endl;
    myfile << "[" << R2(0, 0) << " " << R2(0, 1) << " " << R2(0, 2) << ";" << std::endl;
    myfile << R2(1, 0) << " " << R2(1, 1) << " " << R2(1, 2) << ";" << std::endl;
    myfile << R2(2, 0) << " " << R2(2, 1) << " " << R2(2, 2) << "];" << std::endl;

    myfile << std::endl;
    myfile << std::endl;
    myfile << std::endl;

    myfile << "a0 = " << a0 << ";" << std::endl;
    myfile << "b0 = " << b0 << ";" << std::endl;
    myfile << "t0 = " << t0 << ";" << std::endl;
    myfile << "a1 = " << a1 << ";" << std::endl;
    myfile << "b1 = " << b1 << ";" << std::endl;
    myfile << "t1 = " << t1 << ";" << std::endl;

    myfile << "x0 = " << x0 << ";" << std::endl;
    myfile << "y0 = " << y0 << ";" << std::endl;
    myfile << "z0 = " << z0 << ";" << std::endl;

    myfile << "x1 = " << x1 << ";" << std::endl;
    myfile << "y1 = " << y1 << ";" << std::endl;
    myfile << "z1 = " << z1 << ";" << std::endl;

    myfile << "m0 = ..." << std::endl;
    myfile << "[";
    for (unsigned int i = 0; i < m0.size(); i++) {  // First Row
      myfile << m0[i](0) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m0.size(); i++) {  // Second Row
      myfile << m0[i](1) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m0.size(); i++) {  // Third Row
      myfile << m0[i](2) << " ";
    }
    myfile << "];" << std::endl;

    myfile << "m1 = ..." << std::endl;
    myfile << "[";
    for (unsigned int i = 0; i < m1.size(); i++) {  // First Row
      myfile << m1[i](0) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m1.size(); i++) {  // Second Row
      myfile << m1[i](1) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m1.size(); i++) {  // Third Row
      myfile << m1[i](2) << " ";
    }
    myfile << "];" << std::endl;

    myfile << "m2 = ..." << std::endl;
    myfile << "[";
    for (unsigned int i = 0; i < m2.size(); i++) {  // First Row
      myfile << m2[i](0) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m2.size(); i++) {  // Second Row
      myfile << m2[i](1) << " ";
    }
    myfile << ";" << std::endl;
    for (unsigned int i = 0; i < m2.size(); i++) {  // Third Row
      myfile << m2[i](2) << " ";
    }
    myfile << "];" << std::endl;

    myfile.close();
  }*/
