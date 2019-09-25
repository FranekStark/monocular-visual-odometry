#include "SlidingWindow.hpp"

#include <ros/ros.h>

#include <fstream>
#include <iostream>

SlidingWindow::SlidingWindow(int len, unsigned int features) : _maxFeatures(features), _length(len), _frameCounter(0),
                                                               _frameNow(nullptr) {
}

SlidingWindow::~SlidingWindow() {
  /*Remove Frames*/
  auto *frame = _frameNow;
  while (frame != nullptr) {
    auto nextFrame = frame->_preFrame;
    delete frame;
    frame = nextFrame;
  }
}

Frame &SlidingWindow::getFrame(unsigned int past) const {
  //TODO: Ring-Buffer
  Frame *frame = _frameNow;
  unsigned int cnt = 0;
  while (cnt < past) {
    assert(frame != nullptr);
    frame = frame->_preFrame;
    cnt++;
  }
  return *frame;
}

/**
 *The Tracked FeaturesVectors (including Found) must have the same Size as the FeatureVector of the PreFrame. Also Each
 *tracked Feature has to be the Flollowing of the Feature with the same Index in the PreFrame.
 **/
void SlidingWindow::newFrame(const std::vector<cv::Point2f> &trackedFeaturesNow,
                             const std::vector<cv::Vec3d> &trackedFeaturesNowE, const std::vector<unsigned char> &found,
                             cv::Mat image) {
  assert(trackedFeaturesNow.size() == trackedFeaturesNowE.size() && trackedFeaturesNowE.size() == found.size());

  if (_frameNow != nullptr && _frameNow->_type == TEMP) {
    /*Just Update the Positions, and if nescecerry remove Poisitons*/
    auto frameFeatureIt = _frameNow->_features.begin();
    auto trackedFeaturesNowIt = trackedFeaturesNow.begin();
    auto trackedFeaturesNowEIt = trackedFeaturesNowE.begin();
    auto foundIt = found.begin();

    std::vector<std::vector<Feature>::iterator> toRemove; //TODO: Should remove them?

    while (foundIt != found.end()) {

      if (*foundIt == 1) {
        frameFeatureIt->_positionEuclidian = *trackedFeaturesNowEIt;
        frameFeatureIt->_positionImage = *trackedFeaturesNowIt;

      } else {
        toRemove.push_back(frameFeatureIt);
      }

      frameFeatureIt++;
      trackedFeaturesNowIt++;
      trackedFeaturesNowEIt++;
      foundIt++;
    }

    auto toRemoveIt = toRemove.begin();
    for (; toRemoveIt != toRemove.end(); toRemoveIt++) {
      _frameNow->_features.erase(*toRemoveIt);
    }

    _frameNow->_image = image;
  } else {
    /*Create New Frame*/
    Frame *newFrame = new Frame();
    /*Clip In*/
    newFrame->_preFrame = _frameNow;
    newFrame->_type = TEMP;
    _frameNow = newFrame;
    _frameNow->_preFrame->_afterFrame = _frameNow;
    _frameCounter++;
    if (_frameCounter > _length) {
      /*Remove Oldest*/
      Frame &afterOldestFrame = this->getFrame(_length);
      delete afterOldestFrame._preFrame;
      afterOldestFrame._preFrame = nullptr;
      _frameCounter--;
    }

    /*Add Image*/
    newFrame->_image = image;
    if (_frameCounter > 1) {
      /*Find Pairs and Add them*/
      auto &preFeatures = newFrame->_preFrame->_features;
      assert(trackedFeaturesNow.size() ==
          preFeatures.size());  // The others have the same Size as the first one, this
      // is already tested above.
      auto preFeatureIt = preFeatures.begin();
      auto trackedFeatureNowIt = trackedFeaturesNow.begin();
      auto trackedFeatureNowEIt = trackedFeaturesNowE.begin();
      auto foundIt = found.begin();
      while (foundIt != found.end()) {
        if (*foundIt == 1) {
          newFrame->_features.push_back(Feature(*trackedFeatureNowIt, *trackedFeatureNowEIt,
                                                std::distance(preFeatures.begin(), preFeatureIt),
                                                preFeatureIt->_preFeatureCounter + 1));
        }
        preFeatureIt++;
        trackedFeatureNowIt++;
        trackedFeatureNowEIt++;
        foundIt++;
      }
    }
  }
}

void SlidingWindow::newFrame(cv::Mat image, image_geometry::PinholeCameraModel cameraModel) {
  if (_frameNow != nullptr && _frameNow->_type == TEMP) {
    _frameNow->_image = image;
    _frameNow->_cameraModel = cameraModel;
  } else {
    /*Create New Frame*/
    Frame *newFrame = new Frame();
    /*Clip In*/
    newFrame->_preFrame = _frameNow;
    newFrame->_type = TEMP;
    _frameNow = newFrame;
    _frameNow->_preFrame->_afterFrame = _frameNow;
    _frameCounter++;
    if (_frameCounter > _length) {
      /*Remove Oldest*/
      Frame &afterOldestFrame = this->getFrame(_length);
      delete afterOldestFrame._preFrame;
      afterOldestFrame._preFrame = nullptr;
      _frameCounter--;
    }
    /*Add Image*/
    newFrame->_image = image;
    newFrame->_cameraModel = cameraModel;
  }
}

void SlidingWindow::addTrackedFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                                       const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                                       const std::vector<unsigned char> &found,
                                       Frame &frame) {
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
      /*Push Back the tracked Feature (with increased number of predecessor */
      frame._features.push_back(Feature(*trackedFeature,
                                        *trackedFeatureE,
                                        indexOfBeforeFeature,
                                        0)); //Set it to 0, it will be correct set in Merger
    }
    trackedFeature++;
    trackedFeatureE++;
    featureFound++;
  }
}

void SlidingWindow::updateFeatures(const std::vector<cv::Point2f> &trackedFeaturesNow,
                                   const std::vector<cv::Vec3d> &trackedFeaturesNowE,
                                   const std::vector<unsigned char> &found) {
  assert(_frameCounter > 0);
  assert(trackedFeaturesNow.size() == trackedFeaturesNowE.size());
  assert(found.size() == trackedFeaturesNowE.size());
  assert(trackedFeaturesNowE.size() == _frameNow->_features.size());

  /*Indices to remove*/
  std::vector<unsigned int> toRemove;
  toRemove.reserve(found.size() * 0.2); //Best Guess for needed Size

  /* Iterate trough the tracked Features and update them Frame*/
  auto trackedFeature = trackedFeaturesNow.begin();
  auto trackedFeatureE = trackedFeaturesNowE.begin();
  auto featureFound = found.begin();
  auto knownFeature = _frameNow->_features.begin();
  while (featureFound != found.end()) {
    if (*featureFound == 1) { //If the feature has been found
      knownFeature->_positionImage = *trackedFeature;
      knownFeature->_positionEuclidian = *trackedFeatureE;
    } else {
      /* Keep the Index of the lost Feature, to later remove */
      toRemove.push_back(std::distance(_frameNow->_features.begin(), knownFeature));
    }
    trackedFeature++;
    trackedFeatureE++;
    featureFound++;
    knownFeature++;
  }

  /*Because we Iterate from the begin to the end of the known features, the biggest indices in toRemove are at the end */
  for (auto removeIndex = toRemove.rbegin(); removeIndex != toRemove.rend(); removeIndex++) {
    /*Set the afterindex on predecessor*/
    int indexBeforeFeature = _frameNow->_features[*removeIndex]._preFeature;
    if (indexBeforeFeature >= 0) {
      _frameNow->_preFrame->_features[indexBeforeFeature]._afterFeature = -1;
    }
    /*If the Index is the last one, just remove it. If not, replace it through the Last One.
     *This Method only changed indices of elements bigger than the current. And as  explained above, there wouldn't be any
     * Element to remove with an index bigger than the current.*/
    if (*removeIndex == (_frameNow->_features.size() - 1)) { //Last Feature
      _frameNow->_features.pop_back(); //Remove Last Feature
    } else {
      _frameNow->_features[*removeIndex] = _frameNow->_features.back(); //replace through last
      _frameNow->_features.pop_back();//remove Last one, cause it is on new position
    }
  }
}

void SlidingWindow::persistCurrentFrame() {
  assert(_frameNow->_type == TEMP);
  _frameNow->_type = PERSIST;
}

void SlidingWindow::addNewFeaturesToBeforeFrame(const std::vector<cv::Point2f> &features,
                                                const std::vector<cv::Vec3d> &featuresE) { //TODO: afterFaetures!
  assert(features.size() == featuresE.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  while (featureIT != features.end()) {
    _frameNow->_preFrame->_features.push_back(Feature(*featureIT, *featureEIT, -1, 0));
    _frameNow->_features.push_back(Feature(*featureIT, *featureEIT, _frameNow->_preFrame->_features.size() - 1, 1));
    featureEIT++;
    featureIT++;
  }
}

void SlidingWindow::addFeaturesToFrame(Frame &frame,
                                       const std::vector<cv::Point2f> &features,
                                       const std::vector<cv::Vec3d> &featuresE) {
  assert(features.size() == featuresE.size());
  frame._features.reserve(features.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  while (featureIT != features.end()) {
    frame._features.push_back(Feature(*featureIT, *featureEIT, -1, 0));
    featureEIT++;
    featureIT++;
  }
}

void SlidingWindow::addNewFeaturesToFrame(const std::vector<cv::Point2f> &features,
                                          const std::vector<cv::Vec3d> &featuresE, unsigned int past) {
  assert(features.size() == featuresE.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  auto &frame = this->getFrame(past);
  while (featureIT != features.end()) {
    frame._features.push_back(Feature(*featureIT, *featureEIT, -1, 0));

    featureEIT++;
    featureIT++;
  }
}

void SlidingWindow::getFeatures(unsigned int past, std::vector<cv::Point2f> &features) const {
  Frame &frame = this->getFrame(past);
  features.resize(frame._features.size());
  auto featureFrom = frame._features.begin();
  auto featureTo = features.begin();
  while (featureFrom != frame._features.end()) {
    *featureTo = featureFrom->_positionImage;
    featureFrom++;
    featureTo++;
  }
}

void SlidingWindow::getFeatures(unsigned int past, std::vector<cv::Vec3d> &features) const {
  Frame &frame = this->getFrame(past);
  features.resize(frame._features.size());
  auto featureFrom = frame._features.begin();
  auto featureTo = features.begin();
  while (featureFrom != frame._features.end()) {
    *featureTo = featureFrom->_positionEuclidian;
    featureFrom++;
    featureTo++;
  }
}

const cv::Mat SlidingWindow::getImage(unsigned int past) const {
  return this->getFrame(past)._image;
}

cv::Vec3d &SlidingWindow::getPosition(unsigned int past) const {
  return this->getFrame(past)._position;
}

cv::Matx33d &SlidingWindow::getRotation(unsigned int past) const {
  return this->getFrame(past)._rotation;
}

void SlidingWindow::setPosition(const cv::Vec3d &position, unsigned int past) {
  this->getFrame(past)._position = position;
}

void SlidingWindow::setRotation(const cv::Matx33d &rotation, unsigned int past) {
  this->getFrame(past)._rotation = rotation;
}

/*void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<cv::Point2f> &features1,
                                             std::vector<cv::Point2f> &features2) const {
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  auto &features2SW = this->getFrame(window2Index)._features;

  for (auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++) {
    if (feature2->_preFeatureCounter >= depth) {
      features2.push_back(feature2->_positionImage);
      features2.push_back(getRandom<T>(feature2));

      Feature *feature = &(*feature2);
      Frame *frame = &(this->getFrame(window2Index));
      for (unsigned int i = 0; i < depth; i++) {
        feature = &frame->_preFrame->_features[feature->_preFeature];
        frame = frame->_preFrame;
      }
      features1.push_back(feature->_positionImage);
    }
  }
}*/

template<typename T>
void SlidingWindow::getFeatures(const Frame &frame, std::vector<T> &features) {
  //Check wether the Vector is empty
  assert(features.size() == 0);
  //Reserve needed memory
  features.reserve(frame._features.size());
  //Pushback all Features
  for (auto feature = frame._features.begin(); feature != frame._features.end(); feature++) {
    features.push_back(getFeatureLocation<T>(feature));
  }
}

template<typename T>
void SlidingWindow::getCorrespondingFeatures(const Frame &oldestFrame,
                                             const Frame &newestFrame,
                                             std::vector<std::vector<T> *> features) {
/*Calculate Depth and maximum Number Of Features  and check wether the params are okay*/
  unsigned int depth = 0;
  const Frame *frame = &newestFrame;
  auto outFeatures = features.begin();
  while (frame != &oldestFrame) {
    depth++;
    assert(*outFeatures != nullptr);
    assert((*outFeatures)->size() == 0);
    (*outFeatures)->reserve(frame->_features.size());
    frame = frame->_preFrame;
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

}

const cv::Matx33d &SlidingWindow::getRotation(const Frame &frame) {
  return frame._rotation;
}

const image_geometry::PinholeCameraModel &SlidingWindow::getCameraModel(const Frame &frame) {
  return frame._cameraModel;
}

template<>
const cv::Point2f &SlidingWindow::getFeatureLocation(const Feature &f) {
  return f._positionImage;
}

template<>
const cv::Vec3d &SlidingWindow::getFeatureLocation(const Feature &f) {
  return f._positionEuclidian;
}

void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<cv::Vec3d> &features1,
                                             std::vector<cv::Vec3d> &features2) const {
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  auto &features2SW = this->getFrame(window2Index)._features;

  for (auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++) {
    if (feature2->_preFeatureCounter >= depth) {
      features2.push_back(feature2->_positionEuclidian);
      Feature *feature = &(*feature2);
      Frame *frame = &(this->getFrame(window2Index));
      for (unsigned int i = 0; i < depth; i++) {
        feature = &frame->_preFrame->_features[feature->_preFeature];
        frame = frame->_preFrame;
      }
      features1.push_back(feature->_positionEuclidian);
    }
  }
}

void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<std::vector<cv::Vec3d> *> features) const {
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  assert(features.size() == (depth + 1));

  auto &features2SW = this->getFrame(window2Index)._features;

  for (auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++) {
    if (feature2->_preFeatureCounter >= depth) {
      Feature *feature = &(*feature2);
      Frame *frame = &(this->getFrame(window2Index));
      for (unsigned int i = 0; i <= depth; i++) {
        features[i]->push_back(feature->_positionEuclidian);
        if (feature->_preFeature != -1) {
          feature = &(frame->_preFrame->_features[feature->_preFeature]);
        }
        frame = frame->_preFrame;
      }
    }
  }
}

void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<std::vector<cv::Point2f> *> features) const {
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  assert(features.size() == (depth + 1));

  auto &features2SW = this->getFrame(window2Index)._features;

  for (auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++) {
    if (feature2->_preFeatureCounter >= depth) {
      Feature *feature = &(*feature2);
      Frame *frame = &(this->getFrame(window2Index));
      for (unsigned int i = 0; i <= depth; i++) {
        features[i]->push_back(feature->_positionImage);
        if (feature->_preFeature != -1) {
          feature = &(frame->_preFrame->_features[feature->_preFeature]);
        }
        frame = frame->_preFrame;
      }
    }
  }
}

unsigned int SlidingWindow::getNumberOfCurrentTrackedFeatures() const {
  if (_frameNow == nullptr) {
    return 0;
  } else {
    return _frameNow->_features.size();
  }
}

unsigned int SlidingWindow::getNumberOfKnownFeatures(Frame &frame) {
  return frame._features.size();
}

/*
 * This also deletes the whole "FeatureWrapper"
 */
void SlidingWindow::removeFeatureFromCurrentWindow(const cv::Vec3d &feature) {
  auto featureIT = _frameNow->_features.begin();
  bool found = false;
  for (; featureIT != _frameNow->_features.end(); featureIT++) {
    if (featureIT->_positionEuclidian == feature) {
      found = true;
      break;
    }
  }
  if (found) {
    _frameNow->_features.erase(featureIT);
  }
}

image_geometry::PinholeCameraModel &SlidingWindow::getCameraModel(unsigned int past) {
  return this->getFrame(past)._cameraModel;
}

bool SlidingWindow::isTemporaryFrame(unsigned int past) const {
  return (this->getFrame(past)._type == FrameType::TEMP);
}

void SlidingWindow::removeFeaturesFromFrame(std::vector<unsigned int> &featureIndeces, unsigned int past) {
  /* first sort the Vector of indices ascending to make use of the procedure in 'update' */
  std::sort(featureIndeces.begin(), featureIndeces.end());
  Frame &frame = this->getFrame(past);
  for (auto index = featureIndeces.rbegin(); index != featureIndeces.rend(); index++) {
    /*Update Prefeature*/
    auto preFeatureIndex = frame._features[*index]._preFeature;
    if (preFeatureIndex >= 0) {
      frame._preFrame->_features[preFeatureIndex]._afterFeature = -1;
    }
    /*Update Afterfeature*/
    auto afterFrameIndex = frame._features[*index]._afterFeature;
    if (afterFrameIndex >= 0) {
      frame._afterFrame->_features[afterFrameIndex]._preFeature = -1;
    }
    /*If it is the Last Index, then only remove. Otherwise take last one on this place*/
    if (*index == (frame._features.size() - 1)) { //Remove simply last Feature
      frame._features.pop_back();
    } else { //Replace with the Last
      auto indexAfterFeature = frame._features.back()._afterFeature;
      auto indexPreFeature = frame._features.back()._preFeature;
      frame._features[*index] = frame._features.back();
      frame._features.pop_back();
      /*Inform After Feature*/
      if (indexAfterFeature >= 0) {
        frame._afterFrame->_features[indexAfterFeature]._preFeature = *index;
      }
      if (indexPreFeature >= 0) {
        frame._preFrame->_features[indexPreFeature]._afterFeature = *index;
      }

    }

  }

}

void SlidingWindow::disbandFeatureConnection(const std::vector<unsigned int> &indices, Frame &frame) {
  for (auto index : indices) {
    frame._features[index]._preFeature = -1;
    frame._features[index]._preFeatureCounter = 0;
  }
}

void SlidingWindow::updateFrame(Frame &targetFrame, Frame &sourceFrame) {
  assert(targetFrame._preFrame == &sourceFrame);
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
  delete &targetFrame;
}

void SlidingWindow::mergeFrame(Frame &targetFrame, Frame &sourceFrame) {
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
  Frame * frame = &sourceFrame;
  while(frame->_preFrame != &targetFrame){//Remove the Preframe if its not the target Frame
    Frame * deleteFrame = frame->_preFrame;
    //Connect the pre correct
    frame->_preFrame = deleteFrame->_preFrame;
    delete deleteFrame;
  }
}

void SlidingWindow::calculateFeaturePreCounter(Frame &frame) {
  assert(frame._preFrame != nullptr);
  Frame & preFrame = *(frame._preFrame);
  for(auto feature = frame._features.begin(); feature!=frame._features.end(); feature++){
    if(feature->_preFeature >= 0) {
      feature->_preFeatureCounter = preFrame._features[feature->_preFeature]._preFeatureCounter + 1;
    }
  }
}



  const cv::Vec3d &SlidingWindow::getBaseLineToPrevious(const Frame &frame) {
    return frame._baseLine;
  }

  Frame *SlidingWindow::getNewestFrame() {
    //TODO: only, when _frame not null
    return _frameNow;
  }

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
  }
