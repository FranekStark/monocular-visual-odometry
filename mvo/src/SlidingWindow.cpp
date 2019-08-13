#include "SlidingWindow.hpp"

#include <ros/ros.h>

SlidingWindow::SlidingWindow(int len) : _length(len), _frameCounter(0), _frameNow(nullptr)
{
}

SlidingWindow::~SlidingWindow()
{
  /*Remove Frames*/
  auto* frame = _frameNow;
  while (frame != nullptr)
  {
    auto nextFrame = frame->_preFrame;
    delete frame;
    frame = nextFrame;
  }
}

Frame & SlidingWindow::getFrame(unsigned int past) const
{
  Frame* frame = _frameNow;
  unsigned int cnt = 0;
  while (cnt < past)
  {
    assert(frame != nullptr);
    frame = frame->_preFrame;
    cnt++;
  }
  return *frame;
}

/**
 *The Tracked FeaturesVectors (including Found) must have the same Size as the FeatureVector of the PreFrame. Also Each
 *tracked Feature has to be the followwing of the Feature with the same Index in the PreFrame.
 **/
void SlidingWindow::newFrame(const std::vector<cv::Point2f>& trackedFeaturesNow,
                             const std::vector<cv::Vec3d>& trackedFeaturesNowE, const std::vector<unsigned char>& found,
                             cv::Mat image)
{
  assert(trackedFeaturesNow.size() == trackedFeaturesNowE.size() && trackedFeaturesNowE.size() == found.size());
  /*Create New Frame*/
  Frame* newFrame = new Frame();
  /*Clip In*/
  newFrame->_preFrame = _frameNow;
  _frameNow = newFrame;
  _frameCounter++;
  if (_frameCounter > _length)
  {
    /*Remove Oldest*/
    Frame & afterOldestFrame = this->getFrame(_length);
    delete afterOldestFrame._preFrame;
    afterOldestFrame._preFrame = nullptr;
    _frameCounter--;
  }

  /*Add Image*/
  newFrame->_image = image;
  if(_frameCounter > 1){
  /*Find Pairs and Add them*/
  auto & preFeatures = newFrame->_preFrame->_features;
  assert(trackedFeaturesNow.size() == preFeatures.size());  // The others have the same Size as the first one, this is
                                                            // already tested above.
  auto preFeatureIt = preFeatures.begin();
  auto trackedFeatureNowIt = trackedFeaturesNow.begin();
  auto trackedFeatureNowEIt = trackedFeaturesNowE.begin();
  auto foundIt = found.begin();
  while (foundIt != found.end())
  {
    if (*foundIt == 1)
    {
      newFrame->_features.push_back(Feature(
        *trackedFeatureNowIt,
        *trackedFeatureNowEIt,
        &(*preFeatureIt),
        preFeatureIt->_preFeatureCounter + 1
      ));
    }
    preFeatureIt++;
    trackedFeatureNowIt++;
    trackedFeatureNowEIt++;
    foundIt++;
  }
  }
}

void SlidingWindow::addNewFeaturesToCurrentFrame(const std::vector<cv::Point2f>& features,
                                                 const std::vector<cv::Vec3d>& featuresE)
{
  assert(features.size() == featuresE.size());
  auto featureIT = features.begin();
  auto featureEIT = featuresE.begin();
  while (featureIT != features.end())
  {
    _frameNow->_features.push_back(Feature(
      *featureIT,
      *featureEIT,
      nullptr,
      0
    ));

    featureEIT++;
    featureIT++;
  }
}

void SlidingWindow::getFeatures(unsigned int past, std::vector<cv::Point2f> & features) const
{ 
  Frame & frame = this->getFrame(past);
  features.resize(frame._features.size());
  auto featureFrom = frame._features.begin();
  auto featureTo = features.begin();
  while(featureFrom != frame._features.end()){
    *featureTo = featureFrom->_positionImage;
    featureFrom++;
    featureTo++;
  }
}

void SlidingWindow::getFeatures(unsigned int past, std::vector<cv::Vec3d> & features) const
{ 
  Frame & frame = this->getFrame(past);
  features.resize(frame._features.size());
  auto featureFrom = frame._features.begin();
  auto featureTo = features.begin();
  while(featureFrom != frame._features.end()){
    *featureTo = featureFrom->_positionEuclidian;
    featureFrom++;
    featureTo++;
  }
}

const cv::Mat SlidingWindow::getImage(unsigned int past) const
{
  return this->getFrame(past)._image;
}

cv::Vec3d& SlidingWindow::getPosition(unsigned int past) const
{
  return this->getFrame(past)._position;
}

cv::Matx33d& SlidingWindow::getRotation(unsigned int past) const
{
  return this->getFrame(past)._rotation;
}


void SlidingWindow::setPosition(const cv::Vec3d & position, unsigned int past){
  this->getFrame(past)._position = position;
}

void SlidingWindow::setRotation(const cv::Matx33d & rotation, unsigned int past){
  this->getFrame(past)._rotation = rotation;
}



void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Point2f>& features1,
                                             std::vector<cv::Point2f>& features2) const
{
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  auto & features2SW = this->getFrame(window2Index)._features;

  for(auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++){
    if(feature2->_preFeatureCounter >= depth){
      features2.push_back(feature2->_positionImage);
      Feature * feature = &(*feature2);
      for(unsigned int i = 0; i < depth; i++){
        feature = feature->_preFeature;
      }
      features1.push_back(feature->_positionImage);
    }
  }

}

void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Vec3d>& features1,
                                             std::vector<cv::Vec3d>& features2) const
{
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  auto & features2SW = this->getFrame(window2Index)._features;

  for(auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++){
    if(feature2->_preFeatureCounter >= depth){
      features2.push_back(feature2->_positionEuclidian);
      Feature * feature = &(*feature2);
      for(unsigned int i = 0; i < depth; i++){
        feature = feature->_preFeature;
      }
      features1.push_back(feature->_positionEuclidian);
    }
  }

}


void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<std::vector<cv::Vec3d>*> features) const
{
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  assert(features.size() == (depth + 1));

  auto & features2SW = this->getFrame(window2Index)._features;

  for(auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++){
    if(feature2->_preFeatureCounter >= depth){
      Feature * feature = &(*feature2);
      for(unsigned int i = 0; i <= depth; i++){
        features[i]->push_back(feature->_positionEuclidian);
        feature = feature->_preFeature;;
      }
    }
  }
  
}

void SlidingWindow::getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index,
                                             std::vector<std::vector<cv::Point2f>*> features) const
{
  assert(window1Index < _frameCounter && window2Index < _frameCounter);
  unsigned int depth = window1Index - window2Index;
  assert(features.size() == (depth + 1));

  auto & features2SW = this->getFrame(window2Index)._features;

  for(auto feature2 = features2SW.begin(); feature2 != features2SW.end(); feature2++){
    if(feature2->_preFeatureCounter >= depth){
      Feature * feature = &(*feature2);
      for(unsigned int i = 0; i <= depth; i++){
        features[i]->push_back(feature->_positionImage);
        feature = feature->_preFeature;;
      }
    }
  }
  
}


unsigned int SlidingWindow::getNumberOfCurrentTrackedFeatures() const
{
  if (_frameNow == nullptr)
  {
    return 0;
  }
  else
  {
    return _frameNow->_features.size();
  }
}

/*
 * This also deletes tze hole "FeatureWrapper"
 */
void SlidingWindow::removeFeatureFromCurrentWindow(const cv::Vec3d& feature)
{
  ROS_INFO_STREAM("To Remove" << std::endl);
  auto featureIT = _frameNow->_features.begin();
  bool found = false;
  for(; featureIT != _frameNow->_features.end(); featureIT++){
    if(featureIT->_positionEuclidian == feature){
      found = true;
    }
  }
  if(found){
    _frameNow->_features.erase(featureIT);
    ROS_INFO_STREAM("removed" << std::endl);
  }
}

