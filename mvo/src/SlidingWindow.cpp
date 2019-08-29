#include "SlidingWindow.hpp"

#include <ros/ros.h>

#include <iostream>
#include <fstream>

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
 * This also deletes the whole "FeatureWrapper"
 */
void SlidingWindow::removeFeatureFromCurrentWindow(const cv::Vec3d& feature)
{
  auto featureIT = _frameNow->_features.begin();
  bool found = false;
  for(; featureIT != _frameNow->_features.end(); featureIT++){
    if(featureIT->_positionEuclidian == feature){
      found = true;
      break;
    }
  }
  if(found){
    _frameNow->_features.erase(featureIT);
  }
}

void SlidingWindow::exportMatlabData(){
  

  cv::Vec3d& st0 = this->getPosition(0);
  cv::Vec3d& st1 = this->getPosition(1);
  cv::Vec3d& st2 = this->getPosition(2);

  double n0 = cv::norm(st0 - st1);
  double n1 = cv::norm(st1 - st2);
  cv::Vec3d u0 = (st0 - st1) / n0;
  cv::Vec3d u1 = (st1 - st2) / n1;


  double a0 = 0.0;                                                                                // A0
  double b0 = 0.0;                                                                                // B0
  double t0 = 1.0;                                                                                // T0
  double x0 = u0(0);                                                                              // X0
  double y0 = u0(1);                                                                              // Y0
  double z0 = u0(2);                                                                              // Z0

  double a1 = 0.0;                                                                                // A1
  double b1 = 0.0;                                                                                // B1
  double t1 = 1.0;                                                                                // T1
  double x1 = u1(0);                                                                              // X1
  double y1 = u1(1);                                                                              // Y1
  double z1 = u1(2);                                                                              // Z1                                                                         // Z1

  cv::Matx33d R0 = this->getRotation(0);
  cv::Matx33d R1 = this->getRotation(1);
  cv::Matx33d R2 = this->getRotation(2);

  std::vector<cv::Vec3d> m0, m1, m2;
  std::vector<std::vector<cv::Vec3d>*> vectors{ &(m0), &(m1), &(m2) };

  this->getCorrespondingFeatures(3 - 1, 0, vectors);

  //File
  std::ofstream myfile;
  myfile.open("/home/franek/Repos/ba/MATLAB2/dataTEMP.m");

  myfile << "R0 = ..." << std::endl;
  myfile << "[" << R0(0,0) << " " << R0(0,1) << " " << R0(0,2) << ";" << std::endl;
  myfile        << R0(1,0) << " " << R0(1,1) << " " << R0(1,2) << ";" << std::endl;
  myfile        << R0(2,0) << " " << R0(2,1) << " " << R0(2,2) << "];" << std::endl;

  myfile << std::endl;
  myfile << std::endl;
  myfile << std::endl;

  myfile << "R1 = ..." << std::endl;
  myfile << "[" << R1(0,0) << " " << R1(0,1) << " " << R1(0,2) << ";" << std::endl;
  myfile        << R1(1,0) << " " << R1(1,1) << " " << R1(1,2) << ";" << std::endl;
  myfile        << R1(2,0) << " " << R1(2,1) << " " << R1(2,2) << "];" << std::endl;

  myfile << std::endl;
  myfile << std::endl;
  myfile << std::endl;

  myfile << "R2 = ..." << std::endl;
  myfile << "[" << R2(0,0) << " " << R2(0,1) << " " << R2(0,2) << ";" << std::endl;
  myfile        << R2(1,0) << " " << R2(1,1) << " " << R2(1,2) << ";" << std::endl;
  myfile        << R2(2,0) << " " << R2(2,1) << " " << R2(2,2) << "];" << std::endl;

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
  for(unsigned int i = 0; i < m0.size(); i++){ //First Row
    myfile << m0[i](0) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m0.size(); i++){ //Second Row
    myfile << m0[i](1) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m0.size(); i++){ //Third Row
    myfile << m0[i](2) << " ";
  }
  myfile << "];" << std::endl;

   myfile << "m1 = ..." << std::endl;
  myfile << "[";
  for(unsigned int i = 0; i < m1.size(); i++){ //First Row
    myfile << m1[i](0) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m1.size(); i++){ //Second Row
    myfile << m1[i](1) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m1.size(); i++){ //Third Row
    myfile << m1[i](2) << " ";
  }
  myfile << "];" << std::endl;

  myfile << "m2 = ..." << std::endl;
  myfile << "[";
  for(unsigned int i = 0; i < m2.size(); i++){ //First Row
    myfile << m2[i](0) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m2.size(); i++){ //Second Row
    myfile << m2[i](1) << " ";
  }
  myfile << ";" << std::endl;
  for(unsigned int i = 0; i < m2.size(); i++){ //Third Row
    myfile << m2[i](2) << " ";
  }
  myfile << "];" << std::endl;




  myfile.close();

}

