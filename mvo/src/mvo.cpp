#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), _frameCounter(0)
{
}

MVO::~MVO()
{
}

OdomData MVO::handleImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel, const cv::Matx33d &R){
  ROS_INFO_STREAM("Rotation: " << R << std::endl);
  cv::cvtColor(image, _debugImage, cv::ColorConversionCodes::COLOR_GRAY2RGB);

  std::vector<cv::Point2f> newFeatures(20);
  _cornerTracker.detectFeatures(newFeatures, image, 20); //TODO: Parametrice Number
  ROS_ERROR_STREAM_COND(newFeatures.size() < 4, "New Feature Set is very small: " << newFeatures.size() << std::endl);
  cv::Vec3d b(0,0,0);

  if(_frameCounter == 0){
     //ROS_INFO_STREAM("-> First Frame " << std::endl);
    _slidingWindow.newWindow(std::vector<cv::Point2f>(), std::vector<uchar>(), image); //First Two are Dummies
    _slidingWindow.addFeaturesToCurrentWindow(newFeatures); //all, because first Frame
    _slidingWindow.addTransformationToCurrentWindow(b, R);

    this->drawDebugPoints(newFeatures, cv::Scalar(0,0,255), _debugImage);
    //ROS_INFO_STREAM("<- First Frame " << std::endl);
  }else{
   std::vector<cv::Point2f> & prevFeatures = _slidingWindow.getFeatures(0);
   cv::Mat prevImage = _slidingWindow.getImage(0);
   std::vector<cv::Point2f> trackedFeatures;
   std::vector<unsigned char> found;
   //ROS_INFO_STREAM("->  Track Features " << std::endl);
   _cornerTracker.trackFeatures(prevImage, image, prevFeatures, trackedFeatures, found);
   //ROS_INFO_STREAM("<-  Track Features " << std::endl);

   this->drawDebugPoints(trackedFeatures, cv::Scalar(0,255,0), _debugImage);
    if(!(this->checkEnoughDisparity(trackedFeatures, prevFeatures))){
    OdomData od;
     od.b = b;
     od.s = _slidingWindow.getPosition(0);
     return od;
    }
    
   _slidingWindow.newWindow(trackedFeatures, found, image);
   this->sortOutSameFeatures(trackedFeatures, newFeatures);
   _slidingWindow.addFeaturesToCurrentWindow(newFeatures);

   this->drawDebugPoints(newFeatures, cv::Scalar(0,0,255), _debugImage);

  cv::Matx33d rBefore = _slidingWindow.getRotation(1);
  cv::Matx33d rDiff = rBefore.t() * R; //Difference Rotation
  //ROS_INFO_STREAM("Relative Rotation: " << rDiff << std::endl);
  std::vector<cv::Point2f> thisCorespFeatures, beforeCorespFeatures;
  std::vector<cv::Vec3d> thisCorespFeaturesE, beforeCorespFeaturesE;
  _slidingWindow.getCorrespondingFeatures(1,0,beforeCorespFeatures, thisCorespFeatures);
  this->euclidNormFeatures(thisCorespFeatures, thisCorespFeaturesE, cameraModel);
  this->euclidNormFeatures(beforeCorespFeatures, beforeCorespFeaturesE,cameraModel);
  std::vector<cv::Vec3d> thisCorespFeaturesEUnrotate;
  this->unrotateFeatures(thisCorespFeaturesE, thisCorespFeaturesEUnrotate, rDiff);

 //ROS_INFO_STREAM("-> Estimate Baseline " << std::endl);
  b = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesEUnrotate);
//ROS_INFO_STREAM("<- Estimate Baseline " << std::endl);
  ROS_INFO_STREAM("b: " << b << std::endl);
 //Scale vote 
   std::vector<double> depths(beforeCorespFeaturesE.size());
   //ROS_INFO_STREAM("-> Reconstruct Depth " << std::endl);
   this->reconstructDepth(depths, thisCorespFeaturesE, beforeCorespFeaturesE, rDiff, b);
   int sign = 0;
   for(auto depth = depths.begin(); depth!=depths.end(); depth++){
     if((*depth) < 0){
      sign--;
     }else if((*depth)>0){
      sign++;
     }
   }

   if(sign<0){
     b = b*-1;
   }
  // ROS_INFO_STREAM("<- Reconstruct Depth " << std::endl);
   ROS_INFO_STREAM("b after Depth: " << b << std::endl);
  
  if(false && _frameCounter > 1){//IterativeRefinemen -> Scale Estimation?
    std::vector<cv::Point2f> thisFirsCorespFeatures, beforeFirstCorepsFeatures;
    std::vector<cv::Vec3d> thisFirstCorespFeaturesE, beforeFirstCorespFeaturesE;
    _slidingWindow.getCorrespondingFeatures(2,0, beforeFirstCorepsFeatures, thisFirsCorespFeatures);
    const cv::Vec3d & shi = _slidingWindow.getPosition(2);
    cv::Vec3d st = _slidingWindow.getPosition(1) + b;
    //ROS_INFO_STREAM("st before Refinement: " << st << std::endl);
    auto rhi = _slidingWindow.getRotation(2);
    this->euclidNormFeatures(beforeFirstCorepsFeatures, beforeFirstCorespFeaturesE, cameraModel);
    this->euclidNormFeatures(thisFirsCorespFeatures, thisFirstCorespFeaturesE, cameraModel);
    _iterativeRefinement.iterativeRefinement(thisFirstCorespFeaturesE, R, beforeFirstCorespFeaturesE, rhi, shi, st);
    _slidingWindow.addTransformationToCurrentWindow(st, R);
  }else{
    _slidingWindow.addTransformationToCurrentWindow(_slidingWindow.getPosition(1) + b, R);
  }

}
this->drawDebugImage(b, _debugImage);
 _frameCounter++;
 OdomData od;
 od.b = b;
 od.s = _slidingWindow.getPosition(0);
 return od; 
}

void MVO::sortOutSameFeatures(const std::vector<cv::Point2f> & beforeFeatures, std::vector<cv::Point2f> & newFeatures){
  for(auto beforeFeature = beforeFeatures.begin(); beforeFeature != beforeFeatures.end(); beforeFeature++){
    for(auto newFeature = newFeatures.begin(); newFeature != newFeatures.end(); newFeature++){
      double distance = cv::norm((*beforeFeature)-(*newFeature));
      if(distance < 20){ //TODO: Pram Threshold
        newFeatures.erase(newFeature);
        break;
      }
    }
  }
}

void MVO::euclidNormFeatures(const std::vector<cv::Point2f> &features, std::vector<cv::Vec3d> & featuresE,const image_geometry::PinholeCameraModel & cameraModel){
  for(auto feature = features.begin(); feature != features.end(); feature++){
    featuresE.push_back(cameraModel.projectPixelTo3dRay(*feature));
  }
}

void MVO::drawDebugImage(const cv::Vec3d baseLine, cv::Mat & image){
  auto baseLineNorm =  cv::normalize(baseLine);

 int mitX = double(image.cols) / 2.0;
    int mitY = double(image.rows) / 2.0;
    double scaleX = (image.cols - mitX) / 1.5;
    double scaleY = (image.rows - mitY) / 1.5;
    cv::arrowedLine(image, cv::Point(mitX, mitY),
                    cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY),
                    cv::Scalar(0, 255, 0), 10);
    cv::line(image, cv::Point(mitY, 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), 20), cv::Scalar(0, 255, 0),
             10);
}

void MVO::drawDebugPoints(const std::vector<cv::Point2f> & points, const cv::Scalar & color, cv::Mat & image){
  for (auto point = points.begin(); point != points.end(); point++)
  {
    cv::circle(image, cv::Point(*point), 10, color, -10);
    std::string index;
    cv::putText(image,std::to_string(std::distance(points.begin(), point)), cv::Point(*point),cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255,255,255));
  }
}


void MVO::reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                           const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r,
                           const cv::Vec3d &b)
{
  assert(m1L.size() == m2L.size());
  for (auto m1 = m1L.begin(), m2 = m2L.begin(); m1 != m1L.end() && m2 != m2L.end(); m1++, m2++)
  {
    cv::Matx33d C;
    C << 1, 0, -(*m2)(0),  //
        0, 1, -(*m2)(1),   //
        -(*m2)(0), -(*m2)(1), (*m2)(0) * (*m2)(0) + (*m2)(1) * (*m2)(1);
    double Z = (m1->t() * r * C * r.t() * b)(0) / (m1->t() * r * C * r.t() * (*m1))(0);
    depth.push_back(Z);
  }
}

bool MVO::checkEnoughDisparity(std::vector<cv::Point2f> & first, std::vector<cv::Point2f> & second){
  assert(first.size() == second.size());
  //ROS_INFO_STREAM("Points: " << std::endl);
  double diff = 0;
  for(auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++){
    diff += cv::norm((*p1)-(*p2));
    //ROS_INFO_STREAM(*p1 << " - " << *p2 << std::endl);
  }
  diff = diff / first.size();
  //ROS_INFO_STREAM("diff: " << diff << std::endl);
  return diff > 10; //TODO: Thresh
}

void MVO::unrotateFeatures(const std::vector<cv::Vec3d> & features, std::vector<cv::Vec3d> & unrotatedFeatures, const cv::Matx33d & R){
  for(auto feature = features.begin(); feature != features.end(); feature++){
    auto unrotatedFeature = R * (*feature);
    unrotatedFeature = unrotatedFeature / unrotatedFeature(2);
    unrotatedFeatures.push_back(unrotatedFeature);
  }
}