#include "mvo.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <map>
#include <random>
MVO::MVO() : _slidingWindow(5), _frameCounter(0)
{
  cv::namedWindow("original", cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow("cornerImage", cv::WINDOW_GUI_EXPANDED);
}

MVO::~MVO()
{
  cv::destroyWindow("original");
  cv::destroyWindow("cornerImage");
}

void MVO::handleImage(const cv::Mat image, const image_geometry::PinholeCameraModel &cameraModel){
  cv::imshow("original", image);
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BayerBG2GRAY);

  std::vector<cv::Point2f> newFeatures(20);
  _cornerTracker.detectFeatures(newFeatures, grayImage, 20); //TODO: Parametrice Number
  cv::Matx33d r = cv::Matx33d::eye();
  cv::Vec3d b(0,0,0);


  if(_frameCounter == 0){
    _slidingWindow.newWindow(std::vector<cv::Point2f>(), std::vector<uchar>(), grayImage); //First Two are Dummies
    _slidingWindow.addFeaturesToCurrentWindow(newFeatures); //all, because first Frame
    _slidingWindow.addTransformationToCurrentWindow(b, r);

  }else{
   std::vector<cv::Point2f> & prevFeatures = _slidingWindow.getFeatures(0);
   cv::Mat prevImage = _slidingWindow.getImage(0);
   std::vector<cv::Point2f> trackedFeatures;
   std::vector<unsigned char> found;
   _cornerTracker.trackFeatures(prevImage, grayImage, prevFeatures, trackedFeatures, found);
    if(!(this->checkEnoughDisparity(trackedFeatures, prevFeatures))){
      return;
    }
   _slidingWindow.newWindow(trackedFeatures, found, grayImage);
   this->sortOutSameFeatures(trackedFeatures, newFeatures);
   _slidingWindow.addFeaturesToCurrentWindow(newFeatures);

  std::vector<cv::Point2f> thisCorespFeatures, beforeCorespFeatures;
  std::vector<cv::Vec3d> thisCorespFeaturesE, beforeCorespFeaturesE; //TODO: Presize?
  _slidingWindow.getCorrespondingFeatures(1,0,beforeCorespFeatures, thisCorespFeatures);
  this->euclidNormFeatures(thisCorespFeatures, thisCorespFeaturesE, cameraModel);
  this->euclidNormFeatures(beforeCorespFeatures, beforeCorespFeaturesE,cameraModel);

  b = _epipolarGeometry.estimateBaseLine(beforeCorespFeaturesE, thisCorespFeaturesE, r);


  if(_frameCounter ==1){//Scale vote 
   std::vector<double> depths(beforeCorespFeaturesE.size());
   this->reconstructDepth(depths, thisCorespFeaturesE, beforeCorespFeaturesE, r, b);
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
  _slidingWindow.addTransformationToCurrentWindow(b, r);
  }else{//IterativeRefinemen -> Scale Estimation?
    std::vector<cv::Point2f> thisFirsCorespFeatures, beforeFirstCorepsFeatures;
    std::vector<cv::Vec3d> thisFirstCorespFeaturesE, beforeFirstCorespFeaturesE;
    _slidingWindow.getCorrespondingFeatures(2,0, beforeFirstCorepsFeatures, thisFirsCorespFeatures);
    cv::Vec3d & shi = _slidingWindow.getPosition(2);
    auto rhi = cv::Matx33d::eye(); //TODO real Rotation
    this->euclidNormFeatures(beforeFirstCorepsFeatures, beforeFirstCorespFeaturesE, cameraModel);
    this->euclidNormFeatures(thisFirsCorespFeatures, thisFirstCorespFeaturesE, cameraModel);
    _iterativeRefinement.iterativeRefinement(thisFirstCorespFeaturesE, r, beforeFirstCorespFeaturesE, rhi, shi, b);
    _slidingWindow.addTransformationToCurrentWindow(shi + b, r);
  }

 
 
  ROS_INFO_STREAM("b: " << b << std::endl);
}
 this->drawDebugImage(_slidingWindow.getFeatures(0), b, grayImage);
 cv::waitKey(1);
  _frameCounter++;
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

void MVO::drawDebugImage(const std::vector<cv::Point2f> points, const cv::Vec3d baseLine, cv::Mat & image){
   cv::Mat cornerImage = image.clone();
  std::stringstream text;
  text << "Number of Features: " << points.size();
  cv::putText(cornerImage, text.str(), cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255));
  for (auto const &corner :points)
  {
    cv::circle(cornerImage, cv::Point(corner), 10, cv::Scalar(0, 0, 255), -10);
  }

 auto baseLineNorm =  cv::normalize(baseLine);

 ROS_INFO_STREAM("Norm: " << baseLineNorm << std::endl);

 int mitX = double(cornerImage.cols) / 2.0;
    int mitY = double(cornerImage.rows) / 2.0;
    double scaleX = (cornerImage.cols - mitX) / 1.5;
    double scaleY = (cornerImage.rows - mitY) / 1.5;
    cv::arrowedLine(cornerImage, cv::Point(mitX, mitY),
                    cv::Point(scaleX * baseLineNorm(0) + mitX, (scaleY * baseLineNorm(1)) + mitY),
                    cv::Scalar(0, 255, 0), 10);
    cv::line(cornerImage, cv::Point(mitY, 20), cv::Point(mitY + (scaleX * baseLineNorm(2)), 20), cv::Scalar(0, 255, 0),
             10);
    cv::imshow("cornerImage", cornerImage);
}


void MVO::reconstructDepth(std::vector<double> &depth, const std::vector<cv::Vec3d> &m2L,
                           const std::vector<cv::Vec3d> &m1L, const cv::Matx33d &r,
                           const cv::Vec3d &b)
{
  // TODO: ROTATION
  (void)(r);
  assert(m1L.size() == m2L.size());
  for (auto m1 = m1L.begin(), m2 = m2L.begin(); m1 != m1L.end() && m2 != m2L.end(); m1++, m2++)
  {
    cv::Matx33d C;
    C << 1, 0, -(*m2)(0),  //
        0, 1, -(*m2)(1),   //
        -(*m2)(0), -(*m2)(1), (*m2)(0) * (*m2)(0) + (*m2)(1) * (*m2)(1);
    double Z = (m1->t() * C * b)(0) / (m1->t() * C * (*m1))(0);
    depth.push_back(Z);
  }
}

bool MVO::checkEnoughDisparity(std::vector<cv::Point2f> & first, std::vector<cv::Point2f> & second){
  assert(first.size() == second.size());
  double diff = 0;
  for(auto p1 = first.begin(), p2 = second.begin(); p1 != first.end() && p2 != second.end(); p1++, p2++){
    diff += cv::norm((*p1)-(*p2));
  }
  diff = diff / first.size();
  return diff > 20; //TODO: Thresh
}