#include "CornerTracker.hpp"
#include  <ros/ros.h>

CornerTracker::CornerTracker() : _blockSize(2), _apertureSize(3), _k(0.02), _thresh(200)
{
}

CornerTracker::~CornerTracker()
{
}

void CornerTracker::detectFeatures(std::vector<cv::Point2f> &corner, const cv::Mat &image, int numberToDetect, const std::vector<cv::Point2f> & existingFeatures)
{
  if (numberToDetect <= 0)
  {
    return;
  }
  //Create Mask
  cv::Mat mask(image.size(), CV_8U);
  mask = cv::Scalar(255); //First use all Pixels
  for(auto existingFeature : existingFeatures){
    cv::circle(mask, existingFeature, 40, cv::Scalar(0), -1); //Here no Feature Detection! //TODO: param
  }

  cv::goodFeaturesToTrack(image, corner, numberToDetect, double(0.10), double(50.0), mask, _blockSize,
                          bool(true),
                          _k);  // Corners berechnen TODO: More params -> especially the detection Quality

  // Subpixel-genau:
  if (corner.size() > 0)
  {
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria =
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(image, corner, winSize, zeroZone, criteria);
  }
}

void CornerTracker::trackFeatures(const cv::Mat &prevImage, const cv::Mat &currentImage,
                                  const std::vector<cv::Point2f> &prevFeatures,
                                  std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found)
{
  std::vector<cv::Mat> nowPyramide;
  std::vector<cv::Mat> prevPyramide;
  cv::Size winSize(21, 21);  // Has to be the same as in calcOpeitcalcOpticalFLow
  int maxLevel = 3;
  // TODO: The call of these Funtions only make sense, if we store the pyramids for reuse.
  // Otherwise calcOpticalFlow could do this on its own.
  cv::buildOpticalFlowPyramid(currentImage, nowPyramide, winSize, maxLevel);
  cv::buildOpticalFlowPyramid(prevImage, prevPyramide, winSize, maxLevel);

  // TODO: Handle Case, when no before Features!
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(prevPyramide, nowPyramide, prevFeatures, trackedFeatures, found, error);  // TODO: more

  //TODO: Sort out to Close features!
  std::vector<cv::Point2f>::iterator f1,f2;
   std::vector<unsigned char>::iterator f1Found, f2Found;
  for(f1 = trackedFeatures.begin(), f1Found = found.begin(); f1 != trackedFeatures.end(); f1++, f1Found++){
    if(*f1Found == 0){
      continue;
    }
    for(f2 = trackedFeatures.begin(), f2Found = found.begin(); f2 != trackedFeatures.end(); f2++, f2Found++){
      if(*f2Found == 0){
        continue;
      }
      if(f1 == f2){
        continue; //Don't sort out the same Feature
      }

      if(cv::norm(*f1 - *f2) < 40){ //TODO: Param
        *f2Found = 0; //Sort Out
      }

    }
  }

}
void CornerTracker::setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh)
{
  _blockSize = blockSize;
  _apertureSize = aperatureSize;
  _k = k;
  _thresh = thresh;
}