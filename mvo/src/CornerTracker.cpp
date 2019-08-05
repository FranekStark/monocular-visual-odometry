#include "CornerTracker.hpp"

CornerTracker::CornerTracker() : _blockSize(2), _apertureSize(3), _k(0.04), _thresh(200)
{
}

CornerTracker::~CornerTracker()
{
}

void CornerTracker::detectFeatures(std::vector<cv::Point2f> & corner, const cv::Mat & image, int numberToDetect){
  if(numberToDetect <= 0){
    return;
  }
  cv::goodFeaturesToTrack(image, corner, numberToDetect, double(0.01), double(10.0), cv::noArray(), _blockSize, bool(true),
                          _k);  // Corners berechnen TODO: More params

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

void CornerTracker::trackFeatures(const cv::Mat &prevImage, const cv::Mat &currentImage, const std::vector<cv::Point2f> &prevFeatures,
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
  
}
  void CornerTracker::setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh){
  _blockSize = blockSize;
  _apertureSize = aperatureSize;
  _k = k;
  _thresh = thresh;
  }