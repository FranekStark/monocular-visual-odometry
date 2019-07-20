#include "mvo.hpp"

#include <map>

MVO::MVO() : _slidingWindow(5), _blockSize(2), _apertureSize(3), _k(0.04), _thresh(200)
{
  cv::namedWindow("original", cv::WINDOW_GUI_EXPANDED);
  cv::namedWindow("cornerImage", cv::WINDOW_GUI_EXPANDED);

  cv::Point2f p;
  cv::Point2f *rp = &p;
  int i;

  ROS_INFO_STREAM("raw: " << sizeof(p) << "ref: " << sizeof(rp) << "int: " << sizeof(i) << std::endl);
}

MVO::~MVO()
{
  cv::destroyWindow("original");
  cv::destroyWindow("cornerImage");
}

void MVO::handleImage(const cv::Mat &image)
{
  /*Original */
  cv::imshow("original", image);
  /*Grayscale */
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY, 3);

  /*Track Features */
  std::vector<cv::Point2f> trackedFeatures;
  std::vector<unsigned char> found;
  cv::Mat *prevImage = _slidingWindow.getImage(0);
  std::vector<cv::Point2f> *prevFeatures = _slidingWindow.getFeatures(0);
  if (prevImage != nullptr)
  {  // Otherwise it is the first Frame
    this->trackFeatures(grayImage, *prevImage, *prevFeatures, trackedFeatures, found);
  }

  /*New Window-Frame */
  _slidingWindow.newWindow(trackedFeatures, *prevFeatures, found, grayImage);

  /*NewFeatures */
  std::vector<cv::Point2f> newFeatures = this->detectCorners(grayImage, 40);
  _slidingWindow.addFeaturesToCurrentWindow(newFeatures);

  /*Mark Features*/
  cv::Mat cornerImage = image.clone();
  std::stringstream text;
  text << "Number of Features: " << _slidingWindow.getFeatures(0)->size();
  cv::putText(cornerImage, text.str(), cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255));
  for (auto const &corner : *(_slidingWindow.getFeatures(0)))
  {
    cv::circle(cornerImage, cv::Point(corner), 10, cv::Scalar(0, 0, 255), -10);
  }
  imshow("cornerImage", cornerImage);
  cv::waitKey(1);
}

// Must be Grayscale
std::vector<cv::Point2f> MVO::detectCorners(const cv::Mat &image, int num)
{
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(image, corners, num, double(0.01), double(10.0), cv::noArray(), _blockSize, bool(true),
                          _k);  // Corners berechnen TODO: More params

  // Subpixel-genau:
  if (corners.size() > 0)
  {
    cv::Size winSize = cv::Size(5, 5);
    cv::Size zeroZone = cv::Size(-1, -1);
    cv::TermCriteria criteria =
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(image, corners, winSize, zeroZone, criteria);
  }
  return corners;
  std::vector<cv::Point2f> result;
 
}

void MVO::setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh)
{
  // TODO: Concurrent, when using more Threads
  _blockSize = blockSize;
  _apertureSize = aperatureSize;
  _k = k;
  _thresh = thresh;
}

void MVO::trackFeatures(const cv::Mat &nowImage, const cv::Mat &prevImage, const std::vector<cv::Point2f> &prevFeatures,
                        std::vector<cv::Point2f> &trackedFeatures, std::vector<unsigned char> &found)
{
  std::vector<cv::Mat> nowPyramide;
  std::vector<cv::Mat> prevPyramide;
  cv::Size winSize(21, 21);  // Has to be the same as in calcOpeitcalcOpticalFLow
  int maxLevel = 3;
  // TODO: The call of these Funtions only make sense, if we store the pyramids for reuse.
  // Otherwise calcOpticalFlow could to this on its own.
  cv::buildOpticalFlowPyramid(nowImage, nowPyramide, winSize, maxLevel);
  cv::buildOpticalFlowPyramid(prevImage, prevPyramide, winSize, maxLevel);

  // TODO: Handle Case, when no Before FEatures!
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(prevPyramide, nowPyramide, prevFeatures, trackedFeatures, found, error);  // TODO: more
                                                                                                     // Params
}