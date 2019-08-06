#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

class CornerTracker
{
private:
  int _blockSize;
  int _apertureSize;
  double _k;
  int _thresh;

public:
  CornerTracker();
  ~CornerTracker();
  void detectFeatures(std::vector<cv::Point2f> &corner, const cv::Mat &image, int numberToDetect);
  void trackFeatures(const cv::Mat &prevImage, const cv::Mat &currentImage,
                     const std::vector<cv::Point2f> &prevFeatures, std::vector<cv::Point2f> &trackedFeatures,
                     std::vector<unsigned char> &found);
  void setCornerDetectorParams(int blockSize, int aperatureSize, double k, int thresh);
};
