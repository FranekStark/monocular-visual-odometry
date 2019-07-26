#include <opencv2/core.hpp>
#include <vector>

#include "Window.hpp"

class SlidingWindow
{
private:
  /*Size of the SlidingWindow (how many Windows will be kept) */
  int _length;
  /*Last (timely) Window in the Past, which is still there*/
  Window* _firstWindow;
  /*Window /Current (timly last)*/
  Window* _lastWindow;

  /**
   * 0 means the Window NOW/Current
   * If the Window is not available, then nullptr
   */
  Window* getWindow(int past);

public:
  SlidingWindow(int len);
  ~SlidingWindow();

  void newWindow(const std::vector<cv::Point2f> & trackedFeaturesNow,
                 const std::vector<unsigned char>& found, cv::Mat image);
  void addFeaturesToCurrentWindow(std::vector<cv::Point2f> & features);

  void addTransformationToCurrentWindow(const cv::Vec3d & position, const cv::Matx33d & rotation);

  /**
   * past = 0, means Current Windows Features
   * When no Window at this time, nullptr is returned
   */
  std::vector<cv::Point2f>& getFeatures(int past);
  cv::Mat getImage(int past);
  cv::Vec3d & getPosition(int past);

/**
 * Gives Back Corresponding Featurelocations between two Windows.
 * Window2 is timely after Window1. While 0 means NOW, and the Maximum is _length.
 * Usually Window 2 is 0.
 */
  void getCorrespondingFeatures(int window1Index, int window2Index, std::vector<cv::Point2f>& features1,
                                std::vector<cv::Point2f>& features2);

  void getCorrespondingPosition(int window1Index, int window2Index, cv::Vec3d & position1, cv::Vec3d & position2, cv::Matx33d &rotation1, cv::Matx33d &rotation2);
  
};
