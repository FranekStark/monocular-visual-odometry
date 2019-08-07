#ifndef SLIDING_WINDOW_HPP
#define SLIDING_WINDOW_HPP
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
  Window* getWindow(int past) const;

public:
  SlidingWindow(int len);
  ~SlidingWindow();

  void newWindow(const std::vector<cv::Point2f> & trackedFeaturesNow,
                 const std::vector<cv::Vec3d> & trackedFeaturesNowE,
                 const std::vector<unsigned char> & found, cv::Mat image);
  void addFeaturesToCurrentWindow(std::vector<cv::Point2f> & features, const std::vector<cv::Vec3d> & featuresE);

  void addTransformationToCurrentWindow(const cv::Vec3d & position, const cv::Matx33d & rotation);

  /**
   * past = 0, means Current Windows Features
   * When no Window at this time, nullptr is returned
   */
  const std::vector<cv::Point2f>& getFeatures(int past) const;
  const std::vector<cv::Vec3d>& SlidingWindow::getFeaturesE(int past) const;
  const cv::Mat getImage(int past) const;
  cv::Vec3d & getPosition(int past) const;
  cv::Matx33d & getRotation(int past) const;

/**
 * Gives Back Corresponding Featurelocations between two Windows.
 * Window2 is timely after Window1. While 0 means NOW, and the Maximum is _length.
 * Usually Window 2 is 0.
 */
  void getCorrespondingFeatures(int window1Index, int window2Index, std::vector<cv::Point2f>& features1,
                                std::vector<cv::Point2f>& features2) const;
  void getCorrespondingFeatures(int window1Index, int window2Index, std::vector<cv::Vec3d>& features1,
                                std::vector<cv::Vec3d>& features2) const;                          

  void getCorrespondingPosition(int window1Index, int window2Index, cv::Vec3d & position1, cv::Vec3d & position2, cv::Matx33d &rotation1, cv::Matx33d &rotation2) const;
  


  unsigned int getNumberOfCurrentTrackedFeatures() const;

  void removeFeatureFromCurrentWindow(const cv::Point2f & feature);
  void removeFeatureFromCurrentWindow(const cv::Vec3d & feature)
};
#endif //SLIDING_WINDOW_HPP