#ifndef SLIDING_WINDOW_HPP
#define SLIDING_WINDOW_HPP
#include <opencv2/core.hpp>
#include <vector>

#include "Frame.hpp"

class SlidingWindow
{
private:
  /*Size of the SlidingWindow (how many Windows will be kept) */
  unsigned int _length;
  /*Number Of current Frames*/
  unsigned int _frameCounter;
  /*Last (timely) Window in the Past, which is still there*/
  Frame* _frameNow;

  /**
   * 0 means the Window NOW/Current
   * If the Window is not available, then nullptr
   */
  Frame & getFrame(unsigned int past) const;

public:
  SlidingWindow(int len);
  ~SlidingWindow();

  void newFrame(const std::vector<cv::Point2f> & trackedFeaturesNow,
                 const std::vector<cv::Vec3d> & trackedFeaturesNowE,
                 const std::vector<unsigned char> & found, cv::Mat image);

  void addNewFeaturesToCurrentFrame(const std::vector<cv::Point2f> & features, const std::vector<cv::Vec3d> & featuresE);


  /**
   * past = 0, means Current Windows Features
   * When no Window at this time, nullptr is returned
   */
  void getFeatures(unsigned int past, std::vector<cv::Point2f> & features) const;
  void getFeatures(unsigned int past, std::vector<cv::Vec3d> & features) const;

  const cv::Mat getImage(unsigned int past) const;
  cv::Vec3d & getPosition(unsigned int past) const;
  cv::Matx33d & getRotation(unsigned int past) const;

  void setPosition(const cv::Vec3d & position, unsigned int past);
  void setRotation(const cv::Matx33d & rotation, unsigned int past);

/**
 * Gives Back Corresponding Featurelocations between two Windows.
 * Window2 is timely after Window1. While 0 means NOW, and the Maximum is _length.
 * Usually Window 2 is 0.
 */
  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Point2f>& features1,
                                             std::vector<cv::Point2f>& features2) const;
  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<cv::Vec3d>& features1,
                                std::vector<cv::Vec3d>& features2) const;         

  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<std::vector<cv::Vec3d>*> features) const;   
  void getCorrespondingFeatures(unsigned int window1Index, unsigned int window2Index, std::vector<std::vector<cv::Point2f>*> features) const;   

  unsigned int getNumberOfCurrentTrackedFeatures() const;

  void removeFeatureFromCurrentWindow(const cv::Vec3d & feature);
};
#endif //SLIDING_WINDOW_HPP