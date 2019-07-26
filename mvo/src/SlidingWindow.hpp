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

  void newWindow(std::vector<cv::Point2d> trackedFeaturesNow, std::vector<cv::Point2d>& trackedFeaturesBefore,
                 std::vector<unsigned char>& found, cv::Mat image);
  void addFeaturesToCurrentWindow(std::vector<cv::Point2d> features);

  void addTransformationToCurrentWindow(cv::Mat translation, cv::Mat rotation);

  /**
   * past = 0, means Current Windows Features
   * When no Window at this time, nullptr is returned
   */
  std::vector<cv::Point2d>* getFeatures(int past);
  cv::Mat* getImage(int past);
  cv::Mat getPosition(int past);

/**
 * Gives Back Corresponding Featurelocations between two Windows.
 * Window2 is timely after Window1. While 0 means NOW, and the Maximum is _length.
 * Usually Window 2 is 0.
 */
  void getCorrespondingFeatures(int window1Index, int window2Index, std::vector<cv::Point2d>& features1,
                                std::vector<cv::Point2d>& features2);

  void getCorrespondingPosition(int window1Index, int window2Index, cv::Mat & position1, cv::Mat & position2, cv::Mat &rotation1, cv::Mat &rotation2);
  
};
