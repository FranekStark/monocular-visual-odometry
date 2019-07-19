#include <opencv2/core.hpp>
#include <vector>

#include "Window.hpp"

class SlidingWindow
{
private:
    /*Size of the SlidingWindow (how many Windows will be kept) */
    int _length;
    /*Last (timely) Window in the Past, which is still there*/
    Window * _firstWindow;   
    /*Window /Current (timly last)*/
    Window * _lastWindow;

    

     /**
     * 0 means the Window NOW/Current
     * If the Window is not available, then nullptr
     */
    Window * getWindow(int past);

public:
    SlidingWindow(int len);
    ~SlidingWindow();

    void newWindow(std::vector<cv::Point2f> trackedFeaturesNow, std::vector<cv::Point2f> & trackedFeaturesBefore, std::vector<unsigned char> & found, cv::Mat image);
    void addFeaturesToCurrentWindow(std::vector<cv::Point2f> features);

    /**
     * past = 0, means Current Windows Features
     * When no Window at this time, nullptr is returned
     */
    std::vector<cv::Point2f> * getFeatures(int past);
    cv::Mat * getImage(int past);
};

