#include <opencv2/core.hpp>
#include <vector>
#include <boost/bimap.hpp>


class Window
{
public:
    cv::Mat _image; //TODO:: Maybe remove TODO:maybe replace through pyramide
    std::vector<cv::Point2f> _features;    
    
    cv::Vec3d _position;
    cv::Matx33d _rotation;

    /*Points to the Window before (timely)*/
    Window * _windowBefore;

    /*Left is this Window feature, Right is the Feature Before. If no entry, then no Connection. */
    boost::bimap<long unsigned int, long unsigned int> _featuresBefore;
    
    Window();
    ~Window();
};

