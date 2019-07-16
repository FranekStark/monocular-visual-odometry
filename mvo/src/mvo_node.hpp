#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "mvo.hpp"


class MVO_node
{
private:
    ros::NodeHandle _nodeHandle;
    ros::NodeHandle _privateNodeHandle;
    image_transport::ImageTransport _imageTransport;
    image_transport::Subscriber _imageSubscriber;
    std::string _imageSubscriberTopic;

    MVO _mvo;

    void init();
public:
    MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~MVO_node();
    void imageCallback(const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&);
};


