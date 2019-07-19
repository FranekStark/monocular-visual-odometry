#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <mvo/corner_detectorConfig.h>
#include "mvo.hpp"

class MVO_node
{
private:
    ros::NodeHandle _nodeHandle;
    ros::NodeHandle _privateNodeHandle;
    image_transport::ImageTransport _imageTransport;
    image_transport::CameraSubscriber _imageSubscriber;
    std::string _imageSubscriberTopic;

    dynamic_reconfigure::Server<mvo::corner_detectorConfig> _dynamicConfigServer;
    dynamic_reconfigure::Server<mvo::corner_detectorConfig>::CallbackType _dynamicConfigCallBackType; 

    MVO _mvo;

    void init();

public:
    MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~MVO_node();
    void imageCallback(const sensor_msgs::ImageConstPtr &, const sensor_msgs::CameraInfoConstPtr &);
    void dynamicConfigCallback(mvo::corner_detectorConfig & config, uint32_t level);
};
