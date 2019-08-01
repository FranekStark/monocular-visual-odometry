#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <dynamic_reconfigure/server.h>
#include <mvo/corner_detectorConfig.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/Odometry.h>
#include "mvo.hpp"

class MVO_node
{
private:
    ros::NodeHandle _nodeHandle;
    ros::NodeHandle _privateNodeHandle;
    image_transport::ImageTransport _imageTransport;
    image_transport::CameraSubscriber _imageSubscriber;
    std::string _imageSubscriberTopic;
    ros::Publisher _odomPublisher;
    tf2_ros::TransformBroadcaster _odomTfBroadcaster;
    tf2_ros::Buffer _tfBuffer;
    tf2_ros::TransformListener _transFormListener;

    dynamic_reconfigure::Server<mvo::corner_detectorConfig> _dynamicConfigServer;
    dynamic_reconfigure::Server<mvo::corner_detectorConfig>::CallbackType _dynamicConfigCallBackType; 

    image_transport::Publisher _debugImagePublisher;

    MVO _mvo;

    cv::Matx33d _worldToCameraProjectionMatrix;

    void init();

public:
    MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~MVO_node();
    void imageCallback(const sensor_msgs::ImageConstPtr &, const sensor_msgs::CameraInfoConstPtr &);
    void dynamicConfigCallback(mvo::corner_detectorConfig & config, uint32_t level);
};
