#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <dynamic_reconfigure/server.h>
#include <mvo/mvoConfig.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <mutex>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "mvo.hpp"

class MVO_node
{
private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Imu> SyncPolicie;
    ros::NodeHandle _nodeHandle;
    ros::NodeHandle _privateNodeHandle;
    image_transport::ImageTransport _imageTransport;
    std::string _imageSubscriberTopic;
    image_transport::SubscriberFilter _imageSubscriber;
    message_filters::Subscriber<sensor_msgs::CameraInfo> _cameraInfoSubscriber;
    message_filters::Subscriber<sensor_msgs::Imu> _imuSubscriber;
    message_filters::Synchronizer<SyncPolicie> _synchronizer;
    tf2_ros::TransformBroadcaster _odomTfBroadcaster;
    ros::Publisher _odomPublisher;

    dynamic_reconfigure::Server<mvo::mvoConfig> _dynamicConfigServer;
    dynamic_reconfigure::Server<mvo::mvoConfig>::CallbackType _dynamicConfigCallBackType; 

    image_transport::Publisher _debugImagePublisher;
    image_transport::Publisher _debugImage2Publisher;

    MVO _mvo;

    cv::Matx33d _transformWorldToCamera;


    void init();

public:
    MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~MVO_node();
    void imageCallback(const sensor_msgs::ImageConstPtr &, const sensor_msgs::CameraInfoConstPtr &, const sensor_msgs::ImuConstPtr &);
    void dynamicConfigCallback(mvo::mvoConfig & config, uint32_t level);
};
