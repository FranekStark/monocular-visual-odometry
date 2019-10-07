#ifndef MVO_SRC_MVO_NODE_HPP_
#define MVO_SRC_MVO_NODE_HPP_

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
#include <tf2_msgs/TFMessage.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <mutex>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "mvo.hpp"
#include "Utils.hpp"

class MVO_node {
 private:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Imu>
      SyncPolicie;
  ros::NodeHandle & _nodeHandle;
  ros::NodeHandle & _privateNodeHandle;
  image_transport::ImageTransport  _imageTransport;
  image_transport::SubscriberFilter *  _imageSubscriber;
  message_filters::Subscriber<sensor_msgs::CameraInfo>  * _cameraInfoSubscriber;
  message_filters::Subscriber<sensor_msgs::Imu> * _imuSubscriber;
  message_filters::Synchronizer<SyncPolicie> * _synchronizer;

  ros::Publisher _estimatedOdomPublisher;
  ros::Publisher _refined1OdomPublisher;
  ros::Publisher _refined2OdomPublisher;
  dynamic_reconfigure::Server<mvo::mvoConfig> _dynamicConfigServer;
  dynamic_reconfigure::Server<mvo::mvoConfig>::CallbackType _dynamicConfigCallBackType;
  mvo::mvoConfig _currentConfig;
  std::mutex _configLock;

  cv::Matx33d _transformWorldToCamera;

  MVO _mvo;

  void init();

  geometry_msgs::Pose worldPoseFromCameraPosition(const cv::Point3d & position, const cv::Matx33d & orientation);


 public:
  MVO_node(ros::NodeHandle & nh, ros::NodeHandle & pnh);

  ~MVO_node();

  void imageCallback(const sensor_msgs::ImageConstPtr &, const sensor_msgs::CameraInfoConstPtr &,
                     const sensor_msgs::ImuConstPtr &);

  void dynamicConfigCallback(mvo::mvoConfig &config, uint32_t level);

  void publishEstimatedPosition(cv::Point3d position, cv::Matx33d orientation);
  void publishRefinedPosition(cv::Point3d position, cv::Matx33d orientation, int stage);


};
#endif //MVO_SRC_MVO_NODE_HPP_