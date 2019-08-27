#include "mvo_node.hpp"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mvo_node");
  MVO_node node(ros::NodeHandle(""), ros::NodeHandle("~"));
  ros::AsyncSpinner spinner(2); //Two Threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}

MVO_node::MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh)
  : _nodeHandle(nh), _privateNodeHandle(pnh), _imageTransport(nh), _transFormListener(_tfBuffer), _transformWorldToCamera(0,0,1,-1,0,0,0,-1,0), _rotation(1,0,0,0,1,0,0,0,1)
{
  //ROS_INFO_STREAM("M = " << _transformWorldToCamera << std::endl);
  this->init();
}

MVO_node::~MVO_node()
{
}

void MVO_node::init()
{
  _imageSubscriberTopic = "/pylon_camera_node/image_rect";
  _imageSubscriber = _imageTransport.subscribeCamera(_imageSubscriberTopic, 3, &MVO_node::imageCallback, this);
  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
  _odomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom", 10, true);
  _debugImagePublisher = _imageTransport.advertise("debug/image", 3, true);
  _debugImage2Publisher = _imageTransport.advertise("debug/image2", 3, true);
  _imuSubscriber = _nodeHandle.subscribe("/imu/data", 10, &MVO_node::imuCallback, this);
}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camInfo)
{
  (void)(camInfo);  // TODO: unused
  cv_bridge::CvImageConstPtr bridgeImage = cv_bridge::toCvShare(image, "mono8");
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(camInfo);

  _rotationMutex.lock();
  cv::Matx33d rotation = _rotation;
  _rotationMutex.unlock();

  auto od = _mvo.handleImage(bridgeImage->image, model, rotation);  
  //ROS_INFO_STREAM("before: " << od.s << std::endl);
  od.b = _transformWorldToCamera * od.b;
  od.s = _transformWorldToCamera * od.s; 
  //ROS_INFO_STREAM("after: " << od.s << std::endl);
  /*Publish */
  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = ros::Time::now();
  odomMsg.header.frame_id = "odom";
  odomMsg.child_frame_id = "base_footprint";
  odomMsg.pose.pose.position.x = od.s(0);
  odomMsg.pose.pose.position.y = od.s(1);
  odomMsg.pose.pose.position.z = od.s(2);
  odomMsg.pose.pose.orientation.w = 1;
  odomMsg.twist.twist.linear.x = od.b(0);
  odomMsg.twist.twist.linear.y = od.b(1);
  odomMsg.twist.twist.linear.z = od.b(2);
  _odomPublisher.publish(odomMsg);

  geometry_msgs::TransformStamped transformMsg;
  transformMsg.header.stamp = ros::Time::now();
  transformMsg.header.frame_id = "odom";
  transformMsg.child_frame_id = "base_footprint";
  transformMsg.transform.translation.x = od.s(0);
  transformMsg.transform.translation.y = od.s(1);
  transformMsg.transform.translation.z = od.s(2);
  transformMsg.transform.rotation.w = 1;
  _odomTfBroadcaster.sendTransform(transformMsg);

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "camera_link";
  cv_bridge::CvImage debugImage(header,"rgb8",_mvo._debugImage);
  _debugImagePublisher.publish(debugImage.toImageMsg());
  cv_bridge::CvImage debugImage2(header,"rgb8",_mvo._debugImage2);
  _debugImage2Publisher.publish(debugImage2.toImageMsg());
}

void MVO_node::dynamicConfigCallback(mvo::corner_detectorConfig &config, uint32_t level)
{
  (void)(level);  // TODO: unused
  _mvo._cornerTracker.setCornerDetectorParams(config.block_size, config.aperture_size, config.k, config.threshold);
}

void MVO_node::imuCallback(const sensor_msgs::ImuConstPtr & imu_msg){
  // //Anuglar Velocity Vector
  // cv::Vec3d angular_velocity(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
  // //Transform into Camera Cordinates
  // angular_velocity = _transformWorldToCamera.t() * angular_velocity;
  // //Caluclate relative Rotation:
  // double theta = cv::norm(angular_velocity);
  // cv::Matx33d angular_velocity_cross( 0                          ,-1.0 * angular_velocity[2], angular_velocity[1],
  //                                     angular_velocity[2]        ,0                         , -1.0 * angular_velocity[1],
  //                                     -1.0 * angular_velocity[1] ,angular_velocity[0]       ,0                           );
                                    
  // cv::Matx33d Rd = cv::Matx33d::eye() + (std::sin(theta)/theta) * angular_velocity_cross + 
  // ((1.0 - std::cos(theta)) / theta*theta) *  angular_velocity_cross * angular_velocity_cross; //TODO: fasten up through before calc

  // _rotationMutex.lock();
  // _rotation = _rotation * Rd;
  // ROS_INFO_STREAM( " new Rotationdelta: " << std::endl << Rd << std::endl);
  // tf2::Matrix3x3 rot(_rotation(0,0), _rotation(0,1),_rotation(0,2),
  //                _rotation(1,0), _rotation(1,1),_rotation(1,2),
  //                _rotation(2,0), _rotation(2,1),_rotation(2,2));

  
  tf2::Quaternion quat(imu_msg->orientation.x,imu_msg->orientation.y,imu_msg->orientation.z, imu_msg->orientation.w);
  tf2::Matrix3x3 rot(quat);
  double yaw, pitch, roll;
  rot.getEulerYPR(yaw, pitch, roll);
  rot.setRPY(-pitch,-yaw,roll);
  _rotationMutex.lock();
  _rotation(0,0) = rot[0][0]; _rotation(0,1) = rot[0][1]; _rotation(0,2) = rot[0][2];
  _rotation(1,0) = rot[1][0]; _rotation(1,1) = rot[1][1]; _rotation(1,2) = rot[1][2];
  _rotation(2,0) = rot[2][0]; _rotation(2,1) = rot[2][1]; _rotation(2,2) = rot[2][2];
  _rotationMutex.unlock();



}