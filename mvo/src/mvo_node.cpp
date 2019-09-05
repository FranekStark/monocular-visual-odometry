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
  : _nodeHandle(nh), _privateNodeHandle(pnh), _imageTransport(nh), _transFormListener(_tfBuffer), _transformWorldToCamera(0,0,1,-1,0,0,0,-1,0)
{
  //ROS_INFO_STREAM("M = " << _transformWorldToCamera << std::endl);
  this->init();
}

MVO_node::~MVO_node()
{
}

void MVO_node::init()
{
  _imageSubscriberTopic = "/camera_ueye/image_rect";
  _imageSubscriber = _imageTransport.subscribeCamera(_imageSubscriberTopic, 10, &MVO_node::imageCallback, this);
  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
  _odomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom", 10, true);
  _debugImagePublisher = _imageTransport.advertise("debug/image", 3, true);
  _debugImage2Publisher = _imageTransport.advertise("debug/image2", 3, true);
}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camInfo)
{
  assert(image->header.stamp == camInfo->header.stamp);

  (void)(camInfo);  // TODO: unused
  cv_bridge::CvImageConstPtr bridgeImage = cv_bridge::toCvCopy(image, "mono8");
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(camInfo);

  //Get Rotation for The Image Timestamp
  auto orientation = _tfBuffer.lookupTransform("base_footprint", "base_link", camInfo->header.stamp);
  tf2::Quaternion rotationQuat(orientation.transform.rotation.x, orientation.transform.rotation.y, orientation.transform.rotation.z, orientation.transform.rotation.w);
  tf2::Matrix3x3 rotation(rotationQuat);

  assert(image->header.stamp == orientation.header.stamp);


  double yaw, pitch, roll;
  rotation.getEulerYPR(yaw, pitch, roll);
  rotation.setEulerYPR(roll,-yaw,-pitch);
  cv::Matx33d rotationCV;
  rotationCV(0,0) = rotation[0][0]; rotationCV(0,1) = rotation[0][1]; rotationCV(0,2) = rotation[0][2];
  rotationCV(1,0) = rotation[1][0]; rotationCV(1,1) = rotation[1][1]; rotationCV(1,2) = rotation[1][2];
  rotationCV(2,0) = rotation[2][0]; rotationCV(2,1) = rotation[2][1]; rotationCV(2,2) = rotation[2][2];
  
  auto od = _mvo.handleImage(bridgeImage->image, model, rotationCV);  
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
  _mvo._cornerTracker.setCornerDetectorParams(config.block_size, config.minDifPercent, config.qualityLevel);
}
