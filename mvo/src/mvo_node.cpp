#include "mvo_node.hpp"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "mvo_node");
  MVO_node node(ros::NodeHandle(""), ros::NodeHandle("~"));
  ros::AsyncSpinner spinner(2); //Two Threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}

MVO_node::MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh)
    : _nodeHandle(nh),
      _privateNodeHandle(pnh),
      _imageTransport(nh),
      _imageSubscriberTopic("/camera_ueye/image_rect"),
      _imageSubscriber(_imageTransport, "/camera_ueye/image_rect", 10),
      _cameraInfoSubscriber(_nodeHandle, "/camera_ueye/camera_info", 10),
      _imuSubscriber(_nodeHandle, "/imu/data", 100),
      _synchronizer(SyncPolicie(200), _imageSubscriber, _cameraInfoSubscriber, _imuSubscriber),
      _transformWorldToCamera(0, 0, 1, -1, 0, 0, 0, -1, 0),
      _mvo(std::function<(void) cv::Point3d>(), 0, std::function<(void) cv::Point3d>()) {

  this->init();
}

MVO_node::~MVO_node() {
}

void MVO_node::init() {
  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
  _odomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom", 10, true);
  _debugImagePublisher = _imageTransport.advertise("debug/image", 3, true);
  _debugImage2Publisher = _imageTransport.advertise("debug/image2", 3, true);
  _debugImage3Publisher = _imageTransport.advertise("debug/image3", 3, true);
  _debugImage4Publisher = _imageTransport.advertise("debug/image4", 3, true);
  _synchronizer.registerCallback(boost::bind(&MVO_node::imageCallback, this, _1, _2, _3));

}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camInfo,
                             const sensor_msgs::ImuConstPtr &imu) {
  /**
   * Convert the Image and the CameraInfo
   **/
  cv_bridge::CvImageConstPtr bridgeImage = cv_bridge::toCvCopy(image, "mono8");
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(camInfo);

  /**
   * Convert the Orientation
   **/
  tf2::Quaternion orientationQuat(imu->orientation.x, imu->orientation.y, imu->orientation.z, imu->orientation.w);
  tf2::Matrix3x3 orientationMat(orientationQuat);
  cv::Matx33d orientationMatCV(orientationMat[0][0], orientationMat[0][1], orientationMat[0][2],
                               orientationMat[1][0], orientationMat[1][1], orientationMat[1][2],
                               orientationMat[2][0], orientationMat[2][1], orientationMat[2][2]
  );
  /**
   * Project To Camera Coordinates
   **/
  orientationMatCV = _transformWorldToCamera.t() * orientationMatCV * _transformWorldToCamera;

  /**
   * Call the Algorithm
   **/
  //auto od = _mvo.handleImage(bridgeImage->image, model, orientationMatCV);

  /**
   * Reproject TO World Coordinates
   */
  od.b = _transformWorldToCamera * od.b;
  od.s = _transformWorldToCamera * od.s;
  orientationMatCV = _transformWorldToCamera * od.o * _transformWorldToCamera.t();


  /**
   * Reconvert
   **/
  orientationMat[0][0] = orientationMatCV(0, 0);
  orientationMat[0][1] = orientationMatCV(0, 1);
  orientationMat[0][2] = orientationMatCV(0, 2);
  orientationMat[1][0] = orientationMatCV(1, 0);
  orientationMat[1][1] = orientationMatCV(1, 1);
  orientationMat[1][2] = orientationMatCV(1, 2);
  orientationMat[2][0] = orientationMatCV(2, 0);
  orientationMat[2][1] = orientationMatCV(2, 1);
  orientationMat[2][2] = orientationMatCV(2, 2);
  orientationMat.getRotation(orientationQuat);



  /**
   * Pack Message and Publish
   **/
  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = ros::Time::now();
  odomMsg.header.frame_id = "odom";
  odomMsg.child_frame_id = "base_footprint";
  odomMsg.pose.pose.position.x = od.s(0);
  odomMsg.pose.pose.position.y = od.s(1);
  odomMsg.pose.pose.position.z = od.s(2);
  odomMsg.pose.pose.orientation.w = orientationQuat.getW();
  odomMsg.pose.pose.orientation.x = orientationQuat.getX();
  odomMsg.pose.pose.orientation.y = orientationQuat.getY();
  odomMsg.pose.pose.orientation.z = orientationQuat.getZ();
  odomMsg.twist.twist.linear.x = od.b(0);
  odomMsg.twist.twist.linear.y = od.b(1);
  odomMsg.twist.twist.linear.z = od.b(2);
  _odomPublisher.publish(odomMsg);

  /**
   * Pack Tranform and Publish
   **/
  geometry_msgs::TransformStamped transformMsg;
  transformMsg.header.stamp = ros::Time::now();
  transformMsg.header.frame_id = "odom";
  transformMsg.child_frame_id = "base_footprint";
  transformMsg.transform.translation.x = od.s(0);
  transformMsg.transform.translation.y = od.s(1);
  transformMsg.transform.translation.z = od.s(2);
  transformMsg.transform.rotation.w = orientationQuat.getW();
  transformMsg.transform.rotation.x = orientationQuat.getX();
  transformMsg.transform.rotation.y = orientationQuat.getY();
  transformMsg.transform.rotation.z = orientationQuat.getZ();
  _odomTfBroadcaster.sendTransform(transformMsg);

  /**
   * Pack DebugImages and Publish
   */
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "camera_link";
  cv_bridge::CvImage debugImage(header, "rgb8", _mvo._debugImage);
  _debugImagePublisher.publish(debugImage.toImageMsg());
  cv_bridge::CvImage debugImage2(header, "rgb8", _mvo._debugImage2);
  _debugImage2Publisher.publish(debugImage2.toImageMsg());
  cv_bridge::CvImage debugImage3(header, "rgb8", _mvo._debugImage3);
  _debugImage3Publisher.publish(debugImage3.toImageMsg());
  cv_bridge::CvImage debugImage4(header, "rgb8", _mvo._debugImage4);
  _debugImage4Publisher.publish(debugImage4.toImageMsg());
}

void MVO_node::dynamicConfigCallback(mvo::mvoConfig &config, uint32_t level) {
  (void) (level);  // TODO: unused
  _mvo._cornerTracker.setCornerDetectorParams(config.block_size, config.minDifPercent, config.qualityLevel,
                                              config.trackerWindowSize, config.maxPyramideLevel);
  _mvo.setParameters(config.numberOfFeatures, config.disparityThreshold);
}
