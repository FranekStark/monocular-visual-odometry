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
      _mvo([this](cv::Point3d position, cv::Matx33d orientation){
          this->publishEstimatedPosition(position, orientation);
        },
            [this](cv::Point3d position, cv::Matx33d orientation){
        this->publishRefinedPosition(position, orientation);
      }) {

  this->init();
}

MVO_node::~MVO_node() {
}

void MVO_node::init() {
  //Set Log-Level:
  if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
    ros::console::notifyLoggerLevelsChanged();
  }

  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
  _estimatedOdomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom_estimated", 10, true);
  _refinedOdomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom_refined", 10, true);
  /*_debugImagePublisher = _imageTransport.advertise("debug/image", 3, true);
  _debugImage2Publisher = _imageTransport.advertise("debug/image2", 3, true);
  _debugImage3Publisher = _imageTransport.advertise("debug/image3", 3, true);
  _debugImage4Publisher = _imageTransport.advertise("debug/image4", 3, true);*/
  _synchronizer.registerCallback(boost::bind(&MVO_node::imageCallback, this, _1, _2, _3));

  //Callbacks

}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camInfo,
                             const sensor_msgs::ImuConstPtr &imu) {

 LOG_DEBUG("Image Callback");
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
  _mvo.newImage(bridgeImage->image, model, orientationMatCV);

/*  *//**
   * Pack DebugImages and Publish
   *//*
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
  _debugImage4Publisher.publish(debugImage4.toImageMsg());*/
}

void MVO_node::dynamicConfigCallback(mvo::mvoConfig &config, uint32_t level) {
  (void) (level);  // TODO: unused
  (void) (config);
  ROS_ERROR("Not implemented!");
}

void MVO_node::publishEstimatedPosition(cv::Point3d position, cv::Matx33d orientation) {
  //Pack Message
  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = ros::Time::now(); //TODO: Time of Image?
  odomMsg.header.frame_id = "odom";
  odomMsg.child_frame_id = "base_footprint";
  odomMsg.pose.pose = worldPoseFromCameraPosition(position, orientation);
  //Publish
  _estimatedOdomPublisher.publish(odomMsg);
}
void MVO_node::publishRefinedPosition(cv::Point3d position, cv::Matx33d orientation) {
  //Pack Message
  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = ros::Time::now(); //TODO: Time of Image?
  odomMsg.header.frame_id = "odom";
  odomMsg.child_frame_id = "base_footprint";
  odomMsg.pose.pose = worldPoseFromCameraPosition(position, orientation);
  //Publish
  _refinedOdomPublisher.publish(odomMsg);
}
geometry_msgs::Pose MVO_node::worldPoseFromCameraPosition(const cv::Point3d &position, const cv::Matx33d &orientation) {
  /**
   * Reproject TO World Coordinates
   */
  auto positionWorld = _transformWorldToCamera * position;
  auto orientationWorldCV = _transformWorldToCamera * orientation * _transformWorldToCamera.t();
  tf2::Matrix3x3 orientationMat;
  /**
   * Reconvert
   **/
  orientationMat[0][0] = orientationWorldCV(0, 0);
  orientationMat[0][1] = orientationWorldCV(0, 1);
  orientationMat[0][2] = orientationWorldCV(0, 2);
  orientationMat[1][0] = orientationWorldCV(1, 0);
  orientationMat[1][1] = orientationWorldCV(1, 1);
  orientationMat[1][2] = orientationWorldCV(1, 2);
  orientationMat[2][0] = orientationWorldCV(2, 0);
  orientationMat[2][1] = orientationWorldCV(2, 1);
  orientationMat[2][2] = orientationWorldCV(2, 2);
  tf2::Quaternion orientationQuat;
  orientationMat.getRotation(orientationQuat);
  /**
   * Pack
   **/
  geometry_msgs::Pose pose;
  pose.position.x = positionWorld.x;
  pose.position.y = positionWorld.y;
  pose.position.z = positionWorld.z;
  pose.orientation.y = orientationQuat.getY();
  pose.orientation.w = orientationQuat.getW();
  pose.orientation.z = orientationQuat.getZ();
  pose.orientation.x = orientationQuat.getX();
  //Return Position
  return pose;
}
