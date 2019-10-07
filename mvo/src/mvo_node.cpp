#include "mvo_node.hpp"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "mvo_node");
  auto nh = ros::NodeHandle("");
  auto pnh = ros::NodeHandle("~");
  MVO_node node(nh, pnh);
  ros::AsyncSpinner spinner(2); //Two Threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}

MVO_node::MVO_node(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : _nodeHandle(nh),
      _privateNodeHandle(pnh),
      _imageTransport(nh),
      _currentConfig(mvo::mvoConfig::__getDefault__()),
      _transformWorldToCamera(0, 0, 1, -1, 0, 0, 0, -1, 0),
      _mvo([this](cv::Point3d position, cv::Matx33d orientation) {
             this->publishEstimatedPosition(position, orientation);
           },
           [this](cv::Point3d position, cv::Matx33d orientation) {
             this->publishRefinedPosition(position, orientation,1);
           },
           [this](cv::Point3d position, cv::Matx33d orientation) {
             this->publishRefinedPosition(position, orientation,2);
           }) {
  this->init();
}

MVO_node::~MVO_node() {
  delete _synchronizer;
  delete _imuSubscriber;
  delete _cameraInfoSubscriber;
  delete _imageSubscriber;
}

void MVO_node::init() {
  //Get Params
  std::string _imageTopic = "/pylon_camera_node/image_rect";
  std::string _camInfoTopic = "/pylon_camera_node/camera_info";
  std::string _imuTopic = "/imu/data";
  bool _logDebug = false;
  bool _useCompressed = false;

  _privateNodeHandle.param<std::string>("imageTopic", _imageTopic, _imageTopic);
  _privateNodeHandle.param<std::string>("cameraInfoTopic", _camInfoTopic, _camInfoTopic);
  _privateNodeHandle.param<std::string>("imuTopic", _imuTopic, _imuTopic);
  _privateNodeHandle.param<bool>("logDebug", _logDebug, _logDebug);
  _privateNodeHandle.param<bool>("useCompressed", _useCompressed, _useCompressed);




  //Get Params for Algo-Config;
  _privateNodeHandle.param<int>("numberOfFeatures", _currentConfig.numberOfFeatures, _currentConfig.numberOfFeatures);
  _privateNodeHandle.param<double>("qualityLevel", _currentConfig.qualityLevel, _currentConfig.qualityLevel);
  _privateNodeHandle.param<int>("windowSizeX", _currentConfig.windowSizeX, _currentConfig.windowSizeX);
  _privateNodeHandle.param<int>("windowSizeY", _currentConfig.windowSizeY, _currentConfig.windowSizeY);
  _privateNodeHandle.param<double>("k", _currentConfig.k, _currentConfig.k);
  _privateNodeHandle.param<double>("blockSize", _currentConfig.blockSize, _currentConfig.blockSize);
  _privateNodeHandle.param<double>("mindDiffPercent", _currentConfig.mindDiffPercent, _currentConfig.mindDiffPercent);
  _privateNodeHandle.param<int>("pyramidDepth", _currentConfig.pyramidDepth, _currentConfig.pyramidDepth);
  _privateNodeHandle.param<double>("sameDisparityThreshold",
                                   _currentConfig.sameDisparityThreshold,
                                   _currentConfig.sameDisparityThreshold);
  _privateNodeHandle.param<double>("movementDisparityThreshold",
                                   _currentConfig.movementDisparityThreshold,
                                   _currentConfig.movementDisparityThreshold);
  _privateNodeHandle.param<double>("thresholdOutlier",
                                   _currentConfig.thresholdOutlier,
                                   _currentConfig.thresholdOutlier);
  _privateNodeHandle.param<double>("bestFitProbability",
                                   _currentConfig.bestFitProbability,
                                   _currentConfig.bestFitProbability);
  _privateNodeHandle.param<int>("maxNumThreads", _currentConfig.maxNumThreads, _currentConfig.maxNumThreads);
  _privateNodeHandle.param<int>("maxNumIterations", _currentConfig.maxNumIterations, _currentConfig.maxNumIterations);
  _privateNodeHandle.param<double>("functionTolerance",
                                   _currentConfig.functionTolerance,
                                   _currentConfig.functionTolerance);
  _privateNodeHandle.param<bool>("useLossFunction", _currentConfig.useLossFunction, _currentConfig.useLossFunction);
  _privateNodeHandle.param<double>("lowestLength", _currentConfig.lowestLength, _currentConfig.lowestLength);
  _privateNodeHandle.param<double>("highestLength", _currentConfig.highestLength, _currentConfig.highestLength);

  _dynamicConfigServer.updateConfig(_currentConfig);


  //Set Log-Level:
  auto logLevel = ros::console::levels::Info;
  if (_logDebug) {
    logLevel = ros::console::levels::Info;
  }

  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, logLevel)) {
    ros::console::notifyLoggerLevelsChanged();
  }

  _imageSubscriber = new image_transport::SubscriberFilter(_imageTransport, _imageTopic, 10);
  _cameraInfoSubscriber =
      new message_filters::Subscriber<sensor_msgs::CameraInfo>(_nodeHandle, _camInfoTopic, 10);
  _imuSubscriber = new message_filters::Subscriber<sensor_msgs::Imu>(_nodeHandle, _imuTopic, 100);
  _synchronizer = new message_filters::Synchronizer<SyncPolicie>(SyncPolicie(200),
                                                                 *_imageSubscriber,
                                                                 *_cameraInfoSubscriber,
                                                                 *_imuSubscriber);

  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
  _estimatedOdomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom_estimated", 10, true);
  _refined1OdomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom_refined_once", 10, true);
  _refined2OdomPublisher = _nodeHandle.advertise<nav_msgs::Odometry>("odom_refined_twice", 10, true);
  _synchronizer->registerCallback(boost::bind(&MVO_node::imageCallback, this, _1, _2, _3));

  if (_useCompressed) {
    _nodeHandle.setParam("image_transport", "compressed");
  }

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
  _configLock.lock();
  _mvo.newImage(bridgeImage->image, model, orientationMatCV, _currentConfig);
  _configLock.unlock();
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
  _configLock.lock();
  _currentConfig = config;
  _configLock.unlock();
  ROS_INFO_STREAM("Parameter changed! ");
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
void MVO_node::publishRefinedPosition(cv::Point3d position, cv::Matx33d orientation, int stage) {
  //Pack Message
  nav_msgs::Odometry odomMsg;
  odomMsg.header.stamp = ros::Time::now(); //TODO: Time of Image?
  odomMsg.header.frame_id = "odom";
  odomMsg.child_frame_id = "base_footprint";
  odomMsg.pose.pose = worldPoseFromCameraPosition(position, orientation);
  //Publish
  switch (stage) {
    case 1:_refined1OdomPublisher.publish(odomMsg);
      break;
    case 2:
      _refined2OdomPublisher.publish(odomMsg);
      break;
  }

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
