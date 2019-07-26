#include "mvo_node.hpp"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mvo_node");
  MVO_node node(ros::NodeHandle(""), ros::NodeHandle("~"));
  ros::spin();
  return 0;
}

MVO_node::MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh)
  : _nodeHandle(nh), _privateNodeHandle(pnh), _imageTransport(nh)
{
  this->init();
}

MVO_node::~MVO_node()
{
}

void MVO_node::init()
{
  _imageSubscriberTopic = "/pylon_camera_node/image_rect";
  _imageSubscriber = _imageTransport.subscribeCamera(_imageSubscriberTopic, 1, &MVO_node::imageCallback, this);
  _dynamicConfigCallBackType = boost::bind(&MVO_node::dynamicConfigCallback, this, _1, _2);
  _dynamicConfigServer.setCallback(_dynamicConfigCallBackType);
}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camInfo)
{
  (void)(camInfo);  // TODO: unused
  cv_bridge::CvImageConstPtr bridgeImage = cv_bridge::toCvShare(image);
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(camInfo);
  _mvo.handleImage(bridgeImage->image, model);
}

void MVO_node::dynamicConfigCallback(mvo::corner_detectorConfig &config, uint32_t level)
{
  (void)(level);  // TODO: unused
  _mvo._cornerTracker.setCornerDetectorParams(config.block_size, config.aperture_size, config.k, config.threshold);
}