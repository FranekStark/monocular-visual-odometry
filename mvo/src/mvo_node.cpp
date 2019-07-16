
#include "mvo_node.hpp"

int main(int argc, char const *argv[])
{
    ros::init(argc, argv, "mvo_node");
    MVO_node node(ros::NodeHandle(""), ros::NodeHandle("~"));
    return 0;
}


MVO_node::MVO_node(ros::NodeHandle nh, ros::NodeHandle pnh):
_nodeHandle(nh),
_privateNodeHandle(pnh),
_imageTransport(nh),
{
    this->init();
}

MVO_node::~MVO_node()
{
}

MVO_node::init(){
    _imageSubscriberTopic = "/cam/rect";
    _imageSubscriber = _imageTransport.subscribeCamera(_imageSubscriberTopic, 1)
}

void MVO_node::imageCallback(const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&){

}