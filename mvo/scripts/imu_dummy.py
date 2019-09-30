#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
import std_msgs.msg
import math
import tf

SAMPLE_FREQ = 100

def publishIMU(publisher):
    imu_raw = Imu()
    # Header:
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    h.frame_id = 'imu_filter'
    imu_raw.header = h

    # Data:
    imu_raw.orientation_covariance[0] = -1
    imu_raw.linear_acceleration.x = 0
    imu_raw.linear_acceleration.y = 0
    imu_raw.linear_acceleration.z = 0
    imu_raw.linear_acceleration_covariance[0] = -1

    imu_raw.angular_velocity.x = 0
    imu_raw.angular_velocity.y = 0
    imu_raw.angular_velocity.z = 0
    imu_raw.angular_velocity_covariance[0] = -1

    #####################################################################
    ###### magnetometer values should not be stored in orientation ######
    #####################################################################
    imu_raw.orientation.x = 0
    imu_raw.orientation.y = 0
    imu_raw.orientation.z = 0
    imu_raw.orientation_covariance[0] = -1

    publisher.publish(imu_raw)


def imu_node():
    pubIMU = rospy.Publisher("/imu/data", Imu, queue_size=100)

    rospy.init_node("imu_dummy_node", anonymous=True)
    rate = rospy.Rate(SAMPLE_FREQ)  # Pull data with 100hz

    while not rospy.is_shutdown():
        publishIMU(pubIMU)
        rate.sleep()


if __name__ == '__main__':
    try:
        imu_node()
    except rospy.ROSInterruptException:
        pass
