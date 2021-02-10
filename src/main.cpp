#include <ros/ros.h>
#include "yolo2lidar.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo2lidar");
    ros::NodeHandle nh;
    Object_Detection::CORE_Perception Obj_detection;

    while(ros::ok())
    {
        ros::spin();
    }
}
