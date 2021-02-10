#ifndef INCLUDE_POINT_TYPES_H_
#define INCLUDE_POINT_TYPES_H_

#pragma once
#pragma GCC diagnostic ignored "-Wwrite-strings"
//c++
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <time.h>

//ros
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseArray.h>

//OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

//darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>

//darknet
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "image_interface.h"
#include "blas.h"
#include "darknet.h"
#include "option_list.h"
#include <sys/time.h>

//PointCloud
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/pointcloudXYZIR.h>

#endif  // INCLUDE_POINT_TYPES_H_


namespace Object_Detection {
//! Bounding box of the detected object.
typedef struct
{
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;

class CORE_Perception{
private:
    ros::NodeHandle node;
    ros::Subscriber hesai_sub; //Hesai Pandar40M

    // Load to Lidar
    pcl::PointCloud <pcl::PointXYZI> m_lidar_point;
    std::vector<int> m_ring;
    int total_num;

    cv::Mat bbox;

    //darknet Parameter
    char* datacfg_;
    char* cfg_;
    char* weights_;
    char* filename_;

    char **names;
    image **alphabet;
    network net;
    int frameWidth_;
    int frameHeight_;
    RosBox_ *roiBoxes_;
    std::vector<std::vector<RosBox_> > rosBoxes_;
    std::vector<int> rosBoxCounter_;
    darknet_ros_msgs::BoundingBoxes boundingBoxesResults_;

public:
    CORE_Perception();
    void HesaiCallback(const sensor_msgs::PointCloud2Ptr scan); //Hesai Callback function

    void init(); //Total management

    //Polar_view Transfer
    cv::Mat PCL_TO_PolarView(pcl::PointCloud <pcl::PointXYZI> pt); //Make Polar View
    pcl::PointCloud <pcl::PointXYZI> PolarView_TO_PCL(cv::Mat img);

    //Darknet Detector
    cv::Mat image_to_mat(image input);
    image mat_to_image(cv::Mat mat);

    void ros_detector(char *datacfg, char *cfgfile, char *weightfile, image input_im, float thresh, float hier_thresh, int dont_show,
                      int ext_output, int save_labels, int letter_box, int benchmark_layers);
};
}
