#include "yolo2lidar.h"
#include <X11/Xlib.h>

namespace Object_Detection {
CORE_Perception::CORE_Perception(){
    // sub & pub
    hesai_sub = node.subscribe<sensor_msgs::PointCloud2Ptr> ("/pandar_points", 100, &CORE_Perception::HesaiCallback, this); //Subscribe Pandar 40M

    // Polar View Data
    total_num = 0;

    // Darknet load to data
    datacfg_ = "/home/a/catkin_ws/src/yolo2lidar/darknet/data/obj.data";
    cfg_ = "/home/a/catkin_ws/src/yolo2lidar/yolo_network_config/cfg/Gaussian_yolov3_BDD.cfg";
//    cfg_ = "/home/a/catkin_ws/src/yolo2lidar/darknet/cfg/yolov4.cfg";
    weights_ = "/home/a/catkin_ws/src/yolo2lidar/yolo_network_config/weights/Gaussian_yolov3_BDD_best.weights";
//    weights_ = "/home/a/catkin_ws/src/yolo2lidar/darknet/backup/yolov4.weights";

    list *options = read_data_cfg(datacfg_);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int name_size = 0;
    names = get_labels_custom(name_list, &name_size); //obj.data & obj.name

    alphabet = load_alphabet(); //class name write(label)
    net = parse_network_cfg_custom(cfg_, 1, 1); //xxx.cfg and batch = 1
    if(weights_){
        load_weights(&net, weights_); //xxx.weights load
    }
    net.benchmark_layers = 0;
    fuse_conv_batchnorm(net); //convolution and shortcut layer
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != name_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, name_size, net.layers[net.n - 1].classes, cfg_);
        if (net.layers[net.n - 1].classes > name_size) getchar();
    }
}

float lowest_vertical_ang = -25.0, highest_vertical_ang = 15.0;
float prev_dis = (float)INT_MAX;

void CORE_Perception::HesaiCallback(const sensor_msgs::PointCloud2Ptr scan){
    m_lidar_point.clear();
    m_ring.clear();

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> arr;
    pcl::fromROSMsg(*scan, arr);

    for(int i = 0; i < arr.size(); i++){
        pcl::PointXYZI pt;
        pt.x         = arr[i].x;
        pt.y         = arr[i].y;
        pt.z         = arr[i].z;
        pt.intensity = arr[i].intensity;

        m_ring.push_back(arr[i].ring);
        m_lidar_point.push_back(pt);
    }

    init();
}

void CORE_Perception::init(){
    cv::Mat polar_img = PCL_TO_PolarView(m_lidar_point);
    image convert_img = mat_to_image(polar_img);

    ros::Time t_total = ros::Time::now();
    ros_detector(datacfg_, cfg_, weights_, convert_img, .25, .5, 0, 0, 0, 0, 0);
    ros::Duration d_total = ros::Time::now() - t_total;
    std::cout << d_total.toSec()*1000.0 << "ms" << std::endl;
}

cv::Mat CORE_Perception::PCL_TO_PolarView(pcl::PointCloud<pcl::PointXYZI> pt){
    std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > pp = m_lidar_point.points;
    cv::Mat img = cv::Mat::zeros(164, 1440, CV_8UC3);
    uchar *img_data = img.data;

    for(int i=0; i<pp.size(); i++)
    {
        pcl::PointXYZI po = pp[i];
        if((-3.56 < po.x) && (po.x < 0.3) && (-1.275 < po.y) && (po.y < 0.1)) continue;
        if(po.z < -1.0 || po.z > 3.0) continue;

        float dis = sqrt(po.x*po.x + po.y*po.y + po.z*po.z);
        if(dis > 80.0 || dis <= 1.0) continue;

        float deg1 = -(atan2(po.y, po.x) * 57.29578 - 180.0); //180.0 / M_PI = 57.29578
        float deg2 = acos((po.x*po.x + po.y*po.y) / (sqrt(po.x*po.x + po.y*po.y + po.z*po.z) * sqrt(po.x*po.x + po.y*po.y))) * 57.29578; //180.0 / M_PI = 57.29578
        if(po.z < 0.0) deg2 = -deg2;

        if(deg2 > highest_vertical_ang || deg2 < lowest_vertical_ang) continue;

        float x = (int)(deg1 * 4.0);
        float y = (int)((deg2 - highest_vertical_ang) * (-4.1));
        if((y < 0.0) || (y > img.rows)) continue;

        float d_dis = prev_dis - dis;
        if(d_dis < 0) d_dis = -d_dis;

//        unsigned char z3_ = 0, image3_z_ = img3_data[(int)y * img3.cols + (int)x];
        unsigned char z3 = 0, image3_z = img_data[((int)y * img.cols * 3 + (int)x * 3) + 0];

        if(d_dis <= 0.2) z3 = 255;
        else if((0.2 < d_dis) && (d_dis <= 1.0)) z3 = (d_dis - 1.0) * (-318.75);  // = 255 / (0.2 - 1.0);
        else if(d_dis > 1.0) z3 = 0;

        prev_dis = dis;

        if (z3 > image3_z)
        {
//            img3_data[(int)y * img3.cols + (int)x] = z3;

//            if(y-1 > 0) img3_data[(int)(y-1) * img3.cols + (int)x] = z3;
//            if(y+1 < img3.rows) img3_data[(int)(y+1) * img3.cols + (int)x] = z3;
//            if(y-2 > 0) img3_data[(int)(y-2) * img3.cols + (int)x] = z3;
//            if(y+2 < img3.rows) img3_data[(int)(y+2) * img3.cols + (int)x] = z3;


            img_data[((int)y * img.cols * 3 + (int)x * 3) + 0] = z3;

            if(y-1 > 0) img_data[((int)(y-1) * img.cols * 3 + (int)x * 3) + 0] = z3;
            if(y+1 < img.rows) img_data[((int)(y+1) * img.cols * 3 + (int)x * 3) + 0] = z3;
            if(y-2 > 0) img_data[((int)(y-2) * img.cols * 3 + (int)x * 3) + 0] = z3;
            if(y+2 < img.rows) img_data[((int)(y+2) * img.cols * 3 + (int)x * 3) + 0] = z3;
        }

//        unsigned char z = 0, image1_z = img1_data[(int)y * img1.cols + (int)x];//img1.at <unsigned char>((int)y, (int)x);
        unsigned char z = 0, image1_z = img_data[((int)y * img.cols * 3 + (int)x * 3) + 2];

        if((1.0 < dis) && (dis <= 3.0)) z = 255;
        else if((3.0 < dis) && (dis <= 30.0))
        {
            z = (dis - 30.0) * (-9.44); //-9.44 = 255 / (3.0 - 30.0);
        }

        if (z > image1_z)
        {
//            img1_data[(int)y * img1.cols + (int)x] = z;

//            if(y-1 > 0) img1_data[(int)(y-1) * img1.cols + (int)x] = z;
//            if(y+1 < img1.rows) img1_data[(int)(y+1) * img1.cols + (int)x] = z;
//            if(y-2 > 0) img1_data[(int)(y-2) * img1.cols + (int)x] = z;
//            if(y+2 < img1.rows) img1_data[(int)(y+2) * img1.cols + (int)x] = z;

            img_data[((int)y * img.cols * 3 + (int)x * 3) + 2] = z;

            if(y-1 > 0) img_data[((int)(y-1) * img.cols * 3 + (int)x * 3) + 2] = z;
            if(y+1 < img.rows) img_data[((int)(y+1) * img.cols * 3 + (int)x * 3) + 2] = z;
            if(y-2 > 0) img_data[((int)(y-2) * img.cols * 3 + (int)x * 3) + 2] = z;
            if(y+2 < img.rows) img_data[((int)(y+2) * img.cols * 3 + (int)x * 3) + 2] = z;

//            img1_data[(int)y * img.cols + (int)x] = z;

//            if(y-1 > 0) img1_data[((int)(y-1) * img1.cols * 1 + (int)x * 1)] = z;
//            if(y+1 < img1.rows) img1_data[((int)(y+1) * img1.cols * 1 + (int)x * 1)] = z;
//            if(y-2 > 0) img1_data[((int)(y-2) * img1.cols * 1 + (int)x * 1)] = z;
//            if(y+2 < img1.rows) img1_data[((int)(y+2) * img1.cols * 1 + (int)x * 1)] = z;
        }

//        unsigned char z2 = 0, image2_z = img2_data[(int)y * img2.cols + (int)x];
        unsigned char z2 = 0, image2_z = img_data[((int)y * img.cols * 3 + (int)x * 3) + 1];

        if(po.intensity <= 5) z2 = 0;
        else if((5.0 < po.intensity) && (po.intensity <= 50.0))
        {
            z2 = (po.intensity - 5.0) * 5.6667; //6.375 = 255 / (50.0 - 10.0);
        }
        else if(50.0 < po.intensity) z2 = 255;

        if (z2 > image2_z)
        {
//            img2_data[(int)y * img2.cols + (int)x] = z2;

//            if(y-1 > 0) img2_data[(int)(y-1) * img2.cols + (int)x] = z2;
//            if(y+1 < img2.rows) img2_data[(int)(y+1) * img2.cols + (int)x] = z2;
//            if(y-2 > 0) img2_data[(int)(y-2) * img2.cols + (int)x] = z2;
//            if(y+2 < img2.rows) img2_data[(int)(y+2) * img2.cols + (int)x] = z2;

            img_data[((int)y * img.cols * 3 + (int)x * 3) + 1] = z2;

            if(y-1 > 0) img_data[((int)(y-1) * img.cols * 3 + (int)x * 3) + 1] = z2;
            if(y+1 < img.rows) img_data[((int)(y+1) * img.cols * 3 + (int)x * 3) + 1] = z2;
            if(y-2 > 0) img_data[((int)(y-2) * img.cols * 3 + (int)x * 3) + 1] = z2;
            if(y+2 < img.rows) img_data[((int)(y+2) * img.cols * 3 + (int)x * 3) + 1] = z2;
        }
    }

//    cv::imshow("img1", img1);
//    cv::imshow("img2", img2);
//    cv::imshow("img3", img3);
//    cv::imshow("img", img);
//    cv::waitKey(1);
//    double max_distance = 120.0; //sensor spec
//    double res = 5.0;
//    int max_ring = 40;

//    cv::Mat img = cv::Mat::zeros(max_ring*2, 360*res, CV_8UC3);

//    std::vector<cv::Point3f> pre_pt;
//    pre_pt.resize(max_ring);
//    for(int i = 0; i < pt.size(); i++){
//        double dis = sqrt(pow(pt.at(i).x, 2) + pow(pt.at(i).y, 2) + pow(pt.at(i).z, 2));
//        double theta = atan2(pt.at(i).y, pt.at(i).x);

//        theta = theta*(180.0/3.14159265359);

//        img.at<cv::Vec3b>((max_ring - m_ring.at(i) - 1)*2, (360*res*0.5) + theta*res)[0] = 255*(dis/max_distance);
//        img.at<cv::Vec3b>((max_ring - m_ring.at(i) - 1)*2, (360*res*0.5) + theta*res)[1] = pt.at(i).intensity;

//        if(i >= max_ring){
//            double sub_dis = sqrt(pow(pt.at(i).x - pre_pt.at(m_ring.at(i)).x, 2)
//                                  + pow(pt.at(i).y - pre_pt.at(m_ring.at(i)).y, 2)
//                                  + pow(pt.at(i).z - pre_pt.at(m_ring.at(i)).z, 2));

//            img.at<cv::Vec3b>((max_ring - m_ring.at(i) - 1)*2, (360*res*0.5) + theta*res)[2] = 255*(sub_dis/max_distance);
//        }
//        cv::Point3f pre_pt_arr;
//        pre_pt_arr.x = pt.at(i).x;
//        pre_pt_arr.y = pt.at(i).y;
//        pre_pt_arr.z = pt.at(i).z;

//        pre_pt.at(m_ring.at(i)) = pre_pt_arr;
//    }

//    cv::imshow("polar_view", img);
//    cv::waitKey(4);

//    std::string file_name;

//    if(total_num % 5 == 0){
//        file_name = "/home/a/hesai_data/";
//        file_name += std::to_string(total_num);
//        file_name += ".png";
//        cv::imwrite(file_name, img);
//    }
//    total_num++;

    return img;
}

pcl::PointCloud<pcl::PointXYZI> CORE_Perception::PolarView_TO_PCL(cv::Mat img){
    pcl::PointCloud <pcl::PointXYZI> result;



    return result;
}

image CORE_Perception::mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

cv::Mat CORE_Perception::image_to_mat(image img){
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}

void CORE_Perception::ros_detector(char *datacfg, char *cfgfile, char *weightfile, const image input_im, float thresh,
                                 float hier_thresh, int dont_show, int ext_output, int save_labels, int letter_box, int benchmark_layers){
    frameWidth_ = 1800; frameHeight_ = 39;
    srand(2222222);
    float nms = .45;

    image im = input_im;
    image sized;
    if(letter_box) sized = letterbox_image(im, net.w, net.h);
    else sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n-1];
    roiBoxes_ = (Object_Detection::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(Object_Detection::RosBox_));
    rosBoxes_ = std::vector<std::vector<RosBox_> >(l.classes);
    rosBoxCounter_ = std::vector<int>(l.classes);



    float *X = sized.data;

    double time = get_time_point();
    network_predict(net, X);

    int nboxes = 0;
    detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
    if (nms) {
        if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
    }
    draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

    std::cout << nboxes << std::endl;

    int i, j;
    int count = 0;
    for(i = 0; i < nboxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if(xmin < 0)
            xmin = 0;
        if(ymin < 0)
            ymin = 0;
        if(xmax > 1)
            xmax = 1;
        if(ymax > 1)
            ymax = 1;

        // iterate through possible boxes and collect the bounding boxes
        for(j = 0; j < l.classes; ++j){
            if(dets[i].prob[j]){
                float x_center = (xmin + xmax) / 2;
                float y_center = (ymin + ymax) / 2;
                float BoundingBox_width = xmax - xmin;
                float BoundingBox_height = ymax - ymin;

                // define bounding box
                // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
                if(BoundingBox_width > 0.01 && BoundingBox_height > 0.01){
                    roiBoxes_[count].x = x_center;
                    roiBoxes_[count].y = y_center;
                    roiBoxes_[count].w = BoundingBox_width;
                    roiBoxes_[count].h = BoundingBox_height;
                    roiBoxes_[count].Class = j;
                    roiBoxes_[count].prob = dets[i].prob[j];
                    count++;
                }
            }
        }
    }
    if(count == 0){
        roiBoxes_[0].num = 0;
    }
    else{
        roiBoxes_[0].num = count;
    }

    int num = roiBoxes_[0].num;
    if(num > 0 && num <= 100){
        for(int i = 0; i < num; i++){
            for(int j = 0; j < l.classes; j++){
                if(roiBoxes_[i].Class == j){
                    rosBoxes_[j].push_back(roiBoxes_[i]);
                    rosBoxCounter_[j]++;
                }
            }
        }

        for(int i = 0; i < l.classes; i++){
            if(rosBoxCounter_[i] > 0){
                darknet_ros_msgs::BoundingBox boundingBox;

                for(int j = 0; j < rosBoxCounter_[i]; j++){
                    int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
                    int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
                    int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
                    int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

                    boundingBox.Class = "car";
                    boundingBox.probability = rosBoxes_[i][j].prob;
                    boundingBox.xmin = xmin;
                    boundingBox.ymin = ymin;
                    boundingBox.xmax = xmax;
                    boundingBox.ymax = ymax;
                    boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
                }
            }
        }
        boundingBoxesResults_.header.stamp = ros::Time::now();
        boundingBoxesResults_.header.frame_id = "detection";
    }
    cv::Mat output;
    sensor_msgs::ImagePtr msg;
    output = image_to_mat(im);
    msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output).toImageMsg();

    cv::imshow("result", output);
    cv::waitKey(4);
}
}
