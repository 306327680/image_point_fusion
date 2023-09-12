//
// Created by echo on 23-9-9.
//
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CompressedImage.h"
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <livox_ros_driver/CustomMsg.h>

Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Zero();
Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
Eigen::Vector3d translation;
Eigen::Vector3d point_3d, point_t3d, point_3d_3;
double k1, k2, k3, p1, p2; //distortion parameters
cv::Mat mat;
pcl::PointCloud<pcl::PointXYZI> cloudin;

bool cloud_in = false;
bool image_in = false;
sensor_msgs::PointCloud2 lastPcd;
livox_ros_driver::CustomMsg lastPcd_livox;
void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    cloud_in = true;
    int plsize = msg->point_num;
    cloudin.clear();
    cloudin.resize(plsize);
    for (uint i = 1; i < plsize; i++) {
        if (((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            cloudin[i].x = msg->points[i].x;
            cloudin[i].y = msg->points[i].y;
            cloudin[i].z = msg->points[i].z;
            cloudin[i].intensity = msg->points[i].reflectivity;
        }
    }
    lastPcd.header = msg->header;
}

void PCDCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::fromROSMsg(*msg, cloudin);
    cloud_in = true;
    lastPcd = *msg;
}

void ImageCallback(const sensor_msgs::CompressedImage::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_cam = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat = cv_cam->image;
    image_in = true;
}


pcl::PointCloud<pcl::PointXYZRGB> pclalignImg2LiDAR(cv::Mat mat, pcl::PointCloud<pcl::PointXYZI> cloudin) {
    //去畸变
    cv::Mat image = mat;
    cv::Mat imageCalibration;
    cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = intrinsics(0, 0);
    cameraMatrix.at<double>(0, 1) = intrinsics(0, 1);
    cameraMatrix.at<double>(0, 2) = intrinsics(0, 2);
    cameraMatrix.at<double>(1, 1) = intrinsics(1, 1);
    cameraMatrix.at<double>(1, 2) = intrinsics(1, 2);
    cameraMatrix.at<double>(2, 2) = 1;
    cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << -0.003828, -0.007873, -0.000222, -0.000447, 0.000000);
    k1 = distCoeffs.at<float>(cv::Point(0, 0));
    k2 = distCoeffs.at<float>(cv::Point(1, 0));
    k3 = distCoeffs.at<float>(cv::Point(2, 0));
    p1 = distCoeffs.at<float>(cv::Point(3, 0));
    p2 = distCoeffs.at<float>(cv::Point(4, 0));
    cv::Mat view, rview, map1, map2;
    cv::Size imageSize;
    imageSize = image.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
                            CV_16SC2, map1, map2);
    remap(image, imageCalibration, map1, map2, cv::INTER_LINEAR);
    mat = imageCalibration;
    std::cerr << cameraMatrix << std::endl;
    std::cerr << distCoeffs << std::endl;
    std::cerr << k1 << " " << k2 << " " << k3 << " " << p1 << " " << p2 << std::endl;
    //正经程序
    pcl::PointCloud<pcl::PointXYZRGB> coloured_point_cloud;
    cv::Mat projected_image1;
    for (int i = 0; i < cloudin.size(); ++i) {
        pcl::PointXYZRGB temp_point;
        temp_point.x = cloudin[i].x;
        temp_point.y = cloudin[i].y;
        temp_point.z = cloudin[i].z;

        point_3d << cloudin[i].x, cloudin[i].y, cloudin[i].z;
        point_3d_3 = rotation * point_3d + translation;//外参平移现在的点
        point_t3d = point_3d_3 / point_3d_3[2];//z轴归一化

        float r_2 = point_t3d[0] * point_t3d[0] + point_t3d[1] * point_t3d[1];
        float x_ori = point_t3d[0];
        float y_ori = point_t3d[1];

        point_t3d[0] = x_ori * (1 + k1 * r_2 + k2 * pow(r_2, 2) + k3 * pow(r_2, 3)) + 2 * p1 * x_ori * y_ori +
                       p2 * (r_2 + 2 * pow(x_ori, 2));
        point_t3d[1] = y_ori * (1 + k1 * r_2 + k2 * pow(r_2, 2) + k3 * pow(r_2, 3)) + 2 * p2 * x_ori * y_ori +
                       p1 * (r_2 + 2 * pow(y_ori, 2));

        int image_u = (int) (intrinsics(0, 0) * point_t3d[0] + intrinsics(0, 2));
        int image_v = (int) (intrinsics(1, 1) * point_t3d[1] + intrinsics(1, 2));
        int u = mat.size().width;
        int v = mat.size().height;

        if (10 <= image_u && image_u < mat.size().width - 10 && 10 <= image_v && image_v < mat.size().height - 10 &&
            temp_point.x > 0) {
            cv::Vec3b colour = mat.at<cv::Vec3b>(cv::Point(image_u, image_v));
            temp_point.r = colour[2];
            temp_point.g = colour[1];
            temp_point.b = colour[0];
            coloured_point_cloud.push_back(temp_point);
        }
    }
    return coloured_point_cloud;
}

int main(int argc, char **argv) {

    pcl::PointCloud<pcl::PointXYZRGB> result;
    Eigen::Matrix4d T;
    T << -0.002151, -0.999995, -0.00178768, 0.0381084,
            -0.00911536, 0.00180711, -0.999956, -0.0718582,
            0.999956, -0.00213453, -0.00911927, -0.0663774,
            0, 0, 0, 1;
    Eigen::Affine3d TT;
    TT.matrix() = T;
    rotation = TT.rotation();
    translation = TT.translation();
    intrinsics << 878.903459, 0.000000, 979.188754,
            0.000000, 878.645312, 546.835879,
            0.000000, 0.000000, 1.000000;

    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::Publisher chatter_pub = n.advertise<sensor_msgs::PointCloud2>("/ColorPcd", 1000);
    ros::Rate loop_rate(10);
    ros::Subscriber sub = n.subscribe("/cloud_registered_body", 1000, PCDCallback);
   // ros::Subscriber sub0 = n.subscribe("/livox/lidar", 1000, avia_handler);
    ros::Subscriber sub1 = n.subscribe("/usb_cam/image_raw/compressed", 1000, ImageCallback);
   // ros::Subscriber sub2 = n.subscribe("/Odometry", 1000, avia_handler);

    while (ros::ok()) {
        sensor_msgs::PointCloud2 colorPCD;
        pcl::PointCloud<pcl::PointXYZRGB> result_pub;
        pcl::PCLPointCloud2 pcl_frame;
        if (cloud_in && image_in) {
            result_pub = pclalignImg2LiDAR(mat, cloudin);
            pcl::toPCLPointCloud2(result_pub, pcl_frame);
            pcl_conversions::fromPCL(pcl_frame, colorPCD);
            colorPCD.header = lastPcd.header;
            colorPCD.header.frame_id = "body";
            std::cerr << result_pub.size() << std::endl;
            chatter_pub.publish(colorPCD);
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}