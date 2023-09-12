cmake_minimum_required(VERSION 3.0.2)
project(image_point_fusion)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARIES_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        rosbag
        sensor_msgs
        cv_bridge
        livox_ros_driver
        message_generation
        )

catkin_package(

)


include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)
add_executable(main src/main.cpp)
target_link_libraries(main ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})