cmake_minimum_required(VERSION 2.8.3)
project(mvo)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  image_geometry
  cv_bridge
  dynamic_reconfigure
  tf2_ros
  nav_msgs
)

include_directories(
	${catkin_INCLUDE_DIRS}		
)


generate_dynamic_reconfigure_options(
  cfg/dyn_config.cfg
)

catkin_package()
find_package(OpenCV 4.1.0 REQUIRED )

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(mvo_node src/mvo_node.cpp src/sliding_window/SlidingWindow.cpp src/mvo.cpp src/IterativeRefinement.cpp src/CornerTracking.cpp src/EpipolarGeometry.cpp src/PipelineStage.cpp src/PipelineStage.h src/Merger.cpp src/Merger.h src/FeatureOperations.cpp src/FeatureOperations.h src/TrackerDetector.cpp src/TrackerDetector.hpp src/BaselineEstimator.cpp src/BaselineEstimator.hpp src/Refiner.cpp src/Refiner.hpp src/nils_lib/Ringbuffer.hpp)
target_link_libraries(mvo_node ${catkin_LIBRARIES})
target_link_libraries(mvo_node ${CERES_LIBRARIES})
target_link_libraries (mvo_node Eigen3::Eigen)
add_dependencies(mvo_node mvo_gencfg)

target_compile_options(mvo_node PRIVATE -Wall -Werror -pedantic -Wextra)