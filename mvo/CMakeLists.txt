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


add_executable(mvo_node src/mvo_node.cpp src/sliding_window/SlidingWindow.cpp src/mvo.cpp src/algorithms/IterativeRefinement.cpp src/algorithms/CornerTracking.cpp src/algorithms/EpipolarGeometry.cpp src/pipeline/PipelineStage.cpp src/pipeline/PipelineStage.h src/pipeline/Merger.cpp src/pipeline/Merger.h src/operations/FeatureOperations.cpp src/operations/FeatureOperations.h src/pipeline/TrackerDetector.cpp src/pipeline/TrackerDetector.hpp src/pipeline/BaselineEstimator.cpp src/pipeline/BaselineEstimator.hpp src/pipeline/Refiner.cpp src/pipeline/Refiner.hpp src/nils_lib/Ringbuffer.hpp src/operations/VisualisationUtils.cpp src/operations/VisualisationUtils.cpp src/pipeline/PipelineBegin.cpp src/pipeline/PipelineBegin.cpp src/pipeline/PipeLineEnd.cpp src/pipeline/PipeLineEnd.hpp)
target_link_libraries(mvo_node ${catkin_LIBRARIES})
target_link_libraries(mvo_node ${CERES_LIBRARIES})
target_link_libraries (mvo_node Eigen3::Eigen)
add_dependencies(mvo_node mvo_gencfg)

target_compile_options(mvo_node PRIVATE -Wall -Werror -pedantic -Wextra)