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
)

include_directories(
	${catkin_INCLUDE_DIRS}		
)


generate_dynamic_reconfigure_options(
  cfg/dyn_config.cfg
)

catkin_package()
find_package(OpenCV 4.1.0 REQUIRED )



add_executable(mvo_node src/mvo_node.cpp src/SlidingWindow.cpp src/Window.cpp src/mvo.cpp src/IterativeRefinement.cpp src/CornerTracker.cpp src/EpipolarGeometry.cpp)
target_link_libraries(mvo_node ${catkin_LIBRARIES})
add_dependencies(mvo_node mvo_gencfg)

target_compile_options(mvo_node PRIVATE -Werror -Wall -Wextra)
