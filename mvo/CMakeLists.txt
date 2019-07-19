cmake_minimum_required(VERSION 2.8.3)
project(mvo)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
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



add_library(MVO src/mvo.cpp)
find_package(OpenCV 4.1.0 REQUIRED )

add_executable(mvo_node src/mvo_node.cpp)

target_link_libraries(MVO ${OpenCV_LIBS})
target_link_libraries(mvo_node ${catkin_LIBRARIES} MVO)
add_dependencies(mvo_node mvo_gencfg)
