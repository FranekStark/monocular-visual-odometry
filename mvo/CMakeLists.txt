cmake_minimum_required(VERSION 2.8.3)
project(mvo)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  cv_bridge
)

include_directories(
	${catkin_INCLUDE_DIRS}		
)

catkin_package()

add_library(MVO src/mvo.cpp)
find_package(OpenCV REQUIRED )

add_executable(mvo_node src/mvo_node.cpp)
target_link_libraries(MVO ${OpenCV_LIBS})
target_link_libraries(mvo_node ${catkin_LIBRARIES} MVO)

