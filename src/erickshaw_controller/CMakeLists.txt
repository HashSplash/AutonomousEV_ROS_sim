cmake_minimum_required(VERSION 3.0.2)
project(erickshaw_controller)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)


catkin_package(

)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/erickshaw_controller/simple_controller.py
  src/erickshaw_teleop/keyboard.py
  # src/erickshaw_controller/simple_controller.py  
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)