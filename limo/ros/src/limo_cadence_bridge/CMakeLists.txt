cmake_minimum_required(VERSION 3.0.2)
project(limo_cadence_bridge)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch worlds DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
