cmake_minimum_required(VERSION 3.8)
project(calicam_ros LANGUAGES C CXX)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)


# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# find ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# find calicam dependencies
find_package(CUDA REQUIRED)
find_package(LibSGM REQUIRED)
find_package(OpenCV REQUIRED)


file(GLOB SOURCES
    include/*.h
    src/*.cpp
)

if (LIBSGM_FOUND AND CUDA_FOUND)
	message(INFO "Found LibSGM")
	message(INFO "Found CUDA")
  cuda_add_executable(${PROJECT_NAME} ${SOURCES})
else()
  message(INFO "Could NOT find LibSGM")
	message(INFO "Could NOT find CUDA")
  add_executable(${PROJECT_NAME} ${SOURCES})
endif()

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  image_geometry
  cv_bridge
  image_transport
  std_msgs
  geometry_msgs
)

if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

include_directories(include ${OpenCV_INCLUDE_DIRS})

if (LIBSGM_FOUND AND CUDA_FOUND)
	include_directories(${LIBSGM_INCLUDE_DIRS})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUDA_STEREO_SGM")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_STEREO_SGM")
	include_directories(${LIBSGM_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${LIBSGM_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
else()
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
