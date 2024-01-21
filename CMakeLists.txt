cmake_minimum_required(VERSION 3.0)
project( IncrementalSFM )

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(g2o REQUIRED)
find_package(fmt REQUIRED)  # needed by g2o

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(two_view_triangulation src/two_view_triangulation.cpp)

target_link_libraries(two_view_triangulation ${OpenCV_LIBS})
target_link_libraries(two_view_triangulation ${PCL_LIBRARIES})
target_link_libraries(two_view_triangulation g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d)
target_link_libraries(two_view_triangulation fmt::fmt)

target_compile_features(two_view_triangulation PRIVATE cxx_std_11)
