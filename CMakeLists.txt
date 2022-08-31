cmake_minimum_required(VERSION 3.1)
project(tracker)

set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

LIST( APPEND CMAKE_MODULE_PATH /home/vagrant/g2o/cmake_modules )
SET( G2O_ROOT /home/vagrant/g2o )



############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
# find_package(OpenCV 3.0 REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_DIR /home/vagrant/opencv/build)
set( OpenCV_FOUND 1 )

find_package(OpenCV 4.4.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4.4.0)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV >= 4.4.0 not found")
    endif()
endif()
message(STATUS "Use OpenCV ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})


# glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})


# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        )

enable_testing()

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(tracker_node src/tracker.cpp)

target_link_libraries( tracker_node ${THIRD_PARTY_LIBS} Sophus::Sophus)
# add_subdirectory(src)
# add_subdirectory(test)
# add_subdirectory(app)
