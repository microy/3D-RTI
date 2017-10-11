CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(Photometric-Stereo)

# required packages
FIND_PACKAGE(OpenCV REQUIRED)

#required macros and variables
ADD_DEFINITIONS(-std=c++11)

# add all sources automagically
FILE(GLOB SOURCES main.cpp)

ADD_EXECUTABLE(main ${SOURCES})
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})
