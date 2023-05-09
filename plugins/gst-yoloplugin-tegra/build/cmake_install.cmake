# Install script for directory: /home/nvidia/YOLOv3/deepstream_reference_apps/yolo/plugins/gst-yoloplugin-tegra

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee]|[Dd][Ee][Bb][Uu][Gg])$")
    if(EXISTS "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so")
      file(RPATH_CHECK
           FILE "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so"
           RPATH "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib")
    endif()
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "/usr/lib/aarch64-linux-gnu/gstreamer-1.0" TYPE SHARED_LIBRARY FILES "/home/nvidia/YOLOv3/deepstream_reference_apps/yolo/plugins/gst-yoloplugin-tegra/build/libgstnvyolo.so")
    if(EXISTS "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so")
      file(RPATH_CHANGE
           FILE "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so"
           OLD_RPATH "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib:"
           NEW_RPATH "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvyolo.so")
      endif()
    endif()
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee]|[Dd][Ee][Bb][Uu][Gg])$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/nvidia/YOLOv3/deepstream_reference_apps/yolo/plugins/gst-yoloplugin-tegra/build/lib/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/nvidia/YOLOv3/deepstream_reference_apps/yolo/plugins/gst-yoloplugin-tegra/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
