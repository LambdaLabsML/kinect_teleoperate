# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Release)
# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(MuJoCo_Include_Dir ${CMAKE_CURRENT_SOURCE_DIR}/../mujoco-3.1.5/include)
set(MuJoCo_Library_Dir ${CMAKE_CURRENT_SOURCE_DIR}/../mujoco-3.1.5/lib)
set(MuJoCo_Libraries mujoco)

link_directories(
    ${MuJoCo_Library_Dir}
    ${hx_idl_cpp_Include_Dir}
    /home/ubuntu/unitree_sdk2/lib/x86_64
    /home/ubuntu/unitree_sdk2/thirdparty/lib/x86_64
)

add_executable(kinect_teleoperate main.cpp)

target_include_directories(kinect_teleoperate 
    PRIVATE 
    ../sample_helper_includes 
    ${MuJoCo_Include_Dir}
    include/
    /home/ubuntu/unitree_sdk2/include
    /home/ubuntu/unitree_sdk2/thirdparty/include
    /home/ubuntu/unitree_sdk2/thirdparty/include/ddscxx
    /home/ubuntu/unitree_sdk2/example/g1/low_level
    # unitree_sdk2
)

# Dependencies of this library
target_link_libraries(kinect_teleoperate PRIVATE 
    k4a
    k4abt
    k4arecord
    window_controller_3d::window_controller_3d
    glfw::glfw
    ${MuJoCo_Libraries}
    Eigen3::Eigen
    unitree_sdk2
    ddsc
    ddscxx
    pthread
)

