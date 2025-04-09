cmake_minimum_required(VERSION 3.10)
project(cuda_opencv_example LANGUAGES CXX CUDA)

# Enable C++11 or later
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Enable CUDA
enable_language(CUDA)

# Add your source file
add_executable(hello blur.cu)

# Link OpenCV
target_include_directories(hello PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hello PRIVATE ${OpenCV_LIBS})
