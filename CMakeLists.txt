cmake_minimum_required (VERSION 3.1)

project (miniml)

configure_file(${PROJECT_SOURCE_DIR}/cmake/types.h.in ${PROJECT_SOURCE_DIR}/include/types.h)

file(GLOB INCLUDE "include/*.h")
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.h")

source_group("Include" FILES ${INCLUDE})
source_group("" FILES ${INCLUDE} ${SOURCES} ${HEADERS}) 
source_group("Source Files" FILES "src/main.cpp") 
source_group("Source Files\\Bayes" FILES "src/Bayes.h" "src/Bayes.cpp")
source_group("Source Files\\Histogram" FILES "src/PDFHistogram.h" "src/PDFHistogram.cpp")
source_group("Source Files\\utilities" FILES "src/timer.h")

# OpenCV package
find_package(OpenCV 4.0 REQUIRED core highgui imgproc imgcodecs PATHS "$ENV{OPENCVDIR}/build")

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sets
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 11)
if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /wd5040") 
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")
endif(MSVC)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 
#set(IB_API_INCLUDE_DIR "" CACHE PATH "Path to the IP API")

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

#definitions
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)

add_executable(miniml ${INCLUDE} ${SOURCES} ${HEADERS})

# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(miniml ${OpenCV_LIBS})
