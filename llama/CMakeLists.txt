# Builds sone-vision against a pinned llama.cpp release. Run by scikit-build-core when the
# sone package is installed; the binary goes next to the environment's scripts.
cmake_minimum_required(VERSION 3.28)
project(sone-vision CXX)

include(FetchContent)
# llama.cpp's API moves fast; bump this tag deliberately and re-check vision results.
set(LLAMA_TAG "b11715" CACHE STRING "llama.cpp release tag")
FetchContent_Declare(
    llama
    URL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_TAG}.tar.gz"
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    EXCLUDE_FROM_ALL
)

# Only the libraries: llama, ggml and mtmd, linked statically into one binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
# Compile for this machine's CPU: on a Raspberry Pi 5 this is ~1.6x faster than generic ARM kernels.
set(GGML_NATIVE ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama)

add_executable(sone-vision sone-vision.cpp)
target_compile_features(sone-vision PRIVATE cxx_std_17)
target_link_libraries(sone-vision PRIVATE mtmd llama vendor::nlohmann)

if(NOT DEFINED SKBUILD_SCRIPTS_DIR)
    set(SKBUILD_SCRIPTS_DIR bin)
endif()
install(TARGETS sone-vision RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}" COMPONENT sone)
