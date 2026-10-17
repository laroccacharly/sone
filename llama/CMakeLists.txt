# Builds sone-vision against the llama.cpp checkout that build.py puts in vendor/.
cmake_minimum_required(VERSION 3.14)
project(sone-vision CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp" CACHE PATH "llama.cpp checkout")

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

add_subdirectory("${LLAMA_DIR}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(sone-vision sone-vision.cpp)
target_compile_features(sone-vision PRIVATE cxx_std_17)
target_link_libraries(sone-vision PRIVATE mtmd llama vendor::nlohmann)
set_target_properties(sone-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
