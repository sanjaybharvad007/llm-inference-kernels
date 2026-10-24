cmake_minimum_required(VERSION 3.24)
project(fused_attention_trt LANGUAGES CXX CUDA)
# Builds libfused_attention_trt.so: the TensorRT plugin + the attention kernels (no PyTorch dependency).
#   cmake -S trt -B trt/build -DTENSORRT_ROOT=/usr/lib/x86_64-linux-gnu -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
#   cmake --build trt/build -j
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE NvInferRuntime.h HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIB nvinfer HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)
if(NOT TENSORRT_INCLUDE OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found; set -DTENSORRT_ROOT")
endif()
add_library(fused_attention_trt SHARED
  fused_attention_plugin.cpp
  ../csrc/flash_attn.cu
  ../csrc/flash_attn_mma.cu)
target_include_directories(fused_attention_trt PRIVATE ${TENSORRT_INCLUDE} ../csrc)
target_link_libraries(fused_attention_trt PRIVATE ${TENSORRT_LIB} CUDA::cudart)
target_compile_options(fused_attention_trt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -lineinfo>)
set_target_properties(fused_attention_trt PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
