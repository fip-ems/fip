cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fip LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "CUDA Architectures: " ${CMAKE_CUDA_ARCHITECTURES})

if(NOT DEFINED NO_VIS)
    # GLAD & GLFW 
    set(GLAD_DIR ./deps/glad)
    include_directories(${GLAD_DIR}/include)
    set(GLFW_DIR ./deps/glfw)
    add_subdirectory(${GLFW_DIR} glfw)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    # IMGUI
    set(IMGUI_DIR ./deps/imgui)
    add_library(IMGUI STATIC)

    target_sources( IMGUI
        PRIVATE
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        )

    target_include_directories( IMGUI
        PUBLIC ${IMGUI_DIR}
        PUBLIC ${IMGUI_DIR}/backends
        )
    
    target_link_libraries(IMGUI PUBLIC glfw ${CMAKE_DL_LIBS})

    # implot
    add_library(implot 
        STATIC 
        ./deps/implot/implot.h
        ./deps/implot/implot_internal.h
        ./deps/implot/implot.cpp
        ./deps/implot/implot_items.cpp
        ./deps/implot/implot_demo.cpp
        )
    target_link_libraries(implot PUBLIC IMGUI)
endif()

# YAML-CPP
add_subdirectory(./deps/yaml-cpp yaml-cpp)


set( SRC_FILES
    src/main.cpp
    src/kernels.h
    src/kernels.cu
    src/simdata.h
    src/simdata.cpp
    src/timeseries.h
    src/timeseries.cpp
    src/boundary_condition.h
    src/boundary_condition.cpp
    src/bc_kernel_gen.h
    src/bc_kernel_gen.cpp
    src/enumhelper.h
    src/utils.h
    )
if(NOT DEFINED NO_VIS)
    set(SRC_FILES ${SRC_FILES}
        src/vis.h
        src/vis.cpp
        src/shaders.h
        src/imgui_impl_opengl3.cpp
        ${GLAD_DIR}/src/glad.c
    )
endif()

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME}
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ./deps
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -extended-lambda >)

target_link_libraries(${PROJECT_NAME} yaml-cpp)

# libraries needed for nvrtc
if (WIN32)
    target_link_libraries(${PROJECT_NAME} nvrtc.lib cuda.lib )
else()
    target_link_libraries(${PROJECT_NAME} libnvrtc.so libcuda.so )
endif()

if(NOT DEFINED NO_VIS)
    target_link_libraries(${PROJECT_NAME} glfw IMGUI implot)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC NO_VIS)
endif()
