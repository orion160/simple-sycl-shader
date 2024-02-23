cmake_minimum_required(VERSION 3.28)
project("Simple SYCL Shader" LANGUAGES CXX)

find_package(OpenEXR CONFIG REQUIRED)
if("${SYCL_IMPLEMENTATION}" STREQUAL "Intel OneAPI")
    find_package(IntelSYCL REQUIRED)
endif()

add_executable(main)
set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_sources(main
    PRIVATE main.cpp
)
target_link_libraries(main PRIVATE OpenEXR::OpenEXR)
add_custom_command(TARGET main POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:main> $<TARGET_RUNTIME_DLLS:main>
  COMMAND_EXPAND_LISTS
)

if("${SYCL_IMPLEMENTATION}" STREQUAL "Intel OneAPI")
    add_sycl_to_target(TARGET main
        SOURCES main.cpp
    )
endif()
