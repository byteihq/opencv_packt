project(chapter2)

set(IMAGE_READ OFF)
set(VIDEO_READ ON)

if (IMAGE_READ)
    add_definitions(-DIMAGE_READ)
endif()

if(VIDEO_READ)
    add_definitions(-DVIDEO_READ)
endif()

add_executable(${PROJECT_NAME} demo.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/samples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
