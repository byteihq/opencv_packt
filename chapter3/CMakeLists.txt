project(chapter3)

add_executable(${PROJECT_NAME} demo.cpp)

find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/samples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
