project(chapter5)


add_executable(${PROJECT_NAME} demo.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/samples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
