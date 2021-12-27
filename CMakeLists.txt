cmake_minimum_required(VERSION 3.0)

project(neuralnetwork)

add_executable(neuralnetwork main.cpp)

install(TARGETS neuralnetwork RUNTIME DESTINATION bin)
