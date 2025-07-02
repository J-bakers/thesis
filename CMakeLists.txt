cmake_minimum_required(VERSION 3.10)
project(FT_BackEnd)

set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM CONFIG REQUIRED)

add_executable(FT_BackEnd FT_BackEnd.cpp)
target_link_libraries(FT_BackEnd gtsam)

add_executable(gnc_optimizer gnc_optimizer.cpp)
target_link_libraries(gnc_optimizer gtsam)

add_executable(count_robots count_robots.cpp)

add_executable(individual_error individual_error.cpp)
target_link_libraries(individual_error gtsam)

add_executable(levenberg_individual_error levenberg_individual_error.cpp)
target_link_libraries(levenberg_individual_error gtsam)

add_executable(gnc_individual_error gnc_individual_error.cpp)
target_link_libraries(gnc_individual_error gtsam)



