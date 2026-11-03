add_executable(hopfield_app main.cpp)
target_link_libraries(hopfield_app PRIVATE hopfield Boost::program_options)
