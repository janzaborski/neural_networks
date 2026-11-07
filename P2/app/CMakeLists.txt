file(GLOB_RECURSE APP_MODULES CONFIGURE_DEPENDS *.cppm)

add_library(hopfield_app_lib)
target_sources(hopfield_app_lib PUBLIC FILE_SET CXX_MODULES FILES ${APP_MODULES})
target_link_libraries(hopfield_app_lib PUBLIC hopfield)

add_executable(hopfield_app main.cpp)
target_link_libraries(hopfield_app PRIVATE hopfield_app_lib)
