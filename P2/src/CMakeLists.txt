file(GLOB_RECURSE HOPFIELD_MODULES CONFIGURE_DEPENDS *.cppm)
file(GLOB_RECURSE HOPFIELD_SOURCES CONFIGURE_DEPENDS *.cpp)

add_library(hopfield)
target_sources(hopfield
    PUBLIC FILE_SET CXX_MODULES FILES ${HOPFIELD_MODULES}
    PRIVATE ${HOPFIELD_SOURCES}
)
target_link_libraries(hopfield
    PUBLIC hopfield_options
    PRIVATE Boost::headers Boost::json PNG::PNG JPEG::JPEG
)
