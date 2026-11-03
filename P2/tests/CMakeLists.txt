file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test_*.cpp)

add_executable(hopfield_tests ${TEST_SOURCES})
target_link_libraries(hopfield_tests PRIVATE hopfield Boost::unit_test_framework)
target_compile_definitions(hopfield_tests PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME hopfield_tests COMMAND hopfield_tests)
