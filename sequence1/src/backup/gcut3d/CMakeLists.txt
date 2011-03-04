project(gcut3d)

add_executable (gcut3d gcut3d_main.cpp)

target_link_libraries (gcut3d ${BOOSTLIBS} X11 pthread)


