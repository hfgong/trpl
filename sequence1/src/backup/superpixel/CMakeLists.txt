project(superpixel)

add_executable (superpixel superpixel_main.cpp)

target_link_libraries (superpixel ${BOOSTLIBS} X11 pthread)

add_executable (superpixel_slice superpixel_slice_main.cpp)

target_link_libraries (superpixel_slice ${BOOSTLIBS} X11 pthread)


add_executable (vis_superpixel vis_superpixel_main.cpp)

target_link_libraries (vis_superpixel ${BOOSTLIBS} X11 pthread)

