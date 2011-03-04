project (video_slice)

add_executable (video_slice video_slice_main.cpp)

target_link_libraries (video_slice ${BOOSTLIBS} X11 pthread)

add_executable (video_slice_mpi video_slice_mpi_main.cpp)

target_link_libraries (video_slice_mpi ${BOOSTLIBS} X11 pthread)
