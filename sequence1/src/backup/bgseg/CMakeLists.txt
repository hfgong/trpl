project(bgseg)


add_executable (bgseg bgseg_main.cpp)


target_link_libraries (bgseg ${BOOSTLIBS}  X11 pthread ${UMFLIBS})

add_executable (bgseg_test bgseg_test.cpp) 


target_link_libraries (bgseg_test ${BOOSTLIBS} X11 pthread ${UMFLIBS})
