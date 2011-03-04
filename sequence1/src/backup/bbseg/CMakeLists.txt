project(bbseg)


add_executable (bbseg bbseg_main.cpp)


target_link_libraries (bbseg ${BOOSTLIBS}  X11 pthread ${UMFLIBS})

#add_executable (bbseg_test bbseg_test.cpp) 


#target_link_libraries (bbseg_test ${BOOSTLIBS} X11 pthread ${UMFLIBS})
