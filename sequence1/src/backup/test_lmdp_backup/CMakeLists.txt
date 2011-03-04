project (test_lmdp) 

add_executable (test_lmdp test_lmdp.cpp)

target_link_libraries (test_lmdp ${BOOSTLIBS}  X11 pthread ${UMFLIBS})

add_executable (test_lmdp_learn test_lmdp_learn.cpp)

target_link_libraries (test_lmdp_learn ${BOOSTLIBS}  X11 pthread ${UMFLIBS})

add_executable (lmdp_qlearn lmdp_qlearn_main.cpp)

target_link_libraries (lmdp_qlearn ${BOOSTLIBS}  X11 pthread ${UMFLIBS})
