project (kvector)

add_executable ( graph_test graph_test.cpp)

add_executable ( homotopy_eth homotopy_eth_main.cpp )

target_link_libraries ( homotopy_eth ${BOOSTLIBS}  X11 pthread ${UMFLIBS})


#add_executable ( homotopy homotopy_main.cpp )

#target_link_libraries ( homotopy ${BOOSTLIBS}  X11 pthread ${UMFLIBS})

