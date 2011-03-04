project(test)


# Add executable called "helloDemo" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
add_executable (test test.cpp)

# Link the executable to the Hello library.
target_link_libraries (test boost_mpi-mt)
 

