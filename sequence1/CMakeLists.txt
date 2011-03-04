
cmake_minimum_required (VERSION 2.6)
#set(CMAKE_CXX_COMPILER /usr/bin/g++-4.1)

project (sequence)


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(DEP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../whole_sequence/)

#release
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -msse3 -mfpmath=sse")

#debug
set(CMAKE_CXX_FLAGS_DEBUG  -ggdb3)


set(UMFDIR ${DEP_PATH}/UMFPACK_ALL)

set(GLPKDIR ${DEP_PATH}/glpk-4.44/build)

set(BOOSTLIBS mpi_cxx boost_mpi-mt boost_date_time-mt 
	      boost_filesystem-mt boost_serialization-mt)

set(UMFLIBS umfpack amd cholmod camd colamd ccolamd metis blas glpk)

include_directories (${DEP_PATH}/boost-1.40.0.cmake4/ 
	${DEP_PATH}/openmpi-1.4.2/build/include/
        ${DEP_PATH}/CImg-1.3.2/
	${UMFDIR}/UMFPACK/Include
	${UMFDIR}/UFconfig
	${UMFDIR}/AMD/Include
	${GLPKDIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include)


link_directories (${DEP_PATH}/boost-1.40.0.cmake4/build/lib/
	${DEP_PATH}/openmpi-1.4.2/build/lib/
	${UMFDIR}/UMFPACK/Lib
	${UMFDIR}/AMD/Lib
	${UMFDIR}/CHOLMOD/Lib
	${UMFDIR}/CAMD/Lib
	${UMFDIR}/COLAMD/Lib
	${UMFDIR}/CCOLAMD/Lib
	${UMFDIR}/metis-4.0
	${GLPKDIR}/lib)

add_subdirectory (src)
