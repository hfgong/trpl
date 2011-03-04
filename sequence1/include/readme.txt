
cvpr_array_traits.hpp --- traits to adapt ublas::matrix, ublas::vector, multi_array and CImg
detail --- detailed code to be included in a header file.

labelmap_as_graph.hpp --- adapt labelmap as a boost::graph, working in progress
labelmap.hpp --- convert a labelmap as a boost::graph.

misc_utils.hpp --- misc utils, such as std::copy_if

multi_array_serialization.hpp --- multi_array serialization, to work with mpi, but failed because of the size check in the operator=. 
			      ublas::matrix ships with serialization by default. So now I use ublas::matrix for mpi.
real_timer.hpp --- real time timer, based on boost::datetime. The reason for this is that boost::timer counts only cpu time.

statistics.hpp --- mean, var, gaussian et al.
text_file.hpp --- read and write matrices

ublas_cimg.hpp --- conversion between ublas::matrix and CImg, not necessary any more because of cvpr_array_traits.hpp
ublas_random.hpp --- random number and matrices. The generator cannot work with std::generate because the copy constructor and pointer members. 
		 I am goint to correct this issue.

