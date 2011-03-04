#ifndef __TRACK_EXISTED_OBJECTS_HPP__INCLUDED__
#define __TRACK_EXISTED_OBJECTS_HPP__INCLUDED__

struct parameter_t;
struct directory_structure_t;
struct geometric_info_t;
struct object_info_t;


template <typename Float>
void track_existed_objects(parameter_t const&,  directory_structure_t const&,
			   geometric_info_t const& , object_info_t &,
			   vector<std::vector<std::string> > const &seq,
			   int tt,
			   vector<CImg<unsigned char> > const& images,
			   vector<matrix<float> > const& grd );
#ifdef USE_MPI

template <typename Float>
void track_existed_objects(mpi::communicator& world,
			   parameter_t const&,  directory_structure_t const&,
			   geometric_info_t const& , object_info_t &,
			   vector<std::vector<std::string> > const &seq,
			   int tt,
			   vector<CImg<unsigned char> > const& images,
			   vector<matrix<float> > const& grd );
#endif


#endif
