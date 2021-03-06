#ifndef __PROPOSE_NEW_OBJECTS__HPP__INCLUDED__
#define __PROPOSE_NEW_OBJECTS__HPP__INCLUDED__

struct parameter_t;
struct directory_structure_t;
struct geometric_info_t;
struct object_info_t;


template <typename Float>
void propose_new_objects(parameter_t const&,  directory_structure_t const&,
			 geometric_info_t const& , object_info_t const&,
    			 vector<std::vector<std::string> > const &seq,
			 int tt,
			 vector<CImg<unsigned char> > const& images,
			 vector<matrix<float> > const& grd,
			 vector<matrix<float> >& detected_rects);

#ifdef USE_MPI
template <typename Float>
void propose_new_objects(mpi::communicator& world,
			 parameter_t const&,  directory_structure_t const&,
			 geometric_info_t const& , object_info_t const&,
    			 vector<std::vector<std::string> > const &seq,
			 int tt,
			 vector<CImg<unsigned char> > const& images,
			 vector<matrix<float> > const& grd,
			 vector<matrix<float> >& detected_rects);
#endif

#endif
