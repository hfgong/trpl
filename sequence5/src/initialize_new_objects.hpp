#ifndef __INITIALIZE_NEW_OBJECTS__INCLUDED__
#define __INITIALIZE_NEW_OBJECTS__INCLUDED__


template <typename Float>
void initialize_new_objects(parameter_t const& P,  directory_structure_t const& ds,
			    geometric_info_t const& gi, object_info_t& oi,
			    vector<std::vector<std::string> > const &seq, int tt,
			    vector<CImg<unsigned char> > const& images,
			    vector<matrix<float> > const& grd,
			    vector<matrix<float> >& detected_rects);
#ifdef USE_MPI
template <typename Float>
void initialize_new_objects(mpi::communicator& world,
			    parameter_t const& P,  directory_structure_t const& ds,
			    geometric_info_t const& gi, object_info_t& oi,
			    vector<std::vector<std::string> > const &seq, int tt,
			    vector<CImg<unsigned char> > const& images,
			    vector<matrix<float> > const& grd,
			    vector<matrix<float> >& detected_rects);
#endif


#endif
