#ifndef __LINPROG_ALONE_HPP_INCLUDED__
#define __LINPROG_ALONE_HPP_INCLUDED__

//#include "planning.hpp"
void prepare_alone_affinity(mpi::communicator& world,
			   vector<std::vector<std::string> > const& seq,
			   geometric_info_t const& gi,
			   parameter_t const& P, int plan_advance,
			   float plff_thr,
			   vector<object_trj_t> const& trlet_list,
			   vector<array<float, 4> > const& model,
			   matrix<float> const& Tff,
			   matrix<float>& Plff,
			   matrix<object_trj_t>& gap_trlet_list,
			   matrix<int>& gap_rind,
			   matrix<matrix<int> > & gap_paths,
			   directory_structure_t &ds);


#endif
