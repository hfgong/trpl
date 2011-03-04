#ifndef __LINPROG_PLAN_HPP_INCLUDED__
#define __LINPROG_PLAN_HPP_INCLUDED__

#include "planning.hpp"
void prepare_plan_affinity(mpi::communicator& world,
			   vector<std::vector<std::string> > const& seq,
			   geometric_info_t const& gi,
			   parameter_t const& P, int plan_advance,
			   float plff_thr,
			   vector<object_trj_t> const& trlet_list,
			   vector<array<float, 4> > const& model,
			   matrix<float> const& Tff,
			   vector<int> const& plan_time,
			   vector<vector<planning_result_item_t> > const& plan_results,
			   matrix<vector<matrix<int> > >& reduced_paths,
			   matrix<float>& Plff,
			   matrix<vector<object_trj_t> >& gap_trlet_list,
			   matrix<int>& gap_rind,
			   matrix<matrix<int> > & gap_paths,
			   directory_structure_t &ds);


#endif
