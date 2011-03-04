#ifndef __FINALIZE_TRAJECTORY__HPP__INCLUDED__
#define __FINALIZE_TRAJECTORY__HPP__INCLUDED__

#include <set>

void finalize_trajectory(int Ncam, int T, //vector<int> const& good_trlet_index,
			 matrix<int> const& links,
			 vector<object_trj_t> const& trlet_list,
			 matrix<object_trj_t> const& gap_trlet_list,
			 vector<object_trj_t>& final_trj_list,
			 vector<vector<int> >& final_trj_index,
			 matrix<int>& final_state_list);


#endif
