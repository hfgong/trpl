#ifndef __POST__PROCESSING__HPP__INCLUDED__
#define __POST__PROCESSING__HPP__INCLUDED__

void post_process_trj(vector<object_trj_t> const& trlet_list,
		      vector<object_trj_t> const& final_trj_list,
		      vector<vector<int> > const& final_trj_index,
		      matrix<int> const& final_state_list,
		      int Ncam, 
		      int len_thr,
		      vector<object_trj_t>& merged_trj_list,
		      vector<vector<int> >& merged_trj_index,
		      matrix<int>& merged_state_list);

#endif
