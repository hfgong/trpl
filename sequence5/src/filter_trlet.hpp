#ifndef __FILTER__TRLET__HPP__INCLUDED__
#define __FILTER__TRLET__HPP__INCLUDED__

void compute_seg_score(vector<object_trj_t> const &trlet_list,
		       vector<matrix<matrix<unsigned char> > > const& seg_list,
		       vector<float>& seg_score  );

void filter_trlet(vector<object_trj_t> const &trlet_list,
		  vector<float> const& seg_score,
		  int min_trlet_len,
		  float seg_thresh,
		  vector<object_trj_t> & good_trlet_list,
		  vector<bool>& good_trlet_flag,
		  vector<int>& good_trlet_index);

template <class V1, class V2>
void flag_to_index(V1 const& flag, V2& index);


void prepare_valid_linkset(vector<object_trj_t> const &trlet_list,
			   int t_thresh,
			   float v_thresh,
			   matrix<int>& Tff);



#endif
