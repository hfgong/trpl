#ifndef __LINPROG_APP_HPP_INCLUDED__
#define __LINPROG_APP_HPP_INCLUDED__

template <class Float>
Float appmodel_match(vector<matrix<Float> > const& hp1,
		     vector<matrix<Float> > const& hq1,
		     vector<matrix<Float> > const& hp2,
		     vector<matrix<Float> > const& hq2);

void prepare_app_affinity(matrix<int> const& Tff, //vector<int> const& good_trlet_index,
			  vector<object_trj_t> const& trlet_list, float thr,
			  matrix<float>& Aff);

template <class Float>
Float appmodel_match_one(Float hp1, Float hq1, Float hp2, Float hq2, Float ep);


void enumerate_rects_refine(vector<float> const& rect,
			    vector<float> const& dx, vector<float> const& dy,
			    matrix<float>& cand_rects);

void prepare_occl_affinity(vector<std::vector<std::string> > const& seq,
			   geometric_info_t const& gi,
			   parameter_t const& P,
			   vector<object_trj_t> const& trlet_list,
			   vector<array<float, 4> > const& model,
			   //vector<int> const& good_trlet_index,
			   matrix<float> const& Tff,
			   matrix<float>& Ocff,
			   matrix<object_trj_t>& gap_trlet_list);

void enumerate_rects_refine(vector<float> const& rect,
			    vector<float> const& dx, vector<float> const& dy,
			    matrix<float>& cand_rects);



#endif
