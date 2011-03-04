#ifndef __POST__PROCESSING__IMPL__HPP__INCLUDED__
#define __POST__PROCESSING__IMPL__HPP__INCLUDED__

#include "linprog_app.hpp"

void post_process_trj(vector<object_trj_t> const& trlet_list,
		      vector<object_trj_t> const& final_trj_list,
		      vector<vector<int> > const& final_trj_index,
		      matrix<int> const& final_state_list,
		      int Ncam, 
		      int len_thr,
		      vector<object_trj_t>& merged_trj_list,
		      vector<vector<int> >& merged_trj_index,
		      matrix<int>& merged_state_list)
{
    using namespace boost::lambda;
    int nt = final_trj_list.size();
    vector<int> final_trj_len(scalar_vector<int>(nt, 0));

    for(int ff=0; ff<nt; ++ff)
    {
	final_trj_len(ff) = final_trj_list(ff).endt+1-final_trj_list(ff).startt;
    }

    vector<bool> stable_flag;
    array1d_transform(final_trj_len, stable_flag, _1>=len_thr);
    std::vector<int> stable_idx;
    for(int ii=0; ii<stable_flag.size(); ++ii)
    {
	if(stable_flag(ii)) stable_idx.push_back(ii);
    }

    vector<int> stable_trj_len(stable_idx.size());
    for(int ii=0; ii<stable_idx.size(); ++ii)
    {
	int kk = stable_idx[ii];
	stable_trj_len(ii) = final_trj_len(kk);
    }


    vector<std::pair<int, int> > tmp(stable_idx.size());
    std::transform(stable_idx.begin(), stable_idx.end(),
		   stable_trj_len.begin(), tmp.begin(),
		   std::make_pair<int, int>);
    std::sort(tmp.begin(), tmp.end(),
	      bind(std::greater<int>(),
		   bind(&std::pair<int, int>::second, _1),
		   bind(&std::pair<int, int>::second, _2) ) );


    vector<object_trj_t> stable_trj_list(tmp.size());
    vector<vector<int> > stable_trj_index(tmp.size());
    matrix<int> stable_state_list(tmp.size(), final_state_list.size2());
    for(int ii=0; ii<tmp.size(); ++ii)
    {
	int kk = tmp(ii).first;
	stable_trj_list(ii) = final_trj_list(kk);
	stable_trj_index(ii) = final_trj_index(kk);
	stable_trj_len(ii) = tmp(ii).second;
	row(stable_state_list, ii) = row(final_state_list, kk);
    }

    vector<bool> final_used_flag(scalar_vector<bool>(nt, false));
    std::vector<object_trj_t> final_trj_list2;
    std::vector<vector<int> > final_trj_index2;
    std::vector<vector<int> > final_state_list2;


    int cur = -1;
    for(int kk=0; kk<stable_trj_list.size(); ++kk)
    {
	if(final_used_flag(kk)) continue;
	cur++;
	final_used_flag(kk) = true;
	final_trj_index2.push_back(stable_trj_index(kk));
	final_trj_list2.push_back(stable_trj_list(kk));
	final_state_list2.push_back(row(stable_state_list, kk));
	for(int ff=0; ff<stable_trj_list.size(); ++ff)
	{
	    if(final_used_flag(ff)) continue;
	    if(final_trj_list2[cur].startt > stable_trj_list(ff).endt ||
	       final_trj_list2[cur].endt < stable_trj_list(ff).startt) 
		continue;
	    int conflict = 0;
	    int time1 = std::max(final_trj_list2[cur].startt, stable_trj_list(ff).startt);
	    int time2 = std::min(final_trj_list2[cur].endt, stable_trj_list(ff).endt);

	    for(int tt=time1; tt<=time2; ++tt)
	    {
		if(final_state_list2[cur](tt) == 1 &&
		   stable_state_list(ff, tt) == 1)
		    conflict++;
	    }
	    if(conflict>1)     continue;

	    float overlap_score = 0.0f;
   

	    for(int tt=time1; tt<=time2; ++tt)
	    {
		if(final_state_list2[cur](tt) <0 ||
		   stable_state_list(ff, tt) <0) continue;

		for(int cam=0; cam<Ncam; ++cam)
		{
		    vector<float> r1(row(final_trj_list2[cur].trj(cam), tt));
		    vector<float> r2(row(stable_trj_list(ff).trj(cam), tt));
		    float ar1 = (r1(2)-r1(0))*(r1(3)-r1(1));
		    float ar2 = (r2(2)-r2(0))*(r2(3)-r2(1));
		    float inar = rectint(r1, r2);
		    //(std::min(r1(2), r2(2))-std::max(r1(0), r2(0)))
		    //*(std::min(r1(3), r2(3))-std::max(r1(1), r2(1)));
		    inar = std::max(0.0f, inar);
		    float rate = std::min(inar/ar1, inar/ar2);
		    
		    overlap_score += rate;
		}
	    }
	    overlap_score /= std::max(time2-time1+1, 12)*Ncam;

	    float appsim_score = -100.0f;
	    for(int nn1=0; nn1<final_trj_index2[cur].size(); ++nn1)
	    {
		for(int nn2=0; nn2<stable_trj_index(ff).size(); ++nn2)
		{
		    float asc = appmodel_match(trlet_list(nn1).hist_p,
					      trlet_list(nn1).hist_q,
					      trlet_list(nn2).hist_p,
					      trlet_list(nn2).hist_q);
		    appsim_score = std::max(appsim_score, asc/Ncam);

		}
	    }

	    float merge_score = overlap_score + appsim_score/3.0f;
#if 0
	    std::cout<<project(final_state_list2[cur], range(20, 40))<<std::endl;
	    std::cout<<project(stable_state_list, range(ff, ff+1), range(20, 40))<<std::endl;
	    std::cout<<"overlap_score="<<overlap_score<<std::endl;
	    std::cout<<"appsim_score="<<appsim_score<<std::endl;
#endif
	    //if(merge_score>4.0f)		
	    if(merge_score>1.5f)		
	    {
		final_used_flag(ff) = true;
		int time1 = std::min(final_trj_list2[cur].startt, stable_trj_list(ff).startt);
		int time2 = std::max(final_trj_list2[cur].endt, stable_trj_list(ff).endt);
		object_trj_t trj=final_trj_list2[cur];
		vector<int> state = final_state_list2[cur];
		for(int tt=time1; tt<=time2; ++tt)
		{
		    if(final_state_list2[cur](tt)<stable_state_list(ff, tt))
		    {
			for(int cam=0; cam<Ncam; ++cam)
			{
			    row(trj.trj(cam), tt) = row(stable_trj_list(ff).trj(cam), tt);
			}
			row(trj.trj_3d, tt) = row(stable_trj_list(ff).trj_3d, tt);
		    }
		    if(final_state_list2[cur](tt)==stable_state_list(ff, tt))
		    {
			for(int cam=0; cam<Ncam; ++cam)
			{
			    row(trj.trj(cam), tt) = (
				row(stable_trj_list(ff).trj(cam), tt)+
				row(trj.trj(cam), tt) )/2.0f;
			}
			row(trj.trj_3d, tt) = (
			    row(stable_trj_list(ff).trj_3d, tt) + 
			    row(trj.trj_3d, tt) )/2.0f;
		    }
		    state(tt) = std::max(final_state_list2[cur](tt),
					 stable_state_list(ff, tt));
		}

		trj.startt = time1;
		trj.endt = time2;
		final_trj_list2[cur] = trj; 
		final_state_list2[cur] = state;
		int ni = final_trj_index2[cur].size()+stable_trj_index(ff).size();
		vector<int> index_tmp(ni);
		std::copy(final_trj_index2[cur].begin(), final_trj_index2[cur].end(),
			  index_tmp.begin());
		std::copy(stable_trj_index(ff).begin(), stable_trj_index(ff).end(),
			  index_tmp.begin()+final_trj_index2[cur].size());
		std::sort(index_tmp.begin(), index_tmp.end());
		final_trj_index2[cur] = index_tmp;
	    }

	}

    }

    int T = final_state_list.size2();

    merged_trj_list = vector<object_trj_t>(final_trj_list2.size());
    merged_trj_index = vector<vector<int> >(final_trj_index2.size());
    merged_state_list = matrix<int>(final_state_list2.size(), T);

    std::copy(final_trj_list2.begin(), final_trj_list2.end(),
	      merged_trj_list.begin() );
    std::copy(final_trj_index2.begin(), final_trj_index2.end(),
	      merged_trj_index.begin() );
    for(int ii=0; ii<final_state_list2.size(); ++ii)
    {
	row(merged_state_list, ii) = final_state_list2[ii];
    }

}

#endif
