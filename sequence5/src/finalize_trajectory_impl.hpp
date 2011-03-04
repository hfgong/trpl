#ifndef __FINALIZE__TRAJECTORY__IMPL__HPP__INCLUDED__
#define __FINALIZE__TRAJECTORY__IMPL__HPP__INCLUDED__

void finalize_trajectory(int Ncam, int T, matrix<int> const& links,
			 vector<object_trj_t> const& trlet_list,
			 matrix<object_trj_t> const& gap_trlet_list,
			 vector<object_trj_t>& final_trj_list,
			 vector<vector<int> >& final_trj_index,
			 matrix<int>& final_state_list)
{
    using namespace boost::lambda;
    int ng = trlet_list.size();

    std::set<int> ngs(boost::counting_iterator<int>(0),
		      boost::counting_iterator<int>(ng));
    matrix_column<matrix<int> const> cfrom(links, 0);
    matrix_column<matrix<int> const> cto(links, 1);
    std::set<int> lfrom(cfrom.begin(), cfrom.end());
    std::set<int> lto(cto.begin(), cto.end());
    std::set<int> all;
    std::set<int> isola;
    std::set<int> begin;
    std::set_union(lfrom.begin(), lfrom.end(), lto.begin(), lto.end(),
		   std::inserter(all, all.end()));
    std::set_difference(ngs.begin(), ngs.end(), all.begin(), all.end(),
			std::inserter(isola, isola.end()));
    std::set_difference(lfrom.begin(), lfrom.end(), lto.begin(), lto.end(),
			std::inserter(begin, begin.end()));
    vector<int> tmp(begin.size()+isola.size());
    vector<int>::iterator tmp_begin=std::copy(begin.begin(), begin.end(), tmp.begin());
    std::copy(isola.begin(), isola.end(), tmp_begin);

    vector<std::vector<int> > trj_gindex(tmp.size());
    for(int ii=0; ii<trj_gindex.size(); ++ii)	trj_gindex(ii).push_back(tmp(ii));
    
    vector<int> next(scalar_vector<int>(ng, -1));
    for(int ll=0; ll<links.size1(); ++ll) next(links(ll, 0)) = links(ll, 1);

    for(int ii=0; ii<begin.size(); ++ii)
    {
	int bg = *(trj_gindex(ii).rbegin());
	while(next(bg)>=0)
	{
	    trj_gindex(ii).push_back(next(bg));
	    bg = next(bg);
	}
    }

    vector<vector<int> > trj_index(trj_gindex.size());
    for(int ii=0; ii<trj_gindex.size(); ++ii)
    {
	trj_index(ii) = vector<int>(trj_gindex(ii).size());
	for(int jj=0; jj<trj_gindex(ii).size(); ++jj)
	{
	    trj_index(ii)(jj) = trj_gindex(ii)[jj]; //good_trlet_index(trj_gindex(ii)[jj]);
	}
    }

    int nobj = trj_index.size();
    final_trj_index = trj_index;

	//std::cout<<"next="<<next<<std::endl;
	//std::cout<<"links="<<links<<std::endl;
	//std::cout<<"trj_gindex="<<trj_index<<std::endl;



    final_trj_list = vector<object_trj_t>(nobj);
    final_state_list = scalar_matrix<int>(nobj, T, -1);

    for(int ii=0; ii<nobj; ++ii)
    {
	int nns = trj_index(ii)[0];
	int nne = *(trj_index(ii).rbegin());
	object_trj_t& final_trj = final_trj_list(ii);
	final_trj.startt = trlet_list(nns).startt;
	final_trj.endt = trlet_list(nne).endt;
	final_trj.trj = vector<matrix<float> >(Ncam);
	final_trj.trj_3d = scalar_matrix<float>(T, 2, 0);
	final_trj.scores = scalar_matrix<float>(Ncam, T, 0);
	for(int cam=0; cam<Ncam; ++cam)
	{
	    final_trj.trj(cam) = scalar_matrix<float>(T, 4, 0);
	}

	//fill true trlet
	for(int jj=0; jj<trj_index(ii).size(); ++jj)
	{
	    int nn = trj_index(ii)[jj];
	    object_trj_t const& trlet=trlet_list(nn);
	    int tt1 = trlet.startt;
	    int tt2 = trlet.endt;
	    for(int cam=0; cam<Ncam; ++cam)
	    {
		for(int tt=tt1; tt<=tt2; ++tt)
		{
		    row(final_trj.trj(cam), tt) = row(trlet.trj(cam), tt);
		    final_trj.scores(cam, tt) = trlet.scores(cam, tt);
		}
	    }
	    for(int tt=tt1; tt<=tt2; ++tt)
	    {
		row(final_trj.trj_3d, tt) = row(trlet.trj_3d, tt);
		final_state_list(ii, tt) = 1;
	    }
	}
	//fill gap trlet
	for(int jj=0; jj+1<trj_gindex(ii).size(); ++jj)
	{
	    int nn1 = trj_gindex(ii)[jj];
	    int nn2 = trj_gindex(ii)[jj+1];
	    object_trj_t const& gap_trlet = gap_trlet_list(nn1, nn2);
	    int tt1 = gap_trlet.startt;
	    int tt2 = gap_trlet.endt;
	    for(int cam=0; cam<Ncam; ++cam)
	    {
		for(int tt=tt1; tt<=tt2; ++tt)
		{
		    row(final_trj.trj(cam), tt) = row(gap_trlet.trj(cam), tt);
		    final_trj.scores(cam, tt) = gap_trlet.scores(cam, tt);
		}
	    }
	    for(int tt=tt1; tt<=tt2; ++tt)
	    {
		row(final_trj.trj_3d, tt) = row(gap_trlet.trj_3d, tt);
		final_state_list(ii, tt) = 0;
	    }
	}

    }

}

#endif
