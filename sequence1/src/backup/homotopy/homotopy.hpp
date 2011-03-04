#ifndef __HOMOTOPY__HPP__INCLUDED__
#define __HOMOTOPY__HPP__INCLUDED__


void generate_feature_maps(matrix<int> const& obs, matrix<int> const& dyn_obs,
			   vector<matrix<float> >& feat)
{

    using namespace boost::lambda;
    feat = vector<matrix<float> >(3);
    feat(0) = scalar_matrix<float>(obs.size1(), obs.size2(), 1.0f);
    feat(1) = matrix<float>(obs.size1(), obs.size2());
    feat(2) = matrix<float>(obs.size1(), obs.size2());

    CImg<int> obs_im;
    array2d_copy(obs, obs_im);
    CImg<float> obs_dist = obs_im.get_distance(1);
    
    CImg<int> dyn_obs_im;
    array2d_copy(dyn_obs, dyn_obs_im);
    CImg<float> dyn_obs_dist = dyn_obs_im.get_distance(1);

    float(*fexp)(float) = std::exp;
    array2d_transform(obs_dist, feat(1), bind(fexp, -_1/4.0f));
    array2d_transform(dyn_obs_dist, feat(2), bind(fexp, -_1/4.0f));

}


void get_state_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
		     vector<vector<int> >& sg,
		     matrix<int>& yx2ig,
		     matrix<int>& ig2yx)
{
    using namespace boost::lambda;
    matrix<int> obs_all(obs);//+dyn_obs);
    int ng = std::count(obs_all.data().begin(), obs_all.data().end(), 0);
    sg = vector<vector<int> >(ng);

    yx2ig = scalar_matrix<int>(obs.size1(), obs.size2(), -1);
    ig2yx = scalar_matrix<int>(ng, 2, -1);

    int ig = 0;
    for(int yy=0; yy<obs.size1(); ++yy)
    {
	for(int xx=0; xx<obs.size2(); ++xx)
	{
	    if(obs_all(yy, xx)>0) continue;
	    yx2ig(yy, xx) = ig;
	    ig2yx(ig, 0) = yy;
	    ig2yx(ig, 1) = xx;
	    ++ig;
	}
    }

    vector<std::vector<int> > sgv(ng);
    for(int yy=0; yy<yx2ig.size1(); ++yy)
    {
	for(int xx=0; xx<yx2ig.size2(); ++xx)
	{
	    int ig1 = yx2ig(yy, xx);
	    if(ig1<0) continue;
	    for(int nn=0; nn<8; ++nn)
	    {
		int y2 = yy+nbrhood_t::dy[nn];
		int x2 = xx+nbrhood_t::dx[nn];
		if(y2<0 || y2>=obs.size1()) continue;
		if(x2<0 || x2>=obs.size2()) continue;
		int ig2 = yx2ig(y2, x2);
		if(ig2<0) continue;
		sgv(ig1).push_back(ig2);
	    }
	}
    }

    for(int ii=0; ii<sgv.size(); ++ii)
    {
	array1d_copy(sgv[ii], sg(ii));
    }

}

void get_feature_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
		       vector<matrix<float> > const& feat,
		       vector<vector<int> > const& sg,
		       matrix<int> const& ig2yx,
		       matrix<vector<double> >& fg)
{
    using namespace boost::lambda;

    int ng = sg.size();

    fg = matrix<vector<double> >(feat.size(), ng);
    for(int ff=0; ff<feat.size(); ++ff)
    {
	for(int gg=0; gg<ng; ++gg)
	{
	    fg(ff, gg) = vector<double>(sg(gg).size());
	}
    }

    for(int ff=0; ff<feat.size(); ++ff)
    {
	for(int gg=0; gg<ng; ++gg)
	{

	    int yy = ig2yx(gg, 0);
	    int xx = ig2yx(gg, 1);

	    for(int nn=0; nn<fg(ff, gg).size(); ++nn)
	    {
		int g2 = sg(gg)(nn);
		int y2 = ig2yx(g2, 0);
		int x2 = ig2yx(g2, 1);
		int dx = x2-xx;
		int dy = y2-yy;

		double dist = std::sqrt(static_cast<double>(dx*dx+dy*dy));
		fg(ff, gg)(nn) = (feat(ff)(yy, xx)+feat(ff)(y2, x2))/2*dist;
	    }

	}
    }

}


void get_path_gindex(matrix<int> const& path,
		     matrix<int> const& yx2ig,
		     vector<int>& path_ig)
{
    path_ig = vector<int>(path.size1());
    for(int ii=0; ii<path.size1(); ++ii)
    {
	path_ig(ii) = yx2ig(path(ii, 1), path(ii, 0));
    }
}




void shortest_path(vector<vector<int> >const& sg,
		   vector<vector<double> > const& fdist,
		   int start, int goal,
		   vector<int>& path)
{
    using namespace boost;
    typedef adjacency_list < listS, vecS, directedS,
	no_property, property < edge_weight_t, double > > graph_t;
    typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
    typedef graph_traits < graph_t >::edge_descriptor edge_descriptor;

    int num_nodes = sg.size();
    graph_t g(num_nodes);
    property_map<graph_t, edge_weight_t>::type weightmap
	= get(edge_weight, g);
    for (int ii = 0; ii < num_nodes; ++ii) 
    {
	for(int jj=0; jj < sg(ii).size(); ++jj)
	{
	    edge_descriptor e; bool inserted;
	    tie(e, inserted) = add_edge(ii, sg(ii)(jj), g);
	    weightmap[e] = fdist(ii)(jj);
	}
    }

    std::vector<vertex_descriptor> p(num_vertices(g));
    std::vector<double> d(num_vertices(g));
    vertex_descriptor s = vertex(start, g);

    // VC++ has trouble with the named parameters mechanism
    dijkstra_shortest_paths(g, s,
			    predecessor_map(&p[0]).distance_map(&d[0]));

    std::vector<int> pv;
    for(vertex_descriptor gg = vertex(goal, g); gg!=s; gg=p[gg] )
    {
	pv.push_back(gg);
    }

    path = vector<int>(pv.size());
    std::copy(pv.rbegin(), pv.rend(), path.begin());

#if 0
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi) {
	std::cout << "distance(" << name[*vi] << ") = " << d[*vi] << ", ";
	std::cout << "parent(" << name[*vi] << ") = " << name[p[*vi]] << std::
	    endl;
    }
#endif

}

int pow_int(int base, int n)
{
    int v = 1;
    for(int ii=0; ii<n; ++ii)
    {
	v *= base;
    }
    return v;
}

void wind_angle_planning(vector<vector<int> > const& sg,
			 vector<vector<double> > const & fdist,
			 matrix<int> const& ig2yx,
			 matrix<double> const& obs,
			 int wnum_l,
			 int wnum_u,
			 int start, int goal,
			 vector<vector<int> >& result_path,
			 vector<double>& result_dist,
			 vector<vector<int> >& result_wind_num)
{
    using namespace boost::lambda;
    double pi = boost::math::constants::pi<double>();
    int ng = sg.size();
    int no = obs.size1();

    std::cout<<"ng="<<ng<<std::endl;
    std::cout<<"no="<<no<<std::endl;
    real_timer_t timer1;
    matrix<double> wa(ng, no);
    vector<vector<vector<double> > > dwa(ng);
    for(int gg=0; gg<ng; ++gg)
    {
	double yy = ig2yx(gg, 0);
	double xx = ig2yx(gg, 1);
	for(int oo=0; oo<no; ++oo)
	{
	    wa(gg, oo) = std::atan2(yy-obs(oo, 1), xx-obs(oo, 0));
	}
    }
    std::cout<<"step 1 time: "<<timer1.elapsed()/1000.0f<<std::endl;


    for(int gg=0; gg<ng; ++gg)
    {
	int nnb = sg(gg).size();
	dwa(gg) = vector<vector<double> >(nnb);
	for(int bb=0; bb<nnb; ++bb)
	{
	    int g2 = sg(gg)(bb);

	    dwa(gg)(bb) = vector<double>(no);
	    for(int oo=0; oo<no; ++oo)
	    {
		double da= wa(gg, oo) - wa(g2, oo);
		if(da>pi)
		    da -= 2*pi;
		else if(da<=-pi)
		    da += 2*pi;
		dwa(gg)(bb)(oo) = da;
	    }
	}
    }
    std::cout<<"step 1 time: "<<timer1.elapsed()/1000.0f<<std::endl;

    int nw = wnum_u - wnum_l+1;
    int nlayer = pow_int(nw, no);
    int num_nodes = nlayer*ng;

    int cc=0;

    real_timer_t timer2;
    matrix<double> wind_angle(num_nodes, no); //of augmented graph nodes

    for(int gg=0; gg<ng; ++gg)
    {
	for(int ll=0; ll<nlayer; ++ll)
	{
	    int l2 = ll;
	    for(int oo=0; oo<no; ++oo)
	    {
		int ww = l2%nw;
		wind_angle(cc, oo) = (wnum_l+ww)*pi*2 + wa(gg, oo);
		l2 /= nw;
	    }
	    ++cc;
	}
    }
    std::cout<<"step 2 time: "<<timer2.elapsed()/1000.0f<<std::endl;

    //std::cout<<"wind_angle="<<std::endl;
    //array2d_print(std::cout, wind_angle);

    using namespace boost;
    typedef adjacency_list < listS, vecS, directedS,
	no_property, property < edge_weight_t, double > > graph_t;
    typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
    typedef graph_traits < graph_t >::edge_descriptor edge_descriptor;

    real_timer_t timer3;
    graph_t g(num_nodes);
    property_map<graph_t, edge_weight_t>::type weightmap
	= get(edge_weight, g);
    //weightmap.reserve(8*nlayer);
    std::cout<<"num_nodes="<<num_nodes<<std::endl;
    for(int gg = 0; gg < ng; ++gg)
    {
	for(int ll=0; ll<nlayer; ++ll)
	{
	    int ii = gg*nlayer+ll;
	    for(int nn=0; nn < sg(gg).size(); ++nn)
	    {
		int g2 = sg(gg)(nn);
		for(int l2 = 0; l2<nlayer; ++l2)
		{
		    int jj = g2*nlayer+l2;
		    vector<double> diff = row(wind_angle, ii)
			-row(wind_angle, jj);
		    vector<double> ddf = diff-dwa(gg)(nn);
		    double absdiff = norm_1(ddf);
		    if(absdiff<1e-6)
		    {
			edge_descriptor e; bool inserted;
			tie(e, inserted) = add_edge(ii, jj, g);
			weightmap[e] = fdist(gg)(nn);
			//std::cout<<"connect "<<gg<<", "<<ll<<"\t"
			//	 <<g2<<", "<<l2<<std::endl;
		    }
		}
	    }
	    //std::cout<<"-----------------------"<<std::endl;
	}
    }

    std::cout<<"graph construction timer:"
	     <<timer3.elapsed()/1000.0f<<std::endl;

    std::cout<<"num_edges="<<num_edges(g)<<std::endl;

    //winding number analysis of start-goal pairs
    matrix<vector<int> > wind_num(nlayer, nlayer);
    for(int ls=0; ls<nlayer; ++ls)
    {
	for(int lg=0; lg<nlayer; ++lg)
	{
	    vector<double> diff = row(wind_angle, goal*nlayer+lg)
		-row(wind_angle, start*nlayer+ls);
	    double (*ffloor)(double) = std::floor;
	    array1d_transform(diff, wind_num(ls, lg), ll_static_cast<int>( 
				  bind(ffloor, _1/pi/2.0l)));
	}
    }


    std::vector<vector<int> > selected_wind_num;
    std::vector<vector<int> > selected_path;
    std::vector<double> selected_dist;
    //for(int ls=0; ls<nlayer; ++ls)
    for(int ls=nlayer/4; ls<nlayer-nlayer/4; ++ls)
    {
	std::vector<vertex_descriptor> p(num_vertices(g));
	std::vector<double> d(num_vertices(g));

	vertex_descriptor s = vertex(start*nlayer+ls, g);

	// VC++ has trouble with the named parameters mechanism
	dijkstra_shortest_paths(g, s,
				predecessor_map(&p[0]).distance_map(&d[0]));

	for(int lg=0; lg<nlayer; ++lg)
	{
	    std::vector<int> pv;
	    double dist = d[goal*nlayer+lg];
	    for(vertex_descriptor gg = vertex(goal*nlayer+lg, g); gg!=s; gg=p[gg] )
	    {
		pv.push_back(gg);
	    }
	    pv.push_back(s);
	    std::for_each(pv.begin(), pv.end(), _1/=nlayer);
	    vector<int> count=scalar_vector<int>(sg.size(), 0);
	    bool looped = false;
	    for(int ii=0; ii<pv.size(); ++ii)
	    {
		if(count[pv[ii]]) 
		{
		    looped = true;
		    break;
		}
		count[pv[ii]]++;
	    }
	    if(looped) continue; //remove looped ones

	    vector<int> path;
	    path = vector<int>(pv.size());
	    std::copy(pv.rbegin(), pv.rend(), path.begin());
	    selected_wind_num.push_back(wind_num(ls, lg));
	    selected_dist.push_back(dist);
	    selected_path.push_back(path);
	    
	}
    }

    std::vector<int> uniq;
    for(int ii=0; ii<selected_path.size(); ++ii)
    {
	bool same = false;
	for(int rr=0; rr<uniq.size(); ++rr)
	{
	    int ir = uniq[rr];
	    if(std::abs(selected_dist[ir]-selected_dist[ii])> 1e-6l)
		continue;
	    int dp = norm_1(selected_path[ir]-selected_path[ii]);
	    if(0==dp)
	    {
		same = true;
		break;
	    }

	}
	if(!same) uniq.push_back(ii);
    }

    result_dist = vector<double>(uniq.size());
    for(int rr=0; rr<uniq.size(); ++rr)
    {
	int ii = uniq[rr];
	result_dist(rr) = selected_dist[ii];
    }

    vector<std::pair<int, double> > tmp(uniq.size());
    std::transform(uniq.begin(), uniq.end(), result_dist.begin(),
		   tmp.begin(), bind(std::make_pair<int, double>, _1, _2));

    std::sort(tmp.begin(), tmp.end(),
	      bind(std::less<double>(),
		   bind(&std::pair<int, double>::second, _1),
		   bind(&std::pair<int, double>::second, _2) ) );

    result_wind_num = vector<vector<int> >(tmp.size());
    result_path = vector<vector<int> >(tmp.size());
    for(int rr=0; rr<tmp.size(); ++rr)
    {
	int ii = tmp(rr).first;
	result_wind_num(rr) = selected_wind_num[ii];
	result_path(rr) = selected_path[ii];
	result_dist(rr) = tmp(rr).second;
    }

#if 0
    for(int ii=0; ii<result_path.size(); ++ii)
    {
	std::cout<<"winding number="<<result_wind_num[ii]<<std::endl;
	std::cout<<"distance="<<result_dist[ii]<<std::endl;
	std::cout<<"path="<<result_path[ii]<<std::endl;
    }
#endif
    std::cout<<"step 3 time: "<<timer3.elapsed()/1000.0f<<std::endl;

}

#endif
