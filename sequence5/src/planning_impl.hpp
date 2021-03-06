#ifndef __PLANNING__IMPL__HPP__INCLUDED__
#define __PLANNING__IMPL__HPP__INCLUDED__

#include <boost/math/special_functions/fpclassify.hpp>

void carboxes2carobs(matrix<float> const& cars, 
		     matrix<double> const& img2grd,
		     vector<matrix<float> > & car_poly)
{

    float pi = boost::math::constants::pi<float>();

    int num_cars = cars.size1();
    car_poly = vector<matrix<float> >(num_cars);

    for(int nn=0; nn<num_cars; ++nn)
    {
	matrix_row<matrix<float> const> bb(cars, nn);
	float x1 = bb(0);
	float x2 = bb(2);
	float y2 = bb(3);

	float xc = (x1+x2)/2.0f;
	x1 = (x1-xc)*1.2f+xc;
	x2 = (x2-xc)*1.2f+xc;

	vector<double> img_x(2), img_y(2);
	img_x <<= x1, x2;
	img_y <<= y2, y2;

	vector<double> grd_x(2), grd_y(2);
	apply_homography(img2grd, img_x, img_y, grd_x, grd_y);

	vector<float> p1(2), p4(2);
	p1 <<= grd_x(0), grd_y(0);
	p4 <<= grd_x(1), grd_y(1);

	vector<float> dp = p4-p1;
	float theta = std::atan2(dp(1), dp(0))+pi/2.0f;
	float thick = 2.0f*1.2f;
	thick = thick*20.0f;

	vector<float> p2 = p1;
	p2(0) += cos(theta)*thick;
	p2(1) += sin(theta)*thick;
	vector<float> p3 = p4;
	p3(0) += cos(theta)*thick;
	p3(1) += sin(theta)*thick;

	car_poly(nn) = matrix<float>(4, 2);

	row(car_poly(nn), 0) = p1;
	row(car_poly(nn), 1) = p2;
	row(car_poly(nn), 2) = p3;
	row(car_poly(nn), 3) = p4;
    }
}		     

void load_carboxes(directory_structure_t& ds,
		   vector<std::vector<std::string> > const& seq,
		   matrix<matrix<float> >& cars)
{
    int Ncam = seq.size();
    int T = seq[0].size();
    cars = matrix<matrix<float> >(Ncam, T);
    for(int tt=0; tt<T; ++tt)
    {
	for(int cam=0; cam<Ncam; ++cam)
	{
	    fs::path seq_path(seq[cam][tt]);
	    std::string image_name = fs::basename(seq_path);

	    fs::path car_path = fs::path(ds.workspace)
		/"car_detection"/(image_name+".txt");
	    //fs::path ped_path = fs::path(ds.workspace)/"detection"/(image_name+"_3d_ped.txt");

	    matrix<float> car_boxes;
	    read_text_matrix(car_path.string(), car_boxes);

	    std::vector<int> idx;
	    for(int cc=0; cc<car_boxes.size1(); ++cc)
	    {
		if(car_boxes(cc, 4)>=0)
		    idx.push_back(cc);
	    }

	    cars(cam, tt) = matrix<float>(idx.size(), 5);
	    for(int ii=0; ii<idx.size(); ++ii)
	    {
		int cc = idx[ii];
		matrix_row<matrix<float> > rc(car_boxes, cc);
		row(cars(cam, tt), ii) = project(rc, range(0, 5));
						 
	    }
	    //std::cout<<car_boxes<<std::endl;
      
	    //matrix<float> ped_boxes;
	    //read_text_matrix(ped_path.string(), ped_boxes);

	}
    }

}

void prepare_car_obs(matrix<matrix<float> > const& cars,
		     matrix<matrix<double> > const& img2grd,
		     matrix<vector<matrix<float> > >& car_poly)
{
    int Ncam = cars.size1();
    int T = cars.size2();
    car_poly = matrix<vector<matrix<float> > >(Ncam, T);
    for(int cam=0; cam<Ncam; ++cam)
    {
	for(int tt=0; tt<T; ++tt)
	{
	    carboxes2carobs(cars(cam, tt), img2grd(tt, cam), car_poly(cam, tt));
	}
    }
}

void prepare_ped_obs(vector<object_trj_t> const& good_trlet_list,
		     int T,
		     matrix<matrix<float> >& ped_obs)
{

    vector<object_trj_t> const& trlet_list = good_trlet_list;
    int num_obj = trlet_list.size();

    ped_obs = matrix<matrix<float> >(num_obj, T);

    for(int nn=0; nn<trlet_list.size(); ++nn)
    {

	for(int tt=trlet_list(nn).startt;
	    tt<=trlet_list(nn).endt; ++tt)
	{
	    //p0 = trlet_list(nn).trj3d(tt, :)';'
	    matrix_row<matrix<float> const> p0(trlet_list(nn).trj_3d, tt);
	    float thick = 0.25f*1.2f*100/5.0f;

	    ped_obs(nn, tt) = matrix<float>(4, 2);
	    ped_obs(nn, tt) <<=
		p0(0)-thick, p0(1)-thick,
		p0(0)+thick, p0(1)-thick,
		p0(0)+thick, p0(1)+thick,
		p0(0)-thick, p0(1)+thick;
	}
    }

}


    
void combine_car_obs(matrix<vector<matrix<float> > > const& car_obsz,
		     vector<vector<matrix<float> > > & car_obs)
{
    int Ncam = car_obsz.size1();
    int T = car_obsz.size2();

    car_obs = row(car_obsz, 1); //use right camera

}



void plan_trlet_list(geometric_info_t const& gi,
		     matrix<int> const& Tff,
		     vector<object_trj_t> const& good_trlet_list,
		     vector<vector<matrix<float> > > const& car_obs, 
		     matrix<matrix<float> > const& ped_obs,
		     int plan_advance,
		     vector<int>& plan_time,
		     vector<vector<planning_result_item_t> >& results,
		     directory_structure_t &ds)
{
    std::cout<<"planning ..."<<std::endl;;
    vector<object_trj_t> const& trlet_list = good_trlet_list;

    int N = good_trlet_list.size();
    //std::cout<<"ng = "<<ng<<std::endl;

    
    results = vector<vector<planning_result_item_t> >(N);
    plan_time = vector<int>(N);

    int counter = 0;
    for(int ii=0; ii<N; ++ii)
    {
	int st = sum(row(Tff, ii));
	if(st<=0) continue;

	int endt = trlet_list(ii).endt;
	int startt = trlet_list(ii).startt;
	int plant = endt - plan_advance;
	if(plant<startt)        plant = startt;
    
	plan_time(ii) = plant;

	float start_x = trlet_list(ii).trj_3d(plant, 0);
	float start_y = trlet_list(ii).trj_3d(plant, 1);
	vector<matrix<float> > ped_obs_one = column(ped_obs, plant);

	do_homotopy_planning(gi.ground_lim, gi.poly_ground,
			     ii, start_x, start_y,
			     gi.goal_ground,
			     car_obs(plant), ped_obs_one,
			     results(ii), ds, plant);

	std::cout<<"-------------------------finished "<<ii<<"/"<<N<<::std::endl;
	for(int kk=0; kk<results(ii).size(); ++kk)
	{
	    for(int ll=0; ll<results(ii)(kk).path.size(); ++ll)
	    {
		counter += results(ii)(kk).path(ll).size1();
	    }
	}
	std::cout<<"\t\tcounter="<<counter<<std::endl;
    }

}


void plan_trlet_list(mpi::communicator& world,
		     geometric_info_t const& gi,
		     matrix<int> const& Tff,
		     vector<object_trj_t> const& good_trlet_list,
		     vector<vector<matrix<float> > > const& car_obs, 
		     matrix<matrix<float> > const& ped_obs,
		     int plan_advance,
		     vector<int>& plan_time,
		     vector<vector<planning_result_item_t> >& results,
		     directory_structure_t &ds)
{
    std::cout<<"planning ..."<<std::endl;;
    vector<object_trj_t> const& trlet_list = good_trlet_list;

    int N = good_trlet_list.size();
    //std::cout<<"ng = "<<ng<<std::endl;

    
    results = vector<vector<planning_result_item_t> >(N);
    plan_time = vector<int>(N);

    //int counter = 0;
    for(int ii=world.rank(); ii<N; ii+=world.size())
    {
	int st = sum(row(Tff, ii));
	if(st<=0) continue;

	//if(ii<15) continue;

	int endt = trlet_list(ii).endt;
	int startt = trlet_list(ii).startt;
	int plant = endt - plan_advance;
	if(plant<startt)        plant = startt;
    
	plan_time(ii) = plant;

	float start_x = trlet_list(ii).trj_3d(plant, 0);
	float start_y = trlet_list(ii).trj_3d(plant, 1);
	vector<matrix<float> > ped_obs_one = column(ped_obs, plant);

	do_homotopy_planning(gi.ground_lim, gi.poly_ground,
			     ii, start_x, start_y,
			     gi.goal_ground,
			     car_obs(plant), ped_obs_one,
			     results(ii), ds, plant);

	std::cout<<"-------------------------finished "<<ii<<"/"<<N<<::std::endl;
#if 0
	for(int kk=0; kk<results(ii).size(); ++kk)
	{
	    for(int ll=0; ll<results(ii)(kk).path.size(); ++ll)
	    {
		counter += results(ii)(kk).path(ll).size();
	    }
	}
	std::cout<<"\t\tcounter="<<counter<<std::endl;
#endif
    }

    if(world.rank()==0)
    {
	std::vector<vector<vector<planning_result_item_t> > > v1;
	std::vector<vector<int> > v2;
	mpi::gather(world, results, v1, 0);
	mpi::gather(world, plan_time, v2, 0);

	for(int ii=0; ii<results.size(); ++ii)
	{
	    if(results(ii).size()!=0) continue;

	    for(int kk=0; kk<v1.size(); ++kk)
	    {
		if(v1[kk](ii).size())
		{
		    results(ii) = v1[kk](ii);
		    plan_time(ii) = v2[kk](ii);
		    break;
		}
	    }

	}
    }
    else
    {
	mpi::gather(world, results, 0);
	mpi::gather(world, plan_time, 0);
    }



    mpi::broadcast(world, results, 0);
    mpi::broadcast(world, plan_time, 0);


}



void construct_state_graph( matrix<int> const& obs,
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


void construct_feature_maps(matrix<int> const& obs, matrix<int> const& dyn_obs,
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

    //obs_dist.save_png("../figures/dist.png");

    float(*fexp)(float) = std::exp;
    array2d_transform(obs_dist, feat(1), bind(fexp, -_1/1.0f));
    array2d_transform(dyn_obs_dist, feat(2), bind(fexp, -_1/4.0f));

}

void construct_feature_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
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



float shortest_path(vector<vector<int> >const& sg,
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

    return d[vertex(goal, g)];
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

bool is_looped(std::vector<int> const& pv,
	       vector<vector<int> > const& sg,
	       matrix<int> const& ig2yx)
{
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

    if(looped)   return looped;

    std::set<std::pair<int, int> > cent2;

    for(int ii=0; ii+1<pv.size(); ++ii)
    {
	int kk = pv[ii];
	int k2 = pv[ii+1];
	std::pair<int, int> c(ig2yx(kk, 0)+ig2yx(k2, 0), ig2yx(kk, 1)+ig2yx(k2, 1));
	std::pair<std::set<std::pair<int, int> >::iterator, bool > ret = cent2.insert(c);
	if(!ret.second) return true;
    }
    return false;

}

template <typename Float2>
bool enclose_obstacle(matrix<Float2> const& path1,
		      matrix<Float2> const& path2,
		      matrix<Float2> const& obs)
{
    matrix<Float2> m(path1.size1()+path2.size1()-2, 2);
    project(m, range(0, path1.size1()), range::all()) = path1;
    for(int rr=1; rr+1<path2.size1(); ++rr)
    {
	row(m, m.size1()-rr) = row(path2, rr);
    }
    for(int oo=0; oo<obs.size1(); ++oo)
    {
	matrix_column<matrix<Float2> > mx(m, 0);
	matrix_column<matrix<Float2> > my(m, 1);
	if(point_in_polygon(mx, my, obs(oo, 0), obs(oo, 1)))
	    return true;
    }
    return false;
}


template <typename Float1, typename Float2>
void check_redundancy(std::vector<vector<int> >const& selected_path,
		      std::vector<Float1>const & selected_dist,
		      matrix<int> const& ig2yx,
		      matrix<Float2> const& obs,
		      std::vector<int>& useful)
{
    using namespace boost::lambda;
    int N = selected_dist.size();
    vector<std::pair<int, Float1> > tmp(N);
    std::transform(counting_iterator<int>(0),
		   counting_iterator<int>(N),
		   selected_dist.begin(),
		   tmp.begin(), bind(std::make_pair<int, Float1>, _1, _2));

    std::sort(tmp.begin(), tmp.end(),
	      bind(std::less<Float1>(),
		   bind(&std::pair<int, Float1>::second, _1),
		   bind(&std::pair<int, Float1>::second, _2) ) );

    vector<matrix<Float2> > path(N);
    for(int pp=0; pp<N; ++pp)
    {
	path(pp) = matrix<Float2>(selected_path[pp].size(), 2);
	for(int ss=0; ss<selected_path[pp].size(); ++ss)
	{
	    int kk = selected_path[pp](ss);
	    path(pp)(ss, 0) = ig2yx(kk, 1);
	    path(pp)(ss, 1) = ig2yx(kk, 0);
	}
    }


    for(int pp=0; pp<N; ++pp)
    {
	bool good = true;
	int p1 = tmp(pp).first;
	for(int uu=0; uu<useful.size(); ++uu)
	{
	    int p2 = useful[uu];
	    if(!enclose_obstacle(path(p1), path(p2), obs))
	    {
		good = false;
		break;
	    }

	}
	if(good) useful.push_back(p1);
    }

}


template <typename Float>
void wind_angle_planning(vector<vector<int> > const& sg,
			 vector<vector<Float> > const & fdist,
			 matrix<int> const& ig2yx,
			 matrix<Float> const& obs,
			 int wnum_l,
			 int wnum_u,
			 int start, int goal,
			 vector<vector<int> >& result_path,
			 vector<Float>& result_dist,
			 matrix<int>& result_wind_num)
{
    using namespace boost::lambda;
    Float pi = boost::math::constants::pi<Float>();
    int ng = sg.size();
    int no = obs.size1();

    std::cout<<"ng="<<ng<<"\t";
    std::cout<<"no="<<no<<std::endl;
    real_timer_t timer1;
    matrix<Float> wa(ng, no);
    vector<vector<vector<Float> > > dwa(ng);
    for(int gg=0; gg<ng; ++gg)
    {
	double yy = ig2yx(gg, 0);
	double xx = ig2yx(gg, 1);
	for(int oo=0; oo<no; ++oo)
	{
	    wa(gg, oo) = std::atan2(yy-obs(oo, 1), xx-obs(oo, 0));
	}
    }

    for(int gg=0; gg<ng; ++gg)
    {
	int nnb = sg(gg).size();
	dwa(gg) = vector<vector<Float> >(nnb);
	for(int bb=0; bb<nnb; ++bb)
	{
	    int g2 = sg(gg)(bb);

	    dwa(gg)(bb) = vector<Float>(no);
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
    matrix<Float> wind_angle(num_nodes, no); //of augmented graph nodes

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
	no_property, property < edge_weight_t, Float > > graph_t;
    typedef typename graph_traits < graph_t >::vertex_descriptor
	vertex_descriptor;
    typedef typename graph_traits < graph_t >::edge_descriptor
	edge_descriptor;

    real_timer_t timer3;

#if 1
    graph_t g(num_nodes);
    typename property_map<graph_t, edge_weight_t>::type weightmap
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

#else
    typedef std::pair<int, int> edge_t;
    std::vector<edge_t> edge_vec;
    std::vector<Float> weight_vec;
    int nnz = std::accumulate(sg.begin(), sg.end(), 0, _1+bind(&vector<int>::size, _2));
    std::cout<<"nnz="<<nnz<<std::endl;
    edge_vec.reserve(nnz*nlayer);
    weight_vec.reserve(nnz*nlayer);

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
			edge_vec.push_back(edge_t(ii, jj));
			weight_vec.push_back(fdist(gg)(nn));
			//std::cout<<"connect "<<gg<<", "<<ll<<"\t"
			//	 <<g2<<", "<<l2<<std::endl;
		    }
		}
	    }
	    //std::cout<<"-----------------------"<<std::endl;
	}
    }

    graph_t g(edge_vec.begin(), edge_vec.end(), weight_vec.begin(), num_nodes);
    typename property_map<graph_t, edge_weight_t>::type weightmap
	= get(edge_weight, g);
    //weightmap.reserve(8*nlayer);
    std::cout<<"num_nodes="<<num_nodes<<std::endl;

#endif

    std::cout<<"graph construction timer:"
	     <<timer3.elapsed()/1000.0f<<std::endl;

    std::cout<<"num_edges="<<num_edges(g)<<std::endl;

    //winding number analysis of start-goal pairs
    matrix<vector<int> > wind_num(nlayer, nlayer);
    for(int ls=0; ls<nlayer; ++ls)
    {
	for(int lg=0; lg<nlayer; ++lg)
	{
	    vector<Float> diff = row(wind_angle, goal*nlayer+lg)
		-row(wind_angle, start*nlayer+ls);
	    Float (*ffloor)(Float) = std::floor;
	    array1d_transform(diff, wind_num(ls, lg), ll_static_cast<int>( 
				  bind(ffloor, _1/pi/2.0)));
	}
    }


    std::vector<vector<int> > selected_wind_num;
    std::vector<vector<int> > selected_path;
    std::vector<Float> selected_dist;
    for(int ls=0; ls<nlayer; ++ls)
    //for(int ls=nlayer/4; ls<nlayer-nlayer/4; ++ls)
    {
	std::vector<vertex_descriptor> p(num_vertices(g));
	std::vector<double> d(num_vertices(g));

	vertex_descriptor s = vertex(start*nlayer+ls, g);

	// VC++ has trouble with the named parameters mechanism
	//std::cout<<"begin dijkstra..."<<std::endl;
	dijkstra_shortest_paths(g, s,
				predecessor_map(&p[0]).distance_map(&d[0]));

	//std::cout<<"end dijkstra..."<<std::endl;
	for(int lg=0; lg<nlayer; ++lg)
	{

	    std::vector<int> pv;
	    Float dist = d[goal*nlayer+lg];
	    //std::cout<<"dist="<<dist<<std::endl;

	    if(!boost::math::isfinite<Float>(dist)) continue;

	    for(vertex_descriptor gg = vertex(goal*nlayer+lg, g);
		gg!=s; gg=p[gg] )
	    {
		pv.push_back(gg);
		//if(pv.size()>3000) break;
	    }
	    pv.push_back(s);
	    std::for_each(pv.begin(), pv.end(), _1/=nlayer);
	    if(is_looped(pv, sg, ig2yx)) continue;

	    vector<int> path;
	    path = vector<int>(pv.size());
	    std::copy(pv.rbegin(), pv.rend(), path.begin());
	    selected_wind_num.push_back(wind_num(ls, lg));
	    selected_dist.push_back(dist);
	    selected_path.push_back(path);
	    
	}
	//std::cout<<"end loop check"<<std::endl;
    }

#if 0
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

    result_dist = vector<Float>(uniq.size());
    for(int rr=0; rr<uniq.size(); ++rr)
    {
	int ii = uniq[rr];
	result_dist(rr) = selected_dist[ii];
    }

    vector<std::pair<int, Float> > tmp(uniq.size());
    std::transform(uniq.begin(), uniq.end(), result_dist.begin(),
		   tmp.begin(), bind(std::make_pair<int, Float>, _1, _2));

    std::sort(tmp.begin(), tmp.end(),
	      bind(std::less<Float>(),
		   bind(&std::pair<int, Float>::second, _1),
		   bind(&std::pair<int, Float>::second, _2) ) );

    result_wind_num = matrix<int>(tmp.size(), no);
    result_path = vector<vector<int> >(tmp.size());
    for(int rr=0; rr<tmp.size(); ++rr)
    {
	int ii = tmp(rr).first;
	row(result_wind_num, rr) = selected_wind_num[ii];
	result_path(rr) = selected_path[ii];
	result_dist(rr) = tmp(rr).second;
    }
#endif

    std::vector<int> useful;

    check_redundancy(selected_path, selected_dist, ig2yx, obs, useful);

    result_dist = vector<Float>(useful.size());
    result_wind_num = matrix<int>(useful.size(), no);
    result_path = vector<vector<int> >(useful.size());
    for(int rr=0; rr<useful.size(); ++rr)
    {
	int ii = useful[rr];
	row(result_wind_num, rr) = selected_wind_num[ii];
	result_path(rr) = selected_path[ii];
	result_dist(rr) = selected_dist[ii];
    }


    std::cout<<"step 3 time: "<<timer3.elapsed()/1000.0f<<std::endl;

}


void combine_obstacles(int nn,
		       vector<matrix<float> > const& car_obs, 
		       vector<matrix<float> > const& ped_obs,
		       matrix<double> const & poly_ground,
		       vector<matrix<float> >& obs,
		       matrix<float>& obs_cent)
{
    obs = vector<matrix<float> >(car_obs.size()+ped_obs.size()-1);
    project(obs, range(0, car_obs.size())) = car_obs;
    int oo=car_obs.size();
    for(int pp=0; pp<ped_obs.size(); ++pp)
    {
	if(nn==pp) continue;
	obs(oo) = ped_obs(pp);
	++oo;
    }

    std::vector<int> odx;
    for(int ii=0; ii<obs.size(); ++ii)
    {
	if(obs(ii).size1())
	{
	    odx.push_back(ii);
	}
    }
    //fix_obstalces(obs, obs_cent);
    typedef adjacency_list <vecS, vecS, undirectedS> graph_t;

    graph_t g(odx.size());
    for(int ii=0; ii<odx.size(); ++ii)
    {
	int o1 = odx[ii];
	matrix_column<matrix<float> const> plx(obs(o1), 0);
	matrix_column<matrix<float> const> ply(obs(o1), 1);
	for(int jj=ii+1; jj<odx.size(); ++jj)
	{
	    int o2 = odx[jj];
	    matrix<float> const& poly2=obs(o2);
	    for(int pp=0; pp<poly2.size1(); ++pp)
	    {
		if(point_in_polygon(plx, ply, poly2(pp, 0), poly2(pp, 1)))
		{
		    add_edge(ii, jj, g);
		}
	    }
	}
    }
    
    std::vector<int> component(num_vertices(g));
    int num = connected_components(g, &component[0]);

    vector<std::vector<int> > group(num);

    for(int ii=0; ii<component.size(); ++ii)
    {
	group(component[ii]).push_back(odx[ii]);
    }

    vector<int> attach_bg=scalar_vector<int>(num, 0);
    matrix_row<matrix<double> const> gplx(poly_ground, 0);
    matrix_row<matrix<double> const> gply(poly_ground, 1);
    for(int ii=0; ii<group.size(); ++ii)
    {
	for(int jj=0; jj<group(ii).size(); ++jj)
	{
	    int oo = group(ii)[jj];
	    matrix<float> const& pl=obs(oo);
	    for(int pp=0; pp<pl.size1(); ++pp)
	    {
		if(!point_in_polygon(gplx, gply,
				     double(pl(pp, 0)),
				     double(pl(pp, 1))))
		{
		    attach_bg(ii) = 1;
		    break;
		}
	    }
	    if( attach_bg(ii) == 1 )  break;
	}
    }

    obs_cent = matrix<float>(group.size()-sum(attach_bg), 2);
    int cc = 0;
    for(int oo=0; oo<group.size(); ++oo)
    {
	if(attach_bg(oo)) continue;
	int ii = group(oo)[0];
	matrix_column<matrix<float> const> rx(obs(ii), 0);
	matrix_column<matrix<float> const> ry(obs(ii), 1);
	obs_cent(cc, 0) = sum(rx)/rx.size();
	obs_cent(cc, 1) = sum(ry)/ry.size();
	cc++;
    }

}

void fix_poly_ground(matrix<double> const& poly_ground2,
		     matrix<double> & poly_ground)
{
    double dloopx = poly_ground2(0, 0)
	-poly_ground2(0, poly_ground2.size2()-1);
    double dloopy = poly_ground2(1, 0)
	-poly_ground2(1, poly_ground2.size2()-1);
    if(dloopx*dloopx+dloopy+dloopy<0.5l)
    {
	poly_ground = project(poly_ground2, range::all(),
			      range(0, poly_ground2.size2()-1));
    }
    else poly_ground = poly_ground2;
    for(int ii=0; ii<poly_ground.size1(); ++ii)
    {
	for(int jj=0; jj<poly_ground.size2(); ++jj)
	{
	    if(poly_ground(ii, jj)<0)
		poly_ground(ii, jj) = 0;
	}
    }

}


void construct_obstacle_maps(vector<matrix<float> > const& obs,
			     matrix<double> const& poly_ground, 
			     matrix<double> const& goal_ground,
			     matrix<int>& obs_map,
			     matrix<int>& dyn_obs_map)
{
    matrix_row<matrix<double> const> poly_x(poly_ground, 0);
    matrix_row<matrix<double> const> poly_y(poly_ground, 1);

    matrix_row<matrix<double> const> goal_x(goal_ground, 0);
    matrix_row<matrix<double> const> goal_y(goal_ground, 1);
    int s1 = static_cast<int>(
	std::max(*std::max_element(poly_y.begin(), poly_y.end()),
		 *std::max_element(goal_y.begin(), goal_y.end()))
	+1.5l );
    int s2 = static_cast<int>(
	std::max(*std::max_element(poly_x.begin(), poly_x.end()),
		 *std::max_element(goal_x.begin(), goal_x.end()))
	+1.5l );
    //

    //vector<double> poly_x2 = poly_x;
    //vector<double> poly_y2 = poly_y;
    //mask_from_polygon(obs_map, s1, s2, poly_x, poly_y);
    obs_map=scalar_matrix<int>(s1, s2, 0);
    dyn_obs_map=scalar_matrix<int>(s1, s2, 0);


    for(int yy=0; yy<s1; ++yy)
    {
	for(int xx=0; xx<s2; ++xx)
	{
	    for(int oo=0; oo<obs.size(); ++oo)
	    {
		if(obs(oo).size1()==0) continue;
		matrix_column<matrix<float>const > px(obs(oo), 0);
		matrix_column<matrix<float>const > py(obs(oo), 1);
		if(point_in_polygon(px, py, float(xx), float(yy) ))
		{
		    dyn_obs_map(yy, xx) = 1;
		    break;
		}
	    }
	}
    }

    for(int yy=0; yy<s1; ++yy)
    {
	for(int xx=0; xx<s2; ++xx)
	{
	    if(point_in_polygon(poly_x, poly_y, double(xx), double(yy) ))
	    {
		obs_map(yy, xx) = 0;
	    }
	    else
		obs_map(yy, xx) = 1;
		    
	}
    }

}


void vis_obstacle_maps(matrix<double> const& poly_ground,
		       matrix<double> const& goal_ground,
		       int nn, float start_x, float start_y,
		       matrix<int> const& obs_map,
		       matrix<int> const& dyn_obs_map,
		       directory_structure_t &ds, int tt)
{
    using namespace boost::lambda;
    CImg<unsigned char> obs_im;
    array2d_transform(obs_map, obs_im,
		      ll_static_cast<unsigned char>(_1*255));
    CImg<unsigned char> dyn_im;
    array2d_transform(dyn_obs_map, dyn_im,
		      ll_static_cast<unsigned char>(_1*255));

    matrix<int> ipoly;
    matrix<int> igoal;
    array2d_transform(poly_ground, ipoly, ll_static_cast<int>(_1+0.5l));
    array2d_transform(goal_ground, igoal, ll_static_cast<int>(_1+0.5l));
    {
	obs_im.resize(-100, -100, -100, -300);
	unsigned char lcol[3] = {255, 0, 0};
	for(int pp=0; pp+1<ipoly.size2(); ++pp)
	{
	    obs_im.draw_line(ipoly(0, pp), ipoly(1, pp),
			     ipoly(0, pp+1), ipoly(1, pp+1),
			     lcol, 3);
	}
	unsigned char gcol[3] = {0, 255, 0};
	for(int g=0; g<igoal.size2(); ++g)
	{
	    obs_im.draw_circle(igoal(0, g), igoal(1, g),
			       3, gcol);
	}
	obs_im.draw_circle(static_cast<int>(start_x+0.5f),
			   static_cast<int>(start_y+0.5f), 4, lcol);
	std::string name = ds.figures
	    +str(format("obs_im_%03d_%03d.png")%tt%nn);
	obs_im.save_png(name.c_str());
    }

    {
	std::string name = ds.figures
	    +str(format("dyn_im_%03d_%03d.png")%tt%nn);
	dyn_im.save_png(name.c_str());
    }

}


void compute_feat_dist(vector<vector<int> > const& sg,
		       matrix<vector<double> > const& fg,
		       vector<float> const& wei,
		       vector<vector<float> >& fdist)
{
    fdist = vector<vector<float> >(sg.size());
    for(int xx=0; xx<sg.size(); ++xx)
    {
	fdist(xx) = scalar_vector<float>(sg(xx).size(), 0.0l);
	for(int nn=0; nn<sg(xx).size(); ++nn)
	{
	    for(int ff=0; ff<fg.size1(); ++ff)
	    {
		fdist(xx)(nn) += fg(ff, xx)(nn)*wei(ff);
	    }
	}
    }

}

void vis_planned_path(matrix<double> const& poly_ground,
		      matrix<double> const& goal_ground,
		      int nn, float start_x, float start_y,
		      vector<matrix<float> >const& obs,
		      matrix<int> const& obs_map,
		      matrix<int> const& ig2yx,
		      vector<vector<int> > const& path,
		      directory_structure_t &ds,
		      int tt, int gid)
{
   using namespace boost::lambda;
    CImg<unsigned char> obs_im;
    array2d_transform(obs_map, obs_im,
		      ll_static_cast<unsigned char>(_1*255));

    matrix<int> ipoly;
    matrix<int> igoal;

    array2d_transform(poly_ground, ipoly, ll_static_cast<int>(_1+0.5l));
    array2d_transform(goal_ground, igoal, ll_static_cast<int>(_1+0.5l));

    obs_im.resize(-100, -100, -100, -300);
    unsigned char lcol[3] = {255, 0, 0};
    for(int pp=0; pp+1<ipoly.size2(); ++pp)
    {
	obs_im.draw_line(ipoly(0, pp), ipoly(1, pp),
			 ipoly(0, pp+1), ipoly(1, pp+1),
			 lcol, 3);
    }
    unsigned char gcol[3] = {0, 255, 0};
    for(int g=0; g<igoal.size2(); ++g)
    {
	obs_im.draw_circle(igoal(0, g), igoal(1, g),
			   3, gcol);
    }
    obs_im.draw_circle(static_cast<int>(start_x+0.5f),
		       static_cast<int>(start_y+0.5f), 4, lcol);

    for(int oo=0; oo<obs.size(); ++oo)
    {
	if(obs(oo).size1()==0) continue;
	matrix<int> iob;
	array2d_transform(obs(oo), iob, ll_static_cast<int>(_1+0.5l));
	int sz = iob.size1();
	for(int ss=0; ss<iob.size1(); ++ss)
	{
	    obs_im.draw_line(iob(ss, 0), iob(ss, 1),
			     iob((ss+1)%sz, 0), iob((ss+1)%sz, 1),
			     lcol, 3);
	}

    }

    unsigned char pcol[3] = {0, 0, 255};
    for(int pp=0; pp<path.size(); ++pp)
    {
	CImg<unsigned char> pim=obs_im;

	for(int ss=0; ss+1<path(pp).size(); ++ss)
	{
	    int ig = path(pp)(ss);
	    int ig2 = path(pp)(ss+1);
	    pim.draw_line(ig2yx(ig, 1), ig2yx(ig, 0),
			     ig2yx(ig2, 1), ig2yx(ig2, 0),
			     pcol, 3);
	}

	std::string name = ds.figures
	    +str(format("plan_t%03d_n%03d_g%03d_p%03d.png")%tt%nn%gid%pp);
	pim.mirror('y');
	pim.save_png(name.c_str());
    }

}

void vis_feature(vector<matrix<float> >const& feat,
		 directory_structure_t &ds, 
		 int nn, int tt)
{
    using namespace boost::lambda;
    for(int ff=1; ff<feat.size(); ++ff)
    {
	CImg<unsigned char> fim;
	array2d_transform(feat(ff), fim, ll_static_cast<int>(_1*254));

	std::string name = ds.figures
	    +str(format("feat_t%03d_n%03d_f%03d.png")%tt%nn%ff);
	fim.save_png(name.c_str());
    }
}

void choose_critic_obstacles(matrix<int> const& ig2yx,
			     matrix<float> const& obs_cent,
			     vector<int> const& spath,
			     float sdist,
			     matrix<float> & critic_obs_cent)
{
    using namespace boost::lambda;
    float thr = 150.0f;
    vector<float> dist(obs_cent.size1());
    vector<int> perp(obs_cent.size1());

    vector<float> dist2sp(spath.size());
    for(int cc=0; cc<obs_cent.size1(); ++cc)
    {
	for(int ss=0; ss<spath.size(); ++ss)
	{
	    float dx = obs_cent(cc, 0) - ig2yx(spath(ss), 1);
	    float dy = obs_cent(cc, 1) - ig2yx(spath(ss), 0);
	    dist2sp(ss) = dx*dx+dy*dy;
	}
	int idx = std::min_element(dist2sp.begin(), dist2sp.end())
	    - dist2sp.begin();
	dist(cc) = std::sqrt(dist2sp(idx));
	perp(cc) = idx;
    }
    //std::cout<<"dist="<<dist<<std::endl;
    //std::cout<<"perp="<<perp<<std::endl;
    std::cout<<"dist_ratio="<<dist/sdist<<std::endl;

    std::vector<int> cidx;
    for(int cc=0; cc<obs_cent.size1(); ++cc)
    {
	if(dist(cc)>sdist) continue;

	if(perp(cc)>=5 && perp(cc)<spath.size()-5 && dist(cc)<thr)
	{
	    cidx.push_back(cc);
	}
	else if( (perp(cc)<5 && dist(cc)<thr/2) ||
		 (perp(cc)>=spath.size()-5 && dist(cc)<thr/2) )
	{
	    cidx.push_back(cc);
	}
    }


    vector<float> dist2(cidx.size());

    for(int ii=0; ii<cidx.size(); ++ii)
    {
	dist2(ii) = dist(cidx[ii]);
    }

    //3 critic obstacles at the most
    vector<std::pair<int, float> > tmp(dist2.size());
    std::transform(counting_iterator<int>(0),
		   counting_iterator<int>(tmp.size()),
		   dist.begin(),
		   tmp.begin(), bind(std::make_pair<int, float>, _1, _2));

    std::sort(tmp.begin(), tmp.end(),
	      bind(std::less<float>(),
		   bind(&std::pair<int, float>::second, _1),
		   bind(&std::pair<int, float>::second, _2) ) );

    int nc = std::min(std::size_t(3), cidx.size());
    critic_obs_cent = matrix<float>(nc, 2);

    for(int ii=0; ii < nc; ++ii)
    {
	int cc = cidx[tmp[ii].first];
	row(critic_obs_cent, ii) = row(obs_cent, cc);
    }
}


int get_legal_index(matrix<int> const& yx2ig,
		    matrix<int> const& ig2yx,
		    float start_x, float start_y)
{
    int yy = static_cast<int>(start_y+0.5f);
    int xx = static_cast<int>(start_x+0.5f);
    //std::cout<<"yy="<<yy<<", xx="<<xx<<std::endl;
    if(yy>=0 && yy<yx2ig.size1() && xx>=0 && xx<yx2ig.size2())
    {
	int ig = yx2ig(yy, xx);
	if(ig>=0) return ig;
    }

    vector<int> tmp(ig2yx.size1());
    for(int ii=0; ii<ig2yx.size1(); ++ii)
    {
	int dy = yy-ig2yx(ii, 0);
	int dx = xx-ig2yx(ii, 1);
	tmp(ii) = dy*dy+dx*dx;
    }
    return std::min_element(tmp.begin(), tmp.end()) - tmp.begin();
}

void do_homotopy_planning(ground_lim_t const& glim,
			  matrix<double> const& poly_ground2,
			  int nn, float start_x, float start_y,
			  matrix<double> const& goal_ground,
			  vector<matrix<float> > const& car_obs, 
			  vector<matrix<float> > const& ped_obs,
			  vector<planning_result_item_t>& results,
			  directory_structure_t &ds, int tt)
{
    using namespace boost::lambda;

    matrix<double> poly_ground;
    fix_poly_ground(poly_ground2, poly_ground);

    vector<matrix<float> > obs;
    matrix<float> obs_cent;
    combine_obstacles(nn, car_obs,  ped_obs, poly_ground, obs, obs_cent);


#if 0
    std::cout<<"obs_cent="<<obs_cent<<std::endl;
    std::cout<<"poly_ground="<<poly_ground<<std::endl;
    std::cout<<"goal_ground="<<goal_ground<<std::endl;
    std::cout<<"startx="<<start_x<<std::endl;
    std::cout<<"starty="<<start_y<<std::endl;
    //project(obs, range(car_obs.size(), obs.size())) = ped_obs;
    //std::pair<int&, int> a;

    for(int ii=0; ii<obs.size(); ++ii)
    {
	std::cout<<obs(ii)<<std::endl;
    }
#endif

    matrix<int> obs_map;
    matrix<int> dyn_obs_map;
    construct_obstacle_maps(obs, poly_ground, goal_ground,
			    obs_map, dyn_obs_map);


    vector<matrix<float> > feat;
    construct_feature_maps(obs_map, dyn_obs_map, feat);

    vis_feature(feat, ds, nn, tt);

    vector<vector<int> > sg;
    matrix<int> yx2ig;
    matrix<int> ig2yx;
    construct_state_graph(obs_map, sg, yx2ig, ig2yx);
    vis_obstacle_maps(poly_ground,  goal_ground,
		      nn, start_x, start_y,
		      obs_map, dyn_obs_map, ds, tt);

    //std::cout<<"obs_map(start_y, start_x)="
    //     <<obs_map(start_y+0.5, start_x+0.5)<<std::endl;

    matrix<vector<double> > fg;
    construct_feature_graph(obs_map, dyn_obs_map, feat,
			    sg, ig2yx, fg);


    int wnum_l = -1;
    int wnum_u = 0;
    int start = get_legal_index(yx2ig, ig2yx, start_x, start_y);

    results = vector<planning_result_item_t>(goal_ground.size2());


    vector<float> wei(feat.size()+1);
    //wei <<= 1.61642, 7.03021, 15.3362, 1.23757;
    wei <<= 1.61642, 3, 15, 1.23757;

    vector<vector<float> > fdist;
    compute_feat_dist(sg, fg, wei, fdist);

    for(int gg=0; gg<goal_ground.size2(); ++gg)
    {
	//int gy = static_cast<int>(goal_ground(1, gg)+0.5f);
	//int gx = static_cast<int>(goal_ground(0, gg)+0.5f);
	//int goal = yx2ig(gy, gx);
	int goal = get_legal_index(yx2ig, ig2yx,
				   goal_ground(0, gg), goal_ground(1, gg));

	//std::cout<<"start="<<start<<std::endl;
	//std::cout<<"goal="<<goal<<std::endl;

	vector<int> spath;
	float sdist = shortest_path(sg, fdist, start, goal, spath);

	matrix<float> critic_obs_cent;
	choose_critic_obstacles(ig2yx, obs_cent, spath,
				sdist, critic_obs_cent);


	vector<vector<int> > ipath;
	wind_angle_planning(sg, fdist, ig2yx, critic_obs_cent,
			    wnum_l, wnum_u,
			    start, goal,
			    ipath,
			    results(gg).dist,
			    results(gg).wind_num);

	vis_planned_path(poly_ground, goal_ground, nn, start_x, start_y,
			 obs, obs_map, 
			 ig2yx, ipath, ds, tt, gg);
	results(gg).path = vector<matrix<int> >(ipath.size());
	for(int ii=0; ii<ipath.size(); ++ii)
	{
	    results(gg).path(ii) = matrix<int>(ipath(ii).size(), 2);
	    for(int ss=0; ss<ipath(ii).size(); ++ss)
	    {
		results(gg).path(ii)(ss, 0) = ig2yx(ipath(ii)(ss), 1);
		results(gg).path(ii)(ss, 1) = ig2yx(ipath(ii)(ss), 0);
	    }
	}
    }
 
}

#endif

