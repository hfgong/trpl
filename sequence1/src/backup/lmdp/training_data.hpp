#ifndef __TRAINING_DATA__HPP__INCLUDED__
#define  __TRAINING_DATA__HPP__INCLUDED__


struct training_data_t
{
    matrix<int> obs;
    matrix<int> dyn_obs;
    matrix<int> path;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)	{
	ar & BOOST_SERIALIZATION_NVP(obs);
	ar & BOOST_SERIALIZATION_NVP(dyn_obs);
	ar & BOOST_SERIALIZATION_NVP(path);
    }

    void save(std::string const& fname)	const {
	save_xml(*this, fname, "training_data");
    }
    void load(std::string const& fname) {
	load_xml(*this, fname, "training_data");
    }
};

void draw_training_data(CImg<unsigned char>& vis,
			training_data_t const& tdata)
{

    typedef array3d_traits<CImg<unsigned char> > tr;
    tr::change_size(vis, 3, tdata.obs.size1(), tdata.obs.size2());
    for(int ii=0; ii<tdata.obs.size1(); ++ii)
    {
	for(int jj=0; jj<tdata.obs.size2(); ++jj)
	{
	    tr::ref(vis, 0, ii, jj) = 0;
	    tr::ref(vis, 1, ii, jj) = 0;
	    tr::ref(vis, 2, ii, jj) = 0;
	    if(tdata.obs(ii, jj)) tr::ref(vis, 2, ii, jj) = 255;
	    if(tdata.dyn_obs(ii, jj)) tr::ref(vis, 0, ii, jj) = 255;
	}
    }
    char ccol[3] = {255, 255, 255};

    for(int kk=0; kk+1<tdata.path.size2(); ++kk)
    {
	vis.draw_line(tdata.path(1, kk), tdata.path(0, kk),
		      tdata.path(1, kk+1), tdata.path(0, kk+1), ccol);
    }

}

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


void get_path_ig(matrix<int> const& path,
		 matrix<int> const& yx2ig,
		 vector<int>& path_ig)
{
    path_ig = vector<int>(path.size2());
    for(int ii=0; ii<path.size2(); ++ii)
    {
	path_ig(ii) = yx2ig(path(0, ii), path(1, ii));
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



#endif
