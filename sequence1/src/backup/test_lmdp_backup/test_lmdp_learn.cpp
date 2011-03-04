#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <limits>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>

#include <boost/numeric/ublas/io.hpp>

#include <CImg.h>

#include "statistics.hpp"
#include "cvpr_array_traits.hpp"
#include "real_timer.hpp"
#include "text_file.hpp"

namespace umf=boost::numeric::bindings::umfpack;
using namespace boost::numeric::ublas;
using namespace boost;
using namespace cvpr;
using namespace cimg_library;

namespace fs = boost::filesystem;


typedef cvpr::sparse_matrix_t<double>::type umf_sparse_matrix;


struct directory_structure_t
{
    directory_structure_t() {
	prefix = "../test/";
	output = "../test/output2/";
	workspace = "../test/workspace/";
	figures = "../test/figures/";
    }
    void make_dir(){
	if ( !fs::exists( prefix ) )
	{
	    fs::create_directory( prefix );
	}
	if ( !fs::exists( workspace ) )
	{
	    fs::create_directory( workspace );
	}
	if ( !fs::exists( output ) )
	{
	    fs::create_directory( output );
	}
    	if ( !fs::exists( figures ) )
	{
	    fs::create_directory( figures );
	}
    }


    std::string prefix;
    std::string output;
    std::string workspace;
    std::string figures;
};

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


void generate_toy_training_example1(directory_structure_t& ds, training_data_t& tdata)
{
    int size=10;
    tdata.obs = scalar_matrix<int>(size, size, 0);
    tdata.dyn_obs = scalar_matrix<int>(size, size, 0);

    tdata.path = matrix<int>(2, 11);
    tdata.path <<= 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9,
	0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9;
	
    for(int ii=tdata.obs.size1()-3; ii<tdata.obs.size1(); ++ii)
    {
	for(int jj=0; jj<3; ++jj)
	{
	    tdata.obs(ii, jj) = 1;
	}
    }

    for(int ii=0; ii<3; ++ii)
    {
	for(int jj=tdata.dyn_obs.size2()-3; jj<tdata.dyn_obs.size2(); ++jj)
	{
	    tdata.dyn_obs(ii, jj) = 1;
	}
    }

}

void generate_toy_training_example2(directory_structure_t& ds, training_data_t& tdata)
{
    int size=20;
    tdata.obs = scalar_matrix<int>(size, size, 0);
    tdata.dyn_obs = scalar_matrix<int>(size, size, 0);

    tdata.path = matrix<int>(2, 22);
    tdata.path <<= 
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 15, 16, 17, 18, 19,
	0, 1, 2, 3, 4, 4, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19;
	
    for(int ii=tdata.obs.size1()-5; ii<tdata.obs.size1(); ++ii)
    {
	for(int jj=0; jj<5; ++jj)
	{
	    tdata.obs(ii, jj) = 1;
	}
    }

    for(int ii=0; ii<5; ++ii)
    {
	for(int jj=tdata.dyn_obs.size2()-5; jj<tdata.dyn_obs.size2(); ++jj)
	{
	    tdata.dyn_obs(ii, jj) = 1;
	}
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

    //array2d_print(std::cout, obs_dist);
    //array2d_print(std::cout, dyn_obs_dist);

    float(*fexp)(float) = std::exp;
    array2d_transform(obs_dist, feat(1), bind(fexp, -_1/4.0f));
    array2d_transform(dyn_obs_dist, feat(2), bind(fexp, -_1/4.0f));

}

struct nbrhood
{
    static int dy[8];
    static int dx[8];
};

int nbrhood::dy[8]={0, 1, 0, -1, 1, 1, -1, -1};
int nbrhood::dx[8]={1, 0, -1, 0, 1, -1, -1, 1};

void get_feature_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
		       vector<matrix<float> > const& feat,
		       vector<umf_sparse_matrix>& fg,
		       matrix<int>& yx2ig,
		       matrix<int>& ig2yx)
{
    using namespace boost::lambda;
    matrix<int> obs_all(obs+dyn_obs);
    int ng = std::count(obs_all.data().begin(), obs_all.data().end(), 0);
    fg = vector<umf_sparse_matrix>(feat.size());
    for(int ff=0; ff<feat.size(); ++ff)
    {
	fg(ff) = umf_sparse_matrix(ng, ng, ng*8);
    }


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

    for(int ff=0; ff<feat.size(); ++ff)
    {
	for(int yy=0; yy<yx2ig.size1(); ++yy)
	{
	    for(int xx=0; xx<yx2ig.size2(); ++xx)
	    {
		int ig1 = yx2ig(yy, xx);
		if(ig1<0) continue;
		for(int nn=0; nn<8; ++nn)
		{
		    int y2 = yy+nbrhood::dy[nn];
		    int x2 = xx+nbrhood::dx[nn];
		    if(y2<0 || y2>=obs.size1()) continue;
		    if(x2<0 || x2>=obs.size2()) continue;
		    int ig2 = yx2ig(y2, x2);
		    if(ig2<0) continue;
		    int dx = nbrhood::dx[nn];
		    int dy = nbrhood::dy[nn];
		    double dist = std::sqrt(static_cast<double>(dx*dx+dy*dy));
		    fg(ff)(ig1, ig2) = (feat(ff)(yy, xx)+feat(ff)(y2, x2))/2*dist;
		}
	    }
	}
    }


}

void compute_pps_unnormalized(vector<matrix<float> > const& feat,
			      vector<umf_sparse_matrix> const& fg,
			      matrix<int> const& yx2ig,
			      matrix<int> const& ig2yx,
			      vector<double> const& wei,
			      umf_sparse_matrix& pps)
{
    using namespace boost::lambda;
    if(pps.size1()==0)
    {
	int ng = std::count_if(yx2ig.data().begin(), yx2ig.data().end(), _1>=0);

	pps = umf_sparse_matrix(ng, ng, ng*8);
    }


    for(int yy=0; yy<yx2ig.size1(); ++yy)
    {
	for(int xx=0; xx<yx2ig.size2(); ++xx)
	{
	    int ig1 = yx2ig(yy, xx);
	    if(ig1<0) continue;
	    for(int nn=0; nn<8; ++nn)
	    {
		int y2 = yy+nbrhood::dy[nn];
		int x2 = xx+nbrhood::dx[nn];
		if(y2<0 || y2>=yx2ig.size1()) continue;
		if(x2<0 || x2>=yx2ig.size2()) continue;
		int ig2 = yx2ig(y2, x2);
		if(ig2<0) continue;
		double wsumf = 0;
		for(int ff=0; ff<fg.size(); ++ff)
		{
		    wsumf += fg(ff)(ig1, ig2)*wei(ff);
		}
		pps(ig1, ig2) = std::exp(-wsumf);
	    }

	}
    }

}

void compute_pps(vector<matrix<float> > const& feat,
		 vector<umf_sparse_matrix> const& fg,
		 matrix<int> const& yx2ig,
		 matrix<int> const& ig2yx,
		 vector<double> const& wei,
		 umf_sparse_matrix& pps,
		 vector<double>& qvec)
{
    using namespace boost::lambda;
    compute_pps_unnormalized(feat, fg, yx2ig, ig2yx, wei, pps);
    int ng = pps.size1();
    qvec = vector<double>(ng);
    umf_sparse_matrix::iterator1 it1;
    for(it1=pps.begin1(); it1 != pps.end1(); ++it1)
    {
	double denom = std::accumulate(it1.begin(), it1.end(), 0.0f);
	std::for_each(it1.begin(), it1.end(), _1/=denom);
	qvec(it1.index1()) = -std::log(denom);
    }

}

void compute_pps(vector<matrix<float> > const& feat,
		 vector<umf_sparse_matrix> const& fg,
		 matrix<int> const& yx2ig,
		 matrix<int> const& ig2yx,
		 vector<double> const& wei,
		 umf_sparse_matrix& pps)
{
    using namespace boost::lambda;
    compute_pps_unnormalized(feat, fg, yx2ig, ig2yx, wei, pps);
    umf_sparse_matrix::iterator1 it1;
    for(it1=pps.begin1(); it1 != pps.end1(); ++it1)
    {
	double denom = std::accumulate(it1.begin(), it1.end(), 0.0f);
	std::for_each(it1.begin(), it1.end(), _1/=denom);
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


void compute_grad_L1(vector<umf_sparse_matrix> const& fg,
		     umf_sparse_matrix const& pps,
		     vector<vector<int> >const& path_ig,
		     vector<double>& grad_L1)
{
    int nf = fg.size();
    grad_L1 = scalar_vector<double>(nf, 0.0f);

    for(int ff=0; ff<nf; ++ff)
    {
	for(int cc=0; cc<path_ig.size(); ++cc)
	{
	    for(int tt=0; tt+1<path_ig(cc).size(); ++tt)
	    {
		int ig1 = path_ig(cc)(tt);
		int ig2 = path_ig(cc)(tt+1);
		grad_L1(ff) += -fg(ff)(ig1, ig2);
		for(int ii=0; ii<pps.size2(); ++ii)
		{
		    if(pps(ig1, ii)) 
		    {
			grad_L1(ff) += fg(ff)(ig1, ii)*pps(ig1, ii);
			//std::cout<<"great, "<<fg(ff)(ig1, ii)<<std::endl;
		    }
		}
	    }
	}
    }
}

void compute_hessian_L1(vector<umf_sparse_matrix> const& fg,
			umf_sparse_matrix const& pps,
			vector<vector<int> > const& path_ig,
			matrix<double>& hess_L1)
{
    using namespace boost::lambda;
    typedef matrix_row<umf_sparse_matrix const> const_row;
    int nf = fg.size();
    hess_L1 = scalar_matrix<double>(nf, nf, 0.0f);
    for(int ff=0; ff<nf; ++ff)
    {
	for(int ll=0; ll<nf; ++ll)
	{
	    for(int cc=0; cc<path_ig.size(); ++cc)
	    {
		for(int tt=0; tt+1<path_ig(cc).size(); ++tt)
		{
		    int ig1 = path_ig(cc)(tt);
		    const_row rpps(pps, ig1);
		    const_row rf1(fg(ff), ig1);
		    const_row rf2(fg(ll), ig1);
		    const_row::const_iterator ip;
		    const_row::const_iterator i1;
		    const_row::const_iterator i2;
		    for(ip = rpps.begin(), i1=rf1.begin(), i2=rf2.begin();
			ip!= rpps.end();
			++ip, ++i1, ++i2)
		    {
			hess_L1(ff, ll) += -*ip**i1**i2;
		    }
		    double pf1 = std::inner_product(rpps.begin(), rpps.end(), rf1.begin(), 0.0f);
		    double pf2 = std::inner_product(rpps.begin(), rpps.end(), rf2.begin(), 0.0f);
		    hess_L1(ff, ll) += pf1*pf2;

		}
	    }
	}
    }
}

void learn_weights_L1(vector<matrix<float> > const& feat,
		      vector<umf_sparse_matrix>const& fg,
		      umf_sparse_matrix& pps,
		      matrix<int> const& yx2ig,
		      matrix<int> const& ig2yx,
		      vector<vector<int> > const& path_ig,
		      vector<double>& wei)
{

    for(int it=0; it<4; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	compute_pps(feat, fg, yx2ig, ig2yx, wei, pps);

	vector<double> grad_L1;
	compute_grad_L1(fg, pps, path_ig, grad_L1);

	std::cout<<"grad_L1="<<std::endl;
	array1d_print(std::cout, grad_L1);
	
	matrix<double> hess_L1;
	compute_hessian_L1(fg, pps, path_ig, hess_L1);

	std::cout<<"hess_L1="<<std::endl;
	array2d_print(std::cout, hess_L1);

	permutation_matrix<std::size_t> P(hess_L1.size1());

	int res = lu_factorize(hess_L1, P);  //vv holds L and U
	lu_substitute(hess_L1, P, grad_L1);       //vd holds the solution var_{-1}*data

	//std::cout<<"grad_L1="<<std::endl;
	//std::cout<<-grad_L1<<std::endl;

	wei += -grad_L1;
	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);
    }

}

void learn_weights_L1(vector<matrix<float> > const& feat,
		      vector<umf_sparse_matrix>const& fg,
		      umf_sparse_matrix& pps,
		      matrix<int> const& yx2ig,
		      matrix<int> const& ig2yx,
		      vector<vector<int> > const& path_ig,
		      vector<double>& wei,
		      double reg_eps)
{
    using namespace boost::lambda;
    double(*flog)(double) = std::log;
    double(*fexp)(double) = std::exp;

    for(int it=0; it<40; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	compute_pps(feat, fg, yx2ig, ig2yx, wei, pps);

	vector<double> grad_L1;
	compute_grad_L1(fg, pps, path_ig, grad_L1);

	std::cout<<"grad_L1="<<std::endl;
	array1d_print(std::cout, grad_L1);
	
	matrix<double> hess_L1;
	compute_hessian_L1(fg, pps, path_ig, hess_L1);

	std::cout<<"hess_L1="<<std::endl;
	array2d_print(std::cout, hess_L1);

	vector<double> logw(wei.size());
	std::transform(wei.begin(), wei.end(), logw.begin(), flog);
	vector<double> grad_L1_lw = element_prod(grad_L1, wei)-reg_eps*logw;
	matrix<double> hess_L1_lw = element_prod(hess_L1 , outer_prod(wei, wei))
	    -reg_eps*identity_matrix<double>(wei.size());
	for(int dd=0; dd<logw.size(); ++dd)
	{
	    hess_L1_lw(dd, dd) += grad_L1(dd)*wei(dd);
	}

	permutation_matrix<std::size_t> P(hess_L1_lw.size1());

	int res = lu_factorize(hess_L1_lw, P);
	lu_substitute(hess_L1_lw, P, grad_L1_lw); 

	//std::cout<<"grad_L1="<<std::endl;
	//std::cout<<-grad_L1<<std::endl;

	logw += -grad_L1_lw;
	std::transform(logw.begin(), logw.end(), wei.begin(), fexp);
	//wei += -grad_L1;
	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);
    }

}

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

void draw_sim_path(CImg<unsigned char>& vis,
		   training_data_t const& tdata,
		   vector<vector<int> > const& sim_path,
		   matrix<int> const& ig2yx)
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

    for(int ll=0; ll<sim_path.size(); ++ll)
    {
	for(int kk=0; kk+1<sim_path(ll).size(); ++kk)
	{
	    int ig1 = sim_path(ll)(kk);
	    int ig2 = sim_path(ll)(kk+1);
	    vis.draw_line(ig2yx(ig1, 1), ig2yx(ig1, 0),
			  ig2yx(ig2, 1), ig2yx(ig2, 0), ccol);
	}
    }

}

void draw_value_function(matrix<double>const & v,
			 CImg<unsigned char> & vis_val, 
			 double infty)
{
    using namespace boost::lambda;
    typedef array2d_traits<CImg<unsigned char> > tr;
    tr::change_size(vis_val, v.size1(), v.size2());

    matrix<double> v2=v;
    for(int ii=0; ii<v2.size1(); ++ii)
    {
	for(int jj=0; jj<v2.size2(); ++jj)
	{
	    if(infty-v2(ii, jj)<1e-6) v2(ii, jj) = -1;
	}
    }
    double max_val = *(std::max_element(v2.data().begin(), v2.data().end()));
    array2d_transform(v2, vis_val, ll_static_cast<unsigned char>(_1*254.0f/max_val));

    for(int ii=0; ii<v2.size1(); ++ii)
    {
	for(int jj=0; jj<v2.size2(); ++jj)
	{
	    if(v2(ii, jj)<=-0.5) tr::ref(vis_val, ii, jj) = 255;
	}
    }

}


////////////////////////////////////////////////////////////////

void compute_qp(vector<double> const& q, umf_sparse_matrix const& pp,
		umf_sparse_matrix& qp)
{
    using namespace boost::lambda;
    qp = pp;
    for(umf_sparse_matrix::iterator1 it1=qp.begin1(); it1 != qp.end1(); ++it1)
    {
	double expq = std::exp(-q(it1.index1()));
	std::for_each(it1.begin(), it1.end(), _1 *= expq);
    }

}

void get_qp_block(vector<int> const& goals,
		  umf_sparse_matrix const& qp,
		  umf_sparse_matrix & qpnn,
		  matrix<double>& qpnt)
{
    using namespace boost::lambda;
    int ng = qp.size1();
    int num_n = ng - goals.size();
    int num_t = goals.size();
    vector<int> mapvec(scalar_vector<int>(ng, 0));
    for(int gg=0; gg<goals.size(); ++gg)
    {
	mapvec(goals(gg)) = 1;
    }

    qpnn = umf_sparse_matrix(num_n, num_n, num_n*4);
    qpnt = matrix<double>(num_n, num_t);

    int rr = 0;
    for(umf_sparse_matrix::const_iterator1 it1=qp.begin1(); it1 != qp.end1(); ++it1)
    {
	int ii = it1.index1();
	if(1==mapvec(ii)) continue;
	int cn = 0;
	int ct = 0;
	real_timer_t timer1;
	for(int jj=0; jj<qp.size2(); ++jj)
	{
	    if(mapvec(jj)==1)
	    {
		qpnt(rr, ct) = qp(ii, jj);
		++ct;
	    }
	    else
	    {
		if(qp(ii, jj)) //important for performance
		    qpnn(rr, cn) = qp(ii, jj);
		++cn;
	    }
	}
	++rr;
    }
}


void combine_z(vector<double> const& zn,
	       vector<double> const& zt,
	       vector<int> const& goals,
	       vector<double>& z)
{
    z = scalar_vector<double>(zn.size()+zt.size(), 0);
    vector<int> gv(scalar_vector<int>(z.size(), 0));
    for(int ii = 0; ii<goals.size(); ++ii)
    {
	gv(goals(ii)) = 1;
    }

    int cn = 0;
    int ct = 0;
    for(int ii=0; ii<z.size(); ++ii)
    {
	if(gv(ii)) 
	{
	    z(ii) = zt(ct);
	    ++ct;
	}
	else
	{
	    z(ii) = zn(cn);
	    ++cn;
	}
    }
}

void compute_u(umf_sparse_matrix const& pps,
	       vector<double> const& vvec,
	       vector<double> const& qvec,
	       umf_sparse_matrix& u)
{
    u = umf_sparse_matrix(pps.size1(), pps.size2(), pps.nnz());
    umf_sparse_matrix::const_iterator1 it1;
    umf_sparse_matrix::const_iterator2 it2;
    for(it1=pps.begin1(); it1!=pps.end1(); ++it1)
    {
	int ii = it1.index1();
	for(it2=it1.begin(); it2!=it1.end(); ++it2)
	{
	    int jj = it2.index2();
	    u(ii, jj) = vvec(ii)-vvec(jj)-qvec(ii);
	}
    }
}


void simulate_path(umf_sparse_matrix const& u,
		   vector<int> const& starts, int goal,
		   vector<vector<int> > & sim_path)
{
    int np = starts.size();
    sim_path = vector<vector<int> >(np);

    for(int pp=0; pp<np; ++pp)
    {
	std::vector<int> tp;
	int pos = starts(pp);
	tp.push_back(pos);
	while(pos!=goal)
	{
	    matrix_row<umf_sparse_matrix const> ur(u, pos);
	    matrix_row<umf_sparse_matrix const>::const_iterator it = std::max_element(ur.begin(), ur.end());
	    pos = it.index();
	    tp.push_back(pos);
	}
	//sim_path(pp) = vector<int>(tp.size());
	array1d_copy(tp, sim_path(pp));

    }
}

int main(int argc, char* argv[])
{
    using namespace boost::lambda;
    directory_structure_t ds;
    ds.make_dir();
    //array<int, 2> goal;
    training_data_t tdata;

#if 1
    generate_toy_training_example1(ds, tdata);
    tdata.save(ds.prefix+"training_data1.xml");



    generate_toy_training_example2(ds, tdata);
    tdata.save(ds.prefix+"training_data2.xml");
#endif

    tdata.load(ds.prefix+"training_data2.xml");
    matrix<int>& obs = tdata.obs;
    matrix<int>& dyn_obs = tdata.dyn_obs;
    matrix<int>& path = tdata.path;

    CImg<unsigned char> vis_map;
    draw_training_data(vis_map, tdata);
    {
	std::string name = ds.figures+"vis_map.png";
	vis_map.save_png(name.c_str());
    }
#if 0
    std::cout<<"obs:"<<std::endl;
    array2d_print(std::cout, obs);
    std::cout<<"dyn_obs:"<<std::endl;
    array2d_print(std::cout, dyn_obs);

    matrix<char> vis(obs.size1(), obs.size2());
    for(int ii=0; ii<vis.size1(); ++ii)
    {
	for(int jj=0; jj<vis.size2(); ++jj)
	{
	    vis(ii, jj) = ' ';
	    if(obs(ii, jj)) vis(ii, jj) = 'o';
	    if(dyn_obs(ii, jj)) vis(ii, jj) = 'x';
	}
    }
    for(int kk=0; kk<path.size2(); ++kk)
    {
	vis(path(0, kk), path(1, kk)) = '@';
    }

    std::cout<<"vis:"<<std::endl;
    array2d_print(std::cout, vis);
#endif

    vector<matrix<float> > feat;
    generate_feature_maps(obs, dyn_obs, feat);

#if 0
    for(int ii=0; ii<feat.size(); ++ii)
    {
	array2d_print(std::cout, feat(ii));
    }
#endif

    for(int ii=0; ii<feat.size(); ++ii)
    {
	CImg<unsigned char> vis_feat;
	double maxf = *(std::max_element(feat(ii).data().begin(), feat(ii).data().end()));
	array2d_transform(feat(ii), vis_feat, ll_static_cast<unsigned char>(_1*254/maxf));
	std::string name = ds.figures+str(format("vis_feat%03d.png")%ii);
	vis_feat.save_png(name.c_str());
    }

    vector<umf_sparse_matrix> fg;
    umf_sparse_matrix pps;
    matrix<int> yx2ig;
    matrix<int> ig2yx;

    get_feature_graph(obs, dyn_obs, feat, fg, yx2ig, ig2yx);

    //array2d_print(std::cout, pps);

    vector<vector<int> > path_ig(1);
    get_path_ig(path, yx2ig, path_ig(0));

    vector<double> wei(3);
    wei <<= 0.33, 0.33, 0.33;

    //learn_weights_L1(feat, fg, pps, yx2ig, ig2yx, path_ig, wei, 0.001);
    learn_weights_L1(feat, fg, pps, yx2ig, ig2yx, path_ig, wei);

    if(wei(0)<0) wei(0) = 0.01;

    wei *= 10;

    vector<double> qvec;
    compute_pps(feat, fg, yx2ig, ig2yx, wei, pps, qvec);

    umf_sparse_matrix qp;
    compute_qp(qvec, pps, qp);
    //std::cout<<timer.elapsed()<<": done compute_qp"<<std::endl;

    vector<int> goals(1);
    goals <<= *(path_ig(0).rbegin());
    umf_sparse_matrix qpnn;
    matrix<double> qpnt;
    get_qp_block(goals, qp, qpnn, qpnt);

    int num_t = goals.size();
    vector<double> zt = scalar_vector<double>(num_t, 1.0f);

    vector<double> qpz0 = prod(qpnt, zt);

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    

    //array2d_print(std::cout, qpnn);

    umf_sparse_matrix A(-qpnn);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) += 1.0f;
    }
    //std::cout<<timer.elapsed()<<": done fill A"<<std::endl;

    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    vector<double> zn(qpz0.size());
    umf::solve(A, zn, qpz0, nume);  
    //std::cout<<timer.elapsed()<<": done solve A"<<std::endl;

    //std::cout<<zn<<std::endl;

    vector<double> z;
    combine_z(zn, zt, goals, z);  

    vector<double> vvec(z.size());
    double(*flog)(double) = std::log;
    std::transform(z.begin(), z.end(), vvec.begin(), -bind(flog, _1));

    matrix<double> v(scalar_matrix<double>(tdata.obs.size1(), tdata.obs.size2(), 1e6));

    for(int nn=0; nn<z.size(); ++nn)
    {
	int yy = ig2yx(nn, 0);
	int xx = ig2yx(nn, 1);
	v(yy, xx) = vvec(nn);//-std::log(z(nn));
    }

    //std::cout<<v<<std::endl;
    cvpr::array2d_print(std::cout, v);

    umf_sparse_matrix u;
    compute_u(pps, vvec, qvec, u);

    vector<vector<int> > sim_path;
    vector<int> starts(1);
    starts <<= 0;

    simulate_path(u, starts, goals(0), sim_path);
    std::cout<<sim_path(0)<<std::endl;

    CImg<unsigned char> vis_sim;
    draw_sim_path(vis_sim, tdata, sim_path, ig2yx);
    {
	std::string name = ds.figures+"vis_sim.png";
	vis_sim.save_png(name.c_str());
    }

    CImg<unsigned char> vis_val;
    draw_value_function(v, vis_val, 1e6);
    {
	std::string name = ds.figures+"vis_val.png";
	vis_val.save_png(name.c_str());
    }

    return 0;
}
