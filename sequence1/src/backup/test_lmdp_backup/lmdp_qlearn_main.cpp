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
#include "misc_utils.hpp"

namespace umf=boost::numeric::bindings::umfpack;
using namespace boost::numeric::ublas;
using namespace boost;
using namespace cvpr;
using namespace cimg_library;

namespace fs = boost::filesystem;


typedef cvpr::sparse_matrix_t<double>::type umf_sparse_matrix;

#include "lmdp.hpp"
#include "lmdp_qlearn.hpp"

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



int xmain(int argc, char* argv[])
{
    matrix<double> m(3,3);
    m <<=0.8147,    0.9134,    0.2785,
	0.9058,    0.6324,    0.5469,
	0.1270,    0.0975,    0.9575;
    vector<double> v;
    double l = eigs(m, v);

    std::cout<<l<<std::endl;
    std::cout<<v<<std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
    using namespace boost::lambda;
    directory_structure_t ds;
    ds.make_dir();
    //array<int, 2> goal;
    training_data_t tdata;

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

    vector<matrix<float> > feat;
    generate_feature_maps(obs, dyn_obs, feat);

    for(int ii=0; ii<feat.size(); ++ii)
    {
	CImg<unsigned char> vis_feat;
	double maxf = *(std::max_element(feat(ii).data().begin(), feat(ii).data().end()));
	array2d_transform(feat(ii), vis_feat, ll_static_cast<unsigned char>(_1*254/maxf));
	std::string name = ds.figures+str(format("vis_feat%03d.png")%ii);
	vis_feat.save_png(name.c_str());
    }

    vector<vector<int> > sg; //(num_states, num_neighbors)
    matrix<vector<double> > fg; //(num_features, num_states, num_neighbors)
    vector<vector<double> > log_pps; //(num_states, num_neighbors)
    matrix<int> yx2ig;
    matrix<int> ig2yx;

    get_state_graph(obs, dyn_obs, sg, yx2ig, ig2yx);
    //std::cout<<sg<<std::endl;

    get_feature_graph(obs, dyn_obs, feat, sg, ig2yx, fg);

    //array2d_print(std::cout, pps);

    vector<vector<int> > path_ig(1);
    get_path_ig(path, yx2ig, path_ig(0));

    vector<double> wei(feat.size()+1);
    wei <<= 0.33, 0.33, 0.33, 0.5;

    wei *= 40;
    vector<double> q;
    lmdp_t lmdp;
    lmdp.initialize(fg, sg, yx2ig, ig2yx);


    //learn_weights(lmdp, path_ig, wei);
    unit_test(lmdp, path_ig, wei);
    return 0;
}

//////backup code
void solve_lmdpt(matrix<vector<double> >const& fg,
		vector<vector<int> >const& sg,
		matrix<int> const& ig2yx,
		vector<double> const& wei,
		vector<vector<double> > const& log_pps,
		vector<double> const& q,
		int goal,
		vector<double>& logz)
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0f);

    vector<int> sizes(sg.size());
    std::transform(sg.begin(), sg.end(), sizes.begin(), bind(&vector<int>::size, _1));
    int nnz = sum(sizes);

    umf_sparse_matrix qp(ng, ng, nnz);

    //std::cout<<"q="<<q<<std::endl;
    //std::cout<<"logpps="<<log_pps<<std::endl;

    for(int ig = 0; ig<ng; ++ig)
    {
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    qp(ig, ig2) = std::exp(-q(ig)+log_pps(ig)(nn));
	}
    }
    vector<int> good=scalar_vector<int>(ng, 0);
    good(goal) = 1;

    splitter_t<> splitter(good);
    matrix<umf_sparse_matrix> qp_block = split_sparse_mat<matrix>(splitter, qp);
    //split_sparse_matrix(qp, good, qp_block);

    vector<double> z2(qp_block(0, 1).size2());
    for(int ii=0, jj=0; ii<good.size(); ++ii)
    {
	if(good(ii))
	{
	    z2(jj) = 1.0l;
	    ++jj;
	}
    }


    umf_sparse_matrix A = -qp_block(0,0);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) = 1.0l;
    }

    umf_sparse_matrix B = -qp_block(1,1);
    for(int ii=0; ii<B.size1(); ++ii)
    {
	B(ii, ii) = 1.0l;
    }

    umf_sparse_matrix G = prod(trans(A), A) +
	prod(trans(qp_block(1, 0)), qp_block(1, 0));
    vector<double> Hvec= prod(qp_block(0, 1), z2);
    Hvec += prod(trans(qp_block(1, 0)), z2) ;
    Hvec -= prod(trans(qp_block(0, 0)), prod<vector<double> >(qp_block(0, 1), z2));
    Hvec -= prod(trans(qp_block(1, 0)), prod<vector<double> >(qp_block(1, 1), z2));

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    umf::symbolic (G, symb); 
    umf::numeric (G, symb, nume); 

    vector<double> z1=scalar_vector<double>(Hvec.size(), 0.0l);
    umf::solve(G, z1, Hvec, nume);  
///

    vector<double> z(ng);

    array<int, 2> cc={0,0};
    for(int ii=0; ii<z.size(); ++ii)
    {
	if(good(ii)) 
	{
	    z(ii) = z2(cc[1]);
	    cc[1]++;
	}
	else 
	{
	    z(ii) = z1(cc[0]);
	    cc[0]++;
	}
    }

    logz = vector<double>(z.size());
    std::transform(z.begin(), z.end(), logz.begin(), flog);
    std::cout<<"logz="<<logz<<std::endl;
    //std::cout<<"logzz="<<logz<<std::endl;
    //std::cout<<"z="<<z<<std::endl;

}

void solve_lmdpz(matrix<vector<double> >const& fg,
		vector<vector<int> >const& sg,
		matrix<int> const& ig2yx,
		vector<double> const& wei,
		vector<vector<double> > const& log_pps,
		vector<double> const& q,
		int goal,
		vector<double>& logz)
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0f);

    vector<int> sizes(sg.size());
    std::transform(sg.begin(), sg.end(), sizes.begin(), bind(&vector<int>::size, _1));
    int nnz = sum(sizes);

    umf_sparse_matrix qp(ng, ng, nnz);

    //std::cout<<"q="<<q<<std::endl;
    //std::cout<<"logpps="<<log_pps<<std::endl;

    for(int ig = 0; ig<ng; ++ig)
    {
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    qp(ig, ig2) = std::exp(-q(ig)+log_pps(ig)(nn));
	}
    }
    vector<int> good=scalar_vector<int>(ng, 0);
    good(goal) = 1;

    splitter_t<> splitter(good);
    matrix<umf_sparse_matrix> qp_block = split_sparse_mat<matrix>(splitter, qp);
    //split_sparse_matrix(qp, good, qp_block);

    vector<double> z2(qp_block(0, 1).size2());
    for(int ii=0, jj=0; ii<good.size(); ++ii)
    {
	if(good(ii))
	{
	    z2(jj) = 1.0l;
	    ++jj;
	}
    }

    vector<double> z1=scalar_vector<double>(qp_block(0,0).size1(), 0.0001l);
    vector<double> zz0=prod(qp_block(0, 1), z2);

    for(int it=0; it<100; ++it)
    {
	vector<double> tmp = zz0;
	axpy_prod(qp_block(0,0), z1, tmp, false);
	z1 = tmp;
    }

///

    vector<double> z(ng);

    array<int, 2> cc={0,0};
    for(int ii=0; ii<z.size(); ++ii)
    {
	if(good(ii)) 
	{
	    z(ii) = z2(cc[1]);
	    cc[1]++;
	}
	else 
	{
	    z(ii) = z1(cc[0]);
	    cc[0]++;
	}
    }

    logz = vector<double>(z.size());
    std::transform(z.begin(), z.end(), logz.begin(), flog);
    std::cout<<"logz="<<logz<<std::endl;
    //std::cout<<"logzz="<<logz<<std::endl;
    //std::cout<<"z="<<z<<std::endl;

}


void solve_lmdp_eigs(matrix<vector<double> >const& fg,
		     vector<vector<int> >const& sg,
		     matrix<int> const& ig2yx,
		     vector<double> const& wei,
		     vector<vector<double> > const& log_pps,
		     vector<double> const& q,
		     vector<double>& logz)
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0f);

    vector<int> sizes(sg.size());
    std::transform(sg.begin(), sg.end(), sizes.begin(), bind(&vector<int>::size, _1));
    int nnz = sum(sizes);

    umf_sparse_matrix qp(ng, ng, nnz);

    for(int ig = 0; ig<ng; ++ig)
    {
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    qp(ig, ig2) = std::exp(-q(ig)+log_pps(ig)(nn));
	}
    }

    vector<double> z;
    double lam = eigs(qp, z);

    vector<int> good(z.size());
    std::transform(z.begin(), z.end(), good.begin(), _1>1e-6);

    for(int ii=0; ii<z.size(); ++ii)
    {
	if(good(ii)) logz(ii) = std::log(z(ii));
    }


    vector<int>::iterator fit = std::find(good.begin(), good.end(), 0);
    std::cout<<"lam="<<lam<<std::endl;
    vector<double> logzz(z.size());
    std::transform(z.begin(), z.end(), logzz.begin(), flog);
    //std::cout<<"logz="<<logz<<std::endl;
    std::cout<<"logzz="<<logzz<<std::endl;

    if(fit == good.end())  return;

    splitter_t<> splitter(good);
    matrix<umf_sparse_matrix> qp_block = split_sparse_mat<matrix>(splitter, qp);
    //split_sparse_matrix(qp, good, qp_block);

    vector<double> z2(qp_block(0, 1).size2());
    for(int ii=0, jj=0; ii<good.size(); ++ii)
    {
	if(good(ii))
	{
	    z2(jj) = z(ii);
	    ++jj;
	}
    }

    vector<double> A12z2(qp_block(0, 1).size1());
    axpy_prod(qp_block(0, 1), z2, A12z2, true);
    A12z2 *= 1e14;
    umf_sparse_matrix A = -qp_block(0,0);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) = 1.0l;
    }

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    std::cout<<"A.size="<<A.size1()<<", "<<A.size2()<<std::endl;
    //std::cout<<"z1.size="<<z1.size()<<std::endl;
    std::cout<<"B.size="<<A12z2.size()<<std::endl;

    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    vector<double> z1=scalar_vector<double>(A12z2.size(), 0.0l);
    umf::solve(A, z1, A12z2, nume);  
    vector<double> logz1(z1.size());
    std::transform(z1.begin(), z1.end(), logz1.begin(),
		   bind(flog, _1)-std::log(1e14l) );

    for(int ii=0, jj=0; ii<good.size(); ++ii)
    {
	if(!good(ii))
	{
	    logz(ii) = logz1(jj);
	    ++jj;
	}
    }

    std::cout<<"lam="<<lam<<std::endl;
    //vector<double> logzz(z.size());
    std::transform(z.begin(), z.end(), logzz.begin(), flog);
    //std::cout<<"logz="<<logz<<std::endl;
    //std::cout<<"logzz="<<logzz<<std::endl;
}

#include "lmdp_details.hpp"
