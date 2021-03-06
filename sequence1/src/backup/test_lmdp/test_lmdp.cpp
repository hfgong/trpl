#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>

#include <boost/numeric/ublas/io.hpp>


#include "statistics.hpp"
#include "cvpr_array_traits.hpp"
#include "real_timer.hpp"

namespace umf=boost::numeric::bindings::umfpack;
using namespace boost::numeric::ublas;
using namespace cvpr;

typedef cvpr::sparse_matrix_t<double>::type umf_sparse_matrix;

void create_map(matrix<int>& map)
{
    //map = scalar_matrix<int>(70, 70, 0); // 0-- trepassable
    map = scalar_matrix<int>(10, 10, 0); // 0-- trepassable
    map(0, 0) = 1; //goal

    for(int jj=0; jj<5; ++jj)
    {
	map(2, jj) = -1; //obstacles
	map(4, map.size2()-1-jj) = -1;
    }
}

void create_qvec(matrix<int> const& map,
		 vector<int>& mapvec,
		 vector<double>& q,
		 double scale, double infty)
{

    q = vector<double>(map.size1()*map.size2());
    mapvec = vector<int>(q.size());
    int kk = 0;
    for(int ii = 0; ii<map.size1(); ++ii)
    {
	for(int jj=0; jj<map.size2(); ++jj)
	{
	    switch(map(ii, jj))
	    {
	    case 0:  q(kk) = scale; break;
	    case 1:  q(kk) = 0.0f; break;
	    case -1: q(kk) = infty; break;
	    default: q(kk) = infty;
	    }
	    mapvec(kk) = map(ii, jj);
	    ++kk;
	}
    }
}

void create_passive_prob(matrix<int> const& map,
			 umf_sparse_matrix& pp)
{
    using namespace boost::lambda;
    int nrow = map.size1();
    int ncol = map.size2();
    int np = nrow*ncol;
    int KN = 0;
    int Nbr = (KN*KN+KN)*4+(KN==0)*4;

   
    int ndi[Nbr];
    int ndj[Nbr];
    int bb=0;
    if(KN==0) {
	ndi[0]= 1;  ndj[0]=0;
	ndi[1]= 0;  ndj[1]=1;
	ndi[2]= -1; ndj[2]=0;
	ndi[3]= 0;  ndj[3]=-1;
    }
    else
    {
	for(int di=-KN; di <= KN; ++di)
	{
	    for(int dj = -KN; dj <= KN; ++dj)
	    {
		if(di==0 &&  dj==0) continue;
		ndi[bb] = di;
		ndj[bb] = dj;
		bb++;
	    }
	}
    }

    pp = umf_sparse_matrix(np, np, np*4);

    for(int ii=0; ii<nrow; ++ii)
    {
	for(int jj=0; jj<ncol; ++jj)
	{
	    int iii = ii*ncol+jj;
	    if(map(ii, jj)==1) //sink
	    {
		pp(iii, iii) = 1;
		continue;
	    }
	    for(int nn=0; nn<Nbr; ++nn)
	    {
		int i2 = ii+ndi[nn];
		int j2 = jj+ndj[nn];
		if(i2<0) continue;
		if(j2<0) continue;
		if(i2>=nrow) continue;
		if(j2>=ncol) continue;

		int jjj = i2*ncol+j2;

		pp(iii, jjj) =1;
	    }
	}
    }

    //std::cout<<"pp="<<std::cout<<row(pp, 2)<<std::endl;
    for(umf_sparse_matrix::iterator1 it1=pp.begin1(); it1 != pp.end1(); ++it1)
    {
	double sum = std::accumulate(it1.begin(), it1.end(), 0.0f);
	//std::cout<<"sum="<<sum<<std::endl;
	std::for_each(it1.begin(), it1.end(), _1 /= sum);
    }

}

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

void get_qp_block(vector<int> const& mapvec,
		  umf_sparse_matrix const& qp,
		  umf_sparse_matrix & qpnn,
		  matrix<double>& qpnt)
{
    using namespace boost::lambda;
    int num_n = std::count_if(mapvec.begin(), mapvec.end(), _1!=1);
    int num_t = std::count(mapvec.begin(), mapvec.end(), 1);

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


void test_lmdp()
{

    matrix<int> map;
    create_map(map);

    vector<double> q;
    vector<int> mapvec;
    double scale = 5;
    double infty = 1e6;
    real_timer_t timer;
    create_qvec(map, mapvec, q, scale, infty);
    std::cout<<timer.elapsed()<<": done create_qvec"<<std::endl;

    //std::cout<<q<<std::endl;
    //cvpr::array2d_print(std::cout, map);

    umf_sparse_matrix pps;
    create_passive_prob(map, pps);
    std::cout<<timer.elapsed()<<": done create_passive_prob"<<std::endl;

    //cvpr::array2d_print(std::cout, ppassive);
    umf_sparse_matrix qp;
    compute_qp(q, pps, qp);
    std::cout<<timer.elapsed()<<": done compute_qp"<<std::endl;

    umf_sparse_matrix qpnn;
    matrix<double> qpnt;
    get_qp_block(mapvec, qp, qpnn, qpnt);
    std::cout<<timer.elapsed()<<": done get_qp_block"<<std::endl;

    int num_t = std::count(mapvec.begin(), mapvec.end(), 1);
    vector<double> zt = scalar_vector<double>(num_t, 1.0f);

    vector<double> qpz0 = prod(qpnt, zt);


    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;

    umf_sparse_matrix A(-qpnn);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) += 1.0f;
    }
    std::cout<<timer.elapsed()<<": done fill A"<<std::endl;

    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    vector<double> zn(qpz0.size());
    umf::solve(A, zn, qpz0, nume);  
    std::cout<<timer.elapsed()<<": done solve A"<<std::endl;

    std::cout<<zn<<std::endl;

    matrix<double> v(map.size1(), map.size2());
    v(0, 0) = 0;

    int ll=0;
    for(int ii=0; ii<v.size1(); ++ii)
    {
	for(int jj=0; jj<v.size2(); ++jj)
	{
	    if(ii==0 && jj==0) continue;

	    if(zn(ll)==0) v(ii, jj) = 1e6;
	    else v(ii, jj) = -std::log(zn(ll))/scale;
	    ++ll;
	}
    }

    //std::cout<<v<<std::endl;
    cvpr::array2d_print(std::cout, v);

}

int main(int argc, char* argv[])
{
    real_timer_t timer;
    test_lmdp();
    std::cout<<"time: "<<timer.elapsed()<<"ms."<<std::endl;
    return 0;
}

