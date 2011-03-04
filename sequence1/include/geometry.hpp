#ifndef __GEOMETRY__HPP__INCLUDED__
#define __GEOMETRY__HPP__INCLUDED__

#include <boost/numeric/ublas/lu.hpp>
#include <boost/lambda/lambda.hpp>

#include "cvpr_stub.hpp"

BEGIN_NAMESPACE_CVPR

template <typename Float>
void apply_homography(boost::numeric::ublas::matrix<Float> const& A,
		      boost::numeric::ublas::vector<Float> const& prev_x, 
		      boost::numeric::ublas::vector<Float> const& prev_y, 
		      boost::numeric::ublas::vector<Float> & new_x, 
		      boost::numeric::ublas::vector<Float> & new_y)

{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;
    int num = prev_x.size();
    matrix<Float> tmp(3, num);
    row(tmp, 0) = prev_x;
    row(tmp, 1) = prev_y;
    row(tmp, 2) = scalar_vector<Float>(num, 1);

    matrix<Float> vert2drx(prod(A, tmp));
    vector<Float> vtmp(row(vert2drx, 2));
    std::for_each(vtmp.begin(), vtmp.end(), _1=1/_1);
    new_x = element_prod(row(vert2drx, 0), vtmp);
    new_y = element_prod(row(vert2drx, 1), vtmp);
}

template <typename Float>
void estimate_homography(boost::numeric::ublas::vector<Float> const& prev_x, 
			 boost::numeric::ublas::vector<Float> const& prev_y, 
			 boost::numeric::ublas::vector<Float> const& new_x, 
			 boost::numeric::ublas::vector<Float> const& new_y, 
			 boost::numeric::ublas::matrix<Float>& A)
{
    using namespace boost::numeric::ublas;
    int num = prev_x.size();

    int num_all=2*num;
    matrix<Float> X=zero_matrix<Float>(num_all, 8);
    matrix<Float> a=zero_matrix<Float>(num_all, 1);

    int ia=0;
    for(int nn=0; nn<num; ++nn)
    {
	X(ia, 0) = prev_x(nn);
	X(ia, 1) = prev_y(nn);
	X(ia, 2) = 1;
	X(ia, 3) = 0;
	X(ia, 4) = 0;
	X(ia, 5) = 0;
	X(ia, 6) = -prev_x(nn)*new_x(nn);
	X(ia, 7) = -prev_y(nn)*new_x(nn);
	a(ia, 0) = new_x(nn);
	ia ++;

	X(ia, 0) = 0;
	X(ia, 1) = 0;
	X(ia, 2) = 0;
	X(ia, 3) = prev_x(nn);
	X(ia, 4) = prev_y(nn);
	X(ia, 5) = 1;
	X(ia, 6) = -prev_x(nn)*new_y(nn);
	X(ia, 7) = -prev_y(nn)*new_y(nn);
	a(ia, 0) = new_y(nn);
	ia ++;
    }

    matrix<Float> XTX = prod(trans (X), X);

    matrix<Float> XTx = prod(trans (X), a);

    //permutation_matrix<Float> P(8);
    permutation_matrix<int> P(8);

    lu_factorize(XTX, P);	
    lu_substitute(XTX, P, XTx); 
	
    A = matrix<Float>(3, 3);

    A(0, 0) = XTx(0, 0);
    A(0, 1) = XTx(1, 0);
    A(0, 2) = XTx(2, 0);

    A(1, 0) = XTx(3, 0);
    A(1, 1) = XTx(4, 0);
    A(1, 2) = XTx(5, 0);

    A(2, 0) = XTx(6, 0);
    A(2, 1) = XTx(7, 0);
    A(2, 2) = 1;

}

void estimate_affine(const boost::numeric::ublas::vector<int>& prev_x, 
		     const boost::numeric::ublas::vector<int>& prev_y, 
		     boost::numeric::ublas::vector<int>& est_x, 
		     boost::numeric::ublas::vector<int>& est_y, 
		     boost::numeric::ublas::matrix<float>& A, 
		     boost::numeric::ublas::vector<float>& dis)
{
    using namespace boost::numeric::ublas;
    int num = prev_x.size();

    int num_all=2*num;

    matrix<double> X=zero_matrix<double>(num_all, 6);
    matrix<double> x_new=zero_matrix<double>(num_all, 1);

    int ia=0;
    for(int nn=0; nn<num; ++nn)
    {
	X(ia, 0) = prev_x(nn);
	X(ia, 1) = prev_y(nn);
	X(ia, 2) = 0;
	X(ia, 3) = 0;
	X(ia, 4) = 1;
	X(ia, 5) = 0;
	x_new(ia, 0) = est_x(nn);
	ia ++;
	X(ia, 0) = 0;
	X(ia, 1) = 0;
	X(ia, 2) = prev_x(nn);
	X(ia, 3) = prev_y(nn);
	X(ia, 4) = 0;
	X(ia, 5) = 1;
	x_new(ia, 0) = est_y(nn);
	ia ++;
    }

    matrix<double> XTX=//zero_matrix<float>(6, 6);
	prod(trans (X), X);

    matrix<double> XTx=//zero_matrix<float>(6, 1);
	prod(trans (X), x_new);

    permutation_matrix<int> P(6);


    lu_factorize(XTX,P);	
    lu_substitute(XTX, P, XTx); 
	
    A = matrix<float>(2, 2);
    dis = vector<float>(2);

    A(0, 0) = XTx(0, 0);
    A(0, 1) = XTx(1, 0);
    A(1, 0) = XTx(2, 0);
    A(1, 1) = XTx(3, 0);

    dis(0) = XTx(4, 0);
    dis(1) = XTx(5, 0);

}

END_NAMESPACE_CVPR

#endif
