#ifndef __UBLAS__RANDOM__HPP__INCLUDED__
#define __UBLAS__RANDOM__HPP__INCLUDED__

#include <algorithm>

#include <boost/random.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "cvpr_stub.hpp"

BEGIN_NAMESPACE_CVPR

//todo: fix the problem for std::generator, copy constructor? ref-counting?

// uniform integer distribution
template <class Int>
struct randu_int_t
{
    
    randu_int_t(Int l, Int u) {
	dist = new boost::uniform_int<Int>(l, u);
	rand = new boost::variate_generator<boost::mt19937&, boost::uniform_int<Int> >(rng, *dist);
	//if(0==rand) std::cout<<"rand null"<<std::endl;
    }
    ~randu_int_t() {
	delete rand;
	delete dist;
    }
    Int operator()() {
	//if(0==rand) std::cout<<"rand null"<<std::endl;
	return (*rand)();
    }
    void seed(int s) {
	rng.seed(s);
    }
    void seed() {
	seed(std::time(NULL));
    }
    boost::mt19937 rng;          
    boost::uniform_int<Int> *dist;
    boost::variate_generator<boost::mt19937&, boost::uniform_int<Int> >  *rand;
};


// gaussian distribution
template <class Float>
struct randn_t
{

    randn_t(Float mean, Float sigma)	{
	dist = new boost::normal_distribution<Float>(mean, sigma);
	randn = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<Float> >(rng, *dist);
    }

    ~randn_t()	{
	delete randn;
	delete dist;
    }

    Float operator()()	{
	return (*randn)();
    }
    void seed(int s) {
	rng.seed(s);
    }
    void seed() {
	seed(std::time(NULL));
    }

    boost::mt19937 rng;    
    boost::normal_distribution<Float> *dist;
    boost::variate_generator<boost::mt19937&,   boost::normal_distribution<Float> > *randn; 

};

// uniform distribution
template <class Float>
struct randu_t
{

    randu_t(Float l, Float u)	{
	dist = new boost::uniform_real<Float>(l, u);
	randu = new boost::variate_generator<boost::mt19937&, boost::uniform_real<Float> >(rng, *dist);
    }

    ~randu_t()	{
	delete randu;
	delete dist;

    }

    Float operator()()	{
	return (*randu)();
    }

    void seed(int s) {
	rng.seed(s);
    }
    void seed() {
	seed(std::time(NULL));
    }


    boost::mt19937 rng;    
    boost::uniform_real<Float> *dist;
    boost::variate_generator<boost::mt19937&, boost::uniform_real<Float> > *randu; 

};



template <class Float, class Rand>
boost::numeric::ublas::matrix<Float> rand_matrix(int size1, int size2, Rand& rand)
{
    boost::numeric::ublas::matrix<Float> mat(size1, size2);
    for(int ii=0; ii<mat.size1(); ++ii)
    {
	for(int jj=0; jj<mat.size2(); ++jj)
	{
	    mat(ii, jj) = rand();
	}
    }
    return mat;
}

template <class Float, class Rand>
boost::numeric::ublas::vector<Float> rand_vector(int size,  Rand& rand)
{
    boost::numeric::ublas::vector<Float> vec(size);
    for(int ii=0; ii<vec.size(); ++ii)
    {
	vec(ii) = rand();
    }
    return vec;
}

END_NAMESPACE_CVPR

#endif
