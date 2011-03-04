#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "ublas_cimg.hpp"
#include "ublas_random.hpp"

#include "bgseg.hpp"
#include "labelmap.hpp"
#include "cvpr_array_traits.hpp"

using namespace cimg_library;

namespace mpi = boost::mpi;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;

using namespace cvpr;


using namespace bgseg;


#include "input_output.hpp"


void test_sort_with_index()
{
    using namespace boost::lambda;

    typedef float Float;
    randn_t<Float> rand(1,1); rand.seed();
    matrix<Float> X = rand_matrix<Float>(2, 3, rand);
    typedef std::pair<Float, int> Pair;

    vector<Pair> x2(X.data().size());
    std::transform(X.data().begin(), X.data().end(), boost::counting_iterator<int>(0),
		  x2.begin(), bind(std::make_pair<Float, int>, _1, _2));
    std::sort(x2.begin(), x2.end(), bind(&Pair::first, _1)<
	      bind(&Pair::first, _2));

    vector<Float> x_sorted(X.data().size());
    vector<int> idx_sorted(X.data().size());

    std::transform(x2.begin(), x2.end(), x_sorted.begin(), 
		   bind(&Pair::first, _1) );
    std::transform(x2.begin(), x2.end(), idx_sorted.begin(), 
		   bind(&Pair::second, _1) );
    std::cout<<X<<std::endl;
    std::cout<<x_sorted<<std::endl;
    std::cout<<idx_sorted<<std::endl;

}

void test_pointer_size()
{
    std::cout<<"sizeof(int*)="<<sizeof(int*)<<std::endl;
}

struct test_mem_t
{
    int ii;
    std::string a;
    int c[3];
    int& getc() { return c[1]; }
    int getc() const { return c[1]; }
};


void test_member_array1d()
{
    std::vector<test_mem_t> v;
    for(int ii=0; ii<10; ++ii)
    {
	test_mem_t m;
	m.ii = ii;
	m.a = boost::lexical_cast<std::string>(ii)+"a";
	v.push_back(m);
    }
    member_array1d<std::vector<test_mem_t>, int test_mem_t::*>
	ma(v, &test_mem_t::ii);
    member_array1d<std::vector<test_mem_t>, std::string test_mem_t::*>
	mb(v, &test_mem_t::a);
    std::cout<<"ma="<<std::endl;
    for(int ii=0; ii<ma.size(); ++ii)
    {
	std::cout<<"\t"<<ma[ii]<<std::endl;
    }

    std::cout<<"mb="<<std::endl;
    for(int ii=0; ii<mb.size(); ++ii)
    {
	std::cout<<"\t"<<mb[ii]<<std::endl;
    }
    ublas::vector<int> d;
    array1d_copy(ma, d);

    std::cout<<"d="<<std::endl;
    for(int ii=0; ii<d.size(); ++ii)
    {
	std::cout<<"\t"<<d[ii]<<std::endl;
    }


}

int main(int argc, char * argv[])
{
    test_member_array1d();
    //test_bbsegj();

    return 0;
}
