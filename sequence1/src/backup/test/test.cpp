#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

namespace mpi = boost::mpi;

int xmain(int argc, char* argv[]) 
{
  mpi::environment env(argc, argv);
  mpi::communicator world;
  std::cout << "I am process " << world.rank() << " of " << world.size()
            << "." << std::endl;
  return 0;
}

#include <limits>

void test_lim()
{
    std::cout<<std::numeric_limits<double>::epsilon()<<std::endl;
}


namespace cvpr{

    namespace detail{
	using namespace boost::numeric::ublas;
	template <class T>
	T f(T a)
	{
	    return a;
	}
	template <class T>
	T g(T a, T b)
	{
	    return b+f(a);
	}
	template <class T>
	T g(T a, T b, T c)
	{
	    return b+f(a)/c;
	}
    }

    template <class T>
    T f(T a)
    {
	return detail::f(a);
    }

    template <class T>
    T g(T a, T b)
    {
	return detail::g(a, b);
    }
    template <class T>
    T g(T a, T b, T c)
    {
	return detail::g(a, b, c);
    }

    void test()
    {
	g(1,2, 3);
    }

    detail::vector<float> v;
}

using namespace cvpr;

int main(int argc, char* argv[])
{
    test();
    int a = 100;

    int const * p1;
    p1=&a;
    //*p1 = 10;

    int * const p2=&a;
    //p2 = &a;
    *p2 = 10;

    int const * const p3=&a;
    //p3 = &a;
    //*p3 = 10;

    return 0; 
}
