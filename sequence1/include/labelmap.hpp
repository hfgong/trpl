#ifndef __LABELMAP__HPP__INCLUDED__
#define __LABELMAP__HPP__INCLUDED__

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <boost/lambda/lambda.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include "cvpr_stub.hpp"
#include "real_timer.hpp"

BEGIN_NAMESPACE_CVPR

template <class Mat>
int labelmap_connected_components(const Mat& labelmap, Mat& compmap)
{
    typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS> Graph;
    //real_timer_t timer;

    Graph G(labelmap.data().size());

    for(int ii=0; ii+1<labelmap.size1(); ++ii)
    {
        for(int jj=0; jj<labelmap.size2(); ++jj)
        {
            int offset = jj+ii*labelmap.size2();
            if(labelmap(ii, jj) == labelmap(ii+1, jj))
                add_edge(offset, jj+(ii+1)*labelmap.size2(), G);
        }
    }
    for(int ii=0; ii<labelmap.size1(); ++ii)
    {
        for(int jj=0; jj+1<labelmap.size2(); ++jj)
        {
            int offset = jj+ii*labelmap.size2();
            if(labelmap(ii, jj) == labelmap(ii, jj+1))
                add_edge(offset, jj+1+ii*labelmap.size2(), G);
	}
    }
    //Graph construction consumes more time than the CC function.
    // I will make an adaptor when I have time.
    //std::cout<<"graph construction timer:"<<timer.elapsed()<<std::endl;
    return boost::connected_components(G, compmap.data().begin());

}

template <class Mat, class Vec>
void labelmap_count_components(const Mat& compmap, int ncomp, Vec& count)
{
    using namespace boost::lambda;
    if(count.size()==0) count = Vec(ncomp);
    std::for_each(count.begin(), count.end(), _1=0);
    for(int ii=0; ii<compmap.size1(); ++ii)
    {
	for(int jj=0; jj<compmap.size2(); ++jj)
	{
	    int nn = compmap(ii, jj);
	    if(nn>=0 && nn<ncomp)  count(nn)++;
	}
    }
}

template <typename graph_t, typename Int>
int connected_components(graph_t & g, std::vector<Int>& component) {
    component.reserve(num_vertices(g));
    return boost::connected_components(g, &component[0]);
}

template <typename graph_t, typename Int>
int connected_components(graph_t & g, boost::numeric::ublas::vector<Int>& component) {
    component = boost::numeric::ublas::vector<Int>(num_vertices(g));
    return boost::connected_components(g, &component[0]);
}


END_NAMESPACE_CVPR

#endif
