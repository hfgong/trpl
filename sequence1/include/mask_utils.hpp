#ifndef __MASK__UTILS__HPP__INCLUDED__
#define __MASK__UTILS__HPP__INCLUDED__

#include "cvpr_array_traits.hpp"
#include "cvpr_stub.hpp"
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/type_traits.hpp>

BEGIN_NAMESPACE_CVPR


template <class Vec, typename Int>
bool point_in_polygon(const Vec& poly_x,
		      const Vec& poly_y, Int x, Int y)
{
    typedef typename array1d_traits<Vec>::value_type value_type;

    typedef array1d_traits<Vec> tr;

    std::size_t counter = 0;
    std::size_t i;
    double xinters;
    std::size_t N = (int)poly_x.size();
    value_type p1x, p1y, p2x, p2y;

    p1x = tr::ref(poly_x, 0);
    p1y = tr::ref(poly_y, 0);

    for (i=1; i<=N; i++) 
    {
        p2x = tr::ref(poly_x, (i) % (N));
	p2y = tr::ref(poly_y, (i) % (N));

	if (y > std::min<value_type>(p1y, p2y)) 
        {
	    if (y <= std::max<value_type>(p1y, p2y)) 
            {
		if (x <= std::max<value_type>(p1x, p2x)) 
                {
                    if (p1y != p2y) 
                    {
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x;
                        if (p1x == p2x || x <= xinters)
                            counter++;
                    }
                }
            }
        }
        p1x = p2x;
	p1y = p2y;
    }

    if (counter % 2 == 0)
        return false;
    else
        return true;
}

template <class Mat, class Vec>
void mask_from_polygon_internal(Mat& mask, std::size_t s1, std::size_t s2, 
				Vec const& poly_x,
				Vec const& poly_y, boost::true_type)
{
    typedef array2d_traits<Mat> trm;
    trm::change_size(mask, s1, s2);

    std::size_t x, y, x0, y0, x1, y1;

    x0 = poly_x(0);
    x1 = poly_x(0);
    y0 = poly_y(0);
    y1 = poly_y(0);

    for(std::size_t nn=1; nn<poly_x.size(); ++nn)
    {
	if (poly_x(nn) < x0)         x0 = poly_x(nn);
	if (poly_x(nn) > x1)         x1 = poly_x(nn);

	if (poly_y(nn) < y0)         y0 = poly_y(nn);
	if (poly_y(nn) > y1)         y1 = poly_y(nn);
    }

    for (y = 0; y < s1; ++y)
    {
	for(x = 0; x < s2; ++x)
	{
	    if( (y < y0) || (y > y1) ||
		(x < x0) || (x > x1) ) 
	    {
		trm::ref(mask, y, x) = 0;
		continue;
	    }
	    if(point_in_polygon(poly_x, poly_y, x, y))
	    {
		trm::ref(mask, y, x) = 1;				
	    }
	    else
	    {
		trm::ref(mask, y, x) = 0;
	    }
	}		
    }

}

template <class Mat, class Vec>
void mask_from_polygon_internal(Mat& mask, std::size_t s1, std::size_t s2, 
				Vec const& poly_x,
				Vec const& poly_y, boost::false_type)
{
    using namespace boost::lambda;

    boost::numeric::ublas::vector<std::size_t> px, py;
    array1d_transform(poly_x, px, ll_static_cast<std::size_t>(_1+0.5));
    array1d_transform(poly_y, py, ll_static_cast<std::size_t>(_1+0.5));
    mask_from_polygon_internal(mask, s1, s2, px, py, boost::true_type());
}

template <class Mat, class Vec>
void mask_from_polygon(Mat& mask, std::size_t s1, std::size_t s2, 
		       Vec const& poly_x,
		       Vec const& poly_y)
{
    typename boost::is_integral<typename Vec::value_type>::type tag;
    mask_from_polygon_internal(mask, s1, s2, poly_x, poly_y, tag);

}


namespace {
const int neighbor_dx[4]={1, 0, -1, 0};
const int neighbor_dy[4]={0, 1, 0, -1};
}

template <class Mat>
void dilate_mask(Mat& mask)
{
    typedef array2d_traits<Mat> trm;
    std::size_t s1 = trm::size1(mask);
    std::size_t s2 = trm::size2(mask);
    Mat tmask;
    trm::resize(tmask, s1, s2);
    array2d_copy(mask, tmask);

    for(std::size_t yy=1; yy+1<s1; ++yy)
    {
	for(std::size_t xx=1; xx+1<s2; ++xx)
	{
	    if( trm::ref(mask, yy, xx) )	continue;
	    for(int nn=0; nn<4; ++nn)
	    {
		int dx = neighbor_dx[nn];
		int dy = neighbor_dy[nn];
		if(trm::ref(mask, yy+dy, xx+dx))
		{					
		    trm::ref(tmask, yy, xx) = 1;
		    break;
		}
	    }
	}
    }
    array2d_copy(tmask, mask);
}


template <class Mat>
void erode_mask(Mat& mask)
{
    typedef array2d_traits<Mat> trm;
    std::size_t s1 = trm::size1(mask);
    std::size_t s2 = trm::size2(mask);
    Mat tmask;
    trm::resize(tmask, s1, s2);
    array2d_copy(mask, tmask);


    for(std::size_t yy=1; yy+1<s1; ++yy)
    {
	for(std::size_t xx=1; xx+1<s2; ++xx)
	{
	    if(! trm::ref(mask, yy, xx) )	continue;
	    int ok = 1;
	    for(int nn=0; nn<4; ++nn)
	    {
		int dx = neighbor_dx[nn];
		int dy = neighbor_dy[nn];
		if(!trm::ref(mask, yy+dy, xx+dx))
		{
		    ok = 0;
		    break;
		}
	    }
	    tmask(yy, xx) = ok;
			
	}
    }
    array2d_copy(tmask, mask);

}

END_NAMESPACE_CVPR

#endif
