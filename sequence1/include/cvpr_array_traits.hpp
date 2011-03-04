#ifndef __CVPR__ARRAY__TRAITS__HPP__INCLUDED__
#define __CVPR__ARRAY__TRAITS__HPP__INCLUDED__

/*
A minimal array traits class, for element access, size determination only.
Summarize CImg, ublas, multi_array, IPLImage etc into a common framework.

The dimensionality assumption is like that
1d, x                      --- vector, array, image column, row
2d, y, x                   --- greylevel image or matrix
3d, channel, y, x          --- color image
4d, time, channel, y, x    --- color video
Thus the access of CImg is adjusted accordingly.

The default templates are for boost::ublas::vector/matrix combinations.
1d, vector<T>, also works for std::vector
2d, matrix<T>
3d, vector<matrix<T> >
4d, matrix<matrix<T> >

Specializations have made for CImg and boost::multi_array.
CImg<T>
multi_array<T, 1>
multi_array<T, 2>
multi_array<T, 3>
multi_array<T, 4>

Todo:
Specialization for std::vector<std::vector<T> > etc.

 */
#include <boost/multi_array.hpp>
#include <CImg.h>
#include "cvpr_stub.hpp"
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>


BEGIN_NAMESPACE_CVPR

//--------------------------------------------------------------------------------
// array1d_traits
//--------------------------------------------------------------------------------

template <class Mat>
struct array1d_traits
{
    typedef typename Mat::value_type value_type;
    typedef Mat array_type;
    static inline value_type ref(Mat const& mat, int x)  {
	return mat[x];
    }
    static inline value_type& ref(Mat & mat, int x) {
	return mat[x];
    }
    static inline Mat create(int s1) {
	return Mat(s1);
    }
    static inline void change_size(Mat& mat, int s1) {
	mat = Mat(s1);
    }
    static inline int size(Mat const& mat) {
	return mat.size();
    }
    static inline int size1(Mat const& mat) {
	return size();
    }
private:
    array1d_traits();
};

template <class T>
struct array1d_traits<boost::multi_array<T, 1> >
{
    typedef T value_type;
    typedef boost::multi_array<T, 1> array_type;

    static inline value_type ref(array_type const& mat, int x)  {
	return mat[x];
    }
    static inline value_type& ref(array_type & mat, int x) {
	return mat[x];
    }
    static inline array_type create(int s1) {
	return array_type(boost::extents[s1]);
    }
    static inline void change_size(array_type& m, int s1) {
	m.resize(boost::extents[s1]);
    }
    static inline int size(array_type const& mat) {
	return mat.shape()[0];
    }
    static inline int size1(array_type const& mat) {
	return size();
    }
private:
    array1d_traits();
};

template <class T>
struct array1d_traits<cimg_library::CImg<T> >
{
    typedef T value_type;
    typedef cimg_library::CImg<T> array_type;

    static inline value_type ref(array_type const& mat, int x)  {
	return mat.atXYZC(x, 0, 0, 0);
    }
    static inline value_type& ref(array_type & mat, int x) {
	return mat.atXYZC(x, 0, 0, 0);
    }
    static inline array_type create(int s1) {
	return array_type(s1);
    }
    static inline void change_size(array_type & m, int s1) {
	m = array_type( s1);
    }
    static inline int size(array_type const& mat) {
	return mat.width();
    }
    static inline int size1(array_type const& mat) {
	return size();
    }
private:
    array1d_traits();
};

template <class M1, class M2>
void array1d_copy(M1 const & m1, M2 & m2)
{
    typedef array1d_traits<M1> tr1;
    typedef array1d_traits<M2> tr2;
    if(tr1::size(m1) != tr2::size(m2))
    {
	tr2::change_size(m2, tr1::size(m1));
    }
    for(int xx=0; xx<tr1::size(m1); ++xx)
    {
	tr2::ref(m2, xx) = tr1::ref(m1, xx);
    }

}

template <class M1, class M2, class F>
void array1d_transform(M1 const & m1, M2 & m2, F f)
{
    typedef array1d_traits<M1> tr1;
    typedef array1d_traits<M2> tr2;
    if(tr1::size(m1) != tr2::size(m2))
    {
	tr2::change_size(m2, tr1::size(m1));
    }
    for(int xx=0; xx<tr1::size(m1); ++xx)
    {
	tr2::ref(m2, xx) = f(tr1::ref(m1, xx));
    }

}

template <class M1, class M2, class M3>
bool array1d_cat(M1 const & m1, M2 const & m2, M3 & m3)
{
    typedef array1d_traits<M1> tr1;
    typedef array1d_traits<M2> tr2;
    typedef array1d_traits<M3> tr3;
    if(tr1::size(m1) + tr2::size(m2) != tr3::size(m3) )
    {
	tr3::change_size(m3, tr1::size(m1)+tr2::size(m2) );
    }

    for(int xx=0; xx<tr1::size(m1); ++xx)
    {
	tr3::ref(m3, xx) = tr1::ref(m1, xx);
    }


    for(int xx=0; xx<tr2::size(m2); ++xx)
    {
	tr3::ref(m3, xx+tr1::size(m1)) = tr1::ref(m2, xx);
    }

}

template <class ST, class M>
void array1d_print(ST& cout, M const& m)
{
    typedef array1d_traits<M> tr;

    for(std::size_t xx=0; xx<tr::size(m); ++xx)
    {
	cout<< tr::ref(m, xx) <<" ";
    }
    cout<<std::endl;
}


//--------------------------------------------------------------------------------
// array2d_traits
//--------------------------------------------------------------------------------


template <class Mat>
struct array2d_traits
{
    typedef typename Mat::value_type value_type;
    typedef Mat array_type;
    static inline value_type ref(Mat const& mat, int y, int x)  {
	return mat(y, x);
    }
    static inline value_type& ref(Mat & mat, int y, int x) {
	return mat(y, x);
    }
    static inline Mat create(int s1, int s2) {
	return Mat(s1, s2);
    }
    static inline void change_size(Mat& mat, int s1, int s2) {
	mat = Mat(s1, s2);
    }
    static inline int size1(Mat const& mat) {
	return mat.size1();
    }
    static inline int size2(Mat const& mat) {
	return mat.size2();
    }
private:
    array2d_traits();
};

template <class T>
struct array2d_traits<boost::multi_array<T, 2> >
{
    typedef T value_type;
    typedef boost::multi_array<T, 2> array_type;

    static inline value_type ref(array_type const& mat, int y, int x)  {
	return mat[y][x];
    }
    static inline value_type& ref(array_type & mat, int y, int x) {
	return mat[y][x];
    }
    static inline array_type create(int s1, int s2) {
	return array_type(boost::extents[s1][s2]);
    }
    static inline void change_size(array_type& m, int s1, int s2) {
	m.resize(boost::extents[s1][s2]);
    }
    static inline int size1(array_type const& mat) {
	return mat.shape()[0];
    }
    static inline int size2(array_type const& mat) {
	return mat.shape()[1];
    }
private:
    array2d_traits();
};

template <class T>
struct array2d_traits<cimg_library::CImg<T> >
{
    typedef T value_type;
    typedef cimg_library::CImg<T> array_type;

    static inline value_type ref(array_type const& mat, int y, int x)  {
	return *(mat.data(x, y, 0, 0));
    }
    static inline value_type& ref(array_type & mat, int y, int x) {
	return *(mat.data(x, y, 0, 0));
    }
    static inline array_type create(int s1, int s2) {
	return array_type(s2, s1);
    }
    static inline void change_size(array_type & m, int s1, int s2) {
	m = array_type(s2, s1);
    }
    static inline int size1(array_type const& mat) {
	return mat.height();
    }
    static inline int size2(array_type const& mat) {
	return mat.width();
    }
private:
    array2d_traits();
};

template <class M1, class M2>
void array2d_copy(M1 const & m1, M2 & m2)
{
    typedef array2d_traits<M1> tr1;
    typedef array2d_traits<M2> tr2;
    if(tr1::size1(m1) != tr2::size1(m2) || tr1::size2(m1) != tr2::size2(m2) )
    {
	tr2::change_size(m2, tr1::size1(m1), tr1::size2(m1));
    }
    for(int yy=0; yy<tr1::size1(m1); ++yy)
    {
	for(int xx=0; xx<tr1::size2(m1); ++xx)
	{
	    tr2::ref(m2, yy, xx) = tr1::ref(m1, yy, xx);
	}
    }
}

template <class M1, class M2, class M3>
bool array2d_cat1(M1 const & m1, M2 const & m2, M3 & m3)
{
    typedef array2d_traits<M1> tr1;
    typedef array2d_traits<M2> tr2;
    typedef array2d_traits<M3> tr3;
    if( tr1::size2(m1) != tr2::size2(m2))
    {
	return false;
    }

    if(tr1::size1(m1) + tr2::size1(m2) != tr3::size1(m3) ||
       tr1::size2(m1) != tr3::size2(m3) )
    {
	tr3::change_size(m3, tr1::size1(m1)+tr2::size1(m2), 
			 tr1::size2(m1));
    }
    for(int yy=0; yy<tr1::size1(m1); ++yy)
    {
	for(int xx=0; xx<tr1::size2(m1); ++xx)
	{
	    tr3::ref(m3, yy, xx) = tr1::ref(m1, yy, xx);
	}
    }


    for(int yy=0; yy<tr2::size1(m2); ++yy)
    {
	for(int xx=0; xx<tr2::size2(m2); ++xx)
	{
	    tr3::ref(m3, yy+tr1::size1(m1), xx) = tr1::ref(m2, yy, xx);
	}
    }

}

template <class M1, class M2, class M3>
bool array2d_cat2(M1 const & m1, M2 const& m2, M3& m3)
{
    typedef array2d_traits<M1> tr1;
    typedef array2d_traits<M2> tr2;
    typedef array2d_traits<M3> tr3;

    if( tr1::size1(m1) != tr2::size1(m2) )
    {
	return false;
    }

    if(tr1::size2(m1) + tr2::size2(m2) != tr3::size2(m3) ||
       tr1::size1(m1) != tr3::size1(m3)  )
    {
	tr3::change_size(m3, tr1::size1(m1),
			 tr1::size2(m1)+tr2::size2(m2) );
    }

    for(int yy=0; yy<tr1::size1(m1); ++yy)
    {
	for(int xx=0; xx<tr1::size2(m1); ++xx)
	{
	    tr3::ref(m3, yy, xx) = tr1::ref(m1, yy, xx);
	}
    }



    for(int yy=0; yy<tr2::size1(m2); ++yy)
    {
	for(int xx=0; xx<tr2::size2(m2); ++xx)
	{
	    tr3::ref(m3, yy, xx+tr1::size2(m1)) = tr1::ref(m2, yy, xx);
	}
    }

}

template <class M1, class M2, class F>
bool array2d_transform(M1 const & m1, M2 & m2, F f)
{
    typedef array2d_traits<M1> tr1;
    typedef array2d_traits<M2> tr2;

    if(tr1::size2(m1) !=  tr2::size2(m2) ||
       tr1::size1(m1) != tr2::size1(m2)  )
    {
	tr2::change_size(m2, tr1::size1(m1),  tr1::size2(m1) );
    }

    for(int yy=0; yy<tr1::size1(m1); ++yy)
    {
	for(int xx=0; xx<tr1::size2(m1); ++xx)
	{
	    tr2::ref(m2, yy, xx) = f(tr1::ref(m1, yy, xx));
	}
    }

}

template <class M>
void array2d_max(M const& m, std::size_t& i1, std::size_t& i2)
{
    typedef array2d_traits<M> tr;
    typename tr::value_type v = tr::ref(m, 0, 0);

    i1 = 0; i2 = 0;
    for(std::size_t yy=0; yy<tr::size1(m); ++yy)
    {
	for(std::size_t xx=0; xx<tr::size2(m); ++xx)
	{
	    if( tr::ref(m, yy, xx)> v)
	    {
		v = tr::ref(m, yy, xx);
		i1 = yy;
		i2 = xx;
	    }
	}
    }

}



template <class ST, class M>
void array2d_print(ST& cout, M const& m)
{
    typedef array2d_traits<M> tr;

    for(std::size_t yy=0; yy<tr::size1(m); ++yy)
    {
	for(std::size_t xx=0; xx<tr::size2(m); ++xx)
	{
	    cout<< tr::ref(m, yy, xx) <<" ";
	}
	cout<<std::endl;
    }

}



//--------------------------------------------------------------------------------
// array3d_traits
//--------------------------------------------------------------------------------
template <class Mat>
struct array3d_traits
{
    typedef typename Mat::value_type::value_type value_type;
    typedef Mat array_type;
    typedef typename Mat::value_type sub_array_type;
    static inline value_type ref(Mat const& mat, int c, int y, int x)  {
	return mat(c)(y, x);
    }
    static inline value_type& ref(Mat & mat, int c, int y, int x) {
	return mat(c)(y, x);
    }
    static inline Mat create(int s1, int s2, int s3) {
	Mat m(s1);
	for(int cc=0; cc<s1; ++cc)
	    m(cc) = sub_array_type(s2, s3);
	return m;
    }
    static inline void change_size(Mat & m, int s1, int s2, int s3) {
	m = create(s1, s2, s3);
    }
    static inline int size1(Mat const& mat) {
	return mat.size();
    }
    static inline int size2(Mat const& mat) {
	if(mat.size()>0)
	    return mat(0).size1();
	else return 0;
    }
    static inline int size3(Mat const& mat) {
	if(mat.size()>0)
	    return mat(0).size2();
	else return 0;
    }
private:
    array3d_traits();
};

template <class T>
struct array3d_traits<boost::multi_array<T, 3> >
{
    typedef T value_type;
    typedef boost::multi_array<T, 3> array_type;

    static inline value_type ref(array_type const& mat, int c, int y, int x)  {
	return mat[c][y][x];
    }
    static inline value_type& ref(array_type & mat, int c, int y, int x) {
	return mat[c][y][x];
    }
    static inline array_type create(int s1, int s2, int s3) {
	return array_type(boost::extents[s1][s2][s3]);
    }
    static inline void change_size(array_type& m, int s1, int s2, int s3) {
	m.resize(boost::extents[s1][s2][s3]);
    }
    static inline int size1(array_type const& mat) {
	return mat.shape()[0];
    }
    static inline int size2(array_type const& mat) {
	return mat.shape()[1];
    }
    static inline int size3(array_type const& mat) {
	return mat.shape()[2];
    }
private:
    array3d_traits();
};

template <class T>
struct array3d_traits<cimg_library::CImg<T> >
{
    typedef T value_type;
    typedef cimg_library::CImg<T> array_type;

    static inline value_type ref(array_type const& mat, int c, int y, int x)  {
	return *(mat.data(x, y, 0, c));
    }
    static inline value_type& ref(array_type & mat, int c, int y, int x) {
	return *(mat.data(x, y, 0, c));
    }
    static inline array_type create(int s1, int s2, int s3) {
	return array_type(s3, s2, 1, s1);
    }
    static inline void change_size(array_type& m, int s1, int s2, int s3) {
	m = create(s1, s2, s3);
    }
    static inline int size1(array_type const& mat) {
	return mat.spectrum();
    }
    static inline int size2(array_type const& mat) {
	return mat.height();
    }
    static inline int size3(array_type const& mat) {
	return mat.width();
    }
private:
    array3d_traits();
};

template <class M1, class M2>
void array3d_copy(M1 const & m1, M2 & m2)
{
    typedef array3d_traits<M1> tr1;
    typedef array3d_traits<M2> tr2;
    if(tr1::size1(m1) != tr2::size1(m2) || tr1::size2(m1) != tr2::size2(m2) 
       || tr1::size3(m1) != tr2::size3(m2) )
    {
	tr2::change_size(m2, tr1::size1(m1), tr1::size2(m1), tr1::size3(m1));
    }
    for(int cc=0; cc<tr1::size1(m1); ++cc)
    {
	for(int yy=0; yy<tr1::size2(m1); ++yy)
	{
	    for(int xx=0; xx<tr1::size3(m1); ++xx)
	    {
		tr2::ref(m2, cc, yy, xx) = tr1::ref(m1, cc, yy, xx);
	    }
	}
    }
}

template <class M1, class M2, class F>
void array3d_transform(M1 const & m1, M2 & m2, F f)
{
    typedef array3d_traits<M1> tr1;
    typedef array3d_traits<M2> tr2;
    if(tr1::size1(m1) != tr2::size1(m2) || tr1::size2(m1) != tr2::size2(m2) 
       || tr1::size3(m1) != tr2::size3(m2) )
    {
	tr2::change_size(m2, tr1::size1(m1), tr1::size2(m1), tr1::size3(m1));
    }
    for(int cc=0; cc<tr1::size1(m1); ++cc)
    {
	for(int yy=0; yy<tr1::size2(m1); ++yy)
	{
	    for(int xx=0; xx<tr1::size3(m1); ++xx)
	    {
		tr2::ref(m2, cc, yy, xx) = f(tr1::ref(m1, cc, yy, xx));
	    }
	}
    }
}

template <class M1, class M2, class M3>
bool array3d_cat1(M1 const & m1, M2 const & m2, M3& m3)
{
    typedef array3d_traits<M1> tr1;
    typedef array3d_traits<M2> tr2;
    typedef array3d_traits<M3> tr3;
    if( tr1::size2(m1) != tr2::size2(m2) 
       || tr1::size3(m1) != tr2::size3(m2) )
    {
	return false;
    }

    if(tr1::size1(m1) + tr2::size1(m2) != tr3::size1(m3) ||
       tr1::size2(m1) != tr3::size2(m3) || tr1::size3(m1) != tr3::size3(m3) )
    {
	tr3::change_size(m3, tr1::size1(m1)+tr2::size1(m2), 
			 tr1::size2(m1), tr1::size3(m1));
    }
    for(int cc=0; cc<tr1::size1(m1); ++cc)
    {
	for(int yy=0; yy<tr1::size2(m1); ++yy)
	{
	    for(int xx=0; xx<tr1::size3(m1); ++xx)
	    {
		tr3::ref(m3, cc, yy, xx) = tr1::ref(m1, cc, yy, xx);
	    }
	}
    }

    for(int cc=0; cc<tr2::size1(m2); ++cc)
    {
	for(int yy=0; yy<tr2::size2(m2); ++yy)
	{
	    for(int xx=0; xx<tr2::size3(m2); ++xx)
	    {
		tr3::ref(m3, cc+tr1::size1(m1), yy, xx) = tr1::ref(m2, cc, yy, xx);
	    }
	}
    }
}

template <class M1, class M2, class M3>
bool array3d_cat2(M1 const & m1, M2 const & m2, M3& m3)
{
    typedef array3d_traits<M1> tr1;
    typedef array3d_traits<M2> tr2;
    typedef array3d_traits<M3> tr3;
    if( tr1::size1(m1) != tr2::size1(m2) 
       || tr1::size3(m1) != tr2::size3(m2) )
    {
	return false;
    }

    if(tr1::size2(m1) + tr2::size2(m2) != tr3::size2(m3) ||
       tr1::size1(m1) != tr3::size1(m3) || tr1::size3(m1) != tr3::size3(m3) )
    {
	tr3::change_size(m3, tr1::size1(m1),
			 tr1::size2(m1)+tr2::size2(m2), tr1::size3(m1));
    }
    for(int cc=0; cc<tr1::size1(m1); ++cc)
    {
	for(int yy=0; yy<tr1::size2(m1); ++yy)
	{
	    for(int xx=0; xx<tr1::size3(m1); ++xx)
	    {
		tr3::ref(m3, cc, yy, xx) = tr1::ref(m1, cc, yy, xx);
	    }
	}
    }

    for(int cc=0; cc<tr2::size1(m2); ++cc)
    {
	for(int yy=0; yy<tr2::size2(m2); ++yy)
	{
	    for(int xx=0; xx<tr2::size3(m2); ++xx)
	    {
		tr3::ref(m3, cc, yy+tr1::size2(m1), xx) = tr1::ref(m2, cc, yy, xx);
	    }
	}
    }
}

template <class M1, class M2, class M3>
bool array3d_cat3(M1 const & m1, M2 const & m2, M3& m3)
{
    typedef array3d_traits<M1> tr1;
    typedef array3d_traits<M2> tr2;
    typedef array3d_traits<M3> tr3;
    if( tr1::size1(m1) != tr2::size1(m2) 
       || tr1::size2(m1) != tr2::size2(m2) )
    {
	return false;
    }

    if(tr1::size3(m1) + tr2::size3(m2) != tr3::size3(m3) ||
       tr1::size1(m1) != tr3::size1(m3) || tr1::size2(m1) != tr3::size2(m3) )
    {
	tr3::change_size(m3, tr1::size1(m1),
			 tr1::size2(m1), tr1::size3(m1)+tr2::size3(m2));
    }
    for(int cc=0; cc<tr1::size1(m1); ++cc)
    {
	for(int yy=0; yy<tr1::size2(m1); ++yy)
	{
	    for(int xx=0; xx<tr1::size3(m1); ++xx)
	    {
		tr3::ref(m3, cc, yy, xx) = tr1::ref(m1, cc, yy, xx);
	    }
	}
    }

    for(int cc=0; cc<tr2::size1(m2); ++cc)
    {
	for(int yy=0; yy<tr2::size2(m2); ++yy)
	{
	    for(int xx=0; xx<tr2::size3(m2); ++xx)
	    {
		tr3::ref(m3, cc, yy, xx+tr1::size3(m1)) = tr1::ref(m2, cc, yy, xx);
	    }
	}
    }
}

template <class M>
void array3d_max(M const& m, std::size_t& i1, std::size_t& i2, std::size_t& i3)
{
    typedef array3d_traits<M> tr;
    typename tr::value_type v = tr::ref(m, 0, 0, 0);

    i1 = 0; i2 = 0; i3 = 0;
    for(std::size_t cc=0; cc<tr::size1(m); ++cc)
    {
	for(std::size_t yy=0; yy<tr::size2(m); ++yy)
	{
	    for(std::size_t xx=0; xx<tr::size3(m); ++xx)
	    {
		if( tr::ref(m, cc, yy, xx)> v)
		{
		    v = tr::ref(m, cc, yy, xx);
		    i1 = cc;
		    i2 = yy;
		    i3 = xx;
		}
	    }
	}
    }
}

template <class M>
void array3d_fill(M & m, typename array3d_traits<M>::value_type const& val)
{
    typedef array3d_traits<M> tr;

    for(std::size_t cc=0; cc<tr::size1(m); ++cc)
    {
	for(std::size_t yy=0; yy<tr::size2(m); ++yy)
	{
	    for(std::size_t xx=0; xx<tr::size3(m); ++xx)
	    {
		tr::ref(m, cc, yy, xx) = val;
	    }
	}
    }
}


template <class ST, class M>
void array3d_print(ST& cout, M const& m)
{
    typedef array3d_traits<M> tr;

    for(std::size_t cc=0; cc<tr::size1(m); ++cc)
    {
	cout<<":----------------------"<<std::endl;
	for(std::size_t yy=0; yy<tr::size2(m); ++yy)
	{
	    for(std::size_t xx=0; xx<tr::size3(m); ++xx)
	    {
		cout<< tr::ref(m, cc, yy, xx) <<" ";
	    }
	    cout<<std::endl;
	}
    }
}

//--------------------------------------------------------------------------------
// array4d_traits
//--------------------------------------------------------------------------------
template <class Mat>
struct array4d_traits
{
    typedef typename Mat::value_type::value_type value_type;
    typedef Mat array_type;
    typedef typename Mat::value_type sub_array_type;
    static inline value_type ref(Mat const& mat, int t, int c, int y, int x)  {
	return mat(t, c)(y, x);
    }
    static inline value_type& ref(Mat & mat, int t, int c, int y, int x) {
	return mat(t, c)(y, x);
    }
    static inline Mat create(int s1, int s2, int s3, int s4) {
	Mat m(s1, s2);
	for(int tt=0; tt<s1; ++tt)
	{
	    for(int cc=0; cc<s2; ++cc)
	    {
		m(tt, cc) = sub_array_type(s3, s4);
	    }
	}
	return m;
    }
    static inline void change_size(Mat & m, int s1, int s2, int s3, int s4) {
	m = create(s1, s2, s3, s4);
    }
    static inline int size1(Mat const& mat) {
	return mat.size1();
    }
    static inline int size2(Mat const& mat) {
	return mat.size2();
    }
    static inline int size3(Mat const& mat) {
	if(mat.size1()>0 && mat.size2()>0)
	    return mat(0, 0).size1();
	else return 0;
    }
    static inline int size4(Mat const& mat) {
	if(mat.size1()>0 && mat.size2()>0)
	    return mat(0, 0).size2();
	else return 0;
    }
};

template <class T>
struct array4d_traits<boost::multi_array<T, 3> >
{
    typedef T value_type;
    typedef boost::multi_array<T, 4> array_type;

    static inline value_type ref(array_type const& mat, int t, int c, int y, int x)  {
	return mat[t][c][y][x];
    }
    static inline value_type& ref(array_type & mat, int t, int c, int y, int x) {
	return mat[t][c][y][x];
    }
    static inline array_type create(int s1, int s2, int s3, int s4) {
	return array_type(boost::extents[s1][s2][s3][s4]);
    }
    static inline void change_size(array_type& m, int s1, int s2, int s3, int s4) {
	m.resize(boost::extents[s1][s2][s3][s4]);
    }
    static inline int size1(array_type const& mat) {
	return mat.shape()[0];
    }
    static inline int size2(array_type const& mat) {
	return mat.shape()[1];
    }
    static inline int size3(array_type const& mat) {
	return mat.shape()[2];
    }
    static inline int size4(array_type const& mat) {
	return mat.shape()[3];
    }
};

template <class T>
struct array4d_traits<cimg_library::CImg<T> >
{
    typedef T value_type;
    typedef cimg_library::CImg<T> array_type;

    static inline value_type ref(array_type const& mat, int t, int c, int y, int x)  {
	return *(mat.data(x, y, t, c));
    }
    static inline value_type& ref(array_type & mat, int t, int c, int y, int x) {
	return *(mat.data(x, y, t, c));
    }
    static inline array_type create(int s1, int s2, int s3, int s4) {
	return array_type(s4, s3, s1, s2);
    }
    static inline void change_size(array_type& m, int s1, int s2, int s3, int s4) {
	m = create(s1, s2, s3, s4);
    }
    static inline int size1(array_type const& mat) {
	return mat.spectrum();
    }
    static inline int size2(array_type const& mat) {
	return mat.height();
    }
    static inline int size3(array_type const& mat) {
	return mat.width();
    }
    static inline int size4(array_type const& mat) {
	return mat.depth();
    }
};

template <class M1, class M2>
void array4d_copy(M1 const & m1, M2 & m2)
{
    typedef array4d_traits<M1> tr1;
    typedef array4d_traits<M2> tr2;
    if(tr1::size1(m1) != tr2::size1(m2) || tr1::size2(m1) != tr2::size2(m2) 
       || tr1::size3(m1) != tr2::size3(m2) ||tr1::size4(m1) != tr2::size4(m2) )
    {
	tr2::change_size(m2, tr1::size1(m1), tr1::size2(m1), tr1::size3(m1), tr1::size4(m1));
    }
    for(int tt=0; tt<tr1::size1(m1); ++tt)
    {
	for(int cc=0; cc<tr1::size2(m1); ++cc)
	{
	    for(int yy=0; yy<tr1::size3(m1); ++yy)
	    {
		for(int xx=0; xx<tr1::size4(m1); ++xx)
		{
		    tr2::ref(m2, tt, cc, yy, xx) = tr1::ref(m1, tt, cc, yy, xx);
		}
	    }
	}
    }
}

/*
Member of array of struct as array
 */
template <class M>
struct member_pointer_type
{
};

template <class T, class MT>
struct member_pointer_type<MT T::*>
{
    typedef MT type;
};


template <class Mat, class Func>
struct member_array1d
{
    typedef typename member_pointer_type<Func>::type value_type;
    member_array1d(Mat& m, Func f)
	:mat(m), func(f){}
    value_type const & operator[](std::size_t ii) const {
	typedef array1d_traits<Mat> tr;
	return boost::lambda::bind(func, boost::lambda::_1)
	    (tr::ref(mat, ii));	    
    }
    value_type& operator[](std::size_t ii)  {
	typedef array1d_traits<Mat> tr;
	return boost::lambda::bind(func, boost::lambda::_1)
	    (tr::ref(mat, ii));
    }
    std::size_t size() const {
	typedef array1d_traits<Mat> tr;
	return tr::size(mat);
    }
    Mat& mat;
    Func func;
};

template <class Mat, class Func>
struct member_array2d
{
    typedef typename member_pointer_type<Func>::type value_type;
    member_array2d(Mat& m, Func f)
	:mat(m), func(f){}
    value_type const & operator()(std::size_t yy, std::size_t xx) const {
	typedef array1d_traits<Mat> tr;
	return boost::lambda::bind(func, boost::lambda::_1)
	    (tr::ref(mat, yy, xx));	    
    }
    value_type& operator()(std::size_t yy, std::size_t xx)  {
	typedef array1d_traits<Mat> tr;
	return boost::lambda::bind(func, boost::lambda::_1)
	    (tr::ref(mat, yy, xx));
    }
    std::size_t size1() const {
	typedef array1d_traits<Mat> tr;
	return tr::size1(mat);
    }
    std::size_t size2() const {
	typedef array1d_traits<Mat> tr;
	return tr::size1(mat);
    }
    Mat& mat;
    Func func;
};


template <class M1, class M2>
void add_patch_to_tile(M1 & tile, M2 const& patch, int pos1, int pos2)
{
    typedef array3d_traits<M1> A;
    typedef array3d_traits<M2> B;

    int hh = B::size2(patch);
    int ww = B::size3(patch);

    for(int cc=0; cc<B::size1(patch); ++cc)
    {
	for(int yy=0; yy<hh; ++yy)
	{
	    for(int xx=0; xx<ww; ++xx)
	    {
		A::ref(tile, cc, hh*pos1+yy, ww*pos2+xx) = B::ref(patch, cc, yy, xx);
	    }
	}
    }

}



/* transpose without data copy
 */

/* useful reshapes
 */

/*
vector of vector, matrix of vector, vector of matrix, matrix of matrix
 */

END_NAMESPACE_CVPR

#endif
