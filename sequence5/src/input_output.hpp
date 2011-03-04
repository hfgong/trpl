#ifndef __INPUT__OUTPUT__HPP__INCLUDED__
#define __INPUT__OUTPUT__HPP__INCLUDED__

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>


#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
using namespace cimg_library;

namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;

//const float head_w=0.5;
//const float head_h=0.2;
//const float torso_h = 0.6;
namespace {
const float model_data[3][4]=
{
    {0.25, 0, 0.75, 0.2},
    {0, 0.2, 1, 0.6},
    {0, 0.6, 1, 1}
};
}

template <class Float>
void load_part_model(ublas::vector<boost::array<Float, 4> >& model)
{

    model = ublas::vector<boost::array<Float, 4> >(3);
    for(int ii=0; ii<3; ++ii)
    {
	for(int jj=0; jj<4; ++jj)
	{
	    model(ii)[jj] = model_data[ii][jj];
	}
    }

}

template <class Float>
void load_shape(const std::string& shape_name, CImg<Float>& shape)
{
    CImg<unsigned char> shape_img = CImg<unsigned char>(shape_name.c_str());
    shape = CImg<Float>(shape_img.width(), shape_img.height(), 1, 1);
    for(int yy=0; yy<shape.height(); ++yy)
    {
	for(int xx=0; xx<shape.width(); ++xx)
	{
	    shape(xx, yy, 0, 0) = 0;
	    for(int cc=0; cc<shape.spectrum(); ++cc)
	    {
		shape(xx, yy, 0, 0) += shape_img(xx, yy, 0, cc);
	    }
	    shape(xx, yy, 0, 0) /= shape_img.spectrum();
	}
    }

}

template <class Float>
void draw_detected_boxes(CImg<unsigned char>* images, int cam,
			 const ublas::matrix<Float>& car_boxes,
			 const ublas::matrix<Float>& ped_boxes )
{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;

    CImg<unsigned char>& image = images[cam];

    for(int cc=0; cc<car_boxes.size1(); ++cc)
    {
	if(car_boxes(cc, 4)<=0) continue;
	unsigned char ccol[3]={255, 255, 0};
	boost::array<int, 4> box;
	vector<Float> b = project(row(car_boxes, cc), range(0, 4));
	std::transform(b.begin(), b.end(), box.begin(), ll_static_cast<int>(_1+0.5f));

	image.draw_line(box[0], box[1], 0, box[0], box[3], 0, ccol, 1);
	image.draw_line(box[2], box[1], 0, box[2], box[3], 0, ccol, 1);
	image.draw_line(box[0], box[1], 0, box[2], box[1], 0, ccol, 1);
	image.draw_line(box[0], box[3], 0, box[2], box[3], 0, ccol, 1);

    }

    for(int pp=0; pp<ped_boxes.size1(); ++pp)
    {
	unsigned char pcol[3]={255, 0, 0};
	boost::array<int, 4> box;
	vector<Float> b = row(ped_boxes, pp);
	std::transform(b.begin(), b.end(), box.begin(), ll_static_cast<int>(_1+0.5f));

	image.draw_line(box[0], box[1], 0, box[0], box[3], 0, pcol, 1);
	image.draw_line(box[2], box[1], 0, box[2], box[3], 0, pcol, 1);
	image.draw_line(box[0], box[1], 0, box[2], box[1], 0, pcol, 1);
	image.draw_line(box[0], box[3], 0, box[2], box[3], 0, pcol, 1);
    }

}

void read_sequence_list(const std::string& prefix, boost::array<std::vector<std::string>, 2>& seq)
{
    std::vector<std::string> seql;
    std::vector<std::string> seqr;

    read_string_list(prefix+"image_list_l.txt", seql);
    read_string_list(prefix+"image_list_r.txt", seqr);

    for(int jj=0; jj<seql.size(); ++jj)
    {
	seql[jj] = prefix+"left_rect/"+seql[jj];
    }

    for(int jj=0; jj<seqr.size(); ++jj)
    {
	seqr[jj] = prefix+"right_rect/"+seqr[jj];
    }

    seq[0] = seql;
    seq[1] = seqr;

}

template <class T>
void draw_detection_segments(CImg<T>& image, const ublas::matrix<ublas::matrix<int> >& segs,
			     const ublas::matrix<boost::array<int, 4> >& parts,
			     const ublas::matrix<boost::array<int, 4> >& exts)
{
    using namespace boost;
    for(int ss=0; ss<segs.size1(); ++ss)
    {
	T color[3][3]={ {0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
	for(int bb=0; bb<segs.size2(); ++bb)
	{

	    const array<int, 4>& ext = exts(ss, bb);
#if 0
	    image.draw_line(ext[0], ext[1], 0, ext[0], ext[3], 0, color[bb], 1);
	    image.draw_line(ext[2], ext[1], 0, ext[2], ext[3], 0, color[bb], 1);
	    image.draw_line(ext[0], ext[1], 0, ext[2], ext[1], 0, color[bb], 1);
	    image.draw_line(ext[0], ext[3], 0, ext[2], ext[3], 0, color[bb], 1);
#endif

	    const array<int, 4>& par = parts(ss, bb);
	    image.draw_line(par[0], par[1], 0, par[0], par[3], 0, color[bb], 2);
	    image.draw_line(par[2], par[1], 0, par[2], par[3], 0, color[bb], 2);
	    image.draw_line(par[0], par[1], 0, par[2], par[1], 0, color[bb], 2);
	    image.draw_line(par[0], par[3], 0, par[2], par[3], 0, color[bb], 2);

	    for(int dy=0; dy<segs(ss, bb).size1(); ++dy)
	    {
		for(int dx=0; dx<segs(ss, bb).size2(); ++dx)
		{
		    if(segs(ss, bb)(dy, dx))
		    {
			image.draw_point(ext[0]+dx, ext[1]+dy, color[bb], 0.5);
		    }
		}
	    }
	}
    }
}


#endif
