#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/format.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "ublas_cimg.hpp"
#include "ublas_random.hpp"

#include "bbseg.hpp"
#include "labelmap.hpp"


using namespace cimg_library;

namespace mpi = boost::mpi;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;

using namespace cvpr;


using namespace bbseg;


//const float head_w=0.5;
//const float head_h=0.2;
//const float torso_h = 0.6;
const float model_data[3][4]=
{
    {0.25, 0, 0.75, 0.2},
    {0, 0.2, 1, 0.6},
    {0, 0.6, 1, 1}
};

void load_part_model(ublas::vector<boost::array<float, 4> >& model)
{

    model = ublas::vector<boost::array<float, 4> >(3);
    for(int ii=0; ii<3; ++ii)
    {
	for(int jj=0; jj<4; ++jj)
	{
	    model(ii)[jj] = model_data[ii][jj];
	}
    }

}


void test_weighted_column_mean_var()
{
    using namespace boost::numeric::ublas;
    //bbseg_t bbs;

    matrix<float> data0=randn_matrix<float>(3, 10000, 3.0f, 1.0f);
    //std::cout<<data<<std::endl;

    matrix<float> prj = randn_matrix<float>(2, 3, 1, 1);
    matrix<float> data = prod(prj, data0);

    vector<float> mean;
    matrix<float> var;

    vector<float> fg = scalar_vector<float>(data.size2(), 1);
    weighted_column_mean_var(data, fg, mean, var);

    std::cout<<"mean="<<mean<<std::endl;
    std::cout<<"var="<<var<<std::endl;
    std::cout<<"prj="<<prj<<std::endl;
    std::cout<<"prj*prjT="<<prod(prj, trans(prj))<<std::endl;
}

void test_gaussian_loglike()
{
    typedef float Float;
    using namespace boost::numeric::ublas;
    //bbseg_t bbs;

    matrix<Float> data0=randn_matrix<Float>(3, 10000, 0.0f, 1.0f);
    //std::cout<<data<<std::endl;

    matrix<Float> prj = randn_matrix<Float>(2, 3, 1, 1);
    matrix<Float> data = prod(prj, data0);

    vector<Float> mean;
    matrix<Float> var;

    vector<Float> fg = scalar_vector<Float>(data.size2(), 1);
    weighted_column_mean_var(data, fg, mean, var);

    std::cout<<"mean="<<mean<<std::endl;
    std::cout<<"var="<<var<<std::endl;
    std::cout<<"=========================================="<<std::endl;

    matrix<Float> x = matrix_range<matrix<Float> >(data, range(0, data.size1()), range(0, 3));

    vector<Float> loglike;
    gaussian_loglike<Float>(mean, var, x, loglike);
    std::cout<<"x="<<x<<std::endl;
    std::cout<<"loglike="<<loglike<<std::endl;

}

void test_em()
{
    typedef float Float;
    using namespace boost::numeric::ublas;
    //bbseg_t bbs;

    matrix<Float> data0=randn_matrix<Float>(3, 200, 0.0f, 1.0f);
    matrix<Float> data1=randn_matrix<Float>(3, 800, -6.0f, 1.0f);

    matrix<Float> data(data0.size1(), data0.size2()+data1.size2());
    typedef matrix_range<matrix<Float> > mat_range; 
    project(data, range(0, 3), range(0, data0.size2())) = data0;
    project(data, range(0, 3), range(data0.size2(), data0.size2()+data1.size2()))=data1;

#if 0
    matrix<Float> data(1, 8);
    data(0, 0) = -1; data(0, 1) = 0; data(0, 2) = 0; data(0, 3) = 1;
    data(0, 4) = 9; data(0, 5) = 10; data(0, 6) = 10; data(0, 7) = 11;
#endif

    vector<Float> fgconf;
    int K = 2;
    vector<Float> pis;
    vector<vector<Float> > means;
    vector<matrix<Float> > vars;

    fgconf = scalar_vector<Float>(data.size2(), 1);

    EM_gaussian_mixture(data, K, pis, means, vars, EM_plain_opt<Float>(1e-6));

    std::cout<<"pis="<<pis<<std::endl;
    std::cout<<"means(0)="<<means(0)<<std::endl;
    std::cout<<"means(1)="<<means(1)<<std::endl;

    std::cout<<"vars(0)="<<vars(0)<<std::endl;
    std::cout<<"vars(1)="<<vars(1)<<std::endl;

}

void test_em2()
{
    typedef float Float;
    using namespace boost::numeric::ublas;
    //bbseg_t bbs;

    matrix<Float> data0=randn_matrix<Float>(3, 400, 0.0f, 1.0f);
    matrix<Float> data1=randn_matrix<Float>(3, 600, -2.0f, 1.0f);

    matrix<float> prj1 = randn_matrix<float>(3, 3, 1, 1);
    matrix<float> prj2 = randn_matrix<float>(3, 3, 1, 2);
    matrix<float> data0x = prod(prj1, data0);
    matrix<float> data1x = prod(prj2, data1);

    matrix<Float> data(data0.size1(), data0.size2()+data1.size2());
    //typedef matrix_range<matrix<Float> > mat_range; 
    project(data, range(0, 3), range(0, data0.size2())) = data0x;
    project(data, range(0, 3), range(data0.size2(), data0.size2()+data1.size2()))=data1x;


    int K = 2;
    vector<Float> pis;
    vector<vector<Float> > means;
    vector<matrix<Float> > vars;

    real_timer_t timer;
    EM_gaussian_mixture(data, K, pis, means, vars, EM_plain_opt<Float>(1e-6));
    std::cout<<"time: "<<timer.elapsed()<<std::endl;

    std::cout<<"pis="<<pis<<std::endl;
    std::cout<<"means(0)="<<means(0)<<std::endl;
    std::cout<<"means(1)="<<means(1)<<std::endl;
    std::cout<<"prj2*x  ="<<prod(prj2, scalar_vector<Float>(3, -2.0f))<<std::endl;

    std::cout<<"vars(0)   ="<<vars(0)<<std::endl;
    std::cout<<"prj1*prj1T="<<prod(prj1, trans(prj1))<<std::endl;

    std::cout<<"vars(1)   ="<<vars(1)<<std::endl;
    std::cout<<"prj2*prj2T="<<prod(prj2, trans(prj2))<<std::endl;



}

void test_em3(bool subsamp)
{
    typedef float Float;
    using namespace boost::numeric::ublas;
    //bbseg_t bbs;

    matrix<Float> data0=randn_matrix<Float>(3, 2000, 0.0f, 1.0f);
    matrix<Float> data1=randn_matrix<Float>(3, 3000, 2.0f, 1.0f);
    matrix<Float> data2=randn_matrix<Float>(3, 2000, 6.0f, 1.0f);
    matrix<Float> data3=randn_matrix<Float>(3, 3000, 9.0f, 1.0f);

    matrix<float> prj1 = randn_matrix<float>(3, 3, 1, 1);
    matrix<float> prj2 = randn_matrix<float>(3, 3, 1, 2);
    matrix<float> prj3 = randn_matrix<float>(3, 3, 0, 1);
    matrix<float> prj4 = randn_matrix<float>(3, 3, 0, 2);

    matrix<float> data0x = prod(prj1, data0);
    matrix<float> data1x = prod(prj2, data1);
    matrix<float> data2x = prod(prj3, data2);
    matrix<float> data3x = prod(prj4, data3);

    int s0 = data0.size2();
    int s1 = data1.size2();
    int s2 = data2.size2();
    int s3 = data3.size2();

    int dim = data0.size1();
    int np = s0+s1+s2+s3;

    matrix<Float> data(dim, np);
    //typedef matrix_range<matrix<Float> > mat_range; 
    project(data, range(0, dim), range(0, s0)) = data0x;
    project(data, range(0, dim), range(s0, s0+s1))=data1x;
    project(data, range(0, dim), range(s0+s1, s0+s1+s2)) = data2x;
    project(data, range(0, dim), range(s0+s1+s2, np))=data3x;



    vector<bool> fgconf(np);
    int K = 2;
    vector<Float> pis;
    vector<vector<Float> > means;
    vector<matrix<Float> > vars;

    project(fgconf, range(0, np/2)) = scalar_vector<bool>(np/2, false);
    project(fgconf, range(np/2, np)) = scalar_vector<bool>(np-np/2, true);

    matrix<Float> datax = columns(data, fgconf);

    real_timer_t timer;
    if(subsamp)
	EM_gaussian_mixture(datax, K, pis, means, vars, EM_subsamp_opt<Float>(10, 1e-6));
    else
	EM_gaussian_mixture(datax, K, pis, means, vars, EM_plain_opt<Float>(1e-6));

    std::cout<<"time: "<<timer.elapsed()<<std::endl;

    std::cout<<"pis="<<pis<<std::endl;
#if 0
    std::cout<<"means(0)="<<means(0)<<std::endl;
    std::cout<<"prj1*x  ="<<prod(prj1, scalar_vector<Float>(3, 0.0f))<<std::endl;
    std::cout<<"means(1)="<<means(1)<<std::endl;
    std::cout<<"prj2*x  ="<<prod(prj2, scalar_vector<Float>(3, 2.0f))<<std::endl;

    std::cout<<"vars(0)   ="<<vars(0)<<std::endl;
    std::cout<<"prj1*prj1T="<<prod(prj1, trans(prj1))<<std::endl;

    std::cout<<"vars(1)   ="<<vars(1)<<std::endl;
    std::cout<<"prj2*prj2T="<<prod(prj2, trans(prj2))<<std::endl;
#else
    std::cout<<"means(0)="<<means(0)<<std::endl;
    std::cout<<"prj1*x  ="<<prod(prj3, scalar_vector<Float>(3, 6.0f))<<std::endl;
    std::cout<<"means(1)="<<means(1)<<std::endl;
    std::cout<<"prj2*x  ="<<prod(prj4, scalar_vector<Float>(3, 9.0f))<<std::endl;

    std::cout<<"vars(0)   ="<<vars(0)<<std::endl;
    std::cout<<"prj1*prj1T="<<prod(prj3, trans(prj3))<<std::endl;

    std::cout<<"vars(1)   ="<<vars(1)<<std::endl;
    std::cout<<"prj2*prj2T="<<prod(prj4, trans(prj4))<<std::endl;
#endif


}



void load_shape(const std::string& shape_name, CImg<float>& shape)
{
    CImg<unsigned char> shape_img = CImg<unsigned char>(shape_name.c_str());
    shape = CImg<float>(shape_img.width(), shape_img.height(), 1, 1);
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


void draw_detected_boxes(CImg<unsigned char>* images, int cam,
			 const ublas::matrix<float>& car_boxes,
			 const ublas::matrix<float>& ped_boxes )
{
    for(int cc=0; cc<car_boxes.size1(); ++cc)
    {
	if(car_boxes(cc, 4)<=0) continue;
	unsigned char ccol[3]={255, 255, 0};
	images[cam].draw_rectangle(int(car_boxes(cc, 0)+0.5), int(car_boxes(cc, 1)+0.5), 
				   int(car_boxes(cc, 2)+0.5), int(car_boxes(cc, 3)+0.5), ccol, 0.3);
    }

    for(int pp=0; pp<ped_boxes.size1(); ++pp)
    {
	unsigned char pcol[3]={255, 0, 0};
	images[cam].draw_rectangle(int(ped_boxes(pp, 0)+0.5), int(ped_boxes(pp, 1)+0.5), 
				   int(ped_boxes(pp, 2)+0.5), int(ped_boxes(pp, 3)+0.5), pcol, 0.3);
    }

}


template <class T>
void draw_detection_segments(CImg<T>& image, const matrix<ublas::matrix<int> >& segs,
			     const matrix<array<int, 4> >& parts,
			     const matrix<array<int, 4> >& exts)
{
    for(int ss=0; ss<segs.size1(); ++ss)
    {
	T color[3][3]={ {0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
	for(int bb=0; bb<segs.size2(); ++bb)
	{
	    const array<int, 4>& ext = exts(ss, bb);
	    image.draw_line(ext[0], ext[1], 0, ext[0], ext[3], 0, color[bb], 1);
	    image.draw_line(ext[2], ext[1], 0, ext[2], ext[3], 0, color[bb], 1);
	    image.draw_line(ext[0], ext[1], 0, ext[2], ext[1], 0, color[bb], 1);
	    image.draw_line(ext[0], ext[3], 0, ext[2], ext[3], 0, color[bb], 1);

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

    //seq = new std::vector<std::string>[2];
    seq[0] = seql;
    seq[1] = seqr;

}

//#define  HALF_SIZE

int test_bbseg() 
{
    using namespace boost::numeric::ublas;
    using namespace boost;
    const std::string prefix = "../images/";
    const std::string output = "../output2/";
    const std::string workspace = "../workspace/";
    const std::string figures = "../figures/";

    boost::array<std::vector<std::string>, 2> seq;
    read_sequence_list(prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    vector<boost::array<float, 4> > model;
    load_part_model(model);

    //for(int tt=world.rank(); tt<T; tt+=world.size())
    int tt = 13; //27; //1; 
    {
	CImg<unsigned char> images[Ncam];
	matrix<float> mat[Ncam];
	for(int cam=0; cam<Ncam; ++cam)
	{
	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());
#ifdef HALF_SIZE
	    images[cam].resize_halfXY();
#endif
	    mat[cam] = ublas_matrix<float>(images[cam]);
	}

	for(int cam=0; cam<Ncam; ++cam)
	{
	    fs::path seq_path = seq[cam][tt];

	    std::string image_name = fs::basename(seq_path);
	    fs::path image_folder = seq_path.branch_path();

	    //fs::path ped_path = fs::path(workspace)/(image_name+"_3d_ped.txt");
	    fs::path car_path = fs::path(workspace)/"car_detection"/(image_name+".txt");
	    fs::path ped_path = fs::path(workspace)/"detection_refine"/(image_name+"_3d_ped.txt");


	    matrix<float> ped_boxes;
	    read_text_matrix(ped_path.string(), ped_boxes);
#ifdef HALF_SIZE
	    ped_boxes /= 2.0f;
#endif

	    int nc = images[cam].spectrum();
	    real_timer_t frame_timer;
	    matrix<matrix<int> > segs;
	    matrix<array<int, 4> > exts;
	    matrix<array<int, 4> > parts;

	    bbsegment_image(mat[cam], nc, model, ped_boxes, segs, parts, exts);
	    draw_detection_segments(images[cam], segs, parts, exts);

	    fs::path det_path = fs::path(figures)/(image_name+"_det_test.jpg");
	    images[cam].save_jpeg(det_path.string().c_str(), 90);

	    std::cout<<"frame="<<tt<<" done in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

	}
    }

    return 0;
}


void test_shape_map()
{
    using namespace boost::lambda;
    boost::array<int, 4> ext_ri;
    boost::array<int, 4> in_ri;

    ext_ri[0] = 0; ext_ri[1] = 0;
    ext_ri[2] = 100; ext_ri[3] = 100;
    in_ri[0] = 25; in_ri[1] = 25;
    in_ri[2] = 75; in_ri[3] = 75;

    matrix<float> smap;
    prepare_shape_map(ext_ri, in_ri, smap);

    float minv = *(std::min_element(smap.data().begin(), smap.data().end()));
    float maxv = *(std::max_element(smap.data().begin(), smap.data().end()));

    matrix<unsigned char> disp(smap.size1(), smap.size2());

    std::transform(smap.data().begin(), smap.data().end(), disp.data().begin(),
		  ll_static_cast<unsigned char>((_1-minv)/(maxv-minv)*255.0f));


    CImg<unsigned char> image = cvpr::cimg<unsigned char>(disp);

    CImgDisplay main_disp(image, "smap");
    while (!main_disp.is_closed())
    {
	main_disp.wait();
	if (main_disp.button() && main_disp.mouse_y()>=0) {
	    const int y = main_disp.mouse_y();
	    const int x = main_disp.mouse_x();
	    std::cout<<smap(y, x)<<std::endl;

        }
    }
    //return 0;

}

void test_labelmap()
{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;

    const std::string prefix = "../images/";

    boost::array<std::vector<std::string>, 2> seq;
    read_sequence_list(prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    CImg<unsigned char> images[Ncam];
    matrix<float> mat[Ncam];
    int tt=0;
    for(int cam=0; cam<Ncam; ++cam)
    {
	images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());
	//images[cam].resize_halfXY();
	mat[cam] = ublas_matrix<float>(images[cam]);
    }
    matrix<float> matx = project(mat[0], slice(0, 1, 100), slice(0, 3, 200));
    matrix<int> lb(matx.size1(), matx.size2());
    std::transform(matx.data().begin(), matx.data().end(), lb.data().begin(), ll_static_cast<int>(_1/25));
    matrix<int> cmp(lb.size1(), lb.size2());

    real_timer_t timer;
    int ncomp = labelmap_connected_components(lb, cmp);
    std::cout<<"time: "<<timer.elapsed()<<std::endl;


    vector<int> count=scalar_vector<int>(ncomp, 0);

    //std::for_each(cmp.data().begin(), cmp.data().end(), count(_1)) ;

    labelmap_count_components(cmp, ncomp, count);
    std::cout<<"time: "<<timer.elapsed()<<std::endl;
    std::cout<<count<<std::endl;

    show_image("window_name", lb);
    show_image("window_name", cmp);


    //std::cout<<cmp<<std::endl;

}


void test_sort_with_index()
{
    using namespace boost::lambda;

    typedef float Float;
    matrix<Float> X = randn_matrix<Float>(2, 3, 1, 1);
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


int main(int argc, char * argv[])
{
    //test_weighted_column_mean_var();
    //test_gaussian_loglike();
    //test_em();
    //test_em2();
    
    //test_em3(true);
    //test_em3(false);

    test_bbseg();

    //test_shape_map();

    //test_labelmap();
    //test_sort_with_index();
    //test_pointer_size();

    return 0;
}



