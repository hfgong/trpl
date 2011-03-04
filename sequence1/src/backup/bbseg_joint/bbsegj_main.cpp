#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/format.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "ublas_cimg.hpp"
#include "bbseg_joint.hpp"


using namespace cimg_library;

namespace mpi = boost::mpi;
namespace fs = boost::filesystem;
using namespace boost::numeric::ublas;


using namespace cvpr;

#include "input_output.hpp"

std::string rank_and_time(int rank)
{
    int h, m, s, ms;
    time_of_day(h, m, s, ms);
    return boost::str(boost::format("#%d-%02d:%02d:%02d.%03d")
		      %rank%h%m%s%ms )+":";
}


//#define HALF_IMAGE
int main(int argc, char* argv[]) 
{
    using namespace boost::numeric::ublas;
    using boost::array;
    mpi::environment env(argc, argv);
    mpi::communicator world;
    std::cout << "I am process " << world.rank() << " of " << world.size()
	      << "." << std::endl;

    const std::string prefix = "../images/";
    const std::string output = "../output2/";
    const std::string workspace = "../workspace/";
    const std::string figures = "../figures/";

    if (0==world.rank())
    {
	if ( !fs::exists( output ) )
	{
	    fs::create_directory( output );
	}
    
	if ( !fs::exists( figures ) )
	{
	    fs::create_directory( figures );
	}
    }
    world.barrier();

    real_timer_t timer;
    boost::array<std::vector<std::string>, 2> seq;
    read_sequence_list(prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    std::string shape_name = prefix+"shape.bmp";
    CImg<float> shape;
    load_shape(shape_name, shape);

#ifdef HALF_IMAGE
    shape.resize_halfXY();
#endif

    vector<boost::array<float, 4> > model;
    load_part_model(model);


    for(int tt=world.rank(); tt<T; tt+=world.size())
    {
	CImg<unsigned char> images[Ncam];
	matrix<float> mat[Ncam];
	real_timer_t frame_timer;

	for(int cam=0; cam<Ncam; ++cam)
	{
	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());
#ifdef HALF_IMAGE
	    images[cam].resize_halfXY();
#endif
	    mat[cam] = ublas_matrix<float>(images[cam]);
	}

	for(int cam=0; cam<Ncam; ++cam)
	{
	    fs::path seq_path = seq[cam][tt];

	    std::string image_name = fs::basename(seq_path);
	    fs::path image_folder = seq_path.branch_path();

	    fs::path car_path = fs::path(workspace)/"car_detection"/(image_name+".txt");
	    fs::path ped_path = fs::path(workspace)/"detection_refine"/(image_name+"_3d_ped.txt");


	    matrix<float> car_boxes;
	    read_text_matrix(car_path.string(), car_boxes);
#ifdef HALF_IMAGE
	    car_boxes /= 2.0f;
#endif
      
	    matrix<float> ped_boxes;
	    read_text_matrix(ped_path.string(), ped_boxes);
#ifdef HALF_IMAGE
	    ped_boxes /= 2.0f;
#endif
	    //draw boxes
	    draw_detected_boxes(images, cam, car_boxes, ped_boxes);
	    matrix<matrix<int> > segs;
	    matrix<array<int, 4> > exts;
	    matrix<array<int, 4> > parts;

	    int nc = images[cam].spectrum();
	    //bbsegj::bbsegment_image(mat[cam], nc, model, ped_boxes, segs, parts, exts)
	    bbsegj::bbsegment_image_joint(mat[cam], nc, model, ped_boxes,
				  segs, parts, exts);

	    draw_detection_segments(images[cam], segs, parts, exts);

#ifdef HALF_IMAGE
	    fs::path det_path = fs::path(figures)/(image_name+"_det_half.jpg");
#else
	    fs::path det_path = fs::path(figures)/(image_name+"_det.jpg");
#endif
	    images[cam].save_jpeg(det_path.string().c_str(), 90);

	}
	std::cout<<"rank="<<world.rank()<<", frame="<<tt
		 <<" done in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

    }

    std::cout<<"rank="<<world.rank()<<", \t "<<timer.elapsed()/1000.0f<<"s."<<std::endl;
    return 0;
}
