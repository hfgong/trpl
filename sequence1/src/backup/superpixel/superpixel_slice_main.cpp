/*
Modified by Haifeng GONG 2010.
 */
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/algorithm/string.hpp>

#include <misc_utils.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "ublas_cimg.hpp"

#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image2.h"

using namespace cimg_library;

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

void read_slice_lists(std::string const& prefix,
		      std::vector<std::vector<std::vector<std::string> > >& slice_list)
{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;
    using namespace boost;

    fs::path full_path(prefix);
    std::vector<fs::path> files;
    typedef fs::directory_iterator diter;
    bool (*is_file)(fs::path const&) = &fs::is_regular_file;
    std::copy_if(diter(full_path), diter(), std::back_inserter(files), is_file);
    vector<std::string> file_names(files.size());
    std::transform(files.begin(), files.end(), file_names.begin(),
		   bind(&fs::path::string, _1));

    bool (*check_end)(std::string const&, std::string const&) = &boost::ends_with;
    bool (*check_start)(std::string const&, std::string const&) =&boost::starts_with;

    function<bool (std::string const&)> png_filt = bind(check_end, _1, ".png");
    std::vector<std::string> png_names;
    std::copy_if(file_names.begin(), file_names.end(), std::back_inserter(png_names),
		 png_filt);
    std::for_each(png_names.begin(), png_names.end(), _1=bind(&fs::path::filename, _1));
    //std::for_each(png_names.begin(), png_names.end(), std::cout<<_1<<'\n');

    const std::string orie[2]={"x", "y"};
    for(int cam=0; cam<4; ++cam)
    {
	std::vector<std::vector<std::string> > tmp;
	for(int oo=0; oo<2; ++oo)
	{
	    std::string cams = str(format("cam%dtrans_%s")%(cam+1)%orie[oo]);

	    function<bool (std::string const&)> f=bind(check_start, _1, cams);
	    std::vector<std::string> s;
	    std::copy_if(png_names.begin(), png_names.end(), std::back_inserter(s),
			 f);
	    if(s.size()==0) break;
	    std::sort(s.begin(), s.end());
	    std::for_each(s.begin(), s.end(), _1=prefix+_1);
	    tmp.push_back(s);
	}
	if(tmp.size()==0) break;
	slice_list.push_back(tmp);
    }
}

int const stretch = 4;

template <class Mat1, class Mat2>
void array3d_stretch(Mat1 const & m, Mat2& res, int str)
{
    typedef cvpr::array3d_traits<Mat1> tr1;
    typedef cvpr::array3d_traits<Mat2> tr2;
    int s1 = tr1::size1(m);
    int s2 = tr1::size2(m);
    int s3 = tr1::size3(m);

    res = tr2::create(s1, s2, s3*str);

    for(int cc=0; cc<s1; ++cc)
    {
	for(int yy=0; yy<s2; ++yy)
	{
	    for(int xx=0; xx<s3; ++xx)
	    {
		for(int ss=0; ss<str; ++ss)
		{
		    tr2::ref(res, cc, yy, xx*str+ss) = tr1::ref(m, cc, yy, xx);
		}
	    }
	}
    }

}

//#define HALF_IMAGE
int main(int argc, char* argv[]) 
{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;
    using namespace boost;

    if (argc != 4) {
	std::cerr<<boost::str(boost::format("usage: %s sigma k min\n")%argv[0])<<std::endl;;
	return 1;
    }
  
    float sigma = lexical_cast<float>(argv[1]);
    float k = lexical_cast<float>(argv[2]);
    int min_size = lexical_cast<int>(argv[3]);
	
    mpi::environment env;
    mpi::communicator world;
    std::cout << "I am process " << world.rank() << " of " << world.size()
	      << "." << std::endl;

    const std::string prefix = "../workspace/slice/";
    const std::string output = "../output2/";
    const std::string workspace = "../workspace/ssp/";
    const std::string figures = "../figures/";

    if (0==world.rank())
    {
	if ( !fs::exists( workspace ) )
	{
	    fs::create_directory( workspace );
	}
    
    }
    world.barrier();

    real_timer_t timer;
    std::vector<std::vector<std::vector<std::string> > > seq;
    read_slice_lists(prefix, seq);

    int NS = seq[0].size();
    int Ncam = seq.size();

    for(int cam=0; cam<Ncam; ++cam)
    {
	for(int ss=0; ss<NS; ++ss)
	{
	    int T = seq[cam][ss].size();
	    real_timer_t timer;
	    for(int tt=world.rank(); tt<T; tt+=world.size())
	    {
		CImg<unsigned char> image;
		matrix<float> mat;

		image = CImg<unsigned char>(seq[cam][ss][tt].c_str());

		std::cout<<"\tloading input image."<<std::endl;
		std::cout<<"\tprocessing"<<std::endl;

		CImg<unsigned char> image2;
		array3d_stretch(image, image2, stretch);
		matrix<int> seg;
		CImg<unsigned char> segim;
		int num_ccs = segment_image(image2, sigma, k, min_size, seg, segim);
	 		
		std::string image_name = fs::basename(fs::path(seq[cam][ss][tt]));
		fs::path det_path = fs::path(workspace)/(image_name+"_seg.imat");

		std::ofstream fout(det_path.string().c_str());
		fout<<seg<<std::endl;
		fout.close();

		fs::path vis_path = fs::path(workspace)/(image_name+"_seg.png");
		segim.save_png(vis_path.string().c_str());

		std::cout<<str(format("\tgot %d components\n")%num_ccs);

	    }
	    std::cout<<"rank="<<world.rank()<<", ss="<<ss<<" done in "<<timer.elapsed()/1000.0f<<"s."<<std::endl;

	}


    }

    std::cout<<"rank="<<world.rank()<<", \t "<<timer.elapsed()/1000.0f<<"s."<<std::endl;
    return 0;
}
