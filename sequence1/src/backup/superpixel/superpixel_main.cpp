/*
Modified by Haifeng GONG 2010.
 */

/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>


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

    if (argc != 4) {
	//fprintf(stderr, "usage: %s sigma k min\n", argv[0]);
	std::cerr<<boost::str(boost::format("usage: %s sigma k min\n")%argv[0])<<std::endl;;
	return 1;
    }
  
    float sigma = atof(argv[1]);
    float k = atof(argv[2]);
    int min_size = atoi(argv[3]);
	
    mpi::environment env(argc, argv);
    mpi::communicator world;
    std::cout << "I am process " << world.rank() << " of " << world.size()
	      << "." << std::endl;

    const std::string prefix = "../images/";
    const std::string output = "../output2/";
    const std::string workspace = "../workspace/superpixel";
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
    boost::array<std::vector<std::string>, 2> seq;
    read_sequence_list(prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;



    for(int tt=world.rank(); tt<T; tt+=world.size())
    {
	real_timer_t frame_timer;

	for(int cam=0; cam<Ncam; ++cam)
	{
	    CImg<unsigned char> image;
	    image = CImg<unsigned char>(seq[cam][tt].c_str());
#ifdef HALF_IMAGE
	    image.resize_halfXY();
#endif
	    std::cout<<"\tloading input image."<<std::endl;
	    std::cout<<"\tprocessing"<<std::endl;

	    matrix<int> seg;
	    CImg<unsigned char> segim;
	    int num_ccs = segment_image(image, sigma, k, min_size, seg, segim);
	    
	    std::string image_name = fs::basename(fs::path(seq[cam][tt]));

#ifdef HALF_IMAGE
	    fs::path det_path = fs::path(workspace)/(image_name+"_seg_half.imat");
#else
	    fs::path det_path = fs::path(workspace)/(image_name+"_seg.imat");
#endif

	    std::ofstream fout(det_path.string().c_str());
	    fout<<seg<<std::endl;
	    fout.close();

#ifdef HALF_IMAGE
	    fs::path vis_path = fs::path(workspace)/(image_name+"_seg_half.png");
#else
	    fs::path vis_path = fs::path(workspace)/(image_name+"_seg.png");
#endif
	    segim.save_png(vis_path.string().c_str());

	    std::cout<<boost::str(boost::format("\tgot %d components\n")%num_ccs);
	    std::cout<<"\tdone! uff...thats hard work.\n"<<std::endl;

	}

	std::cout<<"rank="<<world.rank()<<", frame="<<tt
		 <<" done in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;
    }

    std::cout<<"rank="<<world.rank()<<", \t "<<timer.elapsed()/1000.0f<<"s."<<std::endl;
    return 0;
}
