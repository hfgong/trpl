#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/ref.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "text_file.hpp"
#include "real_timer.hpp"
#include "ublas_cimg.hpp"
#include "ublas_random.hpp"

using namespace cimg_library;

namespace fs = boost::filesystem;
using namespace boost::numeric::ublas;

using namespace cvpr;

#include "input_output.hpp"


int main(int argc, char* argv[]) 
{
    using namespace boost::numeric::ublas;
    using boost::array;

    const std::string prefix = "../images/";
    const std::string output = "../output2/";
    const std::string workspace = "../workspace/superpixel";
    const std::string figures = "../figures/";

    if ( !fs::exists( figures ) )
    {
	fs::create_directory( figures );
    }
    if ( !fs::exists( workspace ) )
    {
	fs::create_directory( workspace );
    }


    boost::array<std::vector<std::string>, 2> seq;
    read_sequence_list(prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    for(int tt=0; tt<T; tt++)
    {
	CImg<unsigned char> images[Ncam];
	matrix<float> mat[Ncam];
	real_timer_t frame_timer;

	for(int cam=0; cam<Ncam; ++cam)
	{

	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());

	    mat[cam] = ublas_matrix<float>(images[cam]);

	    fs::path seq_path = seq[cam][tt];
	    std::string image_name = fs::basename(seq_path);

#ifdef HALF_IMAGE
	    fs::path det_path = fs::path(workspace)/(image_name+"_seg_half.imat");
#else
	    fs::path det_path = fs::path(workspace)/(image_name+"_seg.imat");
#endif
	    //images[cam].save_jpeg(det_path.string().c_str(), 90);
	    matrix<int> seg;

	    std::ifstream fin(det_path.string().c_str());
	    fin>>seg;
	    fin.close();

	    matrix<unsigned char> seg_vis(seg.size1(), seg.size2()*3);
	    randu_int_t<int> rand(0, 255); rand.seed();
	    //std::cout<<"rand="<<rand()<<std::endl;
	    std::map<int, array<unsigned char, 3> > colormap;
	    for(int yy=0; yy<seg.size1(); ++yy)
	    {
		for(int xx=0; xx<seg.size2(); ++xx)
		{
		    int comp = seg(yy, xx);
		    std::map<int, array<unsigned char, 3> >::iterator it = colormap.find(comp);
		    if(it == colormap.end()) {
			array<int, 3> c;
			//std::generate(c.begin(), c.end(), boost::ref(rand));
			c[0] = rand();
			c[1] = rand();
			c[2] = rand();
			colormap[comp] = c;
			it = colormap.find(comp);
		    }
		    
		    seg_vis(yy, xx*3+0) = it->second[0];
		    seg_vis(yy, xx*3+1) = it->second[1];
		    seg_vis(yy, xx*3+2) = it->second[2];
		}
	    }
	    std::cout<<"showing"<<std::endl;
	    show_image("Segmap", seg_vis, 3);
	    std::cout<<"showing done"<<std::endl;

	    std::cout<<" finished frame "<<tt<<", cam "<<cam<<std::endl;

	}

    }

    return 0;
}
