//#include <boost/mpi/environment.hpp>
//#include <boost/mpi/communicator.hpp>

#include "tracking_data_package.hpp"

#include "track_existed_objects.hpp"
#include "propose_new_objects.hpp"
#include "initialize_new_objects.hpp"
#include "segment_parts.hpp"


//#define HALF_IMAGE
int main(int argc, char* argv[]) 
{
    using namespace boost::numeric::ublas;
    using namespace boost::lambda;
    using boost::array;

    real_timer_t timer;
    directory_structure_t ds;
    ds.make_dir();

    vector<std::vector<std::string> > seq(2);
    read_sequence_list(ds.prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    array<std::size_t, 2> img_size = {768, 1024};
    geometric_info_t gi;
    gi.load(ds, img_size);

    parameter_t P;

    //int nobj = 200; //% length(objs_l);
    int nobj = 500; 

    vector<matrix<matrix<unsigned char> > > seg_list(Ncam);
    for(int cam=0; cam<Ncam; ++cam)
    {
	seg_list(cam) = matrix<matrix<unsigned char> >(nobj, T);
    }

    object_info_t oi(Ncam, nobj, P);

    for(int tt=0; tt<T; tt++)
	//for(int tt=0; tt<1; tt++)
    {
	real_timer_t frame_timer;
	vector<CImg<unsigned char> > images(Ncam);
	vector<matrix<float> > grd(Ncam);


	for(int cam=0; cam<Ncam; ++cam)
	{
	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());
	}
	//std::cout<<"\t\timage read time:"<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;
#if 0
	for(int cam=0; cam<Ncam; ++cam)
	{
	    fs::path seq_path(seq[cam][tt]);
	    std::string image_name = fs::basename(seq_path);

	    fs::path car_path = fs::path(ds.workspace)/"car_detection"/(image_name+".txt");
	    fs::path ped_path = fs::path(ds.workspace)/"detection"/(image_name+"_3d_ped.txt");

	    matrix<float> car_boxes;
	    read_text_matrix(car_path.string(), car_boxes);
      
	    matrix<float> ped_boxes;
	    read_text_matrix(ped_path.string(), ped_boxes);

	}
#endif

	track_existed_objects<float>(P,  ds, gi, oi, seq, tt, images, grd);
	//std::cout<<"\t\ttrack time:"<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

	vector<matrix<float> > detected_rects;
	propose_new_objects<float>(P,  ds, gi, oi, seq, tt, images, grd, detected_rects);
	//std::cout<<"\t\tpropse time:"<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

	initialize_new_objects<float>(P,  ds, gi, oi, seq, tt, images, grd, detected_rects);
	std::cout<<"---------------- frame="<<tt <<" done in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

	segment_parts(oi, seq, tt, images, seg_list);
	std::cout<<"---------------- frame="<<tt <<" segpart in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

    }

    for(int nn=0; nn<oi.trlet_list.size(); ++nn)
    {
	if(oi.trlet_list(nn).trj.size()==0) break;
	std::string name = ds.workspace+boost::str(boost::format("trlet_%04d.xml")%nn);
	oi.trlet_list(nn).save(name.c_str());
    }

    for(int nn=oi.trlet_list.size(); nn<20000; ++nn)
    {
	std::string name = ds.workspace+boost::str(boost::format("trlet_%04d.xml")%nn);
	if(fs::exists(name))  fs::remove(name);
	else break;
    }

    {
	std::string name = ds.workspace+"seg_list.txt";
	save_seg_list(name, seg_list);
    }

    std::cout<<"============== total time \t "<<timer.elapsed()/1000.0f<<"s."<<std::endl;
    return 0;
}

#include "tracking_detail.hpp"
#include "track_existed_objects_impl.hpp"
#include "propose_new_objects_impl.hpp"
#include "initialize_new_objects_impl.hpp"
#include "segment_parts_impl.hpp"
