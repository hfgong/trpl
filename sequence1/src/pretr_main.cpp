#include <boost/mpi.hpp>

#define USE_MPI 1

#include "tracking_data_package.hpp"

#include "track_existed_objects.hpp"
#include "propose_new_objects.hpp"
#include "initialize_new_objects.hpp"

#include "segment_parts.hpp"

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
    using namespace boost::lambda;
    using boost::array;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    std::cout << "I am process " << world.rank() << " of " << world.size()
	      << "." << std::endl;

    real_timer_t timer;
    directory_structure_t ds;
    if (0==world.rank()) ds.make_dir();
    world.barrier();

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

    for(int tt=0; tt<T; tt++)	//for(int tt=0; tt<1; tt++)
    {
	real_timer_t frame_timer;
	vector<CImg<unsigned char> > images(Ncam);
	vector<matrix<float> > grd(Ncam);

	for(int cam=0; cam<Ncam; ++cam)
	{
	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());
	}

	track_existed_objects<float>(world, P,  ds, gi, oi, seq, tt, images, grd);
	vector<matrix<float> > detected_rects;
	propose_new_objects<float>(P,  ds, gi, oi, seq, tt, images, grd, detected_rects);
	initialize_new_objects<float>(world, P,  ds, gi, oi, seq, tt, images, grd, detected_rects);
	std::cout<<"---------------- frame="<<tt <<" done in "<<frame_timer.elapsed()/1000.0f<<"s."<<std::endl;

	segment_parts(oi, seq, tt, images, seg_list);

    }

    std::cout<<"============== total time \t "<<timer.elapsed()/1000.0f<<"s."<<std::endl;

    if (0==world.rank()) 
    {

	{
	    std::vector<int> idx;
	    for(int nn=0; nn<oi.trlet_list.size(); ++nn)
	    {
		if(oi.trlet_list(nn).trj.size()!=0)
		{
		    idx.push_back(nn);
		}
	    }
	    vector<object_trj_t> trlet_list(idx.size());
	    for(int ii=0; ii<idx.size(); ++ii)
	    {
		trlet_list(ii) = oi.trlet_list(idx[ii]);
	    }
	    std::string name = ds.workspace+"raw_trlet_list.xml";
	    std::ofstream fout(name.c_str());
	    boost::archive::xml_oarchive oa(fout);
	    oa << BOOST_SERIALIZATION_NVP(trlet_list);
	}

	for(int nn=0; nn<oi.trlet_list.size(); ++nn)
	{

	    std::string name = ds.output+boost::str(boost::format("trlet_%04d.xml")%nn);
	    if(fs::exists(name))  fs::remove(name);
	    if(oi.trlet_list(nn).trj.size()==0) break;
	    oi.trlet_list(nn).save(name.c_str());
	}
	for(int nn=oi.trlet_list.size(); nn<20000; ++nn)
	{
	    std::string name = ds.output+boost::str(boost::format("trlet_%04d.xml")%nn);
	    if(fs::exists(name))  fs::remove(name);
	    else break;
	}

	{
	    std::string name = ds.workspace+"seg_list.txt";
	    save_seg_list(name, seg_list);
	}
    }


    return 0;
}

#include "tracking_detail.hpp"
#include "track_existed_objects_impl.hpp"
#include "propose_new_objects_impl.hpp"
#include "initialize_new_objects_impl.hpp"
#include "segment_parts_impl.hpp"
