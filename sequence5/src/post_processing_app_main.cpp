#include <boost/mpi.hpp>

#define USE_MPI 1

#include "tracking_data_package.hpp"
#include "tracking_detail.hpp"
#include "misc_utils.hpp"

#include <boost/math/constants/constants.hpp>

#include "linprog_app.hpp"
#include "post_processing.hpp"


int main(int argc, char* argv[])
{

    directory_structure_t ds;

    vector<object_trj_t> good_trlet_list;
    {
	std::string name = ds.workspace+"good_trlet_list.xml";
	std::ifstream fin(name.c_str());
	boost::archive::xml_iarchive ia(fin);
	ia >> BOOST_SERIALIZATION_NVP(good_trlet_list);
    }

    if(good_trlet_list.size()==0) 
    {
	std::cout<<"reading good_trlet_list.xml fails."<<std::endl;
	return 1;
    }
    int Ncam = good_trlet_list(0).trj.size();

    vector<object_trj_t> final_trj_list;
    vector<vector<int> > final_trj_index;
    matrix<int> final_state_list;

    {
	std::string name = ds.workspace+"final_trj_list.xml";
	std::ifstream fin(name.c_str());
	boost::archive::xml_iarchive ia(fin);
	ia >> BOOST_SERIALIZATION_NVP(final_trj_list);
    }

    {
	std::string name = ds.workspace+"final_state_list.txt";
	std::ifstream fin(name.c_str());
	fin>>final_state_list;
	fin.close();
    }

    {
	std::string name = ds.workspace+"final_trj_index.txt";
	std::ifstream fin(name.c_str());
	fin >> final_trj_index;
	fin.close();
    }

    int len_thr = 5;
    vector<object_trj_t> merged_trj_list;
    vector<vector<int> > merged_trj_index;
    matrix<int> merged_state_list;

    post_process_trj(good_trlet_list, final_trj_list, final_trj_index,
		     final_state_list, Ncam, len_thr,
		     merged_trj_list, merged_trj_index, merged_state_list);

    {
	std::string name = ds.workspace+"merged_trj_list.xml";
	std::ofstream fout(name.c_str());
	boost::archive::xml_oarchive oa(fout);
	oa << BOOST_SERIALIZATION_NVP(merged_trj_list);
    }

    {
	std::string name = ds.workspace+"merged_trj_index.txt";
	std::ofstream fout(name.c_str());
	fout<<merged_trj_index;
	fout.close();
    }

    {
	std::string name = ds.workspace+"merged_state_list.txt";
	std::ofstream fout(name.c_str());
	fout << merged_state_list;
	fout.close();
    }


    return 0;

}

#include "post_processing_impl.hpp"
#include "linprog_app_impl.hpp"
