#include <boost/mpi.hpp>
#include "tracking_data_package.hpp"
#include "tracking_detail.hpp"
#include "planning.hpp"

#include "munkres.hpp"

#include <iostream>

#include "text_file.hpp"

#include "misc_utils.hpp"

#include "real_timer.hpp"

#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;
using namespace cvpr;

void extract_nntt(matrix<float> const& results,
		  vector<int>& rtt,
		  vector<int>& rnn)
{
    using namespace boost::lambda;

    rtt = vector<int>(results.size1());
    matrix_column<matrix<float> const> rtt_tmp(results, 1);
    std::transform(rtt_tmp.begin(), rtt_tmp.end(), rtt.begin(),
		   ll_static_cast<int>(_1+0.5f));

    rnn = vector<int>(results.size1());
    matrix_column<matrix<float> const> rnn_tmp(results, 0);
    std::transform(rnn_tmp.begin(), rnn_tmp.end(), rnn.begin(),
		   ll_static_cast<int>(_1+0.5f));

}

void load_patches(directory_structure_t & ds,
		  matrix<float> const& gt,
		  matrix<CImg<unsigned char> >& patches_gt)
{

    using namespace boost::lambda;
    vector<int> gtt;
    vector<int> gnn;
    extract_nntt(gt, gtt, gnn);


    vector<std::vector<std::string> > seq(2);
    read_sequence_list(ds.prefix, seq);
    int T = seq[0].size();
    //int T = corres.size();
    int Ncam = 2;

    patches_gt = matrix<CImg<unsigned char> >(Ncam, gt.size1());

    int ww = 60;
    int hh = 150;
    for(int tt=0; tt<T; ++tt)
    {
	std::string tstr = str(format("%d")%tt);
	
	for(int cam=0; cam<Ncam; ++cam)
	{
	    CImg<unsigned char> image(seq[cam][tt].c_str());
	    for(int ii=0; ii<gt.size1(); ++ii)
	    {
		int t2 = gtt(ii);
		if(t2!=tt) continue;
		boost::array<int, 4> box = {gt(ii, 2+4*cam), gt(ii, 3+4*cam),
					    gt(ii, 4+4*cam), gt(ii, 5+4*cam)};
		patches_gt(cam, ii) = image.get_crop(box[0], box[1], box[2], box[3]);
		patches_gt(cam, ii).resize(ww, hh);
		unsigned char fcol[] = {255, 255, 255};
		unsigned char bcol[] = {0, 0, 0};
		patches_gt(cam, ii).draw_text (1, 1, tstr.c_str(), fcol, bcol, 1, 35);

	    }
	}
    }
}



void get_pid_map(matrix<float>const& gt, int nngt,
		 std::map<int, int> & pid_gt)
{
    for(int jj=0; jj<gt.size1(); ++jj)
    {
	int nn = static_cast<int>(gt(jj, 0)+0.5f);
	if(nn != nngt) continue;
	int tt = static_cast<int>(gt(jj, 1)+0.5f);
	pid_gt.insert(std::make_pair<int, int>(tt, jj));
    }

}

void vis_pair(directory_structure_t& ds,
	      matrix<CImg<unsigned char> >const& patches_gt,
	      matrix<CImg<unsigned char> >const& patches_rst_plan,
	      matrix<CImg<unsigned char> >const& patches_rst_app,
	      matrix<float>& gt,
	      matrix<float>& results_plan,
	      matrix<float>& results_app)
{
    typedef array3d_traits<CImg<unsigned char> > A;

    matrix<int> idmap(3, 8);
    idmap <<=
	0, 1, 2, 3, 4, 5, 6, 9,
	0, 1, 2, 3, 4, 5, 6, 9,
	4, 0, 1, 2, 3, 4, 6, 9;


    int ww = 60;
    int hh = 150;

    for(int ii=0; ii<idmap.size2(); ++ii)
    {
	int nngt = idmap(0, ii);
	int nnpl = idmap(1, ii);
	int nnap = idmap(2, ii);

	CImg<unsigned char> gt2rst_tile;
	std::map<int, int> pid_gt;
	get_pid_map(gt, nngt, pid_gt);
	std::map<int, int> pid_pl;
	get_pid_map(results_plan, nnpl, pid_pl);
	std::map<int, int> pid_ap;
	get_pid_map(results_app, nnap, pid_ap);

//
	int T0 = std::min(std::min(pid_gt.begin()->first, pid_pl.begin()->first), pid_ap.begin()->first);
	int T1 = 1+std::max(std::max(pid_gt.rbegin()->first, pid_pl.rbegin()->first), pid_ap.rbegin()->first);;
	int dt = 1;
	if(T1-T0>70) dt = 2;
	if(T1-T0>130) dt = 3;

	A::change_size(gt2rst_tile, 3, 3*hh, ((T1-T0-1)/dt+1)*ww);
	array3d_fill(gt2rst_tile, 0);
	for(int tt=T0; tt<T1; tt+=dt)
	{
	    std::map<int, int>::iterator it;
	    it = pid_gt.find(tt) ;
	    if(it != pid_gt.end())
		add_patch_to_tile(gt2rst_tile, patches_gt(0, it->second), 0, (tt-T0)/dt);
	    it = pid_pl.find(tt);
	    if(it != pid_pl.end())
		add_patch_to_tile(gt2rst_tile, patches_rst_plan(0, it->second), 1, (tt-T0)/dt);
	    it = pid_ap.find(tt);
	    if(it != pid_ap.end())
		add_patch_to_tile(gt2rst_tile, patches_rst_app(0, it->second), 2, (tt-T0)/dt);
	}

	{
	    std::string name = ds.figures + str(format("trjpatch_%03d_%03d_%03d.jpg")%nngt%nnpl%nnap);
	    gt2rst_tile.save_jpeg(name.c_str());
	}
    }
}


int main(int argc, char *argv[]) 
{
    directory_structure_t ds;

    matrix<float> gt;
    
    read_text_array2d(ds.workspace+"gt.txt", gt);

    matrix<CImg<unsigned char> > patches_gt;
    load_patches(ds, gt, patches_gt);

    matrix<float> results_plan;
    matrix<CImg<unsigned char> > patches_rst_plan;
    {

	read_text_array2d(ds.workspace+"results_plan.txt", results_plan);
	load_patches(ds, results_plan, patches_rst_plan);

    }

    matrix<float> results_app;
    matrix<CImg<unsigned char> > patches_rst_app;
    {

	read_text_array2d(ds.workspace+"results_app.txt", results_app);
	load_patches(ds, results_app, patches_rst_app);
    }


    vis_pair(ds, patches_gt, patches_rst_plan, patches_rst_app,
	     gt, results_plan, results_app);
}
