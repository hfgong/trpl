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

const float infw = 1e6f;
const float missw = 10000.0f;


void make_idlabel_table(matrix<float> const& results,
			std::vector<int>& rid2label,
			std::map<int, int>& rlabel2id)
{

    for(int ii=0; ii<results.size1(); ++ii)
    {
	int label = static_cast<int>(results(ii, 0)+0.5f);
	if(rlabel2id.find(label) == rlabel2id.end())
	{
	    rlabel2id[label] = rid2label.size();
	    rid2label.push_back(label);
	}
    }
#if 0
    std::cout<<"id2label:"<<std::endl;
    for(int ii=0; ii<rid2label.size(); ++ii)
    {
	std::cout<<ii<<", "<<rid2label[ii]<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<"label2id:"<<std::endl;
    for(std::map<int, int>::iterator it = rlabel2id.begin();
	it != rlabel2id.end(); ++it)
    {
	std::cout<<it->first<<", "<<it->second<<std::endl;
    }
    std::cout<<std::endl;
#endif
}


void  make_connection_matrix(matrix<bool>& conn,
			     vector<matrix<float> > const& gt,
			     vector<matrix<float> > const& results)
{
    int NG = gt(0).size1();
    int NR = results(0).size1();
    conn = scalar_matrix<bool>(NG, NR, false);
    for(int ii=0; ii<gt(0).size1(); ++ii)
    {
	int tt_g = static_cast<int>(gt(0)(ii, 1)+0.5f);
	matrix_row<matrix<float> const> gr0(gt(0), ii);
	matrix_row<matrix<float> const> gr1(gt(1), ii);

	vector<float> r11 = project(gr0, range(2, 6));
	vector<float> r12 = project(gr1, range(2, 6));
	for(int jj=0; jj<results(0).size1(); ++jj)
	{
	    int tt_r = static_cast<int>(results(0)(jj, 1)+0.5f);
	    if(tt_g != tt_r) continue;
	    matrix_row<matrix<float> const> rr0(results(0), jj);
	    matrix_row<matrix<float> const> rr1(results(1), jj);

	    vector<float> r21 = project(rr0, range(2, 6));
	    vector<float> r22 = project(rr1, range(2, 6));
	    float inar1 = rectint(r11, r21);
	    float inar2 = rectint(r12, r22);

	    if(inar1>=0.4f && inar2>=0.4f)
	    {
		conn(ii, jj) = true;
		//std::cout<<gr0<<"\t"<<gr1<<"\t---\t"<<rr0<<"\t"<<rr1<<std::endl;
	    }

	}
    }
    //std::cout<<conn<<std::endl;
}

void make_distance_matrix(matrix<float>& dist,
			  matrix<bool> const& conn,
			  vector<matrix<float> > const& gt,
			  vector<matrix<float> > const& results)
{

    int k = std::max(conn.size1(), conn.size2());
    dist = scalar_matrix<float>(k, k, missw);
    for(int ii=0; ii<conn.size1(); ++ii)
    {
	for(int jj=0; jj<conn.size2(); ++jj)
	{
	    if(conn(ii, jj))
	    {
		float gx1 = (gt(0)(ii, 2)+gt(0)(ii, 4))/2.0f;
		float gx2 = (gt(1)(ii, 2)+gt(1)(ii, 4))/2.0f;
		float gy1 = (gt(0)(ii, 3)+gt(0)(ii, 5))/2.0f;
		float gy2 = (gt(1)(ii, 3)+gt(1)(ii, 5))/2.0f;

		float rx1 = (results(0)(jj, 2)+results(0)(jj, 4))/2.0f;
		float rx2 = (results(1)(jj, 2)+results(1)(jj, 4))/2.0f;
		float ry1 = (results(0)(jj, 3)+results(0)(jj, 5))/2.0f;
		float ry2 = (results(1)(jj, 3)+results(1)(jj, 5))/2.0f;

		float dx1 = gx1-rx1;
		float dx2 = gx2-rx2;

		float dy1 = gy1-ry1;
		float dy2 = gy2-ry2;

		dist(ii, jj) = ( std::sqrt(dx1*dx1+dy1*dy1)
				+std::sqrt(dx2*dx2+dy2*dy2))/2.0f;
	    }
	    else
	    {
		dist(ii, jj) = infw;
	    }
	}
    }
}

void collect_miss_fa(matrix<bool> const& conn,
		     vector<bool>& miss,
		     vector<bool>& fa)
{
    miss = scalar_vector<bool>(conn.size1(), false);
    for(int ii=0; ii<conn.size1(); ++ii)
    {
	bool detected = false;
	for(int jj=0; jj<conn.size2(); ++jj)
	{
	    if(conn(ii, jj)) 
	    {
		detected = true;
		break;
	    }
	}
	if(!detected) miss(ii) = true;
    }

    fa = scalar_vector<bool>(conn.size2(), false);
    for(int jj=0; jj<conn.size2(); ++jj)
    {
	bool hitgt = false;
	for(int ii=0; ii<conn.size1(); ++ii)
	{
	    if(conn(ii, jj))
	    {
		hitgt = true;
		break;
	    }
	}
	if(!hitgt) fa(jj) = true;
    }
}

    
    
void make_reduced_idtable(vector<bool> const& miss,
			  std::vector<int>& reduced2id,
			  std::map<int, int>& id2reduced)
{
    for(int ii=0; ii<miss.size(); ++ii)
    {
	if(miss(ii)) continue;
	reduced2id.push_back(ii);
	id2reduced[reduced2id.size()] = ii;
    }
}

    
void extract_reduced_dist(matrix<float> const& dist,
			  matrix<float> & red_dist,
			  std::vector<int> const& greduced2id,
			  std::vector<int> const& rreduced2id)
{
    int k = std::max(greduced2id.size(), rreduced2id.size());
    red_dist = scalar_matrix<float>(k, k, missw);
    for(int ii=0; ii<greduced2id.size(); ++ii)
    {
	for(int jj=0; jj<rreduced2id.size(); ++jj)
	{
	    red_dist(ii, jj) = dist(greduced2id[ii], rreduced2id[jj]);
	}
    }
}


void perf_eval(vector<matrix<float> > const& results,
	       vector<matrix<float> > const& gt,
	       std::string const& fmt1,
	       std::string const& fmt2)
{
    using namespace boost::lambda;

    //trj id to label translation
    std::vector<int> rtid2label;
    std::map<int, int> rlabel2tid;

    make_idlabel_table(results(0), rtid2label, rlabel2tid);

    //trj id to label translation
    std::vector<int> gtid2label;
    std::map<int, int> glabel2tid;
    make_idlabel_table(gt(0), gtid2label, glabel2tid);

    int NG = gt(0).size1();
    int NR = results(0).size1();

    matrix<bool> conn;
    make_connection_matrix(conn, gt, results);

    vector<bool> miss;
    vector<bool> fa;

    collect_miss_fa(conn, miss, fa);

    std::vector<int> greduced2id;
    std::map<int, int> gid2reduced;
    make_reduced_idtable(miss, greduced2id, gid2reduced);

    std::vector<int> rreduced2id;
    std::map<int, int> rid2reduced;
    make_reduced_idtable(fa, rreduced2id, rid2reduced);


    matrix<float> dist;
    make_distance_matrix(dist, conn, gt, results);

    matrix<float> red_dist;
    extract_reduced_dist(dist, red_dist, greduced2id, rreduced2id);

    // Apply Munkres algorithm to matrix.
    matrix<double> oldw = red_dist;
    Munkres<> m;
    real_timer_t timer;
    m.solve(red_dist);
    std::cout<<"Munkres algorithm elasped time: "<<timer.elapsed()/1000.0f<<std::endl;

    vector<int> gt_coverage = scalar_vector<int>(NG, 0);
    vector<int> rst_coverage = scalar_vector<int>(NR, 0);

    vector<int> gt2rst = scalar_vector<int>(NG, -1);
    vector<int> rst2gt = scalar_vector<int>(NR, -1);

    matrix<int> tid_corres = scalar_matrix<int>(gtid2label.size(),
						rtid2label.size(), 0);
    for(int ii=0; ii<greduced2id.size(); ++ii)
    {
	for(int jj=0; jj<rreduced2id.size(); ++jj)
	{
	    if(red_dist(ii, jj)>=0)
	    {
		//std::cout<<"("<<ii<<", "<<jj<<")"<<std::endl;
		int gii = greduced2id[ii];
		int rjj = rreduced2id[jj];
		if(!conn(gii, rjj)) continue;
		//std::cout<<"["<<gii<<", "<<rjj<<"]="<<conn(gii, rjj)<<std::endl;
		gt_coverage(greduced2id[ii])++;
		rst_coverage(rreduced2id[jj])++;

		gt2rst(greduced2id[ii]) = rreduced2id[jj];
		rst2gt(rreduced2id[jj]) = greduced2id[ii];

		//corres(greduced2id[ii], rreduced2id[jj]) = 1;
		int glab = static_cast<int>(gt(0)(greduced2id[ii], 0)+0.5f);
		int rlab = static_cast<int>(results(0)(rreduced2id[jj], 0)+0.5f);
		int gtid = glabel2tid[glab];
		int rtid = rlabel2tid[rlab];
		tid_corres(gtid, rtid)++;
	    }
	}
    }

    int num_miss = std::count(gt_coverage.begin(), gt_coverage.end(), 0);
    int num_fa = std::count(rst_coverage.begin(), rst_coverage.end(), 0);


    int err1 = 0;
    for(int ii=0; ii<tid_corres.size1(); ++ii)
    {
	matrix_row<matrix<int> > r(tid_corres, ii);
	int tmp = std::count_if(r.begin(), r.end(), _1>0);
	if(tmp>1) err1 += tmp-1;
    }

    int err2 = 0;
    for(int ii=0; ii<tid_corres.size2(); ++ii)
    {
	matrix_column<matrix<int> > r(tid_corres, ii);
	int tmp = std::count_if(r.begin(), r.end(), _1>0);
	if(tmp>1) err2 += tmp-1;
    }

    std::cout<<"num_miss="<<num_miss<<",\tnum_fa="<<num_fa<<",\t";
    std::cout<<"err1="<<err1<<",\terr2="<<err2<<std::endl;

    directory_structure_t ds;
    vector<std::vector<std::string> > seq(2);
    read_sequence_list(ds.prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    matrix<CImg<unsigned char> > patches_gt(Ncam, gt(0).size1());
    matrix<CImg<unsigned char> > patches_rst(Ncam, results(0).size1());

    int ww = 80;
    int hh = 200;
    for(int tt=0; tt<T; ++tt)
    {
	for(int cam=0; cam<Ncam; ++cam)
	{
	    CImg<unsigned char> image(seq[cam][tt].c_str());
	    for(int ii=0; ii<gt(cam).size1(); ++ii)
	    {
		int t2 = static_cast<int>(gt(cam)(ii, 1)+0.5f);
		if(t2!=tt) continue;
		boost::array<int, 4> box = {gt(cam)(ii, 2), gt(cam)(ii, 3),
					    gt(cam)(ii, 4), gt(cam)(ii, 5)};
		patches_gt(cam, ii) = image.get_crop(box[0], box[1], box[2], box[3]);
		patches_gt(cam, ii).resize(ww, hh);
	    }
	    for(int ii=0; ii<results(cam).size1(); ++ii)
	    {
		int t2 = static_cast<int>(results(cam)(ii, 1)+0.5f);
		if(t2!=tt) continue;
		boost::array<int, 4> box = {results(cam)(ii, 2),
					    results(cam)(ii, 3),
					    results(cam)(ii, 4),
					    results(cam)(ii, 5)};
		patches_rst(cam, ii) = image.get_crop(box[0], box[1], box[2], box[3]);
		patches_rst(cam, ii).resize(ww, hh);
	    }
	}
    }
    typedef array3d_traits<CImg<unsigned char> > A;

    vector<CImg<unsigned char> > gt2rst_tile(gtid2label.size());

    for(int ii=0; ii<gt2rst_tile.size(); ++ii)
    {
	//gt2rst_tile(ii) = CImg<unsigned char>(4, T, 1);
	A::change_size(gt2rst_tile(ii), 3, 4*hh, T*ww);
	array3d_fill(gt2rst_tile(ii), 0);
    }

    for(int ii=0; ii<gt2rst.size(); ++ii)
    {
	int jj = gt2rst[ii];
	int tt = static_cast<int>(gt(0)(ii, 1)+0.5f);
	int label = static_cast<int>(gt(0)(ii, 0)+0.5f);
	int tid = glabel2tid[label];

	add_patch_to_tile(gt2rst_tile(tid), patches_gt(0, ii), 0, tt);
	add_patch_to_tile(gt2rst_tile(tid), patches_gt(1, ii), 1, tt);
//#if 0
	if(jj<0) continue;
	add_patch_to_tile(gt2rst_tile(tid), patches_rst(0, jj), 2, tt);
	add_patch_to_tile(gt2rst_tile(tid), patches_rst(1, jj), 3, tt);
//#endif
    }

    for(int ii=0; ii<gt2rst_tile.size(); ++ii)
    {
	//std::cout<<ii<<std::endl;
	std::string name = ds.figures+str(format(fmt1)%ii);
	gt2rst_tile(ii).save_png(name.c_str());
    }

////

    vector<CImg<unsigned char> > rst2gt_tile(rtid2label.size());

    for(int ii=0; ii<rst2gt_tile.size(); ++ii)
    {
	A::change_size(rst2gt_tile(ii), 3, 4*hh, T*ww);
	array3d_fill(rst2gt_tile(ii), 0);
    }

    for(int ii=0; ii<rst2gt.size(); ++ii)
    {
	int jj = rst2gt[ii];
	int tt = static_cast<int>(results(0)(ii, 1)+0.5f);
	int label = static_cast<int>(results(0)(ii, 0)+0.5f);
	int tid = rlabel2tid[label];

	add_patch_to_tile(rst2gt_tile(tid), patches_rst(0, ii), 0, tt);
	add_patch_to_tile(rst2gt_tile(tid), patches_rst(1, ii), 1, tt);
//#if 0
	if(jj<0) continue;
	add_patch_to_tile(rst2gt_tile(tid), patches_gt(0, jj), 2, tt);
	add_patch_to_tile(rst2gt_tile(tid), patches_gt(1, jj), 3, tt);
//#endif
    }

    for(int ii=0; ii<rst2gt_tile.size(); ++ii)
    {
	//std::cout<<ii<<std::endl;
	std::string name = ds.figures+str(format(fmt2)%ii);
	rst2gt_tile(ii).save_png(name.c_str());
    }


}


int main(int argc, char *argv[]) 
{
    directory_structure_t ds;

    matrix<float> gt;
    
    read_text_array2d(ds.workspace+"gt.txt", gt);

    matrix<float> results_plan;
    matrix<float> results_app;
    matrix<float> results_alone;

    {
	read_text_array2d(ds.workspace+"results_plan.txt", results_plan);
	std::cout<<"Evaluating the performance of planning:"<<std::endl;
	//perf_eval(results_plan, gt, "gt2rst_pl%05d.png", "rst2gt_pl%05d.png");
    }
    {
	read_text_array2d(ds.workspace+"results_app.txt", results_app);
	std::cout<<"Evaluating the performance of appearance:"<<std::endl;
	//perf_eval(results_app, gt, "gt2rst_ap%05d.png", "rst2gt_ap%05d.png");
    }
    {
	read_text_array2d(ds.workspace+"results_alone.txt", results_alone);
	std::cout<<"Evaluating the performance of alone:"<<std::endl;
	//perf_eval(results_alone, gt, "gt2rst_al%05d.png", "rst2gt_al%05d.png");
    }

    return 0;
}


#if 0
int mainx(int argc, char *argv[]) 
{

    directory_structure_t ds;
    vector<matrix<float> > gt(2);
    
    read_text_array2d(ds.workspace+"gt0.txt", gt(0));
    read_text_array2d(ds.workspace+"gt1.txt", gt(1));


    matrix<float> gt_all(gt(0).size1(), 10);
    matrix_range<matrix<float> > gta(gt_all, range::all(), range(0, 6));

    gta = gt(0);

    matrix_range<matrix<float> > gtb(gt_all, range::all(), range(6, 10));
    matrix_range<matrix<float> > tmp(gt(1), range::all(), range(2, 6));
    gtb = tmp;

    write_text_array2d(ds.workspace+"gt.txt", gt_all);
    return 0;

    vector<matrix<float> > results_plan(2);
    vector<matrix<float> > results_app(2);
    vector<matrix<float> > results_alone(2);
    {
	read_text_array2d(ds.workspace+"results_plan_0.txt", results_plan(0));
	read_text_array2d(ds.workspace+"results_plan_1.txt", results_plan(1));
	std::cout<<"Evaluating the performance of planning:"<<std::endl;
	perf_eval(results_plan, gt, "gt2rst_pl%05d.png", "rst2gt_pl%05d.png");
    }
    {
	read_text_array2d(ds.workspace+"results_app_0.txt", results_app(0));
	read_text_array2d(ds.workspace+"results_app_1.txt", results_app(1));
	std::cout<<"Evaluating the performance of appearance:"<<std::endl;
	perf_eval(results_app, gt, "gt2rst_ap%05d.png", "rst2gt_ap%05d.png");
    }
    {
	read_text_array2d(ds.workspace+"results_alone_0.txt", results_alone(0));
	read_text_array2d(ds.workspace+"results_alone_1.txt", results_alone(1));
	std::cout<<"Evaluating the performance of alone:"<<std::endl;
	perf_eval(results_alone, gt, "gt2rst_al%05d.png", "rst2gt_al%05d.png");
    }



    return 0;

}
#endif
