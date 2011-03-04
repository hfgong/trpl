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

struct perf_t
{
    int miss;
    int fa;
    int idswitch;
    float miss_rate;
    float fa_rate;
    template <class STR>
    void print(STR& out){
	out<<"miss="<<miss;
	out<<",\tfa="<<fa<<std::endl;
	out<<"miss_rate="<<miss_rate;
	out<<",\tfa_rate="<<fa_rate;
	out<<",\tid_switch="<<idswitch<<std::endl;
    }

};

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

////////////////////////////////////////////////////////////////////////////

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

void make_distance_matrix(matrix<float>& dist,
			  matrix<bool> const& conn,
			  matrix<float> const& gt,
			  matrix<float> const& results)
{

    int k = std::max(conn.size1(), conn.size2());
    dist = scalar_matrix<float>(k, k, missw);
    for(int ii=0; ii<conn.size1(); ++ii)
    {
	for(int jj=0; jj<conn.size2(); ++jj)
	{
	    if(conn(ii, jj))
	    {
		float gx1 = (gt(ii, 2)+gt(ii, 4))/2.0f;
		float gx2 = (gt(ii, 6)+gt(ii, 8))/2.0f;
		float gy1 = (gt(ii, 3)+gt(ii, 5))/2.0f;
		float gy2 = (gt(ii, 7)+gt(ii, 9))/2.0f;

		float rx1 = (results(jj, 2)+results(jj, 4))/2.0f;
		float rx2 = (results(jj, 6)+results(jj, 8))/2.0f;
		float ry1 = (results(jj, 3)+results(jj, 5))/2.0f;
		float ry2 = (results(jj, 7)+results(jj, 9))/2.0f;

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


void perf_eval(matrix<float> const& results,
	       matrix<float> const& gt,
	       vector<std::vector<std::pair<int, int> > > & corres,
	       perf_t& perf)
{
    using namespace boost::lambda;
    vector<int> rtt;
    vector<int> rnn;
    extract_nntt(results, rtt, rnn);
    vector<int> gtt;
    vector<int> gnn;
    extract_nntt(gt, gtt, gnn);

    int T = std::max(*std::max_element(rtt.begin(), rtt.end()),
		     *std::max_element(gtt.begin(), gtt.end()))+1;

    corres = vector<std::vector<std::pair<int, int> > >(T);
    for(int tt=0; tt<T; ++tt)
    {
	std::vector<int> gbbid, rbbid;
	for(int ii=0; ii<gtt.size(); ++ii)
	{
	    if(tt==gtt(ii)) gbbid.push_back(ii);
	}
	for(int ii=0; ii<rtt.size(); ++ii)
	{
	    if(tt==rtt(ii)) rbbid.push_back(ii);
	}


	//prepare valid pairs
	matrix<bool> conn = scalar_matrix<bool>(gbbid.size(), rbbid.size(), false);
	std::vector<std::pair<int, int> > cand;
	std::vector<std::pair<int, int> > cand_kkll;
	for(int kk=0; kk<gbbid.size(); ++kk)
	{
	    int ii = gbbid[kk];
	    matrix_row<matrix<float> const> grow(gt, ii);
	    vector<float> r11 = project(grow, range(2, 6));
	    vector<float> r12 = project(grow, range(6, 10));
	    float ar11 = (r11(2)-r11(0))*(r11(3)-r11(1));
	    float ar12 = (r12(2)-r12(0))*(r12(3)-r12(1));

	    for(int ll=0; ll<rbbid.size(); ++ll)
	    {
		int jj = rbbid[ll];

		matrix_row<matrix<float> const> rrow(results, jj);

		vector<float> r21 = project(rrow, range(2, 6));
		vector<float> r22 = project(rrow, range(6, 10));
		float ar21 = (r21(2)-r21(0))*(r21(3)-r21(1));
		float ar22 = (r22(2)-r22(0))*(r22(3)-r22(1));

		float inar1 = rectint(r11, r21)/std::sqrt(ar11*ar21+1.0f);
		float inar2 = rectint(r12, r22)/std::sqrt(ar12*ar22+1.0f);

		if(inar1>=0.4f && inar2>=0.4f) 
		{
		    conn(kk, ll) = true;
		    cand.push_back(std::make_pair<int, int>(ii, jj));
		    cand_kkll.push_back(std::make_pair<int, int>(kk, ll));
		}

	    }
	}
	//propagate prev corres
	if(tt>0)
	{
	    for(int pp=0; pp<corres(tt-1).size(); ++pp)
	    {
		int l1 = gnn[corres(tt-1)[pp].first];
		int l2 = rnn[corres(tt-1)[pp].second];
		for(int qq=0; qq<cand.size(); ++qq)
		{
		    int l1q = gnn[cand[qq].first];
		    int l2q = rnn[cand[qq].second];
		    if(l1 == l1q && l2 == l2q)
		    {
			corres(tt).push_back(cand[qq]);
			matrix_row<matrix<bool> > connrow(conn, cand_kkll[qq].first);
			std::for_each(connrow.begin(), connrow.end(), _1=false);
			matrix_column<matrix<bool> > conncol(conn, cand_kkll[qq].second);
			std::for_each(conncol.begin(), conncol.end(), _1=false);
		    }
		}
	    }
	}
	std::vector<int> gbbid_remained, rbbid_remained;
	std::vector<int> kk_remained, ll_remained;
	for(int kk=0; kk<conn.size1(); ++kk)
	{
	    int ii = gbbid[kk];
	    matrix_row<matrix<bool> > connrow(conn, kk);
	    if(connrow.end() != std::find(connrow.begin(), connrow.end(), true) )
	    {
		gbbid_remained.push_back(ii);
		kk_remained.push_back(kk);
	    }
	}
	for(int ll=0; ll<conn.size2(); ++ll)
	{
	    int jj = rbbid[ll];
	    matrix_column<matrix<bool> > conncol(conn, ll);
	    if(conncol.end() != std::find(conncol.begin(), conncol.end(), true) )
	    {
		rbbid_remained.push_back(jj);
		ll_remained.push_back(ll);
	    }
	}
	matrix<bool> conn_remained(kk_remained.size(), ll_remained.size());
	for(int ss=0; ss<conn_remained.size1(); ++ss)
	{
	    int kk = kk_remained[ss];
	    for(int zz=0; zz<conn_remained.size2(); ++zz)
	    {
		int ll = ll_remained[zz];
		conn_remained(ss, zz) = conn(kk, ll);
	    }
	}

	matrix<float> gt_remained(gbbid_remained.size(), gt.size2());
	matrix<float> rst_remained(rbbid_remained.size(), results.size2());
	for(int ss=0; ss<gt_remained.size1(); ++ss)
	{
	    int ii = gbbid_remained[ss];
	    matrix_row<matrix<float> > gtr_row(gt_remained, ss);
	    gtr_row = row(gt, ii);
	}
	for(int zz=0; zz<rst_remained.size1(); ++zz)
	{
	    int jj = rbbid_remained[zz];
	    matrix_row<matrix<float> > rstr_row(rst_remained, zz);
	    rstr_row = row(results, jj);
	}

	matrix<float> dist;
	make_distance_matrix(dist, conn_remained, gt_remained, rst_remained);

	// Apply Munkres algorithm to matrix.
	//matrix<double> oldw = dist;
	Munkres<> m;
	real_timer_t timer;
	m.solve(dist);
	//std::cout<<"Munkres algorithm elasped time: "<<timer.elapsed()/1000.0f<<std::endl;

	for(int ss=0; ss<gbbid_remained.size(); ++ss)
	{
	    int ii = gbbid_remained[ss];
	    for(int zz=0; zz<rbbid_remained.size(); ++zz)
	    {
		int jj = rbbid_remained[zz];
		if(dist(ss, zz)>=0)
		    corres(tt).push_back(std::make_pair<int, int>(ii, jj));
	    }
	}
#if 0
	std::cout<<"tt="<<tt<<std::endl;
	for(int cc=0; cc<corres(tt).size(); ++cc)
	{
	    std::cout<<"("<<gnn[corres(tt)[cc].first]<<", "
		     <<rnn[corres(tt)[cc].second]<<")"<<std::endl;
	    
	}
#endif
    }

    vector<int> gcount=scalar_vector<int>(gt.size1(), 0);
    vector<int> rcount=scalar_vector<int>(results.size1(), 0);
    for(int tt=0; tt<corres.size(); ++tt)
    {
	for(int cc=0; cc<corres(tt).size(); ++cc)
	{
	    gcount[corres(tt)[cc].first] ++;
	    rcount[corres(tt)[cc].second] ++;
	}
    }

    int miss = std::count(gcount.begin(), gcount.end(), 0);
    int fa = std::count(rcount.begin(), rcount.end(), 0);

    int mismatch = 0;
    for(int tt=1; tt<corres.size(); ++tt)
    {
	for(int pp=0; pp<corres(tt-1).size(); ++pp)
	{
	    for(int qq=0; qq<corres(tt).size(); ++qq)
	    {
		if(gnn[corres(tt-1)[pp].first] == gnn[corres(tt)[qq].first] &&
		   rnn[corres(tt-1)[pp].second] != rnn[corres(tt)[qq].second])
		{
		    mismatch ++;
		}
		if(gnn[corres(tt-1)[pp].first] != gnn[corres(tt)[qq].first] &&
		   rnn[corres(tt-1)[pp].second] == rnn[corres(tt)[qq].second])
		{
		    mismatch ++;
		}
	    }
	}
    }

    perf.miss = miss;
    perf.fa = fa;
    perf.idswitch = mismatch;
    perf.miss_rate = static_cast<float>(miss)/gt.size1();
    perf.fa_rate = static_cast<float>(fa)/results.size1();

    std::cout<<"miss="<<miss<<"/"<<gt.size1()
	     <<", \tfa="<<fa<<"/"<<results.size1()
	     <<", \tmismatch="<<mismatch<<std::endl;

}

void vis_perf_eval(matrix<float> const& results,
		   matrix<float> const& gt,
		   vector<std::vector<std::pair<int, int> > > const& corres,
		   matrix<CImg<unsigned char> >& patches_gt,
		   matrix<CImg<unsigned char> >& patches_rst,
		   std::string const& fmt1,
		   std::string const& fmt2);


int main(int argc, char *argv[]) 
{
    directory_structure_t ds;

    matrix<float> gt;
    
    read_text_array2d(ds.workspace+"gt.txt", gt);




    perf_t perf_plan, perf_app, perf_alone;
    {
	matrix<float> results_plan;
	read_text_array2d(ds.workspace+"results_plan.txt", results_plan);
	std::cout<<"Evaluating the performance of planning:"<<std::endl;
	vector<std::vector<std::pair<int, int> > > corres;
	perf_eval(results_plan, gt, corres, perf_plan);
	matrix<CImg<unsigned char> > patches_gt;
	matrix<CImg<unsigned char> > patches_rst;
	vis_perf_eval(results_plan, gt, corres, patches_gt, patches_rst,
		      "gt2rst_pl%05d.png", "rst2gt_pl%05d.png");

    }
    {
	matrix<float> results_app;
	read_text_array2d(ds.workspace+"results_app.txt", results_app);
	std::cout<<"Evaluating the performance of appearance:"<<std::endl;
	vector<std::vector<std::pair<int, int> > > corres;
	perf_eval(results_app, gt, corres, perf_app);
	matrix<CImg<unsigned char> > patches_gt;
	matrix<CImg<unsigned char> > patches_rst;
	vis_perf_eval(results_app, gt, corres,  patches_gt, patches_rst,
		      "gt2rst_ap%05d.png", "rst2gt_ap%05d.png");

    }


    {
	matrix<float> results_alone;
	read_text_array2d(ds.workspace+"results_alone.txt", results_alone);
	std::cout<<"Evaluating the performance of alone:"<<std::endl;
	vector<std::vector<std::pair<int, int> > > corres;
	perf_eval(results_alone, gt, corres, perf_alone);
	matrix<CImg<unsigned char> > patches_gt;
	matrix<CImg<unsigned char> > patches_rst;
	vis_perf_eval(results_alone, gt, corres,  patches_gt, patches_rst,
		      "gt2rst_al%05d.png", "rst2gt_al%05d.png");

    }

    std::string name = ds.workspace+"perf.txt";
    std::ofstream fout(name.c_str());
    fout<<"plan perf:"<<std::endl;
    perf_plan.print(fout);
    fout<<"app perf:"<<std::endl;
    perf_app.print(fout); 
    fout<<"alone perf:"<<std::endl;
    perf_alone.print(fout);
    fout.close();

    return 0;
}


void vis_perf_eval(matrix<float> const& results,
		   matrix<float> const& gt,
		   vector<std::vector<std::pair<int, int> > > const& corres,
		   matrix<CImg<unsigned char> >& patches_gt,
		   matrix<CImg<unsigned char> >& patches_rst,
		   std::string const& fmt1,
		   std::string const& fmt2)
{

    using namespace boost::lambda;
    vector<int> rtt;
    vector<int> rnn;
    extract_nntt(results, rtt, rnn);
    vector<int> gtt;
    vector<int> gnn;
    extract_nntt(gt, gtt, gnn);

    directory_structure_t ds;
    vector<std::vector<std::string> > seq(2);
    read_sequence_list(ds.prefix, seq);
    //int T = seq[0].size();
    int T = corres.size();
    int Ncam = 2;

    patches_gt = matrix<CImg<unsigned char> >(Ncam, gt.size1());
    patches_rst = matrix<CImg<unsigned char> >(Ncam, results.size1());

    int ww = 60;
    int hh = 150;
    for(int tt=0; tt<T; ++tt)
    {
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
	    }
	    for(int ii=0; ii<results.size1(); ++ii)
	    {
		int t2 = rtt(ii);
		if(t2!=tt) continue;
		boost::array<int, 4> box = {results(ii, 2+4*cam),
					    results(ii, 3+4*cam),
					    results(ii, 4+4*cam),
					    results(ii, 5+4*cam)};
		patches_rst(cam, ii) = image.get_crop(box[0], box[1], box[2], box[3]);
		patches_rst(cam, ii).resize(ww, hh);
	    }
	}
    }
    typedef array3d_traits<CImg<unsigned char> > A;

    int NG = 1+ *std::max_element(gnn.begin(), gnn.end());
    int NR = 1+ *std::max_element(rnn.begin(), rnn.end());

    vector<CImg<unsigned char> > gt2rst_tile(NG);

    for(int ii=0; ii<NG; ++ii)
    {
	A::change_size(gt2rst_tile(ii), 3, 4*hh, T*ww);
	array3d_fill(gt2rst_tile(ii), 0);
	for(int jj=0; jj<gnn.size(); ++jj)
	{
	    if(gnn[jj] != ii) continue;
	    add_patch_to_tile(gt2rst_tile(ii), patches_gt(0, jj), 0, gtt[jj]);
	    add_patch_to_tile(gt2rst_tile(ii), patches_gt(1, jj), 1, gtt[jj]);
	}
    }

    for(int tt=0; tt<T; ++tt)
    {
	for(int pp=0; pp<corres(tt).size(); ++pp)
	{
	    int ii = corres(tt)[pp].first;
	    int jj = corres(tt)[pp].second;

	    int glabel = gnn[ii];
	    int rlabel = rnn[jj];

	    add_patch_to_tile(gt2rst_tile(glabel), patches_rst(0, jj), 2, tt);
	    add_patch_to_tile(gt2rst_tile(glabel), patches_rst(1, jj), 3, tt);

	}
    }
    for(int ii=0; ii<gt2rst_tile.size(); ++ii)
    {
	std::string name = ds.figures+str(format(fmt1)%ii);
	gt2rst_tile(ii).save_png(name.c_str());
    }

////

    vector<CImg<unsigned char> > rst2gt_tile(NR);

    for(int ii=0; ii<NR; ++ii)
    {
	A::change_size(rst2gt_tile(ii), 3, 4*hh, T*ww);
	array3d_fill(rst2gt_tile(ii), 0);
	for(int jj=0; jj<rnn.size(); ++jj)
	{
	    if(rnn[jj] != ii) continue;
	    add_patch_to_tile(rst2gt_tile(ii), patches_rst(0, jj), 0, rtt[jj]);
	    add_patch_to_tile(rst2gt_tile(ii), patches_rst(1, jj), 1, rtt[jj]);
	}

    }

    for(int tt=0; tt<T; ++tt)
    {
	for(int pp=0; pp<corres(tt).size(); ++pp)
	{
	    int ii = corres(tt)[pp].first;
	    int jj = corres(tt)[pp].second;

	    int glabel = gnn[ii];
	    int rlabel = rnn[jj];
	    add_patch_to_tile(rst2gt_tile(rlabel), patches_gt(0, ii), 2, tt);
	    add_patch_to_tile(rst2gt_tile(rlabel), patches_gt(1, ii), 3, tt);
	}
    }

    for(int ii=0; ii<rst2gt_tile.size(); ++ii)
    {
	std::string name = ds.figures+str(format(fmt2)%ii);
	rst2gt_tile(ii).save_png(name.c_str());
    }

}

