#ifndef __BBSEG__HPP__INCLUDED__
#define __BBSEG__HPP__INCLUDED__

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/math/constants/constants.hpp>

#include <boost/array.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

#include <algorithm>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "labelmap.hpp"
#include "misc_utils.hpp"
#include "statistics.hpp"

//define for better formatting in editor
#define BEGIN_NAMESPACE_BBSEG namespace bbseg {

#define END_NAMESPACE_BBSEG };


BEGIN_NAMESPACE_BBSEG

using namespace boost::numeric::ublas;
using namespace boost::lambda;
namespace umf=boost::numeric::bindings::umfpack;

using namespace cvpr;

using boost::array;


//sparse matrix used by UMFPACK, double-only
typedef sparse_matrix_t<double>::type umf_sparse_matrix;

template <class Mat, class Float>
void preapre_affinity_matrix(umf_sparse_matrix& W,
			     const Mat& mat, int nc,
			     Float sig, int KN)
{
    typedef require_same_type<typename Mat::value_type, Float>
	Mat_value_type_should_be_the_same_as_Float;

    int nrow = mat.size1();
    int ncol = mat.size2()/nc;
    int np = nrow*ncol;
    int Nbr = (KN*KN+KN)*4+(KN==0)*4;

    W = umf_sparse_matrix(np, np, (Nbr+1)*np);
    int ndi[Nbr];
    int ndj[Nbr];
    int bb=0;
    if(KN==0) {
	ndi[0]= 1;  ndj[0]=0;
	ndi[1]= 0;  ndj[1]=1;
	ndi[2]= -1; ndj[2]=0;
	ndi[3]= 0;  ndj[3]=-1;
    }
    else
    {
	for(int di=-KN; di <= KN; ++di)
	{
	    for(int dj = -KN; dj <= KN; ++dj)
	    {
		if(di==0 &&  dj==0) continue;
		ndi[bb] = di;
		ndj[bb] = dj;
		bb++;
	    }
	}
    }

    for(int ii=0; ii<nrow; ++ii)
    {
	for(int jj=0; jj<ncol; ++jj)
	{
	    int iii = ii*ncol+jj;
	    W(iii, iii) = 1e-10; //keep room for diagonal D
	    for(int nn=0; nn<Nbr; ++nn)
	    {
		int i2 = ii+ndi[nn];
		int j2 = jj+ndj[nn];
		if(i2<0) continue;
		if(j2<0) continue;
		if(i2>=nrow) continue;
		if(j2>=ncol) continue;

		int jjj = i2*ncol+j2;
		if(jjj>=iii) continue;
		Float dI=0;
		for(int cc=0; cc<nc; ++cc)
		{
		    Float dpix = mat(ii, jj*nc+cc)-mat(i2, j2*nc+cc);
		    dI += dpix*dpix;
		}
		Float sdI = std::sqrt(dI)/sig;
		W(iii, jjj) = std::exp(-dI*dI/2);
		W(jjj, iii) = W(iii, jjj);
	    }
	}
    }
    //std::cout<<"\t\t\ttime to build W is "<<timer.elapsed()<<std::endl;

}

template <class Mat, class Float>
void biased_graph_cut(const Mat& mat, int nc, Float sig, int KN,
		      Float alpha, const matrix<Float>& prior,
		      umf_sparse_matrix& W, matrix<Float>& X)
{
    typedef require_same_type<typename Mat::value_type, Float>
	Mat_value_type_should_be_the_same_as_Float;

    int nrow = mat.size1();
    int ncol = mat.size2()/nc;
    int np = nrow*ncol;

    // Compute W matrix if necessary
    if(W.size1()==0 || W.size2()==0)
    {
	//real_timer_t timer;
	preapre_affinity_matrix(W, mat, nc, sig, KN);
    }

    vector<double> Bhat(np);
    std::copy(prior.data().begin(), prior.data().end(), Bhat.data().begin());

    // Using correct access order, super fast
    vector<Float> Dv=scalar_vector<Float>(np, 0);
    typedef umf_sparse_matrix::const_iterator2 citer2;
    typedef umf_sparse_matrix::const_iterator1 citer1;
    typedef umf_sparse_matrix::iterator2 iter2;
    typedef umf_sparse_matrix::iterator1 iter1;

    for(citer2 it=W.begin2(); it != W.end2(); ++it)
    {
	for(citer1 it2=it.begin(); it2 != it.end(); ++it2)
	{
	    Dv(it2.index1()) += *it2;
	}
    }

    Float meanDv = sum(Dv)/Dv.size();
    umf_sparse_matrix D_W = W;
    for(iter2 it= D_W.begin2(); it != D_W.end2(); ++it)
    {
	int jj = it.index2();
	for(iter1 it2=it.begin(); it2 != it.end(); ++it2)
	{
	    int ii = it2.index1();
	    if(ii==jj) { *it2 = Dv(ii)+alpha*meanDv - *it2;  }
	    else   {  *it2 = -*it2;   }
	}
    }

    vector<double> Xhat(np);

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    umf::symbolic (D_W, symb); 
    umf::numeric (D_W, symb, nume); 
    umf::solve(D_W, Xhat, Bhat, nume);  

    X = matrix<Float>(nrow, ncol);
    for(int ii=0; ii<nrow; ++ii)    {
	for(int jj=0; jj<ncol; ++jj)	{
	    X(ii, jj) = Xhat(ii*ncol+jj);
	}
    }

}

void prepare_ext_box(const vector<float>& box,
		     int s1, int s2,
		     boost::array<int, 4>& ext_ri,
		     boost::array<int, 4>& in_ri)
{
    float rect_cent_x = (box(0)+box(2))/2;
    float rect_cent_y = (box(1)+box(3))/2;

    float rect_w = box(2)-box(0);
    float rect_h = box(3)-box(1);

    vector<float> ext_rect(4);
    ext_rect(0) = rect_cent_x-rect_w/2*2;
    ext_rect(1) = rect_cent_y-rect_h/2*1.2;
    ext_rect(2) = rect_cent_x+rect_w/2*2;
    ext_rect(3) = rect_cent_y+rect_h/2*1.2;
   

    std::transform(ext_rect.begin(), ext_rect.end(), ext_ri.begin(), 
		   ll_static_cast<int>(_1+0.5f));
    std::transform(box.begin(), box.end(), in_ri.begin(), 
		   ll_static_cast<int>(_1+0.5f));


    if(ext_ri[0]<0)    {  ext_ri[0] = 0;   }
    if(ext_ri[1]<0)    {  ext_ri[1] = 0;   }
    if(ext_ri[2]>=s2)    {  ext_ri[2] = s2-1;   }
    if(ext_ri[3]>=s1)    {  ext_ri[3] = s1-1;   }

}

void prepare_shape_map(const boost::array<int, 4>& ext_ri,
		       const boost::array<int, 4>& in_ri,
		       matrix<float>& smap)
{
    int s2 = ext_ri[2]-ext_ri[0]+1;
    int s1 = ext_ri[3]-ext_ri[1]+1;

    int w = in_ri[2]-in_ri[0]+1;
    int h = in_ri[3]-in_ri[1]+1;

    float centx = (in_ri[2]+in_ri[0])/2.0f;
    float centy = (in_ri[3]+in_ri[1])/2.0f;

    float dcx = centx-ext_ri[0];
    float dcy = centy-ext_ri[1];

    float sc = 3.0f;
    float rate = std::sqrt((1-1/sc/sc)/(2*std::log(sc)));
    float sigw = w/2.0f*rate;
    float sigh = h/2.0f*rate;

    smap = matrix<float>(s1, s2);

    for(int yy=0; yy<s1; ++yy)
    {
	float dy = yy-dcy;
	dy /= sigh;
	for(int xx=0; xx<s2; ++xx)
	{
	    float dx = xx-dcx;
	    dx /= sigw;
	    float d2 = dx*dx+dy*dy;
	    float score = -0.5*(1-1/sc/sc)*d2+std::log(sc);
	    smap(yy, xx) = score;
	}
    }

}

// remove small segments from a binary segment map
template <class Int>
void remove_small_connected_components(matrix<Int>& seg, Int thr)
{
    matrix<Int> cmpmap(seg.size1(), seg.size2());
    int ncomp = labelmap_connected_components(seg, cmpmap);
    if(ncomp>2)     //remove small regions
    {
	vector<Int> cmpcount=scalar_vector<int>(ncomp, 0);
	labelmap_count_components(cmpmap, ncomp, cmpcount);
	vector<bool> small_flag(ncomp);
	std::transform(cmpcount.begin(), cmpcount.end(), small_flag.begin(), _1<=thr);
	matrix<Int> seg2 = seg;
	tile_panel_t tp(1, 3, seg.size1(), seg.size2());
	tp.add_image_gray(0, 0, seg2);
	for(int ii=0; ii<seg.size1(); ++ii)
	{
	    for(int jj=0; jj<seg.size2(); ++jj)
	    {
		int cpid = cmpmap(ii, jj);
		if(small_flag(cpid)) seg(ii, jj) = 1-seg(ii, jj);
	    }
	}
	tp.add_image_gray(0, 1, seg);
	matrix<int> diff=seg-seg2;
	tp.add_image_gray(0, 2, diff);

    }

}

//threshold a score map into a binary segment map
// prevent collapsing into a single segment
template <class Mat1, class Mat2>
void score_to_segment(const Mat1& X, Mat2& seg, float fgrl, float fgru)
{
    using namespace boost::lambda;
    std::transform(X.data().begin(), X.data().end(), seg.data().begin(), ll_static_cast<int>(_1>0.0f));
    int fg_count = std::count_if(seg.data().begin(), seg.data().end(), _1>0);
    int np = X.data().size();
    int fgl = static_cast<int>(np*fgrl);
    int fgu = static_cast<int>(np*fgru);
    if(fg_count >= fgl&& fg_count <= fgu)  return;
    if(fg_count < fgl) fg_count = fgl; //prevent degeneration when low contrast
    if(fg_count > fgu) fg_count = fgu; //prevent degeneration when low contrast

    typedef typename Mat1::value_type Float;
    typedef std::pair<Float, int> Pair;

    vector<Pair> x2(X.data().size());
    std::transform(X.data().begin(), X.data().end(), boost::counting_iterator<int>(0),
		  x2.begin(), bind(std::make_pair<Float, int>, _1, _2));
    std::sort(x2.begin(), x2.end(), bind(&Pair::first, _1)< bind(&Pair::first, _2));

    for(int ii=0; ii<x2.size(); ++ii)
    {
	int jj = x2(ii).second;
	seg.data()[jj] = (ii>=np-fg_count);
    }

}

template <class Float>
void do_em_gmm(int K, gaussian_mixture_t<Float>& model, const matrix<Float>& datax,
	       int stride, int it)
{
    int np = datax.size2();
    int ndim = datax.size1();
    if(np<=1000 || it>0)
    {
	EM_subsamp_opt<Float> opt(stride, it!=0); //self init (random)
	EM_gaussian_mixture(datax, K, model, opt);
    }
    else
    {
	int s = np/500;
	slice current_slice(0, s, (np-1)/s+1);
	slice all_dim(0, 1, ndim);

	matrix<Float> dataz = project(datax, all_dim, current_slice);

	EM_plain_opt<Float> opt; //self init (random)
	EM_gaussian_mixture(dataz, K, model, opt);
	EM_subsamp_opt<Float> opt2(stride, true); 
	EM_gaussian_mixture(datax, K, model, opt2);

    }
}

template <class Float>
void bbseg_one(const matrix<Float>& matrect, int nc, const matrix<float>& smap,
	       matrix<int>& seg) 
{

    //Initialize segmentation
    int ss1 = matrect.size1(); //(ext_ri[3]-ext_ri[1]+1);
    int ss2 = matrect.size2()/nc; //(ext_ri[2]-ext_ri[0]+1);
    int np = ss1*ss2;


    boost::array<int, 2> K = {2, 3};
    Float sig = 60;
    int KN = 0; //Four neighborhood
    //int KN = 1; //8 neighborhood
    Float alpha = 0.2;//0.05; //0.1;

    boost::array<gaussian_mixture_t<Float>, 2> models;

    matrix<Float> data;
    reshape_yxf2fp(matrect, nc, data);

    umf_sparse_matrix W;  //affinity matrix
    boost::array<vector<Float>, 2> loglike; //gmm loglike
    matrix<Float> X; //gcut solution

    boost::array<vector<bool>, 2> z;
    z[0] = vector<bool>(np);
    z[1] = vector<bool>(np);

    std::transform(smap.data().begin(), smap.data().end(), z[0].data().begin(), _1>0.0f);
    std::transform(smap.data().begin(), smap.data().end(), z[1].data().begin(), _1<=0.0f);

    seg = matrix<int>(ss1, ss2);
    //std::cout<<"bbsize="<<ss1<<"x"<<ss2<<"="<<ss1*ss2<<std::endl;
    for(int it=0; it<3; ++it)
    {
	//Initialize background and foreground model
	real_timer_t timer1;

	int fg_count = std::count(z[0].begin(), z[0].end(), true);
	int bg_count = np-fg_count;

	boost::array<int, 2> count = {fg_count, bg_count};
	//std::cout<<"fgcount="<<fg_count<<"/bgount="<<bg_count<<std::endl;

	for(int mm=0; mm<models.size(); ++mm)
	{
	    const int NS=5;//4;//3;
	    int stride  = count[mm]/5000*NS;
	    if(stride<NS) stride = NS;
	    matrix<Float> datax = columns(data, z[mm]);
	    do_em_gmm(K[mm], models[mm], datax, stride, it);
	}

	for(int mm=0; mm<models.size(); ++mm)
	{
	    gaussian_mixture_loglike(models[mm], data, loglike[mm]);
	}

	Float logpf = std::log(Float(fg_count));
	Float logpb = std::log(Float(bg_count));

	vector<Float> lldiff = loglike[0]-loglike[1];
	matrix<Float> lr_image(ss1, ss2);
	std::transform(lldiff.begin(), lldiff.end(), lr_image.data().begin(), _1+logpf-logpb);

	if(it==0)
	    std::cout<<"\t\ttime to first em is "<<timer1.elapsed()<<std::endl;
	else
	    std::cout<<"\t\t\ttime to other em is "<<timer1.elapsed()<<std::endl;

	real_timer_t timer;

	matrix<Float> mask = lr_image*0.8+smap*0.5;
	biased_graph_cut(matrect, nc, sig, KN, alpha, mask, W, X);
	//std::cout<<"\t\ttime to gc is "<<timer.elapsed()<<std::endl;

	real_timer_t timer_cc;

	score_to_segment(X, seg, 1.0/6.0, 2.0/3.0);
	
	// too small threshold will result in null segment
	remove_small_connected_components(seg, /*thr=*/ss1*ss2*1/200);
	//std::cout<<"\t\ttime to cc is "<<timer_cc.elapsed()<<std::endl;

	std::transform(seg.data().begin(), seg.data().end(), z[0].data().begin(), _1>0);
	std::transform(seg.data().begin(), seg.data().end(), z[1].data().begin(), _1<=0);

    }


}


template <class Vec, class Mat>
void split_ped_parts(const vector<boost::array<float, 4> >& model,
		     const Vec& ped_box, Mat& part_boxes)
{
    typedef typename Vec::value_type Float;
    typedef require_same_type<Float, typename Mat::value_type> 
	Vec_and_Mat_Must_Have_The_Same_value_type;

    part_boxes = matrix<Float>(3, 4);
    Float h = ped_box(3)-ped_box(1);
    Float w = ped_box(2)-ped_box(0);

    for(int bb=0; bb<model.size(); ++bb)
    {
	part_boxes(bb, 0) = model(bb)[0]*w+ped_box(0);
	part_boxes(bb, 2) = model(bb)[2]*w+ped_box(0);
	part_boxes(bb, 1) = model(bb)[1]*h+ped_box(1);
	part_boxes(bb, 3) = model(bb)[3]*h+ped_box(1);
    }

}

template <class Float>
void bbsegment_image(const matrix<Float>& mat, int nc, 
		     const vector<array<float, 4> >& shape_model,
		     const matrix<float>& ped_boxes,
		     matrix<matrix<int> >& segs,
		     matrix<array<int, 4> >& parts,
		     matrix<array<int, 4> >& exts) 
{
    int np = ped_boxes.size1();
    int nb = shape_model.size();
    segs = matrix<matrix<int> >(np, nb);
    exts = matrix<array<int, 4> >(np, nb);
    parts = matrix<array<int, 4> >(np, nb);


    int s1 = mat.size1();
    int s2 = mat.size2()/nc;

    for(int pp=0; pp<np; ++pp)
    {
	matrix<Float> part_boxes;
	split_ped_parts(shape_model, row(ped_boxes, pp), part_boxes);
	for(int bb=0; bb<part_boxes.size1(); ++bb)
	{
	    array<int, 4> ext_ri;
	    array<int, 4> in_ri;

	    prepare_ext_box(row(part_boxes, bb), s1, s2, ext_ri, in_ri);

	    range ry(ext_ri[1], ext_ri[3]+1);
	    range rx(ext_ri[0]*nc, ext_ri[2]*nc+1*nc);
	    matrix<Float> matrect=project(mat, ry, rx);

	    matrix<float> smap;
	    prepare_shape_map(ext_ri, in_ri, smap);

	    std::copy(ext_ri.begin(), ext_ri.end(), exts(pp, bb).begin());
	    std::copy(in_ri.begin(), in_ri.end(), parts(pp, bb).begin());

	    bbseg_one(matrect, nc, smap, segs(pp, bb));

	}
    }
}


END_NAMESPACE_BBSEG

#endif
