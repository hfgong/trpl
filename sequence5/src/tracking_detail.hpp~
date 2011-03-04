#ifndef __TRACKING__DETAIL__HPP__INCLUDED__
#define __TRACKING__DETAIL__HPP__INCLUDED__

#include <boost/math/special_functions/fpclassify.hpp>
#include <numeric>
#include "multi_array_serialization.hpp"

template <class Float>
void compute_part_rects(Float x, Float y, Float w, Float h,
			vector<array<float, 4> > const& model,
			matrix<Float>& rects)
{
    int Npart = model.size();
    rects = matrix<Float>(Npart, 4);


    for(int ii=0; ii<Npart; ++ii)
    {
	rects(ii, 0) = model[ii][0]*w+x;
	rects(ii, 1) = model[ii][1]*h+y;
	rects(ii, 2) = model[ii][2]*w+x;
	rects(ii, 3) = model[ii][3]*h+y;
    }
}

template <class Float, class Float2>
void  collect_hist(CImg<unsigned char> const& image,
		   matrix<Float> const & rects,
		   matrix<Float2>& hist_p,
		   matrix<Float2>& hist_q)
{
    using namespace boost::lambda;
    int np = rects.size1();
    vector<Float> rcx((column(rects, 0)+column(rects, 2))/2);
    vector<Float> rcy((column(rects, 1)+column(rects, 3))/2);
    vector<Float> rw(column(rects, 2)-column(rects, 0));
    vector<Float> rh(column(rects, 3)-column(rects, 1));

    matrix<Float> exbb(np, 4);

    column(exbb, 0) = rcx-rw/2*2;
    column(exbb, 1) = rcy-rh/2*1.5;
    column(exbb, 2) = rcx+rw/2*2;
    column(exbb, 3) = rcy+rh/2*1.5;

    matrix<int> exbbi(np, 4);
    matrix<int> inbbi(np, 4);

    array2d_transform(exbb, exbbi, ll_static_cast<int>(_1+0.5));
    array2d_transform(rects, inbbi, ll_static_cast<int>(_1+0.5));

    typedef cvpr::array3d_traits<CImg<unsigned char> > traits;
    int s1 = traits::size2(image);
    int s2 = traits::size3(image);

    for(int ii=0; ii<np; ++ii)
    {
	if(exbbi(ii, 0)<0) exbbi(ii, 0) = 0;
	if(exbbi(ii, 1)<0) exbbi(ii, 1) = 0;
	if(exbbi(ii, 2)>s2-1) exbbi(ii, 2) = s2-1;
	if(exbbi(ii, 3)>s1-1) exbbi(ii, 3) = s1-1;
    }


    matrix<int> hp(scalar_matrix<int>(np, 8*8*8, 0));
    matrix<int> hq(scalar_matrix<int>(np, 8*8*8, 0));

    for(int pp=0; pp<np; ++pp)
    {

	Float dy = (exbbi(pp, 3)-exbbi(pp, 1))/30.0;
	Float dx = (exbbi(pp, 2)-exbbi(pp, 0))/15.0;

	if(dy<1)  dy = 1;
	if(dx<1)  dx = 1;

	for(Float yy=exbbi(pp, 1); yy<=exbbi(pp, 3); yy+=dy)
	{
	    int yi = static_cast<int>(yy+0.5);
	    for(Float xx=exbbi(pp, 0); xx<=exbbi(pp, 2); xx+=dx)
	    {
		int xi = static_cast<int>(xx+0.5);
		int ir = static_cast<int>(traits::ref(image, 0, yi, xi)/32);
		int ig = static_cast<int>(traits::ref(image, 1, yi, xi)/32);
		int ib = static_cast<int>(traits::ref(image, 2, yi, xi)/32);

		int ibin = ir+ig*8+ib*8*8;

		if(yy>=inbbi(pp, 1) && yy<= inbbi(pp, 3)
		   && xx>=inbbi(pp, 0) && xx<=inbbi(pp, 2))
		    ++hp(pp, ibin);
		else ++hq(pp, ibin);
	    }
	}
    }

    hist_p = matrix<Float2>(np, 8*8*8);
    hist_q = matrix<Float2>(np, 8*8*8);

    array2d_copy(hp, hist_p);
    array2d_copy(hq, hist_q);

    for(int pp=0; pp<np; ++pp)
    {
	row(hist_p, pp) /= sum(row(hist_p, pp));
	row(hist_q, pp) /= sum(row(hist_q, pp));
    }

}


template <typename Float>
struct candidate_array
{
    vector<Float> fx;
    vector<Float> fy;
    //CImg<bool> gridijs_mask;
    //CImg<Float> gridijs_score;
    Float default_value;

    //typedef array3d_traits<CImg<bool> > traitsb;
    //typedef array3d_traits<CImg<Float> > traitsf;

    vector<matrix<bool> > gridijs_mask;
    vector<matrix<Float> > gridijs_score;
    typedef array3d_traits<vector<matrix<bool> > > traitsb;
    typedef array3d_traits<vector<matrix<Float> > > traitsf;

    std::size_t size1() const { return traitsf::size1(gridijs_score); }
    std::size_t size2() const { return traitsf::size2(gridijs_score); }
    std::size_t size3() const { return traitsf::size3(gridijs_score); }

    void fill_fxfy(Float feetx, Float feety,
		   vector<Float> const& xr, 
		   vector<Float> const& yr,
		   int ns) {
	using namespace boost::lambda;
	fx = xr;
	fy = yr;
	std::for_each(fx.begin(), fx.end(), _1 += feetx);
	std::for_each(fy.begin(), fy.end(), _1 += feety);

	traitsb::change_size(gridijs_mask, fy.size(), fx.size(), ns);
	traitsf::change_size(gridijs_score, fy.size(), fx.size(), ns);
    }

    void fill_score(vector<Float> const& cand_score,
		    matrix<int> const& cand_ijs,
		    Float def_val=-5) {
	//gridijs_mask.reset(0);
	default_value = def_val;
	//std::fill(gridijs_mask.origin(), gridijs_mask.origin()+gridijs_mask.num_elements(), 0);
	//std::fill(gridijs_score.origin(), gridijs_score.origin()+gridijs_score.num_elements(),
	//    default_value);
	array3d_fill(gridijs_mask, false);
	array3d_fill(gridijs_score, default_value);
	for(int ii=0; ii<cand_score.size(); ++ii)
	{
	    int y = cand_ijs(ii, 0);
	    int x = cand_ijs(ii, 1);
	    int s = cand_ijs(ii, 2);
	    traitsf::ref(gridijs_score, y, x, s) = cand_score(ii);
	    traitsb::ref(gridijs_mask, y, x, s) = 1;
	}

    }

    void get_subpixel_score(Float y, Float x, vector<Float>& sc) const {
	int ns = size3(); //gridijs_score.shape()[2];

	if( (y>=fy(fy.size()-1)) || (y<fy(0))
	    || (x>=fx(fx.size()-1)) || (x<fx(0)) )
	{
	    sc = scalar_vector<Float>(ns, default_value);
	    return;
	}

	int i0 = 0, i1 = fy.size()-1;
	while(i0+1<i1)
	{
	    int i2 = (i0+i1)/2;
	    if(y<fy(i2)) i1 = i2;
	    else i0 = i2;
	}
	int j0 = 0, j1 = fx.size()-1;
	while(j0+1<j1)
	{
	    int j2 = (j0+j1)/2;
	    if(x<fx(j2)) j1 = j2;
	    else j0 = j2;
	}
	Float ly = (y-fy(i0))/(fy(i1)-fy(i0));
	Float lx = (x-fx(j0))/(fx(j1)-fx(j0));
	if(sc.size()!=ns)   sc = vector<Float>(ns);
	//std::cout<<"en~~~, great!"<<std::endl;
	for(int ss=0; ss<ns; ++ss)
	{
	    sc(ss) = (1-ly)*(1-lx)*traitsf::ref(gridijs_score, i0, j0, ss) +
		(1-ly)*lx*traitsf::ref(gridijs_score, i0, j1, ss) +
		ly*(1-lx)*traitsf::ref(gridijs_score, i1, j0, ss) +
		ly*lx*traitsf::ref(gridijs_score, i1, j1, ss);
#if 0
	    if(gridijs_mask[i0][j0][ss] &&
	       gridijs_mask[i0][j1][ss] &&
	       gridijs_mask[i1][j0][ss] &&
	       gridijs_mask[i1][j1][ss]  )
#endif
#if 0
	    else sc(ss) = default_value;
#endif
	}

    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)	{

	ar & fx;
	ar & fy;
	ar & gridijs_mask;
	ar & gridijs_score;
	ar & default_value;

    }

};



void  enumerate_rects_inpoly(CImg<unsigned char> const& image,
			     pmodel_t const& pmodel, 
			     float feetx, float feety,
			     vector<float> const& xr, 
			     vector<float> const& yr,
			     vector<float> const& scales,
			     float horiz_mean, float horiz_sig,
			     matrix<double> const& poly_im,
			     matrix<float>& cand_rects,
			     vector<float>& cand_scale,
			     matrix<int>& cand_ijs,
			     candidate_array<float>& cand_array)
{

    int nx = xr.size();
    int ny = yr.size();
    int ns = scales.size();

    matrix<float> candr(nx*ny*ns, 4);
    vector<float> cands(nx*ny*ns);
    matrix<int> candi(nx*ny*ns, 3);

    //std::cout<<"candr.size="<<candr.size1()<<", "<<candr.size2()<<std::endl;

    vector<int> flag(scalar_vector<int>(nx*ny*ns, 1));

    int ic = 0;

    float hpre = pmodel.hpre;
    cand_array.fill_fxfy(feetx, feety, xr, yr, ns);

    for(int ii=0; ii<yr.size(); ++ii)
    {
	float dy = yr(ii);
	float cur_fy = dy+feety;
	float cur_hy = horiz_mean+hpre*(cur_fy-horiz_mean);

	//std::cout<<"hpre="<<hpre<<std::endl;


	float zoom = (cur_fy-cur_hy)/pmodel.bh;
	for(int jj=0; jj<xr.size(); ++jj)
	{
	    float dx = xr(jj);
	    float cur_fx = dx+feetx;
	    for(int kk=0; kk<scales.size(); ++kk)
	    {
		float ds = scales(kk)*zoom;
		float w = ds*pmodel.bw;
		float h = ds*pmodel.bh;

		vector<float> tmp(4);
		tmp <<= (cur_fx-w/2), cur_hy, (cur_fx+w/2), cur_fy;
		row(candr, ic) =tmp;

		cands(ic) = kk;

		if(cur_hy > horiz_mean+6*horiz_sig ||
		   cur_hy < horiz_mean-6*horiz_sig)
		{
// 		    std::cout<<"cur_hy="<<cur_hy<<", \t";
// 		    std::cout<<"horiz_mean="<<horiz_mean<<", \t";
// 		    std::cout<<"horiz_sig="<<horiz_sig<<std::endl;
		    flag(ic) = 0;
		}
#if 1
		else if(!point_in_polygon(row(poly_im, 0), row(poly_im, 1),
					  cur_fx, cur_fy))
		    flag(ic) = 0;
#endif

		candi(ic, 0) = ii;
		candi(ic, 1) = jj;
		candi(ic, 2) = kk;
		++ic;
	    }
	}
    }

    int num = sum(flag);

    cand_rects = matrix<float>(num, 4);
    cand_scale = vector<float>(num);
    cand_ijs = matrix<int>(num, 3);

    ic = 0;
    for(int ii=0; ii<flag.size(); ++ii)
    {
	if(!flag(ii)) continue;
	row(cand_rects, ic) = row(candr, ii);
	row(cand_ijs, ic) = row(candi, ii);
	cand_scale(ic) = cands(ii);
	ic ++;
    }

}

void rects_to_pmodel_geom(vector<float> const& rect,
			  float horiz_mean, 
			  pmodel_t& pmodel)
{
    pmodel.bw = rect(2)-rect(0);
    pmodel.bh = rect(3)-rect(1);

    pmodel.hpre = (rect(1)-horiz_mean)/(rect(3)-horiz_mean);
}


template <typename Vec>
inline typename Vec::value_type kldivergence(Vec const& hp,
				      Vec const& hq)
{
    using namespace boost::lambda;
    typedef typename Vec::value_type Float;
#if 0
    vector<Float> tmp(hp.size());
    std::transform(hp.begin(), hp.end(),
		   hq.begin(), tmp.begin(),
		   bind( (Float (*)(Float) )std::log, (_1+1e-6f)/(_2+1e-6f) )
	);
    Float v = inner_prod(hp, tmp);
    if(v<0) v = 0;
#endif
    Float (*flog)(Float) = std::log;
    Float v = std::inner_product(hp.begin(), hp.end(), hq.begin(), 0.0f, std::plus<Float>(),
				 _1*bind( flog, (_1+1e-6f)/(_2+1e-6f) ) );

    if(v<0) v = 0;

#if 0
    if(isnan(v)) {
	v = 0;
	std::cout<<"hp="<<hp<<std::endl;
	std::cout<<"hq="<<hq<<std::endl;
    }
#endif

    return v;
}

template <typename Float>
inline void kldivergence(matrix<Float> const& hp,
		  matrix<Float> const& hq, vector<Float> & v)
{
    int nf = hp.size1();
    if(nf!=v.size()) v = vector<Float>(nf);
    for(int ff=0; ff<nf; ++ff)
    {
	v(ff) = kldivergence(row(hp, ff), row(hq, ff));
    }
}

template <typename Float>
inline Float sat(Float v, Float u)
{
    if(v>u) return u;
    return v;
}

template <typename Float>
inline Float sat2(Float v, Float l, Float u)
{
    if(v>u) return u;
    if(v<l) return l;
    return v;
}

template <typename Float>
inline void sat(vector<Float>& v, Float u)
{
    using namespace boost::lambda;
    Float(*fsat)(Float, Float) = sat;
    std::for_each(v.begin(), v.end(), _1 = bind(fsat, _1, u) );
}


template <typename Vec>
inline typename Vec::value_type expected_llratio(Vec const& hp,
					  typename Vec::value_type ep,
					  Vec const& p,
					  Vec const& q)
{
    using namespace boost::lambda;
    typedef typename Vec::value_type Float;
    vector<Float> tmp(hp.size());
    std::transform(p.begin(), p.end(), q.begin(), tmp.begin(),
		   bind((Float(*)(Float))std::log, (_1+ep)/(_2+ep)));
    return inner_prod(tmp, hp);
}

template <typename Float>
inline void compute_consistent_score(matrix<Float> const& hist_p,
			      matrix<Float> const& hist_q,
			      matrix<Float> const& p,
			      matrix<Float> const& q,
			      Float ep,
			      vector<Float>& cs)
{
    int np = p.size1();
    if(np!=cs.size())    cs = vector<Float>(np);
    for(int pp=0; pp<np; ++pp)
    {
	float v1 = expected_llratio(row(hist_p, pp), ep,
				    row(p, pp), row(q, pp) );
	float v2 = expected_llratio(row(p, pp), ep,
				    row(hist_p, pp), row(hist_q, pp));
	cs(pp) = sat((v1+v2)/2, 2.5f)*2;
    }
}

template <typename Float>
void get_cand_hist_score(CImg<unsigned char> const& image,
			 vector<array<float, 4> > const& model,
			 vector<float> const& logp1,
			 vector<float> const& logp2,
			 matrix<Float> const& p,
			 matrix<Float> const& q,
			 matrix<Float> const& cand_rects,
			 vector<Float>& score_map,
			 matrix<Float>& feature_scores)
{
    using namespace boost::lambda;

    int nc = cand_rects.size1();
    Float ep = 1e-4;
    score_map = scalar_vector<Float>(nc, 0);
    feature_scores = scalar_matrix<Float>(nc, 6, 0);

    Float wk = 0.5;
    Float wc = 2;

    Float (*flog)(Float) = std::log;
    Float (*fexp)(Float) = std::exp;


    vector<Float> kll;
    vector<Float> cs;
    for(int ii=0; ii<nc; ++ii)
    {
	//real_timer_t timer;

	Float x = cand_rects(ii, 0);
	Float y = cand_rects(ii, 1);
	Float w = cand_rects(ii, 2) - x;
	Float h = cand_rects(ii, 3) - y;
	matrix<Float> rects;
	compute_part_rects(x, y, w, h, model, rects);

	matrix<Float> hist_p;
	matrix<Float> hist_q;

	collect_hist(image, rects, hist_p, hist_q);

	kldivergence(hist_p, hist_q, kll);

	sat(kll, 6.0f);
	kll *= 0.56f;

	compute_consistent_score(hist_p, hist_q, p, q, ep, cs);

	vector<Float> lscore1(  wk*kll+wc*cs+logp1  );
	vector<Float> lscore2(  wk*kll/2    +logp2  );
	vector<Float> lscore0(  lscore1.size()     );
	Float const& (*fmax)(Float const&, Float const&) = std::max;
	std::transform(lscore1.begin(), lscore1.end(),
		       lscore2.begin(), lscore0.begin(),
		       fmax);

	vector<Float> combined_scores(p.size1());
	lscore1 -= lscore0;
	lscore2 -= lscore0;
	std::transform(lscore1.begin(), lscore1.end(),
		       lscore2.begin(), combined_scores.begin(),
		       bind(flog, bind(fexp, _1)+bind(fexp, _2) ));
	combined_scores += lscore0;

	score_map(ii) = sum(combined_scores);
	//std::cout<<"combined_scores="<<combined_scores<<std::endl;

	vector<Float> fv(kll.size()+cs.size());
	project(fv, range(0, kll.size()) ) = kll;
	project(fv, range(kll.size(), fv.size()) ) = cs;
	row(feature_scores, ii) = fv;
	//std::cout<<"\t\t\tcollect_hist time:"<<timer.elapsed()/1000.0f<<"s."<<std::endl;

    }

}

template <class Float>
void  compute_extbox(CImg<unsigned char> const& image,
		    matrix<Float> const & rects,
		    matrix<int>& exbbi,
		    matrix<int>& inbbi )
{
    using namespace boost::lambda;
    int np = rects.size1();
    vector<Float> rcx((column(rects, 0)+column(rects, 2))/2);
    vector<Float> rcy((column(rects, 1)+column(rects, 3))/2);
    vector<Float> rw(column(rects, 2)-column(rects, 0));
    vector<Float> rh(column(rects, 3)-column(rects, 1));

    matrix<Float> exbb(np, 4);

    column(exbb, 0) = rcx-rw/2*2;
    column(exbb, 1) = rcy-rh/2*1.5;
    column(exbb, 2) = rcx+rw/2*2;
    column(exbb, 3) = rcy+rh/2*1.5;

    exbbi=matrix<int> (np, 4);
    inbbi=matrix<int> (np, 4);

    array2d_transform(exbb, exbbi, ll_static_cast<int>(_1+0.5));
    array2d_transform(rects, inbbi, ll_static_cast<int>(_1+0.5));

    typedef cvpr::array3d_traits<CImg<unsigned char> > traits;
    int s1 = traits::size2(image);
    int s2 = traits::size3(image);

    for(int ii=0; ii<np; ++ii)
    {
	if(exbbi(ii, 0)<0) exbbi(ii, 0) = 0;
	if(exbbi(ii, 1)<0) exbbi(ii, 1) = 0;
	if(exbbi(ii, 2)>s2-1) exbbi(ii, 2) = s2-1;
	if(exbbi(ii, 3)>s1-1) exbbi(ii, 3) = s1-1;

	if(inbbi(ii, 0)<0) inbbi(ii, 0) = 0;
	if(inbbi(ii, 1)<0) inbbi(ii, 1) = 0;
	if(inbbi(ii, 2)>s2-1) inbbi(ii, 2) = s2-1;
	if(inbbi(ii, 3)>s1-1) inbbi(ii, 3) = s1-1;
    }

}

template <class Int> 
void compute_integral_histogram(CImg<unsigned char> const& image,
				vector<int> const& outbox,
				multi_array<Int, 3>& int_hist,
				int sub=4)
{
    
    typedef cvpr::array3d_traits<CImg<unsigned char> > traits;
    int s1 = traits::size2(image)/sub;
    int s2 = traits::size3(image)/sub;
    int ss1 = 1+(outbox(3)-outbox(1))/sub;
    int ss2 = 1+(outbox(2)-outbox(0))/sub;

    //std::cout<<"outbox="<<outbox<<std::endl;
    //std::cout<<"ss="<<ss1<<", "<<ss2<<std::endl;

    int_hist.resize(extents[ss1+1][ss2+1][8*8*8]);
    multi_array<Int, 3> histr(extents[ss1][ss2][8*8*8]);
    multi_array<Int, 3> histc(extents[ss1][ss2][8*8*8]);

    vector<int> iy(ss1);
    for(int yy=0; yy<ss1; ++yy)
    {
	iy(yy) = yy*sub+outbox(1);
    }
    vector<int> ix(ss2);
    for(int xx=0; xx<ss2; ++xx)
    {
	ix(xx) = xx*sub+outbox(0);
    }
    matrix<int> ibin(ss1, ss2);

    for(int yy=0; yy<ss1; ++yy)
    {
	for(int xx=0; xx<ss2; ++xx)
	{
	    int ir = static_cast<int>(traits::ref(image, 0, iy(yy), ix(xx))/32);
	    int ig = static_cast<int>(traits::ref(image, 1, iy(yy), ix(xx))/32);
	    int ib = static_cast<int>(traits::ref(image, 2, iy(yy), ix(xx))/32);
	    ibin(yy, xx) = ir+ig*8+ib*8*8;
	}
    }

    for(int yy=0; yy<ss1; ++yy)
    {
	for(int bb=0; bb<histr.shape()[2]; ++bb)
	{
	    histr[yy][0][bb] = 0;
	}
	int ib = ibin(yy, 0);
	histr[yy][0][ib] = 1;

	for(int xx=1; xx<ss2; ++xx)
	{
	    for(int bb=0; bb<histr.shape()[2]; ++bb)
	    {
		histr[yy][xx][bb] = histr[yy][xx-1][bb];
	    }
	    int ib = ibin(yy, xx);
	    histr[yy][xx][ib] ++;
	}
    }
    for(int xx=0; xx<ss2; ++xx)
    {
	for(int bb=0; bb<histr.shape()[2]; ++bb)
	{
	    histc[0][xx][bb] = 0;
	}
	int ib = ibin(0, xx);
	histc[0][xx][ib] ++;

	for(int yy=1; yy<ss1; ++yy)
	{
	    for(int bb=0; bb<histr.shape()[2]; ++bb)
	    {
		histc[yy][xx][bb] = histc[yy-1][xx][bb];
	    }
	    int ib = ibin(yy, xx);
	    histc[yy][xx][ib] ++;
	}
    }

    for(int yy=-1; yy<ss1; ++yy)
    {
	for(int bb=0; bb<histr.shape()[2]; ++bb)
	{
	    int_hist[1+yy][0][bb] = 0;
	}
    }
    for(int xx=0; xx<ss2; ++xx)
    {
	for(int bb=0; bb<histr.shape()[2]; ++bb)
	{
	    int_hist[0][1+xx][bb] = 0;
	}
    }
    for(int yy=0; yy<ss1; ++yy)
    {
	for(int xx=1; xx<ss2; ++xx)
	{
	    for(int bb=0; bb<histr.shape()[2]; ++bb)
	    {
		int_hist[1+yy][1+xx][bb] = histr[yy][xx][bb]+histc[yy][xx][bb]
		    + int_hist[yy][xx][bb];
	    }
	    int ib = ibin(yy, xx);
	    int_hist[1+yy][1+xx][ib] --;
	}
    }

}

//collect hist for integral histogram
template <class Float>
void collect_hist(multi_array<int, 3> const& int_hist,
		  vector<int> const& outbox,
		  int sub,
		  matrix<int> const& inbb,
		  matrix<int> const& exbb,
		  matrix<Float>& hist_p,
		  matrix<Float>& hist_q)
{
    int np = inbb.size1();
    int x0 = outbox(0);
    int y0 = outbox(1);

    hist_p = scalar_matrix<Float>(np, 8*8*8, 0);
    hist_q = scalar_matrix<Float>(np, 8*8*8, 0);

    for(int pp=0; pp<np; ++pp)
    {
	std::cout<<"outbox="<<outbox<<std::endl;
	std::cout<<"inbb="<<row(inbb, pp)<<std::endl;
	std::cout<<"int_hist.size="<<int_hist.shape()[0]
	  <<", "<<int_hist.shape()[1]
	  <<", "<<int_hist.shape()[2]<<std::endl;
	int ex_x0 = (exbb(pp, 0)-x0)/sub;
	int ex_y0 = (exbb(pp, 1)-y0)/sub;
	int ex_x1 = (exbb(pp, 2)-x0)/sub;
	int ex_y1 = (exbb(pp, 3)-y0)/sub;

	int in_x0 = (inbb(pp, 0)-x0)/sub;
	int in_y0 = (inbb(pp, 1)-y0)/sub;
	int in_x1 = (inbb(pp, 2)-x0)/sub;
	int in_y1 = (inbb(pp, 3)-y0)/sub;
	std::cout<<ex_x0<<","<<ex_y0<<","<<ex_x1<<","<<ex_y1<<std::endl;
	std::cout<<in_x0<<","<<in_y0<<","<<in_x1<<","<<in_y1<<std::endl;

	for(int bb=0; bb<int_hist.shape()[2]; ++bb)
	{
	    hist_q(pp, bb) = (Float)(   int_hist[ex_y1+1][ex_x1+1][bb]
				      + int_hist[ex_y0][ex_x0][bb]
				      - int_hist[ex_y1+1][ex_x0][bb]
				      - int_hist[ex_y0][ex_x1+1][bb]);
	    hist_p(pp, bb) = (Float)(   int_hist[in_y1+1][in_x1+1][bb]
				      + int_hist[in_y0][in_x0][bb]
				      - int_hist[in_y1+1][in_x0][bb]
				      - int_hist[in_y0][in_x1+1][bb]);
	    hist_q(pp, bb) -= hist_p(pp, bb);
	}
	row(hist_q, pp) /= sum(row(hist_q, pp));
	row(hist_p, pp) /= sum(row(hist_p, pp));
    }
}

template <typename Float>
void get_cand_hist_score2(CImg<unsigned char> const& image,
			  vector<array<float, 4> > const& model,
			  vector<float> const& logp1,
			  vector<float> const& logp2,
			  matrix<Float> const& p,
			  matrix<Float> const& q,
			  matrix<Float> const& cand_rects,
			  vector<Float>& score_map,
			  matrix<Float>& feature_scores)
{
    using namespace boost::lambda;

    int nc = cand_rects.size1();
    Float ep = 1e-4;
    score_map = scalar_vector<Float>(nc, 0);
    feature_scores = scalar_matrix<Float>(nc, 6, 0);

    Float wk = 0.5;
    Float wc = 2;

    Float (*flog)(Float) = std::log;
    Float (*fexp)(Float) = std::exp;

    vector<matrix<int> > part_exbb(nc);
    vector<matrix<int> > part_inbb(nc);
    vector<int> outbox(4);
    for(int ii=0; ii<nc; ++ii)
    {
	//real_timer_t timer;

	Float x = cand_rects(ii, 0);
	Float y = cand_rects(ii, 1);
	Float w = cand_rects(ii, 2) - x;
	Float h = cand_rects(ii, 3) - y;
	matrix<Float> rects;
	compute_part_rects(x, y, w, h, model, rects);
	matrix<int>& exbbi=part_exbb(ii);
	matrix<int>& inbbi=part_inbb(ii);
	compute_extbox(image, rects, exbbi, inbbi);

	vector<int> x0a(column(exbbi, 0));
	int x0 = *std::min_element(x0a.begin(), x0a.end());
	vector<int> y0a(column(exbbi, 1));
	int y0 = *std::min_element(y0a.begin(), y0a.end());
	vector<int> x1a(column(exbbi, 2));
	int x1 = *std::max_element(x1a.begin(), x1a.end());
	vector<int> y1a(column(exbbi, 3));
	int y1 = *std::max_element(y1a.begin(), y1a.end());
	if(ii==0) outbox <<= x0, y0, x1, y1;
	else 
	{
	    outbox(0) = std::min(x0, outbox(0));
	    outbox(1) = std::min(y0, outbox(1));
	    outbox(2) = std::max(x1, outbox(2));
	    outbox(3) = std::max(y1, outbox(3));
	}
    }
    int sub = (outbox(2)-outbox(0))/40;
    if(sub < 1) sub = 1;
    multi_array<int, 3> int_hist;
    compute_integral_histogram(image, outbox, int_hist, sub);

    for(int ii=0; ii<nc; ++ii)
    {
	matrix<Float> hist_p;
	matrix<Float> hist_q;

	// todo collect_hist(image, rects, hist_p, hist_q);
	collect_hist(int_hist, outbox, sub, part_inbb(ii), part_exbb(ii), hist_p, hist_q);

	vector<Float> kll;
	kldivergence(hist_p, hist_q, kll);

	sat(kll, 6.0f);
	kll *= 0.56f;
	//float contrast_score = sum(kll);
	//std::cout<<"kll="<<kll<<std::endl;

	vector<Float> cs;
	compute_consistent_score(hist_p, hist_q, p, q, ep, cs);
	//std::cout<<"cs="<<cs<<std::endl;

	//float consistent_score = sum(cs);
	//std::cout<<"kll.size="<<kll.size()<<", cs.size="<<cs.size()<<", logp1.size="<<logp1.size()<<std::endl;
	vector<Float> lscore1(  wk*kll+wc*cs+logp1  );
	vector<Float> lscore2(  wk*kll/2    +logp2  );
	vector<Float> lscore0(  lscore1.size()     );
	Float const& (*fmax)(Float const&, Float const&) = std::max;
	std::transform(lscore1.begin(), lscore1.end(),
		       lscore2.begin(), lscore0.begin(),
		       bind(fmax, _1, _2));
	//std::cout<<"lscore1="<<lscore1<<std::endl;
	//std::cout<<"lscore2="<<lscore2<<std::endl;
	//std::cout<<"lscore0="<<lscore0<<std::endl;

	vector<Float> combined_scores(p.size1());
	lscore1 -= lscore0;
	lscore2 -= lscore0;
	std::transform(lscore1.begin(), lscore1.end(),
		       lscore2.begin(), combined_scores.begin(),
		       bind(flog, bind(fexp, _1)+bind(fexp, _2) ));
	combined_scores += lscore0;

	score_map(ii) = sum(combined_scores);
	//std::cout<<"combined_scores="<<combined_scores<<std::endl;

	vector<Float> fv(kll.size()+cs.size());
	project(fv, range(0, kll.size()) ) = kll;
	project(fv, range(kll.size(), fv.size()) ) = cs;
	row(feature_scores, ii) = fv;
	//std::cout<<"\t\t\tcollect_hist time:"<<timer.elapsed()/1000.0f<<"s."<<std::endl;

    }

}


template <typename Float>
struct ground_scoremap_t
{
    int x0;
    int y0;
    multi_array<Float, 3> scores;

    void init(Float minx, Float maxx, Float miny, Float maxy, int ns) {
	x0 = static_cast<int>(minx);
	y0 = static_cast<int>(miny);
	int x1 = static_cast<int>(maxx+1);
	int y1 = static_cast<int>(maxy+1);
	scores.resize(extents[y1-y0][x1-x0][ns]);
    }
    int y1() const { return y0+scores.shape()[0]; }
    int x1() const { return x0+scores.shape()[1]; }

    void peak(int& py, int& px, int& ps) const {
	std::size_t i1, i2, i3;

	array3d_max(scores, i1, i2, i3);
	//std::cout<<" scores.size="<<scores.shape()[0]<<","
	//	 <<scores.shape()[1]<<","
	//	 <<scores.shape()[2]<<std::endl;
	//array3d_print(std::cout, scores);
	//std::cout<<" i1="<<i1<<", i2="<<i2<<", i3="<<i3<<std::endl;
	py = y0+i1; px = x0+i2; ps = i3;
    }

};

template <typename Float>
void combine_ground_score(vector<candidate_array<Float> > const& cand_array,
			   ground_scoremap_t<Float> & grd_scoremap,
			   geometric_info_t const& gi)
{
    using namespace boost::lambda;
    int Ncam = cand_array.size();
    int ns = cand_array(0).size3();

    vector<vector<double> > grbd_x(Ncam), grbd_y(Ncam);
    Float maxx = gi.ground_lim.xmin;
    Float minx = gi.ground_lim.xmax;
    Float maxy = gi.ground_lim.ymin;
    Float miny = gi.ground_lim.ymax;

    real_timer_t timer;
    //std::cout<<"Begin=========================="<<std::endl;
    for(int cam=0; cam < Ncam; ++cam)
    {
	candidate_array<Float> const& ca=cand_array(cam);
	Float x0 = ca.fx(0);
	Float x1 = ca.fx(ca.fx.size()-1);
	Float y0 = ca.fy(0);
	Float y1 = ca.fy(ca.fy.size()-1);
	vector<double> imbd_x(4), imbd_y(4);
	imbd_x <<= x0, x1, x1, x0;
	imbd_y <<= y0, y0, y1, y1;
	apply_homography(gi.img2grd(cam), imbd_x, imbd_y, grbd_x(cam), grbd_y(cam));
	//std::cout<<"imbdx="<<imbd_x<<std::endl;
	//std::cout<<"imbdy="<<imbd_y<<std::endl;

	//std::cout<<"grbdx="<<grbd_x(cam)<<std::endl;
	//std::cout<<"grbdy="<<grbd_y(cam)<<std::endl;

	maxx = std::max(maxx, static_cast<Float>(*std::max_element(grbd_x(cam).begin(),
								   grbd_x(cam).end())));
	maxy = std::max(maxy, static_cast<Float>(*std::max_element(grbd_y(cam).begin(),
								   grbd_y(cam).end())));
	minx = std::min(minx, static_cast<Float>(*std::min_element(grbd_x(cam).begin(),
								   grbd_x(cam).end())));
	miny = std::min(miny, static_cast<Float>(*std::min_element(grbd_y(cam).begin(),
								   grbd_y(cam).end())));
    }
    if(maxx> gi.ground_lim.xmax) maxx = gi.ground_lim.xmax;
    if(maxy> gi.ground_lim.ymax) maxy = gi.ground_lim.ymax;
    if(minx< gi.ground_lim.xmin) minx = gi.ground_lim.xmin;
    if(miny< gi.ground_lim.ymin) miny = gi.ground_lim.ymin;

    //std::cout<<"maxx="<<maxx<<", maxy="<<maxy<<std::endl;
    //std::cout<<"minx="<<minx<<", miny="<<miny<<std::endl;
    //std::cout<<"area="<<(maxx-minx+1)*(maxy-miny+1)<<std::endl;

    grd_scoremap.init(minx, maxx, miny, maxy, ns);

    int gy0 = grd_scoremap.y0;
    int gy1 = grd_scoremap.y1();
    int gx0 = grd_scoremap.x0;
    int gx1 = grd_scoremap.x1();

    //std::cout<<"gx0="<<gx0<<", gx1="<<gx1<<std::endl;
    //std::cout<<"gy0="<<gy0<<", gy1="<<gy1<<std::endl;

    multi_array<float, 4> wrap_score(extents[Ncam][gy1-gy0][gx1-gx0][ns]);
    for(int cam=0; cam<Ncam; ++cam)
    {
	for(int yy=gy0; yy<gy1; ++yy)
	{
	    int ii = yy-gy0;
	    vector<double> grx(gx1-gx0);
	    vector<double> gry(scalar_vector<float>(grx.size(), yy));
	    std::transform(counting_iterator<int>(gx0),
			   counting_iterator<int>(gx1),
			   grx.begin(), ll_static_cast<double>(_1));
	    vector<double> imx, imy;
	    apply_homography(gi.grd2img(cam), grx, gry, imx, imy);
	    //std::cout<<"grx="<<grx<<std::endl;
	    //std::cout<<"gry="<<gry<<std::endl;
	    //std::cout<<"imx="<<imx<<std::endl;
	    //std::cout<<"imy="<<imy<<std::endl;
	    //std::cout<<"fx="<<cand_array(cam).fx<<std::endl;
	    //std::cout<<"fy="<<cand_array(cam).fy<<std::endl;

	    for(int xx=gx0; xx<gx1; ++xx)
	    {
		int jj = xx-gx0;
		Float imxx = imx(jj);
		Float imyy = imy(jj);
		vector<Float> sc(ns);
		cand_array(cam).get_subpixel_score(imyy, imxx, sc);
		for(int ss=0; ss<ns; ++ss)
		    wrap_score[cam][ii][jj][ss] = sc(ss);
	    }
	}
    }

    //std::cout<<" scoremap="<<std::endl;
    for(int yy=0; yy<grd_scoremap.scores.shape()[0]; ++yy)
    {
	for(int xx=0; xx<grd_scoremap.scores.shape()[1]; ++xx)
	{
	    for(int ss=0; ss<grd_scoremap.scores.shape()[2]; ++ss)
	    {
		grd_scoremap.scores[yy][xx][ss] = 0;
		for(int cam=0; cam<Ncam; ++cam)
		{
		    grd_scoremap.scores[yy][xx][ss] += wrap_score[cam][yy][xx][ss];
		}
		//std::cout<<grd_scoremap.scores[yy][xx][ss]<<", ";
	    }
	}
    }

    //std::cout<<std::endl;
    std::cout<<"\t\t"<<(maxx-minx+1)*(maxy-miny+1)
	     <<" ground points time: "<<timer.elapsed()/1000.0f<<"s."<<std::endl;

}


#endif
