#ifndef __TRACK__EXISTED__OBJECTS__IMPL__HPP__INCLUDED__
#define __TRACK__EXISTED__OBJECTS__IMPL__HPP__INCLUDED__

template <typename Float>
int determine_trlet_state(vector<Float> const& scores, parameter_t const& P)
{
    int Ncam = scores.size();
    int state=1;
    if(Ncam==1)
    {
	if(isnan(scores(0)) || scores(0) < P.occl_thr1)
	{
	    state = 2;
	}
    }
    else
    {
	vector<int> val(Ncam*2);
	bool hasnan = false;
	for(int cam=0; cam<Ncam; ++cam)
	{
	    if( isnan(scores(cam)) ) 
	    {
		hasnan = true;
		break;
	    }
	    val(cam*2+0) = (scores(cam) >= P.occl_thr1);
	    val(cam*2+1) = (scores(cam) >= P.occl_thr2);
	}
	if(hasnan || sum(val)<3)
	{
	    state = 2;
	}
    }
    return state;

}

template <typename Float>
void track_one(parameter_t const& P,  directory_structure_t const& ds,
	       geometric_info_t const& gi, object_info_t & oi,
	       vector<std::vector<std::string> > const &seq,
	       int tt, int cam, int nn,
	       vector<CImg<unsigned char> > const& images,
	       candidate_array<Float>& cand_array,
	       vector<matrix<float> > const& grd )
{
    vector<object_trj_t> & trlet_list=oi.trlet_list;

    float xrange = P.xrange;
    float yrange = P.yrange;
    float xstep = P.xstep;
    float ystep = P.ystep;

    if(tt>0)
    {
	if(trlet_list(nn).scores(cam, tt-1)<P.thr)
	{
	    xrange *= 2;
	    yrange *=2;
	    xstep *= 2;
	    ystep *= 2;
	}
    }
    std::vector<float> sxr, syr;
    for(float v=-xrange; v<=xrange; v+=xstep)
    {
	sxr.push_back(v);
    }
    for(float v=-yrange; v<=yrange; v+=ystep)
    {
	syr.push_back(v);
    }
    vector<float> xr(sxr.size()), yr(syr.size());
    std::copy(sxr.begin(), sxr.end(), xr.begin());
    std::copy(syr.begin(), syr.end(), yr.begin());

    float feetx = (trlet_list(nn).trj(cam)(tt-1, 0)
		   +trlet_list(nn).trj(cam)(tt-1, 2))/2;
    float feety = trlet_list(nn).trj(cam)(tt-1, 3);

    matrix<Float> cand_rects;
    vector<Float> cand_scale;
    matrix<int> cand_ijs;

    enumerate_rects_inpoly(images(cam), oi.pmodel_list(cam, nn),
			   feetx, feety,
			   xr, yr, P.scales, gi.horiz_mean, gi.horiz_sig,
			   gi.polys_im(cam),
			   cand_rects, cand_scale,
			   cand_ijs, cand_array);

    real_timer_t timer;
    vector<Float> cand_hist_score;
    matrix<Float> hist_fscores;

    get_cand_hist_score(images(cam), oi.model, P.logp1, P.logp2,
			trlet_list(nn).hist_p(cam),
			trlet_list(nn).hist_q(cam),
			cand_rects,
			cand_hist_score, hist_fscores);

    vector<Float> cand_score=cand_hist_score;
    int idx_max = std::max_element(cand_score.begin(), cand_score.end())
	- cand_score.begin();
    
    std::cout<<"\t\t"<<cand_rects.size1()<<" rects, \tget_cand_hist_score time:"
	     <<timer.elapsed()/1000.0f<<"s."<<std::endl;

    row(trlet_list(nn).trj(cam), tt) = row(cand_rects, idx_max);
    trlet_list(nn).scores(cam, tt) = cand_score(idx_max); //max_score_npl;
    column(trlet_list(nn).fscores(cam), tt) = row(hist_fscores, idx_max);

    cand_array.fill_score(cand_score, cand_ijs);

}

template <typename Float>
void update_one(parameter_t const& P,  directory_structure_t const& ds,
	       geometric_info_t const& gi, object_info_t & oi,
	       vector<std::vector<std::string> > const &seq,
	       int tt, int cam, int nn,
	       vector<CImg<unsigned char> > const& images,
	       vector<matrix<float> > const& grd )
{

    object_trj_t& trlet = oi.trlet_list(nn);

    Float ww = trlet.trj(cam)(tt, 2) - trlet.trj(cam)(tt, 0);
    Float hh = trlet.trj(cam)(tt, 3) - trlet.trj(cam)(tt, 1);

    matrix<Float> rects;
    compute_part_rects(trlet.trj(cam)(tt, 0), trlet.trj(cam)(tt, 1),
		       ww, hh, oi.model, rects);
    Float max_score_npl = trlet.scores(cam, tt);
    Float fglr2 = P.fglr/(1+std::exp(P.thr-max_score_npl));
    matrix<Float> hist_p, hist_q;
    collect_hist(images(cam), rects, hist_p, hist_q);
    trlet.hist_p(cam) = hist_p*fglr2+trlet.hist_p(cam)*(1-fglr2);
    trlet.hist_q(cam) = hist_q*P.bglr+trlet.hist_q(cam)*(1-P.bglr);


}


#ifdef USE_MPI
template <typename Float>
void track_one(mpi::communicator& world,
	       parameter_t const& P,  directory_structure_t const& ds,
	       geometric_info_t const& gi, object_info_t & oi,
	       vector<std::vector<std::string> > const &seq,
	       int tt, int cam, int nn,
	       vector<CImg<unsigned char> > const& images,
	       candidate_array<Float>& cand_array,
	       vector<matrix<float> > const& grd )
{
    vector<object_trj_t> & trlet_list=oi.trlet_list;
    matrix<Float> cand_rects;
    vector<Float> cand_scale;
    matrix<int> cand_ijs;

    if(0==world.rank())
    {
	float xrange = P.xrange;
	float yrange = P.yrange;
	float xstep = P.xstep;
	float ystep = P.ystep;

	if(tt>0)
	{
	    if(trlet_list(nn).scores(cam, tt-1)<P.thr)
	    {
		xrange *= 2;
		yrange *=2;
		xstep *= 2;
		ystep *= 2;
	    }
	}
	std::vector<float> sxr, syr;
	for(float v=-xrange; v<=xrange; v+=xstep)
	{
	    sxr.push_back(v);
	}
	for(float v=-yrange; v<=yrange; v+=ystep)
	{
	    syr.push_back(v);
	}
	vector<float> xr(sxr.size()), yr(syr.size());
	std::copy(sxr.begin(), sxr.end(), xr.begin());
	std::copy(syr.begin(), syr.end(), yr.begin());

	float feetx = (trlet_list(nn).trj(cam)(tt-1, 0)
		       +trlet_list(nn).trj(cam)(tt-1, 2))/2;
	float feety = trlet_list(nn).trj(cam)(tt-1, 3);

	enumerate_rects_inpoly(images(cam), oi.pmodel_list(cam, nn),
			       feetx, feety,
			       xr, yr, P.scales, gi.horiz_mean, gi.horiz_sig,
			       gi.polys_im(cam),
			       cand_rects, cand_scale,
			       cand_ijs, cand_array);
    }
    mpi::broadcast(world, cand_rects, 0);


    real_timer_t timer;
    vector<Float> cand_hist_score(cand_rects.size1());
    matrix<Float> hist_fscores;

    range rrank(world.rank()*cand_rects.size1()/world.size(), 
		(world.rank()+1)*cand_rects.size1()/world.size());
    matrix<Float> cand_rects_rank(project(cand_rects, rrank, range(0, 4)));
    vector<Float> cand_hist_score_rank;
    matrix<Float> hist_fscores_rank;
    get_cand_hist_score(images(cam), oi.model, P.logp1, P.logp2,
			trlet_list(nn).hist_p(cam),
			trlet_list(nn).hist_q(cam),
			cand_rects_rank,
			cand_hist_score_rank, hist_fscores_rank);
    if(world.rank()==0)
    {
	std::vector<vector<Float> > v1;
	std::vector<matrix<Float> > v2;
	mpi::gather(world, cand_hist_score_rank, v1, 0);
	mpi::gather(world, hist_fscores_rank, v2, 0);
	hist_fscores = matrix<Float>(cand_rects.size1(),
				     hist_fscores_rank.size2());
	for(int r=0; r<world.size(); ++r)
	{
	    int start = r*cand_rects.size1()/world.size();
	    for(int vv=0; vv<v1[r].size(); ++vv)
	    {
		cand_hist_score(start+vv) = v1[r](vv);
	    }
	    for(int vv=0; vv<v2[r].size1(); ++vv)
	    {
		row(hist_fscores, start+vv) = row(v2[r], vv);
	    }
	}
    }
    else
    {
	mpi::gather(world, cand_hist_score_rank, 0);
	mpi::gather(world, hist_fscores_rank, 0);
    }

    if(world.rank()==0)
	std::cout<<"\t\t"<<cand_rects.size1()<<" rects, \tget_cand_hist_score time:"
		 <<timer.elapsed()/1000.0f<<"s."<<std::endl;


    mpi::broadcast(world, cand_hist_score, 0);
    mpi::broadcast(world, hist_fscores, 0);

    if(0==world.rank())
    {
	vector<Float> cand_score=cand_hist_score;
	int idx_max = std::max_element(cand_score.begin(), cand_score.end())
	    - cand_score.begin();

	column(trlet_list(nn).fscores(cam), tt) = row(hist_fscores, idx_max);
	row(trlet_list(nn).trj(cam), tt) = row(cand_rects, idx_max);
	trlet_list(nn).scores(cam, tt) = cand_score(idx_max);//max_score_npl;

	cand_array.fill_score(cand_score, cand_ijs);

    }
    mpi::broadcast(world, cand_array, 0);
    mpi::broadcast(world, trlet_list(nn).scores(cam, tt), 0);

    vector<Float> fscore_col;
    vector<Float> trj_row;
    if(0==world.rank())
    {
	fscore_col = column(trlet_list(nn).fscores(cam), tt);
	trj_row = row(trlet_list(nn).trj(cam), tt);
    }
    mpi::broadcast(world, fscore_col, 0);
    mpi::broadcast(world, trj_row, 0);
    if(0!=world.rank())
    {
	column(trlet_list(nn).fscores(cam), tt) = fscore_col;
	row(trlet_list(nn).trj(cam), tt) = trj_row;
    }

}

template <typename Float>
void update_one(mpi::communicator& world,
	       parameter_t const& P,  directory_structure_t const& ds,
	       geometric_info_t const& gi, object_info_t & oi,
	       vector<std::vector<std::string> > const &seq,
	       int tt, int cam, int nn,
	       vector<CImg<unsigned char> > const& images,
	       vector<matrix<float> > const& grd )
{

    object_trj_t& trlet = oi.trlet_list(nn);
    if(0==world.rank())
    {
	Float ww = trlet.trj(cam)(tt, 2) - trlet.trj(cam)(tt, 0);
	Float hh = trlet.trj(cam)(tt, 3) - trlet.trj(cam)(tt, 1);

	matrix<Float> rects;
	compute_part_rects(trlet.trj(cam)(tt, 0), trlet.trj(cam)(tt, 1),
			   ww, hh, oi.model, rects);

	Float max_score_npl = trlet.scores(cam, tt);
	Float fglr2 = P.fglr/(1+std::exp(P.thr-max_score_npl));
	matrix<Float> hist_p, hist_q;
	collect_hist(images(cam), rects, hist_p, hist_q);
	trlet.hist_p(cam) = hist_p*fglr2+trlet.hist_p(cam)*(1-fglr2);
	trlet.hist_q(cam) = hist_q*P.bglr+trlet.hist_q(cam)*(1-P.bglr);

    }
    mpi::broadcast(world, trlet.hist_p(cam), 0);
    mpi::broadcast(world, trlet.hist_q(cam), 0);
}

#endif


template <typename Float>
void track_existed_objects(parameter_t const& P,  directory_structure_t const& ds,
			 geometric_info_t const& gi, object_info_t & oi,
    			 vector<std::vector<std::string> > const &seq,
			 int tt,
			 vector<CImg<unsigned char> > const& images,
			 vector<matrix<float> > const& grd )
{
    int Ncam = seq.size();
    vector<object_trj_t> & trlet_list=oi.trlet_list;
    int nobj = trlet_list.size();
    int T = seq[0].size();
    int np = oi.model.size();

    int num_scales = P.scales.size();

    for(int nn=0; nn<oi.curr_num_obj; ++nn)
    {
	if(trlet_list(nn).trj.size()==0)
	    continue;
	if(trlet_list(nn).state>=2)
	   continue;

	vector<candidate_array<Float> > cand_array(Ncam);

	for(int cam=0; cam<Ncam; ++cam)
	{
	    pmodel_t const& pmodel = oi.pmodel_list(cam, nn);
	    smodel_t const& smodel = oi.smodel_list(cam, nn);
	    track_one(P, ds, gi, oi, seq, tt, cam, nn, images, cand_array(cam), grd);
	}
	ground_scoremap_t<Float> grd_scoremap;
	combine_ground_score(cand_array, grd_scoremap, gi);
	int best_y, best_x, best_s;
	grd_scoremap.peak(best_y, best_x, best_s);	

	for(int cam=0; cam<Ncam; ++cam)
	{
	    vector<double> bx(1), by(1), ix, iy;
	    bx <<= best_x; by <<= best_y;
	    apply_homography(gi.grd2img(cam), bx, by, ix, iy);
	    float hpre = oi.pmodel_list(cam, nn).hpre;
	    float cur_fy = iy(0);
	    float cur_fx = ix(0);
	    float cur_hy = gi.horiz_mean+hpre*(cur_fy-gi.horiz_mean);
	    float ds = P.scales(best_s)*(cur_fy-cur_hy)/oi.pmodel_list(cam, nn).bh;
	    float ww = ds*oi.pmodel_list(cam, nn).bw;
	    float hh = cur_fy - cur_hy;

	    vector<Float> tmp(4);
	    tmp <<= (cur_fx-ww/2), cur_hy, (cur_fx+ww/2), cur_fy;
	    row(trlet_list(nn).trj(cam), tt) = tmp;

	}//endfor cam

	vector<Float> scores(column(trlet_list(nn).scores, tt));
	trlet_list(nn).state = determine_trlet_state(scores, P);

	if(trlet_list(nn).state <= 1)
	{
	    trlet_list(nn).endt = tt;
	    for(int cam=0; cam <Ncam; ++cam)
	    {
		update_one<Float>(P, ds, gi, oi, seq, tt, cam, nn, images, grd );
	    }

	}
	if(trlet_list(nn).state == 2)
	{
	    for(int cam=0; cam<Ncam; ++cam)
	    {
		trlet_list(nn).scores(cam, tt) = 0;
		column(trlet_list(nn).fscores(cam), tt) =
		    scalar_vector<Float>(trlet_list(nn).fscores(cam).size1());
	    }
	}

    }


}

#ifdef USE_MPI
template <typename Float>
void track_existed_objects(mpi::communicator& world,
			   parameter_t const& P,  directory_structure_t const& ds,
			   geometric_info_t const& gi, object_info_t & oi,
			   vector<std::vector<std::string> > const &seq,
			   int tt,
			   vector<CImg<unsigned char> > const& images,
			   vector<matrix<float> > const& grd )
{
    int Ncam = seq.size();
    vector<object_trj_t> & trlet_list=oi.trlet_list;
    int nobj = trlet_list.size();
    int T = seq[0].size();
    int np = oi.model.size();

    int num_scales = P.scales.size();

    for(int nn=0; nn<oi.curr_num_obj; ++nn)
    {
	if(trlet_list(nn).trj.size()==0)
	    continue;
	if(trlet_list(nn).state>=2)
	   continue;

	vector<candidate_array<Float> > cand_array(Ncam);

	for(int cam=0; cam<Ncam; ++cam)
	{
	    pmodel_t const& pmodel = oi.pmodel_list(cam, nn);
	    smodel_t const& smodel = oi.smodel_list(cam, nn);
	    track_one(world, P, ds, gi, oi, seq, tt, cam, nn, images,
		      cand_array(cam), grd);
	}

	int best_y, best_x, best_s;
	if(0==world.rank())
	{
	    ground_scoremap_t<Float> grd_scoremap;
	    combine_ground_score(cand_array, grd_scoremap, gi);
	    grd_scoremap.peak(best_y, best_x, best_s);	
	}
	mpi::broadcast(world, best_y, 0);
	mpi::broadcast(world, best_x, 0);

	trlet_list(nn).trj_3d(tt, 0) = best_x;
	trlet_list(nn).trj_3d(tt, 1) = best_y;
	for(int cam=0; cam<Ncam; ++cam)
	{
	    vector<Float> trj_row(4);
	    if(0==world.rank())
	    {
		vector<double> bx(1), by(1), ix, iy;
		bx <<= best_x; by <<= best_y;
		apply_homography(gi.grd2img(cam), bx, by, ix, iy);
		float hpre = oi.pmodel_list(cam, nn).hpre;
		float cur_fy = iy(0);
		float cur_fx = ix(0);
		float cur_hy = gi.horiz_mean+hpre*(cur_fy-gi.horiz_mean);
		float ds = P.scales(best_s)*(cur_fy-cur_hy)/oi.pmodel_list(cam, nn).bh;
		float ww = ds*oi.pmodel_list(cam, nn).bw;
		float hh = cur_fy - cur_hy;

		trj_row <<= (cur_fx-ww/2), cur_hy, (cur_fx+ww/2), cur_fy;
	    }
	    mpi::broadcast(world, trj_row, 0);
	    row(trlet_list(nn).trj(cam), tt) = trj_row;

	}//endfor cam


	vector<Float> scores(column(trlet_list(nn).scores, tt));
	trlet_list(nn).state = determine_trlet_state(scores, P);

	if(trlet_list(nn).state <= 1)
	{
	    trlet_list(nn).endt = tt;
	    for(int cam=0; cam <Ncam; ++cam)
	    {
		update_one<Float>(world, P, ds, gi, oi, seq, tt, cam, nn, images, grd );
	    }

	}
	if(trlet_list(nn).state == 2)
	{
	    for(int cam=0; cam<Ncam; ++cam)
	    {
		trlet_list(nn).scores(cam, tt) = 0;
		column(trlet_list(nn).fscores(cam), tt) =
		    scalar_vector<Float>(trlet_list(nn).fscores(cam).size1());
	    }
	}

    }

}

#endif


#endif
