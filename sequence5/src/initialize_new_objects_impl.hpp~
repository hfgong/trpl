#ifndef __INITIALIZE_NEW_OBJECTS_IMPL_HPP_INCLUDED__
#define __INITIALIZE_NEW_OBJECTS_IMPL_HPP_INCLUDED__

template <typename Float>
void initialize_new_objects(parameter_t const& P,  directory_structure_t const& ds,
			    geometric_info_t const& gi, object_info_t& oi,
			    vector<std::vector<std::string> > const &seq, int tt,
			    vector<CImg<unsigned char> > const& images,
			    vector<matrix<float> > const& grd,
			    vector<matrix<float> >& detected_rects)
{
    int Ncam = seq.size();
    vector<object_trj_t> & trlet_list=oi.trlet_list;
    int nobj = trlet_list.size();
    int num_new_obj = detected_rects(0).size1();
    int T = seq[0].size();
    int np = oi.model.size();

    int num_scales = P.scales.size();

    for(int oo=0; oo<num_new_obj; ++oo)
    {
	int nn = oi.curr_num_obj + oo;


	trlet_list(nn).startt = tt;
	trlet_list(nn).endt = tt;
	trlet_list(nn).state = 1;
	trlet_list(nn).trj = vector<matrix<float> >(Ncam);
	for(int cam=0; cam<Ncam; ++cam)
	{
	    trlet_list(nn).trj(cam) = scalar_matrix<float>(T, 4, 0);
	}
	trlet_list(nn).trj_3d = scalar_matrix<float>(T, 2, 0);

	trlet_list(nn).hist_p = vector<matrix<float> >(Ncam);
	trlet_list(nn).hist_q = vector<matrix<float> >(Ncam);

	trlet_list(nn).fscores = vector<matrix<float> >(Ncam);
	trlet_list(nn).scores = scalar_matrix<float>(Ncam, T, 0);

	vector<candidate_array<Float> > cand_array(Ncam);

	for(int cam=0; cam<Ncam; ++cam)
	{

	    trlet_list(nn).fscores(cam) = scalar_matrix<float>(np*2, T, 0);

	    float w = detected_rects(cam)(oo, 2)-detected_rects(cam)(oo, 0);
	    float h = detected_rects(cam)(oo, 3)-detected_rects(cam)(oo, 1);
	    row(trlet_list(nn).trj(cam), tt) = row(detected_rects(cam), oo);

	    matrix<float> rects;
	    compute_part_rects(detected_rects(cam)(oo, 0), detected_rects(cam)(oo, 1),
			  w, h, oi.model, rects);

	    pmodel_t pmodel;

	    vector<float> br(row(detected_rects(cam), oo));
	    rects_to_pmodel_geom(br, gi.horiz_mean, pmodel);
	    oi.pmodel_list(cam, nn) = pmodel;

	    //collect_sift(grd(cam), );
	    matrix<float> hist_p, hist_q;
	    collect_hist(images(cam), rects, hist_p, hist_q);
	    trlet_list(nn).hist_p(cam) = hist_p;
	    trlet_list(nn).hist_q(cam) = hist_q;


	    std::vector<float> sxr, syr;
	    for(float v=-P.xrange/2; v<=P.xrange/2; v+=P.xstep)
	    {
		sxr.push_back(v);
	    }
	    for(float v=-P.yrange/2; v<=P.yrange/2; v+=P.ystep)
	    {
		syr.push_back(v);
	    }
	    vector<float> xr(sxr.size()), yr(syr.size());
	    std::copy(sxr.begin(), sxr.end(), xr.begin());
	    std::copy(syr.begin(), syr.end(), yr.begin());

	    float feetx = (trlet_list(nn).trj(cam)(tt, 0)
			   +trlet_list(nn).trj(cam)(tt, 2))/2;
	    float feety = trlet_list(nn).trj(cam)(tt, 3);

	    matrix<Float> cand_rects;
	    vector<Float> cand_scale;
	    matrix<int> cand_ijs;

	    enumerate_rects_inpoly(images(cam), oi.pmodel_list(cam, nn),
				   feetx, feety,
				   xr, yr, P.scales, gi.horiz_mean, gi.horiz_sig,
				   gi.polys_im(cam),
				   cand_rects, cand_scale,
				   cand_ijs, cand_array(cam));


	    vector<Float> cand_hist_score;
	    matrix<Float> hist_fscores;

	    real_timer_t timer;
	    get_cand_hist_score(images(cam), oi.model, P.logp1, P.logp2,
				trlet_list(nn).hist_p(cam),
				trlet_list(nn).hist_q(cam),
				cand_rects,
				cand_hist_score, hist_fscores);

	    vector<Float> cand_score=cand_hist_score;
	    std::cout<<"\t\t"<<cand_rects.size1()<<" rects, \tget_cand_hist_score time:"
		     <<timer.elapsed()/1000.0f<<"s."<<std::endl;

	    int idx_max = std::max_element(cand_score.begin(), cand_score.end())
		- cand_score.begin();

	    column(trlet_list(nn).fscores(cam), tt) = row(hist_fscores, idx_max);
	    trlet_list(nn).scores(cam, tt) = cand_score(idx_max);
	    cand_array(cam).fill_score(cand_score, cand_ijs);


	}//end for cam

	real_timer_t timer;

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

	    //std::cout<<"trlet_list(nn).trj(cam)(tt, :)="
	    //     <<row(trlet_list(nn).trj(cam), tt)<<std::endl;

	}//endfor cam

    }//endfor oo

    oi.curr_num_obj += num_new_obj;
}

#ifdef USE_MPI
template <typename Float>
void initialize_new_objects(mpi::communicator& world,
			    parameter_t const& P,  directory_structure_t const& ds,
			    geometric_info_t const& gi, object_info_t& oi,
			    vector<std::vector<std::string> > const &seq, int tt,
			    vector<CImg<unsigned char> > const& images,
			    vector<matrix<float> > const& grd,
			    vector<matrix<float> >& detected_rects)
{
    int Ncam = seq.size();
    vector<object_trj_t> & trlet_list=oi.trlet_list;
    int nobj = trlet_list.size();
    int num_new_obj = detected_rects(0).size1();
    int T = seq[0].size();
    int np = oi.model.size();

    int num_scales = P.scales.size();

    for(int oo=0; oo<num_new_obj; ++oo)
    {
	int nn = oi.curr_num_obj + oo;

	trlet_list(nn).startt = tt;
	trlet_list(nn).endt = tt;
	trlet_list(nn).state = 1;
	trlet_list(nn).trj = vector<matrix<float> >(Ncam);
	for(int cam=0; cam<Ncam; ++cam)
	{
	    trlet_list(nn).trj(cam) = scalar_matrix<float>(T, 4, 0);
	}
	trlet_list(nn).trj_3d = scalar_matrix<float>(T, 2, 0);

	trlet_list(nn).hist_p = vector<matrix<float> >(Ncam);
	trlet_list(nn).hist_q = vector<matrix<float> >(Ncam);

	trlet_list(nn).fscores = vector<matrix<float> >(Ncam);
	trlet_list(nn).scores = scalar_matrix<float>(Ncam, T, 0);

	vector<candidate_array<Float> > cand_array(Ncam);
	for(int cam=0; cam<Ncam; ++cam)
	{

	    trlet_list(nn).fscores(cam) = scalar_matrix<float>(np*2, T, 0);

	    float w = detected_rects(cam)(oo, 2)-detected_rects(cam)(oo, 0);
	    float h = detected_rects(cam)(oo, 3)-detected_rects(cam)(oo, 1);
	    row(trlet_list(nn).trj(cam), tt) = row(detected_rects(cam), oo);

	    matrix<float> rects;
	    compute_part_rects(detected_rects(cam)(oo, 0), detected_rects(cam)(oo, 1),
			  w, h, oi.model, rects);

	    pmodel_t pmodel;

	    vector<float> br(row(detected_rects(cam), oo));
	    rects_to_pmodel_geom(br, gi.horiz_mean, pmodel);
	    oi.pmodel_list(cam, nn) = pmodel;

	    //collect_sift(grd(cam), );
	    matrix<float> hist_p, hist_q;
	    collect_hist(images(cam), rects, hist_p, hist_q);
	    trlet_list(nn).hist_p(cam) = hist_p;
	    trlet_list(nn).hist_q(cam) = hist_q;

	    matrix<Float> cand_rects;
	    vector<Float> cand_scale;
	    matrix<int> cand_ijs;

	    if(0==world.rank())
	    {

		std::vector<float> sxr, syr;
		for(float v=-P.xrange/2; v<=P.xrange/2; v+=P.xstep)
		{
		    sxr.push_back(v);
		}
		for(float v=-P.yrange/2; v<=P.yrange/2; v+=P.ystep)
		{
		    syr.push_back(v);
		}
		vector<float> xr(sxr.size()), yr(syr.size());
		std::copy(sxr.begin(), sxr.end(), xr.begin());
		std::copy(syr.begin(), syr.end(), yr.begin());

		float feetx = (trlet_list(nn).trj(cam)(tt, 0)
			       +trlet_list(nn).trj(cam)(tt, 2))/2;
		float feety = trlet_list(nn).trj(cam)(tt, 3);


		enumerate_rects_inpoly(images(cam), oi.pmodel_list(cam, nn),
				       feetx, feety,
				       xr, yr, P.scales, gi.horiz_mean, gi.horiz_sig,
				       gi.polys_im(cam),
				       cand_rects, cand_scale,
				       cand_ijs, cand_array(cam));

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

	    mpi::broadcast(world, cand_hist_score, 0);
	    mpi::broadcast(world, hist_fscores, 0);


	    vector<Float> cand_score=cand_hist_score;
	    if(0==world.rank())
	    std::cout<<"\t\t"<<cand_rects.size1()<<" rects, \tget_cand_hist_score time:"
		     <<timer.elapsed()/1000.0f<<"s."<<std::endl;

	    if(0==world.rank())
	    {
		int idx_max = std::max_element(cand_score.begin(), cand_score.end())
		    - cand_score.begin();

		column(trlet_list(nn).fscores(cam), tt) = row(hist_fscores, idx_max);
		trlet_list(nn).scores(cam, tt) = cand_score(idx_max);
		cand_array(cam).fill_score(cand_score, cand_ijs);
	    }
	    mpi::broadcast(world, cand_array(cam), 0);
	    mpi::broadcast(world, trlet_list(nn).scores(cam, tt), 0);
	    vector<Float> fscore_col;
	    if(0==world.rank())
	    {
		fscore_col = column(trlet_list(nn).fscores(cam), tt);
	    }
	    mpi::broadcast(world, fscore_col, 0);
	    if(0!=world.rank())
	    {
		column(trlet_list(nn).fscores(cam), tt) = fscore_col;
	    }


	}//end for cam

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

    }//endfor oo

    oi.curr_num_obj += num_new_obj;
}

#endif

#endif
