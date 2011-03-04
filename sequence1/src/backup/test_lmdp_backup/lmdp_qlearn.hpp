#ifndef __LMDP__QLEARN__HPP__INCLUDED__
#define __LMDP__QLEARN__HPP__INCLUDED__

struct training_data_t
{
    matrix<int> obs;
    matrix<int> dyn_obs;
    matrix<int> path;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)	{
	ar & BOOST_SERIALIZATION_NVP(obs);
	ar & BOOST_SERIALIZATION_NVP(dyn_obs);
	ar & BOOST_SERIALIZATION_NVP(path);
    }

    void save(std::string const& fname)	const {
	save_xml(*this, fname, "training_data");
    }
    void load(std::string const& fname) {
	load_xml(*this, fname, "training_data");
    }
};

void generate_feature_maps(matrix<int> const& obs, matrix<int> const& dyn_obs,
			   vector<matrix<float> >& feat)
{

    using namespace boost::lambda;
    feat = vector<matrix<float> >(3);
    feat(0) = scalar_matrix<float>(obs.size1(), obs.size2(), 1.0f);
    feat(1) = matrix<float>(obs.size1(), obs.size2());
    feat(2) = matrix<float>(obs.size1(), obs.size2());

    CImg<int> obs_im;
    array2d_copy(obs, obs_im);
    CImg<float> obs_dist = obs_im.get_distance(1);
    
    CImg<int> dyn_obs_im;
    array2d_copy(dyn_obs, dyn_obs_im);
    CImg<float> dyn_obs_dist = dyn_obs_im.get_distance(1);

    //array2d_print(std::cout, obs_dist);
    //array2d_print(std::cout, dyn_obs_dist);

    float(*fexp)(float) = std::exp;
    array2d_transform(obs_dist, feat(1), bind(fexp, -_1/4.0f));
    array2d_transform(dyn_obs_dist, feat(2), bind(fexp, -_1/4.0f));

}


void get_state_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
		     vector<vector<int> >& sg,
		     matrix<int>& yx2ig,
		     matrix<int>& ig2yx)
{
    using namespace boost::lambda;
    matrix<int> obs_all(obs+dyn_obs);
    int ng = std::count(obs_all.data().begin(), obs_all.data().end(), 0);
    sg = vector<vector<int> >(ng);

    yx2ig = scalar_matrix<int>(obs.size1(), obs.size2(), -1);
    ig2yx = scalar_matrix<int>(ng, 2, -1);

    int ig = 0;
    for(int yy=0; yy<obs.size1(); ++yy)
    {
	for(int xx=0; xx<obs.size2(); ++xx)
	{
	    if(obs_all(yy, xx)>0) continue;
	    yx2ig(yy, xx) = ig;
	    ig2yx(ig, 0) = yy;
	    ig2yx(ig, 1) = xx;
	    ++ig;
	}
    }

    vector<std::vector<int> > sgv(ng);
    for(int yy=0; yy<yx2ig.size1(); ++yy)
    {
	for(int xx=0; xx<yx2ig.size2(); ++xx)
	{
	    int ig1 = yx2ig(yy, xx);
	    if(ig1<0) continue;
	    for(int nn=0; nn<8; ++nn)
	    {
		int y2 = yy+nbrhood_t::dy[nn];
		int x2 = xx+nbrhood_t::dx[nn];
		if(y2<0 || y2>=obs.size1()) continue;
		if(x2<0 || x2>=obs.size2()) continue;
		int ig2 = yx2ig(y2, x2);
		if(ig2<0) continue;
		sgv(ig1).push_back(ig2);
	    }
	}
    }

    for(int ii=0; ii<sgv.size(); ++ii)
    {
	array1d_copy(sgv[ii], sg(ii));
    }

}

void get_feature_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
		       vector<matrix<float> > const& feat,
		       vector<vector<int> > const& sg,
		       matrix<int> const& ig2yx,
		       matrix<vector<double> >& fg)
{
    using namespace boost::lambda;

    int ng = sg.size();

    fg = matrix<vector<double> >(feat.size(), ng);
    for(int ff=0; ff<feat.size(); ++ff)
    {
	for(int gg=0; gg<ng; ++gg)
	{
	    fg(ff, gg) = vector<double>(sg(gg).size());
	}
    }

    for(int ff=0; ff<feat.size(); ++ff)
    {
	for(int gg=0; gg<ng; ++gg)
	{

	    int yy = ig2yx(gg, 0);
	    int xx = ig2yx(gg, 1);

	    for(int nn=0; nn<fg(ff, gg).size(); ++nn)
	    {
		int g2 = sg(gg)(nn);
		int y2 = ig2yx(g2, 0);
		int x2 = ig2yx(g2, 1);
		int dx = x2-xx;
		int dy = y2-yy;

		double dist = std::sqrt(static_cast<double>(dx*dx+dy*dy));
		fg(ff, gg)(nn) = (feat(ff)(yy, xx)+feat(ff)(y2, x2))/2*dist;
	    }

	}
    }

}


void get_path_ig(matrix<int> const& path,
		 matrix<int> const& yx2ig,
		 vector<int>& path_ig)
{
    path_ig = vector<int>(path.size2());
    for(int ii=0; ii<path.size2(); ++ii)
    {
	path_ig(ii) = yx2ig(path(0, ii), path(1, ii));
    }
}


struct grad_state_t
{
    vector<double> grad_q;       //num_feat
    matrix<double> grad_log_pps; //num_feat, num_action

    vector<matrix<double> > grad_log_p_tilde; //num_feat, num_neighbor, num_action
    matrix<double> grad_l_tilde, grad_nent; //num_feat, num_action
};

struct hess_state_t
{
    matrix<double> hess_q;       //num_feat, num_feat
    matrix<vector<double> > hess_log_pps; //num_feat, num_feat, num_action
    matrix<matrix<double> > hess_log_p_tilde; //num_feat, num_feat, num_neighbor, num_action
    matrix<vector<double> > hess_l_tilde, hess_nent; //num_feat, num_feat, num_action
};

void compute_grad_plh(lmdp_t const& lmdp, int ig, grad_state_t& grad_state)
{
    using namespace boost::lambda;

    matrix<vector<double> >const& fg = lmdp.fg;
    vector<double> const& wei = lmdp.wei;
    vector<vector<int> > const& sg = lmdp.sg;
    vector<matrix<double> > const& p_tilde = lmdp.p_tilde;
    vector<matrix<double> > const& log_p_tilde = lmdp.log_p_tilde;
    matrix<int> const& ig2yx = lmdp.ig2yx;

    vector<matrix<double> >& grad_log_p_tilde = grad_state.grad_log_p_tilde;
    matrix<double> & grad_l_tilde = grad_state.grad_l_tilde;
    matrix<double> & grad_nent = grad_state.grad_nent;

    int nf = fg.size1()+1; //+1 for lambda
    int nnb = sg(ig).size();

    grad_log_p_tilde = vector<matrix<double> >(nf);
    for(int ff=0; ff<nf; ++ff)
    {
	grad_log_p_tilde(ff) = scalar_matrix<double>(nnb, nnb, 0.0f);
    }
    for(int aa=0; aa<grad_log_p_tilde(nf-1).size1(); ++aa)
    {
	int iga = sg(ig)(aa);
	int ya = ig2yx(iga, 0);
	int xa = ig2yx(iga, 1);
	double denom = 0;
	for(int jj=0; jj<grad_log_p_tilde(nf-1).size2(); ++jj)
	{
	    int igj = sg(ig)(jj);
	    int yj = ig2yx(igj, 0);
	    int xj = ig2yx(igj, 1);
	    double ds = (xa-xj)*(xa-xj)+(ya-yj)*(ya-yj);
	    grad_log_p_tilde(nf-1)(aa, jj) = -ds;
	    denom += p_tilde(ig)(aa, jj)*ds;
	}
	for(int jj=0; jj<grad_log_p_tilde(nf-1).size2(); ++jj)
	{
	    grad_log_p_tilde(nf-1)(aa, jj) += denom;
	}
    }


    grad_l_tilde = scalar_matrix<double>(nf, nnb, 0.0f);
    for(int ff=0; ff+1<nf; ++ff)
    {
	for(int aa=0; aa<grad_l_tilde.size2(); ++aa)
	{
	    grad_l_tilde(ff, aa) = inner_prod(row(p_tilde(ig), aa), fg(ff, ig));
	}
    }

    vector<double> di=scalar_vector<double>(nnb, 0.0f);
    for(int ff=0; ff+1<nf; ++ff)
    {
	for(int nn=0; nn<nnb; ++nn)
	{
	    di(nn) += wei(ff)*fg(ff, ig)(nn);
	}
    }
    //std::cout<<di<<std::endl;
    grad_nent = scalar_matrix<double>(nf, nnb, 0.0f);
    for(int aa=0; aa<grad_l_tilde.size2(); ++aa)
    {
	vector<double> pgrad_logp = element_prod(row(p_tilde(ig), aa), row(grad_log_p_tilde(nf-1), aa));
	grad_l_tilde(nf-1, aa) = inner_prod(pgrad_logp, di);
	vector<double> logp_plus1= row(log_p_tilde(ig), aa);
	std::for_each(logp_plus1.begin(), logp_plus1.end(), _1+=1);
	grad_nent(nf-1, aa) = inner_prod(pgrad_logp, logp_plus1);
    }


}

void compute_grad_q_pps(lmdp_t const& lmdp, int ig, grad_state_t& grad_state)
{
    using namespace boost::lambda;
    matrix<double> const& p_tilde = lmdp.p_tilde(ig);
    vector<double> const& log_pps = lmdp.log_pps(ig);
			    
    vector<matrix<double> > const& grad_log_p_tilde = grad_state.grad_log_p_tilde;
    matrix<double> const& grad_l_tilde = grad_state.grad_l_tilde;
    matrix<double> const& grad_nent = grad_state.grad_nent;
    vector<double> & grad_q = grad_state.grad_q;
    matrix<double> & grad_log_pps = grad_state.grad_log_pps;

    int nf = grad_log_p_tilde.size();
    int nnb = grad_l_tilde.size2();
    grad_log_pps = matrix<double>(nf, nnb);
    grad_q = vector<double>(nf);

    matrix<double> B = p_tilde;
    permutation_matrix<std::size_t> P(nnb);
    int res = lu_factorize(B, P);  //B holds L and U

    vector<double> pps(log_pps.size());
    double (*fexp)(double) = std::exp;
    std::transform(log_pps.begin(), log_pps.end(), pps.begin(), fexp);
    for(int ff=0; ff<nf; ++ff)
    {
	matrix<double> plogp= element_prod(p_tilde, grad_log_p_tilde(ff));
	vector<double> nvec = prod(plogp, log_pps);
	nvec += row(grad_l_tilde, ff)-row(grad_nent, ff);

	lu_substitute(B, P, nvec);       //nvec holds the solution B_{-1}*nvec
	vector<double> m_bar = -nvec;

	grad_q(ff) = inner_prod(pps, nvec);

	std::for_each(m_bar.begin(), m_bar.end(), _1 += grad_q(ff));
	row(grad_log_pps, ff) = m_bar;
    }

}

void compute_hess_plh(lmdp_t const& lmdp, int ig, grad_state_t const& grad_state,
		      hess_state_t & hess_state)
{
    using namespace boost::lambda;
    matrix<vector<double> >const& fg = lmdp.fg;
    vector<double> const& wei = lmdp.wei;
    vector<vector<int> > const& sg = lmdp.sg;
    vector<matrix<double> > const& p_tilde = lmdp.p_tilde;
    vector<matrix<double> > const& log_p_tilde = lmdp.log_p_tilde;
    matrix<int> const& ig2yx = lmdp.ig2yx;

    vector<matrix<double> > const& grad_log_p_tilde = grad_state.grad_log_p_tilde;
    matrix<double> const& grad_l_tilde = grad_state.grad_l_tilde;
    matrix<double> const& grad_nent = grad_nent;

    matrix<matrix<double> > & hess_log_p_tilde = hess_state.hess_log_p_tilde;
    matrix<vector<double> > & hess_l_tilde = hess_state.hess_l_tilde;
    matrix<vector<double> > & hess_nent = hess_state.hess_nent;

    int nf = fg.size1()+1; //+1 for lambda
    int nnb = sg(ig).size();

    hess_log_p_tilde = matrix<matrix<double> >(nf, nf);
    hess_l_tilde = matrix<vector<double> >(nf, nf);
    hess_nent = matrix<vector<double> >(nf, nf);
    for(int f1=0; f1<nf; ++f1)
    {
	for(int f2=0; f2<nf; ++f2)
	{
	    hess_log_p_tilde(f1, f2) = scalar_matrix<double>(nnb, nnb, 0.0f);
	    hess_l_tilde(f1, f2) = scalar_vector<double>(nnb, 0.0f);
	    hess_nent(f1, f2) = scalar_vector<double>(nnb, 0.0f);
	}
    }

    matrix<double> pgrad_logp = element_prod(p_tilde(ig), grad_log_p_tilde(nf-1));

    for(int ff=0; ff+1<nf; ++ff)
    {
	hess_l_tilde(ff, nf-1) = prod(pgrad_logp, fg(ff, ig));
	hess_l_tilde(nf-1, ff) = prod(pgrad_logp, fg(ff, ig));
    }

    matrix<double> dist_mat(nnb, nnb);
    for(int aa=0; aa<nnb; ++aa)
    {
	int ya = ig2yx(sg(ig)(aa), 0);
	int xa = ig2yx(sg(ig)(aa), 1);
	for(int jj=0; jj<nnb; ++jj)
	{
	    int yj = ig2yx(sg(ig)(jj), 0);
	    int xj = ig2yx(sg(ig)(jj), 1);
	    dist_mat(aa, jj) = (ya-yj)*(ya-yj) + (xa-xj)*(xa-xj);
	}
    }

    vector<double> hlptilde_one(nnb);
    for(int aa=0; aa<nnb; ++aa)
    {
	hlptilde_one(aa) = inner_prod(row(pgrad_logp, aa), row(dist_mat, aa));
    }
    for(int jj=0; jj<nnb; ++jj)
    {
	column(hess_log_p_tilde(nf-1, nf-1), jj) = hlptilde_one;
    }


    matrix<double> pgrad_logp_sq = element_prod(pgrad_logp, grad_log_p_tilde(nf-1));
    matrix<double> phess_logp = element_prod(p_tilde(ig), hess_log_p_tilde(nf-1, nf-1));

    vector<double> di=scalar_vector<double>(nnb, 0.0f);
    for(int ff=0; ff+1<nf; ++ff)
    {
	for(int nn=0; nn<nnb; ++nn)
	{
	    di(nn) += wei(ff)*fg(ff, ig)(nn);
	}
    }

    hess_l_tilde(nf-1, nf-1) = prod(phess_logp+pgrad_logp_sq, di);

    matrix<double> log_p_tilde_plus1;
    array2d_transform(log_p_tilde(ig), log_p_tilde_plus1, _1+1);
    //std::for_each(log_p_tilde_plus1.begin(), log_p_tilde_plus1.end(), _1+=1);

    vector<double> rs;
    row_sum(element_prod(phess_logp+pgrad_logp_sq, log_p_tilde_plus1)+pgrad_logp_sq, rs);
    hess_nent(nf-1, nf-1) =  rs;

    //std::cout<<"hess_log_p_tilde-trans="<<hess_log_p_tilde-trans(hess_log_p_tilde)<<std::endl;
    //std::cout<<"hess_l_tilde-trans="<<hess_l_tilde-trans(hess_l_tilde)<<std::endl;
    //std::cout<<"hess_nent-trans="<<hess_nent-trans(hess_nent)<<std::endl;
}

void compute_hess_q_pps(lmdp_t const& lmdp, int ig, grad_state_t const& grad_state,
    hess_state_t& hess_state)
{
    using namespace boost::lambda;

    matrix<double> const& p_tilde = lmdp.p_tilde(ig);
    vector<double> const& log_pps = lmdp.log_pps(ig);
    vector<matrix<double> > const& grad_log_p_tilde = grad_state.grad_log_p_tilde;
    matrix<double> const& grad_l_tilde = grad_state.grad_l_tilde;
    matrix<double> const& grad_nent = grad_state.grad_nent;
    vector<double> const& grad_q = grad_state.grad_q;
    matrix<double> const& grad_log_pps = grad_state.grad_log_pps;

    matrix<matrix<double> > const& hess_log_p_tilde = hess_state.hess_log_p_tilde;
    matrix<vector<double> > const& hess_l_tilde = hess_state.hess_l_tilde;
    matrix<vector<double> > const& hess_nent = hess_state.hess_nent;
    matrix<double>& hess_q = hess_state.hess_q;
    matrix<vector<double> >& hess_log_pps = hess_state.hess_log_pps;

    int nf = grad_log_p_tilde.size();
    int nnb = grad_l_tilde.size2();
    hess_log_pps = matrix<vector<double> >(nf, nf);
    hess_q = matrix<double>(nf, nf);

    matrix<double> B = p_tilde;
    permutation_matrix<std::size_t> P(nnb);
    int res = lu_factorize(B, P);  //B holds L and U

    vector<double> pps(log_pps.size());
    double (*fexp)(double) = std::exp;
    std::transform(log_pps.begin(), log_pps.end(), pps.begin(), fexp);
    for(int ee=0; ee<nf; ++ee)
    {
	for(int ff=0; ff<nf; ++ff)
	{	
	    //matrix<double> pglogp= element_prod(p_tilde, grad_log_p_tilde(ff));
	    matrix<double> gradl_lp_times_gradk_lp = element_prod(grad_log_p_tilde(ee), grad_log_p_tilde(ff));
	    matrix<double> hess_pt = element_prod(p_tilde, gradl_lp_times_gradk_lp+hess_log_p_tilde(ee, ff));
	    vector<double> nvec = prod(hess_pt, log_pps);
	    //row_sum(hess_pt, nvec);
	    matrix<double> gradl_p = element_prod(p_tilde, grad_log_p_tilde(ee));
	    matrix<double> gradk_p = element_prod(p_tilde, grad_log_p_tilde(ff));
	    nvec += prod(gradl_p, row(grad_log_pps, ff) )
		+ prod(gradk_p, row(grad_log_pps, ee) )
		+ hess_l_tilde(ee, ff)-hess_nent(ee, ff);   
	    std::cout<<"nvec="<<nvec<<std::endl;

	    lu_substitute(B, P, nvec);       //nvec holds the solution B_{-1}*nvec

	    double sx = -inner_prod(pps, element_prod(row(grad_log_pps, ee), row(grad_log_pps, ff)));

	    hess_q(ee, ff) = inner_prod(pps, nvec) + sx;

	    vector<double> m_bar = -nvec;
	    std::for_each(m_bar.begin(), m_bar.end(), _1 += hess_q(ee, ff));
	    hess_log_pps(ee, ff) = m_bar;
	}
    }

    //std::cout<<"hess_q-hess_qT="<<hess_q-trans(hess_q)<<std::endl;
    //std::cout<<"hess_log_pps-hess_log_ppsT="<<hess_log_pps-trans(hess_log_pps)<<std::endl;
    //std::cout<<"hess_q="<<hess_q<<std::endl;
    //std::cout<<"hess_log_pps="<<hess_log_pps<<std::endl;

}

void compute_grad_lmdp(lmdp_t const& lmdp,
		       vector<grad_state_t>& grad_lmdp)
{
    grad_lmdp = vector<grad_state_t>(lmdp.sg.size());
    for(int ig=0; ig < lmdp.sg.size(); ++ig)
    {
	grad_state_t& grad_state = grad_lmdp(ig);
	compute_grad_plh(lmdp, ig, grad_state);
	compute_grad_q_pps(lmdp, ig, grad_state);
    }

}

void compute_hess_lmdp(lmdp_t const& lmdp,
		       vector<grad_state_t> const& grad_lmdp,
		       vector<hess_state_t>& hess_lmdp)
{
    hess_lmdp = vector<hess_state_t>(lmdp.sg.size());
    for(int ig=0; ig < lmdp.sg.size(); ++ig)
    {
	grad_state_t const& grad_state = grad_lmdp(ig);
	hess_state_t& hess_state = hess_lmdp(ig);
	compute_hess_plh(lmdp, ig, grad_state, hess_state);
	compute_hess_q_pps(lmdp, ig, grad_state, hess_state);
    }
}


void compute_grad_hess_logz(lmdp_t const& lmdp,
			    vector<grad_state_t> const& grad_lmdp,
			    vector<hess_state_t> const& hess_lmdp,
			    vector<vector<int> >const& path_ig,
			    vector<vector<double> >const & logz,
			    vector<matrix<double> >& grad_logz,
			    matrix<matrix<double> >& hess_logz)
{
    using namespace boost::lambda;

    //matrix<vector<double> >const& fg = lmdp.fg;
    vector<double> const& wei = lmdp.wei;
    vector<vector<int> > const& sg = lmdp.sg;
    vector<vector<double> > const& log_pps = lmdp.log_pps;
    vector<double> const& q = lmdp.q;

    int nf = lmdp.fg.size1()+1; //+1 for lambda position
    int ng = sg.size();

    int nnz = lmdp.count_sg_nzz();

    //compute gradients and hessians of logz
    grad_logz =  //cc, ng, nf
	scalar_vector<matrix<double> >(logz.size(),
				       scalar_matrix<double>(ng, nf, 0.0f) );
    hess_logz =  //cc, ng, ee, ff
	scalar_matrix<matrix<double> >(logz.size(), ng,
				       scalar_matrix<double>(nf, nf, 0.0f) );

    for(int cc=0; cc<logz.size(); ++cc)
    {

	vector<int> good=scalar_vector<int>(ng, 0);
	good(*(path_ig(cc).rbegin())) = 1;

	umf_sparse_matrix A(ng, ng, ng+nnz);
	matrix<double> nvec=scalar_matrix<double>(nf, ng, 0.0f);

	matrix<vector<double> > nvec2 =
	    scalar_matrix<vector<double> >(nf, nf,
					   scalar_vector<double>(ng, 0.0f));//(nf, nf, ng);


	vector<vector<double> > pexp_list(ng);
	for(int ig=0; ig<ng; ++ig)
	{
	    A(ig, ig) = 1;
	    double vi = -logz(cc)(ig);
	    double qi = q(ig);
	    vector<double> pexp(sg(ig).size());

	    vector<double> vj(sg(ig).size());
	    for(int nn=0; nn<sg(ig).size(); ++nn)
	    {
		vj(nn) = -logz(cc)(sg(ig)(nn));
	    }
	    for(int nn=0; nn<sg(ig).size(); ++nn)
	    {
		double lpexp = log_pps(ig)(nn)+vi-qi-vj(nn);
		pexp(nn) = std::exp(lpexp);
		A(ig, sg(ig)(nn)) = -pexp(nn);
	    }
	    //if(ig==1) std::cout<<":Pexp.size="<<pexp.size()<<std::endl;
	    pexp_list(ig) = pexp;
	}

	for(int ff=0; ff<nf; ++ff)
	{
	    for(int ig=0; ig<ng; ++ig)
	    {
		nvec(ff, ig) = inner_prod(pexp_list(ig),
					  row(grad_lmdp(ig).grad_log_pps, ff)) - grad_lmdp(ig).grad_q(ff);
	    }
	}

	splitter_t<> splitter(good);
	matrix<umf_sparse_matrix> Ablock = split_sparse_mat<matrix>(splitter, A);
	//split_sparse_matrix(A, good, Ablock);


	umf::symbolic_type<double> symb;
	umf::numeric_type<double> nume;

	umf::symbolic (Ablock(0,0), symb); 
	umf::numeric (Ablock(0,0), symb, nume); 

	for(int ff=0; ff<nf; ++ff)
	{
	    vector<double> grad_logz_one=scalar_vector<double>(ng, 0.0l);
	    vector<double> nvec_one(row(nvec, ff));
	    vector<vector<double> > nvec_block =  splitter(nvec_one);
	    vector<vector<double> > sol_block =   splitter(grad_logz_one);


	    vector<double> nvec_new = nvec_block(0) - prod(Ablock(0, 1), sol_block(1));

	    umf::solve(Ablock(0,0), sol_block(0), nvec_new, nume);  

	    grad_logz_one = splitter.merge(sol_block);

	    column(grad_logz(cc), ff) = grad_logz_one;

	    //std::cout<<"grad_logz_one="<<project(grad_logz_one, range(340, 350))<<std::endl;
	}

	    
	for(int ee=0; ee<nf; ++ee)
	{
	    for(int ff=0; ff<nf; ++ff)
	    {
		for(int ig=0; ig<ng; ++ig)
		{
		    nvec2(ee, ff)(ig) = - hess_lmdp(ig).hess_q(ee, ff)
			- (grad_lmdp(ig).grad_q(ee)+grad_logz(cc)(ig, ee))
			* (grad_lmdp(ig).grad_q(ff)+grad_logz(cc)(ig, ff));
		    vector<double> tmp(sg(ig).size());
		    for(int nn=0; nn<sg(ig).size(); ++nn)
		    {
			tmp(nn) = hess_lmdp(ig).hess_log_pps(ee, ff)(nn) 
			    + (grad_lmdp(ig).grad_log_pps(ee, nn) + grad_logz(cc)(ig, ee))
			    * (grad_lmdp(ig).grad_log_pps(ff, nn) + grad_logz(cc)(ig, ff));
		    }
		    //std::cout<<"tmp.size="<<tmp.size()<<std::endl;
		    //std::cout<<"pexp.size="<<pexp_list(ig).size()<<std::endl;
		    nvec2(ee, ff)(ig) += inner_prod(tmp, pexp_list(ig));
		}
		vector<double> hess_logz_one=scalar_vector<double>(ng, 0.0l);
		vector<vector<double> > nvec2_block = splitter(nvec2(ee, ff));
		vector<vector<double> > sol_block = splitter(hess_logz_one);

		vector<double> nvec2_new = nvec2_block(0) - prod(Ablock(0, 1), sol_block(1));

		umf::solve(Ablock(0,0), sol_block(0), nvec2_new, nume);  

		hess_logz_one = splitter.merge(sol_block);

		//row(hess_logz(cc), ff) = hess_logz_one;
		for(int ig=0; ig<ng; ++ig)
		{
		    hess_logz(cc, ig)(ee, ff) = hess_logz_one(ig);
		}

	    }
	}

	
    }


}


void compute_grad_hess_L(lmdp_t const& lmdp,
			 vector<grad_state_t> const& grad_lmdp,
			 vector<hess_state_t> const& hess_lmdp,
			 vector<vector<int> >const& path_ig,
			 vector<matrix<double> > const& grad_logz,
			 matrix<matrix<double> > const& hess_logz,
			 vector<double>& grad_L,
			 matrix<double>& hess_L)
{
    vector<vector<int> > const& sg = lmdp.sg;

    int nf = lmdp.fg.size1()+1; //+1 for lambda position
    int ng = sg.size();

    double alpha = -1; //-0.0001;// -1;

    //regularization
    grad_L = alpha*lmdp.wei; //grad_L = scalar_vector<double>(nf, 0.0f);
    hess_L = alpha*identity_matrix<double>(nf); //hess_L = scalar_matrix<double>(nf, nf, 0.0f);
    for(int cc=0; cc<path_ig.size(); ++cc)
    {
	grad_L += row(grad_logz(cc), path_ig(cc)(0) );
	hess_L += hess_logz(cc, path_ig(cc)(0));
	std::cout<<"grad_logz(x_0)="<<row(grad_logz(cc), path_ig(cc)(0) )<<std::endl;
	std::cout<<"hess_logz(x_0)="<<hess_logz(cc, path_ig(cc)(0))<<std::endl;
	for(int tt=0; tt+1<path_ig(cc).size(); ++tt)
	{
	    int ig = path_ig(cc)(tt);
	    int ig2 = path_ig(cc)(tt+1);
	    vector<double> const& grad_q = grad_lmdp(ig).grad_q; //num_feat
	    matrix<double> const& grad_log_pps = grad_lmdp(ig).grad_log_pps; //num_feat, num_action
	    matrix<double> const& hess_q = hess_lmdp(ig).hess_q; //num_feat, num_feat
	    matrix<vector<double> > const& hess_log_pps = hess_lmdp(ig).hess_log_pps; //num_feat, num_feat, num_action

	    int nn = std::find(sg(ig).begin(), sg(ig).end(), ig2)-sg(ig).begin();
	    //std::cout<<"nn="<<nn<<std::endl;
	    grad_L += column(grad_log_pps, nn) - grad_q;

	    matrix<double> hess_lpps_one(hess_log_pps.size1(), hess_log_pps.size2());
	    for(int ee=0; ee<hess_lpps_one.size1(); ++ee)
	    {
		for(int ff=0; ff<hess_lpps_one.size2(); ++ff)
		{
		    hess_lpps_one(ee, ff) = hess_log_pps(ee, ff)(nn);
		}
	    }
	    //std::cout<<"hess_lpps_one="<<std::endl;
	    //array2d_print(std::cout, hess_lpps_one);
	    //std::cout<<"hess_q="<<std::endl;
	    //array2d_print(std::cout, hess_q);
	    hess_L += hess_lpps_one - hess_q;
	}
    }

}

void learn_weights(lmdp_t& lmdp,
		   vector<vector<int> > const& path_ig,
		   vector<double>& wei)
{

    for(int it=0; it<1; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	lmdp.embed(wei);

	vector<vector<double> > logz(path_ig.size());

	for(int cc=0; cc<path_ig.size(); ++cc)
	{
	    lmdp.solve(*(path_ig(cc).rbegin()), logz(cc));

	    for(int ff=0; ff<wei.size(); ++ff)
	    {
		lmdp_t lmdp2;
		lmdp2.initialize(lmdp.fg, lmdp.sg, lmdp.yx2ig, lmdp.ig2yx);

		vector<double> wei2 = wei;
		wei2(ff) += 0.1l;

		lmdp2.embed(wei2);

		vector<double> logz2;
		lmdp2.solve(*(path_ig(cc).rbegin()), logz2);
		vector<double> grad_logz2 = (logz2-logz(cc))/0.1l;
		std::cout<<"grad_logz2="<<project(grad_logz2, range(340, 350))<<std::endl;
	    }
	}

	//compute gradient and hessian of q and pps
	vector<grad_state_t> grad_lmdp;
	vector<hess_state_t> hess_lmdp;
	compute_grad_lmdp(lmdp, grad_lmdp);
	compute_hess_lmdp(lmdp, grad_lmdp, hess_lmdp);

	vector<matrix<double> > grad_logz;
	matrix<matrix<double> > hess_logz;
	compute_grad_hess_logz(lmdp, grad_lmdp, hess_lmdp, path_ig, logz, grad_logz, hess_logz);

	vector<double> grad_L;
	matrix<double> hess_L;
	compute_grad_hess_L(lmdp, grad_lmdp, hess_lmdp, path_ig, grad_logz, hess_logz, grad_L, hess_L);

#if 1
	std::cout<<"grad_L="<<std::endl;
	array1d_print(std::cout, grad_L);
	std::cout<<"hess_L="<<std::endl;
	array2d_print(std::cout, hess_L);
#endif

	permutation_matrix<std::size_t> P(hess_L.size1());

	int res = lu_factorize(hess_L, P);  //hess_L holds L and U
	lu_substitute(hess_L, P, grad_L);       //grad_L holds the solution hess_L_{-1}*grad_L


	wei += -grad_L;
	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);


    }

}



struct lmdp_1f_t
{
    vector<double> q;
    vector<vector<double> > log_pps;

    vector<matrix<double> > log_p_tilde;
    vector<vector<double> > l_tilde;
    vector<vector<double> > nent;

};

void compute_grad_lmdp_numeric(lmdp_t const& lmdp, vector<lmdp_1f_t>& grad_lmdp)
{
    using namespace boost::lambda;
    int nf = lmdp.wei.size();
    double dw = 0.01l;

    grad_lmdp = vector<lmdp_1f_t>(nf);

    for(int ff=0; ff<nf; ++ff)
    {
	lmdp_t lmdp2;
	lmdp2.initialize(lmdp.fg, lmdp.sg, lmdp.yx2ig, lmdp.ig2yx);

	vector<double> wei2 = lmdp.wei;
	wei2(ff) += dw;

	lmdp2.embed(wei2);
	grad_lmdp(ff).q  = (lmdp2.q-lmdp.q)/dw;
	grad_lmdp(ff).log_pps = (lmdp2.log_pps-lmdp.log_pps);// = (log_pps2-log_pps);
	std::for_each(grad_lmdp(ff).log_pps.begin(), grad_lmdp(ff).log_pps.end(), _1/=dw);

	grad_lmdp(ff).log_p_tilde = lmdp2.log_p_tilde-lmdp.log_p_tilde;
	std::for_each(grad_lmdp(ff).log_p_tilde.begin(), grad_lmdp(ff).log_p_tilde.end(), _1/=dw);

	grad_lmdp(ff).l_tilde = lmdp2.l_tilde-lmdp.l_tilde;
	std::for_each(grad_lmdp(ff).l_tilde.begin(), grad_lmdp(ff).l_tilde.end(), _1/=dw);

 	grad_lmdp(ff).nent = lmdp2.nent-lmdp.nent;
	std::for_each(grad_lmdp(ff).nent.begin(), grad_lmdp(ff).nent.end(), _1/=dw);
    }

}

void compute_hess_lmdp_numeric(lmdp_t const& lmdp, matrix<lmdp_1f_t>& hess_lmdp)
{
    using namespace boost::lambda;
    int nf = lmdp.wei.size();
    double dw = 0.01l;

    hess_lmdp = matrix<lmdp_1f_t>(nf, nf);
    vector<lmdp_1f_t> grad_lmdp;

    compute_grad_lmdp_numeric(lmdp, grad_lmdp);

    for(int ff=0; ff<nf; ++ff)
    {
	lmdp_t lmdp2;
	lmdp2.initialize(lmdp.fg, lmdp.sg, lmdp.yx2ig, lmdp.ig2yx);

	vector<double> wei2 = lmdp.wei;
	wei2(ff) += dw;

	lmdp2.embed(wei2);

	vector<lmdp_1f_t> grad_lmdp2;
	compute_grad_lmdp_numeric(lmdp2, grad_lmdp2);
	for(int ee=0; ee<nf; ++ee)
	{
	    hess_lmdp(ff, ee).q = (grad_lmdp2(ee).q-grad_lmdp(ee).q)/dw;

	    hess_lmdp(ff, ee).log_pps = grad_lmdp2(ee).log_pps - grad_lmdp(ee).log_pps;
	    std::for_each(hess_lmdp(ff, ee).log_pps.begin(), hess_lmdp(ff, ee).log_pps.end(), _1/=dw);

	    hess_lmdp(ff, ee).log_p_tilde = grad_lmdp2(ee).log_p_tilde-grad_lmdp(ee).log_p_tilde;
	    std::for_each(hess_lmdp(ff, ee).log_p_tilde.begin(), hess_lmdp(ff, ee).log_p_tilde.end(), _1/=dw);

	    hess_lmdp(ff, ee).l_tilde = grad_lmdp2(ee).l_tilde-grad_lmdp(ee).l_tilde;
	    std::for_each(hess_lmdp(ff, ee).l_tilde.begin(), hess_lmdp(ff, ee).l_tilde.end(), _1/=dw);

	    hess_lmdp(ff, ee).nent = grad_lmdp2(ee).nent-grad_lmdp(ee).nent;
	    std::for_each(hess_lmdp(ff, ee).nent.begin(), hess_lmdp(ff, ee).nent.end(), _1/=dw);

	}
    }
}


void unit_test(lmdp_t& lmdp,
	       vector<vector<int> > const& path_ig,
	       vector<double>& wei)
{
    lmdp.embed(wei);


    vector<grad_state_t> grad_lmdp;
    vector<hess_state_t> hess_lmdp;
    compute_grad_lmdp(lmdp, grad_lmdp);
    compute_hess_lmdp(lmdp, grad_lmdp, hess_lmdp);

    vector<lmdp_1f_t> grad_lmdp2;
    compute_grad_lmdp_numeric(lmdp, grad_lmdp2);

    int nf = lmdp.fg.size1();
    int ng = lmdp.sg.size();

#if 0
    for(int ff=0; ff<nf; ++ff)
    {
	for(int ig=0; ig<ng; ++ig)
	{
	    double v1 = grad_lmdp2(ff).q(ig);
	    double v2 = grad_lmdp(ig).grad_q(ff);
	    std::cout<<v1<<", \t"<<v2 <<"\t\t"<<v1-v2<<"\t\t"<<std::abs(v1-v2)/(std::abs(v1)+std::abs(v2))
		     <<std::endl;


	    for(int nn=0; nn<lmdp.sg(ig).size(); ++nn)
	    {
		std::cout<<grad_lmdp2(ff).log_pps(ig)(nn)<<", \t"<<grad_lmdp(ig).grad_log_pps(ff, nn)
			 <<"\t\t"<<grad_lmdp2(ff).log_pps(ig)(nn)-grad_lmdp(ig).grad_log_pps(ff, nn)
			 <<std::endl;
	    }


	}
    }
#endif

#if 0
    for(int ff=0; ff<nf; ++ff)
    {
	for(int ig=0; ig<ng; ++ig)
	{
	    std::cout<<"grad1="<<grad_lmdp2(ff).l_tilde(ig)<<std::endl;
	    std::cout<<"grad2="<<row(grad_lmdp(ig).grad_l_tilde, ff)<<std::endl;
	    std::cout<<"dgrad="<<grad_lmdp2(ff).l_tilde(ig)- row(grad_lmdp(ig).grad_l_tilde, ff)<<std::endl;
	}
    }
#endif

    matrix<lmdp_1f_t> hess_lmdp2;
    compute_hess_lmdp_numeric(lmdp, hess_lmdp2);

    matrix<lmdp_1f_t> hess_lmdp3 = trans(hess_lmdp2);

    std::cout<<"check hessian"<<std::endl;

#if 0
    for(int ff=0; ff<nf; ++ff)
    {
	for(int ee=0; ee<nf; ++ee)
	{
	    if(ff+1<nf || ee+1<nf) continue;
	    for(int ig=0; ig<ng; ++ig)
	    {
		//std::cout<<hess_lmdp(ig).hess_log_p_tilde(ff, ee)<<std::endl;
		//std::cout<< hess_lmdp2(ff, ee).log_p_tilde(ig)<<std::endl;
		//std::cout<< hess_lmdp2(ff, ee).log_p_tilde(ig)-hess_lmdp(ig).hess_log_p_tilde(ff, ee)<<std::endl;
		std::cout<<"hess1="<< hess_lmdp2(ff, ee).l_tilde(ig)<<std::endl;
		std::cout<<"hess2="<< hess_lmdp(ig).hess_l_tilde(ff, ee)<<std::endl;
		std::cout<<"dhess="<< hess_lmdp2(ff, ee).l_tilde(ig)-hess_lmdp(ig).hess_l_tilde(ff, ee)<<std::endl;
	    }
	}
    }
#endif

    for(int ff=0; ff<nf; ++ff)
    {
	for(int ee=0; ee<nf; ++ee)
	{
	    if(ee==ff) continue;
	    for(int ig=0; ig<ng; ++ig)
	    {

#if 0
		std::cout<<hess_lmdp2(ff, ee).q(ig)<<", \t"<<hess_lmdp2(ee, ff).q(ig)
			 <<"\t\t"<<hess_lmdp2(ff, ee).q(ig)-hess_lmdp2(ee, ff).q(ig)
			 <<std::endl;
#endif

#if 1
		std::cout<<hess_lmdp2(ff, ee).q(ig)<<", \t"<<hess_lmdp(ig).hess_q(ff, ee)
			 <<"\t\t"<<hess_lmdp2(ff, ee).q(ig)-hess_lmdp(ig).hess_q(ff, ee)
			 <<std::endl;
#endif

#if 0
		for(int nn=0; nn<lmdp.sg(ig).size(); ++nn)
		{
		    std::cout<<hess_lmdp2(ff, ee).log_pps(ig)(nn)<<", \t"<<hess_lmdp(ig).hess_log_pps(ff, ee)(nn)
			     <<"\t\t"<<hess_lmdp2(ff, ee).log_pps(ig)(nn)-hess_lmdp(ig).hess_log_pps(ff, ee)(nn)
			     <<std::endl;
		}
#endif
	    }
	}
    }

}

#endif

