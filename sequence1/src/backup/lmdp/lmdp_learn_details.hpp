#ifndef __LMDP__LEARN__DETAILS__HPP__INCLUDED__
#define __LMDP__LEARN__DETAILS__HPP__INCLUDED__

void compute_grad_plh(lmdp_t const& lmdp, int ig, grad_state_t& grad_state)
{
    using namespace boost::lambda;

    matrix<vector<double> >const& fg = lmdp.fg;
    vector<double> const& wei = lmdp.wei;
    vector<vector<int> > const& sg = lmdp.sg;
    vector<matrix<double> > const& p_tilde = lmdp.p_tilde;
    vector<matrix<double> > const& log_p_tilde = lmdp.log_p_tilde;
    matrix<int> const& ig2yx = lmdp.ig2yx;

    vector<matrix<double> > & grad_log_p_tilde = grad_state.grad_log_p_tilde;
    vector<vector<double> > & grad_l_tilde = grad_state.grad_l_tilde;
    vector<vector<double> > & grad_nent = grad_state.grad_nent;

    int nf = fg.size1()+1; //+1 for lambda
    int nnb = sg(ig).size();

    grad_log_p_tilde =
	scalar_vector<matrix<double> >(nf,
				       scalar_matrix<double>(nnb,nnb, 0.0l));
    grad_l_tilde =
	scalar_vector<vector<double> >(nf,
				       scalar_vector<double>(nnb, 0.0l));
    grad_nent =
	scalar_vector<vector<double> >(nf,
				       scalar_vector<double>(nnb, 0.0l));

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


    grad_l_tilde =
	scalar_vector<vector<double> >(nf,
				       scalar_vector<double>(nnb, 0.0l) );
    for(int ff=0; ff+1<nf; ++ff)
    {
	for(int aa=0; aa<grad_l_tilde(ff).size(); ++aa)
	{
	    grad_l_tilde(ff)(aa) = inner_prod(row(p_tilde(ig), aa), fg(ff, ig));
	}
    }

    vector<double> di=scalar_vector<double>(nnb, 0.0l);
    for(int ff=0; ff+1<nf; ++ff)
    {
	for(int nn=0; nn<nnb; ++nn)
	{
	    di(nn) += wei(ff)*fg(ff, ig)(nn);
	}
    }
    //std::cout<<di<<std::endl;
    grad_nent = scalar_vector<scalar_vector<double> >(nf, scalar_vector<double>(nnb, 0.0l) );
    for(int aa=0; aa<nnb; ++aa)
    {
	vector<double> pgrad_logp = element_prod(row(p_tilde(ig), aa), row(grad_log_p_tilde(nf-1), aa));
	grad_l_tilde(nf-1)(aa) = inner_prod(pgrad_logp, di);
	vector<double> logp_plus1= row(log_p_tilde(ig), aa);
	std::for_each(logp_plus1.begin(), logp_plus1.end(), _1+=1);
	grad_nent(nf-1)(aa) = inner_prod(pgrad_logp, logp_plus1);
    }


}

void compute_grad_q_pps(lmdp_t const& lmdp, int ig, grad_state_t& grad_state)
{
    using namespace boost::lambda;
    matrix<double> const& p_tilde = lmdp.p_tilde(ig);
    vector<double> const& log_pps = lmdp.log_pps(ig);
			    
    vector<matrix<double> > const& grad_log_ptd = grad_state.grad_log_p_tilde;
    vector<vector<double> > const& grad_ltd = grad_state.grad_l_tilde;
    vector<vector<double> > const& grad_nent = grad_state.grad_nent;
    vector<double> & grad_q = grad_state.grad_q;
    vector<vector<double> > & grad_log_pps = grad_state.grad_log_pps;

    int nf = grad_log_ptd.size();
    int nnb = log_pps.size();
    grad_log_pps = scalar_vector<vector<double> >(nf, scalar_vector<double>(nnb, 0.0l) );
    grad_q = scalar_vector<double>(nf, 0.0l);

    matrix<double> B = p_tilde;
    permutation_matrix<std::size_t> P(nnb);
    int res = lu_factorize(B, P);  //B holds L and U

    vector<double> pps(log_pps.size());
    double (*fexp)(double) = std::exp;
    std::transform(log_pps.begin(), log_pps.end(), pps.begin(), fexp);

    for(int ff=0; ff<nf; ++ff)
    {
	matrix<double> pglogp= element_prod(p_tilde, grad_log_ptd(ff));
	vector<double> nvec = prod(pglogp, log_pps)
	    + grad_ltd(ff)-grad_nent(ff);

	lu_substitute(B, P, nvec); //nvec holds the solution B_{-1}*nvec

	grad_q(ff) = inner_prod(pps, nvec);

	vector<double> m_bar = -nvec;
	std::for_each(m_bar.begin(), m_bar.end(), _1 += grad_q(ff));
	grad_log_pps(ff) = m_bar;
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
    vector<vector<double> > const& grad_l_tilde = grad_state.grad_l_tilde;
    vector<vector<double> > const& grad_nent = grad_nent;

    matrix<matrix<double> > & hess_log_p_tilde = hess_state.hess_log_p_tilde;
    matrix<vector<double> > & hess_l_tilde = hess_state.hess_l_tilde;
    matrix<vector<double> > & hess_nent = hess_state.hess_nent;

    int nf = fg.size1()+1; //+1 for lambda
    int nnb = sg(ig).size();

    hess_log_p_tilde = scalar_matrix<matrix<double> >(nf, nf, scalar_matrix<double>(nnb, nnb, 0.0l));
    hess_l_tilde = scalar_matrix<vector<double> >(nf, nf, scalar_vector<double>(nnb, 0.0l));
    hess_nent = scalar_matrix<vector<double> >(nf, nf, scalar_vector<double>(nnb, 0.0l));

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

    vector<double> di=scalar_vector<double>(nnb, 0.0l);
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
    vector<vector<double> > const& grad_l_tilde = grad_state.grad_l_tilde;
    vector<vector<double> > const& grad_nent = grad_state.grad_nent;
    vector<double> const& grad_q = grad_state.grad_q;
    vector<vector<double> > const& grad_log_pps = grad_state.grad_log_pps;

    matrix<matrix<double> > const& hess_log_p_tilde = hess_state.hess_log_p_tilde;
    matrix<vector<double> > const& hess_l_tilde = hess_state.hess_l_tilde;
    matrix<vector<double> > const& hess_nent = hess_state.hess_nent;
    matrix<double>& hess_q = hess_state.hess_q;
    matrix<vector<double> >& hess_log_pps = hess_state.hess_log_pps;

    int nf = grad_log_p_tilde.size();
    int nnb = log_pps.size();
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
	    nvec += prod(gradl_p, grad_log_pps(ff) )
		+ prod(gradk_p, grad_log_pps(ee) )
		+ hess_l_tilde(ee, ff)-hess_nent(ee, ff);   
	    //std::cout<<"nvec="<<nvec<<std::endl;

	    lu_substitute(B, P, nvec);       //nvec holds the solution B_{-1}*nvec

	    double sx = -inner_prod(pps, element_prod(grad_log_pps(ee), grad_log_pps(ff)));

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
		       vector<lmdp_1f_t>& grad_lmdp)
{
    int ng = lmdp.sg.size();
    int nf = lmdp.wei.size();
    grad_lmdp = scalar_vector<lmdp_1f_t>(nf, lmdp_1f_t(ng));
    for(int ig=0; ig < lmdp.sg.size(); ++ig)
    {
	grad_state_t grad_state; // = grad_lmdp(ig);
	compute_grad_plh(lmdp, ig, grad_state);
	compute_grad_q_pps(lmdp, ig, grad_state);

	for(int ff=0; ff<lmdp.wei.size(); ++ff)
	{
	    grad_lmdp(ff).q(ig) = grad_state.grad_q(ff);
	    grad_lmdp(ff).log_pps(ig) = grad_state.grad_log_pps(ff);
	    grad_lmdp(ff).log_p_tilde(ig) = grad_state.grad_log_p_tilde(ff);
	    grad_lmdp(ff).l_tilde(ig) = grad_state.grad_l_tilde(ff);
	    grad_lmdp(ff).nent(ig) = grad_state.grad_nent(ff);
	}

    }

}

void compute_hess_lmdp(lmdp_t const& lmdp,
		       vector<lmdp_1f_t> const& grad_lmdp,
		       matrix<lmdp_1f_t>& hess_lmdp)
{
    int ng = lmdp.sg.size();
    int nf = lmdp.wei.size();
    hess_lmdp = scalar_matrix<lmdp_1f_t>(nf, nf,
					 lmdp_1f_t(ng));
    for(int ig=0; ig < lmdp.sg.size(); ++ig)
    {
	grad_state_t grad_state(nf);

	for(int ff=0; ff<nf; ++ff)
	{
	    grad_state.grad_q(ff) = grad_lmdp(ff).q(ig);
	    grad_state.grad_log_pps(ff) = grad_lmdp(ff).log_pps(ig);
	    grad_state.grad_log_p_tilde(ff) = grad_lmdp(ff).log_p_tilde(ig);
	    grad_state.grad_l_tilde(ff) = grad_lmdp(ff).l_tilde(ig);
	    grad_state.grad_nent(ff) = grad_lmdp(ff).nent(ig);
	}

	hess_state_t hess_state;// = hess_lmdp(ig);
	compute_hess_plh(lmdp, ig, grad_state, hess_state);
	compute_hess_q_pps(lmdp, ig, grad_state, hess_state);

	for(int ff=0; ff<lmdp.wei.size(); ++ff)
	{
	    for(int ee=0; ee<nf; ++ee)
	    {
		hess_lmdp(ff, ee).q(ig) = hess_state.hess_q(ff, ee);
		hess_lmdp(ff, ee).log_pps(ig) = hess_state.hess_log_pps(ff, ee);
		hess_lmdp(ff, ee).log_p_tilde(ig) = hess_state.hess_log_p_tilde(ff, ee);
		hess_lmdp(ff, ee).l_tilde(ig) = hess_state.hess_l_tilde(ff, ee);
		hess_lmdp(ff, ee).nent(ig) = hess_state.hess_nent(ff, ee);
	    }
	}

    }
}


void compute_grad_hess_logz(lmdp_t const& lmdp,
			    vector<lmdp_1f_t> const& grad_lmdp,
			    matrix<lmdp_1f_t> const& hess_lmdp,
			    vector<vector<int> >const& path_ig,
			    vector<vector<double> >const & logz,
			    vector<vector<vector<double> > >& grad_logz,
			    matrix<vector<vector<double> > >& hess_logz)
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
    vector<vector<double> > zero_vec_vec=
	scalar_vector<vector<double> >(logz.size(),
				       scalar_vector<double>(ng, 0.0l));
    grad_logz = scalar_vector<vector<vector<double> > >(nf, zero_vec_vec);
    //std::cout<<grad_logz(0)(0).size()<<std::endl;

    hess_logz =	scalar_matrix<vector<vector<double> > >(nf, nf, zero_vec_vec);

    for(int cc=0; cc<logz.size(); ++cc)
    {

	vector<int> good=scalar_vector<int>(ng, 0);
	good(*(path_ig(cc).rbegin())) = 1;

	umf_sparse_matrix A(ng, ng, ng+nnz);
	matrix<double> nvec=scalar_matrix<double>(nf, ng, 0.0l);

	matrix<vector<double> > nvec2 =
	    scalar_matrix<vector<double> >(nf, nf,
					   scalar_vector<double>(ng, 0.0l));//(nf, nf, ng);


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
					  grad_lmdp(ff).log_pps(ig))
		    - grad_lmdp(ff).q(ig);
	    }
	}

	splitter_t<vector<int> > splitter(good);
	matrix<umf_sparse_matrix> Ablock;
	splitter.split_sparse_mat(A, Ablock);
	//split_sparse_matrix(A, good, Ablock);


	umf::symbolic_type<double> symb;
	umf::numeric_type<double> nume;

	umf::symbolic (Ablock(0,0), symb); 
	umf::numeric (Ablock(0,0), symb, nume); 

	for(int ff=0; ff<nf; ++ff)
	{
	    vector<double> grad_logz_one=scalar_vector<double>(ng, 0.0l);
	    vector<double> nvec_one(row(nvec, ff));
	    vector<vector<double> > nvec_block;
	    splitter.split(nvec_one, nvec_block);
	    vector<vector<double> > sol_block;
	    splitter.split(grad_logz_one, sol_block);

	    vector<double> nvec_new = nvec_block(0)
		- prod(Ablock(0, 1), sol_block(1));

	    umf::solve(Ablock(0,0), sol_block(0), nvec_new, nume);  

	    //grad_logz_one = splitter.merge(sol_block);
	    splitter.merge(sol_block, grad_logz_one);

	    //column(grad_logz(cc), ff) = grad_logz_one;
	    grad_logz(ff)(cc) = grad_logz_one;;

	    //std::cout<<"grad_logz_one="<<project(grad_logz_one, range(340, 350))<<std::endl;
	}

	    
	for(int ee=0; ee<nf; ++ee)
	{
	    for(int ff=0; ff<nf; ++ff)
	    {
		for(int ig=0; ig<ng; ++ig)
		{
		    nvec2(ee, ff)(ig) = - hess_lmdp(ee, ff).q(ig)
			- (grad_lmdp(ee).q(ig)+grad_logz(ee)(cc)(ig))
			* (grad_lmdp(ff).q(ig)+grad_logz(ff)(cc)(ig));
		    vector<double> tmp(sg(ig).size());
		    for(int nn=0; nn<sg(ig).size(); ++nn)
		    {
			tmp(nn) = hess_lmdp(ee, ff).log_pps(ig)(nn) 
			    + (grad_lmdp(ee).log_pps(ig)(nn) + grad_logz(ee)(cc)(ig))
			    * (grad_lmdp(ff).log_pps(ig)(nn) + grad_logz(ff)(cc)(ig));
		    }
		    //std::cout<<"tmp.size="<<tmp.size()<<std::endl;
		    //std::cout<<"pexp.size="<<pexp_list(ig).size()<<std::endl;
		    nvec2(ee, ff)(ig) += inner_prod(tmp, pexp_list(ig));
		}
		vector<double> hess_logz_one=scalar_vector<double>(ng, 0.0l);
		vector<vector<double> > nvec2_block;
		splitter.split(nvec2(ee, ff), nvec2_block);
		vector<vector<double> > sol_block;
		splitter.split(hess_logz_one, sol_block);

		vector<double> nvec2_new = nvec2_block(0)
		    - prod(Ablock(0, 1), sol_block(1));

		umf::solve(Ablock(0,0), sol_block(0), nvec2_new, nume);  

		//hess_logz_one = splitter.merge(sol_block);
		splitter.merge(sol_block, hess_logz_one);

		//row(hess_logz(cc), ff) = hess_logz_one;
		//for(int ig=0; ig<ng; ++ig)
		//{
		//    hess_logz(cc, ig)(ee, ff) = hess_logz_one(ig);
		//}
		hess_logz(ee, ff)(cc) = hess_logz_one;

	    }
	}

	
    }


}


void compute_grad_hess_L(lmdp_t const& lmdp,
			 vector<lmdp_1f_t> const& grad_lmdp,
			 matrix<lmdp_1f_t> const& hess_lmdp,
			 vector<vector<int> >const& path_ig,
			 vector<vector<vector<double> > > const& grad_logz,
			 matrix<vector<vector<double> > > const& hess_logz,
			 vector<double>& grad_L,
			 matrix<double>& hess_L)
{
    using namespace boost::lambda;
    vector<vector<int> > const& sg = lmdp.sg;

    int nf = lmdp.fg.size1()+1; //+1 for lambda position
    int ng = sg.size();

    double alpha = -1;//-0.1; //-0.0001;// -1;

    //regularization
    grad_L = alpha*lmdp.wei; //grad_L = scalar_vector<double>(nf, 0.0l);
    hess_L = alpha*identity_matrix<double>(nf); //hess_L = scalar_matrix<double>(nf, nf, 0.0l);

    typedef vector<double> vec_t;
    typedef vector<vector<double> > vec_vec_t;
    vec_t::const_reference (vec_t::*cref)(vec_t::size_type) const
	= &vec_t::operator();
    vec_vec_t::const_reference (vec_vec_t::*cvref)(vec_vec_t::size_type) const
	= &vec_vec_t::operator();

    for(int cc=0; cc<path_ig.size(); ++cc)
    {
	vector<double> grad_logz0(nf);
	array1d_transform(grad_logz, grad_logz0,
			  bind(cref, bind(cvref, _1, cc), path_ig(cc)(0)));
	matrix<double> hess_logz0(nf, nf);
	array2d_transform(hess_logz, hess_logz0,
			  bind(cref, bind(cvref, _1, cc), path_ig(cc)(0)));

	grad_L += -grad_logz0; //row(grad_logz(cc), path_ig(cc)(0) );
	hess_L += -hess_logz0; //hess_logz(cc, path_ig(cc)(0));

	for(int tt=0; tt+1<path_ig(cc).size(); ++tt)
	{
	    int ig = path_ig(cc)(tt);
	    int ig2 = path_ig(cc)(tt+1);

	    vector<double> grad_q(nf);
	    array1d_transform(grad_lmdp, grad_q,
			      bind(cref, bind(&lmdp_1f_t::q, _1), ig ));
	    matrix<double> hess_q(nf, nf);
	    array2d_transform(hess_lmdp, hess_q,
			      bind(cref, bind(&lmdp_1f_t::q, _1), ig ));

	    vector<vector<double> > grad_log_pps;
	    array1d_transform(grad_lmdp, grad_log_pps,
			      bind(cvref,
				   bind(&lmdp_1f_t::log_pps, _1), ig));
	    matrix<vector<double> > hess_log_pps;
	    array2d_transform(hess_lmdp, hess_log_pps,
			      bind(cvref,
				   bind(&lmdp_1f_t::log_pps, _1), ig));

	    int nn = std::find(sg(ig).begin(), sg(ig).end(), ig2)
		-sg(ig).begin();
	
	    vector<double> grad_lpps_one(nf);
	    array1d_transform(grad_log_pps, grad_lpps_one,
			      bind(cref, _1, nn));
	    grad_L += grad_lpps_one - grad_q;

	    matrix<double> hess_lpps_one(nf, nf);
	    array2d_transform(hess_log_pps, hess_lpps_one,
			      bind(cref, _1, nn));
	    hess_L += hess_lpps_one - hess_q;
	}

	{
	    int tt = path_ig(cc).size()-1;
	    int ig = path_ig(cc)(tt);

	    vector<double> grad_q(nf);
	    array1d_transform(grad_lmdp, grad_q,
			      bind(cref, bind(&lmdp_1f_t::q, _1), ig ));
	    matrix<double> hess_q(nf, nf);
	    array2d_transform(hess_lmdp, hess_q,
			      bind(cref, bind(&lmdp_1f_t::q, _1), ig ));

	    grad_L += - grad_q;
	    hess_L += - hess_q;
	}

    }

}

void compute_L(lmdp_t const& lmdp,
	       vector<vector<int> >const& path_ig,
	       vector<vector<double> > const& logz,
	       vector<double>& L)
{
    using namespace boost::lambda;
    vector<vector<int> > const& sg = lmdp.sg;

    int nf = lmdp.fg.size1()+1; //+1 for lambda position
    int ng = sg.size();

    int nc = path_ig.size();
    L = scalar_vector<double>(nc, 0.0l);

    for(int cc=0; cc<nc; ++cc)
    {

	int ig0 = path_ig(cc)(0);
	L(cc) += -logz(cc)(ig0);

	for(int tt=0; tt+1<path_ig(cc).size(); ++tt)
	{
	    int ig = path_ig(cc)(tt);
	    int ig2 = path_ig(cc)(tt+1);

	    int nn = std::find(sg(ig).begin(), sg(ig).end(), ig2)
		-sg(ig).begin();
	
	    L(cc) += lmdp.log_pps(ig)(nn) - lmdp.q(ig);

	}
	{
	    int tt = path_ig(cc).size()-1;
	    int ig = path_ig(cc)(tt);

	    L(cc) += -lmdp.q(ig);
	}

    }

}



void learn_weights(lmdp_t& lmdp,
		   vector<vector<int> > const& path_ig,
		   vector<double>& wei)
{

    for(int it=0; it<10; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	lmdp.embed(wei);

	vector<vector<double> > logz(path_ig.size());

	for(int cc=0; cc<path_ig.size(); ++cc)
	{
	    lmdp.solve(*(path_ig(cc).rbegin()), logz(cc));
	}

	//compute gradient and hessian of q and pps
	vector<lmdp_1f_t> grad_lmdp;
	matrix<lmdp_1f_t> hess_lmdp;
	compute_grad_lmdp(lmdp, grad_lmdp);
	compute_hess_lmdp(lmdp, grad_lmdp, hess_lmdp);

	vector<vector<vector<double> > > grad_logz;
	matrix<vector<vector<double> > > hess_logz;
	compute_grad_hess_logz(lmdp, grad_lmdp, hess_lmdp, path_ig, logz, grad_logz, hess_logz);

	vector<double> grad_L;
	matrix<double> hess_L;
	compute_grad_hess_L(lmdp, grad_lmdp, hess_lmdp, path_ig, grad_logz, hess_logz, grad_L, hess_L);

	permutation_matrix<std::size_t> P(hess_L.size1());

	int res = lu_factorize(hess_L, P);  //hess_L holds L and U
	lu_substitute(hess_L, P, grad_L);       //grad_L holds the solution hess_L_{-1}*grad_L


	wei += -grad_L;
	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);

	vector<double> L;
	compute_L(lmdp, path_ig, logz, L);
	std::cout<<"L="<<L(0)<<std::endl;

    }

}


void learn_weights_greedy(lmdp_t& lmdp,
			  vector<vector<int> > const& path_ig,
			  vector<double>& wei)
{

    for(int it=0; it<20; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	lmdp.embed(wei);

	vector<vector<double> > logz(path_ig.size());

	lmdp.solve(path_ig, logz);
	vector<double> L;
	compute_L(lmdp, path_ig, logz, L);
	double sum_L = sum(L);

	std::vector<vector<double> > weis;
	for(int ff=0; ff<wei.size(); ++ff)
	{
	    vector<double> wei2 = wei;
	    for(int kk=-7; kk<7; ++kk)
	    {
		wei2(ff) = std::exp(std::log(wei(ff))+kk*0.03l);
		weis.push_back(wei2);
	    }
	}

	vector<double> scores(weis.size());
	for(int ww=0; ww<weis.size(); ++ww)
	{
	    lmdp_t lmdp2 = lmdp;
	    lmdp2.embed(weis[ww]);

	    vector<vector<double> > logz2;

	    lmdp2.solve(path_ig, logz2);
	    vector<double> L2;
	    compute_L(lmdp2, path_ig, logz2, L2);
	    scores(ww) = sum(L2);
	    if(std::numeric_limits<double>::infinity()==scores(ww))
		scores(ww) = -std::numeric_limits<double>::infinity();
	}

	int idx = std::max_element(scores.begin(), scores.end())
	    - scores.begin();

	wei = weis[idx];

	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);

	std::cout<<"L="<<L(0)<<std::endl;

	vector<lmdp_1f_t> grad_lmdp;
	matrix<lmdp_1f_t> hess_lmdp;
	compute_grad_lmdp(lmdp, grad_lmdp);
	compute_hess_lmdp(lmdp, grad_lmdp, hess_lmdp);

	vector<vector<vector<double> > > grad_logz;
	matrix<vector<vector<double> > > hess_logz;
	compute_grad_hess_logz(lmdp, grad_lmdp, hess_lmdp, path_ig, logz, grad_logz, hess_logz);

	vector<double> grad_L;
	matrix<double> hess_L;
	compute_grad_hess_L(lmdp, grad_lmdp, hess_lmdp, path_ig, grad_logz, hess_logz, grad_L, hess_L);

	std::cout<<"grad_L="<<grad_L<<std::endl;


    }

}



#endif
