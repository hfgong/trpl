#ifndef __LMDP__DETAILS__HPP__INCLUDED__
#define __LMDP__DETAILS__HPP__INCLUDED__
#include <boost/iterator/counting_iterator.hpp>
#include "lmdp.hpp"

void lmdp_t::initialize(matrix<vector<double> > const& fg_,
			vector<vector<int> > const& sg_,
			matrix<int> const& yx2ig_,
			matrix<int> const& ig2yx_)
{
    fg = fg_;
    sg = sg_;
    yx2ig = yx2ig_;
    ig2yx = ig2yx_;
}


void lmdp_t::embed(vector<double> const& wei_)
{
    using namespace boost::lambda;
    wei = wei_;
    int ng = ig2yx.size1();//std::count_if(yx2ig.data().begin(), yx2ig.data().end(), _1>=0);
    double lambda = *(wei.rbegin());
    if(log_pps.size()==0)
    {
	q = scalar_vector<double>(ng, 0.0l);

	log_pps = vector<vector<double> >(ng);
	for(int ii=0; ii<ng; ++ii)
	{
	    log_pps(ii) = scalar_vector<double>(sg(ii).size(), 0);
	}
	p_tilde = vector<matrix<double> >(ng);
	log_p_tilde = vector<matrix<double> >(ng);
	for(int ii=0; ii<ng; ++ii)
	{
	    p_tilde(ii) = scalar_matrix<double>(sg(ii).size(), sg(ii).size(), 0);
	    log_p_tilde(ii) = scalar_matrix<double>(sg(ii).size(), sg(ii).size(), 0);
	}

	l_tilde = vector<vector<double> >(ng);
	nent = vector<vector<double> >(ng);
	for(int ii=0; ii<ng; ++ii)
	{
	    l_tilde(ii) = scalar_vector<double>(sg(ii).size(), 0.0l);
	    nent(ii) = scalar_vector<double>(sg(ii).size(), 0.0l);
	}
    }

    vector<vector<double> > fdist(ng);
    for(int gg=0; gg<ng; ++gg)
    {
	fdist(gg) = scalar_vector<double>(sg(gg).size(), 0.0l);
	for(int nn=0; nn<sg(gg).size(); ++nn)
	{
	    for(int ff=0; ff<fg.size1(); ++ff)
	    {
		fdist(gg)(nn) += fg(ff, gg)(nn)*wei(ff);
	    }
	}
    }

    double (*fexp)(double) = std::exp;

    for(int ii=0; ii<ng; ++ii)
    {
	int nnb = sg(ii).size();

	for(int aa=0; aa<nnb; ++aa)
	{
	    int yy = ig2yx(sg(ii)(aa), 0);
	    int xx = ig2yx(sg(ii)(aa), 1);
	    vector<double> nldist(nnb);
	    for(int jj=0; jj<nnb; ++jj)
	    {
		int y2 = ig2yx(sg(ii)(jj), 0);
		int x2 = ig2yx(sg(ii)(jj), 1);
		double dist = (yy-y2)*(yy-y2)+(xx-x2)*(xx-x2);
		nldist(jj) = -lambda*dist;
	    }
	    double mv = *std::max_element(nldist.begin(), nldist.end());
	    std::for_each(nldist.begin(), nldist.end(), _1-=mv);
	    vector<double> prob(nnb);
	    std::transform(nldist.begin(), nldist.end(), prob.begin(), fexp);

	    double denom = sum(prob);
	    std::for_each(prob.begin(), prob.end(), _1/=denom);
	    std::for_each(nldist.begin(), nldist.end(), _1-=std::log(denom));
	    row(p_tilde(ii), aa) = prob;
	    row(log_p_tilde(ii), aa) = nldist;
	    l_tilde(ii)(aa) = inner_prod(prob, fdist(ii));
	    nent(ii)(aa) = inner_prod(prob, nldist);

	    //Below inner product code is error, which costs me one week 
	    //to debug, "0.0l" should be changed to "0.0l".
	    //Now I use ublas::inner_prod, which has no such sucks.
	    //l_tilde(ii)(aa) = std::inner_product(prob.begin(), prob.end(), fdist(ii).begin(), 0.0f);
	    //nent(ii)(aa) = std::inner_product(prob.begin(), prob.end(), nldist.begin(), 0.0f);
	}
	vector<double> nvec(l_tilde(ii)-nent(ii));
	matrix<double> B = p_tilde(ii); 
	permutation_matrix<std::size_t> P(nvec.size());
	//array2d_print(std::cout, B);


	int res = lu_factorize(B, P);  //B holds L and U
	lu_substitute(B, P, nvec);       //nvec holds the solution B_{-1}*nvec
	vector<double> m_bar = -nvec;

	q(ii) = -log_sum_exp(m_bar);
	vector<double> log_pps_row(m_bar.size());
	std::transform(m_bar.begin(), m_bar.end(), log_pps_row.begin(), _1+=q(ii));
	log_pps(ii) = log_pps_row;
    }

    //std::cout<<q<<std::endl;
}



#if 0
void lmdp_t::solve(int goal, vector<double>& logz) const
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0l);

    int nnz = count_sg_nzz();

    umf_sparse_matrix qp(ng, ng, nnz);

    for(int ig = 0; ig<ng; ++ig)
    {
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    qp(ig, ig2) = std::exp(-q(ig)+log_pps(ig)(nn));
	}
    }
    vector<int> good=scalar_vector<int>(ng, 0);
    good(goal) = 1;

    splitter_t<vector<int> > splitter(good);
    matrix<umf_sparse_matrix> qp_block;
    splitter.split_sparse_mat(qp, qp_block);

    vector<double> z(scalar_vector<double>(ng, 1.0l));
    vector<vector<double> > z_block;
    splitter.split(z, z_block);

    vector<double> A12z2(qp_block(0, 1).size1());
    axpy_prod(qp_block(0, 1), z_block(1), A12z2, true);
    umf_sparse_matrix A = -qp_block(0,0);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) = 1.0l;
    }

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    umf::solve(A, z_block(0), A12z2, nume);  
    splitter.merge(z_block, z);

#if 0
    vector<double> error = z-prod(qp, z);
    vector<double> error_rate(error.size());
    std::transform(error.begin(), error.end(), z.begin(),
		   error_rate.begin(), _1/_2);
    double max_er = *std::max_element(error_rate.begin(),
				      error_rate.end()-1);
    double min_er = *std::min_element(error_rate.begin(),
				      error_rate.end()-1);
    std::cout<<"error_rate in"<<min_er<<", "<<max_er<<std::endl;
    std::cout<<"log error_rate in"<<std::log(-min_er)
	     <<", "<<std::log(max_er)<<std::endl;
#endif

    logz = vector<double>(z.size());
    std::transform(z.begin(), z.end(), logz.begin(), flog);

}
#endif

void lmdp_t::solve(int goal, vector<double>& logz) const
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0l);

    int nnz = count_sg_nzz();

    umf_sparse_matrix qp(ng, ng, nnz);

    for(int ig = 0; ig<ng; ++ig)
    {
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    qp(ig, ig2) = std::exp(-q(ig)+log_pps(ig)(nn));
	}
    }
    vector<int> good=scalar_vector<int>(ng, 0);
    good(goal) = 1;

    splitter_t<vector<int> > splitter(good);
    matrix<umf_sparse_matrix> qp_block;
    //split_sparse_mat<matrix>(splitter, qp);
    splitter.split_sparse_mat(qp, qp_block);


    vector<double> z(scalar_vector<double>(ng, 1.0l));
    vector<vector<double> > z_block;
    splitter.split(z, z_block);

    vector<double> A12z2(qp_block(0, 1).size1());
    axpy_prod(qp_block(0, 1), z_block(1), A12z2, true);
    umf_sparse_matrix A = -qp_block(0,0);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) = 1.0l;
    }

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    umf::solve(A, z_block(0), A12z2, nume);  
    splitter.merge(z_block, z);

#if 0
    vector<double> error = z-prod(qp, z);
    vector<double> error_rate(error.size());
    std::transform(error.begin(), error.end(), z.begin(),
		   error_rate.begin(), _1/_2);
    double max_er = *std::max_element(error_rate.begin(),
				      error_rate.end()-1);
    double min_er = *std::min_element(error_rate.begin(),
				      error_rate.end()-1);
    std::cout<<"error_rate in"<<min_er<<", "<<max_er<<std::endl;
    std::cout<<"log error_rate in"<<std::log(-min_er)
	     <<", "<<std::log(max_er)<<std::endl;
#endif

    logz = vector<double>(z.size());
    std::transform(z.begin(), z.end(), logz.begin(), flog);

#if 0
    vector<long double> log_qpz(logz.size());

    for(int ig = 0; ig<ng-1; ++ig)
    {
	vector<long double> log_qp_z(sg(ig).size());
	for(int nn=0; nn<sg(ig).size(); ++nn)
	{
	    int ig2 = sg(ig)(nn);
	    log_qp_z(nn) = -q(ig)+log_pps(ig)(nn) + logz(ig2);
	}
	log_qpz(ig) = log_sum_exp_sort(log_qp_z);
    }

    vector<long double> log_error = logz-log_qpz;
    //std::cout<<"log_error="<<log_error<<std::endl;
    long double max_er = *std::max_element(log_error.begin(),
				      log_error.end()-1);
    long double min_er = *std::min_element(log_error.begin(),
				      log_error.end()-1);
    std::cout<<"log_error in "<<min_er<<", "<<max_er<<std::endl;
#endif

    vector<std::pair<double, int> > logz_idx(logz.size());
    std::transform(logz.begin(), logz.end(), counting_iterator<int>(0),
		   logz_idx.begin(),
		   std::make_pair<double, int>);

    std::sort(logz_idx.begin(), logz_idx.end(),
	      bind(std::greater<double>(),
	      bind(&std::pair<double, int>::first, _1),
	      bind(&std::pair<double, int>::first, _2) ) );

#if 0
    for(int ii=0; ii<logz_idx.size(); ++ii)
	std::cout<<logz_idx(ii).first<<", ";
    std::cout<<std::endl;
#endif

#if 0

    vector<long double> logz2(logz);

    for(int it=0; it<100; ++it)
    {
	for(int ii = 0; ii<ng; ++ii)
	{
	    int ig = logz_idx(ii).second;
	    if(ig == goal) continue;

	    vector<long double> log_qp_z(sg(ig).size());
	    for(int nn=0; nn<sg(ig).size(); ++nn)
	    {
		int ig2 = sg(ig)(nn);
		log_qp_z(nn) = -q(ig)+log_pps(ig)(nn) + logz2(ig2);
	    }
	    logz2(ig) = log_sum_exp_sort(log_qp_z);
	}
    }

    vector<double> logz2d = logz2;
    vector<double> dlogz = logz2d-logz;
    std::cout<<"dlogz="<<dlogz<<std::endl;
#endif

}


void lmdp_t::solve(vector<vector<int> > const& path_ig,
		   vector<vector<double> >& logz) const
{
    logz = vector<vector<double> >(path_ig.size());
    for(int cc=0; cc<path_ig.size(); ++cc)
    {
	solve(*(path_ig(cc).rbegin()), logz(cc));
    }

}
#endif
