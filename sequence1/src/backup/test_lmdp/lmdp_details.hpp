#ifndef __LMDP__DETAILS__HPP__INCLUDED__
#define __LMDP__DETAILS__HPP__INCLUDED__

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
	q = scalar_vector<double>(ng, 0.0f);

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
	    l_tilde(ii) = scalar_vector<double>(sg(ii).size(), 0);
	    nent(ii) = scalar_vector<double>(sg(ii).size(), 0);
	}
    }

    vector<vector<double> > fdist(ng);
    for(int gg=0; gg<ng; ++gg)
    {
	fdist(gg) = scalar_vector<double>(sg(gg).size(), 0);
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
	    std::transform(nldist.begin(), nldist.end(), prob.begin(), bind(fexp, _1));

	    double denom = std::accumulate(prob.begin(), prob.end(), 0.0f);
	    std::for_each(prob.begin(), prob.end(), _1/=denom);
	    std::for_each(nldist.begin(), nldist.end(), _1-=std::log(denom));
	    row(p_tilde(ii), aa) = prob;
	    row(log_p_tilde(ii), aa) = nldist;
	    l_tilde(ii)(aa) = std::inner_product(prob.begin(), prob.end(), fdist(ii).begin(), 0.0f);
	    nent(ii)(aa) = std::inner_product(prob.begin(), prob.end(), nldist.begin(), 0.0f);;
	}
	vector<double> nvec(l_tilde(ii)-nent(ii));
	matrix<double> B = p_tilde(ii); 
	permutation_matrix<std::size_t> P(nvec.size());
	//std::cout<<"B.size="<<B.size1()<<","<<B.size2()<<std::endl;
	//std::cout<<"nvec.size="<<nvec.size()<<std::endl;
	//std::cout<<"B="<<std::endl;
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



void lmdp_t::solve(int goal, vector<double>& logz) const
{
    using namespace boost::lambda;
    double (*flog)(double) = std::log;

    int ng = ig2yx.size1();
    logz = scalar_vector<double>(ng, 0.0f);

    vector<int> sizes(sg.size());
    std::transform(sg.begin(), sg.end(), sizes.begin(), bind(&vector<int>::size, _1));
    int nnz = sum(sizes);

    umf_sparse_matrix qp(ng, ng, nnz);

    //std::cout<<"q="<<q<<std::endl;
    //std::cout<<"logpps="<<log_pps<<std::endl;

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

    splitter_t<> splitter(good);
    matrix<umf_sparse_matrix> qp_block = split_sparse_mat<matrix>(splitter, qp);

    vector<double> z2(qp_block(0, 1).size2());
    for(int ii=0, jj=0; ii<good.size(); ++ii)
    {
	if(good(ii))
	{
	    z2(jj) = 1.0l;
	    ++jj;
	}
    }

    vector<double> A12z2(qp_block(0, 1).size1());
    axpy_prod(qp_block(0, 1), z2, A12z2, true);
    umf_sparse_matrix A = -qp_block(0,0);
    for(int ii=0; ii<A.size1(); ++ii)
    {
	A(ii, ii) = 1.0l;
    }

    umf::symbolic_type<double> symb;
    umf::numeric_type<double> nume;
    umf::symbolic (A, symb); 
    umf::numeric (A, symb, nume); 

    vector<double> z1=scalar_vector<double>(A12z2.size(), 0.0l);
    umf::solve(A, z1, A12z2, nume);  
///

    vector<double> z(ng);

    array<int, 2> cc={0,0};
    for(int ii=0; ii<z.size(); ++ii)
    {
	if(good(ii)) 
	{
	    z(ii) = z2(cc[1]);
	    cc[1]++;
	}
	else 
	{
	    z(ii) = z1(cc[0]);
	    cc[0]++;
	}
    }

    logz = vector<double>(z.size());
    std::transform(z.begin(), z.end(), logz.begin(), flog);
    //std::cout<<"logz="<<logz<<std::endl;

    //std::cout<<"z="<<z<<std::endl;

}

#endif
