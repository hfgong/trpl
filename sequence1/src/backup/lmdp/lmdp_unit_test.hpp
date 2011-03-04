#ifndef __LMDP__UNIT__TEST__HPP__INCLUDED__
#define __LMDP__UNIT__TEST__HPP__INCLUDED__

#include <boost/tuple/tuple.hpp>

void compute_grad_lmdp_numeric(lmdp_t const& lmdp,
			       vector<lmdp_1f_t>& grad_lmdp)
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
	grad_lmdp(ff).log_pps = (lmdp2.log_pps-lmdp.log_pps);
	grad_lmdp(ff).log_pps /= dw;

	grad_lmdp(ff).log_p_tilde = lmdp2.log_p_tilde-lmdp.log_p_tilde;
	grad_lmdp(ff).log_p_tilde /= dw;

	grad_lmdp(ff).l_tilde = lmdp2.l_tilde-lmdp.l_tilde;
	grad_lmdp(ff).l_tilde /= dw;

 	grad_lmdp(ff).nent = lmdp2.nent-lmdp.nent;
	grad_lmdp(ff).nent /= dw;
    }

}

void compute_hess_lmdp_numeric(lmdp_t const& lmdp,
			       matrix<lmdp_1f_t>& hess_lmdp)
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
	    hess_lmdp(ff, ee).log_pps /= dw;

	    hess_lmdp(ff, ee).log_p_tilde = grad_lmdp2(ee).log_p_tilde-grad_lmdp(ee).log_p_tilde;
	    hess_lmdp(ff, ee).log_p_tilde /= dw;

	    hess_lmdp(ff, ee).l_tilde = grad_lmdp2(ee).l_tilde-grad_lmdp(ee).l_tilde;
	    hess_lmdp(ff, ee).l_tilde /= dw;

	    hess_lmdp(ff, ee).nent = grad_lmdp2(ee).nent-grad_lmdp(ee).nent;
	    hess_lmdp(ff, ee).nent /= dw;

	}
    }
}

void compute_grad_logz_numeric(lmdp_t const& lmdp,
			       vector<vector<int> > const& path_ig,
			       vector<vector<vector<double> > > & grad_logz)
{
    using namespace boost::lambda;
    int nf = lmdp.wei.size();
    double dw = 0.01l;

    vector<vector<double> > logz(path_ig.size());
    for(int cc=0; cc<path_ig.size(); ++cc)
    {
	lmdp.solve(*(path_ig(cc).rbegin()), logz(cc));
    }

    grad_logz = vector<vector<vector<double> > >(nf);
    for(int ff=0; ff<nf; ++ff)
    {
	lmdp_t lmdp2;
	lmdp2.initialize(lmdp.fg, lmdp.sg, lmdp.yx2ig, lmdp.ig2yx);

	vector<double> wei2 = lmdp.wei;
	wei2(ff) += dw;

	lmdp2.embed(wei2);

	vector<vector<double> > logz2(path_ig.size());
	for(int cc=0; cc<path_ig.size(); ++cc)
	{
	    lmdp2.solve(*(path_ig(cc).rbegin()), logz2(cc));
	}
	grad_logz(ff) = logz2 - logz;
	grad_logz(ff) /= dw;
    }


}

void compute_hess_logz_numeric(lmdp_t const& lmdp,
			       vector<vector<int> > const& path_ig,			       
			       matrix<vector<vector<double> > > & hess_logz)
{
    using namespace boost::lambda;
    int nf = lmdp.wei.size();
    double dw = 0.01l;

    vector<vector<vector<double> > >  grad_logz;
    compute_grad_logz_numeric(lmdp, path_ig, grad_logz);

    hess_logz = matrix<vector<vector<double> > >(nf, nf);
    for(int ff=0; ff<nf; ++ff)
    {
	lmdp_t lmdp2;
	lmdp2.initialize(lmdp.fg, lmdp.sg, lmdp.yx2ig, lmdp.ig2yx);

	vector<double> wei2 = lmdp.wei;
	wei2(ff) += dw;

	lmdp2.embed(wei2);

	vector<vector<vector<double> > >  grad_logz2;
	compute_grad_logz_numeric(lmdp2, path_ig, grad_logz2);

	for(int ee=0; ee<nf; ++ee)
	{
	    hess_logz(ff, ee) = grad_logz2(ee)-grad_logz(ee);
	    hess_logz(ff, ee) /= dw;
	}
    }

}


void learn_weights_numeric(lmdp_t& lmdp,
			   vector<vector<int> > const& path_ig,
			   vector<double>& wei)
{

    for(int it=0; it<1; ++it)
    {
	std::cout<<"it="<<it<<"-----------------------------"<<std::endl;
	lmdp.embed(wei);

	vector<lmdp_1f_t> grad_lmdp2;
	compute_grad_lmdp_numeric(lmdp, grad_lmdp2);

	matrix<lmdp_1f_t> hess_lmdp2;
	compute_hess_lmdp_numeric(lmdp, hess_lmdp2);

	vector<vector<double> > logz(path_ig.size());
	for(int cc=0; cc<path_ig.size(); ++cc)
	{
	    lmdp.solve(*(path_ig(cc).rbegin()), logz(cc));
	}

	vector<vector<vector<double> > >  grad_logz2;
	matrix<vector<vector<double> > >  hess_logz2;
	compute_grad_logz_numeric(lmdp, path_ig, grad_logz2);
	compute_hess_logz_numeric(lmdp,  path_ig, hess_logz2);
			       
	vector<double> grad_L;
	matrix<double> hess_L;
	compute_grad_hess_L(lmdp, grad_lmdp2, hess_lmdp2,
			    path_ig, grad_logz2, hess_logz2, grad_L, hess_L);

	permutation_matrix<std::size_t> P(hess_L.size1());

	int res = lu_factorize(hess_L, P);  //hess_L holds L and U
	lu_substitute(hess_L, P, grad_L); 


	wei += -grad_L;
	std::cout<<"wei="<<std::endl;
	array1d_print(std::cout, wei);

    }

}

template <class DType>
tuple<DType, DType, DType> 
analyze_consistency_one(vector<DType> const& ditem,
			vector<DType> const& aitem,
			bool verbose)
{
    using namespace boost::lambda;
    DType (*fabs)(DType) = std::abs;
    DType max_d = *std::max_element(ditem.begin(),
				    ditem.end());
    DType min_d = *std::min_element(ditem.begin(),
				    ditem.end());
    vector<DType> ratio(ditem.size());
    std::transform(ditem.begin(),
		   ditem.end(),
		   aitem.begin(),
		   ratio.begin(),
		   bind(fabs, _1/(_2+1e-10l)));
    DType max_r = *std::max_element(ratio.begin(), ratio.end());
    if(verbose)
	std::cout<<str(format("\t\tmax_d=%f\tmin_d=%f, \tmax_r=%f")
		       %max_d%min_d%max_r)<<std::endl;

    return make_tuple(max_d, min_d, max_r);
}

template <class DType>
tuple<DType, DType, DType>
analyze_consistency_one(vector<matrix<DType> > const& ditem,
			vector<matrix<DType> > const& aitem,
			bool verbose)
{
    using namespace boost::lambda;
    DType (*fabs)(DType) = std::abs;
    vector<DType> max_dv(ditem.size());
    vector<DType> min_dv(ditem.size());
    vector<DType> max_rv(ditem.size());
    for(int cc=0; cc<ditem.size(); ++cc)
    {
	max_dv(cc) = *std::max_element(ditem(cc).data().begin(),
				       ditem(cc).data().end());
	min_dv(cc) = *std::min_element(ditem(cc).data().begin(),
				       ditem(cc).data().end());
	matrix<DType> ratio(ditem(cc).size1(), ditem(cc).size2());
	std::transform(ditem(cc).data().begin(),
		       ditem(cc).data().end(),
		       aitem(cc).data().begin(),
		       ratio.data().begin(),
		       bind(fabs, _1/(_2+1e-10l)));
	max_rv(cc) = *std::max_element(ratio.data().begin(),
					ratio.data().end());

	if(verbose)
	    std::cout<<str(format("\t\tmax_d=%f\tmin_d=%f, \tmax_r=%f")
			   %max_dv(cc)%min_dv(cc)%max_rv(cc))<<std::endl;
    }
    DType max_d = *std::max_element(max_dv.begin(), max_dv.end());
    DType min_d = *std::min_element(min_dv.begin(), min_dv.end());
    DType max_r = *std::max_element(max_rv.begin(), max_rv.end());

    return make_tuple(max_d, min_d, max_r);

}


template <class DType>
tuple<DType, DType, DType>
analyze_consistency_one(vector<vector<DType> > const& ditem,
			vector<vector<DType> > const& aitem,
			bool verbose)
{
    using namespace boost::lambda;
    DType (*fabs)(DType) = std::abs;
    vector<DType> max_dv(ditem.size());
    vector<DType> min_dv(ditem.size());
    vector<DType> max_rv(ditem.size());
    for(int cc=0; cc<ditem.size(); ++cc)
    {
	max_dv(cc) = *std::max_element(ditem(cc).begin(),
					ditem(cc).end());
	min_dv(cc) = *std::min_element(ditem(cc).begin(),
					 ditem(cc).end());
	vector<DType> ratio(ditem(cc).size());
	std::transform(ditem(cc).begin(),
		       ditem(cc).end(),
		       aitem(cc).begin(),
		       ratio.begin(),
		       bind(fabs, _1/(_2+1e-10l)));
	max_rv(cc) = *std::max_element(ratio.begin(), ratio.end());
	if(verbose)
	    std::cout<<str(format("\t\tmax_d=%f\tmin_d=%f, \tmax_r=%f")
			   %max_dv(cc)%min_dv(cc)%max_rv(cc))<<std::endl;
    }
    DType max_d = *std::max_element(max_dv.begin(), max_dv.end());
    DType min_d = *std::min_element(min_dv.begin(), min_dv.end());
    DType max_r = *std::max_element(max_rv.begin(), max_rv.end());

    return make_tuple(max_d, min_d, max_r);
}


template <class DType>
void analyze_consistency(vector<DType> const& data1,
			 vector<DType> const& data2,
			 bool verbose = false)
{
    vector<DType> ddata = data1-data2;
    vector<DType> adata = data1+data2;

    for(int ff=0; ff<data1.size(); ++ff)
    {

	DType& ditem = ddata(ff);
	DType& aitem = adata(ff);
	typename leaf_type_t<DType>::type max_d, min_d, max_r;
	tie(max_d, min_d, max_r) =
	    analyze_consistency_one(ditem, aitem, verbose);
	std::cout<<"\tff="<<ff
		 <<str(format("\t\tmax_d=%f\tmin_d=%f, \tmax_r=%f")
		       %max_d%min_d%max_r)<<std::endl;

    }

}

template <class DType>
void analyze_consistency(matrix<DType> const& data1,
			 matrix<DType> const& data2,
			 bool verbose = false)
{
    matrix<DType> ddata = data1-data2;
    matrix<DType> adata = data1+data2;

    for(int ff=0; ff<data1.size1(); ++ff)
    {
	for(int ee=0; ee<data1.size2(); ++ee)
	{
	    DType& ditem = ddata(ff, ee);
	    DType& aitem = adata(ff, ee);
	    typename leaf_type_t<DType>::type max_d, min_d, max_r;
	    tie(max_d, min_d, max_r) =
		analyze_consistency_one(ditem, aitem, verbose);
	    std::cout<<"\tff="<<ff<<",\tee="<<ee
		     <<str(format("\t\tmax_d=%f\tmin_d=%f, \tmax_r=%f")
			   %max_d%min_d%max_r)<<std::endl;

	}
    }

}

void unit_test(lmdp_t& lmdp,
	       vector<vector<int> > const& path_ig,
 	       vector<double>& wei)
{
    using namespace boost::lambda;
    lmdp.embed(wei);
    int nf = lmdp.fg.size1();
    int ng = lmdp.sg.size();

    vector<lmdp_1f_t> grad_lmdp;
    matrix<lmdp_1f_t> hess_lmdp;
    compute_grad_lmdp(lmdp, grad_lmdp);
    compute_hess_lmdp(lmdp, grad_lmdp, hess_lmdp);

    vector<lmdp_1f_t> grad_lmdp2;
    matrix<lmdp_1f_t> hess_lmdp2;
    compute_grad_lmdp_numeric(lmdp, grad_lmdp2);
    compute_hess_lmdp_numeric(lmdp, hess_lmdp2);

    std::cout<<"grad_log_p_tilde check"<<std::endl;
    vector<vector<matrix<double> > > grad_log_ptld(nf), grad_log_ptld2(nf);
    array1d_transform(grad_lmdp, grad_log_ptld,
		      bind(&lmdp_1f_t::log_p_tilde, _1));
    array1d_transform(grad_lmdp2, grad_log_ptld2,
		      bind(&lmdp_1f_t::log_p_tilde, _1));
    analyze_consistency(grad_log_ptld, grad_log_ptld2);

    std::cout<<"hess_log_p_tilde check"<<std::endl;
    matrix<vector<matrix<double> > > hess_log_ptld(nf, nf),
	hess_log_ptld2(nf, nf);
    array2d_transform(hess_lmdp, hess_log_ptld,
		      bind(&lmdp_1f_t::log_p_tilde, _1));
    array2d_transform(hess_lmdp2, hess_log_ptld2,
		      bind(&lmdp_1f_t::log_p_tilde, _1));
    analyze_consistency(hess_log_ptld, hess_log_ptld2);


    std::cout<<"grad_l_tilde check"<<std::endl;
    vector<vector<vector<double> > > grad_lt(nf), grad_lt2(nf);
    array1d_transform(grad_lmdp, grad_lt,
		      bind(&lmdp_1f_t::l_tilde, _1));
    array1d_transform(grad_lmdp2, grad_lt2,
		      bind(&lmdp_1f_t::l_tilde, _1));
    analyze_consistency(grad_lt, grad_lt2);

    std::cout<<"hess_l_tilde check"<<std::endl;
    matrix<vector<vector<double> > > hess_lt(nf, nf), hess_lt2(nf, nf);
    array2d_transform(hess_lmdp, hess_lt,
		      bind(&lmdp_1f_t::l_tilde, _1));
    array2d_transform(hess_lmdp2, hess_lt2,
		      bind(&lmdp_1f_t::l_tilde, _1));
    analyze_consistency(hess_lt, hess_lt2);


    std::cout<<"grad_nent check"<<std::endl;
    vector<vector<vector<double> > > grad_nent(nf), grad_nent2(nf);
    array1d_transform(grad_lmdp, grad_nent,
		      bind(&lmdp_1f_t::nent, _1));
    array1d_transform(grad_lmdp2, grad_nent2,
		      bind(&lmdp_1f_t::nent, _1));
    analyze_consistency(grad_nent, grad_nent2);

    std::cout<<"hess_nent check"<<std::endl;
    matrix<vector<vector<double> > > hess_nent(nf, nf), hess_nent2(nf, nf);
    array2d_transform(hess_lmdp, hess_nent,
		      bind(&lmdp_1f_t::nent, _1));
    array2d_transform(hess_lmdp2, hess_nent2,
		      bind(&lmdp_1f_t::nent, _1));
    analyze_consistency(hess_nent, hess_nent2);


    std::cout<<"grad_q check"<<std::endl;
    vector<vector<double> > grad_q(nf), grad_q2(nf);
    array1d_transform(grad_lmdp, grad_q, bind(&lmdp_1f_t::q, _1));
    array1d_transform(grad_lmdp2, grad_q2, bind(&lmdp_1f_t::q, _1));
    analyze_consistency(grad_q, grad_q2);

    std::cout<<"hess_q check"<<std::endl;
    matrix<vector<double> > hess_q(nf, nf), hess_q2(nf, nf);
    array2d_transform(hess_lmdp, hess_q, bind(&lmdp_1f_t::q, _1));
    array2d_transform(hess_lmdp2, hess_q2, bind(&lmdp_1f_t::q, _1));
    analyze_consistency(hess_q, hess_q2);



    std::cout<<"grad_log_pps check"<<std::endl;
    vector<vector<vector<double> > > grad_log_pps(nf), grad_log_pps2(nf);
    array1d_transform(grad_lmdp, grad_log_pps,
		      bind(&lmdp_1f_t::log_pps, _1));
    array1d_transform(grad_lmdp2, grad_log_pps2,
		      bind(&lmdp_1f_t::log_pps, _1));
    analyze_consistency(grad_log_pps, grad_log_pps2);


    std::cout<<"hess_log_pps check"<<std::endl;
    matrix<vector<vector<double> > > hess_log_pps(nf, nf),
	hess_log_pps2(nf, nf);
    array2d_transform(hess_lmdp, hess_log_pps,
		      bind(&lmdp_1f_t::log_pps, _1));
    array2d_transform(hess_lmdp2, hess_log_pps2,
		      bind(&lmdp_1f_t::log_pps, _1));
    analyze_consistency(hess_log_pps, hess_log_pps2);

#if 0
    {
	lmdp_t lmdp2(lmdp);
	lmdp2.wei(0) += 0.01l;
	lmdp_t lmdp3(lmdp);
	lmdp3.wei(0) += 0.02l;

	lmdp2.embed(lmdp2.wei);
	lmdp3.embed(lmdp3.wei);

	std::cout<<lmdp.p_tilde(1)<<std::endl;
	std::cout<<lmdp.p_tilde(1)-lmdp2.p_tilde(1)<<std::endl;
	std::cout<<lmdp2.p_tilde(1)-lmdp3.p_tilde(1)<<std::endl;


	std::cout<<">>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;
	lmdp_t* ls[3]={&lmdp, &lmdp2, &lmdp3};
	vector<vector<double> > lt(3);
	for(int kk=0; kk<3; ++kk)
	{
	    int ig = 1;
	    lmdp_t& l = *ls[kk];
	    vector<double> dist=scalar_vector<double>(l.sg(ig).size(), 0.0l);
	    for(int ff=0; ff< l.fg.size1(); ++ff)
	    {
		for(int nn=0; nn<l.sg(ig).size(); ++nn)
		{
		    dist(nn) += l.fg(ff, 1)(nn)*l.wei(ff);
		}
	    }
	    //std::cout<<"dist="<<dist<<std::endl;
	    lt(kk) = prod(l.p_tilde(ig), dist);
	    std::cout<<"p*dist="<<lt(kk)<<std::endl;
	    std::cout<<"dist="<<dist<<std::endl;
	}
	std::cout<<"hess lt="<<lt(0)+lt(2)-2*lt(1)<<std::endl;

	std::cout<<">>>>>>>>>>>>>>>>>>>>>>>"<<std::endl;
	std::cout<<lmdp.l_tilde(1)<<std::endl;
	std::cout<<lmdp2.l_tilde(1)<<std::endl;
	std::cout<<lmdp3.l_tilde(1)<<std::endl;
	std::cout<<(lmdp.l_tilde(1)+lmdp3.l_tilde(1)-lmdp2.l_tilde(1)*2)
	    /0.01l/0.01l<<std::endl;
    }
#endif
    vector<vector<double> > logz(path_ig.size());
    for(int cc=0; cc<path_ig.size(); ++cc)
    {
	lmdp.solve(*(path_ig(cc).rbegin()), logz(cc));
    }

    vector<vector<vector<double> > > grad_logz;
    matrix<vector<vector<double> > > hess_logz;

    compute_grad_hess_logz(lmdp, grad_lmdp, hess_lmdp,
			   path_ig, logz, grad_logz, hess_logz);

    vector<vector<vector<double> > >  grad_logz2;
    matrix<vector<vector<double> > >  hess_logz2;
    compute_grad_logz_numeric(lmdp, path_ig, grad_logz2);
    compute_hess_logz_numeric(lmdp, path_ig, hess_logz2);

    std::cout<<"grad_logz check"<<std::endl;
    analyze_consistency(grad_logz, grad_logz2);

    std::cout<<"hess_logz check"<<std::endl;
    analyze_consistency(hess_logz, hess_logz2);

    {
	//numerical stability check
	double dw = 0.07l;
	lmdp_t lmdp2(lmdp);
	lmdp2.wei(0) += dw;
	lmdp_t lmdp3(lmdp);
	lmdp3.wei(0) += dw;

	lmdp2.embed(lmdp2.wei);
	lmdp3.embed(lmdp3.wei);

	int goal = *(path_ig(0).rbegin());
	vector<vector<double> > lz(3);

	lmdp.solve(goal, lz(0));
	lmdp2.solve(goal, lz(1));
	lmdp3.solve(goal, lz(2));

	vector<double> hz00 = (lz(0)+lz(2)-2*lz(1))/dw/dw;

	for(int ii=0; ii<hz00.size(); ++ii)
	{
	    double v1 = hz00(ii);
	    double v2 = hess_logz(0, 0)(0)(ii);
	    double v3 = hess_logz2(0, 0)(0)(ii);
	    std::cout<<"$$\t"<<v1<<"\t"<<v2<<"\t"<<v3<<std::endl;
	}
	std::cout<<lz(0)<<std::endl;
    }

}


#endif

