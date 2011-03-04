#ifndef __STATISTICS__DETAIL__INCLUDED__
#define __STATISTICS__DETAIL__INCLUDED__

BEGIN_NAMESPACE_CVPR_DETAIL

using namespace boost::numeric::ublas;
using namespace boost::lambda;


template <class Float>
void column_mean_var(matrix<Float> const& data,
		     vector<Float>& mean, matrix<Float>& var) 
{

    int ndim = data.size1();
    int ndata = data.size2();

    mean = scalar_vector<Float>(ndim, 0);

    for(int ii=0; ii<ndata; ++ii)
    {
	mean += column(data, ii);
    }

    mean /= ndata;

    var  = scalar_matrix<float>(ndim, ndim, 0);
    for(int ii=0; ii<ndata; ++ii)
    {
	vector<Float>  dv = column(data, ii) - mean;
	noalias(var) += outer_prod(dv, dv);
    }
    var /= ndata;

}

template <class Float>
void row_mean_var(matrix<Float> const& data,
		  vector<Float>& mean, matrix<Float>& var) 
{

    int ndim = data.size2();
    int ndata = data.size1();

    mean = scalar_vector<Float>(ndim, 0);

    for(int ii=0; ii<ndata; ++ii)
    {
	mean += row(data, ii);
    }

    mean /= ndata;

    var  = scalar_matrix<float>(ndim, ndim, 0);
    for(int ii=0; ii<ndata; ++ii)
    {
	vector<Float>  dv = row(data, ii) - mean;
	noalias(var) += outer_prod(dv, dv);
    }
    var /= ndata;

}


//compute weighted mean and var of a set of data, with minor weights pruning
template <class Float>
void weighted_column_mean_var_prune(matrix<Float> const& data,
				    vector<Float> const& weight,
				    Float thr,
				    vector<Float>& mean, matrix<Float>& var) 
{

    int ndim = data.size1();
    int ndata = data.size2();

    mean = scalar_vector<Float>(ndim, 0);

    Float sum_w = 0;

    for(int ii=0; ii<ndata; ++ii)
    {
	if(weight(ii)<thr) continue;
	mean += weight(ii)*column(data, ii);
	sum_w += weight(ii);
    }

    mean /= sum_w;

    var  = scalar_matrix<float>(ndim, ndim, 0);
    for(int ii=0; ii<ndata; ++ii)
    {
	if(weight(ii)<thr) continue;
	vector<Float>  dv = column(data, ii) - mean;
	noalias(var) += outer_prod(dv, dv)*weight(ii);
    }
    var /= sum_w;

}

template <class Float>
void weighted_column_mean_var(matrix<Float> const& data,
			      vector<Float> const& weight,
			      vector<Float>& mean, matrix<Float>& var) 
{
    weighted_column_mean_var_prune(data, weight, -1.0f, mean, var);
}


// compute log likelihood of multivariate gaussian, using LU factorization
template <class Float>
void gaussian_loglike(vector<Float> const& mean, matrix<Float> const& var,
		      matrix<Float> const& data, vector<Float>& loglike) 
{

    using namespace boost::math::constants;

    int ndim = mean.size(); 
    int np = data.size2();

    matrix<Float> vd = data; //working copy of data
    for(int ii=0; ii<vd.size2(); ++ii)
    {
	column(vd, ii) -= mean;
    }
    matrix<Float> data0 = vd;

    matrix<Float> vv = var;  //working copy of var
    permutation_matrix<std::size_t> P(ndim);
    //std::cout<<"np="<<np<<", vv="<<vv<<std::endl;

    int res = lu_factorize(vv, P);  //vv holds L and U
    lu_substitute(vv, P, vd);       //vd holds the solution var_{-1}*data

    matrix_vector_range<matrix<Float> > diag(vv, range(0, ndim), range(0, ndim));
    Float logz = 0;
    for(int dd=0; dd<ndim; ++dd) {
	logz += std::log(std::abs(diag(dd)))/2 + std::log(2*pi<Float>())/2;
    }

    loglike = vector<Float>(np);
    for(int pp=0; pp<np; ++pp)  {
	loglike(pp) = -logz -inner_prod(column(vd, pp), column(data0, pp))/2;
    }

}

template <class Float>
void gaussian_loglike(gaussian_t<Float> const& g,
		      matrix<Float> const& data, vector<Float>& loglike) 
{

    detail::gaussian_loglike(g.mean, g.var, data, loglike);

}

// Robust computation of log sum_i exp(ml_i)
// It is useful for log-likelihood of mixture models
// robust computation with underflow prevention
template <class Float>
inline Float log_sum_exp(vector<Float> const& ml)
{
    Float maxml = *(std::max_element(ml.begin(), ml.end()));
    Float temp=0;
    for(int mm=0; mm<ml.size(); ++mm)    {
	Float v = ml(mm)-maxml;
	if(v<-12.0f)  continue; // exp(-12) is too small compare to 1.0
	temp += std::exp(v);
    }
    return maxml+std::log(temp);
}

//compute posterio from log-likelihood array
// robust computation with underflow prevention
template <class V1, class Float>
void posterior_from_loglike(V1 const& loglike,
			   vector<Float> const& logpis,
			   vector<Float>& post)
{
    int K = logpis.size();
    post = vector<Float>(K);

    for(int kk=0; kk<K; ++kk)   {
	post(kk) = logpis(kk)+loglike(kk);
    }
    Float maxv = *(std::max_element(post.begin(), post.end()));
    for(int kk=0; kk<K; ++kk)   {
	post(kk) -= maxv;
	post(kk) = std::exp(post(kk));
    }
    post /= sum(post);
}

// compute log likelihood of mixtures of multivariate gaussian
template <class Float>
void gaussian_mixture_loglike(gaussian_mixture_t<Float> const& mix,
			      matrix<Float> const& data,
			      vector<Float>& loglike)
{

    int K = mix.K();
    int np = data.size2();
    loglike = vector<Float>(np);
    vector<Float> logpi(K);
    matrix<Float> modlike(K, np);
    for(int mm=0; mm<K; ++mm)
    {
	vector<Float> ll;
	detail::gaussian_loglike(mix.items(mm), data, ll);
	row(modlike, mm) = ll;
	logpi(mm) = std::log(mix.pis(mm));
    }

    for(int pp=0; pp<np; ++pp)
    {
	vector<Float> ml = column(modlike, pp)+logpi;
	loglike(pp) = log_sum_exp(ml);
    }
}

//EM algorithm for mixture of multivariate gaussian
template <class Float>
void EM_gaussian_mixture(matrix<Float> const& data,
			 int K,  gaussian_mixture_t<Float>& mix,
			 EM_plain_opt<Float> const& opt) 
{

    Float minvar = opt.minvar;
    int max_it = 12*K; //20*K
    int np = data.size2();
    int ndim = data.size1();
    mix = gaussian_mixture_t<Float>(K);
    //means = vector<vector<Float> >(K);
    //vars = vector<matrix<Float> >(K);
    //pis = vector<Float>(K);

    matrix<Float> mem = scalar_matrix<Float>(K, np, 0);
    // initialize membership
    for(int kk=0; kk<K; ++kk)
    {
	int a1= np*kk/K, a2= np*(kk+1)/K;
	project(mem, range(kk, kk+1), range(a1, a2)) = scalar_matrix<Float>(1, a2-a1, 1);
    }

    for(int it=0; it<max_it; ++it)
    {
	// M step
	//real_timer_t timer_m;
	for(int kk=0; kk<K; ++kk)
	{
	    vector<Float> w = row(mem, kk);
	    weighted_column_mean_var_prune(data, w, 1e-5f,
					   mix.items(kk).mean, mix.items(kk).var);
	    for(int dd=0; dd<ndim; ++dd)  mix.items(kk).var(dd, dd) += minvar;
	}

	// E step
	//noalias(pis) = prod(mem, fgconf);
	//for(int kk=0; kk<K; ++kk)	{
	//    pis(kk) = sum(row(mem, kk));
	//}
	row_sum(mem, mix.pis);
	mix.pis /= sum(mix.pis);

	matrix<Float> loglike(K, np);
	for(int kk=0; kk<K; ++kk)
	{
	    vector<Float> ll;
	    detail::gaussian_loglike(mix.items(kk), data, ll);
	    row(loglike, kk) = ll;
	}

	//    Recompute membership
	Float dmem = 0;
	vector<Float> logpis(K);

	std::transform(mix.pis.begin(), mix.pis.end(), logpis.begin(), (Float (*)(Float)) std::log);

	for(int pp=0; pp<np; ++pp)
	{
	    vector<Float> newmem;
	    posterior_from_loglike(column(loglike, pp), logpis, newmem);

	    vector<Float> dm = column(mem, pp) - newmem;
	    dmem += inner_prod(dm, dm);
	    column(mem, pp) = newmem;
	}

	//  Check membership changes over time
	dmem = std::sqrt(dmem);
	if(dmem<EM_DMEM_THR*np/100.0) 
	{
	    std::cout<<"\t\tit="<<it<<", \tdmem="<<dmem<<std::endl;
	    break;
	}

    }

}

//EM algorithm for mixture of multivariate gaussian
//with subsampling of training examples
template <class Float>
void EM_gaussian_mixture(matrix<Float> const& datao,
			 int K,  gaussian_mixture_t<Float>& mix,
			 EM_subsamp_opt<Float> const& opt) 
{

    int max_it = 12*K; //16*K; //20*K;
    int ndim = datao.size1();
    int npo = datao.size2();
    int stride = opt.stride;
    Float minvar = opt.minvar;

    //std::cout<<"npo="<<npo<<std::endl;
    matrix<Float> memo = scalar_matrix<Float>(K, npo, 0);
 
    if(opt.use_prev_model)
    {

	{
	    vector<Float> logpis(K);
	    std::transform(mix.pis.begin(), mix.pis.end(), logpis.begin(), (Float (*)(Float)) std::log);
	
	    matrix<Float> loglike(K, npo);
	    for(int kk=0; kk<K; ++kk)
	    {
		vector<Float> ll;
		detail::gaussian_loglike(mix.items(kk), datao, ll);
		row(loglike, kk) = ll;
	    }

	    for(int pp=0; pp<npo; ++pp)
	    {
		vector<Float> pmem;
		detail::posterior_from_loglike(column(loglike, pp), logpis, pmem);

		column(memo, pp) = pmem;
	    }
	}

    }
    else 
    {
	mix = gaussian_mixture_t<Float>(K);

	// initialize membership
	for(int kk=0; kk<K; ++kk)
	{
	    int a1= npo*kk/K, a2= npo*(kk+1)/K;	    
	    project(memo, range(kk, kk+1), range(a1, a2)) = scalar_matrix<Float>(1, a2-a1, 1);
	}
    }

    Float dmem;

    //Update a slice at each iteration
    for(int it=0; it<max_it; ++it)
    {
	slice current_slice(it%stride, stride, (npo-it%stride-1)/stride+1);
	slice all_dim(0, 1, ndim);
	slice all_mix(0, 1, K);

	matrix<Float> data = project(datao, all_dim, current_slice);
	matrix_slice<matrix<Float> > mem(memo, all_mix, current_slice);

	int np = data.size2();

	//M step

	for(int kk=0; kk<K; ++kk)
	{
	    vector<Float> w = row(mem, kk);
	    weighted_column_mean_var_prune(data, w, 1e-5f,
					   mix.items(kk).mean, mix.items(kk).var);
	    for(int dd=0; dd<ndim; ++dd)  mix.items(kk).var(dd, dd) += minvar;
	}

	//E step
	//real_timer_t timer_e;
	//noalias(pis) = prod(mem, fgconf);
	//for(int kk=0; kk<K; ++kk) {
	//    pis(kk) = sum(row(mem, kk));
	//}
	row_sum(mem, mix.pis);
	mix.pis /= sum(mix.pis);

	matrix<Float> loglike(K, np);
	for(int kk=0; kk<K; ++kk)
	{
	    vector<Float> ll;
	    detail::gaussian_loglike(mix.items(kk), data, ll);
	    row(loglike, kk) = ll;
	}

	//    Recompute membership
	dmem = 0;
	vector<Float> logpis(K);
	std::transform(mix.pis.begin(), mix.pis.end(), logpis.begin(), (Float (*)(Float)) std::log);

	for(int pp=0; pp<np; ++pp)
	{
	    vector<Float> newmem;
	    detail::posterior_from_loglike(column(loglike, pp), logpis, newmem);

	    vector<Float> dm = column(mem, pp) - newmem;
	    dmem += inner_prod(dm, dm);
	    column(mem, pp) = newmem;
	}

	//  Check membership changes over time
	dmem = std::sqrt(dmem);
	if(dmem<EM_DMEM_THR*np/100.0) 
	{
	    //std::cout<<"\t\tit="<<it<<", \tdmem="<<dmem<<std::endl;
	    break;
	}
    }
    //std::cout<<"\t\t\t\tdmem_rate="<<dmem/npo*stride<<std::endl;

}

END_NAMESPACE_CVPR_DETAIL

#endif
