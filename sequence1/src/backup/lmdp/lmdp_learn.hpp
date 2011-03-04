#ifndef __LMDP__LEARN__HPP__INCLUDED__
#define __LMDP__LEARN__HPP__INCLUDED__



struct lmdp_1f_t
{
    vector<double> q;
    vector<vector<double> > log_pps;

    vector<matrix<double> > log_p_tilde;
    vector<vector<double> > l_tilde;
    vector<vector<double> > nent;
    lmdp_1f_t() {}
    explicit lmdp_1f_t(int ng):
	q(ng), log_pps(ng), log_p_tilde(ng),
	l_tilde(ng), nent(ng) {}

};


struct grad_state_t
{
    vector<double> grad_q;       //num_feat
    vector<vector<double> > grad_log_pps; //num_feat, num_action

    vector<matrix<double> > grad_log_p_tilde; //num_feat, num_neighbor, num_action
    vector<vector<double> > grad_l_tilde, grad_nent; //num_feat, num_action
    grad_state_t(){}
    explicit grad_state_t(int nf):
	grad_q(nf), grad_log_pps(nf), grad_log_p_tilde(nf),
	grad_l_tilde(nf), grad_nent(nf) {}
};

struct hess_state_t
{
    matrix<double> hess_q;       //num_feat, num_feat
    matrix<vector<double> > hess_log_pps; //num_feat, num_feat, num_action
    matrix<matrix<double> > hess_log_p_tilde; //num_feat, num_feat, num_neighbor, num_action
    matrix<vector<double> > hess_l_tilde, hess_nent; //num_feat, num_feat, num_action
    hess_state_t(){}
    explicit hess_state_t(int nf):
	hess_q(nf, nf), hess_log_pps(nf, nf), hess_log_p_tilde(nf, nf),
	hess_l_tilde(nf, nf), hess_nent(nf, nf)
	{}
};

void compute_grad_plh(lmdp_t const& lmdp, int ig, grad_state_t& grad_state);

void compute_grad_q_pps(lmdp_t const& lmdp, int ig,
			grad_state_t& grad_state);

void compute_hess_plh(lmdp_t const& lmdp, int ig,
		      grad_state_t const& grad_state,
		      hess_state_t & hess_state);

void compute_hess_q_pps(lmdp_t const& lmdp, int ig,
			grad_state_t const& grad_state,
			hess_state_t& hess_state);

void compute_grad_lmdp(lmdp_t const& lmdp,
		       vector<lmdp_1f_t>& grad_lmdp);

void compute_hess_lmdp(lmdp_t const& lmdp,
		       vector<lmdp_1f_t> const& grad_lmdp,
		       matrix<lmdp_1f_t>& hess_lmdp);

void compute_grad_hess_logz(lmdp_t const& lmdp,
			    vector<lmdp_1f_t> const& grad_lmdp,
			    matrix<lmdp_1f_t> const& hess_lmdp,
			    vector<vector<int> >const& path_ig,
			    vector<vector<double> >const & logz,
			    vector<vector<vector<double> > >& grad_logz,
			    matrix<vector<vector<double> > >& hess_logz);

void compute_grad_hess_L(lmdp_t const& lmdp,
			 vector<lmdp_1f_t> const& grad_lmdp,
			 matrix<lmdp_1f_t> const& hess_lmdp,
			 vector<vector<int> >const& path_ig,
			 vector<vector<vector<double> > > const& grad_logz,
			 matrix<vector<vector<double> > > const& hess_logz,
			 vector<double>& grad_L,
			 matrix<double>& hess_L);

void learn_weights(lmdp_t& lmdp,
		   vector<vector<int> > const& path_ig,
		   vector<double>& wei);

void learn_weights_greedy(lmdp_t& lmdp,
			  vector<vector<int> > const& path_ig,
			  vector<double>& wei);


#endif

