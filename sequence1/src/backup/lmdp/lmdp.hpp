#ifndef __LMDP__HPP__INCLUDED__
#define __LMDP__HPP__INCLUDED__

struct lmdp_t
{
    matrix<vector<double> >  fg;
    vector<vector<int> > sg;
    matrix<int> yx2ig;
    matrix<int>  ig2yx;
    vector<double> wei;
    vector<matrix<double> > p_tilde;
    vector<matrix<double> > log_p_tilde;
    vector<vector<double> > l_tilde;
    vector<vector<double> > nent;
    vector<vector<double> > log_pps;
    vector<double> q;

    void initialize(matrix<vector<double> > const& fg_,
		    vector<vector<int> > const& sg_,
		    matrix<int> const& yx2ig_,
		    matrix<int> const& ig2yx_);
    void embed(vector<double> const& wei_);
    void solve(int goal, vector<double>& logz) const;
    void solve(vector<vector<int> > const& path_ig,
	       vector<vector<double> >& logz) const;


    inline int count_sg_nzz() const	{
	using namespace boost::lambda;
	vector<int> sizes(sg.size());
	std::transform(sg.begin(), sg.end(), sizes.begin(), bind(&vector<int>::size, _1));
	return sum(sizes);

    }
};


#endif
