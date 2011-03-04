#ifndef __LP___IMPL__HPP__INCLUDED__
#define __LP___IMPL__HPP__INCLUDED__

float glpk_solve_links(vector<float> const& Affv, matrix<int> const& c1, matrix<int> const& c2,
		       vector<float>& Lv);

void solve_linprog(matrix<int> const& Tff, matrix<float> const& Aff,
		   matrix<int>& LMat, matrix<int>& links)
{
    using namespace boost::lambda;
    int ng = Tff.size1();
    int dim = std::count_if(Tff.data().begin(), Tff.data().end(), _1>0);
    vector<int> dd2ii(scalar_vector<int>(dim, -1));
    vector<int> dd2jj(scalar_vector<int>(dim, -1));
    matrix<int> ij2dd(scalar_matrix<int>(ng, ng, -1));

    vector<float> Affv(scalar_vector<float>(dim, 0));

    int dd = 0;

    for(int ii=0; ii<ng; ++ii)
    {
	for(int jj=0; jj<ng; ++jj)
	{
	    if(!Tff(ii, jj)) continue;
	    ij2dd(ii, jj) = dd;
	    dd2ii(dd) = ii;
	    dd2jj(dd) = jj;
	    Affv(dd) = Aff(ii, jj);
	    dd++;
	}
    }

    matrix<int> const1(scalar_matrix<int>(ng, dim, 0));
    std::vector<int> idx1;
    for(int ii=0; ii<ng; ++ii)
    {
	for(int jj=0; jj<ng; ++jj)
	{
	    int dd = ij2dd(ii, jj);
	    if(dd>=0) const1(ii, dd) = 1;
	}
	float v = sum(row(const1, ii));
	if(v>0) idx1.push_back(ii);
    }

    matrix<int> const2(scalar_matrix<int>(ng, dim, 0));
    std::vector<int> idx2;
    for(int ii=0; ii<ng; ++ii)
    {
	for(int jj=0; jj<ng; ++jj)
	{
	    int dd = ij2dd(jj, ii);
	    if(dd>=0) const2(ii, dd) = 1;
	}
	float v = sum(row(const2, ii));
	if(v>0) idx2.push_back(ii);
    }

    matrix<int> c1(idx1.size(), dim);
    for(int ii=0; ii<idx1.size(); ++ii)
    {
	row(c1, ii) = row(const1, idx1[ii]);
    }

    matrix<int> c2(idx2.size(), dim);
    for(int ii=0; ii<idx2.size(); ++ii)
    {
	row(c2, ii) = row(const2, idx2[ii]);
    }

    //matrix<float>

    vector<float> Lv;
    glpk_solve_links(Affv, c1, c2, Lv);

    std::cout<<"Lv="<<Lv<<std::endl;

    LMat = scalar_matrix<int>(ng, ng, 0);
    for(int dd=0; dd<dim; ++dd)
    {
	LMat(dd2ii(dd), dd2jj(dd)) = (Lv(dd)>0.5);
    }

    matrix<float> Aff2(Aff.size1(), Aff.size2());
    for(int ii=0; ii<Aff2.size1(); ++ii)
    {
	for(int jj=0; jj<Aff2.size2(); ++jj)
	{
	    if(LMat(ii, jj)) Aff2(ii, jj) = Aff(ii, jj);
	    else Aff2(ii, jj) = 0;
	}
    }

    std::vector<array<int, 2> > lv;
    int np = static_cast<int>(sum(Lv)+0.5f);
    for(int pp=0; pp<np; ++pp)
    {
	std::size_t ii, jj;
	array2d_max(Aff2, ii, jj);
	float vv = Aff2(ii, jj);
	if(vv>0)
	{
	    array<int, 2> tmp = {ii, jj};
	    lv.push_back(tmp);
	    matrix_row<matrix<float> > r(Aff2, ii);
	    std::for_each(r.begin(), r.end(), _1=0.0f);
	    matrix_column<matrix<float> > c(Aff2, jj);
	    std::for_each(c.begin(), c.end(), _1=0.0f);
	}
	else break;
    }

    links = scalar_matrix<int>(lv.size(), 2, 0);
    for(int ii=0; ii<links.size1(); ++ii)
    {
	for(int jj=0; jj<2; ++jj)
	{
	    links(ii, jj)=lv[ii][jj];
	}
    }

}

#include <glpk.h>

float glpk_solve_links(vector<float> const& Affv, matrix<int> const& c1, matrix<int> const& c2,
		      vector<float>& Lv)
{
    glp_prob *lp;
    int ia[1+2000], ja[1+2000];
    double ar[1+2000];//, z, x1, x2, x3;

    lp = glp_create_prob();

    glp_set_prob_name(lp, "links");

    glp_set_obj_dir(lp, GLP_MAX);

    glp_add_rows(lp, c1.size1()+c2.size2());

    for(int ii=0; ii<c1.size1(); ++ii)
    {
	std::string name = str(format("p%03d")%ii);
	glp_set_row_name(lp, ii+1, name.c_str());
	glp_set_row_bnds(lp, ii+1, GLP_DB, 0.0, 1.0);
    }
    for(int ii=0; ii<c2.size1(); ++ii)
    {
	std::string name = str(format("q%03d")%ii);
	glp_set_row_name(lp, c1.size1()+ii+1, name.c_str());
	glp_set_row_bnds(lp, c1.size1()+ii+1, GLP_DB, 0.0, 1.0);
    }

    glp_add_cols(lp, Affv.size());

    for(int ii=0; ii<Affv.size(); ++ii)
    {
	std::string name = str(format("x%03d")%ii);
	glp_set_col_name(lp, ii+1, name.c_str());
	glp_set_col_bnds(lp, ii+1, GLP_DB, 0.0, 1.0);
	glp_set_obj_coef(lp, ii+1, Affv(ii));
    }

    int ll=1;
    for(int ii=0; ii<c1.size1(); ++ii)
    {
	for(int jj=0; jj<c1.size2(); ++jj)
	{
	    if(!c1(ii, jj)) continue;
	    ia[ll] = ii+1;
	    ja[ll] = jj+1;
	    ar[ll] = 1.0f;
	    ll++;
	}
    }
    for(int ii=0; ii<c2.size1(); ++ii)
    {
	for(int jj=0; jj<c2.size2(); ++jj)
	{
	    if(!c2(ii, jj)) continue;
	    ia[ll] = c1.size1()+ii+1;
	    ja[ll] = jj+1;
	    ar[ll] = 1.0f;
	    ll++;
	}
    }

    glp_load_matrix(lp, ll-1, ia, ja, ar);

    glp_simplex(lp, NULL);
    float z = static_cast<float>(glp_get_obj_val(lp));

    Lv = vector<float>(Affv.size());

    for(int ii=0; ii<Lv.size(); ++ii)
    {
	Lv(ii) = static_cast<float>(glp_get_col_prim(lp, ii+1));
    }

    glp_delete_prob(lp);

    return z;
}


#endif
