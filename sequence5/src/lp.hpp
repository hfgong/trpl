#ifndef __LP__HPP__INCLUDED__
#define __LP__HPP__INCLUDED__

#include <boost/numeric/ublas/matrix.hpp>

void solve_linprog(matrix<int> const& Tff, matrix<float> const& Aff,
		   matrix<int>& LMat, matrix<int>& links);

#endif
