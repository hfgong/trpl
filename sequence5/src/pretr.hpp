#ifndef __BBSEG_JOINT__HPP__INCLUDED__
#define __BBSEG_HOINT__HPP__INCLUDED__

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/math/constants/constants.hpp>

#include <boost/array.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

#include <algorithm>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "labelmap.hpp"
#include "misc_utils.hpp"
#include "statistics.hpp"

//define for better formatting in editor
#define BEGIN_NAMESPACE_BGSEG namespace bgseg {

#define END_NAMESPACE_BGSEG };


BEGIN_NAMESPACE_BGSEG

using namespace boost::numeric::ublas;
using namespace boost::lambda;
namespace umf=boost::numeric::bindings::umfpack;

using namespace cvpr;

using boost::array;

//sparse matrix used by UMFPACK, double-only
typedef sparse_matrix_t<double>::type umf_sparse_matrix;




END_NAMESPACE_BGSEG

#endif
