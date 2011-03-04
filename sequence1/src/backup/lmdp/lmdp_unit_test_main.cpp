#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <limits>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>

#include <boost/numeric/ublas/io.hpp>

#include <CImg.h>

#include "statistics.hpp"
#include "cvpr_array_traits.hpp"
#include "real_timer.hpp"
#include "text_file.hpp"
#include "misc_utils.hpp"

namespace umf=boost::numeric::bindings::umfpack;
using namespace boost::numeric::ublas;
using namespace boost;
using namespace cvpr;
using namespace cimg_library;

namespace fs = boost::filesystem;


typedef cvpr::sparse_matrix_t<double>::type umf_sparse_matrix;

#include "lmdp.hpp"
#include "lmdp_learn.hpp"
#include "lmdp_unit_test.hpp"
#include "training_data.hpp"

struct directory_structure_t
{
    directory_structure_t() {
	prefix = "../test/";
	output = "../test/output2/";
	workspace = "../test/workspace/";
	figures = "../test/figures/";
    }
    void make_dir(){
	if ( !fs::exists( prefix ) )
	{
	    fs::create_directory( prefix );
	}
	if ( !fs::exists( workspace ) )
	{
	    fs::create_directory( workspace );
	}
	if ( !fs::exists( output ) )
	{
	    fs::create_directory( output );
	}
    	if ( !fs::exists( figures ) )
	{
	    fs::create_directory( figures );
	}
    }


    std::string prefix;
    std::string output;
    std::string workspace;
    std::string figures;
};


int main(int argc, char* argv[])
{
    using namespace boost::lambda;
    directory_structure_t ds;
    ds.make_dir();

    training_data_t tdata;

    tdata.load(ds.prefix+"training_data2.xml");
    matrix<int>& obs = tdata.obs;
    matrix<int>& dyn_obs = tdata.dyn_obs;
    matrix<int>& path = tdata.path;


    vector<matrix<float> > feat;
    generate_feature_maps(obs, dyn_obs, feat);

    vector<vector<int> > sg; //(num_states, num_neighbors)
    matrix<vector<double> > fg; //(num_features, num_states, num_neighbors)

    matrix<int> yx2ig;
    matrix<int> ig2yx;

    get_state_graph(obs, dyn_obs, sg, yx2ig, ig2yx);
    get_feature_graph(obs, dyn_obs, feat, sg, ig2yx, fg);

    vector<vector<int> > path_ig(1);
    get_path_ig(path, yx2ig, path_ig(0));

    vector<double> wei(feat.size()+1);
    wei <<= 0.33, 0.33, 0.33, 0.05;

    wei *= 40;

    lmdp_t lmdp;
    lmdp.initialize(fg, sg, yx2ig, ig2yx);

    unit_test(lmdp, path_ig, wei);
    //learn_weights_numeric(lmdp, path_ig, wei);
    return 0;
}



#include "lmdp_details.hpp"
#include "lmdp_learn_details.hpp"
