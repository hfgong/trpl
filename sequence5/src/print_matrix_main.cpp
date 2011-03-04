#include <boost/filesystem.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <iostream>
#include <fstream>

#include <CImg.h>

#include "real_timer.hpp"
#include "text_file.hpp"
#include "ublas_cimg.hpp"
#include "ublas_random.hpp"

#include "cvpr_array_traits.hpp"
using namespace boost;
using namespace boost::numeric::ublas;
using namespace cvpr;

int main(int argc, char* argv[])
{

    if(argc!=2 && argc !=4 && argc !=6)
    {
	std::cout<<"Usage: print_matrix xxx.txt "<<std::endl;
	std::cout<<"\tor print_matrix xxx.txt row_start row_end"<<std::endl;
	std::cout<<"\tor print_matrix xxx.txt row0 row1 col0 col1"<<std::endl;
	return 1;
    }
    matrix<float> mat;
    {
	std::ifstream fs(argv[1]);
	fs>>mat;
	fs.close();
    }
    if(mat.size1()<=0)
    {
	std::cout<<"error reading matrix!"<<std::endl;
	return 1;
    }
    if(argc==2)
    {
	array2d_print(std::cout, mat);
    }
    if(argc==4)
    {
	int row0 = lexical_cast<int>(argv[2]);
	int row1 = lexical_cast<int>(argv[3]);
	array2d_print(std::cout, project(mat, range(row0, row1+1), range(0, mat.size2())));
    }
    if(argc==6)
    {
	int row0 = lexical_cast<int>(argv[2]);
	int row1 = lexical_cast<int>(argv[3]);
	int col0 = lexical_cast<int>(argv[4]);
	int col1 = lexical_cast<int>(argv[5]);
	array2d_print(std::cout, project(mat, range(row0, row1+1), range(col0, col1+1)));
    }
    return 0;
}

