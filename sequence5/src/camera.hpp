#ifndef __PRETR__CAMERA__HPP__INCLUDED__
#define __PRETR__CAMERA__HPP__INCLUDED__

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>


using namespace boost::numeric::ublas;
using namespace boost;


template <typename Float>
struct camera_param_t
{
    matrix<Float> KK_left_new;
    matrix<Float> KK_right_new;
    vector<Float> R_new;
    vector<Float> T_new;
    void print() const {
	std::cout<<KK_left_new<<std::endl;
	std::cout<<KK_right_new<<std::endl;
	std::cout<<R_new<<std::endl;
	std::cout<<T_new<<std::endl;
    }
};

template <typename Float>
void load_camera_param(std::string const& folder, camera_param_t<Float>& param)
{
    read_text_array2d(folder+"KK_left_new.txt", param.KK_left_new);
    read_text_array2d(folder+"KK_right_new.txt", param.KK_right_new);
    read_text_array1d(folder+"R_new.txt", param.R_new);
    read_text_array1d(folder+"T_new.txt", param.T_new);
}



/* Matrix inversion routine.
   Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
bool inverse(const matrix<T>& input, matrix<T>& inv)
{
    typedef permutation_matrix<std::size_t> pmatrix;

    // create a working copy of the input
    matrix<T> A(input);

    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = lu_factorize(A, pm);
    if (res != 0)
	return false;

    // create identity matrix of "inverse"
    inv = (identity_matrix<T> (A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(A, pm, inv);

    return true;
}


template <typename Float>
void get_plane_intersection(matrix<Float> const& KK, 
			    vector<Float> const& plane,
			    matrix<Float> const& point2d, 
			    matrix<Float> & point3d)
{
    using namespace boost::lambda;
    matrix<Float> invKK;
    inverse(KK, invKK);
    matrix<Float> point2dx(point2d.size1()+1, point2d.size2());
    project(point2dx, range(0, point2d.size1()), range(0, point2d.size2())) = point2d;
    row(point2dx, point2d.size1()) = scalar_vector<Float>(point2d.size2(), 1);

    matrix<Float> tmp(prod(invKK, point2dx));
    vector<Float> planev(3);
    std::copy(&plane[0], &plane[3], planev.begin());
    vector<Float> w(prod(planev, tmp));
    std::for_each(w.begin(), w.end(), _1=-plane(3)/_1);

    matrix<Float> w2(3, w.size());
    for(int ii=0; ii<3; ++ii)  row(w2, ii) = w;
    point3d = element_prod(tmp, w2);

#if 0
    std::cout<<"get_plane_intersection:"<<std::endl;
    std::cout<<"KK="<<KK<<std::endl;
    std::cout<<"plane="<<plane<<std::endl;
    std::cout<<"point2d="<<point2d<<std::endl;
    std::cout<<"point3d="<<point3d<<std::endl;
//passed testing
//matlab function: get_plane_point_backprojection
#endif
}

struct ground_lim_t
{
    int xmin, xmax, ymin, ymax;
};

template <class Float>
void compute_binocular_transform(camera_param_t<double> const& cam_param,
				 matrix<Float> const& goal2d,
				 matrix<Float> const& poly2d,
				 array<std::size_t, 2> const& img_size,
				 matrix<matrix<unsigned char> >& poly_mask,
				 matrix<Float>& goal_ground,
				 matrix<Float>& poly_ground,
				 matrix<matrix<Float> > & goals_im,
				 matrix<matrix<Float> > & polys_im,
				 matrix<matrix<double> > const& img2grd,
				 matrix<matrix<double> > const& grd2img,
				 ground_lim_t& ground_lim)
{
    using namespace boost::lambda;
    // Ground plane estimation(Manual estimation)
    double camera_height = 1800;
    vector<double> gNorm(3);// = [0 -1 -0.04];
    gNorm(0) = 0; gNorm(1) = -1; gNorm(2) = -0.04;
    vector<double> gPlane(4);
    project(gPlane, range(0, 3))=gNorm/norm_2(gNorm);
    gPlane(3) = camera_height;
    double yHorizon = 430;

    int T = img2grd.size1();
    int Ncam = img2grd.size2();
    //std::cout<<"img2grd="<<img2grd<<std::endl;

    poly_mask = matrix<matrix<unsigned char> >(T, Ncam);
    mask_from_polygon(poly_mask(0, 0), img_size[0], img_size[1],
		      row(poly2d, 0), row(poly2d, 1));

    matrix<Float> poly3d;
    get_plane_intersection(cam_param.KK_left_new, gPlane, poly2d, 
			   poly3d);
    //std::cout<<"end, poly3d"<<std::endl;

    matrix<Float> goal3d;
    get_plane_intersection(cam_param.KK_left_new, gPlane, goal2d, 
			   goal3d);
    //std::cout<<"end, goal3d"<<std::endl;

    float xmin3d, xmax3d, zmin3d, zmax3d;
    {
	vector<double> tmp(row(poly3d, 0));
	xmin3d = *(std::min_element(tmp.begin(), tmp.end()));
	xmax3d = *(std::max_element(tmp.begin(), tmp.end()));
    }
    {
	vector<double> tmp(row(poly3d, 2));
	zmin3d = *(std::min_element(tmp.begin(), tmp.end()));
	zmax3d = *(std::max_element(tmp.begin(), tmp.end()));
    }
    //std::cout<<"xmin3d="<<xmin3d<<", xmax3d="<<xmax3d<<std::endl;
    //std::cout<<"zmin3d="<<zmin3d<<", zmax3d="<<zmax3d<<std::endl;

    float grid_size=50; //5cm=1pixel
    array<int, 2> gimg_size;
    gimg_size[0] = static_cast<int>((zmax3d-zmin3d)/grid_size+0.5);
    gimg_size[1] = static_cast<int>((xmax3d-xmin3d)/grid_size+0.5);

    //gImgSize = 548   409
    // int32(xmin3D)     -17780
    // int32(zmin3D)      9819


    //std::cout<<"img2grd(0)="<<img2grd(0)<<std::endl;
    //std::cout<<"grd2img(0)="<<grd2img(0)<<std::endl;

//% Convert Left side ground polygon to Right side ground polygon
    matrix<Float> poly3dr(poly3d);
    for(int nn=0; nn<poly3dr.size2(); ++nn)
    {
	column(poly3dr, nn) += cam_param.T_new;
    }
    matrix<Float> poly2drx(prod(cam_param.KK_right_new, poly3dr));
    matrix<Float> poly2dr(2, poly2drx.size2());
    vector<Float> vtmp(row(poly2drx, 2));
    std::for_each(vtmp.begin(), vtmp.end(), _1=1/_1);
    row(poly2dr, 0) = element_prod(row(poly2drx, 0), vtmp);
    row(poly2dr, 1) = element_prod(row(poly2drx, 1), vtmp);

    mask_from_polygon(poly_mask(0, 1), img_size[0], img_size[1],
		      row(poly2dr, 0), row(poly2dr, 1));

/////
    matrix<Float> goal3dr(goal3d);
    for(int nn=0; nn<goal3dr.size2(); ++nn)
    {
	column(goal3dr, nn) += cam_param.T_new;
    }
    matrix<Float> goal2drx(prod(cam_param.KK_right_new, goal3dr));
    matrix<Float> goal2dr(2, goal2drx.size2());
    vector<Float> vtmp2(row(goal2drx, 2));
    std::for_each(vtmp2.begin(), vtmp2.end(), _1=1/_1);
    row(goal2dr, 0) = element_prod(row(goal2drx, 0), vtmp2);
    row(goal2dr, 1) = element_prod(row(goal2drx, 1), vtmp2);

    polys_im = matrix<matrix<Float> > (T, Ncam);
    polys_im(0, 0) = poly2d;
    polys_im(0, 1) = poly2dr;

    goals_im = matrix<matrix<Float> > (T, Ncam);
    goals_im(0, 0) = goal2d;
    goals_im(0, 1) = goal2dr;

    vector<Float> px(row(polys_im(0, 0), 0)), py(row(polys_im(0, 0), 1));
    vector<Float> tmpx, tmpy;
    apply_homography(img2grd(0, 0), px, py, tmpx, tmpy);
    poly_ground = matrix<Float>(2, tmpx.size());
    row(poly_ground, 0) = tmpx;
    row(poly_ground, 1) = tmpy;

    ground_lim.xmin = static_cast<int>(*(std::min_element(tmpx.begin(), tmpx.end())));
    ground_lim.xmax = static_cast<int>(*(std::max_element(tmpx.begin(), tmpx.end()))+1);
    ground_lim.ymin = static_cast<int>(*(std::min_element(tmpy.begin(), tmpy.end())));
    ground_lim.ymax = static_cast<int>(*(std::max_element(tmpy.begin(), tmpy.end()))+1);

    px = row(goals_im(0, 0), 0); py = row(goals_im(0, 0), 1);
    apply_homography(img2grd(0, 0), px, py, tmpx, tmpy);
    goal_ground = matrix<Float>(2, tmpx.size());
    row(goal_ground, 0) = tmpx;
    row(goal_ground, 1) = tmpy;


    vector<Float> goal_x = row(goal_ground, 0);
    vector<Float> goal_y = row(goal_ground, 1);
    vector<Float> poly_x = row(poly_ground, 0);
    vector<Float> poly_y = row(poly_ground, 1);
    for(int tt=1; tt<T; ++tt)
    {
	for(int cam=0; cam<Ncam; ++cam)
	{
	    vector<Float> gi_x, gi_y;
	    apply_homography(grd2img(tt, cam), goal_x, goal_y, gi_x, gi_y);
	    goals_im(tt, cam) = matrix<Float>(2, gi_x.size());
	    row(goals_im(tt, cam), 0) = gi_x;
	    row(goals_im(tt, cam), 1) = gi_y;

	    apply_homography(grd2img(tt, cam), poly_x, poly_y, gi_x, gi_y);
	    polys_im(tt, cam) = matrix<Float>(2, gi_x.size());
	    row(polys_im(tt, cam), 0) = gi_x;
	    row(polys_im(tt, cam), 1) = gi_y;
	}
    }


}

#endif
