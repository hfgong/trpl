#ifndef __PLANNING__HPP__INCLUDED__
#define __PLANNING__HPP__INCLUDED__

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>


struct planning_result_item_t
{
    vector<matrix<int> > path;
    vector<float> dist;
    matrix<int> wind_num;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)	{
	ar & BOOST_SERIALIZATION_NVP(path);
	ar & BOOST_SERIALIZATION_NVP(dist);
	ar & BOOST_SERIALIZATION_NVP(wind_num);
    }
};

void carboxes2carobs(matrix<float> const& cars, 
		     matrix<double> const& img2grd,
		     vector<matrix<float> > & car_poly);

void load_carboxes(directory_structure_t& ds,
		   vector<std::vector<std::string> > const& seq,
		   matrix<matrix<float> >& cars);

void prepare_car_obs(matrix<matrix<float> > const& cars,
		     matrix<matrix<double> > const& img2grd,
		     matrix<vector<matrix<float> > >& car_poly);

void prepare_ped_obs(vector<object_trj_t> const& good_trlet_list,
		     int T,
		     matrix<matrix<float> >& ped_obs);

    
void combine_car_obs(matrix<vector<matrix<float> > > const& car_obsz,
		     vector<vector<matrix<float> > > & car_obs);

void do_homotopy_planning(ground_lim_t const& glim,
			  matrix<double> const& poly_ground,
			  int nn, float start_x, float start_y,
			  matrix<double> const& goal_ground,
			  vector<matrix<float> > const& car_obs, 
			  vector<matrix<float> > const& ped_obs,
			  vector<planning_result_item_t>& results,
			  directory_structure_t &ds, int tt);


void plan_trlet_list(geometric_info_t const& gi,
		     matrix<int> const& Tff,
		     vector<object_trj_t> const& good_trlet_list,
		     vector<vector<matrix<float> > > const& car_obs, 
		     matrix<matrix<float> > const& ped_obs,
		     int plan_advance,
		     vector<int>& plan_time,
		     vector<vector<planning_result_item_t> >& results,
		     directory_structure_t &ds);


void plan_trlet_list(mpi::communicator& world,
		     geometric_info_t const& gi,
		     matrix<int> const& Tff,
		     vector<object_trj_t> const& good_trlet_list,
		     vector<vector<matrix<float> > > const& car_obs, 
		     matrix<matrix<float> > const& ped_obs,
		     int plan_advance,
		     vector<int>& plan_time,
		     vector<vector<planning_result_item_t> >& results,
		     directory_structure_t &ds);

void construct_state_graph( matrix<int> const& obs,
			    vector<vector<int> >& sg,
			    matrix<int>& yx2ig,
			    matrix<int>& ig2yx);


void construct_feature_maps(matrix<int> const& obs, matrix<int> const& dyn_obs,
			    vector<matrix<float> >& feat);

void construct_feature_graph(matrix<int> const& obs, matrix<int> const& dyn_obs,
			     vector<matrix<float> > const& feat,
			     vector<vector<int> > const& sg,
			     matrix<int> const& ig2yx,
			     matrix<vector<double> >& fg);



float shortest_path(vector<vector<int> >const& sg,
		   vector<vector<double> > const& fdist,
		   int start, int goal,
		    vector<int>& path);

template <typename Float>
void wind_angle_planning(vector<vector<int> > const& sg,
			 vector<vector<Float> > const & fdist,
			 matrix<int> const& ig2yx,
			 matrix<Float> const& obs,
			 int wnum_l,
			 int wnum_u,
			 int start, int goal,
			 vector<vector<int> >& result_path,
			 vector<Float>& result_dist,
			 vector<vector<int> >& result_wind_num);


void combine_obstacles(int nn,
		       vector<matrix<float> > const& car_obs, 
		       vector<matrix<float> > const& ped_obs,
		       matrix<double> const & poly_ground,
		       vector<matrix<float> >& obs,
		       matrix<float>& obs_cent);

void fix_poly_ground(matrix<double> const& poly_ground2,
		     matrix<double> & poly_ground);

void construct_obstacle_maps(vector<matrix<float> > const& obs,
			     matrix<double> const& poly_ground, 
			     matrix<double> const& goal_ground,
			     matrix<int>& obs_map,
			     matrix<int>& dyn_obs_map);

void vis_obstacle_maps(matrix<double> const& poly_ground,
		       matrix<double> const& goal_ground,
		       int nn, float start_x, float start_y,
		       matrix<int> const& obs_map,
		       matrix<int> const& dyn_obs_map,
		       directory_structure_t &ds, int tt);


void compute_feat_dist(vector<vector<int> > const& sg,
		       matrix<vector<double> > const& fg,
		       vector<float> const& wei,
		       vector<vector<float> >& fdist);

void vis_planned_path(matrix<double> const& poly_ground,
		      matrix<double> const& goal_ground,
		      int nn, float start_x, float start_y,
		      vector<matrix<float> >const& obs,
		      matrix<int> const& obs_map,
		      matrix<int> const& ig2yx,
		      vector<vector<int> > const& path,
		      directory_structure_t &ds,
		      int tt, int gid);

void vis_feature(vector<matrix<float> >const& feat,
		 directory_structure_t &ds, 
		 int nn, int tt);

void choose_critic_obstacles(matrix<int> const& ig2yx,
			     matrix<float> const& obs_cent,
			     vector<int> const& spath,
			     float sdist,
			     matrix<float> & critic_obs_cent);

void do_homotopy_planning(ground_lim_t const& glim,
			  matrix<double> const& poly_ground2,
			  int nn, float start_x, float start_y,
			  matrix<double> const& goal_ground,
			  vector<matrix<float> > const& car_obs, 
			  vector<matrix<float> > const& ped_obs,
			  vector<planning_result_item_t>& results,
			  directory_structure_t &ds, int tt);

#endif

