
#include "tracking_data_package.hpp"
#include "tracking_detail.hpp"

struct data_package_t
{
    vector<object_trj_t> good_trlet_list;
    vector<object_trj_t> final_trj_list;
    matrix<int> final_state_list;
    vector<vector<int> > final_trj_index;
    matrix<float> Aff;
    matrix<object_trj_t> gap_trlet_list;
    matrix<float> Ocff;
    matrix<int> Tff;
    matrix<int> links;

    vector<object_trj_t> merged_trj_list;
    vector<vector<int> > merged_trj_index;
    matrix<int> merged_state_list;


    void load( directory_structure_t const& ds)	{
	{
	    std::string name = ds.workspace+"good_trlet_list.xml";
	    std::ifstream fin(name.c_str());
	    boost::archive::xml_iarchive ia(fin);
	    ia >> BOOST_SERIALIZATION_NVP(good_trlet_list);
	}

	{
	    std::string name = ds.workspace+"final_trj_list_alone.xml";
	    std::ifstream fs(name.c_str());
	    boost::archive::xml_iarchive ia(fs);
	    ia >> BOOST_SERIALIZATION_NVP(final_trj_list);
	}
	{
	    std::string name = ds.workspace+"final_state_list_alone.txt";
	    std::ifstream fs(name.c_str());
	    fs>>final_state_list;
	    fs.close();
	}
	{
	    std::string name = ds.workspace+"final_trj_index_alone.txt";
	    std::ifstream fs(name.c_str());
	    fs >> final_trj_index;
	    fs.close();
	}

	{
	    std::string name = ds.workspace+"Aff.txt";
	    std::ifstream fs(name.c_str());
	    fs>>Aff;
	    fs.close();
	}

	{
	    std::string name = ds.workspace+"gap_trlet_list_alone.xml";
	    std::ifstream fs(name.c_str());
	    boost::archive::xml_iarchive ar(fs);
	    ar >> BOOST_SERIALIZATION_NVP(gap_trlet_list);
	}

	{
	    std::string name = ds.workspace+"Alnff.txt";
	    std::ifstream fs(name.c_str());
	    fs >> Ocff;
	    fs.close();
	}

	{
	    std::string name = ds.workspace+"Tff.txt";
	    std::ifstream fs(name.c_str());
	    fs >> Tff;
	    fs.close();
	}
	{
	    std::string name = ds.workspace+"links_alone.txt";
	    std::ifstream fs(name.c_str());
	    fs >> links;
	    fs.close();
	}

	{
	    std::string name = ds.workspace+"merged_trj_list_alone.xml";
	    std::ifstream fin(name.c_str());
	    boost::archive::xml_iarchive ia(fin);
	    ia >> BOOST_SERIALIZATION_NVP(merged_trj_list);
	}

	{
	    std::string name = ds.workspace+"merged_trj_index_alone.txt";
	    std::ifstream fin(name.c_str());
	    fin>>merged_trj_index;
	    fin.close();
	}

	{
	    std::string name = ds.workspace+"merged_state_list_alone.txt";
	    std::ifstream fin(name.c_str());
	    fin >> merged_state_list;
	    fin.close();
	}


    }

};


void vis_final_trj_patches(
    directory_structure_t const& ds,
    std::string const& fig_path,
    vector<std::vector<std::string> > const& seq,
    int T,
    int Ncam,
    vector<object_trj_t>const & good_trlet_list,
    vector<object_trj_t>const & final_trj_list,
    matrix<int>const & final_state_list,
    vector<vector<int> >const & final_trj_index)
{
    using namespace boost::lambda;
    int num_obj = final_trj_list.size();
    vector<matrix<CImg<unsigned char> > > patches(
	scalar_vector<matrix<CImg<unsigned char> > >(
	    num_obj, matrix<CImg<unsigned char> >(Ncam, T) ) );

    unsigned char ccol[3]={255, 255, 0};
    unsigned char fcol[3]={255, 255, 255};
    unsigned char bcol[3]={0, 0, 0};
    for(int tt=0; tt<T; tt++)
    {
	vector<CImg<unsigned char> > images(Ncam);

	for(int cam=0; cam<Ncam; ++cam)
	{
	    images[cam] = CImg<unsigned char>(seq[cam][tt].c_str());

	    for(int nn=0; nn<num_obj; ++nn)
	    {
		if(tt<final_trj_list(nn).startt) continue;
		if(tt>final_trj_list(nn).endt) continue;
		vector<float> b = row(final_trj_list(nn).trj(cam), tt);
		boost::array<int, 4> box;
		std::transform(b.begin(), b.end(), box.begin(),
			       ll_static_cast<int>(_1+0.5f));

		patches(nn)(cam,tt) = images[cam].get_crop(
		    box[0], box[1], box[2], box[3]);
	    }	  
	}
    }

    typedef array3d_traits<CImg<unsigned char> > A;
    int ww = 100;
    int hh = 250;

    CImg<unsigned char> tile;
    A::change_size(tile, 3, hh*Ncam*num_obj, ww*T);
    array3d_fill(tile, 255);
    for(int nn=0; nn<num_obj; ++nn)
    {
	for(int cam=0; cam<Ncam; ++cam)
	{
	    for(int tt=0; tt<T; ++tt)
	    {
		if(A::size1(patches(nn)(cam, tt)) ==0) continue;
		patches(nn)(cam, tt).resize(ww, hh);
		if(final_state_list(nn, tt)==0)
		{
		    patches(nn)(cam, tt).draw_line(1, 1, ww-1, hh-1, ccol, 2);
		    patches(nn)(cam, tt).draw_line(ww-1, 1, 1, hh-1, ccol, 2);
		}
		std::string numstr(lexical_cast<std::string>(tt));
		patches(nn)(cam, tt).draw_text (1, 1, numstr.c_str(), fcol, bcol, 1, 40);
		for(int ii=0; ii<final_trj_index(nn).size(); ++ii)
		{
		    int nn0 = final_trj_index(nn)(ii);
		    if(tt==good_trlet_list(nn0).startt)
		    {
			std::string nstr(lexical_cast<std::string>(nn0));
			patches(nn)(cam, tt).draw_text (1, hh/2, nstr.c_str(),
							fcol, bcol, 1, 100);
		    }
		}
		add_patch_to_tile(tile, patches(nn)(cam, tt), cam+nn*Ncam, tt);
	    }
	}

    }

    {
	tile.save_jpeg(fig_path.c_str(), 90);
    }


}


bool draw_ground_fig(object_trj_t const& trj, int tt,
		     CImg<unsigned char>& ground_fig,
		     geometric_info_t const& gi)
{

    array3d_traits<CImg<unsigned char> >::change_size(
	ground_fig, 3, gi.ground_lim.ymax, gi.ground_lim.xmax);

    array3d_fill(ground_fig, 255);

    unsigned char pcol[3]={200, 200, 255};
    unsigned char tcol[3]={255, 0, 0};
    unsigned char yellow[3]={255, 255, 0};
    unsigned char black[3]={0, 0, 0};

    unsigned char green[3]={0, 255, 0};

    unsigned char cyan[3] = {0, 255, 255};
    unsigned char yellow2[3]={128, 128, 0};

    CImg<double> poly;
    array2d_copy(gi.poly_ground, poly);

    ground_fig.draw_polygon(poly, pcol);

    ground_fig.draw_polygon(poly, black, 1, ~0U);


    if(trj.trj.size()==0) return false;
    int t1 = trj.startt;
    int t2 = trj.endt;

    if(tt<t1 || tt>t2) return false;

    object_trj_t::trj_3d_t const& s=trj.trj_3d;


    for(int tx=t1; tx<tt; ++tx)
    {
	int th = 1;
	for(int dx=-th; dx<=th; ++dx)
	{
	    for(int dy=-th; dy<=th; ++dy)
	    {

		ground_fig.draw_line(static_cast<int>(s(tx, 0)+0.5f)+dx,
				     static_cast<int>(s(tx, 1)+0.5f)+dy, 
				     static_cast<int>(s(tx+1, 0)+0.5f)+dx,
				     static_cast<int>(s(tx+1, 1)+0.5f)+dy, 
				     tcol, 1);
	    }
	}
    }

    ground_fig.draw_circle(static_cast<int>(s(tt, 0)+0.5f),
			   static_cast<int>(s(tt, 1)+0.5f), 
			   10, yellow2);
    ground_fig.draw_circle(static_cast<int>(s(tt, 0)+0.5f),
			   static_cast<int>(s(tt, 1)+0.5f), 
			   10, black, 1, 1);


}



void vis_sequences(directory_structure_t const& ds,
		   vector<std::vector<std::string> > const& seq,
		   geometric_info_t const& gi,
		   vector<object_trj_t> const& good_trlet_list,
		   vector<object_trj_t> const& trj_list,
		   matrix<int> const& state_list,
		   vector<vector<int> > const& trj_index)
{
    using namespace boost::lambda;
    int T = seq[0].size();
    int Ncam = 2;

    //vector<matrix<double> > const& grd2img = gi.grd2img;
    //vector<matrix<double> > const& goals_im = gi.goals_im;

    unsigned char ccol[3]={255, 255, 0};

    unsigned char fcol[3]={255, 255, 255};
    unsigned char bcol[3]={0, 0, 0};

    unsigned char green[3] = {0, 255, 0};
    unsigned char cyan[3] = {0, 255, 255};

    unsigned char yellow[3]={255, 255, 0};
    typedef array3d_traits<CImg<unsigned char> > A;

    for(int nn=0; nn<trj_list.size(); ++nn)
    {
	object_trj_t const& trj = trj_list(nn);
	for(int tt=trj.startt; tt<=trj.endt; ++tt)
	{
	    //vector<CImg<unsigned char> > images(Ncam);

	    CImg<unsigned char> ground_fig;
	    draw_ground_fig(trj, tt, ground_fig, gi);


	    {
		ground_fig.mirror('y');
		std::string name = ds.output+
		    str(format("grd_trj_aln_%03d_%03d.png")%nn%tt);
		ground_fig.save_png(name.c_str());
	    }

	    for(int cam=0; cam<Ncam; ++cam)
	    {
		CImg<unsigned char> vis = CImg<unsigned char>(seq[cam][tt].c_str());
		vector<float> b = row(trj.trj(cam), tt);
		boost::array<int, 4> box;
		std::transform(b.begin(), b.end(), box.begin(),
			       ll_static_cast<int>(_1+0.5f));
		for(int cc=0; cc<A::size1(vis); ++cc)
		{
		    for(int yy=0; yy<A::size2(vis); ++yy)
		    {
			for(int xx=0; xx<A::size3(vis); ++xx)
			{
			    if(xx<box[0] || xx> box[2] ||
			       yy<box[1] || yy> box[3])
			    A::ref(vis, cc, yy, xx) = A::ref(vis, cc, yy, xx)/3+32;
			}
		    }
		}
		for(int dx=-1; dx<=1; ++dx)
		{
		    for(int dy=-1; dy<=1; ++dy)
		    {
			vis.draw_line(box[0]+dx, box[1]+dy, 0,
				      box[0]+dx, box[3]+dy, 0, ccol, 1);
			vis.draw_line(box[2]+dx, box[1]+dy, 0,
				      box[2]+dx, box[3]+dy, 0, ccol, 1);
			vis.draw_line(box[0]+dx, box[1]+dy, 0,
				      box[2]+dx, box[1]+dy, 0, ccol, 1);
			vis.draw_line(box[0]+dx, box[3]+dy, 0,
				      box[2]+dx, box[3]+dy, 0, ccol, 1);
		    }
		}


		std::string name = ds.output+str(format("trj_aln_%03d_%1d_%03d.jpg")
		    %nn%cam%tt);
		vis.save_jpeg(name.c_str(), 90);
	    }
	}
    }
}


void write_bounding_boxes(vector<object_trj_t> const& trj_list,
			   std::string const& name)
{

    std::ofstream fout(name.c_str());
    for(int nn=0; nn<trj_list.size(); ++nn)
    {
	for(int tt=trj_list(nn).startt; tt<=trj_list(nn).endt; ++tt)
	{
	    fout<<nn<<" "<<tt;
	    for(int cam=0; cam<trj_list(nn).trj.size(); ++cam)
	    {
		fout<<" "<<trj_list(nn).trj(cam)(tt, 0)
		    <<" "<<trj_list(nn).trj(cam)(tt, 1)
		    <<" "<<trj_list(nn).trj(cam)(tt, 2)
		    <<" "<<trj_list(nn).trj(cam)(tt, 3);
	    }
	    fout<<std::endl;
	}
    }
    fout.close();
}



int main(int argc, char* argv[])
{
    directory_structure_t ds;

    vector<std::vector<std::string> > seq(2);
    read_sequence_list(ds.prefix, seq);
    int T = seq[0].size();
    int Ncam = 2;

    data_package_t data;
    data.load(ds);

    array<std::size_t, 2> img_size = {768, 1024};
    geometric_info_t gi;
    gi.load(ds, img_size);


    vector<object_trj_t>& good_trlet_list = data.good_trlet_list;
    vector<object_trj_t>& final_trj_list = data.final_trj_list;
    matrix<int>& final_state_list = data.final_state_list;
    vector<vector<int> >& final_trj_index = data.final_trj_index;
    matrix<float>& Aff = data.Aff;
    matrix<object_trj_t>& gap_trlet_list = data.gap_trlet_list;
    matrix<float>& Ocff = data.Ocff;
    matrix<int>& links = data.links;
    matrix<int>& Tff = data.Tff;

    vector<object_trj_t>& merged_trj_list = data.merged_trj_list;
    matrix<int>& merged_state_list = data.merged_state_list;
    vector<vector<int> >& merged_trj_index = data.merged_trj_index;

    write_bounding_boxes(merged_trj_list, ds.workspace+"results_alone.txt");

    vis_final_trj_patches(ds, ds.figures+"lp_alone_patches.jpg", seq, T, Ncam,
			  good_trlet_list,
			  final_trj_list, final_state_list, final_trj_index);

    vis_final_trj_patches(ds, ds.figures+"lp_alone_merged_patches.jpg", seq, T, Ncam,
			  good_trlet_list,
			  merged_trj_list, merged_state_list, merged_trj_index);

#if 1
     vis_sequences(ds, seq, gi, good_trlet_list, 
		  merged_trj_list, merged_state_list, merged_trj_index);
#endif
    return 0;
}

